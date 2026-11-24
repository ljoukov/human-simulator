"""Reproduce safe Pareto pruning of the exact bootstrap on M4 CPU and Modal H100.

Run: python pareto.py --reference /path/to/winners.npy
The optional reference contains original indices from the exhaustive search.
Without --reference the script regenerates that reference on the CPU.
"""
import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import modal
import numpy as np
import modal_benchmark as mb

app = modal.App('hip-ensemble-pareto')

# Each thread marks one candidate. Anchor vectors are small enough to cache.
DOMINANCE_KERNEL = r'''
extern "C" __global__ void screen(const float* loss, const float* anchors,
                                   bool* keep, int n, int nanchors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int a = 0; a < nanchors; ++a) {
        bool worse = true;
        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            worse &= loss[p*n + i] > anchors[a*6 + p];
        }
        if (worse) { keep[i] = false; return; }
    }
    keep[i] = true;
}
'''


def anchor_weights(weights):
    """6 pure + 15 equal pairs + 30 asymmetric pairs + 1 uniform mixture."""
    anchor_rows = [row for row in weights if sorted(row, reverse=True) in (
        [6, 0, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0], [5, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    )]
    if len(anchor_rows) != 52:
        raise ValueError('Expected all exact six-participant resamples')
    return np.stack(anchor_rows)


def cpu_sweep(kernel, study, frames, weights, count, seed, threads):
    raw, params = study.sample(count, seed)
    del raw
    mixtures = anchor_weights(weights)
    winners, survivor_counts = [], []
    for mm in np.linspace(-15, 15, 13):
        loss = np.stack([
            study.score(kernel, mb.offset_moment(x, mm), params, threads) for x in frames
        ])
        anchors = np.unique(np.argmin(mixtures @ loss[:, :4096], axis=1))
        ids = np.arange(count)
        for a in anchors:
            # Limit the next test to previous survivors; preserve their ID order.
            worse = loss[0, ids] > loss[0, a]
            for subject in range(1, 6):
                worse &= loss[subject, ids] > loss[subject, a]
            ids = ids[~worse]
        winners.append(ids[np.argmin(weights @ loss[:, ids], axis=1)])
        survivor_counts.append(len(ids))
    return np.stack(winners), survivor_counts


@app.function(image=mb.image.add_local_python_source('modal_benchmark'), gpu='H100!', timeout=300, scaledown_window=15)
def gpu_sweep(source, frame_bytes, weight_bytes, count, seed):
    import cupy as cp
    from study import sample

    hardware = mb.identity()
    frames = [np.frombuffer(x, np.float32).reshape(-1, 8) for x in frame_bytes]
    weights = np.frombuffer(weight_bytes, np.float64).reshape(-1, 6)
    mixtures = cp.asarray(anchor_weights(weights))
    weights = cp.asarray(weights)
    raw, params = sample(count, seed)
    del raw
    p = cp.asarray(params)
    score = cp.RawKernel(source, 'hip_score', options=('--std=c++14', '--fmad=false'))
    screen = cp.RawKernel(DOMINANCE_KERNEL, 'screen')
    shape = ((count + 127) // 128,)
    losses = cp.empty((6, count), cp.float32)
    keep = cp.empty(count, dtype=cp.bool_)
    winners, counts = [], []
    t0, t1 = cp.cuda.Event(), cp.cuda.Event()
    t0.record()
    for mm in np.linspace(-15, 15, 13):
        for i, x in enumerate(frames):
            xf = cp.asarray(mb.offset_moment(x, mm))
            score(shape, (128,), (xf, p, losses[i], np.int32(len(x)), np.int32(count)))
        anchors = cp.argmin(mixtures @ losses[:, :4096].astype(cp.float64), axis=1)
        anchor_losses = cp.ascontiguousarray(losses[:, anchors].T)
        screen(shape, (128,), (losses, anchor_losses, keep, np.int32(count), np.int32(len(anchors))))
        ids = cp.flatnonzero(keep)
        winners.append(ids[cp.argmin(weights @ losses[:, ids].astype(cp.float64), axis=1)])
        counts.append(len(ids))
    t1.record()
    t1.synchronize()
    return cp.stack(winners).get(), counts, hardware, cp.cuda.get_elapsed_time(t0, t1)


def main():
    import bootstrap
    import study

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--mode', choices=['cpu', 'gpu', 'both'], default='both')
    ap.add_argument('--output', type=Path, default=study.DATA / 'benchmark-pareto.json')
    args = ap.parse_args()
    count, seed, threads = 1048576, 20261122, os.cpu_count() or 1

    trials = json.loads((study.DATA/'implants.json').read_text())['trials']
    geometry = json.loads((study.DATA/'geometry.json').read_text())
    training = [t for t in trials if t['split'] == 'calibration']
    subjects = sorted({t['subject'] for t in training})
    xs = {t['id']: study.encode(t, geometry) for t in training}
    frames = [study.combine([t for t in training if t['subject'] == s], xs) for s in subjects]
    weights, _ = bootstrap.distributions(len(subjects))
    if args.reference:
        reference = np.load(args.reference, allow_pickle=False)
    else:
        with tempfile.TemporaryDirectory() as d:
            cpu, _, _ = study.compile_cpu(Path(d))
            raw, params = study.sample(count, seed)
            del raw
            reference = np.stack([
                bootstrap.cpu_bootstrap(cpu, [mb.offset_moment(x, mm) for x in frames], params, threads, weights)
                for mm in np.linspace(-15, 15, 13)
            ])
    if reference.shape != (13, 462):
        raise ValueError('Reference must contain every offset and resample')
    measurements = dict(
        candidates=count, seed=seed, resamples=len(weights), offsetsMm=np.linspace(-15, 15, 13).tolist(),
        referenceSha256=hashlib.sha256(reference.astype('<i8').tobytes()).hexdigest(),
        inputsSha256=hashlib.sha256(b''.join(x.tobytes() for x in frames)).hexdigest(),
        sourceSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__).resolve(), study.SOURCE)},
        anchorRule='52 fixed weight vectors on the first 4,096 candidate IDs. Remove a candidate iff at least one anchor has strictly lower loss on every participant. Retain order for exact tie-breaking.',
        timingScope='Starts with prepared clinical frames. Regenerates the candidate set for every job. Cloud RPC includes sending inputs and receiving the winners; first app time includes app initialization with a prebuilt image.',
    )
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous['referenceSha256'] != measurements['referenceSha256']:
            raise RuntimeError('Reference changed from earlier run')
        if previous['anchorRule'] != measurements['anchorRule']:
            raise RuntimeError('Anchor screen changed from earlier run')
        for backend in ('cpu', 'h100'):
            if backend in previous:
                measurements[backend] = previous[backend]
    if args.mode in ('gpu', 'both'):
        durations, device_times = [], []
        args_gpu = (study.SOURCE.read_text(), [x.tobytes() for x in frames], weights.tobytes(), count, seed)
        init_t = time.perf_counter()
        with app.run():
            for rep in range(4):
                t = time.perf_counter()
                winners, survivors, gpu, elapsed = gpu_sweep.remote(*args_gpu)
                wall = time.perf_counter() - t
                if not np.array_equal(winners, reference):
                    raise RuntimeError(f'H100 failed parity on {np.sum(winners != reference)} winners')
                if rep:
                    durations.append(wall)
                    device_times.append(elapsed)
                else:
                    first = time.perf_counter() - init_t
            measurements['h100'] = dict(hardware=gpu, appId=app.app_id, firstAppSeconds=first,
                warmRpcSeconds=durations, medianSeconds=statistics.median(durations),
                warmDeviceSamplesMs=device_times, matches=int(winners.size), survivors=survivors)
        study.dump(args.output, measurements)
        print(json.dumps(measurements['h100']), flush=True)
    if args.mode in ('cpu', 'both'):
        durations = []
        init_t = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            cpu, _, _ = study.compile_cpu(Path(d))
            for rep in range(4):
                t = time.perf_counter()
                winners, survivors = cpu_sweep(cpu, study, frames, weights, count, seed, threads)
                wall = time.perf_counter() - t
                if not np.array_equal(winners, reference):
                    raise RuntimeError(f'CPU failed parity on {np.sum(winners != reference)} winners')
                if rep:
                    durations.append(wall)
                else:
                    first = time.perf_counter() - init_t
        measurements['cpu'] = dict(hardware=study.hardware(), threads=threads, firstLocalSeconds=first,
            warmSeconds=durations, medianSeconds=statistics.median(durations),
            matches=int(winners.size), survivors=survivors)
        study.dump(args.output, measurements)
        print(json.dumps(measurements['cpu']), flush=True)


if __name__ == '__main__':
    main()
