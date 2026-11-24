"""Exact participant-bootstrap sensitivity of the best hip-load model.

Run with the same Modal and NumPy environment used by modal_benchmark.py.
Only the six calibration participants enter model selection. Every distinct
size-six, with-replacement resample is enumerated with its exact multiplicity.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import modal_benchmark
import study


def distributions(n):
    # Stars and bars enumerates every vector of subject counts summing to n.
    vectors = []
    for cuts in itertools.combinations(range(2*n-1), n-1):
        counts = np.diff((-1, *cuts, 2*n-1))-1
        vectors.append(counts)
    weights = np.array(vectors, dtype=np.float64)
    permutations = np.array([
        math.factorial(n) // math.prod(math.factorial(int(x)) for x in row)
        for row in weights
    ])
    if int(permutations.sum()) != n**n:
        raise RuntimeError("Bootstrap support has wrong multiplicity")
    return weights, permutations


def cpu_bootstrap(kernel, frames, params, threads, weights):
    participant_losses = np.stack([
        study.score(kernel, x, params, threads) for x in frames
    ]).astype(np.float64)
    batches = []
    for lo in range(0, len(weights), 16):
        batches.append(np.argmin(weights[lo:lo+16] @ participant_losses, axis=1))
    return np.concatenate(batches)


def quantiles(values, probability):
    order = np.argsort(values)
    cdf = np.cumsum(probability[order])
    return [float(values[order[np.searchsorted(cdf, q)]]) for q in (.05,.5,.95)]


def offset_sweep(args, frames, weights, prob, subjects, trials, xs):
    # Systematic assumptions shared by all participants, examined independently.
    offsets = np.linspace(-15,15,13).tolist()
    cpu_times, gpu_times = [], []
    with tempfile.TemporaryDirectory() as temp:
        cold_start = time.perf_counter()
        cpu, _, _ = study.compile_cpu(Path(temp))
        for rep in range(3):
            start = time.perf_counter()
            raw, parameters = study.sample(args.count, args.seed)
            del raw
            indices = np.stack([
                cpu_bootstrap(cpu, [modal_benchmark.offset_moment(x, mm) for x in frames],
                              parameters, args.threads, weights) for mm in offsets
            ])
            cpu_times.append(time.perf_counter()-start)
            if not rep:
                cold_cpu = time.perf_counter()-cold_start
                winners = indices
            elif not np.array_equal(winners, indices):
                raise RuntimeError("Laptop sweep is nondeterministic")
    print("CPU complete",cpu_times,flush=True)
    digest = hashlib.sha256(parameters.tobytes()).hexdigest()
    request = (study.SOURCE.read_text(), [x.tobytes() for x in frames], weights.tobytes(),
               len(subjects), args.count, args.seed, 1, offsets, False)
    cold_start = time.perf_counter()
    with modal_benchmark.app.run():
        for rep in range(4):
            start = time.perf_counter()
            remote = modal_benchmark.bootstrap_gpu.remote(*request)
            elapsed = time.perf_counter()-start
            if not rep:
                cold_gpu = time.perf_counter()-cold_start
            else:
                gpu_times.append(elapsed)
            if remote['parameterDigest']!=digest or not np.array_equal(remote['winners'],winners):
                raise RuntimeError("Different full-search winners on H100 vs laptop")
        app_id = modal_benchmark.app.app_id
    reference = winners[offsets.index(0.)]
    selected_case = sorted(trials,key=lambda t:(t['activity']!='walk',t['subject'],t['id']))[0]
    nominal = xs[selected_case['id']]
    frame = int(np.argmax(nominal[:,0]))
    cases = []
    for mm, group in zip(offsets,winners):
        ids, inverse = np.unique(group, return_inverse=True)
        masses = np.bincount(inverse, weights=prob)
        x = modal_benchmark.offset_moment(nominal, mm)
        joint, tendon, _ = study.simulate(x, parameters[ids])
        cases.append(dict(offsetMm=mm, distinctWinners=len(ids),
                          probabilitySameWinner=float(prob[group==reference].sum()),
                          jointForceBwQuantiles=quantiles(joint[:,frame],masses),
                          glutealTensionBwQuantiles=quantiles(tendon[:,frame],masses)))
    result = dict(
        method="Sweep the signed frontal lever arm by a common offset across participants, M' = M + Fz * offset. At each offset calibrate every participant bootstrap and project onto the same held-out walking record. The offset range is a sensitivity assumption.",
        seed=args.seed, subjects=subjects, candidates=args.count, resamples=len(weights),
        candidateResampleOffsetCombinations=int(len(offsets)*len(weights)*args.count),
        winnerAgreement=int(winners.size), heldOutTrialId=selected_case['id'],
        heldOutPhase=selected_case['frames'][frame]['phase'], measuredBw=float(nominal[frame,5]),
        cases=cases, timing=dict(cpu=study.hardware(), cpuThreads=args.threads,
            hardware=remote['hardware'], modalAppId=app_id,
            firstLocalSeconds=cold_cpu, firstCloudAppSeconds=cold_gpu,
            cpuSamplesSeconds=cpu_times, warmCloudSamplesSeconds=gpu_times,
            warmRpcSpeedup=statistics.median(cpu_times)/statistics.median(gpu_times),
            firstAppSpeedup=cold_cpu/cold_gpu,
            scope="Same 13 offsets and 462 resamples per search. From preprocessed clinical features to all winning indices. Laptop includes local parameter generation and fitting. Cloud includes request packing/transport by Modal, regenerating and uploading parameters, computing, and receiving all winners. First-job comparison additionally includes local C++ compilation or Modal app initialization and first request with the prebuilt image. Python startup, clinical feature preparation, and reporting are outside both timings; the host image cache state was uncontrolled."
        ),
    )
    study.dump(study.DATA/'bootstrap-sensitivity.json', result)
    print(json.dumps(result['timing'],indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offset-sweep', action='store_true', help='Compare complete local/cloud searches across a 13-point marker lever sensitivity sweep.')
    ap.add_argument('--count', type=int, default=1048576)
    ap.add_argument('--seed', type=int, default=20261122)
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    trials = json.loads((study.DATA/'implants.json').read_text())['trials']
    geometry = json.loads((study.DATA/'geometry.json').read_text())
    training = [t for t in trials if t['split']=='calibration']
    held = [t for t in trials if t['split']=='evaluation']
    subjects = sorted({t['subject'] for t in training})
    if set(subjects) & {t['subject'] for t in held}:
        raise ValueError("Participant leakage into held-out data")
    xs = {t['id']:study.encode(t, geometry) for t in trials}
    frames = [study.combine([t for t in training if t['subject']==s], xs) for s in subjects]
    weights, multiplicities = distributions(len(subjects))
    prob = multiplicities / multiplicities.sum()
    if args.offset_sweep:
        return offset_sweep(args, frames, weights, prob, subjects, held, xs)
    raw, params = study.sample(args.count, args.seed)
    del raw
    with tempfile.TemporaryDirectory() as temp:
        cpu, flags, compiler = study.compile_cpu(Path(temp))
        winners = cpu_bootstrap(cpu, frames, params, args.threads, weights)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            another = cpu_bootstrap(cpu, frames, params, args.threads, weights)
            timings.append(1000*(time.perf_counter()-start))
            if not np.array_equal(winners, another):
                raise ValueError("CPU winner mismatch across repetitions")
    print(f"CPU median {statistics.median(timings):.2f} ms, {len(weights)} resamples, "
          f"{len(np.unique(winners))} distinct fitted models", flush=True)
    with modal_benchmark.app.run():
        start = time.perf_counter()
        gpu = modal_benchmark.bootstrap_gpu.remote(
            study.SOURCE.read_text(), [x.tobytes() for x in frames], weights.tobytes(),
            len(subjects), args.count, args.seed, args.repeats,
        )
        wall = time.perf_counter()-start
        app_id = modal_benchmark.app.app_id
    if hashlib.sha256(params.tobytes()).hexdigest() != gpu['parameterDigest']:
        raise RuntimeError("H100 generated different parameters")
    if not np.array_equal(gpu['winners'], winners):
        raise RuntimeError("H100 selected different models")
    ids, inv = np.unique(winners, return_inverse=True)
    masses = np.bincount(inv, weights=prob)
    mode = int(np.argmax(masses))
    phases = []
    for trial in held:
        x = xs[trial['id']]
        frame = int(np.argmax(x[:,0]))
        force, tension, _ = study.simulate(x, params[ids])
        phases.append(dict(
            trialId=trial['id'], participant=trial['subject'], activity=trial['activity'],
            phase=trial['frames'][frame]['phase'], measuredBw=float(x[frame,5]),
            jointForceBwQuantiles=quantiles(force[:,frame], masses),
            glutealTensionBwQuantiles=quantiles(tension[:,frame], masses),
        ))
    result = dict(
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="Exact nonparametric participant bootstrap. Sum precomputed per-participant losses and choose the minimum over the same virtual anatomies. Equal weight per participant. Percentiles describe bootstrap model sensitivity at peak vertical GRF.",
        subjects=subjects, candidates=args.count, frames=sum(map(len, frames)),
        seed=args.seed, uniqueResamples=len(weights), orderedResamples=int(multiplicities.sum()),
        candidateResampleCombinations=int(args.count*len(weights)),
        distinctSelectedCandidates=len(ids), modeIndex=int(ids[mode]), modeProbability=float(masses[mode]),
        heldOut=phases,
        timing=dict(
            cpuMs=statistics.median(timings), cpuSamplesMs=timings, cpu=study.hardware(),
            compiler=compiler, flags=flags, threads=args.threads, numpy=np.__version__,
            gpu=gpu['hardware'], gpuMs=statistics.median(gpu['ms']), gpuSamplesMs=gpu['ms'],
            speedup=statistics.median(timings)/statistics.median(gpu['ms']),
            modalAppId=app_id, rpcWallSeconds=wall, winnerAgreement=len(weights),
            scope="CPU wall vs GPU device time for both subject scoring and all weighted fits, resident arrays, after warm-up. RPC includes cloud provisioning and candidate regeneration.",
        ),
    )
    study.dump(study.DATA/'bootstrap.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
