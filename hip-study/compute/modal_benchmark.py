"""Run the shared hip-score kernel on a single Modal H100.

modal run hip-study/compute/modal_benchmark.py --inputs /path/benchmark-input.npz

The input contains float32 frames, parameters and laptop cpuScores. The benchmark
uploads only features and model parameters; parity is computed on the laptop.
"""
import datetime
import hashlib
import io
import json
import statistics
import time
from pathlib import Path

import modal
import numpy as np


COMPUTE = Path(__file__).resolve().parent
app = modal.App("hip-loading-ensemble")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("cupy-cuda12x==13.6.0", "numpy==2.2.6")
    .add_local_file(COMPUTE / "study.py", "/root/study.py")
)
DEFAULT_SIZES = (4096, 65536, 262144, 1048576)


def identity():
    import cupy as cp
    import subprocess

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"].decode()
    if "H100" not in name:
        raise RuntimeError(f"Expected H100, received {name}")
    return {
        "device": name,
        "gpuMemoryBytes": properties["totalGlobalMem"],
        "multiprocessors": properties["multiProcessorCount"],
        "computeCapability": cp.cuda.Device().compute_capability,
        "cudaRuntime": cp.cuda.runtime.runtimeGetVersion(),
        "cupy": cp.__version__,
        "driverVersion": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }


@app.function(image=image, gpu="H100!", timeout=300, scaledown_window=15)
def run_gpu(source, encoded_input, sizes, repeats, warm_up=True, seed=None):
    import cupy as cp

    hardware = identity()
    if not encoded_input:
        smoke = cp.RawKernel(
            'extern "C" __global__ void square(float *x) { x[0] *= x[0]; }',
            "square",
        )
        v = cp.asarray([3], dtype=cp.float32)
        smoke((1,), (1,), (v,))
        if v.get().item() != 9:
            raise RuntimeError("CUDA smoke test failed")
        return hardware

    with np.load(io.BytesIO(encoded_input), allow_pickle=False) as arrays:
        frames = arrays["frames"]
        if seed is None:
            parameters = arrays["parameters"]
        else:
            from study import sample

            sample_start = time.perf_counter()
            raw, parameters = sample(max(sizes), seed)
            del raw
            sample_ms = 1000 * (time.perf_counter() - sample_start)
    t0 = time.perf_counter()
    score = cp.RawKernel(source, "hip_score", options=("--std=c++14", "--fmad=false"))
    score.compile()
    compile_ms = (time.perf_counter() - t0) * 1000
    runs = []
    for count in sizes:
        t0 = time.perf_counter()
        dframes = cp.asarray(frames)
        dparams = cp.asarray(parameters[:count])
        out = cp.empty(count, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        upload_ms = (time.perf_counter() - t0) * 1000
        args = (dframes, dparams, out, np.int32(len(frames)), np.int32(count))
        block = 128
        grid = ((count + block - 1) // block,)
        if warm_up:
            score(grid, (block,), args)
            cp.cuda.get_current_stream().synchronize()
        times = []
        for _ in range(repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            score(grid, (block,), args)
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))
        if seed is not None:
            # Only winning indices and scores cross the network in compact mode.
            top_k = min(256, count)
            winners = cp.argsort(out)[:top_k]
            return {
                "indices": winners.get().tolist(), "scores": out[winners].get().tolist(),
                "parametersSha256": hashlib.sha256(parameters.tobytes()).hexdigest(),
                "samplingMs": sample_ms, "kernelMs": times[-1],
            }
        t0 = time.perf_counter()
        scores = out.get()
        download_ms = (time.perf_counter() - t0) * 1000
        median_ms = statistics.median(times)
        runs.append({
            "candidates": count, "frames": len(frames),
            "blockThreads": block,
            "kernelMs": median_ms, "kernelSamplesMs": times,
            "uploadAndAllocateMs": upload_ms, "downloadMs": download_ms,
            "candidateFramesPerSecond": count * len(frames) / (median_ms / 1000),
            "scores": scores.tobytes(),
        })
    return {"hardware": hardware, "compileMs": compile_ms, "runs": runs}


@app.local_entrypoint()
def main(inputs: str = "", output: str = "", repeats: int = 7, probe_only: bool = False, compact_seed: int = -1):
    if probe_only:
        print(json.dumps(run_gpu.remote("", b"", [], 0), indent=2))
        return
    if not inputs or repeats < 3:
        raise ValueError("Pass --inputs and at least three repetitions.")
    with np.load(inputs, allow_pickle=False) as arrays:
        frames, params, reference = (
            np.ascontiguousarray(arrays[key], dtype=np.float32)
            for key in ("frames", "parameters", "cpuScores")
        )
    if frames.ndim != 2 or frames.shape[1] != 8 or params.ndim != 2 or params.shape[1] != 9:
        raise ValueError("Expected frame stride 8 and parameter stride 9.")
    if reference.shape != (len(params),) or not len(params) or not len(frames):
        raise ValueError("CPU reference must cover every candidate.")
    if not all(np.all(np.isfinite(array)) for array in (frames, params, reference)):
        raise ValueError("Expected finite inputs and scores.")
    payload = io.BytesIO()
    np.savez(payload, frames=frames, parameters=params)
    sizes = sorted({size for size in DEFAULT_SIZES if size <= len(params)} | {len(params)})
    source = (COMPUTE / "search.cu").read_text()
    t0 = time.perf_counter()
    result = run_gpu.remote(source, payload.getvalue(), sizes, repeats)
    result["rpcWallSeconds"] = time.perf_counter() - t0
    # Measure a second real client request, with one pass and all scores returned.
    # It deliberately transfers the entire input again; resident throughput above
    # applies when the application can keep parameters on the GPU.
    t0 = time.perf_counter()
    single = run_gpu.remote(source, payload.getvalue(), [len(params)], 1, False)
    result["singlePassRpcSeconds"] = time.perf_counter() - t0
    result["singlePassKernelMs"] = single["runs"][0]["kernelMs"]
    if single["runs"][0]["scores"] != result["runs"][-1]["scores"]:
        raise RuntimeError("Repeated H100 runs gave different scores")
    if compact_seed >= 0:
        frame_message = io.BytesIO()
        np.savez(frame_message, frames=frames)
        compact_best = np.argsort(reference, kind="stable")[:min(256, len(params))]
        t0 = time.perf_counter()
        compact = run_gpu.remote(source, frame_message.getvalue(), [len(params)], 1, False, compact_seed)
        elapsed = time.perf_counter() - t0
        same_parameters = hashlib.sha256(params.tobytes()).hexdigest() == compact["parametersSha256"]
        agreement = np.intersect1d(compact_best, compact["indices"]).size
        if not same_parameters or agreement != compact_best.size:
            raise RuntimeError("Compact cloud search disagrees with the supplied candidate matrix or laptop ranking")
        result["compact"] = {
            "seed": compact_seed, "candidates": len(params), "rpcSeconds": elapsed,
            "requestArrayBytes": len(frame_message.getvalue()),
            "topKAgreement": int(agreement), "parametersSha256": compact["parametersSha256"],
            "samplingMs": compact["samplingMs"], "kernelMs": compact["kernelMs"],
        }
    result["modalAppId"] = app.app_id
    result["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["kernelSha256"] = hashlib.sha256(source.encode()).hexdigest()
    result["payloadSha256"] = hashlib.sha256(payload.getvalue()).hexdigest()
    result["notes"] = "Kernel time uses CUDA events after warm-up. RPC wall time includes scheduling, startup, network, compilation, and all repetitions, excluding app initialization."
    for run in result["runs"]:
        count = run["candidates"]
        candidate_scores = np.frombuffer(run.pop("scores"), np.float32)
        reference_scores = reference[:count]
        if len(candidate_scores) != count or not np.all(np.isfinite(candidate_scores)):
            raise RuntimeError("Invalid CUDA scores")
        k = min(256, count)
        top_gpu = np.argsort(candidate_scores, kind="stable")[:k]
        top_cpu = np.argsort(reference_scores, kind="stable")[:k]
        run["maxAbsoluteError"] = float(np.max(np.abs(candidate_scores - reference_scores)))
        run["topK"] = k
        run["topKAgreement"] = int(np.intersect1d(top_cpu, top_gpu).size)
        run["winningCandidate"] = int(top_gpu[0])
        run["winningCpuCandidate"] = int(top_cpu[0])
        run["allClose"] = bool(np.allclose(candidate_scores, reference_scores, rtol=5e-5, atol=1e-5))
    path = Path(output) if output else COMPUTE.parent / "data" / "benchmark-gpu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(row["allClose"] for row in result["runs"]):
        raise RuntimeError("CUDA/laptop score mismatch; inspect benchmark data.")


def offset_moment(frames, mm):
    """Apply a uniform medial/lateral lever perturbation, M_new = M + delta_x * Fz."""
    x = frames.copy()
    signed = x[:,2] - x[:,3] + np.float32(mm / 1000) * x[:,0]
    x[:,2] = np.maximum(signed, 0)
    x[:,3] = np.maximum(-signed, 0)
    return x


@app.function(image=image, gpu="H100!", timeout=300, scaledown_window=10)
def bootstrap_gpu(source, frame_arrays, weights_bytes, nsubjects, count, seed,
                  repeat, offsets_mm=(0,), warm_up=True):
    """Evaluate all bootstrap resamples using per-participant losses only once."""
    import cupy as cp
    from study import sample

    hardware = identity()
    raw_frames = [np.frombuffer(blob, np.float32).reshape(-1, 8) for blob in frame_arrays]
    frames = [[cp.asarray(offset_moment(x, mm)) for x in raw_frames] for mm in offsets_mm]
    weights = cp.asarray(np.frombuffer(weights_bytes, np.float64).reshape(-1, nsubjects))
    raw, parameters = sample(count, seed)
    del raw
    digest = hashlib.sha256(parameters.tobytes()).hexdigest()
    device_params = cp.asarray(parameters)
    kernel = cp.RawKernel(source, "hip_score", options=("--std=c++14", "--fmad=false"))
    kernel.compile()
    # All GPU timings include participant scores, FP64 weighted reduction, and
    # stable argmin for each resample. The source frames and parameters are resident.
    scores = cp.empty((nsubjects, count), dtype=cp.float32)
    winners = cp.empty((len(offsets_mm), len(weights)), dtype=cp.int64)
    stream = cp.cuda.get_current_stream()
    times = []
    for iteration in range(repeat + int(warm_up)):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        stream.synchronize()
        start.record()
        for j, case in enumerate(frames):
            for i, x in enumerate(case):
                kernel(((count + 127)//128,), (128,),
                       (x, device_params, scores[i], np.int32(len(x)), np.int32(count)))
            loss = scores.astype(cp.float64)
            for lo in range(0, len(weights), 16):
                hi = min(lo + 16, len(weights))
                winners[j,lo:hi] = cp.argmin(weights[lo:hi] @ loss, axis=1)
        end.record()
        end.synchronize()
        if not warm_up or iteration:
            times.append(cp.cuda.get_elapsed_time(start, end))
    values = winners.get()
    if len(offsets_mm) == 1:
        values = values[0]
    return {"hardware": hardware, "ms": times, "winners": values.tolist(),
            "parameterDigest": digest}
