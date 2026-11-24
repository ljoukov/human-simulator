"""Benchmark the same scoring expression and input arrays on Apple Metal."""
import argparse
import hashlib
import io
import json
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def metal_source(source):
    # MSL requires explicit address spaces. Keep the objective and accumulation
    # order from the C++/CUDA source, including any future changes to the model.
    body = source.split("template<typename T>", 1)[1].split("#if defined(__CUDACC__)", 1)[0]
    body = body.replace("const float *", "device const float *").replace("const T *", "device const T *")
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n#define HIP_DEVICE inline\n"
        "inline float hip_sqrt(float x) { return sqrt(x); }\n"
        "template<typename T>\n" + body + """
kernel void hip_score(device const float *frames [[buffer(0)]],
                      device const float *parameters [[buffer(1)]],
                      device float *scores [[buffer(2)]],
                      constant uint &nFrames [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    scores[index] = objective(frames, parameters + index*9, nFrames);
}
"""
    )


def run(inputs, output, repeats):
    with np.load(inputs, allow_pickle=False) as arrays:
        params, frames, reference = (np.ascontiguousarray(arrays[key], dtype=np.float32)
                                     for key in ("parameters", "frames", "cpuScores"))
    if params.shape != (len(reference), 9) or frames.shape != (len(frames), 8):
        raise ValueError("Unexpected frame, parameter or reference shape")
    if not all(np.isfinite(a).all() for a in (frames, params, reference)):
        raise ValueError("Nonfinite input")
    sizes = sorted({n for n in (4096, 65536, 262144, len(params)) if n <= len(params)})
    source = (ROOT / "compute/search.cu").read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        (tmp / "score.metal").write_text(metal_source(source.decode()))
        frames.tofile(tmp / "frames")
        params.tofile(tmp / "parameters")
        subprocess.run(["xcrun", "clang++", "-O3", "-std=c++17", "-fobjc-arc", "-framework", "Metal",
                        "-framework", "Foundation", ROOT / "metal/score.mm", "-o", tmp / "runner"], check=True)
        raw = subprocess.check_output([tmp / "runner", tmp / "score.metal", tmp / "frames", tmp / "parameters",
                                       tmp, str(repeats), *map(str, sizes)])
        result = json.loads(raw)
        for row in result["rows"]:
            count = row["candidates"]
            values = np.fromfile(tmp / f"{count}.bin", dtype=np.float32)
            target = reference[:count]
            if not np.allclose(values, target, atol=1e-5, rtol=5e-5):
                raise RuntimeError("Metal candidate scores disagree with the compiled CPU baseline")
            top = min(256, count)
            row.update(deviceMs=statistics.median(row["deviceSamplesMs"]),
                       dispatchAndWaitMs=statistics.median(row["dispatchAndWaitSamplesMs"]),
                       maxAbsError=float(np.max(abs(values-target))),
                       topK=top, topKAgreement=int(len(np.intersect1d(
                           np.argsort(values, kind="stable")[:top], np.argsort(target, kind="stable")[:top]))))
    payload = io.BytesIO()
    np.savez(payload, frames=frames, parameters=params)
    result.update(platform=platform.platform(), inputSha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                  payloadSha256=hashlib.sha256(payload.getvalue()).hexdigest(),
                  kernelSha256=hashlib.sha256(source).hexdigest())
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark-metal.json")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    results = run(args.inputs, args.output, args.repeats)
    print(results["device"], [(r["candidates"], r["deviceMs"]) for r in results["rows"]])
