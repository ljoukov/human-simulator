"""Run the gait calibration benchmark in an ephemeral Modal H100 container."""

import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import modal


REMOTE_ROOT = Path("/opt/gait-benchmark")
NODE_VERSION = "24.13.0"
NODE_ARCHIVE = f"node-v{NODE_VERSION}-linux-x64.tar.xz"
NODE_SHA256 = "e798599612f4bb71333a3397ab0d095fd62214e115aea45aa858a145fc72d67e"
CUDA_IMAGE = "nvidia/cuda:12.8.1-base-ubuntu22.04@sha256:e869de8a68e95272e51c4c7071b1d39941aaab707f9bc051a99062a3791448b1"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("ca-certificates", "curl", "xz-utils", "cuda-nvcc-12-8=12.8.93-1")
    .run_commands(
        f"cd /tmp && curl -fsSLO https://nodejs.org/dist/v{NODE_VERSION}/{NODE_ARCHIVE} "
        f"&& printf '%s  %s\\n' {NODE_SHA256} {NODE_ARCHIVE} | sha256sum --check "
        f"&& tar -xJf {NODE_ARCHIVE} -C /usr/local --strip-components=1 "
        f"&& rm {NODE_ARCHIVE}"
    )
)

if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for file in (
        "knee-loading/tools/gait-ensemble.mjs",
        "knee-loading/tools/h100_scores.cu",
        "knee-loading/tools/contact_score.hpp",
        "knee-loading/src/mechanics.ts",
        "knee-loading/src/dataset.ts",
        "knee-loading/src/sampling.ts",
        "knee-loading/src/backend-calibration.ts",
        "knee-loading/data/gait.json",
    ):
        image = image.add_local_file(str(root / file), str(REMOTE_ROOT / file))

app = modal.App("gait-ensemble-h100")


@app.function(image=image, gpu="H100", cpu=2, memory=4096, timeout=300, max_containers=1)
def benchmark() -> dict:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(gpu) != 1 or "H100" not in gpu[0]:
        raise RuntimeError(f"Expected one H100, received {gpu}")
    name, memory_mib, driver = map(str.strip, gpu[0].split(","))
    print(f"GPU: {name}, {memory_mib} MiB, driver {driver}", flush=True)

    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "h100-scores"
        output = Path(directory) / "report.json"
        started = perf_counter()
        subprocess.run(
            [
                "nvcc", "-O3", "-std=c++17", "--fmad=false", "-arch=sm_90",
                "knee-loading/tools/h100_scores.cu", "-o", str(executable),
            ],
            check=True,
            cwd=REMOTE_ROOT,
        )
        compile_seconds = perf_counter() - started
        started = perf_counter()
        subprocess.run(
            ["node", "knee-loading/tools/gait-ensemble.mjs", "--cuda", str(executable), "--output", str(output)],
            check=True,
            cwd=REMOTE_ROOT,
        )
        runner_seconds = perf_counter() - started
        report = json.loads(output.read_text(encoding="utf-8"))

    report["modal"] = {
        "gpu": name,
        "gpuMemoryMiB": int(memory_mib),
        "driverVersion": driver,
        "compileSeconds": compile_seconds,
        "runnerSeconds": runner_seconds,
    }
    return report


@app.local_entrypoint()
def main(output: str = "knee-loading/data/gait-ensemble-h100.json"):
    started = perf_counter()
    report = benchmark.remote()
    report["modal"]["remoteCallSeconds"] = perf_counter() - started
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
