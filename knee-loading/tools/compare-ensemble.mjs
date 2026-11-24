import { readFileSync, writeFileSync } from 'node:fs';

const [cudaPath, nativePath, webGpuPath, output] = process.argv.slice(2);
if (!cudaPath || !nativePath || !webGpuPath || !output) {
  throw new Error('Usage: node knee-loading/tools/compare-ensemble.mjs H100.json NATIVE.json WEBGPU.json OUTPUT.json');
}

const [cuda, native, webGpu] = [cudaPath, nativePath, webGpuPath].map((path) => JSON.parse(readFileSync(path, 'utf8')));
if (![native, webGpu].every((report) => JSON.stringify(report.workload) === JSON.stringify(cuda.workload)
  && report.repeats === cuda.repeats && report.rows.length === cuda.rows.length)) {
  throw new Error('The reports must use the same gait records, parameter pool and repeat count.');
}

const checked = (backend) => backend.agreement.mismatchCount === 0
  && backend.agreement.topCandidateAgreement.shared === backend.agreement.topCandidateAgreement.size;

const rows = cuda.rows.map((row, index) => {
  const cpu = native.rows[index];
  const gpu = webGpu.rows[index];
  if (![cpu, gpu].every((other) => other.parametersSha256 === row.parametersSha256 && other.scenarios === row.scenarios)) {
    throw new Error(`Input comparison failed for ${row.scenarios} candidates.`);
  }
  if (!checked(row.cuda) || row.cuda.baselineAgreement.outOfTolerance || !checked(gpu.webGpu)
    || !cpu.nativeCpu.every(checked)) {
    throw new Error(`Score comparison failed for ${row.scenarios} candidates.`);
  }
  const fastestCpu = cpu.nativeCpu.reduce((best, item) => item.batchMs < best.batchMs ? item : best);
  return {
    scenarios: row.scenarios,
    frames: cuda.workload.frames,
    m4NativeMs: fastestCpu.batchMs,
    m4NativeWorkers: fastestCpu.workers,
    m4WebGpuMs: gpu.webGpu.batchMs,
    h100BatchMs: row.cuda.optimized.batchMs,
    h100KernelMs: row.cuda.optimized.kernelMs,
    h100BlockBatchMs: row.cuda.blockBaseline.batchMs,
    h100BlockKernelMs: row.cuda.blockBaseline.kernelMs,
    topCandidatesMatched: row.cuda.agreement.topCandidateAgreement.shared,
    comparedScores: row.scenarios,
  };
});

const smaller = cuda.rows.find(({ scenarios }) => scenarios === 32_768) ?? cuda.rows[0];
const largest = cuda.rows.at(-1);
writeFileSync(output, `${JSON.stringify({
  repeats: cuda.repeats,
  devices: { cloud: cuda.modal.gpu, laptopCpu: native.hardware.cpu, laptopGpu: webGpu.rows[0].webGpu.adapter },
  modalRemoteSeconds: cuda.modal.remoteCallSeconds,
  rows,
  holdout: {
    smallCandidates: smaller.scenarios,
    bigCandidates: largest.scenarios,
    smallMedialRmseBw: smaller.calibration.medialRmseBw,
    bigMedialRmseBw: largest.calibration.medialRmseBw,
    smallLateralRmseBw: smaller.calibration.lateralRmseBw,
    bigLateralRmseBw: largest.calibration.lateralRmseBw,
  },
}, null, 2)}\n`);
