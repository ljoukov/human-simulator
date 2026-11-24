import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { createGpuBackend } from '../src/backend.ts';
import type { SolveBatch } from '../src/calibration.ts';
import type { GaitDataset } from '../src/dataset.ts';
import { runStudy } from '../src/experiment.ts';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    candidates: { type: 'string', default: '16384' },
    retain: { type: 'string', default: '256' },
    gpu: { type: 'boolean' },
    output: { type: 'string', short: 'o' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (values.help || !positionals.length) {
  console.log('Usage: node knee-loading/tools/run-study.ts GAIT_DATA.json [...] [--candidates 16384] [--retain 256] [-o report.json]');
  console.log('For hardware WebGPU, use deno run --allow-read --allow-write --unstable-webgpu with the same script and add --gpu.');
  process.exit(positionals.length || values.help ? 0 : 1);
}

const datasets = await Promise.all(positionals.map(async (path) => JSON.parse(await readFile(path, 'utf8')) as GaitDataset));
const start = performance.now();
let adapter: string | undefined;
const solveGpu: SolveBatch = async (frames, parameters, observations) => {
  const gpu = await createGpuBackend(frames, observations);
  try {
    if (gpu.fallback) throw new Error('The runtime selected a software adapter. Use a hardware-backed WebGPU adapter.');
    adapter = gpu.adapter;
    return await gpu.solve(parameters);
  } finally {
    gpu.close();
  }
};
const report = await runStudy(datasets, {
  candidateCount: Number(values.candidates),
  retainCount: Number(values.retain),
  backend: values.gpu ? 'WebGPU' : 'CPU',
  solve: values.gpu ? solveGpu : undefined,
});
if (adapter) report.audit.backend = `WebGPU · ${adapter}`;
const destination = values.output ?? new URL('../data/report.json', import.meta.url);
await writeFile(destination, JSON.stringify(report));
console.log(`Wrote ${report.study.comparisons.length} comparisons with ${report.audit.backend} in ${((performance.now() - start) / 1000).toFixed(2)} s.`);
for (const subject of report.audit.subjects) {
  console.log(`${subject.id} | calibration ${subject.calibration.length}, held out ${subject.evaluation.length}`);
  for (const [compartment, errors] of Object.entries(subject.error)) {
    console.log(`  ${compartment}: mechanics RMSE ${errors.rmseBw?.toFixed(3) ?? '—'} BW, moments RMSE ${errors.momentsRmseBw?.toFixed(3) ?? '—'} BW, band coverage ${errors.bandCoverage == null ? '—' : (100 * errors.bandCoverage).toFixed(1) + '%'}`);
  }
}
