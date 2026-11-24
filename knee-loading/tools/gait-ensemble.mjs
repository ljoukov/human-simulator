import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { cpus, endianness, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { prepareCalibrationBenchmark, withinStance } from '../src/backend-calibration.ts';
import { encodeFrames, validateDataset } from '../src/dataset.ts';
import { FRAME_STRIDE, OUTPUT_STRIDE, PARAM_STRIDE, solveCpu } from '../src/mechanics.ts';
import { quantile, rankCandidates, sampleParameters } from '../src/sampling.ts';

const { values } = parseArgs({ options: {
  cuda: { type: 'string' },
  native: { type: 'string' },
  webgpu: { type: 'boolean' },
  threads: { type: 'string' },
  batches: { type: 'string', default: '1024,32768,262144,1048576' },
  repeats: { type: 'string', default: '7' },
  output: { type: 'string', short: 'o' },
  help: { type: 'boolean', short: 'h' },
} });
if (values.help) {
  console.log('node knee-loading/tools/gait-ensemble.mjs [--cuda EXECUTABLE] [--native EXECUTABLE --threads 1,4,10] [--batches 1024,32768,262144,1048576] [--repeats 7] [-o FILE]');
  console.log('Use Deno with --unstable-webgpu and pass --webgpu to time a hardware WebGPU adapter.');
  process.exit(0);
}

function integers(csv, minimum, maximum) {
  const numbers = csv.split(',').map(Number);
  if (numbers.some((number) => !Number.isSafeInteger(number) || number < minimum || number > maximum)) {
    throw new RangeError(`Expected integers between ${minimum} and ${maximum}.`);
  }
  return [...new Set(numbers)].sort((a, b) => a - b);
}

const sizes = integers(values.batches, 2, 4_194_304);
const repeats = Number(values.repeats);
if (!Number.isSafeInteger(repeats) || repeats < 1 || repeats > 50) throw new RangeError('Expected 1–50 repeats.');
if (endianness() !== 'LE') throw new Error('The benchmark binary format requires little-endian float32.');
const coreCount = cpus().length;
const threads = integers(values.threads ?? `1,${Math.max(1, Math.floor(coreCount / 2))},${coreCount}`, 1, 256);
const seed = 0xf8ea9e83;
const absTolerance = 2e-5;
const relTolerance = 1e-5;
const dataset = validateDataset(JSON.parse(readFileSync(new URL('../data/gait.json', import.meta.url), 'utf8')));
const training = dataset.trials.find(({ id }) => id === 'ngait_og5-stance-1');
if (!training) throw new Error('The benchmark requires the GC4-JW ordinary-walking reference stance.');
const holdout = dataset.trials.filter(({ role, subject }) => subject === training.subject && role === 'evaluation');
const { frames, observations } = prepareCalibrationBenchmark([training]);
const frameCount = frames.length / FRAME_STRIDE;
const pool = sampleParameters(sizes.at(-1), seed);
const retainedCount = 256;

const bytes = (array) => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const sha256 = (...arrays) => {
  const hash = createHash('sha256');
  arrays.forEach((array) => hash.update(bytes(array)));
  return hash.digest('hex');
};
const top = (scores) => rankCandidates(scores, retainedCount);

function agreement(expected, actual, topExpected) {
  if (actual.length !== expected.length) throw new Error('The backend returned the wrong number of scores.');
  let maxAbsError = 0;
  let mismatchCount = 0;
  for (let index = 0; index < expected.length; index++) {
    const difference = Math.abs(expected[index] - actual[index]);
    if (!Number.isFinite(difference)) {
      mismatchCount++;
      continue;
    }
    maxAbsError = Math.max(maxAbsError, difference);
    if (difference > absTolerance + relTolerance * Math.max(Math.abs(expected[index]), Math.abs(actual[index]))) {
      mismatchCount++;
    }
  }
  const accepted = new Set(topExpected);
  const shared = mismatchCount ? null : top(actual).filter((index) => accepted.has(index)).length;
  return { maxAbsError, mismatchCount, topCandidateAgreement: { size: accepted.size, shared } };
}

const validationFrames = holdout.flatMap(({ frames }) => frames.filter(withinStance));
const validationInput = encodeFrames(holdout.map((trial) => ({ ...trial, frames: trial.frames.filter(withinStance) })));

function evaluateHoldout(candidateIds) {
  const ensemble = new Float32Array(candidateIds.length * PARAM_STRIDE);
  for (const [slot, id] of candidateIds.entries()) {
    ensemble.set(pool.subarray(id * PARAM_STRIDE, (id + 1) * PARAM_STRIDE), slot * PARAM_STRIDE);
  }
  const predicted = solveCpu(validationInput, ensemble);
  const errors = [0, 0];
  const counts = [0, 0];
  for (const [frame, measured] of validationFrames.entries()) {
    for (const [compartment, target] of [measured.medialBw, measured.lateralBw].entries()) {
      if (target == null) continue;
      const estimates = Float32Array.from(candidateIds, (_, candidate) =>
        predicted[(candidate * validationFrames.length + frame) * OUTPUT_STRIDE + compartment]);
      estimates.sort();
      errors[compartment] += (quantile(estimates, 0.5) - target) ** 2;
      counts[compartment]++;
    }
  }
  return {
    medialRmseBw: counts[0] ? Math.sqrt(errors[0] / counts[0]) : null,
    lateralRmseBw: counts[1] ? Math.sqrt(errors[1] / counts[1]) : null,
  };
}

function runExecutable(executable, arguments_, output) {
  const started = performance.now();
  const timings = JSON.parse(execFileSync(resolve(executable), arguments_, { encoding: 'utf8', timeout: 120_000 }));
  const processMs = performance.now() - started;
  const buffer = readFileSync(output);
  return { timings, processMs, scores: new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4) };
}

const report = {
  createdAt: new Date().toISOString(),
  hardware: { runtime: process.version, platform: process.platform, arch: process.arch, cpu: cpus()[0]?.model, logicalCores: coreCount },
  workload: {
    mode: 'quasi-static knee contact calibration', source: dataset.source.url, subject: training.subject,
    training: training.id, frames: frameCount, seed, poolSize: sizes.at(-1), retainedCount,
    holdout: holdout.map(({ id }) => id), holdoutFrames: validationFrames.length,
    frameInputsSha256: sha256(frames, observations),
  },
  repeats, absTolerance, relTolerance,
  rows: [],
};

// Use one JS reference pass for the entire pool. Every timed batch uses a prefix
// of that exact pool, so changing the search budget leaves earlier candidates intact.
const reference = solveCpu(frames, pool, undefined, observations);
let webGpu;
if (values.webgpu) {
  const { createGpuBackend } = await import('../src/backend.ts');
  const started = performance.now();
  webGpu = await createGpuBackend(frames, observations);
  report.webGpuSetupMs = performance.now() - started;
}
const scratch = values.cuda || values.native ? mkdtempSync(join(tmpdir(), 'gait-ensemble-')) : null;
try {
  for (const scenarios of sizes) {
    const parameters = pool.subarray(0, scenarios * PARAM_STRIDE);
    const expected = reference.subarray(0, scenarios);
    const best = top(expected);
    const row = {
      scenarios, frameSolves: scenarios * frameCount, parametersSha256: sha256(parameters),
      calibration: { bestScore: expected[best[0]], bestScenario: best[0], ...evaluateHoldout(best) },
    };
    if (scratch) {
      const input = join(scratch, 'input.bin');
      const output = join(scratch, 'scores.bin');
      const header = Buffer.alloc(8);
      header.writeUInt32LE(frameCount, 0);
      header.writeUInt32LE(scenarios, 4);
      writeFileSync(input, Buffer.concat([header, bytes(frames), bytes(observations), bytes(parameters)]));
      const run = (executable, extra = []) => {
        const { timings, processMs, scores } = runExecutable(executable, [input, output, String(repeats), ...extra], output);
        return { ...timings, processMs, agreement: agreement(expected, scores, best) };
      };
      if (values.cuda) row.cuda = run(values.cuda);
      if (values.native) row.nativeCpu = threads.map((threadCount) => run(values.native, [String(threadCount)]));
    }
    if (webGpu) {
      let result = await webGpu.solve(parameters);
      const batchSamplesMs = [];
      for (let repeat = 0; repeat < repeats; repeat++) {
        const began = performance.now();
        result = await webGpu.solve(parameters);
        batchSamplesMs.push(performance.now() - began);
      }
      row.webGpu = {
        adapter: webGpu.adapter,
        batchMs: quantile(batchSamplesMs.toSorted((left, right) => left - right), 0.5),
        batchSamplesMs,
        agreement: agreement(expected, result, best),
      };
    }
    report.rows.push(row);
    console.error(`Checked ${scenarios.toLocaleString('en-GB')} candidates × ${frameCount} frames`);
  }
} finally {
  webGpu?.close();
  if (scratch) rmSync(scratch, { recursive: true, force: true });
}

const json = `${JSON.stringify(report, null, 2)}\n`;
if (values.output) writeFileSync(values.output, json);
else process.stdout.write(json);
if (report.rows.some(({ cuda, nativeCpu, webGpu }) =>
  [cuda, webGpu, ...(nativeCpu ?? [])].filter(Boolean).some(({ agreement: validation, baselineAgreement }) =>
    validation.mismatchCount || validation.topCandidateAgreement.shared !== validation.topCandidateAgreement.size
    || baselineAgreement?.outOfTolerance))) {
  process.exitCode = 1;
}
