import { runBenchmark } from '../src/backend.ts'
import { prepareCalibrationBenchmark } from '../src/backend-calibration.ts'
import { encodeFrames, validateDataset, type GaitDataset } from '../src/dataset.ts'
import { sampleParameters } from '../src/sampling.ts'

const [inputArgument, outputArgument, trialArgument, ...unusedArguments] = Deno.args
const help = 'Usage: deno run --allow-read --allow-write knee-loading/tools/benchmark.ts [gait.json] [benchmark.json] [trial-id]'
if (inputArgument === '--help') {
  console.log(help)
  Deno.exit(0)
}
if (unusedArguments.length) throw new Error(help)

const input = inputArgument ?? new URL('../data/gait.json', import.meta.url)
const output = outputArgument ?? new URL('../data/benchmark.json', import.meta.url)
const dataset = validateDataset(JSON.parse(await Deno.readTextFile(input)) as GaitDataset)
const representative = trialArgument
  ? dataset.trials.find(({ id }) => id === trialArgument)
  : dataset.trials.find(({ condition, role }) => {
    const normalized = condition.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
    return role !== 'evaluation' && ['normal', 'normal_walking', 'normal_gait', 'level_walking'].includes(normalized)
  })
if (!representative) throw new Error('Choose the ID of a recorded ordinary-walking calibration trial.')

const waveformBatches = [128, 1024, 8192, 32768]
const calibrationBatches = [128, 1024, 8192, 32768, 131072, 262144]
const params = sampleParameters(calibrationBatches.at(-1)!)
const waves = await runBenchmark(encodeFrames([representative]), params, {
  batches: waveformBatches,
  repeats: 5,
  onProgress: (stage) => console.error(`Waveforms · ${stage}`),
})
const fit = prepareCalibrationBenchmark([representative])
const calibration = await runBenchmark(fit.frames, params, {
  batches: calibrationBatches,
  repeats: 5,
  observations: fit.observations,
  onProgress: (stage) => console.error(`Calibration scores · ${stage}`),
})

const report = {
  source: { title: dataset.source.title, url: dataset.source.url, kind: dataset.source.kind },
  trial: { id: representative.id, subject: representative.subject, condition: representative.condition, role: representative.role ?? null },
  runtime: `Deno ${Deno.version.deno} / ${Deno.build.os} ${Deno.build.arch}`,
  notes: [
    'Timings are medians after one warm-up at 128 scenarios. CPU and GPU setup are reported separately.',
    'Batch times include scenario uploads or worker communication and all result readback. Gait frames are resident after setup.',
    'Both calibration backends reduce to one squared error score per candidate. Candidate sorting and backend agreement are checked outside the timed region.',
  ],
  ...waves,
  calibration,
}
await Deno.writeTextFile(output, `${JSON.stringify(report, null, 2)}\n`)
console.log(JSON.stringify(report, null, 2))
if ([waves, calibration].some((workload) => !workload.gpu || workload.rows.some((row) => row.mismatchCount !== 0))) {
  Deno.exitCode = 1
}
