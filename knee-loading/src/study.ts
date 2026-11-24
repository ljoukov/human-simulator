import { runBenchmark } from './backend.ts';
import { prepareCalibrationBenchmark } from './backend-calibration.ts';
import { encodeFrames, validateDataset, type GaitDataset } from './dataset.ts';
import type { StudyReport } from './experiment.ts';
import { sampleParameters } from './sampling.ts';
import type { StudyBenchmark, StudyView } from './view-types.ts';

const files = import.meta.glob<string>(
  [
    '../data/gait.json', '../data/gc5.json', '../data/report.json', '../data/benchmark.json',
    '../data/patella-oks001.json', '../data/patellar-pressure.json', '../data/whole-body-motion.json',
  ],
  { eager: true, query: '?url', import: 'default' },
);

async function loadJson<T>(name: string): Promise<T | undefined> {
  const url = files[`../data/${name}.json`];
  if (!url) return undefined;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Study file ${name} could not be loaded (HTTP ${response.status}).`);
  return response.json() as Promise<T>;
}

const GROUND_TRUTH_SOURCES = [
  { label: 'SimTK Knee Loads Grand Challenge', url: 'https://simtk.org/projects/kneeloads' },
  { label: 'Knee-load validation archive', url: 'https://simtk.org/projects/statop_val' },
  { label: 'Walter et al., instrumented-implant study', url: 'https://doi.org/10.1002/jor.21142' },
];

interface PatellaAudit {
  source: { dataset: string; datasetDoi: string };
  passive: {
    measuredKneeFlexionRangeDeg: [number, number];
    samples: {
      measuredPatellaRotationChangeDeg: number;
      rajagopalPatellaRotationChangeDeg: number;
      glbPatellaRotationChangeDeg: number;
    }[];
  };
  controlledLoads: { changeFrom20To600N: { translationMm: number }[] };
}

function patellaEvidence(audit: PatellaAudit | undefined): StudyView['patella'] {
  const endpoint = audit?.passive.samples.at(-1);
  if (!audit || !endpoint) return undefined;
  const [flexionStartDeg, flexionEndDeg] = audit.passive.measuredKneeFlexionRangeDeg;
  const shifts = audit.controlledLoads.changeFrom20To600N.map(({ translationMm }) => translationMm);
  return {
    flexionStartDeg, flexionEndDeg,
    measuredRotationDeg: endpoint.measuredPatellaRotationChangeDeg,
    referenceRotationDeg: endpoint.rajagopalPatellaRotationChangeDeg,
    viewerRotationDeg: endpoint.glbPatellaRotationChangeDeg,
    summary: `Open Knee(s) specimen oks001 had less passive patellar rotation than either kinematic model. In separate held-load tests, increasing the nominal quadriceps load from 20 N to 600 N shifted the patellar origin by ${Math.min(...shifts).toFixed(2)}–${Math.max(...shifts).toFixed(2)} mm across five flexion angles. This compares one cadaver specimen and is separate from the walking force experiment.`,
    source: { label: audit.source.dataset, url: audit.source.datasetDoi },
  };
}

export async function loadStudy(): Promise<StudyView> {
  const [report, benchmark, patella] = await Promise.all([
    loadJson<StudyReport>('report'), loadJson<StudyBenchmark>('benchmark'), loadJson<PatellaAudit>('patella-oks001'),
  ]);
  const downloads = [
    { name: 'report', label: 'Study results' },
    { name: 'benchmark', label: 'CPU/GPU timings' },
    { name: 'gait', label: 'GC4 stance inputs' },
    { name: 'gc5', label: 'GC5 stance inputs' },
    { name: 'patella-oks001', label: 'Patellar tracking audit' },
    { name: 'patellar-pressure', label: 'Measured patellar pressure grids' },
    { name: 'whole-body-motion', label: 'Whole-body motion and ground reactions' },
  ].flatMap(({ name, label }) => {
    const filename = `${name}.json`;
    const url = files[`../data/${filename}`];
    return url ? [{ label, url, filename }] : [];
  });
  const evidence = { benchmark, benchmarkAvailable: Boolean(files['../data/gait.json']), patella: patellaEvidence(patella), downloads };
  if (report) {
    return { ...report.study, ...evidence };
  }
  return {
    dataset: 'Instrumented knee gait',
    summary: 'Compare knee compression estimated from recorded gait against forces measured by an instrumented knee implant.',
    sources: GROUND_TRUTH_SOURCES,
    methods: [
      'The experiment calibrates the model on ordinary walking, then evaluates implant forces withheld from calibration.',
      'The same force and moment equations run on the CPU and on WebGPU. The benchmark includes output transfer and readback.',
    ],
    comparisons: [],
    ...evidence,
    notice: 'The measured trial bundle is required to generate model results. See the experiment README for preparation commands.',
  };
}

export async function runLiveBenchmark(onProgress?: (stage: string) => void): Promise<StudyBenchmark> {
  const original = await loadJson<GaitDataset>('gait');
  if (!original) throw new Error('No prepared measured gait recording is available for the benchmark.');
  const { trials } = validateDataset(original);
  const trial = trials.find(({ condition }) => condition === 'normal') ?? trials[0];
  onProgress?.(`Benchmarking recording ${trial.id}…`);
  const waveform = await runBenchmark(encodeFrames([trial]), sampleParameters(8_192),
    { batches: [128, 1024, 8192], repeats: 3, onProgress });
  const sameSubject = trials.filter(({ subject, condition, role }) => subject === trial.subject
    && condition === trial.condition && (role === 'calibration' || role == null));
  const { frames, observations } = prepareCalibrationBenchmark(sameSubject.length ? sameSubject : [trial]);
  const calibration = await runBenchmark(frames, sampleParameters(65_536),
    { batches: [1024, 8192, 65_536], repeats: 3, observations, onProgress });
  return { ...waveform, calibration };
}
