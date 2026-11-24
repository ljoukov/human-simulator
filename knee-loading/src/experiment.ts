import {
  COMPARTMENTS, filterParameters, fitProxies, measuredForce, solveBatchCpu, withinStance, type Proxies, type SolveBatch,
} from './calibration.ts';
import { encodeFrames, validateDataset, type GaitDataset, type GaitTrial } from './dataset.ts';
import { DEFAULT_PARAMETERS, FRAME_STRIDE, MOMENT_RESIDUAL_TOLERANCE_M, OUTPUT_STRIDE, PARAM_STRIDE } from './mechanics.ts';
import { mean, quantile, sampleParameters } from './sampling.ts';
import type { Compartment, ComparisonView, MetricView, Samples, StudyView, TrialView, WaveformView } from './view-types.ts';

type Statistic = 'early' | 'late' | 'mean';
const STATISTICS: readonly Statistic[] = ['early', 'late', 'mean'];
const OFFLOAD_THRESHOLD_BW = 0.10;
const statisticLabel: Record<Statistic, string> = {
  early: 'Early-stance peak',
  late: 'Late-stance peak',
  mean: 'Mean stance load',
};

interface TraceTrial {
  source: GaitTrial;
  phases: number[];
  measured: Samples;
  adduction: Samples;
  moments: Samples;
  projected: Float32Array;
  unclosed: number;
  nominal: Float32Array;
}

interface Group {
  view: TrialView;
  estimates: Map<Compartment, TraceTrial[]>;
}

export interface ErrorMetrics {
  frames: number;
  rmseBw: number | null;
  maeBw: number | null;
  bandCoverage: number | null;
  unclosedFraction: number | null;
  nominalRmseBw: number | null;
  adductionRmseBw: number | null;
  momentsRmseBw: number | null;
}

export interface StudyReport {
  study: StudyView;
  audit: {
    candidateCount: number;
    retainCount: number;
    seed: number;
    backend: string;
    sources: GaitDataset['source'][];
    subjects: {
      source: string;
      id: string;
      calibration: string[];
      evaluation: string[];
      error: Partial<Record<Compartment, ErrorMetrics>>;
    }[];
  };
}

const normalizeCondition = (condition: string) => condition.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_');
const isNormal = ({ condition }: GaitTrial) => ['normal', 'normal_walking', 'normal_gait', 'level_walking'].includes(normalizeCondition(condition));
const humanize = (condition: string) => condition.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase());
const digits = (value: number | null) => value == null ? '—' : value.toFixed(2);

function splitTrials(trials: GaitTrial[]) {
  const normal = trials.filter(isNormal);
  const explicit = trials.some(({ role }) => role != null);
  const calibration = explicit
    ? trials.filter((trial) => trial.role === 'calibration' && isNormal(trial))
    : normal.slice(0, Math.max(1, Math.floor(normal.length / 2)));
  if (trials.some((trial) => trial.role === 'calibration' && !isNormal(trial))) {
    throw new Error('Calibration is reserved for ordinary walking.');
  }
  return {
    calibration,
    testNormal: normal.filter((trial) => !calibration.includes(trial)),
    altered: trials.filter((trial) => !isNormal(trial)),
  };
}

async function traceTrials(
  trials: GaitTrial[],
  ensemble: Float32Array,
  proxies: Partial<Record<Compartment, Proxies>>,
  solve: SolveBatch,
): Promise<Group['estimates']> {
  const estimates: Group['estimates'] = new Map();
  const count = ensemble.length / PARAM_STRIDE;
  const inputs = encodeFrames(trials);
  const batches = new Float32Array(ensemble.length + PARAM_STRIDE);
  batches.set(ensemble);
  batches.set(DEFAULT_PARAMETERS, ensemble.length);
  const outputs = await solve(inputs, batches);
  const frameCount = inputs.length / FRAME_STRIDE;
  let cursor = 0;
  for (const source of trials) {
    const phases = source.frames.map((frame) => frame.stancePct);
    const stanceLength = phases.length;
    const activeFrames = source.frames.flatMap((frame, index) => withinStance(frame) ? [index] : []);
    let unclosed = 0;
    for (let scenario = 0; scenario < count; scenario++) {
      for (const frame of activeFrames) {
        const absolute = (scenario * frameCount + cursor + frame) * OUTPUT_STRIDE;
        if (Math.abs(outputs[absolute + 2]) > MOMENT_RESIDUAL_TOLERANCE_M
          || Math.abs(outputs[absolute + 3]) > MOMENT_RESIDUAL_TOLERANCE_M) unclosed++;
      }
    }
    for (const compartment of COMPARTMENTS) {
      const measured = source.frames.map((frame) => measuredForce(frame, compartment));
      if (measured.every((value) => value == null)) continue;
      const projected = new Float32Array(count * stanceLength);
      const nominal = new Float32Array(stanceLength);
      const force = (scenario: number, frame: number) => {
        const index = (scenario * frameCount + cursor + frame) * OUTPUT_STRIDE;
        const medial = outputs[index];
        const lateral = outputs[index + 1];
        if (compartment === 'medial') return medial;
        if (compartment === 'lateral') return lateral;
        return medial + lateral;
      };
      for (let scenario = 0; scenario < count; scenario++) {
        for (let frame = 0; frame < stanceLength; frame++) {
          projected[scenario * stanceLength + frame] = force(scenario, frame);
        }
      }
      for (let frame = 0; frame < stanceLength; frame++) nominal[frame] = force(count, frame);
      const baseline = proxies[compartment];
      const estimate: TraceTrial = {
        source, phases, measured, projected, nominal,
        adduction: source.frames.map((frame) => baseline?.adduction?.(frame) ?? null),
        moments: source.frames.map((frame) => baseline?.moments?.(frame) ?? null),
        unclosed: activeFrames.length ? unclosed / (count * activeFrames.length) : 0,
      };
      if (!estimates.has(compartment)) estimates.set(compartment, []);
      estimates.get(compartment)!.push(estimate);
    }
    cursor += stanceLength;
  }
  return estimates;
}

function selectTraces(trials: GaitTrial[], estimates: Group['estimates']): Group['estimates'] {
  const selected = new Map<Compartment, TraceTrial[]>();
  const included = new Set(trials);
  for (const [compartment, traces] of estimates) {
    const filtered = traces.filter(({ source }) => included.has(source));
    if (filtered.length) selected.set(compartment, filtered);
  }
  return selected;
}

function waveforms(trials: TraceTrial[], count: number): WaveformView {
  const phases = trials[0].phases;
  const result: WaveformView = { measured: [], modeled: [], low: [], high: [], proxy: [] };
  for (let frame = 0; frame < phases.length; frame++) {
    const ensembleMean = Array.from({ length: count }, (_, scenario) => {
      return trials.reduce((sum, trial) => sum + trial.projected[scenario * phases.length + frame], 0) / trials.length;
    }).sort((a, b) => a - b);
    result.measured.push(mean(trials.map((trial) => trial.measured[frame])));
    result.modeled.push(quantile(ensembleMean, 0.5));
    result.low.push(quantile(ensembleMean, 0.05));
    result.high.push(quantile(ensembleMean, 0.95));
    result.proxy!.push(mean(trials.map((trial) => trial.moments[frame] ?? trial.adduction[frame])));
  }
  return result;
}

function buildGroup(label: string, trials: GaitTrial[], estimates: Group['estimates'], count: number): Group {
  const phase = trials[0].frames.map((frame) => frame.stancePct);
  if (trials.some(({ frames }) => frames.length !== phase.length
    || frames.some(({ stancePct }, index) => Math.abs(stancePct - phase[index]) > 1e-6))) {
    throw new Error('Resample trials to the same stance grid before building group waveforms.');
  }
  const kneeFlexionDeg = phase.map((_, index) => mean(trials.map(({ frames }) => frames[index].kneeFlexionDeg ?? null)));
  const compartments: TrialView['compartments'] = {};
  for (const [compartment, trace] of estimates) compartments[compartment] = waveforms(trace, count);
  return { view: { label, phase, kneeFlexionDeg, compartments, trialCount: trials.length }, estimates };
}

function statistic(samples: ArrayLike<number | null>, phase: number[], metric: Statistic) {
  if (metric === 'mean') {
    let area = 0;
    let span = 0;
    for (let index = 1; index < phase.length; index++) {
      const left = samples[index - 1];
      const right = samples[index];
      if (left == null || right == null) continue;
      const width = phase[index] - phase[index - 1];
      area += (left + right) * width / 2;
      span += width;
    }
    return span > 0 ? area / span : null;
  }

  const [start, end] = metric === 'early' ? [5, 45] : [55, 95];
  let maximum: number | null = null;
  for (let index = 0; index < phase.length; index++) {
    const force = samples[index];
    if (force != null && phase[index] >= start && phase[index] <= end) {
      maximum = maximum == null ? force : Math.max(maximum, force);
    }
  }
  return maximum;
}

function impulse(samples: ArrayLike<number | null>, trial: TraceTrial) {
  if (trial.source.stanceSeconds == null || Array.from(samples).some((value) => value == null)) return null;
  const timeMean = statistic(samples, trial.phases, 'mean');
  return timeMean == null ? null : timeMean * trial.source.stanceSeconds;
}

function comparisonMetric(
  reference: TraceTrial[], intervention: TraceTrial[], compartment: Compartment, metric: Statistic | 'impulse', count: number,
): MetricView {
  const scalar = (samples: ArrayLike<number | null>, trial: TraceTrial) => metric === 'impulse'
    ? impulse(samples, trial)
    : statistic(samples, trial.phases, metric);
  const delta = (values: (trial: TraceTrial) => ArrayLike<number | null>) => {
    const before = mean(reference.map((trial) => scalar(values(trial), trial)));
    const after = mean(intervention.map((trial) => scalar(values(trial), trial)));
    return before == null || after == null ? null : after - before;
  };
  const paired = Array.from({ length: count }, (_, scenario) => delta(({ projected, phases }) => {
    return projected.subarray(scenario * phases.length, (scenario + 1) * phases.length);
  })).filter((value) => value != null).sort((a, b) => a - b);
  return {
    label: metric === 'impulse' ? 'Stance impulse' : statisticLabel[metric],
    unit: metric === 'impulse' ? 'BW·s' : 'BW', compartment, delta: true,
    measured: delta((trial) => trial.measured),
    model: quantile(paired, 0.5),
    modelLow: quantile(paired, 0.05),
    modelHigh: quantile(paired, 0.95),
    proxy: delta((trial) => trial.moments.map((value, index) => value ?? trial.adduction[index])),
    ...(metric !== 'impulse' && paired.length > 0 ? {
      offloadShare: paired.filter((value) => value <= -OFFLOAD_THRESHOLD_BW).length / paired.length,
      offloadThresholdBw: OFFLOAD_THRESHOLD_BW,
    } : {}),
  };
}

function earlyAdductionChange(reference: GaitTrial[], intervention: GaitTrial[]): number | null {
  const peak = ({ frames }: GaitTrial) => statistic(frames.map(({ kamNmKg }) => kamNmKg),
    frames.map(({ stancePct }) => stancePct), 'early');
  const before = mean(reference.map(peak));
  const after = mean(intervention.map(peak));
  return before == null || after == null || before <= 1e-6 ? null : 100 * (after - before) / before;
}

function score(trials: TraceTrial[], count: number): ErrorMetrics {
  const errors = { fitted: [] as number[], nominal: [] as number[], adduction: [] as number[], moments: [] as number[] };
  let covered = 0;
  for (const { source, phases, measured, nominal, projected, adduction, moments } of trials) {
    for (let frame = 0; frame < phases.length; frame++) {
      const truth = measured[frame];
      if (truth == null || !withinStance(source.frames[frame])) continue;
      const distribution = Array.from({ length: count }, (_, scenario) => projected[scenario * phases.length + frame])
        .sort((a, b) => a - b);
      const median = quantile(distribution, 0.5)!;
      errors.fitted.push(median - truth);
      errors.nominal.push(nominal[frame] - truth);
      if (adduction[frame] != null) errors.adduction.push(adduction[frame]! - truth);
      if (moments[frame] != null) errors.moments.push(moments[frame]! - truth);
      if (truth >= quantile(distribution, 0.05)! && truth <= quantile(distribution, 0.95)!) covered++;
    }
  }
  const rms = (values: number[]) => values.length ? Math.sqrt(mean(values.map((value) => value ** 2))!) : null;
  return {
    frames: errors.fitted.length, rmseBw: rms(errors.fitted), maeBw: mean(errors.fitted.map(Math.abs)),
    nominalRmseBw: rms(errors.nominal), adductionRmseBw: rms(errors.adduction), momentsRmseBw: rms(errors.moments),
    bandCoverage: errors.fitted.length ? covered / errors.fitted.length : null,
    unclosedFraction: mean(trials.map((trial) => trial.unclosed)),
  };
}

function absoluteMetrics(compartment: Compartment, scores: ErrorMetrics): MetricView[] {
  return [
    { label: 'Held-out waveform RMSE', unit: 'BW', compartment, delta: false,
      measured: null, model: scores.rmseBw, proxy: scores.momentsRmseBw ?? scores.adductionRmseBw },
    { label: 'Measured frames inside the band', unit: '%', compartment, delta: false,
      measured: null, model: scores.bandCoverage == null ? null : 100 * scores.bandCoverage, proxy: null },
    { label: 'Solves within moment tolerance', unit: '%', compartment, delta: false,
      measured: null, model: scores.unclosedFraction == null ? null : 100 * (1 - scores.unclosedFraction), proxy: null },
  ];
}

export async function runStudy(datasets: GaitDataset[], {
  candidateCount = 16_384, retainCount = 256, seed = 0xf8ea9e83, backend = 'CPU', solve = solveBatchCpu,
}: { candidateCount?: number; retainCount?: number; seed?: number; backend?: string; solve?: SolveBatch } = {}): Promise<StudyReport> {
  if (!Number.isSafeInteger(retainCount) || retainCount < 2 || retainCount > candidateCount) {
    throw new RangeError('The retained ensemble must contain 2 to candidateCount scenarios.');
  }
  const candidates = sampleParameters(candidateCount, seed);
  const report: StudyReport = {
    study: {
      dataset: datasets.map(({ source }) => source.title).join(' · '),
      summary: '',
      sources: [],
      methods: [
        `The study samples ${candidateCount.toLocaleString('en-GB')} parameter sets per participant and retains ${retainCount} using only designated ordinary-walking implant force traces. Parameters are paired across conditions.`,
        'The prescribed-motion model balances sagittal and frontal knee moments with bounded extensor and flexor groups, a paired co-contraction term, and nonnegative medial and lateral contact forces. Geometry uses angle-dependent group leverage from Rajagopal2016. Missing flexion uses 15°.',
        'The two-moment comparator is a linear fit from external knee adduction and absolute external flexion moments. The audit also reports the adduction-only fit and an uncalibrated mechanics reference. All fits use the same calibration subset.',
        'Peaks are the maximum force in 5–45% and 55–95% stance. Mean load is integrated over normalized stance. Where physical stance time is available, force is also integrated in BW·s. Held-out waveform scores use 5–95% stance.',
        'Bands describe the central 90% of retained scenarios under the stated parameter ranges. They measure sensitivity to model assumptions. The offload share counts scenarios with a reduction of at least 0.10 body weights. Departures from sagittal or frontal moment balance greater than 0.005 BW·m are counted.',
        'The 3D study pose drives the generic atlas knee with mean sagittal knee flexion from the input inverse kinematics at each stance phase. The geometry and other joint motions follow the reference atlas.',
      ],
      comparisons: [],
      intervalLabel: 'Central 90% of retained scenarios',
    },
    audit: { candidateCount, retainCount, seed, backend, sources: [], subjects: [] },
  };

  for (const input of datasets) {
    const { trials, source } = validateDataset(input);
    report.study.sources.push({ label: `${source.title} · ${source.license}`, url: source.url });
    report.study.methods.push(...source.notes);
    report.audit.sources.push(source);
    const grouped = new Map<string, GaitTrial[]>();
    for (const trial of trials) {
      if (!grouped.has(trial.subject)) grouped.set(trial.subject, []);
      grouped.get(trial.subject)!.push(trial);
    }
    for (const [subject, subjectTrials] of grouped) {
      const { calibration, testNormal, altered } = splitTrials(subjectTrials);
      if (!calibration.length || !(testNormal.length || altered.length)) continue;
      const id = `${report.audit.sources.length}-${subject}`;
      const model = await filterParameters(calibration, candidates, retainCount, solve);
      const proxies = Object.fromEntries(COMPARTMENTS.map((compartment) => [compartment, fitProxies(calibration, compartment)]));
      const evaluation = [...testNormal, ...altered];
      const estimates = await traceTrials([...calibration, ...evaluation], model, proxies, solve);
      const scores = Object.fromEntries([...selectTraces(evaluation, estimates)].map(([compartment, traces]) => {
        return [compartment, score(traces, retainCount)];
      }));
      report.audit.subjects.push({
        source: source.title, id: subject,
        calibration: calibration.map(({ id }) => id),
        evaluation: evaluation.map(({ id }) => id),
        error: scores,
      });

      const ordinary = testNormal.length ? testNormal : calibration;
      const reference = buildGroup(testNormal.length ? 'Ordinary walking · held out' : 'Ordinary walking · calibration',
        ordinary, selectTraces(ordinary, estimates), retainCount);
      const byCondition = new Map<string, GaitTrial[]>();
      for (const trial of altered) {
        if (!byCondition.has(trial.condition)) byCondition.set(trial.condition, []);
        byCondition.get(trial.condition)!.push(trial);
      }
      const context = source.kind === 'trial'
        ? `${calibration.length} ordinary-walking observations used for calibration. ${evaluation.length} recorded observations withheld.`
        : 'Published condition-average curves digitized from figures. Ordinary walking is the calibration reference. Altered conditions were withheld.';

      for (const [condition, interventionTrials] of byCondition) {
        const label = humanize(condition);
        const intervention = buildGroup(label, interventionTrials, selectTraces(interventionTrials, estimates), retainCount);
        const metrics: MetricView[] = [];
        for (const [compartment, traces] of intervention.estimates) {
          const referenceTraces = reference.estimates.get(compartment);
          if (!referenceTraces?.length) continue;
          const metricKeys: (Statistic | 'impulse')[] = [...STATISTICS];
          if ([...referenceTraces, ...traces].every(({ source }) => source.stanceSeconds != null)) metricKeys.push('impulse');
          metrics.push(...metricKeys.map((metric) => comparisonMetric(referenceTraces, traces, compartment, metric, retainCount)));
          metrics.push(...absoluteMetrics(compartment, score(traces, retainCount)));
        }
        report.study.comparisons.push({
          id: `${id}-${normalizeCondition(condition)}`, groupId: id, subject, label, reference: reference.view,
          intervention: intervention.view, metrics, context, side: source.recordedSide,
          kamEarlyChangePercent: earlyAdductionChange(ordinary, interventionTrials),
        });
      }
      if (!altered.length && testNormal.length) {
        const fitted = buildGroup('Ordinary walking · calibration', calibration, selectTraces(calibration, estimates), retainCount);
        report.study.comparisons.push({
          id: `${id}-normal`, groupId: id, subject, label: 'Held-out ordinary walking', reference: fitted.view,
          intervention: reference.view, side: source.recordedSide, metrics: Object.entries(scores).flatMap(([compartment, result]) =>
            absoluteMetrics(compartment as Compartment, result as ErrorMetrics)),
          context, metricsTitle: 'Held-out accuracy',
          metricsIntro: 'Waveform error, empirical interval coverage, and balance across held-out ordinary walking.',
        });
      }
    }
  }

  if (!report.study.comparisons.length) {
    report.study.summary = 'A comparison requires ordinary-walking calibration data and at least one withheld recording.';
  } else {
    const highlighted = report.study.comparisons
      .filter(({ kamEarlyChangePercent }) => kamEarlyChangePercent != null)
      .sort((left, right) => Math.abs(right.kamEarlyChangePercent!) - Math.abs(left.kamEarlyChangePercent!))[0];
    const early = highlighted?.metrics.find(({ label, compartment }) => label === statisticLabel.early && compartment === 'medial');
    if (highlighted && early) {
      report.study.highlightedComparisonId = highlighted.id;
      const adduction = highlighted.kamEarlyChangePercent!;
      report.study.summary = `For ${highlighted.label.toLowerCase()}, the early external adduction moment ${adduction >= 0 ? 'rose' : 'fell'} ${Math.abs(adduction).toFixed(0)}%. Measured medial load changed by ${digits(early.measured)} BW. The mechanics model estimated ${digits(early.model)} BW.`;
    } else {
      report.study.summary = 'Compare calibrated mechanics against implant measurements on withheld recordings, and see where several plausible parameter sets disagree.';
    }
  }
  return report;
}
