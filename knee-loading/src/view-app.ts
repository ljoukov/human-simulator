import type { BenchmarkResult } from './backend.ts';
import { mountBodyStudy } from './view-body.ts';
import { loadStudy, runLiveBenchmark } from './study.ts';
import { drawCurve, forceScale, isNumber, phaseAtPointer, sampleAt } from './view-chart.ts';
import { bindTissueView, type TissueViewController } from './view-tissue.ts';
import type { ComparisonView, Compartment, MetricView, StudyBenchmark, StudyBenchmarkWorkload, StudyView, TrialView } from './view-types.ts';

const compartmentNames: Record<Compartment, string> = {
  medial: 'Medial',
  lateral: 'Lateral',
  total: 'Total tibiofemoral',
};
const categories: Compartment[] = ['medial', 'lateral', 'total'];
const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const ensembleReport = new URL('../data/gait-ensemble-comparison.json', import.meta.url).href;

interface EnsembleBenchmark {
  rows: {
    scenarios: number;
    frames: number;
    m4NativeMs: number;
    m4WebGpuMs: number;
    h100BatchMs: number;
    h100KernelMs: number;
    h100BlockKernelMs: number;
    topCandidatesMatched: number;
    comparedScores: number;
  }[];
  holdout: {
    smallCandidates: number;
    bigCandidates: number;
    smallMedialRmseBw: number;
    bigMedialRmseBw: number;
    smallLateralRmseBw: number;
    bigLateralRmseBw: number;
  };
  repeats: number;
  modalRemoteSeconds: number;
}

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const target = document.getElementById(id);
  if (!target) throw new Error(`Missing UI element ${id}`);
  return target as T;
}

function create<K extends keyof HTMLElementTagNameMap>(tag: K, content?: string, className?: string): HTMLElementTagNameMap[K] {
  const item = document.createElement(tag);
  if (content) item.textContent = content;
  if (className) item.className = className;
  return item;
}

const comparisonPicker = element<HTMLSelectElement>('comparison');
const compartmentPicker = element<HTMLSelectElement>('compartment');
const phasePicker = element<HTMLInputElement>('stance-phase');
const benchmarkButton = element<HTMLButtonElement>('run-benchmark');
const referenceCanvas = element<HTMLCanvasElement>('reference-chart');
const interventionCanvas = element<HTMLCanvasElement>('intervention-chart');
let tissueView: TissueViewController | undefined;

function option(value: string, text: string): HTMLOptionElement {
  return new Option(text, value);
}

function displayNumber(value: number | null | undefined, signed = false): string {
  if (!isNumber(value)) return 'N/A';
  const number = Math.abs(value) < 0.005 ? 0 : value;
  const prefix = number < 0 ? '−' : signed && number > 0 ? '+' : '';
  return `${prefix}${numberFormat.format(Math.abs(number))}`;
}

function displayMetric(value: number | null | undefined, unit: string, signed = false): string {
  if (!isNumber(value)) return 'N/A';
  const separator = unit === '%' || unit === '°' ? '' : ' ';
  return unit ? `${displayNumber(value, signed)}${separator}${unit}` : displayNumber(value, signed);
}

function labeledValue(label: string, value: string, className?: string): HTMLDivElement {
  const row = create('div', undefined, className);
  row.append(create('dt', label), create('dd', value));
  return row;
}

function metricCard(metric: MetricView): HTMLDivElement {
  const panel = create('div', undefined, 'change-row');
  const list = create('dl');
  const delta = metric.delta ?? true;
  if (isNumber(metric.measured)) list.append(labeledValue('Measured', displayMetric(metric.measured, metric.unit, delta), 'measured'));
  if (isNumber(metric.model)) {
    const model = labeledValue('Model', displayMetric(metric.model, metric.unit, delta), 'modeled');
    if (isNumber(metric.modelLow) && isNumber(metric.modelHigh)) {
      const low = displayNumber(metric.modelLow, delta);
      const high = displayNumber(metric.modelHigh, delta);
      model.querySelector('dd')?.append(create('small', `Interval ${low} to ${high}`));
    }
    list.append(model);
  }
  if (isNumber(metric.proxy)) list.append(labeledValue('Moment baseline', displayMetric(metric.proxy, metric.unit, delta)));
  panel.append(create('p', metric.label, 'change-title'), list);
  return panel;
}

function setTrial(trial: TrialView, side: 'reference' | 'intervention', compartment: Compartment, phase: number): void {
  const wave = trial.compartments[compartment];
  element(`${side}-title`).textContent = trial.label;
  const count = trial.trialCount;
  element(`${side}-count`).textContent = count === undefined ? '' : `${count} ${count === 1 ? 'stance segment' : 'stance segments'}`;
  const values = element(`${side}-values`);
  values.replaceChildren();
  const chart = element(`${side}-chart`);
  const prefix = `${trial.label}. ${compartmentNames[compartment]} knee contact forces by stance phase in body weights.`;
  chart.setAttribute('aria-label', `${prefix} ${wave ? 'The stance slider provides individual values.' : 'Waveform unavailable.'}`);
  if (!wave) return;

  const currentMeasured = sampleAt(trial.phase, wave.measured, phase);
  const currentMedian = sampleAt(trial.phase, wave.modeled, phase);
  const currentLow = sampleAt(trial.phase, wave.low, phase);
  const currentHigh = sampleAt(trial.phase, wave.high, phase);
  if (isNumber(currentMeasured)) values.append(labeledValue('Measured', displayMetric(currentMeasured, 'BW')));
  if (isNumber(currentMedian)) values.append(labeledValue('Model', displayMetric(currentMedian, 'BW')));
  if (isNumber(currentLow) && isNumber(currentHigh)) {
    values.append(labeledValue('Interval', `${displayNumber(currentLow)}–${displayNumber(currentHigh)} BW`));
  }
  if (wave.proxy) {
    const currentProxy = sampleAt(trial.phase, wave.proxy, phase);
    if (isNumber(currentProxy)) values.append(labeledValue('Moment baseline', displayMetric(currentProxy, 'BW')));
  }
}

function renderForces(study: StudyView): void {
  const comparison = study.comparisons.find((item) => item.id === comparisonPicker.value);
  if (!comparison) return;

  const compartment = compartmentPicker.value as Compartment;
  const phase = phasePicker.valueAsNumber;
  const referenceWave = comparison.reference.compartments[compartment];
  const interventionWave = comparison.intervention.compartments[compartment];
  const showProxy = [referenceWave, interventionWave].some((wave) => wave?.proxy?.some(isNumber));
  const showInterval = [referenceWave, interventionWave].some((wave) => wave?.low.some(isNumber));
  const scale = forceScale([referenceWave, interventionWave]);

  element('interval-legend').hidden = !showInterval;
  element('interval-label').textContent = study.intervalLabel ?? 'Model interval';
  element('proxy-legend').hidden = !showProxy;
  element('phase-value').textContent = `${phase}%`;
  phasePicker.setAttribute('aria-valuetext', `${phase} percent stance`);
  setTrial(comparison.reference, 'reference', compartment, phase);
  setTrial(comparison.intervention, 'intervention', compartment, phase);
  drawCurve(referenceCanvas, comparison.reference, referenceWave, scale, phase);
  drawCurve(interventionCanvas, comparison.intervention, interventionWave, scale, phase);
  tissueView?.update(comparison, phase);
}

function renderComparison(comparison: ComparisonView): void {
  const compartment = compartmentPicker.value as Compartment;
  element('comparison-caption').textContent = comparison.context ?? `${comparison.subject} · ${comparison.label}`;
  element('comparison-title').textContent = comparison.metricsTitle ?? 'Change with modified gait';
  element('comparison-explanation').textContent = comparison.metricsIntro ?? 'Positive changes indicate higher loads than in the reference gait.';
  const metrics = comparison.metrics.filter((metric) => (!metric.compartment || metric.compartment === compartment)
    && [metric.measured, metric.model, metric.proxy].some(isNumber));
  const hasDeltas = metrics.some(({ delta }) => delta);
  const changes = hasDeltas ? metrics.filter(({ delta }) => delta !== false) : metrics;
  const validation = hasDeltas ? metrics.filter(({ delta }) => delta === false) : [];
  const target = element('changes');
  target.replaceChildren(...changes.map(metricCard));
  if (!metrics.length) target.append(create('p', 'No summary metrics are available for this compartment.', 'muted'));
  element('validation').hidden = !validation.length;
  element('validation-metrics').replaceChildren(...validation.map(metricCard));
}

interface RankedCondition {
  comparison: ComparisonView;
  metric: MetricView;
  measured: number;
}

const rankingTieBw = 0.005;

function pairwiseAgreement(rows: RankedCondition[], predictor: 'model' | 'proxy'): string | null {
  let correct = 0;
  let pairs = 0;
  for (const [index, first] of rows.entries()) {
    for (const second of rows.slice(index + 1)) {
      const a = first.metric[predictor];
      const b = second.metric[predictor];
      const measuredDifference = first.measured - second.measured;
      if (!isNumber(a) || !isNumber(b) || Math.abs(measuredDifference) < rankingTieBw) continue;
      pairs++;
      const estimatedDifference = a - b;
      if (Math.abs(estimatedDifference) >= rankingTieBw && Math.sign(measuredDifference) === Math.sign(estimatedDifference)) correct++;
    }
  }
  return pairs ? `${correct} of ${pairs}` : null;
}

function renderRanking(comparisons: ComparisonView[], selected: ComparisonView): void {
  const grouping = selected.groupId ?? selected.subject;
  const compartment = compartmentPicker.value as Compartment;
  const ranked: RankedCondition[] = [];
  for (const comparison of comparisons) {
    if ((comparison.groupId ?? comparison.subject) !== grouping) continue;
    const metric = comparison.metrics.find((item) => item.label === 'Early-stance peak' && item.compartment === compartment && item.delta);
    if (metric && isNumber(metric.measured)) ranked.push({ comparison, metric, measured: metric.measured });
  }

  const rankings = element('rankings');
  rankings.hidden = !ranked.length;
  if (!ranked.length) return;
  ranked.sort((first, second) => first.measured - second.measured);
  const showKam = ranked.some(({ comparison }) => isNumber(comparison.kamEarlyChangePercent));
  const showOffload = ranked.some(({ metric }) => isNumber(metric.offloadShare));

  element('rankings-scope').textContent = `${compartmentNames[compartment]} early-stance peak. Ranked by measured change from reference gait, with greatest decrease first.`;
  const table = create('table', undefined, 'ranking-table');
  table.setAttribute('aria-labelledby', 'rankings-title rankings-scope');
  const fields = ['Condition', 'Implant force Δ', 'Model force Δ', 'Moment baseline Δ'];
  if (showKam) fields.push('External adduction moment Δ');
  if (showOffload) fields.push('Offload share');
  const header = table.createTHead().insertRow();
  for (const field of fields) {
    const cell = create('th', field);
    cell.scope = 'col';
    header.append(cell);
  }

  const target = element('rankings-results');
  const body = table.createTBody();
  for (const { comparison, metric, measured } of ranked) {
    const row = body.insertRow();
    const active = comparison.id === selected.id;
    row.classList.toggle('active-row', active);
    const rowTitle = create('th');
    rowTitle.scope = 'row';
    const choice = create('button', comparison.label);
    choice.type = 'button';
    choice.setAttribute('aria-pressed', String(active));
    choice.addEventListener('click', () => {
      comparisonPicker.value = comparison.id;
      comparisonPicker.dispatchEvent(new Event('change'));
      target.querySelector<HTMLButtonElement>('button[aria-pressed="true"]')?.focus({ preventScroll: true });
    });
    rowTitle.append(choice);
    row.append(rowTitle);

    const values = [displayMetric(measured, metric.unit, true), displayMetric(metric.model, metric.unit, true), displayMetric(metric.proxy, metric.unit, true)];
    if (showKam) values.push(displayMetric(comparison.kamEarlyChangePercent, '%', true));
    if (showOffload) values.push(displayMetric(isNumber(metric.offloadShare) ? 100 * metric.offloadShare : null, '%'));
    for (const value of values) row.append(create('td', value));
  }
  target.replaceChildren(table);

  const modelAgreement = pairwiseAgreement(ranked, 'model');
  const momentAgreement = pairwiseAgreement(ranked, 'proxy');
  const comparisonNotes: string[] = [];
  if (modelAgreement) comparisonNotes.push(`The model matched ${modelAgreement} measurable pairwise orders.`);
  if (momentAgreement) comparisonNotes.push(`The moment baseline matched ${momentAgreement}.`);
  if (modelAgreement || momentAgreement) comparisonNotes.push(`Differences below ${rankingTieBw} BW count as ties.`);
  const threshold = ranked.find(({ metric }) => isNumber(metric.offloadThresholdBw))?.metric.offloadThresholdBw;
  if (showOffload && isNumber(threshold)) comparisonNotes.push(`Offload share is the fraction of retained models that predict a decrease of at least ${displayMetric(threshold, 'BW')}.`);
  element('rankings-note').textContent = comparisonNotes.join(' ');
}

function setCompartments(comparison: ComparisonView): void {
  const previous = compartmentPicker.value;
  const available = categories.filter((key) => comparison.reference.compartments[key] || comparison.intervention.compartments[key]);
  compartmentPicker.replaceChildren(...available.map((key) => option(key, compartmentNames[key])));
  if (available.some((key) => key === previous)) compartmentPicker.value = previous;
}

function benchmarkTable(result: BenchmarkResult): HTMLTableElement {
  const table = create('table', undefined, 'benchmark-table');
  table.setAttribute('aria-label', 'CPU and GPU batch timings and numerical agreement');
  const header = table.createTHead().insertRow();
  const gpu = Boolean(result.gpu);
  const columns = ['Scenarios', 'Frame solves', 'Single-core CPU'];
  if (result.cpu.workers > 1) columns.push('Multicore CPU');
  if (gpu) columns.push('GPU', 'vs. fastest CPU', 'Disagreements');
  for (const title of columns) {
    const cell = create('th', title);
    cell.scope = 'col';
    header.append(cell);
  }

  const body = table.createTBody();
  for (const resultRow of result.rows) {
    const row = body.insertRow();
    const label = create('th', displayNumber(resultRow.scenarios));
    label.scope = 'row';
    row.append(label);
    const values = [displayNumber(resultRow.frames * resultRow.scenarios), displayMetric(resultRow.cpuSingleMs, 'ms')];
    if (result.cpu.workers > 1) values.push(displayMetric(resultRow.cpuWorkersMs, 'ms'));
    if (gpu) {
      values.push(
        displayMetric(resultRow.gpuMs, 'ms'),
        isNumber(resultRow.speedup) ? `${displayNumber(resultRow.speedup)}×` : 'N/A',
        displayNumber(resultRow.mismatchCount),
      );
    }
    for (const value of values) row.append(create('td', value));
    if (isNumber(resultRow.mismatchCount) && resultRow.mismatchCount > 0) row.lastElementChild?.classList.add('benchmark-check');
  }
  return table;
}

function benchmarkWorkload(result: StudyBenchmarkWorkload, location?: string): HTMLElement {
  const calibration = result.workload === 'calibration';
  const panel = create('section', undefined, 'benchmark-workload');
  const caption = calibration ? 'Parameter calibration' : 'Contact force curves';
  const description = calibration
    ? 'Each scenario returns a single calibration error reduced across stance frames.'
    : 'Each scenario returns forces and balance residuals for every stance frame.';
  const workers = `${result.cpu.workers} CPU ${result.cpu.workers === 1 ? 'worker' : 'workers'}`;
  const device = result.gpu ? `${workers} · ${result.gpu.adapter}` : `${workers}. ${result.gpuUnavailable || 'GPU results unavailable.'}`;
  const largest = result.rows.reduce<typeof result.rows[number] | undefined>((best, row) => (
    (!best || row.scenarios > best.scenarios) && isNumber(row.speedup) ? row : best
  ), undefined);
  panel.append(
    create('h3', caption),
    create('p', description, 'muted'),
    create('p', location ? `${location} · ${device}` : device, 'benchmark-device'),
  );
  if (largest && isNumber(largest.speedup) && largest.speedup > 0) {
    const count = displayNumber(largest.scenarios);
    const ratio = `${displayNumber(largest.speedup >= 1 ? largest.speedup : 1 / largest.speedup)}×`;
    const winningDevice = largest.speedup >= 1 ? 'GPU' : largest.cpuMode === 'workers' ? 'Multicore CPU' : 'Single-core CPU';
    const selection = largest.topCandidateAgreement;
    const agreement = selection ? ` Both backends agreed on ${selection.shared} of ${selection.size} selected candidates.` : '';
    panel.append(create('p', `${winningDevice} ran ${ratio} as fast at ${count} scenarios.${agreement}`, 'benchmark-summary'));
  }
  panel.append(benchmarkTable(result));
  const cpuSetup = displayMetric(result.cpu.setupMs, 'ms');
  const gpuSetup = result.gpu ? ` and GPU setup ${displayMetric(result.gpu.setupMs, 'ms')}` : '';
  const setup = `Median of ${result.repeats} ${result.repeats === 1 ? 'run' : 'runs'} per batch. CPU setup ${cpuSetup}${gpuSetup}. Batch timings include transfers and result readback.`;
  panel.append(create('p', setup, 'benchmark-note'));
  if (result.cpu.parallelUnavailable) panel.append(create('p', `Multicore CPU unavailable. ${result.cpu.parallelUnavailable}`, 'benchmark-note'));
  if (result.gpu) {
    const absTolerance = result.absTolerance.toPrecision(2);
    const relTolerance = result.relTolerance.toPrecision(2);
    const checks = result.rows.map((row) => row.maxAbsError).filter(isNumber);
    if (checks.length) {
      const maximumError = Math.max(...checks);
      const text = `Parity checks use ${absTolerance} absolute and ${relTolerance} relative tolerances. Largest absolute difference ${maximumError.toExponential(2)}.`;
      panel.append(create('p', text, 'benchmark-note'));
    }
    for (const row of result.rows) {
      if (row.gpuError) panel.append(create('p', `The ${displayNumber(row.scenarios)}-scenario GPU batch failed. ${row.gpuError}`, 'benchmark-note'));
    }
  }
  return panel;
}

function renderBenchmark(result: StudyBenchmark, live = false, recordedCalibration?: StudyBenchmarkWorkload): void {
  const status = element('benchmark-status');
  const results = element('benchmark-results');
  const origin = live ? 'Completed on this device' : 'Recorded run';
  const trial = result.trial ? ` · ${result.trial.id}` : '';
  const runtime = result.runtime ? ` · ${result.runtime}` : '';
  status.textContent = `${origin}${trial}${runtime}`;
  results.replaceChildren(benchmarkWorkload(result));
  if (result.calibration) results.append(benchmarkWorkload(result.calibration));
  else if (live && recordedCalibration) results.append(benchmarkWorkload(recordedCalibration, 'Recorded run'));
  results.hidden = false;
}

async function renderEnsembleBenchmark(url: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Gait ensemble benchmark returned HTTP ${response.status}.`);
  const { rows, holdout, repeats, modalRemoteSeconds } = await response.json() as EnsembleBenchmark;
  const largest = rows.reduce<EnsembleBenchmark['rows'][number] | undefined>((best, row) =>
    !best || row.scenarios > best.scenarios ? row : best, undefined);
  if (!largest) return;

  const panel = element('ensemble-benchmark');
  const title = create('h3', 'Gait calibration on a Modal H100');
  title.id = 'ensemble-benchmark-title';
  const batch = create('p', `Scored ${displayNumber(largest.scenarios)} parameter sets across ${displayNumber(largest.frames)} samples of recorded gait in ${displayMetric(largest.h100BatchMs, 'ms')}.`, 'benchmark-summary');
  const speedFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  const speedup = create('p', `${speedFormat.format(largest.m4NativeMs / largest.h100BatchMs)}× the native M4 CPU throughput and ${speedFormat.format(largest.m4WebGpuMs / largest.h100BatchMs)}× the M4 WebGPU throughput for the same batch.`, 'benchmark-note');

  const table = create('table', undefined, 'benchmark-table ensemble-table');
  table.setAttribute('aria-label', 'Median calibration batch and H100 kernel times in milliseconds');
  const header = table.createTHead().insertRow();
  for (const label of ['Parameter sets', 'M4 native CPU', 'M4 WebGPU', 'H100 CUDA', 'H100 kernel only']) {
    const cell = create('th', label);
    cell.scope = 'col';
    header.append(cell);
  }
  const body = table.createTBody();
  for (const measured of rows) {
    const row = body.insertRow();
    const count = create('th', displayNumber(measured.scenarios));
    count.scope = 'row';
    row.append(count, ...[measured.m4NativeMs, measured.m4WebGpuMs, measured.h100BatchMs, measured.h100KernelMs]
      .map((ms) => create('td', `${ms < 1 ? ms.toFixed(3) : displayNumber(ms)} ms`)));
  }
  const tableWrap = create('div', undefined, 'scrollable');
  tableWrap.tabIndex = 0;
  tableWrap.setAttribute('role', 'region');
  tableWrap.setAttribute('aria-label', 'Calibration timings');
  tableWrap.append(table);

  const timingFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
  const layout = `The CUDA kernel reuses precomputed gait geometry and assigns one 32-thread warp to each parameter set. Threads combine scores on the GPU and return one value per set. At the largest batch, this kernel took ${timingFormat.format(largest.h100KernelMs)} ms versus ${timingFormat.format(largest.h100BlockKernelMs)} ms for the block-per-candidate baseline on the same H100.`;
  const parity = `At that size, the H100 matched ${displayNumber(largest.topCandidatesMatched)} of the reference’s top 256 parameter sets and checked ${displayNumber(largest.comparedScores)} scores against the reference.`;
  const errors = `Across six withheld stance segments from the same participant, increasing the search from ${displayNumber(holdout.smallCandidates)} to ${displayNumber(holdout.bigCandidates)} candidates changed medial force RMSE from ${holdout.smallMedialRmseBw.toFixed(3)} to ${holdout.bigMedialRmseBw.toFixed(3)} BW and lateral force RMSE from ${holdout.smallLateralRmseBw.toFixed(3)} to ${holdout.bigLateralRmseBw.toFixed(3)} BW.`;
  const timing = `Medians of ${displayNumber(repeats)} warm runs. GPU batch times include parameter upload and score readback. Cloud startup and orchestration are timed separately. The recorded Modal call took ${displayMetric(modalRemoteSeconds, 's')}.`;
  const download = create('a', 'Download benchmark data', 'subtle-link');
  download.href = url;
  download.download = 'gait-ensemble-comparison.json';
  panel.replaceChildren(title, batch, speedup, tableWrap,
    create('p', layout, 'benchmark-note'), create('p', parity, 'benchmark-note'),
    create('p', errors, 'benchmark-note'), create('p', timing, 'benchmark-note'), download);
  panel.hidden = false;
}

function showStatus(message: string): void {
  const status = element('study-status');
  status.textContent = message;
  status.hidden = !message;
}

function initialize(study: StudyView): void {
  const participants = new Set(study.comparisons.map(({ subject }) => subject)).size;
  element('study-source').textContent = participants > 1
    ? `${participants} implant participants`
    : study.dataset;
  element('study-summary').textContent = study.summary;
  if (study.notice) showStatus(study.notice);
  element('methods').replaceChildren(...study.methods.map((method) => create('p', method)));
  if (study.sources.length) {
    element('sources').replaceChildren(...study.sources.map((source) => {
      const item = create('li');
      const link = create('a', source.label);
      link.href = source.url;
      item.append(link);
      return item;
    }));
  }
  if (study.downloads?.length) {
    element('download-files').hidden = false;
    element('download-links').replaceChildren(...study.downloads.map(({ label, url, filename }) => {
      const item = create('li');
      const link = create('a', label);
      link.href = url;
      link.download = filename;
      item.append(link);
      return item;
    }));
  }
  if (study.patella) {
    const { summary, flexionStartDeg, flexionEndDeg, measuredRotationDeg, referenceRotationDeg, viewerRotationDeg, source } = study.patella;
    element('patella-summary').textContent = summary;
    element('patella-range').textContent = `Net kneecap rotation between ${displayNumber(flexionStartDeg)}° and ${displayNumber(flexionEndDeg)}° of knee flexion.`;
    element('patella-rotation').replaceChildren(
      labeledValue('Measured specimen', displayMetric(measuredRotationDeg, '°')),
      labeledValue('OpenSim reference', displayMetric(referenceRotationDeg, '°')),
      labeledValue('Atlas animation', displayMetric(viewerRotationDeg, '°')),
    );
    const link = element<HTMLAnchorElement>('patella-source');
    link.textContent = source.label;
    link.href = source.url;
    element('patella').hidden = false;
    const container = element('pressure-view');
    const pressureObserver = new IntersectionObserver(([entry], observer) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();
      void import('../../upper-limb/knee-loading/pressure-view.ts')
        .then(({ mountPressureView }) => mountPressureView(container))
        .catch((error: unknown) => {
          console.warn('The pressure view could not be loaded', error);
          container.textContent = 'The measured pressure view is unavailable.';
        });
    }, { rootMargin: '420px 0px' });
    pressureObserver.observe(container);
  }

  if (study.benchmark) renderBenchmark(study.benchmark);
  const available = study.comparisons.filter((comparison) => Object.keys(comparison.reference.compartments).length
    || Object.keys(comparison.intervention.compartments).length);
  benchmarkButton.hidden = !(study.benchmarkAvailable ?? Boolean(available.length));
  benchmarkButton.addEventListener('click', async () => {
    benchmarkButton.disabled = true;
    benchmarkButton.textContent = 'Running benchmark…';
    const status = element('benchmark-status');
    status.textContent = 'Running the matched batches on this device…';
    try {
      const result = await runLiveBenchmark((stage) => { status.textContent = stage; });
      renderBenchmark(result, true, study.benchmark?.calibration);
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'Unknown error';
      status.textContent = `Device benchmark could not finish. ${reason}`;
    } finally {
      benchmarkButton.disabled = false;
      benchmarkButton.textContent = 'Run on this device';
    }
  });

  if (!available.length) {
    if (!study.notice) showStatus('The study has no force curves to display.');
    return;
  }

  comparisonPicker.replaceChildren(...available.map((comparison) => option(comparison.id, `${comparison.subject} · ${comparison.label}`)));
  const initial = available.find(({ id }) => id === study.highlightedComparisonId)
    ?? available.find(({ metrics }) => metrics.some(({ delta }) => delta))
    ?? available[0];
  comparisonPicker.value = initial.id;
  compartmentPicker.addEventListener('change', () => {
    const comparison = available.find((item) => item.id === comparisonPicker.value);
    if (comparison) {
      renderComparison(comparison);
      renderRanking(available, comparison);
    }
    renderForces(study);
  });
  comparisonPicker.addEventListener('change', () => {
    const comparison = available.find((item) => item.id === comparisonPicker.value);
    if (comparison) {
      setCompartments(comparison);
      renderComparison(comparison);
      renderRanking(available, comparison);
    }
    renderForces(study);
  });
  phasePicker.addEventListener('input', () => {
    tissueView?.pause();
    renderForces(study);
  });
  for (const canvas of [referenceCanvas, interventionCanvas]) {
    canvas.addEventListener('click', (event) => {
      tissueView?.pause();
      phasePicker.value = String(phaseAtPointer(canvas, event));
      renderForces(study);
    });
  }
  const figureObserver = new ResizeObserver(() => renderForces(study));
  figureObserver.observe(referenceCanvas);
  figureObserver.observe(interventionCanvas);

  element('forces').hidden = false;
  tissueView = bindTissueView(phasePicker, () => renderForces(study));
  setCompartments(initial);
  renderComparison(initial);
  renderRanking(available, initial);
  renderForces(study);

  const anchor = window.location.hash.slice(1);
  if (anchor === 'gait-path' || anchor === 'tissue' || anchor === 'pressure-view' || anchor === 'kernel') {
    requestAnimationFrame(() => document.getElementById(anchor)?.scrollIntoView());
  }
}

loadStudy().then(initialize).catch((error: unknown) => {
  element('study-summary').textContent = 'Study results unavailable.';
  const reason = error instanceof Error ? error.message : 'Unknown error';
  showStatus(`The study could not be opened. ${reason}`);
});

void renderEnsembleBenchmark(ensembleReport).catch((error: unknown) => {
  console.warn('The measured gait calibration benchmark could not be loaded', error);
});

void mountBodyStudy().catch((error: unknown) => {
  console.warn('The full-body study could not be opened', error);
  element('body-status').textContent = 'The full-body recordings are unavailable. See the knee tissue study for the stance data.';
});
