import type {
  TissueFrame, TissueScene, TissueLayers, TissueView,
} from '../../upper-limb/knee-loading/tissue-scene.ts';
import { isNumber, sampleAt } from './view-chart.ts';
import type { ComparisonView, TrialView } from './view-types.ts';

const format = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const playDurationMs = 3400;

type ForceSource = 'measured' | 'modeled';
type Slot = 'reference' | 'intervention';

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing tissue view element ${id}`);
  return element as T;
}

function forceAt(trial: TrialView, compartment: 'medial' | 'lateral', source: ForceSource, phase: number): number | null {
  const wave = trial.compartments[compartment];
  return wave ? sampleAt(trial.phase, wave[source], phase) : null;
}

function stanceFrame(trial: TrialView, comparison: ComparisonView, source: ForceSource, phase: number): TissueFrame {
  return {
    flexionDeg: trial.kneeFlexionDeg ? sampleAt(trial.phase, trial.kneeFlexionDeg, phase) : null,
    medialBw: forceAt(trial, 'medial', source, phase),
    lateralBw: forceAt(trial, 'lateral', source, phase),
    side: comparison.side,
  };
}

function flexionLabel(value: number | null, side: TissueFrame['side']): string {
  const label = side ? `${side === 'left' ? 'Left' : 'Right'} knee` : 'Reference knee';
  if (!isNumber(value)) return `${label} · Recorded flexion unavailable`;
  const flexion = value < 0 ? `−${format.format(Math.abs(value))}` : format.format(value);
  const endpoint = value < 0 ? ' · Atlas at 0°' : value > 120 ? ' · Atlas at 120°' : '';
  return `${label} · ${flexion}° flexion${endpoint}`;
}

const formatBW = (value: number | null) => isNumber(value) ? `${format.format(value)} BW` : 'Unavailable';

function renderReadout(
  slot: Slot, label: string, frame: TissueFrame, loadColor: ((force: number | null) => string) | undefined,
): void {
  required(`tissue-${slot}-title`).textContent = label;
  required(`tissue-${slot}-angle`).textContent = flexionLabel(frame.flexionDeg, frame.side);
  const medial = required(`tissue-${slot}-medial`);
  const lateral = required(`tissue-${slot}-lateral`);
  for (const [readout, force] of [[medial, frame.medialBw], [lateral, frame.lateralBw]] as const) {
    readout.textContent = formatBW(force);
    readout.style.setProperty('--marker-color', loadColor?.(force) ?? 'transparent');
    readout.classList.toggle('has-force', isNumber(force));
  }
}

export interface TissueViewController {
  update(comparison: ComparisonView, phase: number): void;
  pause(): void;
}

export function bindTissueView(phasePicker: HTMLInputElement, redraw: () => void): TissueViewController {
  const host = required('tissue-canvas');
  const status = required('tissue-status');
  const viewPicker = required<HTMLSelectElement>('tissue-view');
  const zoomIn = required<HTMLButtonElement>('tissue-zoom-in');
  const zoomOut = required<HTMLButtonElement>('tissue-zoom-out');
  const reset = required<HTMLButtonElement>('tissue-reset');
  const isolate = required<HTMLButtonElement>('tissue-isolate');
  const play = required<HTMLButtonElement>('tissue-play');
  const playLabel = required('tissue-play-label');
  const legend = required('tissue-legend');
  const sourceButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-tissue-source]'));
  const layerInputs = Array.from(document.querySelectorAll<HTMLInputElement>('[data-tissue-layer]'));

  let scene: TissueScene | undefined;
  let loadColor: typeof import('../../upper-limb/knee-loading/tissue-scene.ts').tissueLoadColor | undefined;
  let source: ForceSource = 'measured';
  let current: ComparisonView | undefined;
  let requestId = 0;
  let startTime = 0;
  let startPhase = 0;
  let savedLayers: boolean[] | undefined;

  const updateLayers = () => {
    const layers = Object.fromEntries(layerInputs.map((input) => [input.dataset.tissueLayer, input.checked])) as TissueLayers;
    scene?.setLayers(layers);
    legend.hidden = !scene || !layers.loads;
  };

  const update = (comparison: ComparisonView, phase: number) => {
    current = comparison;
    const ordinary = stanceFrame(comparison.reference, comparison, source, phase);
    const modified = stanceFrame(comparison.intervention, comparison, source, phase);
    renderReadout('reference', comparison.reference.label, ordinary, loadColor);
    renderReadout('intervention', comparison.intervention.label, modified, loadColor);
    scene?.setFrames(ordinary, modified);
  };

  const setPlaying = (running: boolean) => {
    play.setAttribute('aria-pressed', String(running));
    const label = running ? 'Pause stance' : 'Play stance';
    play.setAttribute('aria-label', label);
    playLabel.textContent = label;
  };

  const pause = () => {
    if (!requestId) return;
    cancelAnimationFrame(requestId);
    requestId = 0;
    setPlaying(false);
  };

  function tick(time: number) {
    const phase = Math.floor((startPhase + (time - startTime) * 101 / playDurationMs) % 101);
    if (phase !== phasePicker.valueAsNumber) {
      phasePicker.valueAsNumber = phase;
      redraw();
    }
    requestId = requestAnimationFrame(tick);
  }

  play.addEventListener('click', () => {
    if (requestId) {
      pause();
      return;
    }
    startPhase = phasePicker.valueAsNumber;
    startTime = performance.now();
    setPlaying(true);
    requestId = requestAnimationFrame(tick);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause();
  });

  sourceButtons.forEach((button) => button.addEventListener('click', () => {
    source = button.dataset.tissueSource === 'modeled' ? 'modeled' : 'measured';
    for (const item of sourceButtons) item.setAttribute('aria-pressed', String(item === button));
    if (current) update(current, phasePicker.valueAsNumber);
  }));

  const setIsolating = (active: boolean) => {
    isolate.setAttribute('aria-pressed', String(active));
    isolate.textContent = active ? 'Restore layers' : 'Isolate cartilage';
  };
  isolate.addEventListener('click', () => {
    if (savedLayers) {
      layerInputs.forEach((input, index) => { input.checked = savedLayers![index]; });
      savedLayers = undefined;
    } else {
      savedLayers = layerInputs.map((input) => input.checked);
      for (const input of layerInputs) {
        if (input.dataset.tissueLayer !== 'loads') input.checked = input.dataset.tissueLayer === 'cartilage';
      }
    }
    setIsolating(Boolean(savedLayers));
    updateLayers();
  });
  layerInputs.forEach((input) => input.addEventListener('change', () => {
    savedLayers = undefined;
    setIsolating(false);
    updateLayers();
  }));
  viewPicker.addEventListener('change', () => scene?.setView(viewPicker.value as TissueView));
  zoomIn.addEventListener('click', () => scene?.zoomBy(1.22));
  zoomOut.addEventListener('click', () => scene?.zoomBy(1 / 1.22));
  reset.addEventListener('click', () => {
    scene?.reset();
    viewPicker.value = 'anterior';
  });

  void import('../../upper-limb/knee-loading/tissue-scene.ts').then(async (renderer) => {
    scene = await renderer.createTissueScene(host);
    loadColor = renderer.tissueLoadColor;
    const max = renderer.TISSUE_FORCE_MAX_BW;
    const colors = [0, 0.25, 0.5, 0.75, 1].map((fraction) => loadColor!(max * fraction));
    required('tissue-gradient').style.background = `linear-gradient(to right, ${colors.join(', ')})`;
    required('tissue-force-max').textContent = `${max}+ BW`;
    status.hidden = true;
    updateLayers();
    if (current) update(current, phasePicker.valueAsNumber);
    scene.setView(viewPicker.value as TissueView);
  }).catch((error: unknown) => {
    console.warn('The 3D knee could not be loaded', error);
    status.textContent = '3D anatomy is unavailable in this browser. Force readouts and charts remain available.';
    for (const control of [...layerInputs, viewPicker, zoomIn, zoomOut, reset, isolate]) control.disabled = true;
  });

  return { update, pause };
}
