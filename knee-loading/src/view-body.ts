import type { BodyFrame, BodyLayers, BodyScene, BodyView } from '../../upper-limb/knee-loading/body-scene.ts';
import type { TissueFrame, TissueLayers, TissueScene, TissueView } from '../../upper-limb/knee-loading/tissue-scene.ts';
import { validateDataset, type GaitDataset, type GaitTrial } from './dataset.ts';
import { earlyStancePeaks, MEDIAL_KNEE_PAIN_EVIDENCE, sampleGaitPathway } from './pathway.ts';
import { advanceStride, type StridePlayback } from './stride-playback.ts';

type Position = [number, number, number];
type Rotation = [number, number, number, number];
type GroundContact = [number, number, number, number, number, number];
type GroundFrame = { right: GroundContact | null; left: GroundContact | null };
type Mode = 'whole' | 'knee';
type Collection = 'reference' | 'implant';

interface MotionClip {
  id: string;
  condition: string;
  participant?: string;
  label?: string;
  ageYears?: number;
  sex?: string;
  heightM?: number;
  massKg?: number;
  cohort?: string;
  walkingSpeedMps?: number;
  wristExcursionM?: { right: number; left: number };
  phase: 'stride' | 'stance';
  durationSeconds: number;
  stanceEndPct: number;
  heading?: 1 | -1;
  frames: number[][];
  ground?: GroundFrame[];
  signals: { rightKneeFlexionDeg: number[]; pelvisVerticalDisplacementM: number[] };
}

interface MotionArchive {
  segmentOrder: string[];
  landmarks?: Record<string, { segment: string; offsetM: Position }>;
  source: { title: string; url: string };
  clips: MotionClip[];
}

interface RecordedStride {
  archive: MotionArchive;
  clip: MotionClip;
  trial?: GaitTrial;
  label: string;
  poseLabel: string;
  conditionDetail: string;
  heading: 1 | -1;
  peaks?: ReturnType<typeof earlyStancePeaks>;
  pelvisExcursionCm: number;
  wristTravelCm: { right: number | null; left: number | null };
}

const motionUrl = new URL('../data/whole-body-motion.json', import.meta.url).href;
const referenceUrl = new URL('../data/reference-walks.json', import.meta.url).href;
const trialUrl = new URL('../data/gait.json', import.meta.url).href;
const formats = [
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }),
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }),
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }),
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }),
];
const descriptions: Record<string, string> = {
  bouncy: 'The bouncy gait instruction emphasized more up-and-down pelvis movement.',
  forefoot_strike: 'The forefoot-strike instruction changes which part of the foot contacts the ground first.',
  fast_treadmill: 'This recording combines faster walking and a change from overground to a treadmill.',
};
const clipNames: Record<string, string> = {
  normal: 'Ordinary walking',
  bouncy: 'Bouncy walking',
  forefoot_strike: 'Forefoot strike',
  fast_treadmill: 'Fast treadmill',
};

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing full-body study element ${id}`);
  return element as T;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`A recording could not be loaded (HTTP ${response.status}).`);
  return response.json() as Promise<T>;
}

const mix = (a: number, b: number, fraction: number) => a + (b - a) * fraction;

function frameIndex(length: number, phase: number) {
  const index = (length - 1) * Math.max(0, Math.min(100, phase)) / 100;
  const from = Math.floor(index);
  return { from, to: Math.ceil(index), fraction: index - from };
}

function signalAt(values: number[], phase: number): number {
  const { from, to, fraction } = frameIndex(values.length, phase);
  return mix(values[from], values[to], fraction);
}

function rotationAt(a: number[], b: number[], offset: number, fraction: number): Rotation {
  let dot = 0;
  for (let i = 0; i < 4; i++) dot += a[offset + i] * b[offset + i];
  const sign = dot < 0 ? -1 : 1;
  const q: Rotation = [
    mix(a[offset], sign * b[offset], fraction),
    mix(a[offset + 1], sign * b[offset + 1], fraction),
    mix(a[offset + 2], sign * b[offset + 2], fraction),
    mix(a[offset + 3], sign * b[offset + 3], fraction),
  ];
  const length = Math.hypot(...q);
  return [q[0] / length, q[1] / length, q[2] / length, q[3] / length];
}

function worldPoint({ position, rotation }: BodyFrame['segments'][string], [x, y, z]: Position): Position {
  const [qx, qy, qz, qw] = rotation;
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    position[0] + x + qw * tx + qy * tz - qz * ty,
    position[1] + y + qw * ty + qz * tx - qx * tz,
    position[2] + z + qw * tz + qx * ty - qy * tx,
  ];
}

function groundAt(clip: MotionClip, phase: number): GroundFrame | undefined {
  if (!clip.ground) return undefined;
  const { from, to, fraction } = frameIndex(clip.ground.length, phase);
  const contact = (side: keyof GroundFrame) => {
    const a = clip.ground![from][side];
    const b = clip.ground![to][side];
    // A force plate crossing is discrete. Never extrapolate a contact that
    // exists in only one adjacent sample or blend between distinct plates.
    if (a == null || b == null || Math.hypot(a[0] - b[0], a[2] - b[2]) > 0.2) return fraction < 0.5 ? a : b;
    return a.map((value, index) => mix(value, b[index], fraction)) as GroundContact;
  };
  return { right: contact('right'), left: contact('left') };
}

function bodyFrame(
  stride: RecordedStride, phase: number, medialBw: number | null,
): BodyFrame {
  const { archive, clip, heading } = stride;
  const { from, to, fraction } = frameIndex(clip.frames.length, phase);
  const a = clip.frames[from];
  const b = clip.frames[to];
  const pelvis = archive.segmentOrder.indexOf('pelvis') * 7;
  const first = clip.frames[0];
  const last = clip.frames.at(-1)!;
  const cameraOriginM: Position = [mix(first[pelvis], last[pelvis], phase / 100), 0, mix(first[pelvis + 2], last[pelvis + 2], 0.5)];
  const segments: BodyFrame['segments'] = {};
  for (const [index, name] of archive.segmentOrder.entries()) {
    const offset = index * 7;
    segments[name] = {
      position: [
        mix(a[offset], b[offset], fraction),
        mix(a[offset + 1], b[offset + 1], fraction),
        mix(a[offset + 2], b[offset + 2], fraction),
      ],
      rotation: rotationAt(a, b, offset + 3, fraction),
    };
  }
  const landmark = archive.landmarks?.rightKnee;
  return {
    segments, clipId: clip.id, phase, heading, side: 'right', cameraOriginM,
    kneeForceBw: medialBw,
    kneePosition: landmark && segments[landmark.segment] ? worldPoint(segments[landmark.segment], landmark.offsetM) : undefined,
    ground: groundAt(clip, phase),
  };
}

function format(value: number | null | undefined, unit = '', places = 2) {
  if (value == null || !Number.isFinite(value)) return 'N/A';
  return formats[places].format(value) + (unit === '°' || unit === '%' ? unit : unit ? ` ${unit}` : '');
}

function setReadout(list: HTMLElement, label: string, first: string, second: string, baseline: string, variant: string) {
  let row = Array.from(list.children).find((child) => (child as HTMLElement).dataset.readout === label) as HTMLElement | undefined;
  if (!row) {
    row = document.createElement('div');
    row.dataset.readout = label;
    const term = document.createElement('dt');
    term.textContent = label;
    const definition = document.createElement('dd');
    const ordinary = document.createElement('span');
    ordinary.className = 'pathway-reference';
    const arrow = document.createElement('span');
    arrow.className = 'pathway-arrow';
    arrow.textContent = '→';
    arrow.setAttribute('aria-hidden', 'true');
    const selected = document.createElement('strong');
    definition.append(ordinary, arrow, selected);
    row.append(term, definition);
    list.append(row);
  }
  const ordinary = row.querySelector('.pathway-reference')!;
  const selected = row.querySelector('strong')!;
  if (ordinary.textContent !== baseline) ordinary.textContent = baseline;
  if (selected.textContent !== variant) selected.textContent = variant;
  ordinary.setAttribute('aria-label', `${first}: ${baseline}`);
  selected.setAttribute('aria-label', `${second}: ${variant}`);
}

function wholeStrides(archive: MotionArchive, dataset?: GaitDataset): RecordedStride[] {
  const trials = new Map(dataset?.trials.map((trial) => [trial.id, trial]));
  const pelvisOffset = 7 * archive.segmentOrder.indexOf('pelvis');
  return archive.clips.flatMap((clip) => {
    const trial = trials.get(clip.id);
    if ((dataset && !trial) || clip.phase !== 'stride' || clip.frames.length < 2 || pelvisOffset < 0) return [];
    const [, qy, qz] = clip.frames[0].slice(pelvisOffset + 3, pelvisOffset + 7);
    const heading = clip.heading ?? (1 - 2 * (qy * qy + qz * qz) < 0 ? -1 : 1);
    const pelvis = clip.signals.pelvisVerticalDisplacementM;
    const wristTravel = (side: 'r' | 'l') => {
      const raw = clip.wristExcursionM?.[side === 'r' ? 'right' : 'left'];
      if (raw != null) return 100 * raw;
      const hand = archive.segmentOrder.indexOf(`hand_${side}`) * 7;
      if (hand < 0) return null;
      const foreAft = clip.frames.map((frame) => frame[hand] - frame[pelvisOffset]);
      return 100 * (Math.max(...foreAft) - Math.min(...foreAft));
    };
    const participant = clip.participant?.match(/^SUBJ(\d+)$/i);
    const demographics = [
      clip.ageYears == null ? undefined : `${clip.ageYears} years`,
      clip.sex?.toLowerCase(),
    ].filter(Boolean);
    const id = participant ? `#${Number(participant[1])}` : undefined;
    const label = clip.label ?? (id
      ? `${id} · ${demographics.length ? demographics.join(' · ') : 'barefoot walk'}`
      : clipNames[clip.condition] ?? clip.condition);
    const poseLabel = id ? [id, demographics[0]].filter(Boolean).join(' · ') : label;
    const conditionDetail = trial ? 'GC4-JW · instrumented right knee' : [
      clip.sex, clip.cohort?.toLowerCase(), clip.condition.replace(/ walking at preferred speed$/i, '').toLowerCase(),
      clip.walkingSpeedMps == null ? undefined : format(clip.walkingSpeedMps, 'm/s', 2),
    ].filter(Boolean).join(' · ');
    return [{
      archive, clip, trial, heading, label, poseLabel, conditionDetail,
      peaks: trial ? earlyStancePeaks(trial, clip.stanceEndPct) : undefined,
      pelvisExcursionCm: 100 * (Math.max(...pelvis) - Math.min(...pelvis)),
      wristTravelCm: { right: wristTravel('r'), left: wristTravel('l') },
    }];
  });
}

export async function mountBodyStudy(): Promise<void> {
  const status = required('body-status');
  const wholeHost = required('body-scene');
  const kneeHost = required('body-knee-scene');
  const condition = required<HTMLSelectElement>('body-condition');
  const referencePicker = required<HTMLSelectElement>('body-reference');
  const phaseControl = required<HTMLInputElement>('body-phase');
  const phaseValue = required('body-phase-value');
  const play = required<HTMLButtonElement>('body-play');
  const playLabel = required('body-play-label');
  const camera = required<HTMLSelectElement>('body-view');
  const speed = required<HTMLSelectElement>('body-speed');
  const sourceButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-body-source]'));
  const modeButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-body-mode]'));
  const events = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-body-event]'));
  const motionReadouts = required('body-motion-readouts');
  const groundReadouts = required('body-ground-readouts');
  const loadReadouts = required('body-load-readouts');

  const [archive, raw, referenceArchive] = await Promise.all([
    fetchJson<MotionArchive>(motionUrl), fetchJson<GaitDataset>(trialUrl),
    fetchJson<MotionArchive>(referenceUrl).catch((error: unknown) => {
      console.warn('The optical motion recordings could not be loaded', error);
      return null;
    }),
  ]);
  const implant = wholeStrides(archive, validateDataset(raw));
  const normal = implant.find(({ clip }) => clip.condition === 'normal');
  const modified = implant.filter(({ clip }) => clip.condition !== 'normal');
  const recorded = referenceArchive ? wholeStrides(referenceArchive) : [];
  if (!normal || !modified.length) throw new Error('Complete ordinary and modified implant strides are required.');
  const implantVariant = modified.find(({ clip }) => clip.condition === 'bouncy') ?? modified[0];
  const implantReference = normal;
  let implantSelection = implantVariant;
  let recordedFirst = recorded[0];
  let recordedSecond = recorded.slice(1).filter(({ clip }) => clip.ageYears != null)
    .sort((a, b) => a.clip.ageYears! - b.clip.ageYears!)[0] ?? recorded[2] ?? recorded[1];
  let collection: Collection = recorded.length >= 2 ? 'reference' : 'implant';
  let reference = collection === 'reference' ? recordedFirst : normal;
  let selected = collection === 'reference' ? recordedSecond : implantVariant;
  let mode: Mode = 'whole';
  let wholeScene: BodyScene | undefined;
  let kneeScene: TissueScene | undefined;
  let pendingWhole: Promise<void> | undefined;
  let pendingKnee: Promise<void> | undefined;
  let animationFrame = 0;
  let playback: StridePlayback | undefined;
  let currentPhase = 0;
  let lastReadoutTime = -Infinity;
  let userControlled = false;

  for (const control of [condition, phaseControl, play, ...events]) control.disabled = false;
  required('body-pathway').hidden = false;
  const clinical = MEDIAL_KNEE_PAIN_EVIDENCE;
  required('body-clinical-evidence').textContent = `In a randomized trial of ${clinical.participants} people with knee osteoarthritis, personalized foot-angle retraining reduced medial knee pain by ${Math.abs(clinical.betweenGroupDifferencePoints)} points on a ${clinical.outcomeScale} scale versus sham at one year. Pain ratings are unavailable for these recorded strides.`;
  required<HTMLAnchorElement>('body-clinical-source').href = clinical.url;

  function jumpTo(name: string) {
    const toeOff = selected.clip.stanceEndPct;
    if (name === 'peak') return selected.peaks?.medial?.stridePct ?? null;
    if (name === 'toe-off') return toeOff;
    if (name === 'swing') return (toeOff + 100) / 2;
    return 0;
  }

  function eventText(stride: RecordedStride, phase: number) {
    const end = stride.clip.stanceEndPct;
    const knee = signalAt(stride.clip.signals.rightKneeFlexionDeg, phase);
    let contact = phase >= 99.95 ? 'Next foot contact' : phase > end ? 'Right swing' : 'Right stance';
    if (phase <= 0.05) contact = 'Right foot contact';
    else if (Math.abs(phase - end) <= 0.05) contact = 'Right toe-off';
    const clipped = mode === 'knee' && knee < 0 ? ' · Atlas at 0°' : '';
    return `${contact} · ${format(knee, '°', 1)} flexion${clipped}`;
  }

  function showPhase(updateReadouts = true) {
    const phase = currentPhase;
    const sample = (stride: RecordedStride) => stride.trial
      ? sampleGaitPathway(stride.trial, stride.clip.stanceEndPct, phase) : null;
    const first = sample(reference);
    const second = sample(selected);
    const flexion = ({ clip }: RecordedStride) => signalAt(clip.signals.rightKneeFlexionDeg, phase);
    if (mode === 'whole' && wholeScene) wholeScene.setFrames(
      bodyFrame(reference, phase, first?.medialBw ?? null),
      bodyFrame(selected, phase, second?.medialBw ?? null),
    );
    if (mode === 'knee' && kneeScene) {
      const knee = (stride: RecordedStride, data: typeof first): TissueFrame => ({
        flexionDeg: flexion(stride), medialBw: data?.medialBw ?? null, lateralBw: data?.lateralBw ?? null, side: 'right',
      });
      kneeScene.setFrames(knee(reference, first), knee(selected, second));
    }

    if (!updateReadouts) return;
    phaseValue.textContent = format(phase, '%', 1);
    phaseControl.setAttribute('aria-valuetext', `${format(phase, '', 1)} percent of stride`);
    required('body-reference-track').style.setProperty('--phase', `${phase}%`);
    required('body-selected-track').style.setProperty('--phase', `${phase}%`);
    required('body-reference-event').textContent = eventText(reference, phase);
    required('body-selected-event').textContent = eventText(selected, phase);
    const readout = (list: HTMLElement, label: string, a: number | null | undefined, b: number | null | undefined, unit: string, places?: number) =>
      setReadout(list, label, reference.label, selected.label, format(a, unit, places), format(b, unit, places));
    readout(motionReadouts, 'Right knee flexion', flexion(reference), flexion(selected), '°', 1);
    if (collection === 'implant') {
      required('body-reference-ground').textContent = format(first?.grfBw, 'BW');
      required('body-selected-ground').textContent = format(second?.grfBw, 'BW');
      required('body-reference-load').textContent = format(first?.medialBw, 'BW');
      required('body-selected-load').textContent = format(second?.medialBw, 'BW');
      readout(groundReadouts, 'Right vertical ground force', first?.grfBw, second?.grfBw, 'BW');
      readout(groundReadouts, 'External knee adduction moment', first?.kamNmKg, second?.kamNmKg, 'N·m/kg', 3);
      readout(groundReadouts, 'External knee flexion moment', first?.kfmNmKg, second?.kfmNmKg, 'N·m/kg', 3);
      readout(loadReadouts, 'Medial force', first?.medialBw, second?.medialBw, 'BW');
      readout(loadReadouts, 'Lateral force', first?.lateralBw, second?.lateralBw, 'BW');
      readout(loadReadouts, 'Early medial peak', reference.peaks?.medial?.value, selected.peaks?.medial?.value, 'BW');
    }
  }

  function putPhase(phase: number, updateReadouts = true) {
    // The displayed phase can be rounded by its text formatter. Keep the
    // true event time for the pose and stance/swing boundary.
    currentPhase = phase;
    phaseControl.value = String(phase);
    showPhase(updateReadouts);
  }

  function syncSelected() {
    required('body-reference-label').textContent = reference.poseLabel;
    required('body-selected-label').textContent = selected.poseLabel;
    required('body-reference-detail').textContent = reference.conditionDetail;
    required('body-selected-detail').textContent = selected.conditionDetail;
    for (const [id, stride] of [['body-reference-track', reference], ['body-selected-track', selected]] as const) {
      const track = required(id);
      track.style.setProperty('--stance', `${stride.clip.stanceEndPct}%`);
      track.setAttribute('aria-label', `${stride.label}: right toe-off at ${format(stride.clip.stanceEndPct, '%', 1)}`);
    }
    required('body-movement-description').textContent = collection === 'implant'
      ? descriptions[selected.clip.condition] ?? 'Both strides come from the same implant participant.'
      : 'Wrist travel is the recorded fore–aft range relative to the pelvis. The recordings run at the same stride phase.';
    const fixed = (label: string, a: number | null | undefined, b: number | null | undefined, unit: string, places = 1) =>
      setReadout(motionReadouts, label, reference.label, selected.label, format(a, unit, places), format(b, unit, places));
    fixed('Right wrist travel', reference.wristTravelCm.right, selected.wristTravelCm.right, 'cm');
    fixed('Left wrist travel', reference.wristTravelCm.left, selected.wristTravelCm.left, 'cm');
    fixed('Pelvis vertical excursion', reference.pelvisExcursionCm, selected.pelvisExcursionCm, 'cm');
    fixed('Recorded stride duration', reference.clip.durationSeconds, selected.clip.durationSeconds, 's', 2);
    if (collection === 'reference') {
      if (reference.clip.heightM != null || selected.clip.heightM != null) {
        fixed('Recorded height', reference.clip.heightM, selected.clip.heightM, 'm', 2);
      }
      if (reference.clip.massKg != null || selected.clip.massKg != null) {
        fixed('Recorded mass', reference.clip.massKg, selected.clip.massKg, 'kg');
      }
    }
    for (const button of events) button.disabled = jumpTo(button.dataset.bodyEvent ?? '') == null;
  }

  function readLayers<T>(attribute: string): T {
    return Object.fromEntries(Array.from(document.querySelectorAll<HTMLInputElement>(`[data-${attribute}]`))
      .map((input) => [input.getAttribute(`data-${attribute}`), input.checked])) as T;
  }

  function syncLayers() {
    const layers = readLayers<BodyLayers>('body-layer');
    const showSurface = Boolean(layers.context);
    for (const name of ['muscles', 'skeleton'] as const) {
      const checkbox = wholeHost.closest('.body-plate')?.querySelector<HTMLInputElement>(`[data-body-layer="${name}"]`);
      if (checkbox) checkbox.disabled = showSurface;
      if (showSurface) layers[name] = false;
    }
    if (collection === 'reference') {
      layers.forces = false;
      layers.paths = false;
    }
    wholeScene?.setLayers(layers);
    kneeScene?.setLayers(readLayers<TissueLayers>('body-knee-layer'));
  }

  function syncCamera() {
    if (mode === 'whole') wholeScene?.setView(camera.value as BodyView);
    else kneeScene?.setView((camera.value === 'oblique' ? 'anterior' : camera.value === 'side' ? 'lateral' : camera.value) as TissueView);
  }

  function syncPoseNote() {
    required('body-pose-note').textContent = mode === 'knee'
      ? collection === 'implant'
        ? 'The knee atlas follows recorded flexion. Cartilage colors show measured implant forces during instrumented stance.'
        : 'Cartilage, menisci and ligaments follow the recorded right knee flexion on a reference atlas.'
      : collection === 'implant'
        ? 'Ground-force arrows start at the measured center of pressure. Select a knee marker for the synchronized tissue view.'
        : 'The same reference atlas follows each person’s recorded motion. Select a knee marker for the synchronized tissue view.';
  }

  function showMode(next: Mode) {
    mode = next;
    wholeHost.hidden = mode !== 'whole';
    kneeHost.hidden = mode !== 'knee';
    required('body-layers').hidden = mode !== 'whole';
    required('body-knee-layers').hidden = mode !== 'knee';
    syncPoseNote();
    for (const button of modeButtons) button.setAttribute('aria-pressed', String(button.dataset.bodyMode === mode));

    const ready = mode === 'whole' ? wholeScene : kneeScene;
    status.hidden = Boolean(ready);
    if (ready) {
      syncCamera();
      showPhase();
      return;
    }
    status.textContent = mode === 'whole' ? 'Loading the musculoskeletal atlas…' : 'Loading knee tissues…';
    if (mode === 'whole') {
      pendingWhole ??= import('../../upper-limb/knee-loading/body-scene.ts').then(async ({ createBodyScene }) => {
        wholeScene = await createBodyScene(wholeHost, { onKneeActivate: () => showMode('knee') });
      });
    } else {
      pendingKnee ??= import('../../upper-limb/knee-loading/tissue-scene.ts').then(async ({ createTissueScene }) => {
        kneeScene = await createTissueScene(kneeHost);
      });
    }
    const pending = next === 'whole' ? pendingWhole : pendingKnee;
    void pending?.then(() => {
      syncLayers();
      if (mode === next) {
        syncCamera();
        showPhase();
        status.hidden = true;
      }
    }).catch((error: unknown) => {
      if (next === 'whole') pendingWhole = undefined;
      else pendingKnee = undefined;
      if (mode !== next) return;
      console.warn('The gait model could not be loaded', error);
      status.textContent = next === 'whole'
        ? 'Whole-body anatomy could not load. The knee view and recorded readouts are available.'
        : 'Knee anatomy could not load. The recorded gait and load readouts are available.';
    });
  }

  function pause() {
    if (!playback) return;
    cancelAnimationFrame(animationFrame);
    playback = undefined;
    play.setAttribute('aria-pressed', 'false');
    playLabel.textContent = 'Play stride';
    showPhase();
  }

  function tick(time: number) {
    if (!playback) return;
    const strideMs = 1000 * reference.clip.durationSeconds / Number(speed.value);
    const next = advanceStride(playback, time, strideMs);
    playback = next.playback;
    if (next.phase !== currentPhase) {
      const updateReadouts = next.phase === 0 || next.phase === 100 || time - lastReadoutTime >= 70;
      if (updateReadouts) lastReadoutTime = time;
      putPhase(next.phase, updateReadouts);
    }
    animationFrame = requestAnimationFrame(tick);
  }

  function startPlayback() {
    if (playback) return;
    if (currentPhase === 100) putPhase(0);
    playback = { startedAt: performance.now(), startPhase: currentPhase };
    play.setAttribute('aria-pressed', 'true');
    playLabel.textContent = 'Pause stride';
    animationFrame = requestAnimationFrame(tick);
  }

  function fillComparison() {
    const options = collection === 'implant' ? modified : recorded.filter((item) => item !== recordedFirst);
    condition.replaceChildren(...options.map((variant) => new Option(variant.label, variant.clip.id)));
    condition.value = selected.clip.id;
  }

  function chooseCollection(next: Collection) {
    if (next === 'reference' && recorded.length < 2) return;
    pause();
    collection = next;
    const isImplant = collection === 'implant';
    reference = isImplant ? implantReference : recordedFirst;
    selected = isImplant ? implantSelection : recordedSecond;
    referencePicker.value = reference.clip.id;
    required('body-reference-picker').hidden = isImplant;
    required('body-condition-caption').textContent = isImplant ? 'Compare with ordinary walking' : 'Second recording';
    required('body-reference-card').hidden = isImplant;
    required('body-pathway').classList.toggle('is-optical', !isImplant);
    document.querySelectorAll<HTMLElement>('[data-implant-only]').forEach((item) => { item.hidden = !isImplant; });
    for (const button of sourceButtons) button.setAttribute('aria-pressed', String(button.dataset.bodySource === collection));
    const source = required<HTMLAnchorElement>('body-motion-source');
    source.href = reference.archive.source.url;
    source.title = reference.archive.source.title;
    source.textContent = isImplant ? 'Implant data source' : 'Optical motion data source';
    motionReadouts.replaceChildren();
    fillComparison();
    syncSelected();
    syncLayers();
    syncPoseNote();
    putPhase(isImplant ? jumpTo('peak') ?? 15 : 0);
  }

  referencePicker.replaceChildren(...recorded.map((stride) => new Option(stride.label, stride.clip.id)));
  referencePicker.disabled = recorded.length < 2;
  referencePicker.addEventListener('change', () => {
    userControlled = true;
    const next = recorded.find(({ clip }) => clip.id === referencePicker.value);
    if (!next || next === recordedFirst) return;
    const wasPlaying = Boolean(playback);
    pause();
    if (next === recordedSecond) recordedSecond = recordedFirst;
    recordedFirst = reference = next;
    selected = recordedSecond;
    fillComparison();
    syncSelected();
    showPhase();
    if (wasPlaying) startPlayback();
  });
  condition.addEventListener('change', () => {
    userControlled = true;
    const options = collection === 'implant' ? modified : recorded;
    selected = options.find(({ clip }) => clip.id === condition.value) ?? selected;
    if (collection === 'implant') implantSelection = selected;
    else recordedSecond = selected;
    syncSelected();
    showPhase();
  });
  phaseControl.addEventListener('input', () => {
    userControlled = true;
    const next = phaseControl.valueAsNumber;
    pause();
    putPhase(next);
  });
  for (const button of events) button.addEventListener('click', () => {
    userControlled = true;
    const phase = jumpTo(button.dataset.bodyEvent ?? '');
    if (phase == null) return;
    pause();
    putPhase(phase);
  });
  play.addEventListener('click', () => {
    userControlled = true;
    if (playback) pause();
    else startPlayback();
  });
  speed.addEventListener('change', () => {
    if (playback) {
      playback = { startedAt: performance.now(), startPhase: currentPhase };
    }
  });
  for (const button of sourceButtons) {
    button.disabled = button.dataset.bodySource === 'reference' && recorded.length < 2;
    button.addEventListener('click', () => {
      userControlled = true;
      const next = button.dataset.bodySource as Collection;
      if (collection !== next) chooseCollection(next);
    });
  }
  required('body-open-implant').addEventListener('click', () => {
    userControlled = true;
    chooseCollection('implant');
    sourceButtons.find((button) => button.dataset.bodySource === 'implant')?.focus();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause();
  });
  for (const button of modeButtons) button.addEventListener('click', () => showMode(button.dataset.bodyMode === 'knee' ? 'knee' : 'whole'));
  for (const input of document.querySelectorAll<HTMLInputElement>('[data-body-layer], [data-body-knee-layer]')) {
    input.addEventListener('change', syncLayers);
  }
  camera.addEventListener('change', syncCamera);
  required('body-zoom-in').addEventListener('click', () => (mode === 'whole' ? wholeScene : kneeScene)?.zoomBy(1.22));
  required('body-zoom-out').addEventListener('click', () => (mode === 'whole' ? wholeScene : kneeScene)?.zoomBy(1 / 1.22));
  required('body-reset').addEventListener('click', () => {
    (mode === 'whole' ? wholeScene : kneeScene)?.reset();
    camera.value = mode === 'whole' ? 'oblique' : 'anterior';
  });

  chooseCollection(collection);
  showMode('whole');
  let firstView = true;
  let visible = false;
  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) {
      pause();
      return;
    }
    if (firstView && collection === 'reference' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      void pendingWhole?.then(() => {
        if (visible && !userControlled && !document.hidden && collection === 'reference' && mode === 'whole') startPlayback();
      }).catch(() => {});
    }
    firstView = false;
  }, { threshold: 0.2 });
  observer.observe(required('gait-path'));
}
