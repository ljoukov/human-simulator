import articulation from './model/articulation.json' with { type: 'json' };
import stride from './model/stride.json' with { type: 'json' };
import { BAND_STIFFNESS_N_PER_MM, bandPath } from './model/itb-path.js';

// Sex and age describe each virtual profile. The solver scales geometry by stature
// and inertia by mass and stature. It does not assign weakness from age or sex.
export const PROFILES = Object.freeze([
  { id: 'woman40', label: 'Woman · 40 years', age: 40, sex: 'Female', massKg: 66, heightCm: 166 },
  { id: 'man55', label: 'Man · 55 years', age: 55, sex: 'Male', massKg: 82, heightCm: 180 },
  { id: 'woman70', label: 'Woman · 70 years', age: 70, sex: 'Female', massKg: 61, heightCm: 158 },
]);
export const DEFAULT_SETTINGS = Object.freeze({
  profileId: 'woman40',
  gluteCapacityPct: 65,
  bandShorteningMm: 3,
  task: 'walk',
  strategy: 'effort',
});
const RADIANS = Math.PI / 180;
const G = 9.80665;
const CHANNELS = [
  'pelvisDropDeg',
  'trunkLeanDeg',
  'hipAdductionDeg',
  'leftHipAdductionDeg',
  'hipFlexionDeg',
  'leftHipFlexionDeg',
  'kneeFlexionDeg',
  'leftKneeFlexionDeg',
  'ankleDorsiflexionDeg',
  'leftAnkleDorsiflexionDeg',
  'shoulderFlexionDeg',
  'leftShoulderFlexionDeg',
  'shoulderAdductionDeg',
  'leftShoulderAdductionDeg',
  'elbowFlexionDeg',
  'leftElbowFlexionDeg',
];
const N = CHANNELS.length;
// Effective posture servo stiffness at reference size, N·m / rad. Gains describe a
// tracking strategy; their values are not a patient-specific physiological estimate.
const SERVO = [65, 190, 85, 85, 85, 85, 42, 42, 26, 26, 9, 9, 12, 12, 4, 4];
const DAMPING_TIME = 0.115;
const nominalBandLength = bandPath(0, 0, 0).lengthM;
const prepared = new Map();
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = (x) => x - Math.floor(x);

function walkingValue(channel, phase) {
  const values = stride.channels[channel];
  const position = wrap(phase) * values.length;
  const i = Math.floor(position), t = position - i;
  return values[i] + t * (values[(i + 1) % values.length] - values[i]);
}

function referenceAt(task, phase) {
  const angles = new Float64Array(N);
  if (task === 'stance') {
    const w = Math.sin(Math.PI * phase) ** 2;
    angles.set([0, .8 * w, 1.5 + 3.5 * w, 0, 7 - 5 * w, 15 + 12 * w, 7, 15 + 12 * w, 2, 1, 0, 0, -4, -4, 15, 15]);
    return {
      angles,
      rightForce: .5 + .5 * w,
      leftForce: .5 * (1 - w),
      rightMoment: .02 + .10 * w,
      leftMoment: .02 * (1 - w),
    };
  }
  CHANNELS.forEach((name, i) => {
    const left = name.startsWith('left');
    const source = left ? name[4].toLowerCase() + name.slice(5) : name;
    angles[i] = walkingValue(source, phase + (left ? .5 : 0));
  });
  return {
    angles,
    rightForce: walkingValue('groundForceBw', phase),
    leftForce: walkingValue('groundForceBw', phase + .5),
    rightLandingTarget: .45 * walkingValue('groundForceBw', phase + .07),
    leftLandingTarget: .45 * walkingValue('groundForceBw', phase + .57),
    rightMoment: walkingValue('hipMomentBwM', phase),
    leftMoment: walkingValue('hipMomentBwM', phase + .5),
  };
}

function inertiaAt(task, phase, scale) {
  if (task === 'stance') return Float64Array.from(articulation.stanceMatrix, (value) => value * scale);
  const p = phase * (articulation.phases.length - 1);
  const lo = Math.floor(p), hi = Math.min(articulation.phases.length - 1, lo + 1);
  const a = articulation.matrices[lo], b = articulation.matrices[hi];
  return Float64Array.from(a, (value, i) => scale * (value + (p - lo) * (b[i] - value)));
}

// A compliant constraint tracks the ankle's reference position when its foot carries
// weight. The two segment Jacobian projects ankle displacement into joint torques.
function footSupport(K, ref, side, bodyHeight, bw) {
  const right = side === 1, add = right ? 2 : 3, hip = right ? 4 : 5, knee = right ? 6 : 7;
  // Start targeting the next measured placement in late swing while keeping the
  // prescribed vertical load unmodified. The anticipation horizon is 7% stride.
  const weight = clamp(
    right ? Math.max(ref.rightForce, ref.rightLandingTarget ?? 0) : Math.max(ref.leftForce, ref.leftLandingTarget ?? 0),
    0,
    1,
  );
  const thigh = .245 * bodyHeight, shank = .246 * bodyHeight;
  const f = ref.angles[hip] * RADIANS, s = f - ref.angles[knee] * RADIANS;
  const Jx = thigh * Math.cos(f) + shank * Math.cos(s), Jy = thigh * Math.sin(f) + shank * Math.sin(s);
  const spring = 65 * bw / bodyHeight * weight;
  const lateral = [[0, Jx], [add, -side * Jx]];
  const forward = [[hip, Jx], [knee, -shank * Math.cos(s)]];
  const vertical = [[hip, Jy], [knee, -shank * Math.sin(s)]];
  for (const row of [lateral, forward, vertical]) {
    for (const [i, a] of row) for (const [j, b] of row) K[i * N + j] += spring * a * b;
  }
}

function cholesky(matrix) {
  const lower = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      let value = matrix[i * N + j];
      for (let k = 0; k < j; k++) value -= lower[i * N + k] * lower[j * N + k];
      lower[i * N + j] = i === j ? Math.sqrt(value) : value / lower[j * N + j];
    }
  }
  return lower;
}
function solveFactor(lower, b, v) {
  for (let i = 0; i < N; i++) {
    let x = b[i];
    for (let j = 0; j < i; j++) x -= lower[i * N + j] * v[j];
    v[i] = x / lower[i * N + i];
  }
  for (let i = N - 1; i >= 0; i--) {
    let x = v[i];
    for (let j = i + 1; j < N; j++) x -= lower[j * N + i] * v[j];
    v[i] = x / lower[i * N + i];
  }
}

function prepare(profile, task, steps) {
  const key = `${profile.id}/${task}/${steps}`;
  if (prepared.has(key)) return prepared.get(key);
  const height = profile.heightCm / 100, size = height / 1.70, bw = profile.massKg * G;
  const durationSeconds = task === 'walk' ? stride.strideLengthM / stride.referenceSpeedMps * Math.sqrt(size) : 2.2;
  const dt = durationSeconds / steps;
  const torqueScale = bw * size / (articulation.massKg * G);
  const slots = Array.from({ length: steps }, (_, i) => {
    const phase = i / steps, ref = referenceAt(task, phase);
    const mass = inertiaAt(task, phase, profile.massKg / articulation.massKg * size * size);
    const stiffness = new Float64Array(N * N);
    SERVO.forEach((k, j) => {
      stiffness[j * N + j] = k * torqueScale;
    });
    footSupport(stiffness, ref, 1, height, bw);
    footSupport(stiffness, ref, -1, height, bw);
    // The anchored foot approaches its reference on contact faster than the body's
    // postural response. A single damping time would leave excess touchdown drift.
    const supportDampingTime = 0.018;
    const effective = mass.map((m, j) => m + (dt * supportDampingTime + dt * dt) * stiffness[j]);
    SERVO.forEach((k, j) => {
      effective[j * N + j] += dt * (DAMPING_TIME - supportDampingTime) * k * torqueScale;
    });
    const factor = cholesky(effective);
    const path = bandPath(ref.angles[4], ref.angles[2], ref.angles[6]);
    const tension = Math.max(0, (path.lengthM - nominalBandLength) * size * 1000) * BAND_STIFFNESS_N_PER_MM * size;
    return { ref, mass, stiffness, factor, path, tension };
  });
  const model = { slots, dt, size, bw, durationSeconds };
  // Only interactive resolutions are cached; expensive refinement runs can release their workspaces.
  if (steps === 800) prepared.set(key, model);
  return model;
}

function loads(slot, q, model, settings) {
  const { size, bw } = model, { ref } = slot;
  const path = bandPath(
    ref.angles[4] + q[4] / RADIANS,
    ref.angles[2] + q[2] / RADIANS,
    ref.angles[6] + q[6] / RADIANS,
  );
  const tension = Math.max(0, (path.lengthM - nominalBandLength) * size * 1000 + settings.bandShorteningMm) *
    BAND_STIFFNESS_N_PER_MM * size;
  const capacity = .125 * bw * size, rightCap = capacity * settings.gluteCapacityPct / 100;
  const rightDemand = ref.rightMoment * bw * size, leftDemand = ref.leftMoment * bw * size;
  const rightLever = .28 * size * bw * ref.rightForce, leftLever = .28 * size * bw * ref.leftForce;
  const rightExternal = rightDemand - rightLever * Math.sin(q[1]);
  const leftExternal = leftDemand + leftLever * Math.sin(q[1]);
  const recruit = (external, cap) => clamp(external, 0, cap);
  // The group supplies a frontal-plane moment. A bilateral support model and a
  // tracking servo retain the remainder of the measured gait's nominal actuation.
  const rightActive = recruit(rightExternal, rightCap), leftActive = recruit(leftExternal, capacity);
  const rhs = new Float64Array(N);
  // The two assumed controllers use the same capacity and trunk-lean cost.
  // One conserves total effort; the other responds only to demand above capacity.
  // Both use the local linearization of the stance-side gravitational lever arm.
  const leanTarget = (cap) => {
    const leanCost = .21 ** 2;
    const demand = settings.strategy === 'reserve' ? rightDemand - cap : rightDemand;
    return Math.max(0, demand) * rightLever * leanCost / (cap * cap + rightLever * rightLever * leanCost);
  };
  rhs[1] = SERVO[1] * bw * size / (articulation.massKg * G) * (leanTarget(rightCap) - leanTarget(capacity));
  rhs[2] = (rightExternal - rightActive) - (rightDemand - recruit(rightDemand, capacity));
  rhs[3] = (leftExternal - leftActive) - (leftDemand - recruit(leftDemand, capacity));
  for (const [index, derivative] of [[2, 'dAddMPerRad'], [4, 'dFlexMPerRad'], [6, 'dKneeMPerRad']]) {
    rhs[index] -= size * (tension * path[derivative] - slot.tension * slot.path[derivative]);
  }
  return {
    rhs,
    tension,
    groupForce: rightActive / (.052 * size),
    abductorUtilizationPct: 100 * rightActive / rightCap,
    hipMomentNm: rightExternal,
    bandLengthMm: path.lengthM * size * 1000,
  };
}
function sampleState(slot, q, model, settings, phase) {
  const sample = loads(slot, q, model, settings), frame = { phase };
  CHANNELS.forEach((name, i) => {
    frame[name] = slot.ref.angles[i] + q[i] / RADIANS;
  });
  frame.bandForceN = sample.tension;
  frame.bandLengthMm = sample.bandLengthMm;
  frame.abductorForceN = sample.groupForce;
  frame.abductorUtilizationPct = sample.abductorUtilizationPct;
  frame.hipMomentNm = sample.hipMomentNm;
  frame.groundForceBw = slot.ref.rightForce;
  return frame;
}
function trajectory(model, settings) {
  const { slots, dt } = model, q = new Float64Array(N), v = new Float64Array(N), b = new Float64Array(N);
  const frames = [];
  // Repeating the periodic forcing settles the initial transient. Record only the last cycle.
  const warm = 4, count = slots.length;
  for (let cycle = 0; cycle <= warm; cycle++) {
    for (let i = 0; i < count; i++) {
      const slot = slots[i];
      if (cycle === warm && i % (count / 100) === 0) frames.push(sampleState(slot, q, model, settings, i / count));
      const { rhs } = loads(slot, q, model, settings);
      for (let k = 0; k < N; k++) {
        let z = dt * rhs[k];
        for (let j = 0; j < N; j++) z += slot.mass[k * N + j] * v[j] - dt * slot.stiffness[k * N + j] * q[j];
        b[k] = z;
      }
      solveFactor(slot.factor, b, v);
      for (let j = 0; j < N; j++) q[j] += dt * v[j];
    }
  }
  frames.push(sampleState(slots[0], q, model, settings, 1));
  return frames;
}

export function simulateGait(input = {}, { steps = 800 } = {}) {
  if (!Number.isInteger(steps) || steps < 100 || steps > 1600 || steps % 100) {
    throw new RangeError('Integration steps must be a multiple of 100 in [100,1600].');
  }
  const profile = PROFILES.find((p) => p.id === (input.profileId ?? DEFAULT_SETTINGS.profileId));
  const settings = { ...DEFAULT_SETTINGS, ...input };
  if (!profile || !['walk', 'stance'].includes(settings.task)) {
    throw new RangeError('Unknown virtual profile or movement.');
  }
  if (!['effort', 'reserve'].includes(settings.strategy)) throw new RangeError('Unknown compensation strategy.');
  if (!Number.isFinite(settings.gluteCapacityPct) || !Number.isFinite(settings.bandShorteningMm)) {
    throw new TypeError('Finite capacity and shortening required.');
  }
  settings.gluteCapacityPct = clamp(settings.gluteCapacityPct, 40, 100);
  settings.bandShorteningMm = clamp(settings.bandShorteningMm, 0, 8);
  const model = prepare(profile, settings.task, steps);
  const zero = new Float64Array(N);
  const nominal = { ...settings, gluteCapacityPct: 100, bandShorteningMm: 0 };
  const baseline = Array.from(
    { length: 101 },
    (_, i) => sampleState(model.slots[(i * steps / 100) % steps], zero, model, nominal, i / 100),
  );
  const altered = trajectory(model, settings);
  return {
    profile,
    settings,
    durationSeconds: model.durationSeconds,
    baseline,
    altered,
    metrics: {
      peakBandForceN: Math.max(...altered.map((f) => f.bandForceN)),
      peakRightwardLeanDeg: Math.max(...altered.map((f) => f.trunkLeanDeg)),
    },
  };
}

export function sampleFrame(frames, phase01) {
  if (!Number.isFinite(phase01)) throw new TypeError('Finite movement phase required.');
  const p = clamp(phase01, 0, 1) * (frames.length - 1), i = Math.floor(p), t = p - i;
  const a = frames[i], b = frames[Math.min(i + 1, frames.length - 1)];
  if (t === 0) return a;
  return Object.fromEntries(Object.keys(a).map((key) => [key, a[key] + (b[key] - a[key]) * t]));
}
