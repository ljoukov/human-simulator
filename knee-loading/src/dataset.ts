import { FRAME_STRIDE, REFERENCE_FLEXION_RAD } from './mechanics.ts';

export const GRAVITY_M_S2 = 9.80665;
export type KneeSide = 'left' | 'right';

export interface GaitFrame {
  stancePct: number;
  kamNmKg: number;
  kfmNmKg: number;
  grfBw: number;
  compressionBw?: number | null;
  kneeFlexionDeg?: number | null;
  medialBw?: number | null;
  lateralBw?: number | null;
  totalBw?: number | null;
}

export interface GaitTrial {
  id: string;
  subject: string;
  condition: string;
  role?: 'calibration' | 'evaluation';
  massKg: number;
  heightM?: number | null;
  stanceSeconds?: number | null;
  frames: GaitFrame[];
}

export interface GaitDataset {
  source: {
    title: string;
    url: string;
    citation: string;
    license: string;
    notes: string[];
    kind: 'trial' | 'condition-mean';
    recordedSide?: KneeSide;
  };
  trials: GaitTrial[];
}

export function validateDataset(dataset: GaitDataset): GaitDataset {
  if (!dataset.source?.title || !dataset.source?.url || !dataset.trials?.length) {
    throw new Error('Expected a data source and at least one gait trial.');
  }
  const side = dataset.source.recordedSide;
  if (side != null && side !== 'left' && side !== 'right') {
    throw new Error('Recorded knee side must be left or right.');
  }

  const ids = new Set<string>();
  for (const trial of dataset.trials) {
    if (!trial.id || !trial.subject || !trial.condition || !Number.isFinite(trial.massKg) || trial.massKg <= 0) {
      throw new Error('Trial identifiers, participant, condition and body mass are required.');
    }
    const trialKey = JSON.stringify([trial.subject, trial.id]);
    if (ids.has(trialKey)) throw new Error(`Duplicate trial ${trial.id} for ${trial.subject}.`);
    ids.add(trialKey);

    const frames = trial.frames;
    if (!Array.isArray(frames) || frames.length < 20 || frames[0].stancePct !== 0 || frames.at(-1)?.stancePct !== 100) {
      throw new Error(`${trial.id} needs a complete, resampled stance (0–100%).`);
    }
    if (trial.stanceSeconds != null && (!Number.isFinite(trial.stanceSeconds) || trial.stanceSeconds <= 0)) {
      throw new Error(`${trial.id} has an invalid stance duration.`);
    }
    for (const [index, frame] of frames.entries()) {
      const inputs = [frame.stancePct, frame.kamNmKg, frame.kfmNmKg, frame.grfBw];
      const optional = [frame.compressionBw, frame.kneeFlexionDeg, frame.medialBw, frame.lateralBw, frame.totalBw];
      if (inputs.some((value) => !Number.isFinite(value)) || optional.some((value) => value != null && !Number.isFinite(value))) {
        throw new Error(`Invalid sample at frame ${index} of ${trial.id}.`);
      }
      if (index && frame.stancePct <= frames[index - 1].stancePct) {
        throw new Error(`Stance frames must be in increasing order for ${trial.id}.`);
      }
    }
  }
  return dataset;
}

export function encodeFrames(trials: readonly GaitTrial[]): Float32Array {
  const encoded = new Float32Array(trials.reduce((count, trial) => count + trial.frames.length, 0) * FRAME_STRIDE);
  let offset = 0;
  for (const trial of trials) {
    for (const frame of trial.frames) {
      encoded[offset++] = frame.compressionBw ?? frame.grfBw;
      encoded[offset++] = frame.kamNmKg / GRAVITY_M_S2;
      encoded[offset++] = frame.kfmNmKg / GRAVITY_M_S2;
      encoded[offset++] = frame.kneeFlexionDeg == null ? REFERENCE_FLEXION_RAD : frame.kneeFlexionDeg * Math.PI / 180;
    }
  }
  return encoded;
}
