import type { GaitFrame, GaitTrial } from './dataset.ts';

export const MEDIAL_KNEE_PAIN_EVIDENCE = {
  url: 'https://doi.org/10.1016/S2665-9913(25)00151-1',
  authors: 'Uhlrich et al.',
  year: 2025,
  participants: 68,
  followUpMonths: 12,
  betweenGroupDifferencePoints: -1.2,
  confidenceIntervalPoints: [-1.9, -0.5],
  outcomeScale: '0–10',
  description: 'A separate randomized trial in people with medial knee osteoarthritis compared personalized foot-angle retraining with sham retraining. The retraining group reported less medial knee pain at one year.',
} as const;

export interface GaitPathwaySample {
  stridePct: number;
  stancePct: number;
  grfBw: number | null;
  kamNmKg: number | null;
  kfmNmKg: number | null;
  compressionBw: number | null;
  kneeFlexionDeg: number | null;
  medialBw: number | null;
  lateralBw: number | null;
  totalBw: number | null;
}

export interface GaitPathwayPeak {
  value: number;
  stancePct: number;
  stridePct: number;
}

export const EARLY_STANCE_WINDOW_PCT = [5, 45] as const;
const SAMPLE_EPSILON_PCT = 1e-9;

function isValidStanceEnd(stanceEndPct: number | null | undefined): stanceEndPct is number {
  return typeof stanceEndPct === 'number' && Number.isFinite(stanceEndPct)
    && stanceEndPct > 0 && stanceEndPct <= 100;
}

function sampleAtStance(
  frames: readonly GaitFrame[],
  stanceEndPct: number,
  stancePct: number,
): GaitPathwaySample | null {
  if (!frames.length || stancePct < frames[0].stancePct || stancePct > frames[frames.length - 1].stancePct) {
    return null;
  }

  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (frames[mid].stancePct < stancePct - SAMPLE_EPSILON_PCT) lo = mid + 1;
    else hi = mid;
  }

  const right = frames[lo];
  const exact = Math.abs(right.stancePct - stancePct) <= SAMPLE_EPSILON_PCT;
  const left = lo === 0 || exact ? right : frames[lo - 1];
  const weight = left === right ? 0 : (stancePct - left.stancePct) / (right.stancePct - left.stancePct);
  function interpolate(signal: Exclude<keyof GaitFrame, 'stancePct'>): number | null {
    const a = left[signal];
    const b = right[signal];
    if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return null;
    return a + (b - a) * weight;
  }

  return {
    stridePct: stancePct * stanceEndPct / 100,
    stancePct,
    grfBw: interpolate('grfBw'),
    kamNmKg: interpolate('kamNmKg'),
    kfmNmKg: interpolate('kfmNmKg'),
    compressionBw: interpolate('compressionBw'),
    kneeFlexionDeg: interpolate('kneeFlexionDeg'),
    medialBw: interpolate('medialBw'),
    lateralBw: interpolate('lateralBw'),
    totalBw: interpolate('totalBw'),
  };
}

/** The recorded stance starts at 0% of the clip. Loads are unavailable during swing. */
export function sampleGaitPathway(
  trial: GaitTrial | null | undefined,
  stanceEndPct: number | null | undefined,
  stridePct: number,
): GaitPathwaySample | null {
  if (!trial || !isValidStanceEnd(stanceEndPct) || !Number.isFinite(stridePct)
    || stridePct < 0 || stridePct > stanceEndPct) {
    return null;
  }
  return sampleAtStance(trial.frames, stanceEndPct, stridePct / stanceEndPct * 100);
}

type Signal = Exclude<keyof GaitPathwaySample, 'stridePct' | 'stancePct'>;

/** Signed maxima of the recorded signals from 5% to 45% of stance, inclusive. */
export function earlyStancePeaks(trial: GaitTrial | null | undefined, stanceEndPct: number | null | undefined) {
  if (!trial || !isValidStanceEnd(stanceEndPct) || !trial.frames.length) return null;
  const [start, end] = EARLY_STANCE_WINDOW_PCT;
  const positions = [
    start,
    ...trial.frames.filter(({ stancePct }) => stancePct > start && stancePct < end).map(({ stancePct }) => stancePct),
    end,
  ];
  const samples = positions.map((position) => sampleAtStance(trial.frames, stanceEndPct, position));

  function peak(signal: Signal): GaitPathwayPeak | null {
    let best: GaitPathwayPeak | null = null;
    for (const row of samples) {
      const value = row?.[signal];
      if (row && value != null && (best == null || value > best.value)) {
        best = { value, stancePct: row.stancePct, stridePct: row.stridePct };
      }
    }
    return best;
  }

  return {
    grf: peak('grfBw'),
    kam: peak('kamNmKg'),
    kfm: peak('kfmNmKg'),
    medial: peak('medialBw'),
    lateral: peak('lateralBw'),
    total: peak('totalBw'),
    flexion: peak('kneeFlexionDeg'),
  };
}
