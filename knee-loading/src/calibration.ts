import { prepareCalibrationBenchmark, withinStance } from './backend-calibration.ts';
import type { GaitFrame, GaitTrial } from './dataset.ts';
import { PARAM_STRIDE, solveCpu } from './mechanics.ts';
import { rankCandidates } from './sampling.ts';
import type { Compartment } from './view-types.ts';

export { withinStance };

export const COMPARTMENTS: readonly Compartment[] = ['medial', 'lateral', 'total'];

export function measuredForce(frame: GaitFrame, compartment: Compartment): number | null {
  if (compartment === 'medial') return frame.medialBw ?? null;
  if (compartment === 'lateral') return frame.lateralBw ?? null;
  if (frame.totalBw != null) return frame.totalBw;
  if (frame.medialBw != null && frame.lateralBw != null) return frame.medialBw + frame.lateralBw;
  return null;
}

export type SolveBatch = (
  frames: Float32Array, parameters: Float32Array, observations?: Float32Array,
) => Float32Array | Promise<Float32Array>;

export const solveBatchCpu: SolveBatch = (frames, parameters, observations) =>
  solveCpu(frames, parameters, undefined, observations);

export async function filterParameters(
  trials: GaitTrial[], candidates: Float32Array, retain: number, solve: SolveBatch,
): Promise<Float32Array> {
  const { frames, observations } = prepareCalibrationBenchmark(trials);
  const losses = await solve(frames, candidates, observations);
  const ranked = rankCandidates(losses, retain);
  const selected = new Float32Array(ranked.length * PARAM_STRIDE);
  for (let scenario = 0; scenario < ranked.length; scenario++) {
    const index = ranked[scenario];
    selected.set(candidates.subarray(index * PARAM_STRIDE, (index + 1) * PARAM_STRIDE), scenario * PARAM_STRIDE);
  }
  return selected;
}

type Predictor = (frame: GaitFrame) => number;
export interface Proxies {
  adduction?: Predictor;
  moments?: Predictor;
}

/** Least squares with a two-variable centred solve and an intercept. */
export function fitProxies(trials: GaitTrial[], compartment: Compartment): Proxies {
  const rows = trials.flatMap(({ frames }) => frames.filter(withinStance).flatMap((frame) => {
    const force = measuredForce(frame, compartment);
    return force == null ? [] : [[frame.kamNmKg, Math.abs(frame.kfmNmKg), force]];
  }));
  if (rows.length < 3) return {};

  const xMean = rows.reduce((sum, row) => sum + row[0], 0) / rows.length;
  const yMean = rows.reduce((sum, row) => sum + row[1], 0) / rows.length;
  const zMean = rows.reduce((sum, row) => sum + row[2], 0) / rows.length;

  let xx = 0, yy = 0, xy = 0, xz = 0, yz = 0;
  for (const [adduction, flexion, force] of rows) {
    const x = adduction - xMean;
    const y = flexion - yMean;
    const z = force - zMean;
    xx += x * x;
    yy += y * y;
    xy += x * y;
    xz += x * z;
    yz += y * z;
  }

  if (xx <= 1e-12) return {};
  const singleSlope = xz / xx;
  const adduction: Predictor = (frame) => Math.max(0, zMean + singleSlope * (frame.kamNmKg - xMean));
  const determinant = xx * yy - xy * xy;
  if (determinant <= 1e-10 * xx * yy) return { adduction };
  const adductionSlope = (yy * xz - xy * yz) / determinant;
  const flexionSlope = (xx * yz - xy * xz) / determinant;
  const moments: Predictor = (frame) => Math.max(0, zMean
    + adductionSlope * (frame.kamNmKg - xMean)
    + flexionSlope * (Math.abs(frame.kfmNmKg) - yMean));
  return { adduction, moments };
}
