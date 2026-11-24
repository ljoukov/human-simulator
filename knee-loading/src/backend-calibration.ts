import { encodeFrames, type GaitFrame, type GaitTrial } from './dataset.ts'
import {
  CALIBRATION_RESIDUAL_LEVER_M, FRAME_STRIDE, MOMENT_RESIDUAL_TOLERANCE_M, OBSERVATION_STRIDE,
  PARAM_STRIDE, solveCpu, solveGpuCase,
} from './mechanics.ts'

export function withinStance({ stancePct }: GaitFrame): boolean {
  return stancePct >= 5 && stancePct <= 95
}

/** Align nonempty calibration labels with the solver's prescribed-movement frames. */
export function prepareCalibrationBenchmark(trials: readonly GaitTrial[]): { frames: Float32Array; observations: Float32Array } {
  const observations: number[] = []
  const selected = trials.map((trial) => ({
    ...trial,
    frames: trial.frames.filter((frame) => {
      if (!withinStance(frame)) return false
      const medial = frame.medialBw ?? null
      const lateral = frame.lateralBw ?? null
      const total = medial == null || lateral == null ? frame.totalBw ?? null : null
      if (medial == null && lateral == null && total == null) return false
      const mask = Number(medial != null) + 2 * Number(lateral != null) + 4 * Number(total != null)
      observations.push(medial ?? 0, lateral ?? 0, total ?? 0, mask)
      return true
    }),
  }))
  if (!observations.length) throw new Error('Calibration requires measured implant force in the central 90% of stance.')
  return { frames: encodeFrames(selected), observations: Float32Array.from(observations) }
}

export function solveCalibrationCpu(
  frames: Float32Array, observations: Float32Array, params: Float32Array,
  results: Float32Array = new Float32Array(params.length / PARAM_STRIDE),
): Float32Array {
  if (observations.length !== frames.length / FRAME_STRIDE * OBSERVATION_STRIDE) {
    throw new RangeError('Calibration labels must align to every gait frame.')
  }
  return solveCpu(frames, params, results, observations)
}

// One workgroup owns one scenario. Lanes independently solve the recorded
// frames and reduce scores in workgroup memory. Only one number per scenario
// crosses back to the CPU, where candidate ordering takes place.
export const calibrationShader = `
struct Counts {
  frameCount: u32,
  scenarioCount: u32,
}

@group(0) @binding(0) var<storage, read> frames: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;
@group(0) @binding(3) var<uniform> counts: Counts;
@group(0) @binding(4) var<storage, read> observations: array<vec4<f32>>;
var<workgroup> partialLoss: array<f32, 128>;

${solveGpuCase}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  let scenario = group.x + group.y * grid.x;
  if (scenario >= counts.scenarioCount) {
    return;
  }

  var score = 0.0;
  for (var frame = lane; frame < counts.frameCount; frame += 128u) {
    let measured = observations[frame];
    let labelMask = u32(measured.w);
    if (labelMask == 0u) {
      continue;
    }
    let solution = solve_case(scenario, frame);
    let medialError = solution.x - measured.x;
    let lateralError = solution.y - measured.y;
    let totalError = solution.x + solution.y - measured.z;
    score += select(0.0, medialError * medialError, (labelMask & 1u) != 0u);
    score += select(0.0, lateralError * lateralError, (labelMask & 2u) != 0u);
    score += select(0.0, totalError * totalError, (labelMask & 4u) != 0u);
    let excess = max(0.0, abs(solution.z) - ${MOMENT_RESIDUAL_TOLERANCE_M})
      + max(0.0, abs(solution.w) - ${MOMENT_RESIDUAL_TOLERANCE_M});
    let equivalentForce = excess / ${CALIBRATION_RESIDUAL_LEVER_M};
    score += equivalentForce * equivalentForce;
  }

  partialLoss[lane] = score;
  workgroupBarrier();
  for (var offset = 64u; offset > 0u; offset /= 2u) {
    if (lane < offset) {
      partialLoss[lane] += partialLoss[lane + offset];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    results[scenario] = partialLoss[0];
  }
}
`
