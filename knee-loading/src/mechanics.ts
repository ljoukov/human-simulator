// Frame fields: axial compression / BW, KAM / BW, KFM / BW, knee flexion (rad).
// KAM and KFM are external moments. Positive KAM shifts load medially and
// positive KFM calls for knee extensor force. Moments divided by BW have units m.
export const FRAME_STRIDE = 4

// Output fields: medial and lateral force (BW), then signed unmet sagittal
// and frontal moment (m). Output is packed scenario first, then frame.
export const OUTPUT_STRIDE = 4
// Observations: medial, lateral and total measured force (BW), then a mask
// marking which values to score (1 medial, 2 lateral, 4 total).
export const OBSERVATION_STRIDE = 4
export const REFERENCE_FLEXION_RAD = Math.PI / 12
export const MOMENT_RESIDUAL_TOLERANCE_M = 0.005
export const CALIBRATION_RESIDUAL_LEVER_M = 0.05

// Positive contact arms measure the distances from the modeled knee center.
// Offsets use a positive-medial convention. These bounds define sensitivity
// scenarios. They do not represent a measured distribution of human anatomy.
export const PARAMETERS = [
  { key: 'extensorArmM', label: 'Extensor moment arm at 15°', unit: 'm', min: 0.035, max: 0.055, nominal: 0.045 },
  { key: 'flexorArmM', label: 'Flexor moment arm at 15°', unit: 'm', min: 0.025, max: 0.050, nominal: 0.037 },
  { key: 'medialContactArmM', label: 'Medial contact arm', unit: 'm', min: 0.017, max: 0.030, nominal: 0.023 },
  { key: 'lateralContactArmM', label: 'Lateral contact arm', unit: 'm', min: 0.017, max: 0.030, nominal: 0.023 },
  { key: 'muscleMedialOffsetM', label: 'Muscle compression offset', unit: 'm', min: -0.006, max: 0.008, nominal: 0.001 },
  { key: 'axialMedialOffsetM', label: 'External compression offset', unit: 'm', min: -0.010, max: 0.010, nominal: 0 },
  { key: 'axialScale', label: 'External compression scale', unit: '', min: 0.80, max: 1.20, nominal: 1 },
  { key: 'coContractionRatio', label: 'Co-contraction ratio', unit: '', min: 0, max: 0.50, nominal: 0.15 },
  { key: 'extensorCapacityBW', label: 'Extensor force capacity', unit: 'BW', min: 2.5, max: 7, nominal: 4.5 },
  { key: 'flexorCapacityBW', label: 'Flexor force capacity', unit: 'BW', min: 0.8, max: 3, nominal: 1.8 },
  { key: 'extensorProjection', label: 'Extensor axial projection', unit: '', min: 0.85, max: 1, nominal: 0.95 },
  { key: 'flexorProjection', label: 'Flexor axial projection', unit: '', min: 0.75, max: 1, nominal: 0.90 },
] as const
export const PARAM_STRIDE = PARAMETERS.length
export const DEFAULT_PARAMETERS = Float32Array.from(PARAMETERS, (parameter) => parameter.nominal)

// Quadratic least-squares fits to group mean |moment arm| from the repo's
// Rajagopal2016 geometry, 0–60° with the hip and ankle at their defaults.
// Each curve is normalized at 15°. Max errors are 1.70 mm (extensors) and
// 0.97 mm (flexors) versus the group means. Hold the endpoints outside the fit.
const MAX_FLEXION_RAD = Math.PI / 3
const round = Math.fround
const REFERENCE_FLEXION_F32 = round(REFERENCE_FLEXION_RAD)
const MAX_FLEXION_F32 = round(MAX_FLEXION_RAD)
const EXTENSOR_CURVE_LINEAR = round(-0.145985)
const EXTENSOR_CURVE_QUADRATIC = round(-0.1977134)
const FLEXOR_CURVE_LINEAR = round(0.4933224)
const FLEXOR_CURVE_QUADRATIC = round(-0.20718245)

/**
 * Prescribed-motion, quasi-static knee contact with two aggregate muscle groups.
 * Paired antagonist torque leaves the sagittal moment unchanged. Compression is
 * the sum of external axial and projected muscle forces. A nonnegative two-point
 * contact split balances the frontal moment when both compartments can supply it.
 * A capacity or unilateral contact limit leaves a residual in the output.
 * If observations are supplied, return one calibration loss per scenario.
 */
export function solveCpu(
  frames: Float32Array,
  params: Float32Array,
  results?: Float32Array,
  observations?: Float32Array,
): Float32Array {
  if (frames.length % FRAME_STRIDE !== 0 || params.length % PARAM_STRIDE !== 0) {
    throw new RangeError('Mechanics inputs must contain complete frames and parameter sets.')
  }

  const frameCount = frames.length / FRAME_STRIDE
  const scenarioCount = params.length / PARAM_STRIDE
  if (observations && observations.length !== frameCount * OBSERVATION_STRIDE) {
    throw new RangeError('Mechanics observations must contain one complete record per frame.')
  }
  const resultCount = observations ? scenarioCount : frameCount * scenarioCount * OUTPUT_STRIDE
  results ??= new Float32Array(resultCount)
  if (results.length !== resultCount) {
    throw new RangeError('Mechanics output has the wrong length for this batch.')
  }

  // Knee pose is shared across scenarios. The CPU reuses these geometry
  // factors rather than evaluating the same polynomials in the inner loop.
  const extensorShapes = new Float32Array(frameCount)
  const flexorShapes = new Float32Array(frameCount)
  for (let index = 0; index < frameCount; index += 1) {
    const angle = round(Math.min(Math.max(frames[index * FRAME_STRIDE + 3], 0), MAX_FLEXION_F32) - REFERENCE_FLEXION_F32)
    extensorShapes[index] = round(1 + round(angle * round(EXTENSOR_CURVE_LINEAR + round(angle * EXTENSOR_CURVE_QUADRATIC))))
    flexorShapes[index] = round(1 + round(angle * round(FLEXOR_CURVE_LINEAR + round(angle * FLEXOR_CURVE_QUADRATIC))))
  }

  let output = 0
  for (let scenario = 0; scenario < params.length; scenario += PARAM_STRIDE) {
    const extensorArm = params[scenario]
    const flexorArm = params[scenario + 1]
    const medialArm = params[scenario + 2]
    const lateralArm = params[scenario + 3]
    const muscleOffset = params[scenario + 4]
    const axialOffset = params[scenario + 5]
    const axialScale = params[scenario + 6]
    const coContraction = params[scenario + 7]
    const extensorCapacity = params[scenario + 8]
    const flexorCapacity = params[scenario + 9]
    const extensorProjection = params[scenario + 10]
    const flexorProjection = params[scenario + 11]
    const contactSpan = round(medialArm + lateralArm)
    let loss = 0

    for (let index = 0; index < frameCount; index += 1) {
      const frame = index * FRAME_STRIDE
      const observation = index * OBSERVATION_STRIDE
      const mask = observations ? observations[observation + 3] : 0
      if (observations && mask === 0) continue

      const axial = round(frames[frame] * axialScale)
      const externalAdduction = frames[frame + 1]
      const externalFlexion = frames[frame + 2]
      const extensorLever = round(extensorArm * extensorShapes[index])
      const flexorLever = round(flexorArm * flexorShapes[index])
      const extensorLimit = round(extensorLever * extensorCapacity)
      const flexorLimit = round(flexorLever * flexorCapacity)
      const extensorTorque = Math.min(Math.max(externalFlexion, 0), extensorLimit)
      const flexorTorque = Math.min(Math.max(-externalFlexion, 0), flexorLimit)

      const requestedPair = round(round(coContraction * Math.max(axial, 0)) * extensorLever)
      const pair = Math.min(requestedPair, round(extensorLimit - extensorTorque), round(flexorLimit - flexorTorque))
      const extensorForce = round(round(extensorTorque + pair) / extensorLever)
      const flexorForce = round(round(flexorTorque + pair) / flexorLever)
      const muscleCompression = round(round(extensorForce * extensorProjection) + round(flexorForce * flexorProjection))
      const compression = Math.max(0, round(axial + muscleCompression))
      const frontalTarget = round(round(externalAdduction + round(axial * axialOffset)) + round(muscleCompression * muscleOffset))
      const unconstrainedMedial = round(round(frontalTarget + round(lateralArm * compression)) / contactSpan)
      const medial = Math.min(Math.max(unconstrainedMedial, 0), compression)
      const lateral = round(compression - medial)
      const sagittalResidual = round(externalFlexion - round(extensorTorque - flexorTorque))
      const frontalResidual = round(frontalTarget - round(round(medialArm * medial) - round(lateralArm * lateral)))

      if (observations) {
        if (mask & 1) loss += (medial - observations[observation]) ** 2
        if (mask & 2) loss += (lateral - observations[observation + 1]) ** 2
        if (mask & 4) loss += (medial + lateral - observations[observation + 2]) ** 2
        const excessMoment = Math.max(0, Math.abs(sagittalResidual) - MOMENT_RESIDUAL_TOLERANCE_M)
          + Math.max(0, Math.abs(frontalResidual) - MOMENT_RESIDUAL_TOLERANCE_M)
        loss += (excessMoment / CALIBRATION_RESIDUAL_LEVER_M) ** 2
      } else {
        results[output] = medial
        results[output + 1] = lateral
        results[output + 2] = sagittalResidual
        results[output + 3] = frontalResidual
        output += OUTPUT_STRIDE
      }
    }
    if (observations) results[scenario / PARAM_STRIDE] = loss
  }
  return results
}

// These shader functions share the frame and parameter arrays of their caller.
// Keep the f32 operation order consistent with solveCpu.
export const solveGpuCase = `
fn solve_case(scenarioIndex: u32, frameIndex: u32) -> vec4<f32> {
  let scenario = scenarioIndex * ${PARAM_STRIDE}u;
  let frame = frameIndex * ${FRAME_STRIDE}u;
  let medialArm = params[scenario + 2u];
  let lateralArm = params[scenario + 3u];
  let axial = frames[frame] * params[scenario + 6u];
  let externalAdduction = frames[frame + 1u];
  let externalFlexion = frames[frame + 2u];
  let angle = clamp(frames[frame + 3u], 0.0, ${MAX_FLEXION_F32}) - ${REFERENCE_FLEXION_F32};
  let extensorShape = 1.0 + angle * (${EXTENSOR_CURVE_LINEAR} + angle * ${EXTENSOR_CURVE_QUADRATIC});
  let flexorShape = 1.0 + angle * (${FLEXOR_CURVE_LINEAR} + angle * ${FLEXOR_CURVE_QUADRATIC});
  let extensorLever = params[scenario] * extensorShape;
  let flexorLever = params[scenario + 1u] * flexorShape;
  let extensorLimit = extensorLever * params[scenario + 8u];
  let flexorLimit = flexorLever * params[scenario + 9u];
  let extensorTorque = min(max(externalFlexion, 0.0), extensorLimit);
  let flexorTorque = min(max(-externalFlexion, 0.0), flexorLimit);
  let requestedPair = (params[scenario + 7u] * max(axial, 0.0)) * extensorLever;
  let pair = min(requestedPair, min(extensorLimit - extensorTorque, flexorLimit - flexorTorque));
  let extensorForce = (extensorTorque + pair) / extensorLever;
  let flexorForce = (flexorTorque + pair) / flexorLever;
  let muscleCompression = extensorForce * params[scenario + 10u] + flexorForce * params[scenario + 11u];
  let compression = max(0.0, axial + muscleCompression);
  let frontalTarget = (externalAdduction + axial * params[scenario + 5u]) + muscleCompression * params[scenario + 4u];
  let unconstrainedMedial = (frontalTarget + lateralArm * compression) / (medialArm + lateralArm);
  let medial = clamp(unconstrainedMedial, 0.0, compression);
  let lateral = compression - medial;

  let sagittalResidual = externalFlexion - (extensorTorque - flexorTorque);
  let frontalResidual = frontalTarget - (medialArm * medial - lateralArm * lateral);
  return vec4<f32>(medial, lateral, sagittalResidual, frontalResidual);
}
`

// Each invocation owns one (scenario, frame) pair.
export const solveGpuShader = `
struct Counts {
  frameCount: u32,
  scenarioCount: u32,
}

@group(0) @binding(0) var<storage, read> frames: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> counts: Counts;

${solveGpuCase}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
  let index = id.x + id.y * grid.x * 128u;
  let size = counts.frameCount * counts.scenarioCount;
  if (index >= size) {
    return;
  }

  results[index] = solve_case(index / counts.frameCount, index % counts.frameCount);
}
`
