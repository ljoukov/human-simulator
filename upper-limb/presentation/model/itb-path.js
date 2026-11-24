import geometry from './itb-geometry.json' with { type: 'json' };

const { centersDeg, halfRangesDeg, hipCoefficientsM, kneeCoefficientsM } = geometry;
const degreesPerRadian = 180 / Math.PI;
export const BAND_STIFFNESS_N_PER_MM = geometry.stiffnessNPerMm;

/** Direct derivatives of the effective Davis 2026 ITB path. Generalized torque is −tension × dL/dq. */
export function bandPath(hipFlexionDeg, hipAdductionDeg, kneeFlexionDeg) {
  const rawFlex = (hipFlexionDeg - centersDeg[0]) / halfRangesDeg[0];
  const rawAdd = (hipAdductionDeg - centersDeg[1]) / halfRangesDeg[1];
  const rawKnee = (kneeFlexionDeg - centersDeg[2]) / halfRangesDeg[2];
  const flex = Math.max(-1, Math.min(1, rawFlex));
  const add = Math.max(-1, Math.min(1, rawAdd));
  const knee = Math.max(-1, Math.min(1, rawKnee));
  let hipLength = 0;
  let dFlex = 0;
  let dAdd = 0;
  // Nested Horner evaluation returns length and both hip derivatives in one pass.
  for (let i = hipCoefficientsM.length - 1; i >= 0; i--) {
    const coefficients = hipCoefficientsM[i];
    let row = 0;
    let dRow = 0;
    for (let j = coefficients.length - 1; j >= 0; j--) {
      dRow = dRow * add + row;
      row = row * add + coefficients[j];
    }
    dFlex = dFlex * flex + hipLength;
    dAdd = dAdd * flex + dRow;
    hipLength = hipLength * flex + row;
  }
  let kneeLength = 0;
  let dKnee = 0;
  for (let k = kneeCoefficientsM.length - 1; k >= 0; k--) {
    dKnee = dKnee * knee + kneeLength;
    kneeLength = kneeLength * knee + kneeCoefficientsM[k];
  }
  // Use the endpoint tangent for small knee hyperextension beyond the source model.
  // Hip coordinates saturate outside the representative walking envelope.
  return {
    lengthM: hipLength + kneeLength + (rawKnee - knee) * dKnee,
    dFlexMPerRad: (rawFlex === flex ? dFlex : 0) * degreesPerRadian / halfRangesDeg[0],
    dAddMPerRad: (rawAdd === add ? dAdd : 0) * degreesPerRadian / halfRangesDeg[1],
    dKneeMPerRad: dKnee * degreesPerRadian / halfRangesDeg[2],
  };
}
