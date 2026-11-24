import assert from 'node:assert/strict'
import { test } from 'node:test'
import { DEFAULT_PARAMETERS, REFERENCE_FLEXION_RAD, solveCpu } from './mechanics.ts'

function nominalWithoutOffsetsOrCoContraction(): Float32Array {
  const parameters = DEFAULT_PARAMETERS.slice()
  parameters[4] = 0
  parameters[7] = 0
  return parameters
}

function checkClose(actual: number, expected: number): void {
  assert.ok(Math.abs(actual - expected) < 1e-6, `Expected ${expected}, got ${actual}`)
}

test('contact shifts medially with adduction and retains an unmet moment when unilateral', () => {
  const angle = REFERENCE_FLEXION_RAD
  const frames = Float32Array.of(1, 0, 0, angle, 1, 0.0115, 0, angle, 1, 0.1, 0, angle)
  const output = solveCpu(frames, nominalWithoutOffsetsOrCoContraction())
  assert.deepEqual(Array.from(output.slice(0, 4)), [0.5, 0.5, 0, 0])
  checkClose(output[4], 0.75)
  checkClose(output[5], 0.25)
  checkClose(output[7], 0)
  assert.equal(output[8], 1)
  assert.equal(output[9], 0)
  checkClose(output[11], 0.077)
})

test('sagittal moment uses the opposing muscle group and reports capacity limits', () => {
  const angle = REFERENCE_FLEXION_RAD
  const frames = Float32Array.of(1, 0, 0.045, angle, 1, 0, -0.037, angle, 1, 0, 1, angle)
  const output = solveCpu(frames, nominalWithoutOffsetsOrCoContraction())
  checkClose(output[0] + output[1], 1.95)
  checkClose(output[4] + output[5], 1.90)
  checkClose(output[2], 0)
  checkClose(output[6], 0)
  checkClose(output[10], 0.7975)
})

test('calibration scores only observed fields and penalizes residuals on observed frames', () => {
  const angle = REFERENCE_FLEXION_RAD
  const frames = Float32Array.of(1, 0, 0, angle, 1, 0.1, 0, angle, 1, 0, 1, angle)
  const observations = Float32Array.of(0.4, NaN, 1.2, 5, 0.6, NaN, NaN, 1, NaN, NaN, NaN, 0)
  const output = solveCpu(frames, nominalWithoutOffsetsOrCoContraction(), undefined, observations)
  const expected = (0.5 - 0.4) ** 2 + (1 - 1.2) ** 2 + (1 - 0.6) ** 2 + ((0.077 - 0.005) / 0.05) ** 2
  assert.equal(output.length, 1)
  checkClose(output[0], expected)
})
