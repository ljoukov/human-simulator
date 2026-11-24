import { createCpuBackend } from '../src/backend.ts'
import { solveCalibrationCpu } from '../src/backend-calibration.ts'
import { CALIBRATION_RESIDUAL_LEVER_M, MOMENT_RESIDUAL_TOLERANCE_M, OUTPUT_STRIDE, solveCpu } from '../src/mechanics.ts'
import { sampleParameters } from '../src/sampling.ts'

Deno.test('calibration reduction matches a reference sum over raw waveforms', async () => {
  const frames = new Float32Array([
    0.95, 0.035, 0.02, 0.2,
    1.15, -0.026, -0.03, 0.4,
    0.60, 0.072, 0.40, 0.3,
    0.40, -0.07, -0.25, 0.1,
  ])
  const observations = new Float32Array([
    0.85, 0.67, 0, 3,
    0.75, 0, 1.45, 5,
    0, 0, 1.72, 4,
    0, 0, 0, 0,
  ])
  const scenarios = 7
  const params = sampleParameters(scenarios)
  const waveform = solveCpu(frames, params)
  const expected = new Float32Array(scenarios)
  for (let scenario = 0; scenario < scenarios; scenario++) {
    const base = scenario * frames.length
    const residual = (frame: number) => {
      const offset = base + frame * OUTPUT_STRIDE
      const sagittal = Math.max(0, Math.abs(waveform[offset + 2]) - MOMENT_RESIDUAL_TOLERANCE_M)
      const frontal = Math.max(0, Math.abs(waveform[offset + 3]) - MOMENT_RESIDUAL_TOLERANCE_M)
      return ((sagittal + frontal) / CALIBRATION_RESIDUAL_LEVER_M) ** 2
    }
    expected[scenario] = (waveform[base] - observations[0]) ** 2 + (waveform[base + 1] - observations[1]) ** 2 + residual(0)
      + (waveform[base + 4] - observations[4]) ** 2 + (waveform[base + 4] + waveform[base + 5] - observations[6]) ** 2 + residual(1)
      + (waveform[base + 8] + waveform[base + 9] - observations[10]) ** 2 + residual(2)
  }

  const cpu = await createCpuBackend(frames, 3, observations)
  try {
    for (const actual of [solveCalibrationCpu(frames, observations, params), await cpu.solve(params)]) {
      if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
        throw new Error(`Expected one calibration score per scenario: ${expected}. Received ${actual}.`)
      }
    }
  } finally {
    cpu.close()
  }
})
