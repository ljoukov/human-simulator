import { FRAME_STRIDE, OBSERVATION_STRIDE, OUTPUT_STRIDE, PARAM_STRIDE, solveCpu, solveGpuShader } from './mechanics.ts'
import { calibrationShader } from './backend-calibration.ts'

const GROUP_SIZE = 128
const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT

export type WorkerRequest =
  | { id: number; kind: 'frames'; frames: Float32Array; observations?: Float32Array }
  | { id: number; kind: 'solve'; params: Float32Array; output?: Float32Array }
export type WorkerResponse = { id: number; values?: Float32Array; error?: string }

type WorkerJob = Omit<Extract<WorkerRequest, { kind: 'frames' }>, 'id'>
  | Omit<Extract<WorkerRequest, { kind: 'solve' }>, 'id'>

export interface CpuBackend {
  workers: number
  sharedOutput: boolean
  solve(params: Float32Array): Promise<Float32Array>
  close(): void
}

export interface GpuBackend {
  adapter: string
  fallback: boolean
  solve(params: Float32Array): Promise<Float32Array>
  close(): void
}

export interface BenchmarkOptions {
  batches?: number[]
  repeats?: number
  workers?: number
  absTolerance?: number
  relTolerance?: number
  observations?: Float32Array
  onProgress?: (stage: string) => void
}

export interface BenchmarkRow {
  scenarios: number
  frames: number
  cpuMode: 'single' | 'workers'
  cpuMs: number
  cpuSingleMs: number
  cpuWorkersMs: number | null
  gpuMs: number | null
  speedup: number | null
  maxAbsError: number | null
  mismatchCount: number | null
  topCandidateAgreement?: { size: number; shared: number }
  gpuError?: string
}

export interface BenchmarkResult {
  cpu: { workers: number; setupMs: number; sharedOutput?: boolean; parallelUnavailable?: string }
  gpu: { adapter: string; fallback: boolean; setupMs: number } | null
  gpuUnavailable?: string
  repeats: number
  workload?: 'waveforms' | 'calibration'
  absTolerance: number
  relTolerance: number
  rows: BenchmarkRow[]
}

function description(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function countRows(values: Float32Array, stride: number, label: string): number {
  if (!values.length || values.length % stride) {
    throw new RangeError(`${label} must contain one or more complete rows of ${stride} values.`)
  }
  return values.length / stride
}

function gpuInput(values: Float32Array): Float32Array<ArrayBuffer> {
  return values.buffer instanceof ArrayBuffer
    ? new Float32Array(values.buffer, values.byteOffset, values.length)
    : values.slice()
}

function outputSize(frameCount: number, observations?: Float32Array): number {
  if (!observations) return frameCount * OUTPUT_STRIDE
  if (observations.length !== frameCount * OBSERVATION_STRIDE) throw new RangeError('Calibration labels must align to every gait frame.')
  return 1
}

class CpuWorker {
  private readonly worker = new Worker(new URL('./backend-worker.ts', import.meta.url), { type: 'module' })
  private readonly pending = new Map<number, {
    resolve: (value: WorkerResponse) => void
    reject: (error: Error) => void
  }>()
  private nextId = 0
  private failure?: Error

  constructor() {
    this.worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      const job = this.pending.get(data.id)
      if (!job) return
      this.pending.delete(data.id)
      if (data.error) job.reject(new Error(data.error))
      else job.resolve(data)
    }
    this.worker.onerror = (event) => {
      event.preventDefault()
      this.fail(new Error(event.message || 'A CPU benchmark worker failed to start.'))
    }
    this.worker.onmessageerror = () => {
      this.fail(new Error('A CPU benchmark worker returned an unreadable message.'))
    }
  }

  private fail(error: Error): void {
    this.failure = error
    for (const job of this.pending.values()) job.reject(error)
    this.pending.clear()
  }

  request(job: WorkerJob): Promise<WorkerResponse> {
    return new Promise((resolve, reject) => {
      if (this.failure) {
        reject(this.failure)
        return
      }
      const id = this.nextId++
      this.pending.set(id, { resolve, reject })
      try {
        this.worker.postMessage({ ...job, id })
      } catch (error) {
        this.pending.delete(id)
        reject(error)
      }
    })
  }

  close(): void {
    this.worker.terminate()
    this.fail(new Error('The CPU worker has been closed.'))
  }
}

export async function createCpuBackend(
  frames: Float32Array,
  workers = Math.max(1, globalThis.navigator?.hardwareConcurrency ?? 1),
  observations?: Float32Array,
): Promise<CpuBackend> {
  const frameCount = countRows(frames, FRAME_STRIDE, 'Gait frames')
  const valuesPerScenario = outputSize(frameCount, observations)
  if (!Number.isSafeInteger(workers) || workers < 1) throw new RangeError('Worker count must be a positive integer.')
  if (workers === 1) {
    return { workers, sharedOutput: false, solve: async (params) => solveCpu(frames, params, undefined, observations), close() {} }
  }
  if (typeof Worker === 'undefined') throw new Error('This runtime does not provide module workers.')

  const pool: CpuWorker[] = []
  try {
    for (let i = 0; i < workers; i++) pool.push(new CpuWorker())
    await Promise.all(pool.map((worker) => worker.request({ kind: 'frames', frames, observations })))
  } catch (error) {
    pool.forEach((worker) => worker.close())
    throw error
  }

  const sharedOutput = typeof SharedArrayBuffer !== 'undefined'
  return {
    workers,
    sharedOutput,
    async solve(params) {
      const scenarios = countRows(params, PARAM_STRIDE, 'Parameter sets')
      const active = Math.min(scenarios, workers)
      const outputLength = scenarios * valuesPerScenario
      const output = sharedOutput
        ? new Float32Array(new SharedArrayBuffer(outputLength * FLOAT_BYTES))
        : new Float32Array(outputLength)
      await Promise.all(pool.slice(0, active).map(async (worker, index) => {
        const start = Math.floor(index * scenarios / active)
        const end = Math.floor((index + 1) * scenarios / active)
        // A subarray in a structured clone would copy the original input buffer
        // to every worker. Slice here so each worker receives only its own share.
        const result = await worker.request({
          kind: 'solve', params: params.slice(start * PARAM_STRIDE, end * PARAM_STRIDE),
          output: sharedOutput ? output.subarray(start * valuesPerScenario, end * valuesPerScenario) : undefined,
        })
        if (sharedOutput) return
        if (!result.values || result.values.length !== (end - start) * valuesPerScenario) {
          throw new Error('A CPU worker produced a result with the wrong size.')
        }
        output.set(result.values, start * valuesPerScenario)
      }))
      return output
    },
    close() { pool.forEach((worker) => worker.close()) },
  }
}

export async function createGpuBackend(frames: Float32Array, observations?: Float32Array): Promise<GpuBackend> {
  const frameCount = countRows(frames, FRAME_STRIDE, 'Gait frames')
  const valuesPerScenario = outputSize(frameCount, observations)
  const gpu = globalThis.navigator?.gpu
  if (!gpu) throw new Error('WebGPU is unavailable. Use a WebGPU-capable browser on localhost or HTTPS.')
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('WebGPU could not find a usable adapter.')
  if (adapter.info.isFallbackAdapter) throw new Error('WebGPU exposed a software fallback adapter instead of a hardware GPU.')

  const info = adapter.info
  const label = info.description || [info.vendor, info.architecture].filter(Boolean).join(' ') || 'WebGPU hardware adapter'
  const device = await adapter.requestDevice()
  const limits = device.limits
  const resultBytesPerScenario = valuesPerScenario * FLOAT_BYTES
  const maxStorageBytes = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)
  const maxAxis = limits.maxComputeWorkgroupsPerDimension
  const maxDispatchScenarios = observations ? maxAxis * maxAxis : maxAxis * maxAxis * GROUP_SIZE / frameCount
  const scenariosPerDispatch = Math.floor(Math.min(
    maxStorageBytes / (PARAM_STRIDE * FLOAT_BYTES),
    maxStorageBytes / resultBytesPerScenario,
    maxDispatchScenarios,
    0xffffffff / valuesPerScenario,
  ))
  if (frames.byteLength > maxStorageBytes || scenariosPerDispatch < 1) {
    device.destroy()
    throw new RangeError('The selected gait frames exceed this WebGPU adapter’s storage limits.')
  }

  let pipeline: GPUComputePipeline
  let frameBuffer: GPUBuffer
  let observationBuffer: GPUBuffer | undefined
  try {
    const module = device.createShaderModule({ code: observations ? calibrationShader : solveGpuShader, label: 'Knee contact equilibrium' })
    pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
    frameBuffer = device.createBuffer({ size: frames.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(frameBuffer, 0, gpuInput(frames))
    if (observations) {
      observationBuffer = device.createBuffer({ size: observations.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
      device.queue.writeBuffer(observationBuffer, 0, gpuInput(observations))
    }
    await device.queue.onSubmittedWorkDone()
  } catch (error) {
    device.destroy()
    throw error
  }

  async function dispatch(params: Float32Array, scenarios: number, output?: Float32Array, outputOffset = 0): Promise<Float32Array> {
    const resultBytes = resultBytesPerScenario * scenarios
    const buffers: GPUBuffer[] = []
    const makeBuffer = (size: number, usage: GPUBufferUsageFlags) => {
      const buffer = device.createBuffer({ size, usage })
      buffers.push(buffer)
      return buffer
    }
    try {
      const parameterBuffer = makeBuffer(params.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
      const resultBuffer = makeBuffer(resultBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
      const readback = makeBuffer(resultBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
      const counts = makeBuffer(8, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      device.queue.writeBuffer(parameterBuffer, 0, gpuInput(params))
      device.queue.writeBuffer(counts, 0, new Uint32Array([frameCount, scenarios]))
      const bindings = [frameBuffer, parameterBuffer, resultBuffer, counts]
      if (observationBuffer) bindings.push(observationBuffer)
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
      })
      const encoder = device.createCommandEncoder()
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      const groups = observations ? scenarios : Math.ceil(scenarios * frameCount / GROUP_SIZE)
      const columns = Math.min(groups, maxAxis)
      pass.dispatchWorkgroups(columns, Math.ceil(groups / columns))
      pass.end()
      encoder.copyBufferToBuffer(resultBuffer, 0, readback, 0, resultBytes)
      device.queue.submit([encoder.finish()])
      await readback.mapAsync(GPUMapMode.READ)
      const mapped = new Float32Array(readback.getMappedRange())
      if (!output) return mapped.slice()
      output.set(mapped, outputOffset)
      return output
    } finally {
      buffers.forEach((buffer) => buffer.destroy())
    }
  }

  return {
    adapter: label,
    fallback: false,
    async solve(params) {
      const scenarios = countRows(params, PARAM_STRIDE, 'Parameter sets')
      if (scenarios <= scenariosPerDispatch) return dispatch(params, scenarios)
      const output = new Float32Array(scenarios * valuesPerScenario)
      for (let start = 0; start < scenarios; start += scenariosPerDispatch) {
        const end = Math.min(scenarios, start + scenariosPerDispatch)
        await dispatch(params.subarray(start * PARAM_STRIDE, end * PARAM_STRIDE), end - start, output, start * valuesPerScenario)
      }
      return output
    },
    close() { observationBuffer?.destroy(); frameBuffer.destroy(); device.destroy() },
  }
}

async function medianRun(
  repeats: number,
  solve: () => Float32Array | Promise<Float32Array>,
): Promise<{ ms: number; values: Float32Array }> {
  const times: number[] = []
  let values: Float32Array = new Float32Array(0)
  for (let i = 0; i < repeats; i++) {
    const start = performance.now()
    values = await solve()
    times.push(performance.now() - start)
  }
  times.sort((a, b) => a - b)
  const middle = Math.floor(times.length / 2)
  const ms = times.length % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2
  return { ms, values }
}

function compare(cpu: Float32Array, gpu: Float32Array, absTolerance: number, relTolerance: number) {
  if (cpu.length !== gpu.length) throw new Error('The CPU and GPU returned different result sizes.')
  let maxAbsError = 0
  let mismatchCount = 0
  for (let i = 0; i < cpu.length; i++) {
    const difference = Math.abs(cpu[i] - gpu[i])
    if (!Number.isFinite(difference)) {
      mismatchCount++
      continue
    }
    maxAbsError = Math.max(maxAbsError, difference)
    if (difference > absTolerance + relTolerance * Math.max(Math.abs(cpu[i]), Math.abs(gpu[i]))) mismatchCount++
  }
  return { maxAbsError, mismatchCount }
}

function topCandidateAgreement(cpu: Float32Array, gpu: Float32Array): { size: number; shared: number } {
  const size = Math.min(256, cpu.length)
  const top = (scores: Float32Array) => Array.from(scores.keys())
    .sort((left, right) => scores[left] - scores[right] || left - right)
    .slice(0, size)
  const cpuTop = new Set(top(cpu))
  return { size, shared: top(gpu).filter((index) => cpuTop.has(index)).length }
}

export async function runBenchmark(frames: Float32Array, params: Float32Array, options: BenchmarkOptions = {}): Promise<BenchmarkResult> {
  const frameCount = countRows(frames, FRAME_STRIDE, 'Gait frames')
  const scenarioCount = countRows(params, PARAM_STRIDE, 'Parameter sets')
  const largestDefault = Math.min(scenarioCount, 8192)
  const defaults = [128, 1024].filter((size) => size < largestDefault)
  const batches = options.batches ?? [...defaults, largestDefault]
  const repeats = options.repeats ?? 3
  const absTolerance = options.absTolerance ?? 2e-5
  const relTolerance = options.relTolerance ?? 1e-5
  const observations = options.observations
  outputSize(frameCount, observations)
  if (!batches.length || batches.some((size) => !Number.isSafeInteger(size) || size < 1 || size > scenarioCount)) {
    throw new RangeError('Each benchmark batch must be a positive integer within the number of available parameter sets.')
  }
  if (!Number.isSafeInteger(repeats) || repeats < 1) throw new RangeError('Benchmark repeats must be a positive integer.')
  if (absTolerance < 0 || relTolerance < 0 || !Number.isFinite(absTolerance + relTolerance)) {
    throw new RangeError('Benchmark tolerances must be finite and nonnegative.')
  }

  const result: BenchmarkResult = {
    cpu: { workers: 1, setupMs: 0 }, gpu: null, rows: [], repeats, absTolerance, relTolerance,
    workload: observations ? 'calibration' : 'waveforms',
  }
  let cpu: CpuBackend | undefined
  let gpu: GpuBackend | undefined
  const warmup = params.subarray(0, Math.min(128, scenarioCount) * PARAM_STRIDE)
  try {
    options.onProgress?.('Preparing CPU workers')
    const cpuStart = performance.now()
    try {
      cpu = await createCpuBackend(frames, options.workers, observations)
      result.cpu = { workers: cpu.workers, setupMs: performance.now() - cpuStart, sharedOutput: cpu.sharedOutput }
      await cpu.solve(warmup)
    } catch (error) {
      result.cpu.parallelUnavailable = description(error)
      cpu?.close()
      cpu = undefined
    }
    solveCpu(frames, warmup, undefined, observations)

    options.onProgress?.('Preparing WebGPU')
    const gpuStart = performance.now()
    try {
      gpu = await createGpuBackend(frames, observations)
      result.gpu = { adapter: gpu.adapter, fallback: gpu.fallback, setupMs: performance.now() - gpuStart }
      await gpu.solve(warmup)
    } catch (error) {
      result.gpuUnavailable = description(error)
      result.gpu = null
      gpu?.close()
      gpu = undefined
    }

    const cpuBackend = cpu
    const gpuBackend = gpu
    for (const scenarios of batches) {
      options.onProgress?.(`Timing ${scenarios.toLocaleString()} parameter sets`)
      const batch = params.subarray(0, scenarios * PARAM_STRIDE)
      // Yield outside the timings so browser progress updates can paint.
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      const single = await medianRun(repeats, () => solveCpu(frames, batch, undefined, observations))
      const parallel = cpuBackend && cpuBackend.workers > 1 ? await medianRun(repeats, () => cpuBackend.solve(batch)) : undefined
      const useParallel = parallel && parallel.ms < single.ms
      const best = useParallel ? parallel : single
      const row: BenchmarkRow = {
        scenarios, frames: frameCount, cpuMode: useParallel ? 'workers' : 'single',
        cpuMs: best.ms, cpuSingleMs: single.ms, cpuWorkersMs: parallel?.ms ?? null,
        gpuMs: null, speedup: null, maxAbsError: null, mismatchCount: null,
      }
      if (gpuBackend) {
        try {
          const measured = await medianRun(repeats, () => gpuBackend.solve(batch))
          row.gpuMs = measured.ms
          Object.assign(row, compare(best.values, measured.values, absTolerance, relTolerance))
          if (observations && row.mismatchCount === 0) row.topCandidateAgreement = topCandidateAgreement(best.values, measured.values)
          if (row.mismatchCount === 0 && measured.ms > 0) row.speedup = best.ms / measured.ms
        } catch (error) {
          row.gpuError = description(error)
        }
      }
      result.rows.push(row)
    }
  } finally {
    gpu?.close()
    cpu?.close()
  }
  return result
}
