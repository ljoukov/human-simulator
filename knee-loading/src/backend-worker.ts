import { solveCpu } from './mechanics.ts'
import type { WorkerRequest, WorkerResponse } from './backend.ts'

const port = globalThis as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void
}

let frames: Float32Array | undefined
let observations: Float32Array | undefined
port.onmessage = ({ data }) => {
  try {
    if (data.kind === 'frames') {
      frames = data.frames
      observations = data.observations
      port.postMessage({ id: data.id })
      return
    }
    if (!frames) throw new Error('The CPU worker has no gait frames.')
    const values = solveCpu(frames, data.params, data.output, observations)
    if (data.output) port.postMessage({ id: data.id })
    else port.postMessage({ id: data.id, values }, [values.buffer as ArrayBuffer])
  } catch (error) {
    port.postMessage({ id: data.id, error: String(error) })
  }
}
