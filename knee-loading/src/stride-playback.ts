export interface StridePlayback {
  startedAt: number;
  startPhase: number;
}

export function advanceStride(playback: StridePlayback, time: number, strideMs: number) {
  if (playback.startPhase === 100) {
    return { phase: 0, playback: { startedAt: time, startPhase: 0 } };
  }
  const phase = playback.startPhase + 100 * Math.max(0, time - playback.startedAt) / strideMs;
  if (phase >= 100) return { phase: 100, playback: { startedAt: time, startPhase: 100 } };
  return { phase, playback };
}
