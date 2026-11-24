import assert from 'node:assert/strict';
import { test } from 'node:test';
import { advanceStride } from './stride-playback.ts';

test('each repeat displays the exact last and first poses on consecutive animation frames', () => {
  const start = { startedAt: 100, startPhase: 0 };
  assert.equal(advanceStride(start, 600, 1000).phase, 50);
  // A late animation callback still displays the final pose.
  const finish = advanceStride(start, 1120, 1000);
  assert.equal(finish.phase, 100);
  const repeat = advanceStride(finish.playback, 1136, 1000);
  assert.equal(repeat.phase, 0);
  assert.equal(advanceStride(repeat.playback, 1386, 1000).phase, 25);
});

test('changing speed at the end of a scrubbed stride starts the next stride immediately', () => {
  const finish = advanceStride({ startedAt: 0, startPhase: 75 }, 500, 2000);
  assert.equal(finish.phase, 100);
  const repeat = advanceStride(finish.playback, 508, 500);
  assert.equal(repeat.phase, 0);
  assert.equal(advanceStride(repeat.playback, 633, 500).phase, 25);
});
