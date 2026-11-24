import assert from 'node:assert/strict';
import { test } from 'node:test';
import { rankCandidates } from './sampling.ts';

test('bounded candidate selection matches a full stable ranking', () => {
  for (const input of [
    [9, 1, 3, 1, 1, 2, 0, 0, 9, 1],
    Array.from({ length: 1024 }, (_, index) => 1024 - index),
    Array.from({ length: 1024 }, (_, index) => Math.imul(index, 483313) % 193),
  ]) {
    const scores = Float32Array.from(input);
    const sorted = Array.from(scores.keys()).sort((a, b) => scores[a] - scores[b] || a - b);
    for (const retain of [1, 3, 8, 256, scores.length, scores.length + 1]) {
      assert.deepEqual([...rankCandidates(scores, retain)], sorted.slice(0, retain));
    }
  }
});
