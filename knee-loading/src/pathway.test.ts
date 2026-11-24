import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { GaitTrial } from './dataset.ts';
import { earlyStancePeaks, sampleGaitPathway } from './pathway.ts';

const trial: GaitTrial = {
  id: 'test-trial',
  subject: 'test-subject',
  condition: 'normal',
  massKg: 70,
  frames: [
    { stancePct: 0, grfBw: 0.2, kamNmKg: 0, kfmNmKg: -0.2, medialBw: null },
    { stancePct: 20, grfBw: 0.8, kamNmKg: 0.24, kfmNmKg: 0.6, medialBw: 1 },
    { stancePct: 40, grfBw: 1.2, kamNmKg: 0.16, kfmNmKg: 0.1, medialBw: 2 },
    { stancePct: 50, grfBw: 1.5, kamNmKg: 0.2, kfmNmKg: -0.1, medialBw: 2.4 },
    { stancePct: 100, grfBw: 0.2, kamNmKg: -0.1, kfmNmKg: 0.05, medialBw: 0.8 },
  ],
};

test('stride inspection maps recorded stance and preserves unavailable loads', () => {
  assert.equal(sampleGaitPathway(trial, 60, 61), null);
  assert.equal(sampleGaitPathway(trial, null, 25), null);
  assert.equal(sampleGaitPathway(trial, 60, 60)?.medialBw, 0.8);
  assert.equal(sampleGaitPathway(trial, 60, 6)?.medialBw, null);
  assert.equal(sampleGaitPathway(trial, 60, 12)?.medialBw, 1);
  const fractional = sampleGaitPathway(trial, 60, 18.3);
  assert.ok(fractional);
  assert.ok(Math.abs(fractional.stancePct - 30.5) < 1e-9);
  assert.ok(Math.abs(fractional.medialBw! - 1.525) < 1e-9);
  assert.equal(fractional.lateralBw, null);
});

test('early peaks respect the 5–45% window and map into the clip timeline', () => {
  const peaks = earlyStancePeaks(trial, 60);
  assert.ok(peaks);
  assert.deepEqual(peaks.kam, { value: 0.24, stancePct: 20, stridePct: 12 });
  assert.deepEqual(peaks.medial, { value: 2.2, stancePct: 45, stridePct: 27 });
  assert.equal(peaks.lateral, null);
});
