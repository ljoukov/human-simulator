import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { type GaitDataset } from '../src/dataset.ts';
import { runStudy } from '../src/experiment.ts';

const dataset: GaitDataset = JSON.parse(await readFile(new URL('../data/gait.json', import.meta.url), 'utf8'));
const options = { candidateCount: 64, retainCount: 16 };

test('held-out implant forces affect error reporting without affecting fitted predictions', async () => {
  const original = await runStudy([dataset], options);
  const modified = structuredClone(dataset);
  for (const trial of modified.trials) {
    if (trial.role === 'calibration') continue;
    for (const frame of trial.frames) {
      if (frame.medialBw != null) frame.medialBw += 0.2;
    }
  }
  const check = await runStudy([modified], options);

  assert.equal(original.study.comparisons.length, 3);
  assert.deepEqual(original.audit.subjects[0].calibration, ['ngait_og5-stance-1']);
  assert.equal(original.audit.subjects[0].evaluation.length, 6);
  for (let index = 0; index < original.study.comparisons.length; index++) {
    const before = original.study.comparisons[index].intervention.compartments.medial!;
    const after = check.study.comparisons[index].intervention.compartments.medial!;
    assert.deepEqual(after.modeled, before.modeled);
    assert.deepEqual(after.low, before.low);
    assert.deepEqual(after.high, before.high);
    assert.deepEqual(after.proxy, before.proxy);
    assert.notDeepEqual(after.measured, before.measured);
  }
  assert.notEqual(check.audit.subjects[0].error.medial!.rmseBw, original.audit.subjects[0].error.medial!.rmseBw);
});
