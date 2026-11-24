import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PROFILES, sampleFrame, simulateGait } from './dynamics.js';

const delta = (run, key, fn = Math.max) => fn(...run.altered.map((f, i) => f[key] - run.baseline[i][key]));
const TASK_POLICIES = ['walk', 'stance'].flatMap((task) =>
  ['effort', 'reserve'].map((strategy) => ({ task, strategy }))
);

// Evaluate the nonlinear two-link ankle location independently of the solver's
// linearized kinematic Jacobian, with displacement measured from the same hip.
function ankleMm(frame, left, heightCm) {
  const rad = Math.PI / 180;
  const hip = rad * (left ? frame.leftHipFlexionDeg : frame.hipFlexionDeg);
  const knee = rad * (left ? frame.leftKneeFlexionDeg : frame.kneeFlexionDeg);
  const frontal = rad * (frame.pelvisDropDeg + (left ? frame.leftHipAdductionDeg : -frame.hipAdductionDeg));
  const thigh = heightCm * 10 * .245, shank = heightCm * 10 * .246;
  const reach = thigh * Math.cos(hip) + shank * Math.cos(hip - knee);
  return [thigh * Math.sin(hip) + shank * Math.sin(hip - knee), reach * Math.cos(frontal), reach * Math.sin(frontal)];
}

function maxLoadedAnkleExcursion(run, i) {
  const ref = run.baseline[i], changed = run.altered[i];
  const leftBw = run.settings.task === 'walk' ? run.baseline[(i + 50) % 100].groundForceBw : 1 - ref.groundForceBw;
  let maxMm = 0;
  for (const [left, weight] of [[true, leftBw], [false, ref.groundForceBw]]) {
    if (weight < .25) continue;
    const p = ankleMm(ref, left, run.profile.heightCm), q = ankleMm(changed, left, run.profile.heightCm);
    maxMm = Math.max(maxMm, Math.hypot(...q.map((v, j) => v - p[j])));
  }
  return maxMm;
}

test('neutral intervention reproduces the target and internal readouts for every virtual build and strategy', () => {
  for (const profile of PROFILES) {
    for (const choice of TASK_POLICIES) {
      const result = simulateGait({ profileId: profile.id, ...choice, gluteCapacityPct: 100, bandShorteningMm: 0 });
      assert.deepEqual(result.baseline, result.altered);
      assert.deepEqual(sampleFrame(result.altered, 0), result.altered[0]);
      assert.deepEqual(sampleFrame(result.altered, 1), result.altered[100]);
      assert.equal(sampleFrame(result.altered, .535).phase, .535);
    }
  }
});

test('weakness changes lean, pelvic and contralateral motion; cable shortening increases tension and changes hip flexion', () => {
  const weak = simulateGait({ gluteCapacityPct: 55, bandShorteningMm: 0 });
  assert.ok(delta(weak, 'trunkLeanDeg') > 2);
  assert.ok(delta(weak, 'pelvisDropDeg') > .3);
  assert.ok(delta(weak, 'leftHipAdductionDeg', Math.min) < -.1);
  const band = simulateGait({ gluteCapacityPct: 100, bandShorteningMm: 6 });
  assert.ok(delta(band, 'bandForceN') > 50);
  assert.ok(delta(band, 'hipFlexionDeg', Math.min) < -.5);
});

test('an available-strength strategy preserves mild-impairment motion while reducing strength margin', () => {
  const settings = { gluteCapacityPct: 75, bandShorteningMm: 0 };
  const effort = simulateGait(settings);
  const reserve = simulateGait({ ...settings, strategy: 'reserve' });
  assert.ok(delta(effort, 'trunkLeanDeg') > 1);
  for (let i = 0; i < reserve.altered.length; i++) {
    const { abductorUtilizationPct: nominalUse, ...a } = reserve.baseline[i];
    const { abductorUtilizationPct: changedUse, ...b } = reserve.altered[i];
    assert.deepEqual(a, b);
    assert.ok(Math.abs(changedUse * .75 - nominalUse) < 1e-10);
  }
  assert.ok(Math.max(...reserve.altered.map((f) => f.abductorUtilizationPct)) > 99);
});

test('108 profile, strategy, task and impairment combinations converge at doubled integration resolution', () => {
  const worst = { angleDeg: 0, bandN: 0, abductorN: 0, externalMomentNm: 0, seamDeg: 0, loadedAnkleMm: 0 };
  for (const { id: profileId } of PROFILES) {
    for (const choice of TASK_POLICIES) {
      for (const gluteCapacityPct of [40, 65, 100]) {
        for (const bandShorteningMm of [0, 3, 8]) {
          const inputs = { profileId, ...choice, gluteCapacityPct, bandShorteningMm };
          const run = simulateGait(inputs), refined = simulateGait(inputs, { steps: 1600 });
          for (let i = 0; i < run.altered.length; i++) {
            const a = run.altered[i], b = refined.altered[i];
            assert.ok(a.bandForceN >= 0 && a.abductorForceN >= 0);
            assert.ok(a.abductorUtilizationPct >= 0 && a.abductorUtilizationPct <= 100 + 1e-10);
            for (const key of Object.keys(a)) {
              assert.ok(Number.isFinite(a[key]));
              const err = Math.abs(a[key] - b[key]);
              if (key.endsWith('Deg')) {
                worst.angleDeg = Math.max(worst.angleDeg, err);
                worst.seamDeg = Math.max(worst.seamDeg, Math.abs(run.altered[0][key] - run.altered.at(-1)[key]));
              }
            }
            worst.bandN = Math.max(worst.bandN, Math.abs(a.bandForceN - b.bandForceN));
            worst.abductorN = Math.max(worst.abductorN, Math.abs(a.abductorForceN - b.abductorForceN));
            worst.externalMomentNm = Math.max(worst.externalMomentNm, Math.abs(a.hipMomentNm - b.hipMomentNm));
            worst.loadedAnkleMm = Math.max(worst.loadedAnkleMm, maxLoadedAnkleExcursion(run, i));
          }
        }
      }
    }
  }
  console.info('Maximum 800 vs 1600 step differences, periodic closure and loaded-foot excursion', worst);
  assert.ok(worst.angleDeg < .08 && worst.seamDeg < .005);
  assert.ok(worst.bandN < .4 && worst.abductorN < 5 && worst.externalMomentNm < .3);
  assert.ok(worst.loadedAnkleMm < 10);
});
