// Compare the displayed WebAssembly contact solution and JavaScript bursal
// readout against the native chart values at every slider tick in every clip.
// Run from any directory after installing the viewer's npm dependencies.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { Euler, MathUtils, Quaternion } from '../../upper-limb/node_modules/three/build/three.module.js';
import { createBursaMechanics } from '../../upper-limb/hip-study/bursa-mechanics.js';

const read = file => readFileSync(new URL(file, import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const study = JSON.parse(read('../data/bursa-mechanics.json'));
const { clips } = JSON.parse(read('../data/motion.json'));
const wasm = read('../../upper-limb/hip-study/assets/hip-contact.wasm');
const model = read('../data/hip-contact.bin');
assert.equal(digest(wasm), study.source.contactWasmSha256);
assert.equal(digest(model), study.source.contactInputSha256);
assert.equal(digest(read('../contact/contact.c')), study.source.contactSourceSha256);

const { instance } = await WebAssembly.instantiate(wasm, { env: { sin: Math.sin, cos: Math.cos } });
const kernel = instance.exports;
new Uint8Array(kernel.memory.buffer, kernel.contact_input(), model.byteLength).set(model);
assert.equal(kernel.contact_initialize(model.byteLength), 0);
const vertices = new Float64Array(kernel.memory.buffer, kernel.contact_output(), 3 * study.evaluator.bandVertexCount);
const { excursionMm } = createBursaMechanics(study);
const q = new Quaternion(), euler = new Euler();
function solveAt(frames, phase) {
  let right = frames.findIndex(frame => frame.phase >= phase);
  if (right < 0) right = frames.length-1;
  const a = frames[Math.max(0, right-1)], b = frames[right];
  const fraction = b.phase > a.phase ? (phase - a.phase) / (b.phase - a.phase) : 0;
  function angle(key) {
    const start = a[key] ?? 0, end = b[key] ?? 0;
    return MathUtils.degToRad(start + fraction * (end - start));
  }
  q.setFromEuler(euler.set(-angle('hipFlexionDeg'), angle('hipRotationDeg'), angle('hipAdductionDeg'), 'ZXY'));
  const length = Math.hypot(q.x, q.y, q.z);
  const factor = length ? Math.sign(q.w || 1) / length : 0;
  const status = kernel.contact_solve(q.x * factor, length ? q.y * factor : 1, q.z * factor,
    2 * Math.atan2(length, Math.abs(q.w)));
  assert.equal(status, 0, `Contact solver failed at phase ${phase}`);
  return excursionMm(q, vertices);
}

let checked = 0, maxErrorMm = 0;
for (const clip of clips) {
  const { chartExcursionMm: expected } = study.clipChecks.find(record => record.clip === clip.id);
  for (const [index, nativeMm] of expected.entries()) {
    const browserMm = solveAt(clip.frames, index / (expected.length - 1));
    maxErrorMm = Math.max(maxErrorMm, Math.abs(browserMm - nativeMm));
    checked++;
  }
}
assert.ok(maxErrorMm <= 0.0000501, `Chart and live evaluation differ by ${maxErrorMm} mm`);
console.log(`${checked} native/WASM chart values checked. Maximum error including 4-decimal rounding: ${maxErrorMm} mm.`);

// These nearly identical poses exposed 0.118 and 0.074 mm RMS jumps when
// the solver used a discontinuous vertex-only lower bound.
for (const [clipId, first, second] of [
  ['predsim-hip-75', .6783752268473308, .6783752278645834],
  ['h1l-walk', .9967920155843099, .9967920166015625],
]) {
  const { frames } = clips.find(clip => clip.id === clipId);
  const before = solveAt(frames, first), previous = vertices.slice();
  const travelChangeMm = Math.abs(solveAt(frames, second) - before);
  let vertexChangeMm = 0;
  for (let i = 0; i < previous.length; i += 3) {
    const distance = Math.hypot(previous[i]-vertices[i], previous[i+1]-vertices[i+1], previous[i+2]-vertices[i+2]);
    vertexChangeMm = Math.max(vertexChangeMm, 1000*distance);
  }
  assert.ok(travelChangeMm < 1e-5 && vertexChangeMm < 1e-4, `Contact jump at ${clipId} phase ${first}`);
  console.log(`${clipId} event. Travel change ${travelChangeMm} mm. Largest vertex shift ${vertexChangeMm} mm.`);
}
