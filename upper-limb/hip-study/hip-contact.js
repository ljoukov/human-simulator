// The C solver supplies one hip-local position per physical ITB vertex.
// Atlas vertices duplicated at UV seams share that same solved position.
const configUrl = new URL('../../hip-study/data/hip-contact.json', import.meta.url);
const modelUrl = new URL('../../hip-study/data/hip-contact.bin', import.meta.url);
const wasmUrl = new URL('./assets/hip-contact.wasm', import.meta.url);
const bindings = new WeakMap();
let solverPromise;

function vertexKey(x, y, z) {
  return [x, y, z].map(m => Math.round(m * 1e6)).join(',');
}

async function loadSolver() {
  const responses = await Promise.all([configUrl, modelUrl, wasmUrl].map(url => fetch(url)));
  if (responses.some(response => !response.ok)) throw new Error('Hip contact solver assets unavailable.');
  const [config, model, wasm] = await Promise.all([
    responses[0].json(), responses[1].arrayBuffer(), responses[2].arrayBuffer(),
  ]);
  const {instance} = await WebAssembly.instantiate(wasm, {env:{sin:Math.sin, cos:Math.cos}});
  const core = instance.exports;
  new Uint8Array(core.memory.buffer, core.contact_input(), model.byteLength).set(new Uint8Array(model));
  if (core.contact_initialize(model.byteLength)) throw new Error('Invalid hip contact model.');

  // Four header counts precede the (x, y, z, attachment weight) records.
  // Band vertices are the first surface in the binary.
  const [vertexCount, bandCount] = new Uint32Array(model, 0, 2);
  const rest = new Float32Array(model, 16, vertexCount * 4);
  const [cx, cy, cz] = config.centerM;
  const vertexIds = new Map();
  for (let i = 0; i < bandCount; i++) {
    const k = 4 * i;
    vertexIds.set(vertexKey(rest[k] + cx, rest[k+1] + cy, rest[k+2] + cz), i);
  }
  const output = new Float64Array(core.memory.buffer, core.contact_output(), bandCount * 3);

  function solvePositions(q) {
    const length = Math.hypot(q.x, q.y, q.z);
    const scale = length ? (q.w < 0 ? -1 : 1) / length : 0;
    const angle = 2 * Math.atan2(length, Math.abs(q.w));
    const status = core.contact_solve(q.x * scale, length ? q.y * scale : 1, q.z * scale, angle);
    if (status) throw new Error('Hip contact solver returned '+status);
    // Cache entries must outlive the next write to the shared WASM buffer.
    return output.slice();
  }
  return {vertexIds, solvePositions};
}

export async function prepareHipContact(gltf) {
  solverPromise ??= loadSolver().catch(error => { solverPromise = undefined; throw error; });
  const {vertexIds, solvePositions} = await solverPromise;
  if (bindings.has(gltf)) return bindings.get(gltf);
  let band;
  gltf.scene.traverse(object => {
    if (!object.isMesh || (object.userData.label ?? object.parent?.userData.label) !== 'Iliotibial tract') return;
    if (band) throw new Error('Expected one iliotibial tract surface.');
    band = object.geometry.getAttribute('position');
  });
  if (!band) throw new Error('Iliotibial tract missing from the atlas.');
  const packedToWelded = new Uint32Array(band.count);
  for (let i = 0; i < band.count; i++) {
    const vertex = vertexIds.get(vertexKey(band.getX(i), band.getY(i), band.getZ(i)));
    if (vertex === undefined) throw new Error('Atlas and contact model coordinates differ.');
    packedToWelded[i] = vertex;
  }

  // Four recent exact rotations cover the shared motion panes and the paired
  // comparison without interpolating contact states between poses.
  const recent = [];
  function solve(q) {
    const hit = recent.find(entry => entry.pose.equals(q));
    if (hit) return hit.result;
    const result = {positions:solvePositions(q), packedToWelded};
    recent.unshift({pose:q.clone(), result});
    if (recent.length > 4) recent.pop();
    return result;
  }
  const contact = {solve};
  bindings.set(gltf, contact);
  return contact;
}
