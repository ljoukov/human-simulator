import * as THREE from 'three';
import { createContactSkin } from './hip-contact-skinning.js';

const manifestUrl = new URL('../../hip-study/data/hip-deformation.json', import.meta.url);
const weightsUrl = new URL('../../hip-study/data/hip-attachments.bin', import.meta.url);

let assets;
const neutralFieldsByAtlas = new WeakMap();
async function loadFields() {
  const [meta, blob] = await Promise.all([fetch(manifestUrl), fetch(weightsUrl)]);
  if (!meta.ok || !blob.ok) throw new Error('Hip attachment fields unavailable.');
  const manifest = await meta.json();
  const bytes = new DataView(await blob.arrayBuffer());
  if (manifest.version !== 2 || manifest.recordBytes !== 14) throw new Error('Unsupported hip attachment file.');
  const fields = new Map();
  const insertionDistances = new Map();
  for (const [label, record] of Object.entries(manifest.surfaces)) {
    if (record.constant !== undefined) {
      fields.set(label, record.constant);
      continue;
    }
    const field = new Map();
    const stride = record.recordBytes ?? manifest.recordBytes;
    const distances = record.insertionSeeds ? new Map() : null;
    for (let i = 0; i < record.vertices; i++) {
      const a = record.offsetBytes + i * stride;
      const point = [bytes.getInt32(a, true), bytes.getInt32(a + 4, true), bytes.getInt32(a + 8, true)].join(',');
      field.set(point, bytes.getUint16(a + 12, true) / 65535);
      if (distances) distances.set(point, bytes.getUint16(a + 14, true) * manifest.insertionRegion.distanceUnitMm);
    }
    fields.set(label, field);
    if (distances) insertionDistances.set(label, distances);
  }
  return { manifest, fields, insertionDistances };
}

const arcDeclarations = `
  uniform vec4 hipAxisAngle;
  uniform vec3 hipTranslation;
  attribute float hipArcWeight;
  attribute vec3 hipWeightGradient;
  vec3 hipArc(vec3 p, float weight) {
    vec3 axis = hipAxisAngle.xyz;
    float angle = weight * hipAxisAngle.w;
    float c = cos(angle);
    return c*p + sin(angle)*cross(axis,p) + (1.-c)*dot(axis,p)*axis;
  }`;

function patchMaterial(material, state, usesContact) {
  const original = material.onBeforeCompile.bind(material);
  const key = material.customProgramCacheKey.bind(material);
  const originalKey = key();
  material.customProgramCacheKey = () => originalKey + '/surface-geodesic-arc-v1' + (usesContact ? '/contact' : '');
  material.onBeforeCompile = (shader, renderer) => {
    original(shader, renderer);
    shader.uniforms.hipAxisAngle = state.axisAngle;
    shader.uniforms.hipTranslation = state.translation;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + arcDeclarations + (usesContact ? `
        attribute vec3 hipContactPosition;
        attribute vec4 hipContactRotation;
        vec3 orientContact(vec3 v) {
          return v + 2.*cross(hipContactRotation.xyz,hipContactRotation.w*v + cross(hipContactRotation.xyz,v));
        }` : ''))
      .replace('#include <skinning_vertex>', usesContact
        ? 'transformed = hipContactPosition + hipArcWeight * hipTranslation;'
        : 'transformed = hipArc(transformed, hipArcWeight) + hipArcWeight * hipTranslation;')
      .replace('#include <skinnormal_vertex>', `
        vec3 restNormal = normalize(objectNormal);
        vec3 reference = abs(restNormal.x) < 0.8 ? vec3(1.,0.,0.) : vec3(0.,1.,0.);
        vec3 tangentA = normalize(cross(restNormal, reference));
        vec3 tangentB = cross(restNormal, tangentA);
        vec3 dWeight = hipAxisAngle.w * cross(hipAxisAngle.xyz, hipArc(position,hipArcWeight)) + hipTranslation;
        vec3 movedA = hipArc(tangentA,hipArcWeight) + dot(hipWeightGradient,tangentA) * dWeight;
        vec3 movedB = hipArc(tangentB,hipArcWeight) + dot(hipWeightGradient,tangentB) * dWeight;
        vec3 areaNormal = cross(movedA,movedB);
        objectNormal = dot(areaNormal,areaNormal) > 1.e-14 ? normalize(areaNormal) : hipArc(restNormal,hipArcWeight);
        #ifdef USE_TANGENT
          objectTangent = hipArc(objectTangent,hipArcWeight) + dot(hipWeightGradient,objectTangent) * dWeight;
        #endif
      ` + (usesContact ? `
        objectNormal = orientContact(objectNormal);
        #ifdef USE_TANGENT
          objectTangent = orientContact(objectTangent);
        #endif
      ` : ''));
  };
  material.needsUpdate = true;
}

// Triangle gradient of the scalar attachment field, area averaged at shared
// neutral coordinates. Toggling UV seams leaves both displacement and normals continuous.
function gradient(geometry, weights, weld) {
  const pos = geometry.getAttribute('position');
  const idx = geometry.index?.array;
  const sums = new Float64Array(weld.count * 4);
  const a = new THREE.Vector3(), u = new THREE.Vector3(), v = new THREE.Vector3();
  const n = new THREE.Vector3(), grad = new THREE.Vector3(), aux = new THREE.Vector3();
  const count = idx ? idx.length : pos.count;
  for (let i = 0; i < count; i += 3) {
    const ia = idx ? idx[i] : i, ib = idx ? idx[i + 1] : i + 1, ic = idx ? idx[i + 2] : i + 2;
    a.fromBufferAttribute(pos, ia);
    u.fromBufferAttribute(pos, ib).sub(a);
    v.fromBufferAttribute(pos, ic).sub(a);
    n.crossVectors(u, v);
    const area = n.length();
    if (area < 1e-12) continue;
    n.multiplyScalar(1 / area);
    grad.crossVectors(v, n).multiplyScalar(weights[ib] - weights[ia])
      .addScaledVector(aux.crossVectors(n, u), weights[ic] - weights[ia]);
    for (const id of [ia, ib, ic]) {
      const k = weld.ids[id] * 4;
      sums[k] += grad.x; sums[k + 1] += grad.y; sums[k + 2] += grad.z; sums[k + 3] += area;
    }
  }
  const out = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const k = weld.ids[i] * 4, area = sums[k + 3];
    if (area > 0) for (let j = 0; j < 3; j++) out[3 * i + j] = sums[k + j] / area;
  }
  return out;
}

function neutralFields(gltf, { fields, manifest, insertionDistances }) {
  const cached = neutralFieldsByAtlas.get(gltf);
  if (cached) return cached;
  const sources = new WeakMap(), byLabel = new Map();
  const scale = manifest.unitsPerMetre;
  gltf.scene.traverse(obj => {
    if (!obj.isMesh) return;
    const label = obj.userData.label ?? obj.parent?.userData.label ?? obj.name;
    const field = fields.get(label);
    if (field === undefined) throw new Error('Missing hip motion field for ' + label);
    const pos = obj.geometry.getAttribute('position');
    const w = new Float32Array(pos.count);
    const distances = insertionDistances.get(label);
    const distanceMm = distances ? new Float32Array(pos.count) : null;
    const insertion = new Float32Array(pos.count);
    const weld = { ids: new Uint32Array(pos.count), count: 0 };
    const unique = new Map();
    for (let i = 0; i < pos.count; i++) {
      const key = [Math.round(scale * pos.getX(i)), Math.round(scale * pos.getY(i)), Math.round(scale * pos.getZ(i))].join(',');
      const value = typeof field === 'number' ? field : field.get(key);
      if (value === undefined) throw new Error('Stale hip attachment geometry for ' + label);
      w[i] = value;
      if (distances) {
        distanceMm[i] = distances.get(key);
        const { fullThroughMm, endsAtMm } = manifest.insertionRegion;
        insertion[i] = 1 - THREE.MathUtils.smoothstep(distanceMm[i], fullThroughMm, endsAtMm);
      }
      if (!unique.has(key)) unique.set(key, weld.count++);
      weld.ids[i] = unique.get(key);
    }
    const entry = { w, insertion, distanceMm,
      grad: typeof field === 'number' ? new Float32Array(pos.count*3) : gradient(obj.geometry, w, weld) };
    sources.set(obj.geometry, entry);
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push(entry);
  });
  const result = { sources, byLabel };
  neutralFieldsByAtlas.set(gltf, result);
  return result;
}

export async function prepareHipDeformation(gltf) {
  assets ??= loadFields().catch(error => { assets = undefined; throw error; });
  const data = await assets;
  const { manifest } = data;
  const { sources, byLabel } = neutralFields(gltf, data);
  const states = new Map(), depthMaterials = new Set();

  function stateFor(key) {
    if (!states.has(key)) states.set(key, {
      axisAngle: { value: new THREE.Vector4(0, 1, 0, 0) },
      translation: { value: new THREE.Vector3() }, axis: new THREE.Vector3(0, 1, 0), contactSkins:[]
    });
    return states.get(key);
  }
  function entry(label, geom) {
    const match = geom && sources.get(geom);
    if (match) return match;
    const items = (byLabel.get(label) ?? []).filter(e => !geom || e.w.length === geom.getAttribute('position').count);
    if (items.length !== 1) throw new Error('Pass the original geometry for ' + label);
    return items[0];
  }
  function attach(mesh, label, key = 'default', originalGeometry) {
    const source = entry(label, originalGeometry ?? mesh.geometry);
    const state = stateFor(key);
    mesh.geometry.setAttribute('hipArcWeight', new THREE.BufferAttribute(source.w, 1));
    mesh.geometry.setAttribute('hipWeightGradient', new THREE.BufferAttribute(source.grad, 3));
    mesh.geometry.setAttribute('hipInsertion', new THREE.BufferAttribute(source.insertion, 1));
    if (source.distanceMm) mesh.geometry.setAttribute('hipInsertionDistanceMm', new THREE.BufferAttribute(source.distanceMm, 1));
    const contactSkin = label === 'Iliotibial tract' ? createContactSkin(mesh.geometry, source.w) : null;
    if (contactSkin) state.contactSkins.push({mesh,contactSkin});
    for (const m of [mesh.material].flat()) patchMaterial(m, state, Boolean(contactSkin));
    const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    depth.side = mesh.material.side ?? THREE.DoubleSide;
    patchMaterial(depth, state, Boolean(contactSkin));
    mesh.customDepthMaterial = depth;
    depthMaterials.add(depth);
    // Three's skinned raycaster asks the mesh for the current vertex position.
    mesh.applyBoneTransform = (i, v) => {
      const w = source.w[i];
      if (contactSkin) v.fromBufferAttribute(contactSkin.position,i);
      else if (w > 0) v.applyAxisAngle(state.axis,w*state.axisAngle.value.w);
      v.addScaledVector(state.translation.value,w);
      return v;
    };
  }
  function setPose(q, t, key = 'default', contactResult) {
    const state = stateFor(key);
    const sign = q.w < 0 ? -1 : 1;
    const length = Math.hypot(q.x,q.y,q.z);
    if (length > 1e-10) state.axis.set(q.x,q.y,q.z).multiplyScalar(sign / length);
    else state.axis.set(0,1,0);
    state.axisAngle.value.set(state.axis.x,state.axis.y,state.axis.z,2*Math.atan2(length,Math.abs(q.w)));
    if (t) state.translation.value.copy(t);
    else state.translation.value.set(0,0,0);
    if (contactResult) for (const {mesh,contactSkin} of state.contactSkins) {
      const radius = contactSkin.update(contactResult,state.axis,state.axisAngle.value.w);
      // The CPU hit test must also include vertices moved beyond their neutral bounds.
      mesh.boundingSphere ??= new THREE.Sphere();
      mesh.boundingSphere.center.set(0,0,0);
      mesh.boundingSphere.radius = radius + state.translation.value.length();
    }
  }
  return { weights: (label, geom) => entry(label, geom).w, attach, setPose, manifest,
    dispose() { for (const m of depthMaterials) m.dispose(); depthMaterials.clear(); states.clear(); } };
}
