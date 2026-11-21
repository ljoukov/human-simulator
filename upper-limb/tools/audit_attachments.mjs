// node tools/audit_attachments.mjs [--json] [--angles 0,35,65,90,105,112.5,125] before.glb [after.glb]
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { AnimationMixer, Triangle, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const biceps = 'Common tendon of biceps brachii', triceps = 'Common tendon of triceps brachii';
const sheet = 'Bicipital aponeurosis', fascia = 'Antebrachial fascia';
const long = 'Long head of biceps brachii', short = 'Short head of biceps brachii';
const flexors = ['Flexor carpi radialis', 'Palmaris longus muscle', 'Humeral head of pronator teres'];

// Windows in millimeters in the published source atlas coordinate system. Each window
// must contain neutral-pose proximity to the named recipient to form a material pair.
const interfaces = [
  { id: 'biceps-radius', source: biceps, targets: ['Radius'], center: [-237, 1057, -18], radius: 13 },
  { id: 'triceps-ulna', source: triceps, targets: ['Ulna'], center: [-214, 1106, -60], radius: 15 },
  { id: 'biceps-long-head', source: biceps, targets: [long], center: [-224, 1118, -17], radius: 15 },
  { id: 'biceps-short-head', source: biceps, targets: [short], center: [-214, 1117, -16], radius: 15 },
  { id: 'biceps-long-head-nearest', source: biceps, targets: [long], competingHead: short,
    center: [-224, 1118, -17], radius: 15 },
  { id: 'biceps-short-head-nearest', source: biceps, targets: [short], competingHead: long,
    center: [-214, 1117, -16], radius: 15 },
  { id: 'triceps-long-head', source: triceps, targets: ['Long head of triceps brachii'], center: [-205, 1117, -63], radius: 14 },
  { id: 'triceps-lateral-head', source: triceps, targets: ['Lateral head of triceps brachii'], center: [-221, 1117, -62], radius: 14 },
  { id: 'triceps-medial-head', source: triceps, targets: ['Medial head of triceps brachii'], center: [-201, 1117, -51], radius: 14 },
  { id: 'sheet-biceps-tendon', source: sheet, targets: [biceps], center: [-218, 1114, -12], radius: 15 },
  { id: 'sheet-long-head', source: sheet, targets: [long], center: [-223, 1120, -12], radius: 12 },
  { id: 'sheet-short-head', source: sheet, targets: [short], center: [-217, 1119, -11], radius: 12 },
  { id: 'sheet-forearm-fascia', source: sheet, targets: [fascia], center: [-205, 1063, -12], radius: 45,
    yMin: 1036, yMax: 1090, maxGap: 8, forearmFootprint: true, optional: true },
  { id: 'sheet-fascial-rim', source: sheet, targets: [fascia], center: [-205, 1063, -12], radius: 45,
    yMin: 1036, yMax: 1090, maxGap: 8, forearmFootprint: true, maxNormalDot: 0.55, optional: true },
  { id: 'sheet-flexor-neighborhood', source: sheet, targets: flexors, center: [-205, 1080, -12], radius: 31,
    yMin: 1070, yMax: 1090, maxGap: 4, forearmFootprint: true, kind: 'neighbor' },
];

const round = n => Math.round(n * 1000) / 1000;
const xyz = (a, b, c) => new Vector3(a, b, c);
const allLabels = new Set(interfaces.flatMap(i => [i.source, ...i.targets]));
const labelKey = s => s.toLowerCase().replace(/[^a-z]/g, '');
async function openAnimation(path) {
  const bytes = await readFile(path);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const { scene, animations } = await new GLTFLoader().parseAsync(buffer, '');
  const clip = animations.find(a => a.name === 'Elbow flexion');
  if (!clip) throw new Error('Missing Elbow flexion animation in ' + path);
  const mixer = new AnimationMixer(scene), action = mixer.clipAction(clip).play();
  action.paused = true;
  const meshes = new Map();
  scene.traverse(node => {
    if (!node.isMesh) return;
    let parent = node;
    while (parent && !parent.userData.label) parent = parent.parent;
    if (!parent) return;
    const key = labelKey(parent.userData.label);
    const label = [...allLabels].find(l => labelKey(l) === key)
      ?? (/^(?:proximal)?antebrachialfascia(?:cutaway)?$/.test(key) ? fascia : undefined);
    if (label) meshes.set(label, [...(meshes.get(label) ?? []), node]);
  });
  return {
    sha256: createHash('sha256').update(bytes).digest('hex'),
    at(degrees) {
      action.time = clip.duration * degrees / 125;
      mixer.update(0);
      scene.updateMatrixWorld(true);
      const posed = new Map();
      for (const [label, primitives] of meshes) {
        const points = [], faces = [];
        for (const mesh of primitives) {
          const geometry = mesh.geometry, n = geometry.attributes.position.count, offset = points.length;
          for (let i = 0; i < n; i++) {
            points.push(mesh.getVertexPosition(i, new Vector3()).applyMatrix4(mesh.matrixWorld).multiplyScalar(1000));
          }
          const indices = geometry.index?.array ?? Array.from({ length: n }, (_, i) => i);
          for (let i = 0; i < indices.length; i += 3) {
            faces.push([indices[i], indices[i + 1], indices[i + 2]].map(k => k + offset));
          }
        }
        posed.set(label, { points, faces });
      }
      return posed;
    },
  };
}

const tmp = new Vector3(), nearest = new Vector3();
function closest(point, surfaces) {
  let distance = Infinity, face = null;
  for (const entry of surfaces) {
    const { tri } = entry;
    // Ties use the earliest source triangle to make material correspondence deterministic.
    const candidate = tri.closestPointToPoint(point, tmp).distanceToSquared(point);
    if (candidate < distance) {
      distance = candidate; face = entry; nearest.copy(tmp);
    }
  }
  return { gap: Math.sqrt(distance), face, point: nearest.clone() };
}

function makeFrame(t) {
  const u = new Vector3().subVectors(t.b, t.a).normalize();
  const n = t.getNormal(new Vector3());
  return [u, new Vector3().crossVectors(n, u), n];
}

function getSurface({ label, index }, pose) {
  const { points, faces } = pose.get(label), [a, b, c] = faces[index];
  return new Triangle(points[a], points[b], points[c]);
}

function surfacesNear(labels, center, radius, pose) {
  const entries = [];
  for (const label of labels) {
    for (let index = 0; index < pose.get(label).faces.length; index++) {
      const key = { label, index }, tri = getSurface(key, pose);
      if (tri.getArea() > 1e-9 && tri.closestPointToPoint(center, tmp).distanceTo(center) <= radius) {
        entries.push({ ...key, tri });
      }
    }
  }
  return entries;
}

function weldedNormals(points, faces) {
  const keys = points.map(p => p.toArray().map(n => Math.round(n * 1e4)).join(','));
  const sums = new Map(keys.map(key => [key, new Vector3()]));
  for (const [a, b, c] of faces) {
    const twiceAreaNormal = new Vector3().subVectors(points[b], points[a])
      .cross(new Vector3().subVectors(points[c], points[a]));
    for (const index of [a, b, c]) sums.get(keys[index]).add(twiceAreaNormal);
  }
  for (const n of sums.values()) n.normalize();
  return keys.map(key => sums.get(key));
}

function freezePatch(spec, rest, observedRest) {
  const center = xyz(...spec.center), band = spec.maxGap ?? 1.5;
  const unavailable = [spec.source, ...spec.targets].filter(label => !rest.has(label));
  if (unavailable.length) {
    if (!spec.optional) throw new Error('Missing anatomy: ' + unavailable.join(', '));
    return { status: 'missing', unavailable };
  }
  const targets = surfacesNear(spec.targets, center, spec.radius + band + 2, rest);
  const competingHead = spec.competingHead ? surfacesNear([spec.competingHead], center, spec.radius + band + 2, rest) : null;
  const flexorSurfaces = spec.forearmFootprint ? surfacesNear(flexors, center, spec.radius + 4, rest) : [];
  const tendonSurfaces = spec.forearmFootprint ? surfacesNear([biceps], center, spec.radius + 4, rest) : [];
  const { points, faces } = rest.get(spec.source), areas = new Float64Array(points.length);
  const normals = spec.maxNormalDot ? weldedNormals(points, faces) : null;
  for (const [a, b, c] of faces) {
    const share = new Triangle(points[a], points[b], points[c]).getArea() / 3;
    areas[a] += share; areas[b] += share; areas[c] += share;
  }
  const probes = [];
  for (let index = 0; index < points.length; index++) {
    const p = points[index];
    if (!areas[index] || p.distanceTo(center) > spec.radius || p.y < (spec.yMin ?? -Infinity)
      || p.y > (spec.yMax ?? Infinity) || !targets.length) continue;
    if (spec.forearmFootprint && (closest(p, flexorSurfaces).gap >= 4 || closest(p, tendonSurfaces).gap <= 4)) continue;
    const hit = closest(p, targets);
    if (hit.gap > band) continue;
    if (competingHead && closest(p, competingHead).gap < hit.gap) continue;
    const triangle = hit.face.tri;
    if (normals && Math.abs(triangle.getNormal(tmp).dot(normals[index])) >= spec.maxNormalDot) continue;
    const bary = triangle.getBarycoord(hit.point, new Vector3());
    const observed = getSurface(hit.face, observedRest);
    const observedAnchor = new Vector3().addScaledVector(observed.a, bary.x)
      .addScaledVector(observed.b, bary.y).addScaledVector(observed.c, bary.z);
    const difference = new Vector3().subVectors(observedRest.get(spec.source).points[index], observedAnchor);
    const local = makeFrame(observed).map(axis => axis.dot(difference));
    probes.push({ index, weight: areas[index], label: hit.face.label, targetFace: hit.face.index, bary, local });
  }
  const topology = Object.fromEntries([spec.source, ...spec.targets].map(label => [label,
    createHash('sha256').update(JSON.stringify(rest.get(label).faces)).digest('hex')]));
  return { status: probes.length ? 'tracked' : 'empty', targets, probes, topology };
}

function percentile(values, q) {
  const sorted = values.slice().sort((a, b) => a[0] - b[0]);
  const threshold = sorted.reduce((sum, v) => sum + v[1], 0) * q;
  let weight = 0;
  for (const [value, w] of sorted) { weight += w; if (weight >= threshold) return round(value); }
}

function measure(spec, patch, pose) {
  const targets = patch.targets.map(t => ({ ...t, tri: getSurface(t, pose) }));
  const gaps = [], drifts = [];
  const source = pose.get(spec.source);
  for (const p of patch.probes) {
    const point = source.points[p.index], { gap } = closest(point, targets);
    const t = getSurface({ label: p.label, index: p.targetFace }, pose);
    const bary = p.bary;
    const expected = new Vector3().addScaledVector(t.a, bary.x).addScaledVector(t.b, bary.y).addScaledVector(t.c, bary.z);
    makeFrame(t).forEach((axis, i) => expected.addScaledVector(axis, p.local[i]));
    gaps.push([gap, p.weight]);
    drifts.push([point.distanceTo(expected), p.weight]);
  }
  return { gapMedianMm: percentile(gaps, 0.5), gapP95Mm: percentile(gaps, 0.95),
    materialDriftMedianMm: percentile(drifts, 0.5), materialDriftP95Mm: percentile(drifts, 0.95) };
}

// The three thin tissues contain duplicated vertices at hard-normal seams.
// Freeze the pairs before any sewing changes the neutral fan shape.
function seamPairs(points) {
  const pairs = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < i; j++) {
      const restGap = points[i].distanceTo(points[j]);
      if (restGap <= 0.001) pairs.push({ i, j, restGap });
    }
  }
  return pairs;
}

function seamOpening(points, pairs) {
  let max = 0, countOverPointOneMm = 0;
  for (const { i, j, restGap } of pairs) {
    const added = points[i].distanceTo(points[j]) - restGap;
    max = Math.max(max, added);
    if (added > 0.1) countOverPointOneMm++;
  }
  return { maxAddedGapMm: round(max), countOverPointOneMm };
}

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  json: { type: 'boolean' }, angles: { type: 'string', default: '0,35,65,90,105,112.5,125' },
} });
const angles = [...new Set([0, ...values.angles.split(',').map(Number)])];
if (!positionals.length || angles.some(n => !Number.isFinite(n) || n < 0 || n > 125)) {
  throw new Error('Supply one or more GLBs and sample angles in the range 0–125°.');
}
const output = [];
const frozen = new Map();
const frozenSeams = new Map();
let originalRest, sourceSha256;
for (const file of positionals) {
  const model = await openAnimation(file), rest = model.at(0);
  originalRest ??= rest;
  sourceSha256 ??= model.sha256;
  const seamLabels = [biceps, triceps, sheet, fascia].filter(label => rest.has(label));
  for (const label of seamLabels) {
    if (!frozenSeams.has(label)) frozenSeams.set(label, seamPairs(rest.get(label).points));
  }
  const seamResults = Object.fromEntries(seamLabels.map(label =>
    [label, { restPairs: frozenSeams.get(label).length, poses: {} }]));
  const patches = interfaces.map(spec => {
    if (spec.optional && !spec.targets.every(label => rest.has(label))) return { status: 'missing', unavailable: spec.targets.filter(label => !rest.has(label)) };
    const reference = frozen.get(spec.id);
    if (!reference) {
      // A cutaway may add the missing fascia. Use its neutral surface together
      // with the original fan to identify the same source material after sewing.
      const selectionRest = new Map([...rest, ...originalRest]);
      const patch = { ...freezePatch(spec, selectionRest, rest), referenceSha256: model.sha256 };
      if (patch.status === 'tracked') frozen.set(spec.id, patch);
      return patch;
    }
    for (const [label, hash] of Object.entries(reference.topology)) {
      const current = createHash('sha256').update(JSON.stringify(rest.get(label)?.faces ?? null)).digest('hex');
      if (current !== hash) throw new Error('Cannot reuse frozen material points with changed topology: ' + label);
    }
    return reference;
  });
  const results = interfaces.map((spec, i) => {
    const p = patches[i];
    return { ...spec, status: p.status, sourceReferenceSha256: sourceSha256,
      referenceSha256: p.referenceSha256, unavailable: p.unavailable, count: p.probes?.length,
      restAreaMm2: p.probes ? round(p.probes.reduce((a, b) => a + b.weight, 0)) : undefined, poses: {} };
  });
  for (const angle of angles) {
    const pose = angle === 0 ? rest : model.at(angle);
    for (const label of seamLabels) {
      seamResults[label].poses[angle] = seamOpening(pose.get(label).points, frozenSeams.get(label));
    }
    interfaces.forEach((spec, i) => {
      if (patches[i].status === 'tracked') results[i].poses[angle] = measure(spec, patches[i], pose);
    });
  }
  output.push({ file, sha256: model.sha256, anglesDeg: angles,
    method: 'Footprints are selected on the first model’s neutral source tissue, using the first available neutral recipient surface. Vertices lie within the stated anatomy windows and 1.5 mm of their target unless otherwise specified. The distal fan is within 4 mm of a superficial forearm flexor and more than 4 mm from the biceps tendon. An 8 mm window admits the atlas fascia shell. The rim additionally requires an absolute dot product below 0.55 between the welded fan normal and nearest fascia face normal. Quantiles are weighted by first-model neutral source surface area. Gap is unsigned to the local target surface. Material drift tracks the same target triangle and barycentric point through animation, relative to the zero-pose offset of the first model containing both tissues.',
    interfaces: results, tissueSeams: seamResults });
}
if (values.json) console.log(JSON.stringify(output));
else for (const result of output) {
  console.log(result.file + '  ' + result.sha256);
  console.log('Area-weighted P95 values in mm as gap / material drift');
  console.table(result.interfaces.map(r => Object.fromEntries([['interface', r.id], ['probes', r.count ?? r.status],
    ...angles.map(a => [a + '°', r.poses[a] ? r.poses[a].gapP95Mm + ' / ' + r.poses[a].materialDriftP95Mm : r.status])])));
}
