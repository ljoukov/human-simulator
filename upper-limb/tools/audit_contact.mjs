// node tools/audit_contact.mjs [--json] [--grid-mm 1] [--angles 0,65,90,125] model.glb [revised.glb]
// Integrates overlapping interiors using triangle intersections with two independent
// grids of parallel lines. Each solid is sampled over a sphere centered at the elbow.
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { basename } from 'node:path';
import { parseArgs } from 'node:util';
import { AnimationMixer, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const bones = ['Humerus', 'Radius', 'Ulna', 'Articular cartilage of humerus distal end',
  'Articular cartilage of radius head', 'Articular cartilage of ulna (proximal end)'];
const lateral = ['Brachioradialis muscle', 'Extensor carpi radialis longus',
  'Extensor carpi radialis brevis', 'Extensor digitorum', 'Extensor digiti minimi',
  'Humeral head of extensor carpi ulnaris', 'Ulnar head of extensor carpi ulnaris',
  'Anconeus muscle', 'Supinator'];
const medial = ['Flexor carpi radialis', 'Flexor digitorum superficialis humero-ulnar head',
  'Flexor digitorum superficialis radial head', 'Flexor digitorum profundus',
  'Humeral head of flexor carpi ulnaris', 'Ulnar head of flexor carpi ulnaris',
  'Humeral head of pronator teres', 'Ulnar head of pronator teres', 'Palmaris longus muscle'];
const upper = ['Brachialis muscle', 'Long head of biceps brachii', 'Short head of biceps brachii',
  'Long head of triceps brachii', 'Lateral head of triceps brachii', 'Medial head of triceps brachii'];
const muscles = [...lateral, ...medial, ...upper];
const labels = [...muscles, ...bones];
const radius = 130; // mm
const seamTolerance = 0.001; // mm in the unposed, decoded atlas

function readOptions() {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: {
    json: { type: 'boolean' }, 'grid-mm': { type: 'string', default: '1' },
    angles: { type: 'string', default: '0,65,90,125' },
  } });
  const angles = [...new Set([0, ...values.angles.split(',').map(Number)])];
  const step = Number(values['grid-mm']);
  if (!positionals.length || angles.some(x => !Number.isFinite(x) || x < 0 || x > 125)
    || !Number.isFinite(step) || step < 0.25 || step > 4) {
    throw new Error('Supply a GLB, elbow angles from 0–125°, and a grid step of 0.25–4 mm.');
  }
  return { files: positionals, json: values.json, angles, step };
}

async function loadPose(file) {
  const bytes = await readFile(file);
  const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
  const root = gltf.scene;
  const clip = gltf.animations.find(item => item.name === 'Elbow flexion');
  const pivot = root.getObjectByName('ElbowPivot');
  if (!clip || !pivot) throw new Error('Missing elbow animation or elbow pivot in ' + file);
  const grouped = new Map(labels.map(label => [label, []]));
  root.traverse(node => {
    if (!node.isMesh) return;
    for (let p = node; p; p = p.parent) {
      if (!p.userData.label) continue;
      grouped.get(p.userData.label)?.push(node);
      break;
    }
  });
  const missing = labels.filter(label => !grouped.get(label).length);
  if (missing.length) throw new Error('Missing structures: ' + missing.join(', '));
  const mixer = new AnimationMixer(root);
  const action = mixer.clipAction(clip).play();
  action.paused = true;
  root.updateMatrixWorld(true);
  const point = new Vector3();
  // The approximate longitudinal location of a collision can help distinguish
  // a mid-muscle overlap from contact near its origin/insertion. Fractional
  // superior–inferior position is measured on each entire unposed atlas muscle.
  const reference = new Map();
  for (const [label, meshes] of grouped) {
    let min = Infinity, max = -Infinity;
    const restY = meshes.map(mesh => Array.from({ length: mesh.geometry.attributes.position.count }, (_, i) => {
      const y = mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld).y;
      min = Math.min(min, y); max = Math.max(max, y);
      return y;
    }));
    reference.set(label, restY.map(ys => ys.map(y => (y - min) / (max - min))));
  }
  return {
    sha256: createHash('sha256').update(bytes).digest('hex'),
    pose(angle) {
      action.time = clip.duration * angle / 125;
      mixer.update(0);
      root.updateMatrixWorld(true);
      const center = pivot.getWorldPosition(new Vector3());
      const posed = new Map();
      for (const [label, meshes] of grouped) {
        const parts = meshes.map((mesh, part) => {
          const geometry = mesh.geometry;
          const points = Array.from({ length: geometry.attributes.position.count }, (_, i) =>
            mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld).sub(center).multiplyScalar(1000).toArray());
          return { points, fractions: reference.get(label)[part],
            indices: geometry.index?.array ?? points.map((_, i) => i) };
        });
        posed.set(label, parts);
      }
      return posed;
    },
  };
}

// For each line intersecting the triangle's 2D projection, calculate the exact
// transverse intersection with its plane. Use an oblique grid phase to avoid
// consistently sampling seams. Parity only uses closed, unambiguous columns.
function scan(parts, axis, step, phase) {
  const u = (axis + 1) % 3, v = (axis + 2) % 3;
  const width = Math.ceil(2 * radius / step) + 1;
  const originU = -radius + step * phase[0], originV = -radius + step * phase[1];
  const depths = new Map();
  for (const { points, indices, fractions } of parts) {
    for (let t = 0; t < indices.length; t += 3) {
      const a = points[indices[t]], b = points[indices[t + 1]], c = points[indices[t + 2]];
      const bu = b[u] - a[u], bv = b[v] - a[v], cu = c[u] - a[u], cv = c[v] - a[v];
      const determinant = bu * cv - cu * bv;
      if (Math.abs(determinant) < 1e-10) continue;
      const j0 = Math.max(0, Math.ceil((Math.min(a[v], b[v], c[v]) - originV) / step));
      const j1 = Math.min(width - 1, Math.floor((Math.max(a[v], b[v], c[v]) - originV) / step));
      const i0 = Math.max(0, Math.ceil((Math.min(a[u], b[u], c[u]) - originU) / step));
      const i1 = Math.min(width - 1, Math.floor((Math.max(a[u], b[u], c[u]) - originU) / step));
      for (let j = j0; j <= j1; j++) {
        const y = originV + j * step, dv = y - a[v];
        for (let i = i0; i <= i1; i++) {
          const x = originU + i * step, du = x - a[u];
          if (x * x + y * y >= radius * radius) continue;
          const wb = (du * cv - cu * dv) / determinant, wc = (bu * dv - du * bv) / determinant;
          if (wb < 0 || wc < 0 || wb + wc > 1) continue;
          const depth = a[axis] + wb * (b[axis] - a[axis]) + wc * (c[axis] - a[axis]);
          const fa = fractions[indices[t]];
          const fraction = fa + wb * (fractions[indices[t + 1]] - fa) + wc * (fractions[indices[t + 2]] - fa);
          const key = j * width + i;
          if (!depths.has(key)) depths.set(key, []);
          depths.get(key).push([depth, fraction]);
        }
      }
    }
  }
  const intervals = new Map();
  let odd = 0;
  for (const [key, raw] of depths) {
    raw.sort((a, b) => a[0] - b[0]);
    const hits = raw.filter((d, i) => !i || d[0] - raw[i - 1][0] > 1e-7);
    if (hits.length % 2) { odd++; continue; }
    const x = originU + (key % width) * step, y = originV + Math.floor(key / width) * step;
    const extent = Math.sqrt(radius * radius - x * x - y * y);
    const list = [];
    for (let k = 0; k < hits.length; k += 2) {
      const [start, f0] = hits[k], [end, f1] = hits[k + 1];
      const lo = Math.max(-extent, start), hi = Math.min(extent, end);
      if (hi <= lo) continue;
      // Linear interpolation between the two material coordinates along this
      // chord is a location estimate, not a volumetric tissue registration.
      let midLo = Infinity, midHi = -Infinity;
      if (Math.abs(f1 - f0) < 1e-12) {
        if (f0 >= 0.2 && f0 <= 0.8) { midLo = lo; midHi = hi; }
      } else {
        const s0 = start + (0.2 - f0) * (end - start) / (f1 - f0);
        const s1 = start + (0.8 - f0) * (end - start) / (f1 - f0);
        midLo = Math.max(lo, Math.min(s0, s1));
        midHi = Math.min(hi, Math.max(s0, s1));
      }
      list.push([lo, hi, midLo, midHi]);
    }
    if (list.length) intervals.set(key, list);
  }
  return { intervals, odd, total: depths.size };
}

function overlap(first, second, step) {
  let length = 0, midLength = 0;
  for (const [key, intervals] of first) {
    const other = second.get(key);
    if (!other) continue;
    for (const [a, b, midLo, midHi] of intervals) for (const [c, d] of other) {
      length += Math.max(0, Math.min(b, d) - Math.max(a, c));
      midLength += Math.max(0, Math.min(midHi, d) - Math.max(midLo, c));
    }
  }
  return { all: length * step * step, middle: midLength * step * step };
}

// Whole-mesh signed tetrahedron volume complements the local contact score:
// collapsing a muscle could remove collisions while also destroying its shape.
function enclosedVolume(parts) {
  let volume = 0;
  for (const { points, indices } of parts) {
    let sixVolume = 0;
    for (let t = 0; t < indices.length; t += 3) {
      const a = points[indices[t]], b = points[indices[t + 1]], c = points[indices[t + 2]];
      sixVolume += a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    volume += Math.abs(sixVolume) / 6;
  }
  return volume;
}

function squaredSeparation(a, b) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

// Hard normal seams can duplicate a position as two separately indexed vertices.
// Search neighboring spatial buckets as well as the current bucket so round-off
// at cell boundaries cannot conceal a pair. Coincidence is tested within each
// named structure, including all its primitives, and never across muscle names.
function findSeams(rest) {
  const seams = new Map();
  for (const name of muscles) {
    const buckets = new Map(), pairs = [];
    const parts = rest.get(name);
    parts.forEach(({ points }, part) => points.forEach((point, index) => {
      const [x, y, z] = point.map(v => Math.floor(v / seamTolerance));
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const nearby = buckets.get([x + dx, y + dy, z + dz].join(','));
        if (!nearby) continue;
        for (const [otherPart, otherIndex] of nearby) {
          const separation = squaredSeparation(point, parts[otherPart].points[otherIndex]);
          if (separation <= seamTolerance ** 2) pairs.push([part, index, otherPart, otherIndex, Math.sqrt(separation)]);
        }
      }
      const key = [x, y, z].join(',');
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push([part, index]);
    }));
    seams.set(name, pairs);
  }
  return seams;
}

function measureSeams(pose, seams) {
  return Object.fromEntries([...seams].map(([name, pairs]) => {
    let maxGap = 0, count = 0;
    const parts = pose.get(name);
    for (const [part, index, otherPart, otherIndex, restGap] of pairs) {
      const gap = Math.sqrt(squaredSeparation(parts[part].points[index], parts[otherPart].points[otherIndex])) - restGap;
      maxGap = Math.max(maxGap, gap);
      if (gap > 0.1) count++;
    }
    return [name, { maxAddedGapMm: Math.round(maxGap * 1000) / 1000, pairsOverPointOneMm: count }];
  }));
}

function measure(pose, step) {
  const scans = [[0, [0.213, 0.439]], [2, [0.631, 0.377]]].map(([axis, phase]) =>
    new Map([...pose].map(([label, parts]) => [label, scan(parts, axis, step, phase)])));
  const pairs = {};
  const oddRayCounts = {};
  for (const label of labels) {
    oddRayCounts[label] = scans.map(set => {
      const { odd, total } = set.get(label);
      return [odd, total];
    });
  }
  for (let i = 0; i < muscles.length; i++) for (let j = i + 1; j < labels.length; j++) {
    const key = labels[i] + ' / ' + labels[j];
    pairs[key] = scans.map(set => overlap(set.get(labels[i]).intervals, set.get(labels[j]).intervals, step));
  }
  const meshVolumes = Object.fromEntries(muscles.map(label => [label, enclosedVolume(pose.get(label))]));
  return { pairs, oddRayCounts, meshVolumes };
}

const mean = pair => (pair[0] + pair[1]) / 2;
const round = n => Math.round(n * 10) / 10;
const options = readOptions();
const output = [];
for (const file of options.files) {
  const model = await loadPose(file);
  const poses = {};
  const rest = model.pose(0), seams = findSeams(rest);
  for (const angle of options.angles) {
    const pose = angle === 0 ? rest : model.pose(angle);
    poses[angle] = measure(pose, options.step);
    poses[angle].seams = measureSeams(pose, seams);
  }
  const rows = Object.keys(poses[0].pairs).map(pair => {
    const values = options.angles.map(angle => poses[angle].pairs[pair].map(grid => grid.all));
    const middle = options.angles.map(angle => mean(poses[angle].pairs[pair].map(grid => grid.middle)));
    const peakAdded = Math.max(...values.map(v => mean(v) - mean(values[0])));
    return { pair, meanMm3: values.map(v => round(mean(v))), gridMm3: values.map(v => v.map(round)),
      firstStructureMidspanMm3: middle.map(round), peakAddedMm3: round(peakAdded) };
  }).sort((a, b) => b.peakAddedMm3 - a.peakAddedMm3);
  // Pairwise overlaps can share voxels, so this score is not a union volume.
  const sumPositivePoseExcessMm3 = options.angles.map(a => round(Object.keys(poses[a].pairs).reduce((sum, pair) => {
    const volume = mean(poses[a].pairs[pair].map(grid => grid.all));
    const atlas = mean(poses[0].pairs[pair].map(grid => grid.all));
    return sum + Math.max(0, volume - atlas);
  }, 0)));
  output.push({ file, sha256: model.sha256, anglesDeg: options.angles, stepMm: options.step, elbowRadiusMm: radius,
    method: 'Parity of triangle intersections on orthogonal X and Z ray grids, clipped to an elbow-centered sphere. The first structure midspan spans the central 60% of its unposed superior–inferior extent, estimated along tissue chords. End regions can include entheses. Open meshes and sub-grid overlaps are uncertain. Odd-parity rays are omitted.',
    pairCount: rows.length, sumPositivePoseExcessMm3, pairs: rows,
    wholeMuscleVolumeMm3: Object.fromEntries(muscles.map(m =>
      [m, options.angles.map(a => round(poses[a].meshVolumes[m]))])),
    muscleSeams: {
      restPairToleranceMm: seamTolerance,
      restPairs: Object.fromEntries([...seams].map(([m, pairs]) => [m, pairs.length])),
      byAngle: Object.fromEntries(options.angles.map(a => [a, poses[a].seams])),
    },
    oddAndTotalRays: Object.fromEntries(options.angles.map(a => [a, poses[a].oddRayCounts])) });
}

function compare(baseline, candidate) {
  const originals = new Map(baseline.pairs.map(row => [row.pair, row]));
  const changes = [];
  for (const row of candidate.pairs) {
    const original = originals.get(row.pair);
    for (let i = 1; i < options.angles.length; i++) {
      const before = original.meanMm3[i] - original.meanMm3[0];
      const after = row.meanMm3[i] - row.meanMm3[0];
      changes.push({ pair: row.pair, angleDeg: options.angles[i],
        baselineExcessMm3: round(before), candidateExcessMm3: round(after), deltaMm3: round(after - before) });
    }
  }
  changes.sort((a, b) => a.deltaMm3 - b.deltaMm3);
  return { baseline: baseline.sha256, pairCount: originals.size, largestDecrease: changes[0],
    largestIncrease: changes.at(-1) };
}
for (const candidate of output.slice(1)) candidate.comparison = compare(output[0], candidate);
if (options.json) {
  console.log(JSON.stringify(output));
} else for (const result of output) {
  console.log(basename(result.file) + '  ' + result.sha256);
  console.log('Mesh interior overlap (mm³), averaged across two ray grids within 130 mm of the elbow.');
  console.table(result.pairs.slice(0, 18).map(row => Object.fromEntries([
    ['pair', row.pair], ...result.anglesDeg.map((a, i) => [a + '°', row.meanMm3[i]]), ['peak added', row.peakAddedMm3],
  ])));
  console.log('Sum of positive pairwise excesses over each pair’s 0° value (mm³), by sampled pose:');
  console.log(Object.fromEntries(result.anglesDeg.map((a, i) => [a + '°', result.sumPositivePoseExcessMm3[i]])));
  const seamRows = result.anglesDeg.flatMap(angleDeg => Object.entries(result.muscleSeams.byAngle[angleDeg])
    .map(([muscle, metrics]) => ({ angleDeg, muscle, ...metrics })));
  console.log('Largest opening of vertices coincident at rest (mm):', seamRows.sort((a, b) => b.maxAddedGapMm - a.maxAddedGapMm)[0]);
  if (result.comparison) console.table([result.comparison.largestDecrease, result.comparison.largestIncrease]);
}
