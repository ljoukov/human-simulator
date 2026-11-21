// From the simulator directory: node tools/audit.mjs [--json] [--self-crossings] [--grid-mm 2] [--angles 0,30,60,90,120] assets/lower-limb.glb
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';

// The upper and lower viewers use the same Three.js dependency.
const require = createRequire(new URL('../../upper-limb/package.json', import.meta.url));
const { AnimationMixer, Ray, Triangle, Vector3 } = await import(
  pathToFileURL(join(dirname(require.resolve('three')), 'three.module.js'))
);
const { GLTFLoader } = await import(pathToFileURL(require.resolve('three/addons/loaders/GLTFLoader.js')));

const quad = 'Quadriceps common tendon and patellar ligament', bf = 'Common tendon of biceps femoris';
const achilles = 'Calcaneal tendon',
  semimem = 'Semimembranosus muscle tendon',
  pes = 'Pes anserinus common tendon';
const vasti = ['Vastus intermedius muscle', 'Vastus lateralis muscle', 'Vastus medialis muscle'];
const biceps = ['Long head of biceps femoris', 'Short head of biceps femoris'];
const gastroc = ['Lateral head of gastrocnemius', 'Medial head of gastrocnemius'];
const fibularNerve = 'Common fibular nerve', sciaticNerve = 'Sciatic nerve';
const muscles = [
  'Rectus femoris',
  ...vasti,
  ...biceps,
  ...gastroc,
  'Semimembranosus muscle',
  'Semitendinosus muscle',
  'Sartorius muscle',
  'Gracilis muscle',
  'Popliteus muscle',
  'Plantaris muscle',
  'Soleus muscle',
  'Tibialis anterior muscle',
  'Tibialis posterior muscle',
  'Fibularis longus muscle',
];
const bones = ['Femur', 'Tibia', 'Fibula', 'Patella'];
const cartilage = ['Art cart of femur distal end', 'Art cart of tibia proximal end', 'Art cart of patella'];
// Spheres in the neutral Open3DModel atlas, in mm. The audit selects only source
// vertices in the sphere that also lie within 1.5 mm of the named target surface.
const footprints = [
  [quad, 'Patella', [-86, 443, 9], 31],
  [quad, 'Tibia', [-84, 385, 2], 23],
  ['Rectus femoris', quad, [-91, 494, 19], 27],
  [vasti[0], quad, [-95, 506, 11], 29],
  [vasti[1], quad, [-111, 498, 5], 30],
  [vasti[2], quad, [-62, 495, 8], 31],
  [biceps[0], bf, [-123, 434, -49], 19],
  [biceps[1], bf, [-121, 437, -49], 19],
  [bf, 'Fibula', [-121, 402, -45], 23],
  [biceps[1], 'Femur', [-100, 624, -20], 75],
  ['Popliteus muscle', 'Femur', [-115, 458, -30], 20],
  ['Popliteus muscle', 'Tibia', [-64, 368, -34], 33],
  ['Plantaris muscle', 'Femur', [-95, 476, -39], 22],
  [gastroc[0], 'Femur', [-105, 474, -42], 20],
  [gastroc[1], 'Femur', [-59, 478, -36], 22],
  [gastroc[0], achilles, [-92, 204, -78], 22],
  [gastroc[1], achilles, [-65, 210, -72], 22],
  ['Soleus muscle', achilles, [-80, 183, -72], 42],
  [achilles, 'Calcaneus', [-70, 40, -89], 19],
  ['Semimembranosus muscle', semimem, [-36, 445, -50], 24],
  [semimem, 'Tibia', [-56, 391, -41], 45],
  ['Semitendinosus muscle', pes, [-39, 413, -46], 27],
  ['Sartorius muscle', pes, [-40, 420, -21], 27],
  ['Gracilis muscle', pes, [-38, 414, -30], 24],
  [pes, 'Tibia', [-63, 364, -18], 25],
  ['Tibial nerve', sciaticNerve, [-66, 695, -29], 9],
  [fibularNerve, sciaticNerve, [-66, 695, -29], 9],
  ['Deep fibular nerve', fibularNerve, [-119, 405, -54], 7],
];
const tissues = [...new Set([...muscles, quad, bf, achilles, semimem, pes])];
const medialThigh = ['Semitendinosus muscle', 'Gracilis muscle', 'Sartorius muscle'];
const names = new Set([...muscles, ...bones, ...cartilage, ...footprints.flatMap((v) => v.slice(0, 2))]);
const round = (n) => Math.round(n * 1000) / 1000;
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const scratch = new Vector3();

async function load(file) {
  const bytes = await readFile(file);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const { scene, animations } = await new GLTFLoader().parseAsync(buffer, '');
  const clip = animations.find((a) => a.name === 'Knee flexion'), pivot = scene.getObjectByName('KneePivot');
  if (!clip || !pivot) throw new Error('Expected KneePivot and the Knee flexion animation in ' + file);
  const groups = new Map();
  scene.traverse((node) => {
    if (!node.isMesh) return;
    let owner = node;
    while (owner && !owner.userData.label) owner = owner.parent;
    const label = owner?.userData.label;
    if (names.has(label)) groups.set(label, [...(groups.get(label) ?? []), node]);
  });
  const missing = [...names].filter((name) => !groups.has(name));
  if (missing.length) throw new Error('Missing required named anatomy: ' + missing.join(', '));
  const topology = new Map();
  for (const [label, nodes] of groups) {
    let count = 0;
    const faces = [];
    for (const mesh of nodes) {
      const n = mesh.geometry.attributes.position.count,
        indices = mesh.geometry.index?.array ?? Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < indices.length; i += 3) {
        faces.push([indices[i] + count, indices[i + 1] + count, indices[i + 2] + count]);
      }
      count += n;
    }
    topology.set(label, { faces, hash: hash(JSON.stringify(faces)) });
  }
  const mixer = new AnimationMixer(scene), action = mixer.clipAction(clip).play();
  action.paused = true;
  return {
    sha256: hash(bytes),
    topology,
    pose(angle) {
      action.time = clip.duration * angle / 120;
      mixer.update(0);
      scene.updateMatrixWorld(true);
      const pose = new Map();
      for (const [label, group] of groups) {
        const points = [];
        for (const mesh of group) {
          for (let i = 0; i < mesh.geometry.attributes.position.count; i++) {
            points.push(
              mesh.getVertexPosition(i, new Vector3()).applyMatrix4(mesh.matrixWorld).multiplyScalar(1000),
            );
          }
        }
        pose.set(label, { points, faces: topology.get(label).faces });
      }
      return { pose, center: pivot.getWorldPosition(new Vector3()).multiplyScalar(1000) };
    },
  };
}

function triangle(mesh, i) {
  const [a, b, c] = mesh.faces[i];
  return new Triangle(mesh.points[a], mesh.points[b], mesh.points[c]);
}

function frame(t) {
  const u = new Vector3().subVectors(t.b, t.a).normalize(), n = t.getNormal(new Vector3());
  return [u, new Vector3().crossVectors(n, u), n];
}

function nearest(point, candidates) {
  let distance = Infinity, picked = null, projected = new Vector3();
  for (const entry of candidates) {
    const t = entry.tri;
    // A bounding box test avoids most exact point-to-triangle evaluations.
    const dx = Math.max(entry.lo.x - point.x, 0, point.x - entry.hi.x);
    const dy = Math.max(entry.lo.y - point.y, 0, point.y - entry.hi.y);
    const dz = Math.max(entry.lo.z - point.z, 0, point.z - entry.hi.z);
    if (dx * dx + dy * dy + dz * dz >= distance) continue;
    const d = t.closestPointToPoint(point, scratch).distanceToSquared(point);
    if (d < distance) {
      distance = d;
      picked = entry;
      projected.copy(scratch);
    }
  }
  return { gap: Math.sqrt(distance), entry: picked, point: projected };
}

function candidate(mesh, face) {
  const tri = triangle(mesh, face);
  return { face, tri, lo: tri.a.clone().min(tri.b).min(tri.c), hi: tri.a.clone().max(tri.b).max(tri.c) };
}

function freezeAnchors(rest) {
  return footprints.map(([source, target, coords, radius]) => {
    const mesh = rest.get(source), recipient = rest.get(target), center = new Vector3(...coords);
    const vertices = new Float64Array(mesh.points.length);
    for (let i = 0; i < mesh.faces.length; i++) {
      const area = triangle(mesh, i).getArea() / 3;
      for (const index of mesh.faces[i]) vertices[index] += area;
    }
    const candidates = [];
    for (let f = 0; f < recipient.faces.length; f++) {
      const t = triangle(recipient, f);
      if (t.getArea() > 1e-8 && t.closestPointToPoint(center, scratch).distanceTo(center) <= radius + 5) {
        candidates.push(candidate(recipient, f));
      }
    }
    const anchors = [];
    mesh.points.forEach((p, index) => {
      if (p.distanceTo(center) > radius || vertices[index] === 0) return;
      const hit = nearest(p, candidates);
      if (hit.gap > 1.5) return;
      const tri = hit.entry.tri, bary = tri.getBarycoord(hit.point, new Vector3());
      const offset = new Vector3().subVectors(p, hit.point);
      anchors.push({
        index,
        face: hit.entry.face,
        bary,
        local: frame(tri).map((axis) => axis.dot(offset)),
        weight: vertices[index],
      });
    });
    if (!anchors.length) throw new Error('No neutral material anchors for ' + source + ' / ' + target);
    return { source, target, anchors, faces: candidates.map((c) => c.face) };
  });
}

function quantile(values, q) {
  values.sort((a, b) => a[0] - b[0]);
  const target = values.reduce((s, v) => s + v[1], 0) * q;
  let sum = 0;
  for (const [value, weight] of values) {
    sum += weight;
    if (sum >= target) return round(value);
  }
}

function measureAttachment(patch, pose) {
  const source = pose.get(patch.source), target = pose.get(patch.target);
  const candidates = patch.faces.map((f) => candidate(target, f)), gaps = [], drifts = [];
  for (const { index, face, bary, local, weight } of patch.anchors) {
    const p = source.points[index], t = triangle(target, face);
    const expected = new Vector3().addScaledVector(t.a, bary.x).addScaledVector(t.b, bary.y).addScaledVector(
      t.c,
      bary.z,
    );
    frame(t).forEach((ax, k) => expected.addScaledVector(ax, local[k]));
    gaps.push([nearest(p, candidates).gap, weight]);
    drifts.push([p.distanceTo(expected), weight]);
  }
  return {
    gapMedianMm: quantile(gaps, .5),
    gapP95Mm: quantile(gaps, .95),
    materialDriftP95Mm: quantile(drifts, .95),
  };
}

// The visible crease spans three medial-thigh muscles. Bin by the original
// atlas height so a revised pose cannot silently move a bad patch out of view.
function triangleBasis(t) {
  const u = new Vector3().subVectors(t.b, t.a), v = new Vector3().subVectors(t.c, t.a);
  const x = u.length();
  if (x < 1e-9) return [0, 0, 0];
  return [x, u.dot(v) / x, u.cross(v).length() / x];
}

function principalStretches(reference, posed) {
  const [x, y, z] = reference, [X, Y, Z] = triangleBasis(posed);
  const a = X / x, b = (Y - a * y) / z, c = Z / z;
  const trace = a * a + b * b + c * c, det = a * a * c * c;
  const discriminant = Math.sqrt(Math.max(0, trace * trace - 4 * det));
  return [Math.sqrt(Math.max(0, (trace - discriminant) / 2)), Math.sqrt((trace + discriminant) / 2)];
}

function dihedral(mesh, normals, edge) {
  const axis = new Vector3().subVectors(mesh.points[edge.b], mesh.points[edge.a]).normalize();
  const left = normals[edge.left], right = normals[edge.right];
  return Math.atan2(new Vector3().crossVectors(left, right).dot(axis), left.dot(right));
}

function freezeSurfaceBands(mesh) {
  const bands = new Map(), edges = new Map(), faceCount = mesh.faces.length;
  for (let y = 545; y < 765; y += 10) bands.set(y, { faces: [], edges: [] });
  const bandAt = (y) => bands.get(545 + 10 * Math.floor((y - 545) / 10));
  const normals = Array.from({ length: faceCount }, (_, i) => triangle(mesh, i).getNormal(new Vector3()));
  mesh.faces.forEach(([i, j, k], f) => {
    const t = triangle(mesh, f), area = t.getArea(), band = bandAt((t.a.y + t.b.y + t.c.y) / 3);
    if (band && area > .1) band.faces.push({ f, area, basis: triangleBasis(t) });
    for (const [i1, i2] of [[i, j], [j, k], [k, i]]) {
      const a = Math.min(i1, i2), b = Math.max(i1, i2), key = a * mesh.points.length + b;
      if (edges.has(key)) edges.get(key).right = f;
      else edges.set(key, { a, b, left: f, right: null });
    }
  });
  for (const edge of edges.values()) {
    const a = mesh.points[edge.a],
      b = mesh.points[edge.b],
      band = bandAt((a.y + b.y) / 2),
      length = a.distanceTo(b);
    if (!band || edge.right === null || length <= 1) continue;
    const baseline = dihedral(mesh, normals, edge);
    if (Math.abs(baseline) < 50 * Math.PI / 180) band.edges.push({ ...edge, baseline, length });
  }
  return [...bands].filter(([, b]) => b.faces.length >= 4 && b.edges.length >= 6);
}

function measureSurfaceBands(mesh, patches) {
  const normals = mesh.faces.map((_, i) => triangle(mesh, i).getNormal(new Vector3()));
  return Object.fromEntries(patches.map(([y, band]) => {
    const stretchMin = [], stretchMax = [], bends = [];
    for (const { f, area, basis } of band.faces) {
      const [lo, hi] = principalStretches(basis, triangle(mesh, f));
      stretchMin.push([lo, area]);
      stretchMax.push([hi, area]);
    }
    for (const edge of band.edges) {
      const change = dihedral(mesh, normals, edge) - edge.baseline;
      const degrees = Math.abs(Math.atan2(Math.sin(change), Math.cos(change))) * 180 / Math.PI;
      bends.push([degrees, edge.length]);
    }
    return [y, {
      faces: band.faces.length,
      edges: band.edges.length,
      principalStretchP05: quantile(stretchMin, .05),
      principalStretchP95: quantile(stretchMax, .95),
      addedDihedralP95Deg: quantile(bends, .95),
    }];
  }));
}

function freezeSeams(mesh) {
  const cells = new Map(), pairs = [];
  for (let i = 0; i < mesh.points.length; i++) {
    const point = mesh.points[i], cell = point.toArray().map((n) => Math.floor(n / 0.001));
    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
          const bucket = cells.get([cell[0] + x, cell[1] + y, cell[2] + z].join('/'));
          if (bucket) {
            for (const j of bucket) {
              const d = point.distanceTo(mesh.points[j]);
              if (d <= 0.001) pairs.push([i, j, d]);
            }
          }
        }
      }
    }
    const key = cell.join('/');
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(i);
  }
  return pairs;
}

function topologyCounts(mesh) {
  const edges = new Map(), size = mesh.points.length;
  for (const [i, j, k] of mesh.faces) {
    for (const [a, b] of [[i, j], [j, k], [k, i]]) {
      const key = Math.min(a, b) * size + Math.max(a, b);
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  const counts = [...edges.values()];
  return {
    boundaryEdges: counts.filter((n) => n === 1).length,
    nonManifoldEdges: counts.filter((n) => n > 2).length,
  };
}

const edgeRay = new Ray(), pierce = new Vector3(), barycentric = new Vector3();
function edgePierces(from, to, t) {
  edgeRay.origin.copy(from);
  edgeRay.direction.subVectors(to, from);
  if (!edgeRay.intersectTriangle(t.a, t.b, t.c, false, pierce)) return false;
  const along = from.distanceToSquared(pierce);
  if (along <= 1e-12 || along >= edgeRay.direction.lengthSq() - 1e-12) return false;
  t.getBarycoord(pierce, barycentric);
  return Math.min(barycentric.x, barycentric.y, barycentric.z) > 1e-9;
}

// Count only proper non-coplanar crossings between non-adjacent faces.
// Tangencies and shared vertices/edges can occur on a valid surface.
function countInternalCrossings(mesh) {
  const entries = mesh.faces.map((_, i) => candidate(mesh, i));
  const lo = new Vector3(Infinity, Infinity, Infinity), hi = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const e of entries) {
    lo.min(e.lo);
    hi.max(e.hi);
  }
  const span = hi.sub(lo);
  const axis = span.y > span.x && span.y > span.z ? 'y' : span.z > span.x ? 'z' : 'x';
  entries.sort((a, b) => a.lo[axis] - b.lo[axis]);
  let count = 0;
  for (let i = 0; i < entries.length; i++) {
    const a = entries[i], f = mesh.faces[a.face], t = a.tri;
    for (let j = i + 1; j < entries.length && entries[j].lo[axis] <= a.hi[axis]; j++) {
      const b = entries[j], s = b.tri;
      if (
        b.lo.x > a.hi.x || a.lo.x > b.hi.x || b.lo.y > a.hi.y || a.lo.y > b.hi.y ||
        b.lo.z > a.hi.z || a.lo.z > b.hi.z || mesh.faces[b.face].some((v) => f.includes(v))
      ) continue;
      if (
        edgePierces(t.a, t.b, s) || edgePierces(t.b, t.c, s) || edgePierces(t.c, t.a, s) ||
        edgePierces(s.a, s.b, t) || edgePierces(s.b, s.c, t) || edgePierces(s.c, s.a, t)
      ) count++;
    }
  }
  return count;
}

function shape(mesh, rest, seams, checkFolds) {
  let maxOpening = 0, volume = 0;
  for (const [i, j, d] of seams) maxOpening = Math.max(maxOpening, mesh.points[i].distanceTo(mesh.points[j]) - d);
  const origin = mesh.points[0], a = new Vector3(), b = new Vector3(), c = new Vector3(), edges = [];
  // Signed tetrahedral volume is an indicator of collapse. It assumes a closed,
  // consistently oriented surface and is not independent anatomical ground truth.
  for (const [i, j, k] of mesh.faces) {
    a.subVectors(mesh.points[i], origin);
    b.subVectors(mesh.points[j], origin);
    c.subVectors(mesh.points[k], origin);
    volume += a.dot(b.cross(c)) / 6;
    // One sample per oriented face edge; almost all interior edges occur twice.
    for (const [s, t] of [[i, j], [j, k], [k, i]]) {
      const initial = rest.points[s].distanceTo(rest.points[t]);
      if (initial >= 1) edges.push([mesh.points[s].distanceTo(mesh.points[t]) / initial, initial]);
    }
  }
  return {
    maxAddedSeamMm: round(maxOpening),
    volumeCm3: round(Math.abs(volume) / 1000),
    signedVolumeCm3: round(volume / 1000),
    properSelfCrossings: checkFolds ? countInternalCrossings(mesh) : undefined,
    edgeLengthP05: quantile(edges, .05),
    edgeLengthP95: quantile(edges, .95),
  };
}

// Integrate overlapping closed interiors using exact ray–triangle intersections
// on each of two orthogonal grids, clipped to a sphere centered at the knee.
function scan(mesh, center, axis, step, radius) {
  const u = (axis + 1) % 3, v = (axis + 2) % 3;
  const n = Math.ceil(2 * radius / step) + 1,
    u0 = -radius + step * (axis === 0 ? .219 : .619),
    v0 = -radius + step * (axis === 0 ? .413 : .373);
  const points = mesh.points.map((p) => [p.x - center.x, p.y - center.y, p.z - center.z]), hits = new Map();
  for (const [i, j, k] of mesh.faces) {
    const a = points[i],
      b = points[j],
      c = points[k],
      bu = b[u] - a[u],
      bv = b[v] - a[v],
      cu = c[u] - a[u],
      cv = c[v] - a[v];
    const determinant = bu * cv - cu * bv;
    if (Math.abs(determinant) < 1e-10) continue;
    const x0 = Math.max(0, Math.ceil((Math.min(a[u], b[u], c[u]) - u0) / step)),
      x1 = Math.min(n - 1, Math.floor((Math.max(a[u], b[u], c[u]) - u0) / step));
    const y0 = Math.max(0, Math.ceil((Math.min(a[v], b[v], c[v]) - v0) / step)),
      y1 = Math.min(n - 1, Math.floor((Math.max(a[v], b[v], c[v]) - v0) / step));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const X = u0 + x * step, Y = v0 + y * step;
        if (X * X + Y * Y >= radius * radius) continue;
        const du = X - a[u],
          dv = Y - a[v],
          s = (du * cv - cu * dv) / determinant,
          t = (bu * dv - du * bv) / determinant;
        if (s < 0 || t < 0 || s + t > 1) continue;
        const key = y * n + x;
        if (!hits.has(key)) hits.set(key, []);
        hits.get(key).push(a[axis] + s * (b[axis] - a[axis]) + t * (c[axis] - a[axis]));
      }
    }
  }
  const chords = new Map();
  let odd = 0;
  for (const [key, raw] of hits) {
    raw.sort((a, b) => a - b);
    const depths = raw.filter((v, i) => i === 0 || v - raw[i - 1] > 1e-7);
    if (depths.length % 2) {
      odd++;
      continue;
    }
    const X = u0 + key % n * step,
      Y = v0 + Math.floor(key / n) * step,
      extent = Math.sqrt(radius * radius - X * X - Y * Y),
      intervals = [];
    for (let i = 0; i < depths.length; i += 2) {
      const lo = Math.max(-extent, depths[i]), hi = Math.min(extent, depths[i + 1]);
      if (hi > lo) intervals.push([lo, hi]);
    }
    if (intervals.length) chords.set(key, intervals);
  }
  return { chords, odd, total: hits.size };
}

function overlap(a, b, step) {
  let sum = 0;
  for (const [key, first] of a) {
    const second = b.get(key);
    if (second) {
      for (const [lo, hi] of first) {
        for (const [l, h] of second) sum += Math.max(0, Math.min(hi, h) - Math.max(lo, l));
      }
    }
  }
  return sum * step * step / 1000;
}

const collisionPairs = [];
muscles.forEach((name, i) => {
  for (const other of muscles.slice(i + 1)) collisionPairs.push([name, other, 'muscle–muscle']);
  for (const bone of bones) collisionPairs.push([name, bone, 'muscle–bone']);
});
for (
  const [a, b] of [['Femur', 'Tibia'], ['Femur', 'Patella'], [cartilage[0], cartilage[1]], [
    cartilage[0],
    cartilage[2],
  ]]
) {
  collisionPairs.push([a, b, 'joint surfaces']);
}
for (const bone of ['Femur', 'Patella', 'Tibia']) collisionPairs.push([quad, bone, 'extensor tendon–bone']);
collisionPairs.push([fibularNerve, 'Fibula', 'nerve–bone']);

const options = parseArgs({
  allowPositionals: true,
  options: {
    json: { type: 'boolean' },
    angles: { type: 'string' },
    'step-deg': { type: 'string', default: '5' },
    'grid-mm': { type: 'string', default: '2' },
    'attachments-only': { type: 'boolean' },
    'self-crossings': { type: 'boolean' },
  },
});
const step = Number(options.values['grid-mm']), increment = Number(options.values['step-deg']);
if (
  !options.positionals.length || !Number.isFinite(step) || step < 0.5 || step > 5 ||
  !Number.isFinite(increment) || increment < 0.25 || increment > 120
) {
  throw new Error('Supply a model, a grid of 0.5–5 mm, and an angular step of 0.25–120 degrees.');
}
const angles = options.values.angles
  ? options.values.angles.split(',').map(Number)
  : Array.from({ length: Math.floor(120 / increment) + 1 }, (_, i) => i * increment).concat(120);
angles.push(0);
const sequence = [...new Set(angles)].sort((a, b) => a - b);
if (sequence.some((n) => !Number.isFinite(n) || n < 0 || n > 120)) {
  throw new Error('Angles must fall within 0–120 degrees.');
}

const radius = 160, output = [];
let original, footprintPatches, seams, topology, referenceHash, surfaceBands;
for (const file of options.positionals) {
  const model = await load(file), neutral = model.pose(0);
  if (!original) {
    original = neutral.pose;
    referenceHash = model.sha256;
    topology = model.topology;
    footprintPatches = freezeAnchors(original);
    seams = new Map(tissues.map((n) => [n, freezeSeams(original.get(n))]));
    surfaceBands = new Map(medialThigh.map((n) => [n, freezeSurfaceBands(original.get(n))]));
  }
  for (const [label, ref] of topology) {
    if (model.topology.get(label)?.hash !== ref.hash) throw new Error('Different material topology for ' + label);
  }
  const attachments = footprintPatches.map((p) => ({
    source: p.source,
    target: p.target,
    count: p.anchors.length,
    areaMm2: round(p.anchors.reduce((v, probe) => v + probe.weight, 0)),
    poses: [],
  }));
  const shapeByTissue = Object.fromEntries(
    tissues.map((n) => [n, { restSeamPairs: seams.get(n).length, ...topologyCounts(original.get(n)), poses: [] }]),
  );
  const medialThighSurface = Object.fromEntries(medialThigh.map((n) => [n, []]));
  const contacts = options.values['attachments-only']
    ? []
    : collisionPairs.map(([a, b, kind]) => ({ pair: a + ' / ' + b, kind, meanCm3: [], orthogonalGridsCm3: [] }));
  const oddRays = [];
  for (const angle of sequence) {
    const { pose, center } = angle === 0 ? neutral : model.pose(angle);
    footprintPatches.forEach((p, i) => attachments[i].poses.push(measureAttachment(p, pose)));
    for (const n of tissues) {
      shapeByTissue[n].poses.push(
        shape(pose.get(n), original.get(n), seams.get(n), options.values['self-crossings']),
      );
    }
    for (const n of medialThigh) medialThighSurface[n].push(measureSurfaceBands(pose.get(n), surfaceBands.get(n)));
    if (contacts.length) {
      const solids = new Set(collisionPairs.flatMap((c) => c.slice(0, 2)));
      const grids = [0, 2].map((axis) =>
        new Map([...solids].map((label) => [label, scan(pose.get(label), center, axis, step, radius)]))
      );
      oddRays.push(
        Object.fromEntries(
          [...solids].map((label) => [label, grids.map((g) => [g.get(label).odd, g.get(label).total])]),
        ),
      );
      collisionPairs.forEach(([a, b], i) => {
        const volumes = grids.map((grid) => overlap(grid.get(a).chords, grid.get(b).chords, step));
        contacts[i].meanCm3.push(round((volumes[0] + volumes[1]) / 2));
        contacts[i].orthogonalGridsCm3.push(volumes.map(round));
      });
    }
  }
  for (const row of contacts) row.peakAddedCm3 = round(Math.max(...row.meanCm3) - row.meanCm3[0]);
  const excessSum = contacts.length
    ? sequence.map((_, i) =>
      round(contacts.reduce((sum, row) => sum + Math.max(0, row.meanCm3[i] - row.meanCm3[0]), 0))
    )
    : null;
  contacts.sort((a, b) => b.peakAddedCm3 - a.peakAddedCm3);
  output.push({
    file,
    sha256: model.sha256,
    referenceSha256: referenceHash,
    anglesDeg: sequence,
    gridMm: step,
    kneeRadiusMm: radius,
    method:
      'True posed vertices from GLTFLoader and AnimationMixer. Neutral source vertices within the named atlas windows and 1.5 mm of the recipient are weighted by incident triangle area. Gap is unsigned to that local surface. Relative motion tracks the original recipient triangle, barycentric point, and rest offset in its moving frame. Closed-solid overlap is estimated on two ray grids in a 160 mm knee sphere. Odd-parity rays are omitted. A sum of pairwise excesses may double-count shared volume.',
    surfaceBandMethod:
      'The medial thigh is divided into 10 mm bands in neutral atlas Y. Each band reports area-weighted principal in-surface stretches and an edge-length-weighted 95th percentile of absolute change in signed dihedral angle. The bend measure includes edges longer than 1 mm with neutral bend below 50 degrees. A stretch of 1 preserves local length.',
    attachments,
    shapeByTissue,
    medialThighSurface,
    pairCount: contacts.length,
    contacts,
    sumPositivePairExcessCm3: excessSum,
    oddRays,
  });
}
function printResult(r) {
  console.log(r.file + '  ' + r.sha256);
  console.table(r.attachments.map((p) => {
    const top = p.poses.reduce((best, v, i) => v.gapP95Mm > p.poses[best].gapP95Mm ? i : best, 0);
    return {
      source: p.source,
      target: p.target,
      vertices: p.count,
      restP95mm: p.poses[0].gapP95Mm,
      maxP95mm: p.poses[top].gapP95Mm,
      degrees: r.anglesDeg[top],
      maxDriftP95mm: Math.max(...p.poses.map((v) => v.materialDriftP95Mm)),
    };
  }));
  if (r.contacts.length) {
    console.table(
      r.contacts.slice(0, 10).map((row) => ({
        pair: row.pair,
        restCm3: row.meanCm3[0],
        peakAddedCm3: row.peakAddedCm3,
      })),
    );
  }
  const shapes = Object.entries(r.shapeByTissue).map(([label, s]) => ({
    label,
    lowestSignedVolumeFraction: round(
      Math.min(...s.poses.map((p) => p.signedVolumeCm3 / s.poses[0].signedVolumeCm3)),
    ),
    largestEdgeRatioP95: Math.max(...s.poses.map((p) => p.edgeLengthP95)),
    boundaryEdges: s.boundaryEdges,
  }));
  console.table(shapes.sort((a, b) => a.lowestSignedVolumeFraction - b.lowestSignedVolumeFraction).slice(0, 6));
  const example = r.anglesDeg.reduce(
    (best, a, i) => Math.abs(a - 105) < Math.abs(r.anglesDeg[best] - 105) ? i : best,
    0,
  );
  console.table(medialThigh.map((n) => {
    const low = n === 'Sartorius muscle' ? 635 : 625;
    const band = r.medialThighSurface[n][example][low];
    return {
      label: n,
      degrees: r.anglesDeg[example],
      neutralHeightMm: low + '–' + (low + 10),
      principalStretchP05: band.principalStretchP05,
      addedDihedralP95Deg: band.addedDihedralP95Deg,
    };
  }));
  if (options.values['self-crossings']) {
    const crossingCounts = Object.entries(r.shapeByTissue).map(([label, tissue]) => ({
      label,
      neutral: tissue.poses[0].properSelfCrossings,
      peak: Math.max(...tissue.poses.map((p) => p.properSelfCrossings)),
    }));
    console.table(crossingCounts.sort((a, b) => (b.peak - b.neutral) - (a.peak - a.neutral)).slice(0, 6));
  }
  const seamRecords = Object.entries(r.shapeByTissue).map(([label, s]) => ({
    label,
    restPairs: s.restSeamPairs,
    maxOpeningMm: Math.max(...s.poses.map((p) => p.maxAddedSeamMm)),
  }));
  const totalPairs = seamRecords.reduce((n, s) => n + s.restPairs, 0);
  if (totalPairs) {
    console.log(
      'Maximum added seam gap (mm)',
      Math.max(...seamRecords.map((s) => s.maxOpeningMm)),
      'across',
      totalPairs,
      'coincident pairs.',
    );
  } else {
    console.log('The audited tissues have no separate position indices coincident within 0.001 mm at extension.');
  }
}
if (options.values.json) {
  console.log(JSON.stringify(output));
} else {
  output.forEach(printResult);
}
