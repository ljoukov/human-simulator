import {
  BufferGeometry, Color, DynamicDrawUsage, Float32BufferAttribute, GreaterDepth, Group, Mesh,
  MeshBasicMaterial, MeshStandardMaterial, Vector3,
} from 'three';
import archive from '../../knee-loading/data/muscle-paths.json';
import type { BodyFrame } from './body-scene.ts';

// Each centreline is sampled from an OpenSim GeometryPath at the recording's
// actual pose. Display widths and colors identify guides, not tissue thickness.
const SIDES = 6;
const samplesPerFrame = archive.paths.reduce((sum, path) => sum + path.count, 0);
const decoded = new Map<string, Int16Array>();
const clips: Record<string, string> = archive.clips;
const cos = Array.from({ length: SIDES }, (_, j) => Math.cos(2 * Math.PI * j / SIDES));
const sin = Array.from({ length: SIDES }, (_, j) => Math.sin(2 * Math.PI * j / SIDES));

function clipData(clipId: string): Int16Array | undefined {
  const cached = decoded.get(clipId);
  if (cached) return cached;
  const encoded = clips[clipId];
  if (!encoded) return;
  const raw = atob(encoded);
  if (raw.length !== archive.frameCount * samplesPerFrame * 6) throw new Error('Incomplete OpenSim path archive.');
  const ticks = new Int16Array(raw.length / 2);
  for (let i = 0; i < ticks.length; i++) {
    ticks[i] = raw.charCodeAt(2 * i) | raw.charCodeAt(2 * i + 1) << 8;
  }
  decoded.set(clipId, ticks);
  return ticks;
}

export function createMusclePaths() {
  const object = new Group();
  object.name = 'OpenSim muscle–tendon and patellar ligament paths';
  const positions = new Float32BufferAttribute(new Float32Array(samplesPerFrame * SIDES * 3), 3).setUsage(DynamicDrawUsage);
  const normals = new Float32BufferAttribute(new Float32Array(positions.array.length), 3).setUsage(DynamicDrawUsage);
  const colors = new Float32BufferAttribute(new Float32Array(positions.array.length), 3);
  const indices: number[] = [];
  const center = Array.from({ length: samplesPerFrame }, () => new Vector3());
  const tangent = new Vector3(), normal = new Vector3(), binormal = new Vector3(), outward = new Vector3();
  const muscleColor = new Color('#eda55d'), ligamentColor = new Color('#eee2b8');
  let first = 0;
  const tubes = archive.paths.map(path => {
    const start = first;
    const ligament = path.kind === 'ligament';
    const color = ligament ? ligamentColor : muscleColor;
    for (let i = start; i < start + path.count; i++) {
      for (let j = 0; j < SIDES; j++) {
        const corner = i * SIDES + j;
        colors.setXYZ(corner, color.r, color.g, color.b);
        if (i === start + path.count - 1) continue;
        const next = i * SIDES + (j + 1) % SIDES;
        indices.push(corner, next, corner + SIDES, next, next + SIDES, corner + SIDES);
      }
    }
    first += path.count;
    return { start, end: first, radius: ligament ? 0.0035 : 0.0021 };
  });
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', positions);
  geometry.setAttribute('normal', normals);
  geometry.setAttribute('color', colors);
  geometry.setIndex(indices);

  const surface = new MeshStandardMaterial({ vertexColors: true, roughness: 0.7 });
  const visiblePaths = new Mesh(geometry, surface);
  // Faint occluded guides separate internal routing from exposed segments.
  const hiddenSurface = new MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.24, depthFunc: GreaterDepth, depthWrite: false,
  });
  const hiddenPaths = new Mesh(geometry, hiddenSurface);
  visiblePaths.frustumCulled = hiddenPaths.frustumCulled = false;
  hiddenPaths.renderOrder = 2;
  object.add(visiblePaths, hiddenPaths);

  function setFrame(frame: BodyFrame) {
    const samples = frame.clipId && clipData(frame.clipId);
    const pelvis = frame.segments.pelvis;
    visiblePaths.visible = hiddenPaths.visible = Boolean(samples && pelvis);
    if (!samples || !pelvis) return;
    object.position.set(...pelvis.position);
    const lastFrame = archive.frameCount - 1;
    const time = lastFrame * Math.min(1, Math.max(0, (frame.phase ?? 0) / 100));
    const a = Math.floor(time);
    const b = Math.ceil(time);
    const blend = time - a;
    const aStart = 3 * samplesPerFrame * a, bStart = 3 * samplesPerFrame * b;
    for (let i = 0; i < center.length; i++) {
      const j = 3 * i, from = aStart + j, to = bStart + j;
      center[i].set(
        samples[from] + blend * (samples[to] - samples[from]),
        samples[from + 1] + blend * (samples[to + 1] - samples[from + 1]),
        samples[from + 2] + blend * (samples[to + 2] - samples[from + 2]),
      ).multiplyScalar(0.0001);
    }
    for (const { start, end, radius } of tubes) {
      for (let i = start; i < end; i++) {
        tangent.subVectors(center[Math.min(i + 1, end - 1)], center[Math.max(start, i - 1)]).normalize();
        if (i === start) {
          normal.set(0, 0, 1);
          if (Math.abs(tangent.z) > 0.9) normal.set(1, 0, 0);
        }
        // Project the previous normal onto the new cross-section to keep the
        // six-sided guide continuous through curved ankle and knee wraps.
        binormal.crossVectors(tangent, normal).normalize();
        normal.crossVectors(binormal, tangent).normalize();
        for (let j = 0; j < SIDES; j++) {
          outward.copy(normal).multiplyScalar(cos[j]).addScaledVector(binormal, sin[j]);
          const v = i * SIDES + j, p = center[i];
          positions.setXYZ(v, p.x + radius * outward.x, p.y + radius * outward.y, p.z + radius * outward.z);
          normals.setXYZ(v, outward.x, outward.y, outward.z);
        }
      }
    }
    positions.needsUpdate = normals.needsUpdate = true;
  }

  return {
    object, setFrame,
    dispose() { geometry.dispose(); surface.dispose(); hiddenSurface.dispose(); },
  };
}
