import {
  ACESFilmicToneMapping, BufferGeometry, CanvasTexture, Color, ConeGeometry, CylinderGeometry, DirectionalLight, GridHelper, Group,
  HemisphereLight, Line, LineBasicMaterial, Mesh, MeshBasicMaterial, OrthographicCamera,
  Raycaster, Scene, Sprite, SpriteMaterial, Vector2, Vector3, WebGLRenderer,
  type Material, type Object3D,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createWalkingAtlas } from './body-layers.ts';
import { createMusclePaths } from './muscle-paths.ts';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const BODY_STACK_WIDTH = 520;

type XYZ = readonly [number, number, number];
type XYZW = readonly [number, number, number, number];
type BodySide = 'right' | 'left';
type GroundForce = readonly [number, number, number, number, number, number];
export type BodySlot = 'reference' | 'selected';

export interface BodyFrame {
  segments: Record<string, { position: XYZ; rotation: XYZW }>;
  clipId?: string;
  cameraOriginM?: XYZ;
  headPosition?: XYZ;
  side?: BodySide;
  kneePosition?: XYZ;
  kneeForceBw?: number | null;
  ground?: { right: GroundForce | null; left: GroundForce | null };
  heading?: 1 | -1;
  phase?: number;
}

const VIEWS = {
  side: [0.65, 0.24, 5],
  anterior: [5, 0.28, 0.02],
  oblique: [3.3, 0.48, 4.5],
  posterior: [-5, 0.32, -0.02],
} as const;

export type BodyView = keyof typeof VIEWS;
export type BodyLayers = Partial<Record<'muscles' | 'skeleton' | 'context' | 'paths' | 'knee' | 'ground' | 'forces', boolean>>;

export interface BodyScene {
  setFrames(reference: BodyFrame | null, selected: BodyFrame | null): void;
  setView(view: BodyView): void;
  setLayers(layers: BodyLayers): void;
  zoomBy(factor: number): void;
  reset(): void;
  dispose(): void;
}

interface BodySceneOptions {
  onKneeActivate?: (slot: BodySlot) => void;
}

interface Rig {
  setFrame(frame: BodyFrame): void;
  setLayers(layers: Required<BodyLayers>): void;
  dispose(): void;
  materials: Set<Material>;
  geometry: Set<BufferGeometry>;
  marker: Sprite;
  scene: Scene;
}

const anatomyUrl = new URL('../../knee-loading/assets/walking-atlas.glb', import.meta.url).href;
const BACKGROUND = 0x112625;
const VIEW_HEIGHT_M = 1.99;
const VIEW_MIN_WIDTH_M = .91;
const TARGET = new Vector3(0.045, 0.87, 0);
const Y_UP = new Vector3(0, 1, 0);
const INITIAL_VIEW: BodyView = 'oblique';
const DEFAULT_LAYERS: Required<BodyLayers> = { muscles: true, skeleton: true, paths: false, context: false, knee: true, ground: true, forces: true };
const FORCE_ARROW_METRES_PER_BW = 0.4;

function markerTexture(): CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const drawing = canvas.getContext('2d');
  if (drawing) {
    const center = 64;
    drawing.strokeStyle = '#fff';
    drawing.lineWidth = 3;
    for (let quadrant = 0; quadrant < 4; quadrant++) {
      const angle = quadrant * Math.PI / 2;
      drawing.beginPath();
      drawing.arc(center, center, 46, angle + 0.13, angle + Math.PI / 2 - 0.13);
      drawing.stroke();
      drawing.beginPath();
      drawing.moveTo(center + Math.cos(angle) * 50, center + Math.sin(angle) * 50);
      drawing.lineTo(center + Math.cos(angle) * 56, center + Math.sin(angle) * 56);
      drawing.stroke();
    }
    drawing.globalAlpha = 0.16;
    drawing.fillStyle = '#fff';
    drawing.beginPath();
    drawing.arc(center, center, 42, 0, 2 * Math.PI);
    drawing.fill();
  }
  return new CanvasTexture(canvas);
}

function addLights(scene: Scene): void {
  scene.add(new HemisphereLight(0xfff3e7, 0x869c96, 1.52));
  const key = new DirectionalLight(0xffeedb, 2.17);
  key.position.set(2.5, 3.5, 3.1);
  scene.add(key);
  const rim = new DirectionalLight(0xc3e6e3, 1.84);
  rim.position.set(-2, 2.1, -3);
  scene.add(rim);
  const fill = new DirectionalLight(0xe3e4ef, .62);
  fill.position.set(2, .5, -2);
  scene.add(fill);
}

function makeRig(template: Object3D, indicator: CanvasTexture): Rig {
  const scene = new Scene();
  addLights(scene);

  const alignment = new Group();
  scene.add(alignment);
  const atlas = createWalkingAtlas(template);
  const paths = createMusclePaths();
  const subjects = new Group();
  subjects.add(atlas.object, paths.object);
  alignment.add(subjects);
  const materials = new Set<Material>();
  const geometry = new Set<BufferGeometry>();

  const ground = new Group();
  // Draw the reference plane before the opaque anatomy. An orbit near or
  // below the floor can otherwise put its lines across the visible foot.
  ground.renderOrder = -1;
  alignment.add(ground);
  const grid = new GridHelper(8, 32, 0x41645a, 0x2c4942);
  grid.material.depthWrite = false;
  grid.position.y = -0.006;
  ground.add(grid);
  materials.add(grid.material);
  geometry.add(grid.geometry);
  const directionGeometry = new BufferGeometry().setFromPoints([
    new Vector3(-4, 0.001, 0),
    new Vector3(4, 0.001, 0),
  ]);
  const directionMaterial = new LineBasicMaterial({ color: 0x4e7665, depthWrite: false });
  ground.add(new Line(directionGeometry, directionMaterial));
  materials.add(directionMaterial);
  geometry.add(directionGeometry);

  const markerMaterial = new SpriteMaterial({ color: '#ffcb88', map: indicator, transparent: true, depthTest: false, depthWrite: false, toneMapped: false });
  const marker = new Sprite(markerMaterial);
  marker.scale.setScalar(0.165);
  marker.renderOrder = 5;
  subjects.add(marker);
  materials.add(markerMaterial);

  const loads = new Group();
  subjects.add(loads);
  const shaftShape = new CylinderGeometry(1, 1, 1, 12);
  const headShape = new ConeGeometry(1, 1, 16);
  geometry.add(shaftShape).add(headShape);

  function makeForceArrow(color: number) {
    const material = new MeshBasicMaterial({
      color, transparent: true, opacity: 0.88, depthTest: false, depthWrite: false, toneMapped: false,
    });
    materials.add(material);
    const root = new Group();
    const shaft = new Mesh(shaftShape, material);
    const head = new Mesh(headShape, material);
    shaft.renderOrder = head.renderOrder = 4;
    root.visible = false;
    root.add(shaft, head);
    loads.add(root);
    return { root, shaft, head };
  }
  const arrows = { right: makeForceArrow(0xffc28a), left: makeForceArrow(0xa3ead7) };

  const forceLow = new Color('#ceead0');
  const forceHigh = new Color('#ffad62');
  const forceDirection = new Vector3();
  let layers = DEFAULT_LAYERS;
  let hasKnee = false;
  let lastFrame: BodyFrame | null = null;

  function setFrame(frame: BodyFrame) {
    const pelvis = frame.segments.pelvis;
    lastFrame = frame;
    const center = frame.cameraOriginM ?? pelvis.position;
    const [, qy, qz] = pelvis.rotation;
    const heading = frame.heading ?? (1 - 2 * (qy ** 2 + qz ** 2) < 0 ? -1 : 1);
    // Acquisition sessions can use opposite laboratory headings. Rotate the
    // complete scene by half a turn, including its ground reaction vectors.
    alignment.rotation.y = heading === -1 ? Math.PI : 0;
    subjects.position.set(-center[0], 0, -center[2]);
    // The floor is a reference only. Both views retain the measured vertical
    // positions and move the grid horizontally to expose recorded progression.
    ground.position.set(-center[0], 0, -center[2]);

    atlas.setFrame(frame);
    if (layers.paths) paths.setFrame(frame);
    const side = frame.side ?? 'right';
    const joint = frame.kneePosition ?? frame.segments[side === 'right' ? 'tibia' : 'tibia_l']?.position;
    hasKnee = Boolean(joint);
    marker.visible = hasKnee && layers.knee;
    if (joint) {
      marker.position.set(...joint);
      const force = frame.kneeForceBw;
      const available = force != null && Number.isFinite(force);
      const fraction = available ? Math.max(0, Math.min(1, force / 2.5)) : 0;
      markerMaterial.color.lerpColors(forceLow, forceHigh, fraction);
      markerMaterial.opacity = available ? 0.93 : 0.72;
      marker.scale.setScalar(0.148 + 0.033 * fraction);
    }

    for (const side of ['right', 'left'] as const) {
      const force = frame.ground?.[side];
      const { root, shaft, head } = arrows[side];
      root.visible = force != null;
      if (!force) continue;
      const [x, y, z, fx, fy, fz] = force;
      const magnitude = forceDirection.set(fx, fy, fz).length();
      root.visible = Number.isFinite(magnitude) && magnitude > 1e-4;
      if (!root.visible) continue;
      root.position.set(x, y, z);
      root.quaternion.setFromUnitVectors(Y_UP, forceDirection.divideScalar(magnitude));
      const length = magnitude * FORCE_ARROW_METRES_PER_BW;
      const headLength = Math.min(0.055, length * 0.34);
      const shaftLength = length - headLength;
      const shaftRadius = Math.min(0.006, headLength * 0.19);
      const headRadius = Math.min(0.019, headLength * 0.65);
      shaft.position.y = shaftLength / 2;
      shaft.scale.set(shaftRadius, shaftLength, shaftRadius);
      head.position.y = shaftLength + headLength / 2;
      head.scale.set(headRadius, headLength, headRadius);
    }
  }

  return {
    scene, marker, materials, geometry, setFrame,
    dispose() { atlas.dispose(); paths.dispose(); },
    setLayers(next) {
      layers = next;
      atlas.setLayers(layers.muscles, layers.skeleton, layers.context);
      paths.object.visible = layers.paths;
      if (layers.paths && lastFrame) paths.setFrame(lastFrame);
      ground.visible = layers.ground;
      loads.visible = layers.forces;
      marker.visible = layers.knee && hasKnee;
    },
  };
}

interface Viewport {
  slot: BodySlot;
  index: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export async function createBodyScene(host: HTMLElement, options: BodySceneOptions = {}): Promise<BodyScene> {
  let renderer: WebGLRenderer;
  try {
    renderer = new WebGLRenderer({ antialias: true, alpha: false });
  } catch {
    throw new Error('The whole-body comparison requires WebGL 2.');
  }

  const canvas = renderer.domElement;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.97;
  renderer.setClearColor(BACKGROUND);
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.setAttribute('aria-label', 'Two synchronized recorded gait poses. Arrow keys rotate. Plus and minus zoom. Enter opens knee tissue detail.');
  canvas.setAttribute('role', 'img');
  canvas.tabIndex = 0;
  host.prepend(canvas);

  const sourceMaterials = new Set<Material>();
  const sourceGeometry = new Set<BufferGeometry>();
  const indicator = markerTexture();
  let rigs: Rig[];
  try {
    const model = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(anatomyUrl);
    model.scene.traverse((node) => {
      if (!(node as Mesh).isMesh) return;
      const mesh = node as Mesh;
      sourceGeometry.add(mesh.geometry);
      for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) sourceMaterials.add(material);
    });
    rigs = [makeRig(model.scene, indicator), makeRig(model.scene, indicator)];
  } catch (error) {
    indicator.dispose();
    sourceGeometry.forEach((geometry) => geometry.dispose());
    sourceMaterials.forEach((material) => material.dispose());
    renderer.dispose();
    canvas.remove();
    throw error;
  }

  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.05, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableDamping = false;
  controls.rotateSpeed = 0.65;
  controls.minZoom = 0.65;
  controls.maxZoom = 2.3;
  controls.target.copy(TARGET);

  const frames: [BodyFrame | null, BodyFrame | null] = [null, null];
  const slots: BodySlot[] = ['reference', 'selected'];
  let layers = { ...DEFAULT_LAYERS };
  let screen: Viewport[] = [];
  let width = 0;
  let height = 0;
  let ratio = 0;
  let pendingFrame = 0;
  let disposed = false;

  function frameCamera(viewport: Viewport) {
    const aspect = viewport.width / viewport.height;
    const halfHeight = Math.max(VIEW_HEIGHT_M, VIEW_MIN_WIDTH_M / aspect) / 2;
    camera.left = -halfHeight * aspect;
    camera.right = halfHeight * aspect;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();
  }

  function layout(): Viewport[] {
    const active = frames.flatMap((frame, index) => frame ? [{ slot: slots[index], index }] : []);
    if (active.length === 1) return [{ ...active[0], x: 0, y: 0, width, height }];
    if (active.length < 2) return [];
    if (width < BODY_STACK_WIDTH) {
      const half = Math.floor(height / 2);
      return [
        { ...active[0], x: 0, y: 0, width, height: half },
        { ...active[1], x: 0, y: half, width, height: height - half },
      ];
    }
    const half = Math.floor(width / 2);
    return [
      { ...active[0], x: 0, y: 0, width: half, height },
      { ...active[1], x: half, y: 0, width: width - half, height },
    ];
  }

  function redraw() {
    pendingFrame = 0;
    if (disposed) return;
    const nextWidth = Math.floor(host.clientWidth);
    const nextHeight = Math.floor(host.clientHeight);
    if (!nextWidth || !nextHeight) return;
    const nextRatio = Math.min(window.devicePixelRatio || 1, 2);
    if (nextWidth !== width || nextHeight !== height || nextRatio !== ratio) {
      width = nextWidth;
      height = nextHeight;
      ratio = nextRatio;
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
    }

    screen = layout();
    host.dataset.bodyLayout = screen.length < 2 ? 'single' : width < BODY_STACK_WIDTH ? 'stacked' : 'paired';
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    for (const viewport of screen) {
      frameCamera(viewport);
      const origin = height - viewport.y - viewport.height;
      renderer.setViewport(viewport.x, origin, viewport.width, viewport.height);
      renderer.setScissor(viewport.x, origin, viewport.width, viewport.height);
      renderer.render(rigs[viewport.index].scene, camera);
    }
  }

  function requestRender() {
    if (!pendingFrame && !disposed) pendingFrame = requestAnimationFrame(redraw);
  }

  function setView(view: BodyView) {
    camera.position.set(...VIEWS[view]).add(TARGET);
    camera.up.copy(Y_UP);
    controls.target.copy(TARGET);
    controls.update();
    requestRender();
  }

  function zoomBy(factor: number) {
    const zoom = camera.zoom * factor;
    camera.zoom = Math.max(controls.minZoom, Math.min(controls.maxZoom, zoom));
    camera.updateProjectionMatrix();
    controls.update();
    requestRender();
  }

  const pointer = new Vector2();
  const raycaster = new Raycaster();
  let start: { x: number; y: number; id: number } | undefined;
  function startPointer(event: PointerEvent) {
    start = { x: event.clientX, y: event.clientY, id: event.pointerId };
  }
  function endPointer(event: PointerEvent) {
    if (start?.id !== event.pointerId) return;
    const movement = Math.hypot(start.x - event.clientX, start.y - event.clientY);
    start = undefined;
    if (movement > 5 || !layers.knee) return;
    const bounds = canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const hit = screen.find((viewport) => (
      x >= viewport.x && x < viewport.x + viewport.width
      && y >= viewport.y && y < viewport.y + viewport.height
    ));
    if (!hit || !rigs[hit.index].marker.visible) return;
    frameCamera(hit);
    pointer.set(2 * (x - hit.x) / hit.width - 1, 1 - 2 * (y - hit.y) / hit.height);
    raycaster.setFromCamera(pointer, camera);
    if (raycaster.intersectObject(rigs[hit.index].marker, false).length) options.onKneeActivate?.(hit.slot);
  }

  function keyAction(event: KeyboardEvent) {
    if (event.key === '+' || event.key === '=') {
      zoomBy(1.15);
    } else if (event.key === '-' || event.key === '_') {
      zoomBy(1 / 1.15);
    } else if (event.key === 'Enter' && layers.knee) {
      if (frames[1]) options.onKneeActivate?.('selected');
      else if (frames[0]) options.onKneeActivate?.('reference');
    } else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
      const offset = camera.position.clone().sub(controls.target);
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        offset.applyAxisAngle(Y_UP, event.key === 'ArrowLeft' ? 0.14 : -0.14);
      } else {
        const right = new Vector3().crossVectors(Y_UP, offset).normalize();
        offset.applyAxisAngle(right, event.key === 'ArrowUp' ? -0.12 : 0.12);
      }
      camera.position.copy(controls.target).add(offset);
      controls.update();
      requestRender();
    } else {
      return;
    }
    event.preventDefault();
  }

  const observer = new ResizeObserver(requestRender);
  observer.observe(host);
  controls.addEventListener('change', requestRender);
  canvas.addEventListener('keydown', keyAction);
  canvas.addEventListener('pointerdown', startPointer);
  canvas.addEventListener('pointerup', endPointer);
  setView(INITIAL_VIEW);

  for (const rig of rigs) rig.setLayers(layers);

  return {
    setFrames(reference, selected) {
      frames[0] = reference?.segments.pelvis ? reference : null;
      frames[1] = selected?.segments.pelvis ? selected : null;
      rigs.forEach((rig, index) => {
        const frame = frames[index];
        if (frame) rig.setFrame(frame);
      });
      requestRender();
    },
    setView,
    setLayers(next) {
      layers = { ...layers, ...next };
      for (const rig of rigs) rig.setLayers(layers);
      requestRender();
    },
    zoomBy,
    reset() {
      camera.zoom = 1;
      setView(INITIAL_VIEW);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(pendingFrame);
      observer.disconnect();
      controls.dispose();
      canvas.removeEventListener('keydown', keyAction);
      canvas.removeEventListener('pointerdown', startPointer);
      canvas.removeEventListener('pointerup', endPointer);
      for (const rig of rigs) {
        rig.dispose();
        rig.materials.forEach((material) => material.dispose());
        rig.geometry.forEach((geometry) => geometry.dispose());
      }
      indicator.dispose();
      sourceMaterials.forEach((material) => material.dispose());
      sourceGeometry.forEach((geometry) => geometry.dispose());
      renderer.dispose();
      canvas.remove();
    },
  };
}
