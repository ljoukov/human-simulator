import {
  ACESFilmicToneMapping, AnimationMixer, Color, DirectionalLight, Group, HemisphereLight, LoopOnce,
  MeshStandardMaterial, OrthographicCamera, Scene, Vector3, WebGLRenderer,
  type AnimationClip, type AnimationAction, type Mesh, type Object3D,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneRig } from 'three/addons/utils/SkeletonUtils.js';

export const TISSUE_FORCE_MAX_BW = 2;
export const TISSUE_STACK_WIDTH = 520;

const anatomyUrl = new URL('../../knee-loading/assets/knee.glb', import.meta.url).href;
const STAGE_COLOR = 0x122726;
const NO_SAMPLE = new Color('#74918d');
const LOAD_STOPS = ['#45778a', '#83bab0', '#f1c173', '#e66a48'].map((color) => new Color(color));
const REFERENCE_FLEXION_DEG = 15;
const VIEW_HEIGHT_M = 0.166;
const VIEW_MIN_WIDTH_M = 0.128;
const Y_UP = new Vector3(0, 1, 0);
const TARGET = new Vector3(0, 0, 0);
const VIEWS = {
  anterior: [-0.85, 0.32, 2.8],
  medial: [2.8, 0.28, 0.55],
  lateral: [-2.8, 0.28, 0.55],
  posterior: [0.85, 0.32, -2.8],
  superior: [0.15, 2.8, 0.25],
} as const;

export type TissueView = keyof typeof VIEWS;
export type TissueLayer = 'bone' | 'cartilage' | 'menisci' | 'ligaments' | 'muscles' | 'loads';
export type TissueLayers = Partial<Record<TissueLayer, boolean>>;
export type TissueFrame = {
  flexionDeg: number | null;
  medialBw: number | null;
  lateralBw: number | null;
  side?: 'left' | 'right';
};

export interface TissueScene {
  setFrames(ordinary: TissueFrame | null, selected: TissueFrame | null): void;
  setLayers(layers: TissueLayers): void;
  setView(view: TissueView): void;
  zoomBy(factor: number): void;
  reset(): void;
  dispose(): void;
}

type PhysicalLayer = Exclude<TissueLayer, 'loads'>;
type Side = NonNullable<TissueFrame['side']>;
type LoadColors = { medial: Color; lateral: Color; visible: { value: number } };

interface JointScene {
  scene: Scene;
  group: Group;
  pose: AnimationAction;
  mixer: AnimationMixer;
  structures: Map<Object3D, PhysicalLayer>;
  materials: MeshStandardMaterial[];
  colors: LoadColors;
}

function assignLoadColor(forceBw: number | null, color: Color): Color {
  if (forceBw == null || !Number.isFinite(forceBw)) return color.copy(NO_SAMPLE);
  const index = Math.max(0, Math.min(1, forceBw / TISSUE_FORCE_MAX_BW)) * (LOAD_STOPS.length - 1);
  const low = Math.floor(index);
  return color.lerpColors(LOAD_STOPS[low], LOAD_STOPS[Math.min(low + 1, LOAD_STOPS.length - 1)], index - low);
}

export function tissueLoadColor(forceBw: number | null): string {
  return '#' + assignLoadColor(forceBw, new Color()).getHexString();
}

function anatomyMaterials() {
  return {
    bone: new MeshStandardMaterial({ color: '#e7dcc0', roughness: 0.83 }),
    cartilage: new MeshStandardMaterial({ color: '#7eaaa8', roughness: 0.55 }),
    menisci: new MeshStandardMaterial({ color: '#a8bd9d', roughness: 0.65 }),
    ligaments: new MeshStandardMaterial({ color: '#dfa97b', roughness: 0.7 }),
    muscles: new MeshStandardMaterial({ color: '#a64f46', roughness: 0.82 }),
  };
}

function anatomyLayer(name: string, sourceLayer?: string): PhysicalLayer | undefined {
  if (sourceLayer === 'bones') return 'bone';
  if (name.endsWith('meniscus')) return 'menisci';
  if (sourceLayer === 'cartilage') return 'cartilage';
  if (sourceLayer === 'ligaments' || /patellar ligament/.test(name)) return 'ligaments';
  if (sourceLayer === 'muscles' || sourceLayer === 'tendons') return 'muscles';
  return undefined;
}

/** Tint entire anatomical regions by compartment resultants, leaving the centre neutral. */
function compartmentMaterial(base: MeshStandardMaterial, splitX: number, colors: LoadColors) {
  const material = base.clone();
  material.customProgramCacheKey = () => 'knee-compartment-colors';
  material.onBeforeCompile = (program) => {
    program.uniforms.compartmentSplit = { value: splitX };
    program.uniforms.compartmentMedial = { value: colors.medial };
    program.uniforms.compartmentLateral = { value: colors.lateral };
    program.uniforms.compartmentVisible = colors.visible;
    program.vertexShader = program.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float compartmentSplit;\nvarying float compartmentX;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\ncompartmentX = position.x - compartmentSplit;');
    program.fragmentShader = program.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform vec3 compartmentMedial;
        uniform vec3 compartmentLateral;
        uniform float compartmentVisible;
        varying float compartmentX;`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        if (compartmentVisible > 0.5 && abs(compartmentX) > 0.003) {
          diffuseColor.rgb = mix(compartmentLateral, compartmentMedial, step(0.0, compartmentX));
        }`);
  };
  return material;
}

function makeJoint(template: Object3D, flexion: AnimationClip, tissueCenter: Vector3): JointScene {
  const scene = new Scene();
  scene.add(new HemisphereLight(0xfef5e4, 0x3d7571, 2.15));
  const key = new DirectionalLight(0xffead4, 2.5);
  key.position.set(-0.25, 0.42, 0.6);
  scene.add(key);
  const rim = new DirectionalLight(0x9ccfc8, 1.85);
  rim.position.set(0.4, 0.12, -0.45);
  scene.add(rim);

  const model = cloneRig(template);
  const group = new Group();
  group.add(model);
  scene.add(group);
  model.updateMatrixWorld(true);
  const joint = model.getObjectByName('KneePivot');
  if (!joint) throw new Error('The tissue model does not contain a knee pivot.');
  const kneeCenter = joint.getWorldPosition(new Vector3());

  const mainMaterials = anatomyMaterials();
  const materials = Object.values(mainMaterials);
  const colors: LoadColors = { medial: new Color(), lateral: new Color(), visible: { value: 1 } };
  const structures = new Map<Object3D, PhysicalLayer>();

  model.traverse((object: Object3D) => {
    if (!(object as Mesh).isMesh) return;
    const mesh = object as Mesh;
    const label = (typeof mesh.userData.label === 'string' ? mesh.userData.label : mesh.name).toLowerCase();
    const layer = anatomyLayer(label, mesh.userData.layer);
    if (!layer) {
      mesh.visible = false;
      return;
    }
    structures.set(mesh, layer);
    if (label === 'art cart of tibia proximal end') {
      const splitX = mesh.worldToLocal(kneeCenter.clone()).x;
      const material = compartmentMaterial(mainMaterials.cartilage, splitX, colors);
      materials.push(material);
      mesh.material = material;
    } else {
      mesh.material = mainMaterials[layer];
    }
  });

  model.position.sub(tissueCenter);
  const mixer = new AnimationMixer(model);
  const pose = mixer.clipAction(flexion).setLoop(LoopOnce, 1).play();
  pose.paused = true;
  return { scene, group, pose, mixer, structures, materials, colors };
}

export async function createTissueScene(host: HTMLElement, modelUrl = anatomyUrl): Promise<TissueScene> {
  let renderer: WebGLRenderer;
  try {
    renderer = new WebGLRenderer({ antialias: true, alpha: false });
  } catch {
    throw new Error('The 3D knee comparison requires WebGL 2. Force curves are still available.');
  }

  const canvas = renderer.domElement;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.94;
  renderer.setClearColor(STAGE_COLOR);
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Synchronized 3D knee views. Arrow keys rotate. Plus and minus zoom.');
  canvas.setAttribute('role', 'img');
  host.prepend(canvas);

  let loaded: Awaited<ReturnType<GLTFLoader['loadAsync']>>;
  try {
    loaded = await new GLTFLoader().loadAsync(modelUrl);
    if (!loaded.animations.some((clip) => clip.name === 'Knee flexion') || !loaded.scene.getObjectByName('KneePivot')) {
      throw new Error('The tissue model is missing its fitted flexion animation.');
    }
  } catch (error) {
    renderer.dispose();
    canvas.remove();
    throw error;
  }

  const clip = loaded.animations.find((candidate) => candidate.name === 'Knee flexion')!;
  loaded.scene.updateMatrixWorld(true);
  const tissueCenter = loaded.scene.getObjectByName('KneePivot')!.getWorldPosition(new Vector3());
  tissueCenter.y -= 0.028;
  const joints = [makeJoint(loaded.scene, clip, tissueCenter), makeJoint(loaded.scene, clip, tissueCenter)];
  const camera = new OrthographicCamera(-.1, .1, .1, -.1, 0.05, 10);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false;
  controls.enableDamping = false;
  controls.rotateSpeed = 0.7;
  controls.minZoom = 0.55;
  controls.maxZoom = 2.8;
  controls.target.copy(TARGET);

  const layers: Record<TissueLayer, boolean> = { bone: true, cartilage: true, menisci: true, ligaments: true, muscles: false, loads: true };
  let frames: (TissueFrame | null)[] = [null, null];
  let activeSide: Side = 'right';
  let orientation: TissueView = 'anterior';
  let frameRequest = 0;
  let disposed = false;
  let width = 0;
  let height = 0;
  let pixelRatio = 0;

  function render() {
    frameRequest = 0;
    if (disposed) return;
    const nextWidth = Math.floor(host.clientWidth);
    const nextHeight = Math.floor(host.clientHeight);
    if (nextWidth <= 0 || nextHeight <= 0) return;
    const nextRatio = Math.min(window.devicePixelRatio || 1, 2);
    if (nextWidth !== width || nextHeight !== height || nextRatio !== pixelRatio) {
      width = nextWidth;
      height = nextHeight;
      pixelRatio = nextRatio;
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(width, height, false);
    }

    const active = joints.filter((_, index) => frames[index] != null);
    const stacked = width < TISSUE_STACK_WIDTH;
    host.dataset.tissueLayout = active.length < 2 ? 'single' : stacked ? 'stacked' : 'paired';
    renderer.setScissorTest(false);
    renderer.clear();
    if (!active.length) return;
    renderer.setScissorTest(true);
    for (const [index, joint] of active.entries()) {
      const viewportWidth = active.length === 1 || stacked ? width : Math.floor(width / 2);
      const viewportHeight = active.length === 1 || !stacked ? height : Math.floor(height / 2);
      const x = !stacked && active.length === 2 ? index * viewportWidth : 0;
      const y = stacked && active.length === 2 ? (1 - index) * viewportHeight : 0;
      const w = x === 0 && !stacked && active.length === 2 ? viewportWidth : width - x;
      const h = y === 0 && stacked && active.length === 2 ? viewportHeight : height - y;
      const aspect = w / h;
      const halfHeight = Math.max(VIEW_HEIGHT_M, VIEW_MIN_WIDTH_M / aspect) / 2;
      camera.left = -halfHeight * aspect;
      camera.right = halfHeight * aspect;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.updateProjectionMatrix();
      renderer.setViewport(x, y, w, h);
      renderer.setScissor(x, y, w, h);
      renderer.render(joint.scene, camera);
    }
  }

  function requestRender() {
    if (!disposed && !frameRequest) frameRequest = requestAnimationFrame(render);
  }

  function setView(view: TissueView) {
    orientation = view;
    camera.up.copy(Y_UP);
    camera.position.set(...VIEWS[view]);
    // Both slots compare the same participant. Anatomical presets follow the side.
    if (activeSide === 'left' && (view === 'medial' || view === 'lateral')) camera.position.x *= -1;
    controls.target.copy(TARGET);
    controls.update();
    requestRender();
  }

  function setLayers(next: TissueLayers) {
    Object.assign(layers, next);
    for (const { structures, colors } of joints) {
      for (const [mesh, layer] of structures) mesh.visible = layers[layer];
      colors.visible.value = Number(layers.loads);
    }
    requestRender();
  }

  function zoomBy(factor: number) {
    camera.zoom = Math.max(controls.minZoom, Math.min(controls.maxZoom, camera.zoom * factor));
    camera.updateProjectionMatrix();
    controls.update();
    requestRender();
  }

  function setFrames(ordinary: TissueFrame | null, selected: TissueFrame | null) {
    frames = [ordinary, selected];
    for (const [index, joint] of joints.entries()) {
      const frame = frames[index];
      if (!frame) continue;
      const degrees = frame.flexionDeg == null || !Number.isFinite(frame.flexionDeg) ? REFERENCE_FLEXION_DEG : frame.flexionDeg;
      joint.pose.time = clip.duration * Math.max(0, Math.min(1, degrees / 120));
      joint.mixer.update(0);
      joint.group.scale.x = frame.side === 'left' ? -1 : 1;
      assignLoadColor(frame.medialBw, joint.colors.medial);
      assignLoadColor(frame.lateralBw, joint.colors.lateral);
    }
    const side = (ordinary ?? selected)?.side ?? 'right';
    if (side !== activeSide) {
      activeSide = side;
      if (orientation === 'medial' || orientation === 'lateral') setView(orientation);
    }
    requestRender();
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === '+' || event.key === '=') {
      zoomBy(1.15);
    } else if (event.key === '-' || event.key === '_') {
      zoomBy(1 / 1.15);
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

  const resizeObserver = new ResizeObserver(requestRender);
  resizeObserver.observe(host);
  controls.addEventListener('change', requestRender);
  canvas.addEventListener('keydown', onKey);
  setView('anterior');
  setLayers(layers);

  return {
    setFrames, setLayers, setView, zoomBy,
    reset() {
      camera.zoom = 1;
      setView('anterior');
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(frameRequest);
      resizeObserver.disconnect();
      controls.dispose();
      canvas.removeEventListener('keydown', onKey);
      const meshGeometry = new Set<Mesh['geometry']>();
      const sourceMaterials = new Set<MeshStandardMaterial>();
      loaded.scene.traverse((object: Object3D) => {
        if (!(object as Mesh).isMesh) return;
        const mesh = object as Mesh;
        meshGeometry.add(mesh.geometry);
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const material of materials) sourceMaterials.add(material as MeshStandardMaterial);
      });
      for (const { group, mixer, materials } of joints) {
        mixer.stopAllAction();
        mixer.uncacheRoot(group.children[0]);
        for (const material of materials) material.dispose();
      }
      meshGeometry.forEach((geometry) => geometry.dispose());
      sourceMaterials.forEach((material) => material.dispose());
      renderer.dispose();
      canvas.remove();
    },
  };
}
