import {
  ACESFilmicToneMapping, AnimationMixer, ArrowHelper, Box3, DirectionalLight, HemisphereLight, MathUtils,
  PerspectiveCamera, Raycaster, Scene, Vector2, Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { arm, legs } from './regions.js';

const byId = (id) => document.getElementById(id);
const params = new URLSearchParams(window.location.search);
const legAvailable = Boolean(legs.modelUrl || byId(legs.embeddedModelId));
const requestedRegion = params.get('region') ?? document.body.dataset.defaultRegion;
const region = legAvailable && requestedRegion === 'legs' ? legs : arm;
const isLeg = region === legs;
let isLeft = isLeg && params.get('side') === 'left';
const canvas = byId('anatomy-canvas');
const status = byId('status');
const angle = byId('angle');
const load = byId('load');
const play = byId('play');
const isolate = byId('isolate');
const structurePicker = byId('structure-picker');
const nerveSelect = byId('nerve-select');
const nerveDescription = byId('nerve-description');
const nerveSource = byId('nerve-source');
const detailButton = byId('elbow-detail');
const download = byId('download-model');
const layers = Object.fromEntries(
  ['bones', 'muscles', 'tendons', 'nerves', 'ligaments', 'cartilage', 'fascia']
    .map((name) => [name, byId('layer-' + name)]),
);

const cameraViews = {
  anterior: new Vector3(-0.52, 0.16, 1),
  medial: new Vector3(1, 0.13, 0.27),
  lateral: new Vector3(-1, 0.1, 0.26),
  posterior: new Vector3(-0.38, 0.15, -1),
};
const nerveReferences = region.nerves;
const camera = new PerspectiveCamera(31, 1, 0.01, 20);
const scene = new Scene();
let renderer;
let controls;
let mixer;
let flexionAction;
let flexionDuration;
let loadCenter;
let jointPivot;
let anatomy;
const motionBounds = new Box3();
let selected;
let playingSince;
let currentView = 'medial';
let jointDetail = false;
let pointerStart;
let renderNeeded = true;
const structures = [];
const pickable = [];
const selectionRay = new Raycaster();
const forceArrow = new ArrowHelper(new Vector3(0, -1, 0), new Vector3(), 0.09, 0x247372, 0.023, 0.015);
forceArrow.visible = false;
scene.add(forceArrow);

function configurePage() {
  document.title = region.title + ' · Anatomy Atlas';
  byId('region-leg').hidden = !legAvailable;
  for (const [id, active] of [['region-arm', !isLeg], ['region-leg', isLeg]]) {
    if (active) byId(id).setAttribute('aria-current', 'page');
    else byId(id).removeAttribute('aria-current');
  }
  byId('region-title').textContent = region.title;
  byId('side-controls').hidden = !isLeg;
  configureSide();
  detailButton.textContent = region.joint + ' detail';
  angle.max = String(region.maxAngle);
  angle.value = String(region.startAngle);
  byId('angle-max').textContent = region.maxAngle + '°';
  byId('joint-label').textContent = region.joint + ' flexion';
  byId('load-label').textContent = region.loadSegment + ' load';
  byId('torque-label').textContent = region.joint + ' torque';
  byId('moment-arm-label').textContent = region.muscleLabel + ' moment arm';
  byId('path-length-label').textContent = region.muscleLabel + ' path length';
  byId('control-note').textContent = region.mechanicsNote;
  for (const layer of ['cartilage', 'fascia']) byId(layer + '-layer-row').hidden = !isLeg;
  for (const name of Object.keys(nerveReferences)) {
    if (!Array.from(nerveSelect.options).some((item) => item.value === name)) {
      nerveSelect.add(new Option(name, name));
    }
  }
  for (const option of Array.from(nerveSelect.options)) {
    if (option.value && !nerveReferences[option.value]) option.remove();
  }
  stopPlayback();
  if (isLeg) {
    byId('atlas-credit-text').textContent = 'The 3D model adapts the Open3D Project’s right lower limb. '
      + 'This version selects and recolors structures and adds a knee rig. The adapted mesh is available under';
    byId('atlas-archive').href = region.atlasArchive;
    byId('opensim-credit-text').textContent = 'Knee motion and rectus femoris reference geometry use the Rajagopal et al. OpenSim model.';
    byId('opensim-reference').href = region.modelReference;
    byId('opensim-reference').textContent = 'Rajagopal2016 reference model';
    byId('opensim-license').href = 'https://doi.org/10.1109/TBME.2016.2586891';
    byId('opensim-license').textContent = 'Model publication';
  }
}

function configureSide() {
  const side = isLeft ? 'Left' : 'Right';
  const joint = region.joint.toLowerCase();
  byId('region-subtitle').textContent = side + ' ' + joint + ' flexion & extension' + (isLeft ? ' · mirrored' : '');
  canvas.setAttribute('aria-label', side + ' ' + region.title.toLowerCase()
    + ' model. Drag to rotate, scroll or pinch to zoom, and select a structure.');
  if (isLeg) {
    byId('side-right').setAttribute('aria-pressed', String(!isLeft));
    byId('side-left').setAttribute('aria-pressed', String(isLeft));
    byId('plate-note').textContent = 'Knee motion with the hip and ankle fixed. Nerve paths are illustrative.'
      + (isLeft ? ' The GLB contains the original right-side rig.' : '');
  }
}

function setupScene() {
  renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.085;
  controls.minDistance = 0.22;
  controls.maxDistance = 5;

  scene.add(new HemisphereLight(0xfff5e8, 0x858594, 2.0));
  const key = new DirectionalLight(0xfff3e4, 2.1);
  key.position.set(-2, 3, 4);
  const fill = new DirectionalLight(0xe8eefb, 1.35);
  fill.position.set(3, 1, -2);
  scene.add(key, fill);

  const resizeCanvas = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    renderNeeded = true;
    if (anatomy && currentView) setView(currentView, playingSince !== undefined);
  };
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(canvas);
}

function readModelBytes(encoded) {
  const binary = atob(encoded.trim());
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function openModel() {
  const loader = new GLTFLoader();
  const bundled = byId(region.embeddedModelId);
  download.download = region.filename;
  download.setAttribute('aria-label', isLeg ? 'Download right-leg 3D model' : 'Download 3D model');
  if (bundled) {
    const bytes = readModelBytes(bundled.textContent);
    download.href = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }));
    return loader.parseAsync(bytes.buffer, '');
  }
  download.href = region.modelUrl;
  return loader.loadAsync(region.modelUrl);
}

function indexStructures(root) {
  root.traverse((object) => {
    const { layer, label: sourceLabel } = object.userData;
    if (!layers[layer] || !sourceLabel) return;
    const label = sourceLabel.replace('Musculocutaneus', 'Musculocutaneous');
    const meshes = [];
    object.traverse((part) => {
      if (!part.isMesh) return;
      part.userData.structureId = object.uuid;
      meshes.push({ mesh: part, material: part.material });
      pickable.push(part);
    });
    if (meshes.length) structures.push({ object, layer, label, meshes });
  });
  if (structurePicker) {
    for (const { object, label, layer } of structures.toSorted((a, b) => a.label.localeCompare(b.label))) {
      const option = new Option(label, object.uuid);
      option.dataset.layer = layer;
      structurePicker.append(option);
    }
  }
}

function changeVisibility() {
  if (selected && !layers[selected.layer].checked) selectStructure();
  for (const structure of structures) {
    structure.object.visible = layers[structure.layer].checked
      && (!isolate.checked || !selected || selected === structure);
  }
  updateForceArrow();
  renderNeeded = true;
}

function reapplyMaterial(part) {
  const originals = Array.isArray(part.material) ? part.material : [part.material];
  const highlighted = originals.map((material) => {
    const clone = material.clone();
    clone.emissive?.setHex(0xcc761f);
    clone.emissiveIntensity = 0.22;
    return clone;
  });
  part.mesh.material = Array.isArray(part.material) ? highlighted : highlighted[0];
}

function selectStructure(next) {
  if (selected === next) return;
  for (const part of selected?.meshes ?? []) {
    const highlights = Array.isArray(part.mesh.material) ? part.mesh.material : [part.mesh.material];
    for (const material of highlights) material.dispose();
    part.mesh.material = part.material;
  }
  selected = next;
  for (const part of next?.meshes ?? []) reapplyMaterial(part);
  byId('selection').textContent = next?.label ?? 'All visible structures';
  if (structurePicker) structurePicker.value = next?.object.uuid ?? '';
  const nerve = next?.layer === 'nerves'
    ? Object.keys(nerveReferences).find((name) => next.label.startsWith(name))
    : undefined;
  nerveSelect.value = nerve ?? '';
  const reference = nerveReferences[nerve];
  nerveDescription.textContent = reference?.description ?? '';
  nerveDescription.hidden = !reference;
  nerveSource.hidden = !reference;
  if (reference) nerveSource.href = reference.url;
  else nerveSource.removeAttribute('href');
  changeVisibility();
}

function pickStructure(event) {
  if (!pointerStart || pointerStart.distanceTo(new Vector2(event.clientX, event.clientY)) > 5) return;
  const rect = canvas.getBoundingClientRect();
  const point = new Vector2(
    2 * (event.clientX - rect.left) / rect.width - 1,
    1 - 2 * (event.clientY - rect.top) / rect.height,
  );
  selectionRay.setFromCamera(point, camera);
  const visible = pickable.filter((mesh) => structures.some((structure) => (
    structure.object.uuid === mesh.userData.structureId && structure.object.visible
  )));
  for (const mesh of visible) {
    if (mesh.isSkinnedMesh) mesh.computeBoundingSphere();
  }
  const closest = selectionRay.intersectObjects(visible, false)[0];
  selectStructure(structures.find(({ object }) => object.uuid === closest?.object.userData.structureId));
}

function setView(view, fitMotion = false) {
  if (!anatomy) return;
  currentView = view;
  const bounds = jointDetail
    ? new Box3().setFromCenterAndSize(jointPivot.getWorldPosition(new Vector3()), new Vector3(...region.detailSize))
    : fitMotion ? motionBounds : new Box3().setFromObject(anatomy, true);
  const center = bounds.getCenter(new Vector3());
  const eye = cameraViews[view].clone().normalize();
  if (isLeft) eye.x *= -1;
  const right = new Vector3().crossVectors(camera.up, eye).normalize();
  const up = new Vector3().crossVectors(eye, right);
  const half = bounds.getSize(new Vector3()).multiplyScalar(0.5);
  const project = ({ x, y, z }) => Math.abs(x) * half.x + Math.abs(y) * half.y + Math.abs(z) * half.z;
  const fov = Math.tan(MathUtils.degToRad(camera.fov / 2));
  const projectedSpan = Math.max(project(up), project(right) / camera.aspect);
  const distance = 1.17 * projectedSpan / fov + 0.62 * project(eye);
  camera.position.copy(center).addScaledVector(eye, distance);
  controls.target.copy(center);
  controls.update();
  renderNeeded = true;
  for (const name of Object.keys(cameraViews)) byId('view-' + name).setAttribute('aria-pressed', String(name === view));
}

function renderNumbers(degrees) {
  const data = region.mechanics?.samples[Math.round(degrees)];
  const flexionRadians = MathUtils.degToRad(degrees);
  const torque = Number(load.value) * 0.45359237 * 9.80665 * region.leverM * Math.sin(flexionRadians);
  byId('torque').textContent = torque.toFixed(2) + ' N·m';
  byId('moment-arm').textContent = data ? (data.momentArmM[region.muscle] * 1000).toFixed(1) + ' mm' : '—';
  byId('path-length').textContent = data ? (data.lengthM[region.muscle] * 100).toFixed(1) + ' cm' : '—';
  updateForceArrow();
  renderNeeded = true;
}

function updateForceArrow() {
  forceArrow.visible = Boolean(loadCenter && Number(load.value) && !jointDetail && !(isolate.checked && selected));
  if (forceArrow.visible) {
    loadCenter.getWorldPosition(forceArrow.position);
    forceArrow.setLength(0.055 + Number(load.value) * 0.012, 0.023, 0.015);
  }
}

function setAngle(value) {
  const degrees = MathUtils.clamp(value, 0, region.maxAngle);
  const sliderValue = Math.round(degrees);
  angle.value = String(sliderValue);
  byId('angle-value').textContent = sliderValue + '°';
  poseAnatomy(degrees);
  renderNumbers(degrees);
}

function poseAnatomy(degrees) {
  flexionAction.time = flexionDuration * degrees / region.maxAngle;
  mixer.update(0);
}

function stopPlayback() {
  playingSince = undefined;
  play.setAttribute('aria-pressed', 'false');
  play.textContent = 'Play ' + region.motion;
}

function frame(time) {
  if (playingSince !== undefined) {
    const phase = (time - playingSince) / 950;
    setAngle(region.maxAngle / 2 * (1 - Math.cos(phase)));
  }
  controls.update();
  if (renderNeeded) {
    renderer.render(scene, camera);
    renderNeeded = false;
  }
}

function installControls() {
  for (const side of ['right', 'left']) {
    byId('side-' + side).addEventListener('click', () => {
      if (!isLeg || (side === 'left') === isLeft) return;
      isLeft = side === 'left';
      if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
        const target = new URL(window.location.href);
        target.searchParams.set('region', 'legs');
        target.searchParams.set('side', side);
        window.history.replaceState(window.history.state, '', target);
      }
      anatomy.scale.x *= -1;
      [motionBounds.min.x, motionBounds.max.x] = [-motionBounds.max.x, -motionBounds.min.x];
      configureSide();
      setView(currentView ?? 'medial', playingSince !== undefined);
      updateForceArrow();
    });
  }
  canvas.addEventListener('pointerdown', (event) => {
    pointerStart = new Vector2(event.clientX, event.clientY);
  });
  canvas.addEventListener('pointerup', pickStructure);
  canvas.addEventListener('pointercancel', () => { pointerStart = undefined; });
  for (const checkbox of Object.values(layers)) checkbox.addEventListener('change', changeVisibility);
  isolate.addEventListener('change', changeVisibility);
  structurePicker?.addEventListener('change', () => {
    const next = structures.find(({ object }) => object.uuid === structurePicker.value);
    if (next) layers[next.layer].checked = true;
    selectStructure(next);
  });
  nerveSelect.addEventListener('change', () => {
    const next = structures.find(({ label, layer }) => layer === 'nerves' && label === nerveSelect.value);
    if (next) layers.nerves.checked = true;
    selectStructure(next);
  });
  angle.addEventListener('input', () => {
    stopPlayback();
    setAngle(Number(angle.value));
  });
  angle.addEventListener('change', () => {
    if (currentView) setView(currentView);
  });
  load.addEventListener('change', () => renderNumbers(Number(angle.value)));
  play.addEventListener('click', () => {
    if (playingSince !== undefined) {
      stopPlayback();
      return;
    }
    const currentPhase = Math.acos(1 - 2 * Number(angle.value) / region.maxAngle);
    playingSince = performance.now() - 950 * currentPhase;
    if (currentView) setView(currentView, true);
    play.setAttribute('aria-pressed', 'true');
    play.textContent = 'Pause ' + region.motion;
  });
  for (const view of Object.keys(cameraViews)) byId('view-' + view).addEventListener('click', () => setView(view));
  detailButton.addEventListener('click', () => {
    jointDetail = !jointDetail;
    detailButton.setAttribute('aria-pressed', String(jointDetail));
    setView(currentView ?? 'lateral');
    updateForceArrow();
  });
  byId('reset-view').addEventListener('click', () => {
    jointDetail = false;
    detailButton.setAttribute('aria-pressed', 'false');
    setView('medial');
    updateForceArrow();
  });
  const pose = byId('preset-select');
  pose?.addEventListener('change', () => {
    if (!pose.value) return;
    stopPlayback();
    setAngle(Number(pose.value));
    pose.value = '';
  });
  controls.addEventListener('start', () => {
    if (currentView) byId('view-' + currentView).setAttribute('aria-pressed', 'false');
    currentView = undefined;
  });
  controls.addEventListener('change', () => { renderNeeded = true; });
}

async function main() {
  try {
    configurePage();
    setupScene();
    const { scene: root, animations } = await openModel();
    const flexion = animations.find((clip) => clip.name === region.animation);
    jointPivot = root.getObjectByName(region.pivot);
    if (!flexion || !jointPivot) {
      throw new Error('The model is missing its joint or flexion animation.');
    }
    mixer = new AnimationMixer(root);
    flexionAction = mixer.clipAction(flexion).play();
    flexionAction.paused = true;
    flexionDuration = flexion.duration;
    loadCenter = root.getObjectByName(region.loadCenter);
    if (isLeft) root.scale.x *= -1;
    indexStructures(root);
    if (!structures.length) throw new Error('The model has no selectable anatomy.');
    anatomy = root;
    scene.add(root);
    for (const degrees of [0, 30, 60, 90, region.maxAngle]) {
      poseAnatomy(degrees);
      motionBounds.expandByObject(root, true);
    }
    installControls();
    changeVisibility();
    setAngle(Number(angle.value));
    setView('medial');
    renderer.setAnimationLoop(frame);
    status.hidden = true;
    canvas.dataset.ready = 'true';
  } catch (error) {
    console.error(error);
    status.textContent = 'The 3D scene could not open. Try Chrome with hardware acceleration, or open the GLB in a 3D application.';
  }
}

main();
