import {
  ACESFilmicToneMapping,
  BufferGeometry,
  DirectionalLight,
  GridHelper,
  Group,
  HemisphereLight,
  LineBasicMaterial,
  LineLoop,
  OrthographicCamera,
  Quaternion,
  Scene,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { createWalkingAtlas } from '../knee-loading/body-layers.ts';
import { createPoseSampler } from './render/poses.js';
import { createLandmarkView } from './render/landmarks.js';
import { createMassMarkerOverlay } from './render/mass-markers.js';

const atlasURL = new URL('../../knee-loading/assets/walking-atlas.glb', import.meta.url).href;
const opticalURL = new URL('../../knee-loading/data/reference-walks.json', import.meta.url).href;
const implantURL = new URL('../../knee-loading/data/whole-body-motion.json', import.meta.url).href;
const directions = { front: [1, .045, 0], oblique: [.93, .13, .57], side: [.04, .07, 1], back: [-1, .065, .03] };
let atlasPromise;
let samplerPromise;
async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Gait archive unavailable (' + response.status + ')');
  return response.json();
}
function getSampler() {
  samplerPromise ??= Promise.all([loadJson(opticalURL), loadJson(implantURL)])
    .then(([optical, implant]) => createPoseSampler(optical, implant)).catch((error) => {
      samplerPromise = undefined;
      throw error;
    });
  return samplerPromise;
}
function getTemplate() {
  atlasPromise ??= new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(atlasURL)
    .then((gltf) => gltf.scene).catch((error) => {
      atlasPromise = undefined;
      throw error;
    });
  return atlasPromise;
}
function makeEnvironment(scene) {
  scene.add(new HemisphereLight(0xfffcf4, 0xd3c6ae, 1.16));
  for (
    const [x, y, z, color, intensity] of [[3, 4, 2, 0xfff3e6, 2.05], [-2, 2, -2, 0xf7efe3, 1.04], [
      1,
      1,
      -3,
      0xe5e9e8,
      .65,
    ]]
  ) {
    const lamp = new DirectionalLight(color, intensity);
    lamp.position.set(x, y, z);
    scene.add(lamp);
  }
  const ground = new Group();
  const grid = new GridHelper(3.2, 16, 0x8399a3, 0x677b84);
  grid.material.transparent = true;
  grid.material.opacity = .28;
  grid.material.depthWrite = false;
  grid.position.y = -.013;
  ground.add(grid);
  const radii = [.42, .70];
  for (const r of radii) {
    const vertices = Array.from(
      { length: 96 },
      (_, i) => new Vector3(r * Math.cos(i * Math.PI / 48), -.009, r * Math.sin(i * Math.PI / 48)),
    );
    const ring = new LineLoop(
      new BufferGeometry().setFromPoints(vertices),
      new LineBasicMaterial({ color: 0x92a1a7, transparent: true, opacity: .25, depthWrite: false }),
    );
    ground.add(ring);
  }
  scene.add(ground);
  return ground;
}
const guideLabels = {
  trunkLeanDeg: 'Trunk lean · dashed vertical',
  pelvisDropDeg: 'Pelvic drop · dashed level',
  hipFlexionDeg: 'Right hip flexion · dashed neutral',
  hipAdductionDeg: 'Right hip adduction · dashed neutral',
};

// Construct the solved angular coordinate in its anatomical plane. The neutral
// direction and solved direction share a real atlas landmark and physical length.
function guidePoints(frame, solution, signal) {
  const s = frame.segments;
  const point = (name) => new Vector3(...s[name].position);
  const radians = (value) => value * Math.PI / 180;
  const angle = radians(solution[signal]);
  const hip = point('femur');
  if (signal === 'trunkLeanDeg') {
    const origin = hip.lerp(point('femur_l'), .5);
    const length = point('humerus_r').lerp(point('humerus_l'), .5).distanceTo(origin);
    return [
      origin,
      origin.clone().add(new Vector3(0, length * Math.cos(angle), length * Math.sin(angle))),
      origin.clone().add(new Vector3(0, length * 1.1, 0)),
    ];
  }
  const pelvis = new Quaternion(...s.pelvis.rotation);
  if (signal === 'pelvisDropDeg') {
    // A horizontal transverse axis through the right hip makes pelvic drop
    // apparent from the front, including any residual recorded yaw.
    const across = point('femur_l').sub(hip);
    const width = across.length();
    across.y = 0;
    across.normalize();
    const zero = across.clone().multiplyScalar(width * 1.1).add(hip);
    const actual = across.multiplyScalar(width * Math.cos(angle)).add(hip);
    actual.y -= width * Math.sin(angle);
    return [hip, actual, zero];
  }
  const length = hip.distanceTo(point('tibia'));
  const neutral = new Vector3(0, -length, 0);
  const rightAxis = new Vector3(0, 0, 1);
  if (signal === 'hipAdductionDeg') {
    // OpenSim's hip order is flexion, adduction, then axial rotation.
    pelvis.multiply(new Quaternion().setFromAxisAngle(rightAxis, radians(solution.hipFlexionDeg ?? 0)));
  }
  const axis = signal === 'hipAdductionDeg' ? new Vector3(1, 0, 0) : rightAxis;
  const actual = neutral.clone().applyAxisAngle(axis, angle).applyQuaternion(pelvis).add(hip);
  const zero = neutral.multiplyScalar(1.1).applyQuaternion(pelvis).add(hip);
  return [hip, actual, zero];
}

function angleGuide(host, color) {
  const ns = 'http://www.w3.org/2000/svg';
  const drawing = document.createElementNS(ns, 'svg');
  drawing.classList.add('body-angle-guide');
  drawing.setAttribute('aria-hidden', 'true');
  drawing.style.cssText = 'position:absolute;top:0;pointer-events:none;overflow:hidden;height:100%';
  function add(tag, attributes) {
    const el = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
    drawing.append(el);
    return el;
  }
  const neutralLine = add('line', { stroke: '#e8ebeb', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
  const outline = add('line', { stroke: '#263139', 'stroke-width': 4.5, 'stroke-linecap': 'round' });
  const axis = add('line', { stroke: color, 'stroke-width': 2.5, 'stroke-linecap': 'round' });
  const pivotMark = add('circle', { r: 3.5, fill: color, stroke: '#293239', 'stroke-width': 1.4 });
  const tipMark = add('circle', { r: 2.6, fill: color, stroke: '#293239', 'stroke-width': 1.0 });
  host.append(drawing);
  return {
    show(frame, solution, signal, camera, x, width, height) {
      drawing.style.display = 'block';
      drawing.style.left = x + 'px';
      drawing.style.width = width + 'px';
      const [pivot, tip, straight] = guidePoints(frame, solution, signal).map((point) => {
        point.project(camera);
        return [(point.x + 1) * width * .5, (1 - point.y) * height * .5];
      });
      for (const [line, to] of [[neutralLine, straight], [outline, tip], [axis, tip]]) {
        line.setAttribute('x1', pivot[0]);
        line.setAttribute('y1', pivot[1]);
        line.setAttribute('x2', to[0]);
        line.setAttribute('y2', to[1]);
      }
      for (const [mark, point] of [[pivotMark, pivot], [tipMark, tip]]) {
        mark.setAttribute('cx', point[0]);
        mark.setAttribute('cy', point[1]);
      }
    },
    hide() {
      drawing.style.display = 'none';
    },
    dispose() {
      drawing.remove();
    },
  };
}
function comparisonLabels(host) {
  const rail = document.createElement('div');
  rail.className = 'body-stage-labels';
  rail.style.cssText =
    'position:absolute;inset:10px 0 auto;display:grid;grid-template-columns:1fr 1fr;pointer-events:none;z-index:1;text-align:left;font:600 10px/1.25 system-ui;letter-spacing:.12em;text-transform:uppercase';
  for (const [name, color] of [['Reference', '#afc9d9'], ['Adjusted', '#e3ab8a']]) {
    const wrap = document.createElement('div'), label = document.createElement('span');
    label.style.cssText = 'display:inline-block;padding:var(--marker-padding,4px 8px);color:' + color +
      ';background:#253036e6;border:1px solid ' + color + '47;border-radius:3px';
    wrap.style.paddingLeft = 'var(--marker-inset,10px)';
    label.textContent = name;
    wrap.append(label);
    rail.append(wrap);
  }
  host.append(rail);
  return rail;
}

/** Recorded whole-body motion or paired absolute virtual-patient coordinates. */
export async function createBodyStage(host, options = {}) {
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  const sampler = await getSampler();
  let comparison = Boolean(options.comparison),
    layer = options.layer ?? 'muscles',
    region = options.region ?? 'body';
  let view = options.view ?? 'oblique',
    input = { phase: options.phase ?? .24, clipId: options.clipId },
    active = true,
    disposed = false;
  let width = 0, height = 0, invalidPose = true, paintRequest = 0, ready = false;
  let renderer = null, fallback = null, camera = null, controls = null, scene = null, environment = null;
  let receivingOrbit = false;
  let template = null;
  const rigs = [], guides = [];
  const labels = comparisonLabels(host);
  labels.style.display = comparison ? 'grid' : 'none';
  const scaleKey = document.createElement('span');
  scaleKey.style.cssText =
    'position:absolute;bottom:7px;left:10px;pointer-events:none;color:#d3dcdf;font:10px/1.3 system-ui;background:#263138e8;padding:3px 6px;border-radius:2px';
  scaleKey.hidden = true;
  host.append(scaleKey);
  const frames = [];
  let massOverlay = null;
  function showMassMarkers(points, width, height) {
    if (!points || comparison) {
      massOverlay?.hide();
      return;
    }
    massOverlay ??= createMassMarkerOverlay(host);
    massOverlay.show(points, width, height);
  }
  function disposeGPU() {
    controls?.dispose();
    controls = null;
    for (const rig of rigs) {
      rig.dispose();
      rig.object.removeFromParent();
    }
    rigs.length = 0;
    guides.forEach((guide) => guide.dispose());
    guides.length = 0;
    environment?.traverse((node) => {
      node.geometry?.dispose();
      node.material?.dispose();
    });
    environment?.removeFromParent();
    environment = null;
    renderer?.dispose();
    renderer?.forceContextLoss();
    renderer?.domElement.remove();
    renderer = null;
  }
  function useFallback() {
    scaleKey.hidden = true;
    disposeGPU();
    if (!fallback) fallback = createLandmarkView(host, options.guideColor);
    host.dataset.bodyRenderer = 'landmarks';
    queuePaint();
  }
  function applyLayer(rig) {
    rig.setLayers(layer === 'muscles', layer !== 'surface', layer === 'surface');
  }
  function addRig() {
    const rig = createWalkingAtlas(template, { wireframe: options.wireframe });
    applyLayer(rig);
    scene.add(rig.object);
    rigs.push(rig);
    guides.push(angleGuide(host, options.guideColor ?? (rigs.length === 1 ? '#a1c3da' : '#f0b593')));
  }
  function focalPoint(frame) {
    if (region === 'hip' || region === 'knee') {
      const center = frame.segments[region === 'hip' ? 'femur' : 'tibia'].position;
      return new Vector3(center[0], center[1], center[2]);
    }
    return new Vector3(0, (input.heightM || 1.64) * .55, 0);
  }
  function aim() {
    if (!camera || !controls) return;
    const target = frames.length ? focalPoint(frames[0]) : new Vector3(0, .84, 0);
    controls.target.copy(target);
    camera.position.copy(target).add(
      new Vector3(...(directions[view] ?? directions.oblique)).normalize().multiplyScalar(4),
    );
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.update();
    queuePaint();
  }
  function size(width, height) {
    if (!renderer || !width || !height) return;
    renderer.setSize(width, height, false);
    const slotWidth = width / (comparison ? 2 : 1);
    const stature = input.heightM || 1.65;
    // Both viewports use one camera, so body size and perspective match exactly.
    const span = Math.max(
      stature * (region === 'body' ? 1.25 : .56),
      (region === 'body' ? .86 : .55) * height / slotWidth,
    );
    camera.top = span / 2;
    camera.bottom = -span / 2;
    camera.right = span * .5 * slotWidth / height;
    camera.left = -camera.right;
    camera.updateProjectionMatrix();
  }
  function pose() {
    const common = {
      phase: input.phase ?? 0,
      task: input.task ?? 'walk',
      clipId: input.clipId ?? options.clipId,
      profileHeightM: input.heightM,
      fixedArmSegmentsInTorso: input.fixedArmSegmentsInTorso,
      massMarkers: input.massMarkers,
    };
    frames[0] = sampler.sample({ ...common, solution: input.baseline });
    if (comparison) frames[1] = sampler.sample({ ...common, solution: input.changed ?? input.baseline });
    else frames.length = 1;
    for (let i = 0; i < frames.length && i < rigs.length; i++) {
      rigs[i].setFrame(frames[i]);
    }
    invalidPose = false;
  }
  function paint() {
    if (!active || disposed || !ready) return;
    const w = host.clientWidth, h = host.clientHeight;
    if (w < 2 || h < 2) return;
    if (w !== width || h !== height) {
      width = w;
      height = h;
      size(w, h);
      const compact = comparison && w / 2 < 225;
      labels.style.top = compact ? '4px' : '10px';
      labels.style.fontSize = compact ? '9px' : '10px';
      labels.style.letterSpacing = compact ? '.06em' : '.12em';
      labels.style.setProperty('--marker-padding', compact ? '2px 4px' : '4px 8px');
      labels.style.setProperty('--marker-inset', compact ? '4px' : '10px');
    }
    if (invalidPose) pose();
    if (fallback) {
      const points = fallback.draw(frames, view, region, input.heightM || 1.62);
      showMassMarkers(points, w, h);
      return;
    }
    const count = frames.length;
    const target = focalPoint(frames[0]);
    camera.position.add(target.clone().sub(controls.target));
    controls.target.copy(target);
    controls.update();
    renderer.setScissorTest(true);
    const signal = input.signal ?? 'trunkLeanDeg';
    const showAngle = (comparison || options.guideColor) && Object.hasOwn(guideLabels, signal) &&
      Number.isFinite(input.baseline?.[signal]) &&
      (!comparison || Number.isFinite(input.changed?.[signal])) &&
      (region === 'body' || (region === 'hip' && signal !== 'trunkLeanDeg'));
    if (showAngle && scaleKey.textContent !== guideLabels[signal]) scaleKey.textContent = guideLabels[signal];
    scaleKey.hidden = !showAngle;
    if (!showAngle) guides.forEach((guide) => guide.hide());
    for (let i = 0; i < count; i++) {
      const x = Math.round(i * w / count), right = Math.round((i + 1) * w / count);
      rigs.forEach((rig, j) => {
        rig.object.visible = i === j;
      });
      renderer.setViewport(x, 0, right - x, h);
      renderer.setScissor(x, 0, right - x, h);
      renderer.render(scene, camera);
      if (showAngle) {
        guides[i].show(frames[i], i === 0 ? input.baseline : input.changed, signal, camera, x, right - x, h);
      }
    }
    const markers = frames[0].massMarkers;
    if (markers && !comparison) {
      function project(xyz) {
        const point = new Vector3(...xyz).project(camera);
        return [(point.x + 1) * w * .5, (1 - point.y) * h * .5];
      }
      showMassMarkers({ body: project(markers.body), pelvis: project(markers.pelvis) }, w, h);
    } else {
      massOverlay?.hide();
    }
  }
  function queuePaint() {
    if (!paintRequest && !disposed && active && ready) {
      paintRequest = requestAnimationFrame(() => {
        paintRequest = 0;
        paint();
      });
    }
  }
  try {
    renderer = new WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'default',
      preserveDrawingBuffer: false,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.65));
    renderer.setClearColor(0x283339, 1);
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = .98;
    const canvas = renderer.domElement;
    canvas.className = 'body-stage-canvas';
    canvas.style.cssText = 'display:block;width:100%;height:100%;touch-action:none';
    canvas.setAttribute('aria-label', 'Interactive anatomical atlas. Drag to rotate and scroll to zoom.');
    canvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      if (!disposed && renderer) useFallback();
    }, { once: true });
    host.prepend(canvas);
    host.dataset.bodyRenderer = 'atlas';
    camera = new OrthographicCamera(-1, 1, 1, -1, .05, 25);
    controls = new OrbitControls(camera, canvas);
    controls.enablePan = false;
    controls.enableDamping = false;
    controls.minPolarAngle = Math.PI * .17;
    controls.maxPolarAngle = Math.PI * .73;
    controls.minZoom = .65;
    controls.maxZoom = 3.2;
    controls.addEventListener('change', () => {
      queuePaint();
      if (options.onOrbit && !receivingOrbit && ready) {
        options.onOrbit({
          offset: camera.position.clone().sub(controls.target).toArray(),
          zoom: camera.zoom,
        });
      }
    });
    scene = new Scene();
    environment = makeEnvironment(scene);
    template = await getTemplate();
    if (renderer) {
      addRig();
      if (comparison) addRig();
    }
  } catch (error) {
    console.warn('Displaying the joint-centre projection:', error.message);
    useFallback();
  }
  const resizeObserver = new ResizeObserver(queuePaint);
  resizeObserver.observe(host);
  ready = true;
  pose();
  aim();
  paint();
  return {
    get mode() {
      return renderer ? 'atlas' : 'landmarks';
    },
    setPose(next) {
      const resizeNeeded = input.heightM !== next.heightM;
      input = next;
      invalidPose = true;
      if (resizeNeeded) width = 0;
      queuePaint();
    },
    setView(next) {
      view = Object.hasOwn(directions, next) ? next : 'oblique';
      aim();
      queuePaint();
    },
    setOrbit({ offset, zoom }) {
      if (!controls) return;
      receivingOrbit = true;
      try {
        camera.position.fromArray(offset).add(controls.target);
        camera.zoom = zoom;
        camera.updateProjectionMatrix();
        controls.update();
      } finally {
        receivingOrbit = false;
      }
      queuePaint();
    },
    setRegion(next) {
      region = ['hip', 'knee'].includes(next) ? next : 'body';
      width = 0;
      aim();
      queuePaint();
    },
    setLayer(next) {
      layer = ['bones', 'surface'].includes(next) ? next : 'muscles';
      for (const rig of rigs) applyLayer(rig);
      queuePaint();
    },
    setComparison(next) {
      comparison = Boolean(next);
      if (comparison && rigs.length === 1) addRig();
      labels.style.display = comparison ? 'grid' : 'none';
      invalidPose = true;
      width = 0;
      queuePaint();
    },
    setActive(next) {
      active = Boolean(next);
      if (active) queuePaint();
    },
    reset() {
      aim();
    },
    dispose() {
      disposed = true;
      active = false;
      cancelAnimationFrame(paintRequest);
      resizeObserver.disconnect();
      labels.remove();
      scaleKey.remove();
      massOverlay?.dispose();
      disposeGPU();
      fallback?.dispose();
    },
  };
}
