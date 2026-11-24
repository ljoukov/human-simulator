import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const boneUrl = new URL('../../hip-study/assets/body.glb', import.meta.url).href;
const muscleUrl = new URL('../../hip-study/assets/body-soft-tissue.glb', import.meta.url).href;
const vector = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const midpoint = (a, b) => a.clone().lerp(b, .5);
const difference = (a, b) => a.clone().sub(b);
const rot = new THREE.Matrix4().makeRotationY(Math.PI / 2);
const identity = new THREE.Matrix4();

// Landmarks in the bundled BodyParts3D atlas. Convert the atlas's anterior +Z
// to the walking data's anterior +X and anatomical right +Z. Values are in
// OrganUI's native normalized units; the segment fits below recover metres.
const landmark = (lateral, vertical, anterior) => vector(anterior, vertical, -lateral);
const atlas = { head: landmark(0, 1.84, -.025) };
for (const [side, sign] of [['left', 1], ['right', -1]]) {
  const p = (name, x, y, z) => { atlas[side + name] = landmark(x * sign, y, z); };
  p('Shoulder', .400, 1.270, -.085);
  p('Elbow', .540, .575, -.105);
  p('Wrist', .623, .005, .002);
  p('Hand', .655, -.190, .056);
  p('Hip', .203, .007, -.042);
  p('Knee', .202, -1.025, -.090);
  p('Ankle', .190, -1.895, -.110);
  p('Heel', .198, -1.982, -.232);
  p('Toe', .227, -2.010, .174);
}

function axes(upHint, rightHint) {
  const y = upHint.clone().normalize();
  const x = vector().crossVectors(y, rightHint);
  if (x.lengthSq() < 1e-8) x.crossVectors(y, vector(1, 0, 0));
  x.normalize();
  return new THREE.Matrix4().makeBasis(x, y, vector().crossVectors(x, y).normalize());
}

function poseFrames(p) {
  const hip = midpoint(p.leftHip, p.rightHip);
  const shoulder = midpoint(p.leftShoulder, p.rightShoulder);
  const up = difference(shoulder, hip);
  const hipsRight = difference(p.rightHip, p.leftHip);
  const shoulderRight = difference(p.rightShoulder, p.leftShoulder);
  const shoulderToHead = difference(p.head, shoulder);
  const legLength = (p.leftHip.distanceTo(p.leftKnee) + p.leftKnee.distanceTo(p.leftAnkle)
    + p.rightHip.distanceTo(p.rightKnee) + p.rightKnee.distanceTo(p.rightAnkle)) * .5;
  const fields = {};
  function frame(id, anchor, basis, height, width = height) {
    fields[id] = { anchor, basis, height, width };
  }
  // Width independently aligns both femoral heads with the acetabula.
  frame('pelvis', hip, axes(vector(0, 1, 0).addScaledVector(hipsRight, -hipsRight.y / Math.max(1e-9, hipsRight.lengthSq())), hipsRight),
    up.length(), hipsRight.length());
  frame('thorax', shoulder, axes(up, shoulderRight), up.length(), shoulderRight.length());
  frame('head', p.head, axes(shoulderToHead, shoulderRight), legLength);
  const pair = (id, from, to, rightHint, width) => {
    const axis = difference(from, to);
    frame(id, from, axes(axis, rightHint), axis.length(), width);
  };
  for (const side of ['left', 'right']) {
    const joint = name => p[side + name];
    pair(side + 'Thigh', joint('Hip'), joint('Knee'), hipsRight);
    pair(side + 'Shin', joint('Knee'), joint('Ankle'), hipsRight);
    pair(side + 'Arm', joint('Shoulder'), joint('Elbow'), shoulderRight);
    pair(side + 'Forearm', joint('Elbow'), joint('Wrist'), shoulderRight);
    pair(side + 'Hand', joint('Wrist'), joint('Hand'), shoulderRight, joint('Wrist').distanceTo(joint('Elbow')));
    const toeLine = difference(joint('Toe'), joint('Heel'));
    const footUp = vector().crossVectors(hipsRight, toeLine);
    const footRight = vector().crossVectors(toeLine, footUp);
    frame(side + 'Foot', joint('Ankle'), axes(footUp, footRight), toeLine.length());
  }
  return fields;
}

const bind = poseFrames(atlas);
const segmentNames = Object.keys(bind);
const skinSegments = [...segmentNames, 'lumbar'];
const abductors = new Set(['medius', 'minimus', 'tensor']);
const deepRotators = new Set(['piriformis', 'quadratus', 'obturator']);
const adductors = new Set(['pectineus', 'adductorLongus', 'adductorMagnus']);
const crossesKnee = new Set(['band', 'rectus', 'vastus', 'semitendinosus']);
const tissueColors = {
  maximus: 0xa64d47, medius: 0x239e91, minimus: 0x257787, tensor: 0x4cb09d, band: 0xe6d4b1,
  iliacus: 0xad6254, psoas: 0x914843, piriformis: 0xb76e5d, quadratus: 0x93514e, obturator: 0x975451,
  pectineus: 0xb96c5b, adductorLongus: 0xa65046, adductorMagnus: 0x964741,
  rectus: 0xb76453, vastus: 0xa55049, semitendinosus: 0xaf6555,
};

// Reference atlas units. The deep rotators run mostly across the hip; their
// medial pelvic attachments and lateral femoral insertions share a height.
function pelvicAttachment(type, height, lateral) {
  const blend = THREE.MathUtils.smoothstep;
  if (deepRotators.has(type)) return 1 - blend(lateral, .135, .285);
  if (type === 'vastus') return 0;
  if (adductors.has(type)) return (1 - blend(lateral, .085, .24)) * blend(height, -.40, -.12);
  if (type === 'semitendinosus') return blend(height, -.36, -.125);
  if (type === 'maximus') return Math.max((1 - blend(lateral, .07, .21)) * blend(height, -.38, -.10), blend(height, -.22, .13));
  return blend(height, -.085, .155);
}
const spine = [
  ['lumbar', landmark(0, .253, -.067), 'pelvis', landmark(0, .666, -.061), 'thorax'],
  ['neck', landmark(0, 1.416, -.077), 'thorax', landmark(0, 1.678, -.081), 'head'],
].map(([name, a, lower, b, upper]) => ({
  name, a, lower, b, upper,
  source: {anchor: a, basis: axes(difference(b, a), vector(0, 0, 1)), height: a.distanceTo(b), width: 1},
}));

function segmentMatrix(source, dest) {
  const lengthScale = dest.height / source.height;
  const widthScale = dest.width / source.width;
  const depthScale = Math.sqrt(lengthScale * widthScale);
  const scale = new THREE.Matrix4().makeScale(depthScale, lengthScale, widthScale);
  return new THREE.Matrix4().makeTranslation(...dest.anchor)
    .multiply(dest.basis).multiply(scale).multiply(source.basis.clone().transpose())
    .multiply(new THREE.Matrix4().makeTranslation(-source.anchor.x, -source.anchor.y, -source.anchor.z));
}

function segmentOf(name) {
  const side = name.startsWith('right-') ? 'right' : 'left';
  if (/sacrum|hip-bone/.test(name)) return 'pelvis';
  if (/femur|patella/.test(name)) return side + 'Thigh';
  if (/tibia$|fibula$/.test(name)) return side + 'Shin';
  if (/talus|calcaneus|navicular|cuboid|cuneiform|metatarsal|toe-|of-the-foot/.test(name)) return side + 'Foot';
  if (/humerus/.test(name)) return side + 'Arm';
  if (/radius|ulna/.test(name)) return side + 'Forearm';
  if (/carpal|finger|thumb|scaphoid|lunate|triquetral|pisiform|trapez|capitate|hamate/.test(name)) return side + 'Hand';
  if (/cervical|atlas-c1|axis-c2|disc-c/.test(name)) return 'neck';
  if (/lumbar|disc-l|disc-t12-l1/.test(name)) return 'lumbar';
  if (/vertebr|sternum|rib|costal|clavicle|scapula|xiphoid/.test(name)) return 'thorax';
  return 'head';
}

function prepareRig(boneAsset, tissueAsset) {
  const skeleton = new THREE.Group();
  const atlasMaterials = new Set();
  const boneMaterial = new THREE.MeshStandardMaterial({color: 0xe2dfc9, roughness: .74});
  const cartilage = new THREE.MeshStandardMaterial({color: 0x93b4ab, roughness: .85});
  const groups = new Map();
  const renderedSegments = [];
  const resources = new Set([boneMaterial, cartilage]);
  function releaseMaterials(mesh) {
    for (const material of [mesh.material].flat()) atlasMaterials.add(material);
  }
  boneAsset.traverse(mesh => {
    if (!mesh.isMesh) return;
    const segment = segmentOf(mesh.name);
    const kind = /disc-|cartilage/.test(mesh.name) ? 'cartilage' : 'bone';
    const key = segment + ' ' + kind;
    if (!groups.has(key)) groups.set(key, []);
    mesh.geometry.applyMatrix4(rot);
    // The GLB has distinct sub-meshes for the individual bones. Consolidate
    // draw calls without welding across anatomical boundaries.
    groups.get(key).push(mesh.geometry);
    releaseMaterials(mesh);
  });
  groups.forEach((geometries, key) => {
    const [segment, kind] = key.split(' ');
    const geometry = mergeGeometries(geometries);
    for (const original of geometries) original.dispose();
    const mesh = new THREE.Mesh(geometry, kind === 'bone' ? boneMaterial : cartilage);
    mesh.matrixAutoUpdate = false;
    renderedSegments.push({ mesh, segment });
    resources.add(geometry);
    skeleton.add(mesh);
  });

  const bones = skinSegments.map(() => new THREE.Bone());
  for (const bone of bones) {
    bone.matrixAutoUpdate = false;
    skeleton.add(bone);
  }
  const armature = new THREE.Skeleton(bones, bones.map(() => new THREE.Matrix4()));
  const materials = {};
  for (const [type, color] of Object.entries(tissueColors)) {
    const focus = new THREE.MeshStandardMaterial({
      color, roughness: type === 'band' ? .70 : .75,
      transparent: type === 'band', opacity: type === 'band' ? .5 : 1, depthWrite: type !== 'band',
    });
    const opposite = focus.clone();
    opposite.color.set(abductors.has(type) ? 0xa75e55 : color).lerp(new THREE.Color(0x8b7772), .34).multiplyScalar(.72);
    materials[type] = {focus, opposite};
    resources.add(focus);
    resources.add(opposite);
  }
  const tissues = [];
  const lumbar = skinSegments.indexOf('lumbar'), pelvis = skinSegments.indexOf('pelvis');
  let activeSide = 'right', tissueMode = 'all';
  function updateTissues() {
    for (const {side, type, mesh} of tissues) {
      mesh.visible = tissueMode === 'all' || (tissueMode === 'abductors' && abductors.has(type));
      mesh.material = materials[type][side === activeSide ? 'focus' : 'opposite'];
    }
  }
  tissueAsset.traverse(source => {
    if (!source.isMesh) return;
    const [side, type] = source.name.split('_');
    const geometry = source.geometry.applyMatrix4(rot);
    const vertices = geometry.getAttribute('position');
    const indices = new Uint16Array(vertices.count * 4);
    const weights = new Float32Array(vertices.count * 4);
    const thigh = skinSegments.indexOf(side + 'Thigh'), shin = skinSegments.indexOf(side + 'Shin');
    for (let i = 0; i < vertices.count; i++) {
      const y = vertices.getY(i);
      const proximal = pelvicAttachment(type, y, Math.abs(vertices.getZ(i)));
      const backWeight = type === 'psoas' ? proximal * THREE.MathUtils.smoothstep(y, .26, .50) : 0;
      const shankWeight = crossesKnee.has(type) ? 1 - THREE.MathUtils.smoothstep(y, -1.14, -.99) : 0;
      indices.set([pelvis, thigh, shin, lumbar], 4*i);
      weights.set([proximal - backWeight, 1 - proximal - shankWeight, shankWeight, backWeight], 4*i);
    }
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(indices, 4));
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
    const mesh = new THREE.SkinnedMesh(geometry, materials[type].focus);
    mesh.bind(armature, identity);
    mesh.frustumCulled = false;
    tissues.push({ side, type, mesh });
    skeleton.add(mesh);
    resources.add(geometry);
    releaseMaterials(source);
  });
  for (const material of atlasMaterials) material.dispose();
  updateTissues();
  return {
    object: skeleton,
    apply(points) {
      const current = poseFrames(points);
      const matrices = Object.fromEntries(segmentNames.map(segment => [segment, segmentMatrix(bind[segment], current[segment])]));
      // Keep the cervical and lumbar stacks seated on the adjacent anatomy
      // while headband, shoulder and hip landmarks move independently.
      for (const joint of spine) {
        const a = joint.a.clone().applyMatrix4(matrices[joint.lower]);
        const b = joint.b.clone().applyMatrix4(matrices[joint.upper]);
        const right = vector(0, 0, 1).transformDirection(matrices[joint.lower]);
        const destination = {anchor: a, basis: axes(difference(b, a), right), height: a.distanceTo(b), width: current.head.height / bind.head.height};
        matrices[joint.name] = segmentMatrix(joint.source, destination);
      }
      for (const { segment, mesh } of renderedSegments) {
        mesh.matrix.copy(matrices[segment]);
        mesh.matrixWorldNeedsUpdate = true;
      }
      bones.forEach((bone, i) => { bone.matrix.copy(matrices[skinSegments[i]]); bone.matrixWorldNeedsUpdate = true; });
    },
    focus(side) {
      activeSide = side;
      updateTissues();
    },
    setTissueMode(mode) {
      if (!['all', 'abductors', 'bone'].includes(mode)) return;
      tissueMode = mode;
      updateTissues();
    },
    dispose() { resources.forEach(resource => resource.dispose()); armature.dispose(); }
  };
}

export async function createBodyView(host) {
  const loader = new GLTFLoader();
  const [bones, tissues] = await Promise.all([loader.loadAsync(boneUrl), loader.loadAsync(muscleUrl)]);
  const rig = prepareRig(bones.scene, tissues.scene);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x101c27, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.16;
  const canvas = renderer.domElement;
  canvas.style.cssText = 'display:block;width:100%;height:100%;touch-action:none';
  canvas.setAttribute('aria-label', 'Three-dimensional skeleton with highlighted hip abductors. Drag to orbit and scroll to zoom.');
  host.append(canvas);
  const scene = new THREE.Scene();
  rig.object.visible = false;
  scene.add(rig.object);
  scene.add(new THREE.HemisphereLight(0xd7edfa, 0x778580, 1.45));
  for (const [x, y, z, color, power] of [[3, 4, 2, 0xffe8c8, 2], [-2, 2, -3, 0x8bbec9, 1.65], [1, 1, -2, 0xcde2e2, .55]]) {
    const light = new THREE.DirectionalLight(color, power);
    light.position.set(x, y, z);
    scene.add(light);
  }

  const grid = new THREE.GridHelper(3.2, 16, 0x608787, 0x40616e);
  grid.material.transparent = true;
  grid.material.opacity = .20;
  scene.add(grid);
  const shadowCanvas = document.createElement('canvas');
  shadowCanvas.width = shadowCanvas.height = 128;
  const ctx = shadowCanvas.getContext('2d');
  const halo = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  halo.addColorStop(0, 'rgba(0,0,0,0.62)');
  halo.addColorStop(.45, 'rgba(0,0,0,0.25)');
  halo.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(shadowCanvas);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.1, .72), new THREE.MeshBasicMaterial({map: texture, transparent: true, depthWrite: false}));
  shadow.rotation.x = -Math.PI/2;
  scene.add(shadow);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .01, 15);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false;
  controls.minZoom = .65;
  controls.maxZoom = 4.5;
  controls.minPolarAngle = .12;
  controls.maxPolarAngle = Math.PI * .66;
  let clip, names, currentPoints = {}, phase = 0, view = 'oblique', side = 'right';
  let height = 1.73, centerY = .83, request = 0;
  function paint() {
    if (!request) request = requestAnimationFrame(() => {
      request = 0;
      renderer.render(scene, camera);
    });
  }
  function resize() {
    const { width, height: pixels } = host.getBoundingClientRect();
    if (!width || !pixels) return;
    renderer.setSize(width, pixels, false);
    const aspect = width / pixels;
    const extent = Math.max(height * .56, .50 / aspect);
    camera.top = extent;
    camera.bottom = -extent;
    camera.left = -extent * aspect;
    camera.right = extent * aspect;
    camera.updateProjectionMatrix();
    paint();
  }
  function setView(preset = 'oblique') {
    view = preset;
    const sign = side === 'left' ? -1 : 1;
    const direction = {front: [1,.08,0], back: [-1,.08,0], side: [0,.08,sign], oblique: [.79,.12,sign]};
    controls.target.set(0, centerY, 0);
    camera.position.copy(controls.target).add(vector(...(direction[view] || direction.oblique)).normalize().multiplyScalar(3.2));
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.update();
    paint();
  }
  function setPhase(value) {
    phase = THREE.MathUtils.clamp(value, 0, 1);
    if (!clip?.frames.length) return;
    const frames = clip.frames;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
      const middle = (lo + hi) >>> 1;
      if (frames[middle].phase < phase) lo = middle + 1;
      else hi = middle;
    }
    const a = frames[Math.max(0,lo - 1)], b = frames[lo];
    const mix = b.phase > a.phase ? (phase - a.phase) / (b.phase - a.phase) : 0;
    for (let i=0; i<names.length; i++) {
      const p = a.points[i], q = b.points[i];
      currentPoints[names[i]].set(p[0]+(q[0]-p[0])*mix,p[1]+(q[1]-p[1])*mix,p[2]+(q[2]-p[2])*mix);
    }
    rig.apply(currentPoints);
    paint();
  }
  function setMotion(next, jointNames = next?.jointNames) {
    clip = next;
    rig.object.visible = Boolean(clip?.frames.length && jointNames?.length);
    if (!rig.object.visible) { paint(); return; }
    names = jointNames;
    currentPoints = Object.fromEntries(names.map(name => [name, vector()]));
    side = /^l/i.test(clip.side) ? 'left' : 'right';
    rig.focus(side);
    const iHead = names.indexOf('head');
    const feet = ['leftHeel', 'rightHeel', 'leftToe', 'rightToe'].map(name => names.indexOf(name));
    const top = Math.max(...clip.frames.map(f => f.points[iHead][1])) + .125;
    const floor = Math.min(...clip.frames.flatMap(f => feet.map(i => f.points[i][1]))) - .032;
    grid.position.y = floor - .002;
    shadow.position.y = floor;
    centerY = (top + floor) / 2;
    height = top - floor;
    resize();
    setView(view);
    setPhase(phase);
  }
  controls.addEventListener('change', paint);
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  resize();
  setView();
  return {
    setMotion, setPhase, setView, reset: () => setView('oblique'),
    setTissueMode(mode) { rig.setTissueMode(mode); paint(); },
    dispose() {
      observer.disconnect();
      controls.dispose();
      if (request) cancelAnimationFrame(request);
      shadow.geometry.dispose(); shadow.material.dispose(); texture.dispose();
      grid.geometry.dispose(); grid.material.dispose();
      rig.dispose();
      renderer.dispose();
      canvas.remove();
    }
  };
}
