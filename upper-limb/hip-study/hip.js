import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HIP_PROFILES, describeSurface, profileContains, visibleSurface, atlasMaterial, shadeSection } from './hip-anatomy.js';
import { createLabelVisibility } from './hip-label-visibility.js';
import { prepareHipDeformation } from './hip-deformation.js';
import { prepareHipContact } from './hip-contact.js';
import { loadBursaMechanics } from './bursa-mechanics.js';
import { makeHipReference } from './hip-reference.js';

const ATLAS = new URL('./assets/hip-atlas.glb', import.meta.url).href;
const MODEL_GEOMETRY = new URL('../../hip-study/data/geometry.json', import.meta.url).href;
let referencePaths;
const radians = THREE.MathUtils.degToRad;
const clamp = THREE.MathUtils.clamp;

const VIEWS = {
  anterior: [-0.04,.07,1.3], lateral:[-1.3,.08,.03],
  posterior:[-.42,.15,-1.2], socket:[-1.10,-.26,.56],
};
const SEATED = new THREE.Vector3();
const EXPLODED_FEMUR = new THREE.Vector3(-.022,-.033,-.048);
let atlasPromise;

export function loadHipAtlas() {
  atlasPromise ??= new GLTFLoader().loadAsync(ATLAS).catch(error => {
    atlasPromise = undefined;
    throw error;
  });
  return atlasPromise;
}

function makeAtlas(gltf, deformation, uniforms, profile, configureSurface) {
  const pivot = new THREE.Vector3().fromArray(gltf.parser.json.asset.extras.femoralHeadSphere.centerM);
  const rig = new THREE.Group();
  const pelvis = new THREE.Bone();
  const femur = new THREE.Bone();
  pelvis.add(femur);
  rig.add(pelvis);
  const skeleton = new THREE.Skeleton([pelvis, femur]);
  const surfaces = [];
  gltf.scene.traverse(object => {
    if (!object.isMesh) return;
    const info = describeSurface(object.userData);
    if (!profileContains(profile,info)) return;
    const {label,layer} = info;
    const geometry = object.geometry.clone();
    const points = geometry.getAttribute('position');
    const sourceWeights = deformation.weights(label,object.geometry);
    const joints = new Uint16Array(points.count * 4);
    const weights = new Float32Array(points.count * 4);
    for (let i = 0; i < points.count; i++) {
      const weight = sourceWeights[i];
      joints[i*4+1] = 1;
      weights[i*4] = 1-weight;
      weights[i*4+1] = weight;
    }
    geometry.translate(-pivot.x,-pivot.y,-pivot.z);
    geometry.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(joints,4));
    geometry.setAttribute('skinWeight',new THREE.Float32BufferAttribute(weights,4));
    const material = atlasMaterial(info,uniforms,geometry.hasAttribute('color'),object.material);
    shadeSection(geometry,object.userData.sections,pivot.y,info,material);
    const skin = new THREE.SkinnedMesh(geometry,material);
    skin.bind(skeleton);
    skin.frustumCulled = false;
    skin.castShadow = !['fascia','bursae'].includes(layer);
    skin.receiveShadow = true;
    skin.userData = info;
    const attachedToPelvis = profile === 'joint' && !['Femur','Art cart of femur head'].includes(label);
    deformation.attach(skin,label,attachedToPelvis ? 'pelvic' : undefined,object.geometry);
    configureSurface?.(material, info);
    pelvis.add(skin);
    surfaces.push(skin);
  });
  return {pivot,rig,pelvis,femur,skeleton,surfaces};
}

function pickLandmarks(definitions, surfaces, pivot) {
  const point = new THREE.Vector3();
  return definitions.flatMap(definition => {
    const matching = surfaces.filter(m => m.userData.label === definition.label);
    let least = Infinity;
    let mesh, index;
    const target = new THREE.Vector3().fromArray(definition.xyz).sub(pivot);
    for (const surface of matching) {
      const positions = surface.geometry.attributes.position;
      for (let i=0; i<positions.count; i++) {
        const d = point.fromBufferAttribute(positions,i).distanceToSquared(target);
        if (d<least) { least = d; index = i; mesh = surface; }
      }
    }
    return mesh ? [{...definition,mesh,index}] : [];
  });
}

function makeAnnotations(host, landmarks, bursaMechanics) {
  const overlay = document.createElement('div');
  overlay.className = 'hip-atlas-annotations';
  overlay.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:1';
  host.append(overlay);
  const items = landmarks.map((landmark) => {
    const interactive = bursaMechanics?.label === landmark.label;
    const label = document.createElement(interactive ? 'button' : 'div');
    label.style.cssText = 'position:absolute;display:block;color:#e0e8e8;font:500 10px/1.2 system-ui,sans-serif;letter-spacing:.02em;white-space:nowrap;text-shadow:0 1px 4px #071118,0 0 2px #071118;background:none;border:0;padding:0;text-align:left';
    const caption = document.createElement('span');
    caption.textContent = landmark.text;
    label.append(caption);
    let readout;
    if (interactive) {
      label.type = 'button';
      label.setAttribute('aria-expanded','false');
      label.style.pointerEvents = 'auto';
      label.style.cursor = 'pointer';
      label.title = 'Inspect the modeled movement of the iliotibial band along this bursa';
      readout = document.createElement('span');
      readout.style.cssText = 'display:block;font-size:9px;color:#bad7d7;font-variant-numeric:tabular-nums;line-height:1.5';
      label.append(readout);
    } else label.setAttribute('aria-hidden','true');
    const leader = document.createElement('div');
    leader.style.cssText = 'position:absolute;height:1px;background:rgba(196,224,219,.5);transform-origin:0 50%';
    const dot = document.createElement('div');
    dot.style.cssText = 'position:absolute;width:4px;height:4px;margin:-1.5px;border-radius:50%;background:#d2e1d6;box-shadow:0 0 3px #1b3030';
    overlay.append(leader, dot, label);
    return {label, caption, readout, leader, dot, landmark};
  });
  return {overlay, items};
}

// Display the published actuators' straight control polygons. Each model's hip
// centre and neutral axes are aligned with the atlas, without surface fitting.
function modelGuides(host, pelvis, femur, lowerY, redraw) {
  const group = new THREE.Group();
  pelvis.add(group);
  group.visible = false;
  const paths = [];
  let requested = false;
  let disposed = false;
  const legend = document.createElement('div');
  legend.style.cssText = 'position:absolute;bottom:36px;right:12px;pointer-events:none;background:#101c27e8;padding:6px 9px;border-radius:3px;font:500 10px/1.35 system-ui,sans-serif;color:#e9ece6;z-index:2';
  legend.hidden = true;
  legend.setAttribute('aria-live','polite');
  host.append(legend);
  const colors = {Medius:'#ffd167', Minimus:'#59f7df', 'IT band':'#d6a8ff'};

  function addPath(record) {
    const category = record.name.startsWith('glmed') ? 'Medius' : record.name.startsWith('glmin') ? 'Minimus' : 'IT band';
    const points = [];
    let knotCount = 0;
    for (const p of record.points) {
      const rest = new THREE.Vector3(-p.z, p.y, p.x);
      if (rest.y < lowerY) {
        const last = points.at(-1);
        if (last && last.rest.y > lowerY) {
          const t = (last.rest.y - lowerY) / (last.rest.y - rest.y);
          points.push({rest:last.rest.clone().lerp(rest,t), mobile:true});
        }
        break;
      }
      points.push({rest, mobile:p.body === 'femur_r'});
      knotCount++;
    }
    if (points.length < 2) return;
    const geometry = new THREE.BufferGeometry();
    const positions = new THREE.Float32BufferAttribute(new Float32Array(points.length*3),3);
    geometry.setAttribute('position', positions);
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color:colors[category], depthTest:false, depthWrite:false, transparent:true, opacity:0.9, toneMapped:false,
    }));
    const knots = new THREE.BufferGeometry();
    knots.setAttribute('position', positions);
    knots.setDrawRange(0,knotCount);
    const markers = new THREE.Points(knots, new THREE.PointsMaterial({
      color:colors[category], size:4, sizeAttenuation:false, depthTest:false, depthWrite:false, toneMapped:false,
    }));
    line.frustumCulled = markers.frustumCulled = false;
    line.renderOrder = 4;
    markers.renderOrder = 5;
    group.add(line,markers);
    paths.push({points,positions});
  }

  const moving = new THREE.Vector3();
  function update() {
    if (!group.visible) return;
    for (const {points, positions} of paths) {
      points.forEach(({rest, mobile}, i) => {
        moving.copy(rest);
        if (mobile) moving.applyQuaternion(femur.quaternion);
        positions.setXYZ(i,moving.x,moving.y,moving.z);
      });
      positions.needsUpdate = true;
    }
  }

  function show(visible) {
    group.visible = visible;
    legend.hidden = !visible;
    if (!visible || requested) return;
    requested = true;
    legend.textContent = 'Loading model guides…';
    referencePaths ??= fetch(MODEL_GEOMETRY).then(async response => {
      if (!response.ok) throw new Error('Unable to load reference geometry');
      return (await response.json()).referencePaths.paths;
    }).catch(error => { referencePaths = undefined; throw error; });
    referencePaths.then(records => {
      if (disposed) return;
      records.forEach(addPath);
      legend.textContent = 'Model guides';
      for (const [name, color] of Object.entries(colors)) {
        const item = document.createElement('span');
        item.textContent = ' • '+name;
        item.style.color = color;
        legend.append(item);
      }
      update();
      redraw();
    }).catch(() => {
      if (!disposed) legend.textContent = 'Model guides unavailable';
      requested = false;
    });
  }

  return {show, update, dispose() {
    disposed = true;
    legend.remove();
    group.removeFromParent();
    for (const mesh of group.children) {
      mesh.material.dispose();
      mesh.geometry.dispose();
    }
  }};
}

export async function createHipScene(host, {profile = 'overview', landmarks, framing, configureSurface, onCameraChange} = {}) {
  if (!Object.hasOwn(HIP_PROFILES, profile)) throw new RangeError('Unknown hip dissection profile '+profile);
  const preset = HIP_PROFILES[profile];
  const gltf = await loadHipAtlas();
  const [deformation,bursaMechanics,contact] = await Promise.all([
    prepareHipDeformation(gltf), profile === 'deep' ? loadBursaMechanics() : null,
    profile === 'joint' ? null : prepareHipContact(gltf),
  ]);
  const renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
  renderer.setClearColor(0x101c27, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  const canvas = renderer.domElement;
  canvas.style.cssText = 'width:100%;height:100%;display:block;touch-action:none';
  canvas.setAttribute('aria-label', 'Three-dimensional hip dissection. Tap a structure to identify it, drag to rotate, scroll to zoom, or use arrow keys.');
  canvas.tabIndex = 0;
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.prepend(canvas);
  const uniforms = Object.fromEntries(['showBody','showTendon','jointColor','tendonColor','bandColor'].map(k => [k, {value: 0}]));
  const {pivot, rig, pelvis, femur, skeleton, surfaces} = makeAtlas(gltf, deformation, uniforms, profile, configureSurface);
  const world = new THREE.Group();
  world.add(rig);
  const scene = new THREE.Scene();
  const studio = new RoomEnvironment();
  const generator = new THREE.PMREMGenerator(renderer);
  const radiance = generator.fromScene(studio,.025,.1,100,{size:64});
  scene.environment = radiance.texture;
  scene.environmentIntensity = .25;
  generator.dispose();
  studio.dispose();
  scene.add(world, new THREE.HemisphereLight(0xfff2df,0x8d96a6,.33));
  for (const [color, strength, pos] of [
    [0xffe7ce, 2.18, [-1.85, 1.85, .68]],
    [0xffe0c8, .68, [-.38, .60, -1.8]],
    [0xb4d8db, .82, [1.65, .48, -.72]],
  ]) {
    const light = new THREE.DirectionalLight(color, strength);
    light.position.set(...pos);
    if (color === 0xffe7ce) {
      light.castShadow = true;
      light.shadow.mapSize.set(1024, 1024);
      Object.assign(light.shadow.camera, {left: -.22, right: .22, top: .22, bottom: -.22, near: 2.2, far: 3.3});
      light.shadow.normalBias = .0007;
      light.shadow.bias = -.00008;
    }
    scene.add(light);
  }
  const camera = new THREE.OrthographicCamera(-.2, .2, .2, -.2, .01, 4);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = true;
  controls.enableDamping = false;
  controls.minZoom = .65;
  controls.maxZoom = 4;
  controls.rotateSpeed = .61;
  const anchors = (landmarks ?? preset.anchors).map(a => bursaMechanics?.label === a.label ? {...a,xyz:bursaMechanics.landmarkM} : a);
  const {overlay, items} = makeAnnotations(host,pickLandmarks(anchors,surfaces,pivot),bursaMechanics);
  const hover = document.createElement('span');
  hover.style.cssText = 'position:absolute;z-index:3;pointer-events:none;background:#0c1723eb;color:#f4f1e7;border:1px solid #87969480;border-radius:4px;padding:5px 7px;font:500 11px/1.3 system-ui,sans-serif;max-width:min(240px,90%);display:none';
  host.append(hover);
  const layers = {
    bone:true, muscle:true, tendons:true, cartilage:true, band:true, fascia:true,
    bursae:profile === 'deep', ligaments:true, labrum:true, loads:true, labels:true, rotators:true,
    capsule:false, medius:false, nerves:false, vessels:false, paths:false, envelope:false, separate:profile === 'joint',
  };
  let frame = {};
  let view = 'oblique';
  let side = 1;
  let disposed = false;
  let request = 0;
  let width = 1;
  let height = 1;
  const pose = new THREE.Quaternion();
  const referencePose = new THREE.Quaternion();
  let referenceComparison, comparisonPose;
  const hipEuler = new THREE.Euler(0,0,0,'ZXY');
  const angle = (sample, key) => radians(Number.isFinite(sample[key]) ? sample[key] : 0);
  function hipRotation(sample, target) {
    // Source axes are X medial on the right, Y superior and Z anterior.
    hipEuler.set(-angle(sample, 'hipFlexionDeg'), angle(sample, 'hipRotationDeg'), angle(sample, 'hipAdductionDeg'));
    return target.setFromEuler(hipEuler);
  }
  const point = new THREE.Vector3();
  const ray = new THREE.Raycaster();
  const guides = modelGuides(host, pelvis, femur, .695 - pivot.y, queue);
  const visibility = createLabelVisibility(renderer,scene,camera,surfaces,queue);
  let layerRevision = 0;
  let bursaTravelMm = 0;
  let touchStart;
  let hideTooltip;
  const bursaLabel = items.find(item => item.readout)?.label;
  let selectedBursa = false;
  function bursaDetail() {
    return 'Trochanteric bursa of gluteus maximus. Modeled IT band travel from neutral is '+bursaTravelMm.toFixed(1)+' mm, measured tangentially across the bursal surface (area-weighted RMS).';
  }
  function toggleBursa() {
    selectedBursa = !selectedBursa;
    bursaLabel.setAttribute('aria-expanded',String(selectedBursa));
    if (selectedBursa) {
      hover.textContent = bursaDetail();
      Object.assign(hover.style,{left:'12px',top:'40px',display:'block'});
    } else hover.style.display = 'none';
  }
  bursaLabel?.addEventListener('click',toggleBursa);

  function updateAnnotations() {
    overlay.hidden = !layers.labels || width < 205 || height < 190;
    if (overlay.hidden) return;
    const viewKey = [side,width,height,layerRevision,camera.zoom,...camera.position.toArray(),
      ...controls.target.toArray(),...pose.toArray(),...pelvis.quaternion.toArray()];
    visibility.refresh(width,height,viewKey.map(v => Math.round(v*100)).join('/'));
    const occupied = [];
    for (const {label, caption, readout, leader, dot, landmark} of items) {
      if (layers.labelTarget && landmark.label !== layers.labelTarget) {
        for (const element of [label,leader,dot]) element.style.visibility = 'hidden';
        continue;
      }
      const name = width < 415 ? landmark.compact || landmark.text : landmark.text;
      if (name !== caption.textContent) caption.textContent = name;
      if (readout) {
        const value = (width < 320 ? 'Model ITB travel' : 'Modeled IT band travel')+' · '+bursaTravelMm.toFixed(1)+' mm';
        if (value !== readout.textContent) readout.textContent = value;
      }
      const show = landmark.mesh.visible && (profile === 'joint' || width >= 320 || items.findIndex(item => item.landmark === landmark) < 2);
      landmark.mesh.getVertexPosition(landmark.index,point);
      point.applyMatrix4(landmark.mesh.matrixWorld).project(camera);
      const socketCartilage = profile === 'joint' && landmark.label === 'Articular cartilage of hip bone acetabulum';
      const endpoint = visibility.visiblePoint(landmark.mesh,.5+.5*point.x,.5-.5*point.y,socketCartilage,landmark.narrow);
      if (!endpoint) {
        for (const element of [label,leader,dot]) element.style.visibility = 'hidden';
        continue;
      }
      const x = width*endpoint[0], y = height*endpoint[1];
      const direction = landmark.side * side;
      const labelWidth = label.offsetWidth || 90;
      let labelY = clamp(y - 20, 33, height - 38);
      const [left,right] = visibility.rowBounds((labelY+6)/height);
      const labelX = direction < 0
        ? Math.max(8, Math.min(x-18,left*width-10)-labelWidth)
        : Math.min(width-8-labelWidth,Math.max(x+18,right*width+10));
      for (const box of occupied) {
        if (Math.abs(box.y - labelY) < 18 && Math.abs(box.x - labelX) < 115) labelY += 19;
      }
      const visible = show && x > 6 && x < width-6 && y > 14 && y < height-20;
      for (const element of [label, leader, dot]) element.style.visibility = visible ? 'visible' : 'hidden';
      if (!visible) continue;
      occupied.push({x:labelX,y:labelY});
      label.style.left = labelX+'px';
      label.style.top = labelY+'px';
      const dx = (direction < 0 ? labelX + labelWidth + 4 : labelX - 4) - x;
      const dy = labelY + 6 - y;
      leader.style.width = Math.hypot(dx,dy)+'px';
      leader.style.transform = 'translate('+x+'px,'+y+'px) rotate('+Math.atan2(dy,dx)+'rad)';
      dot.style.transform = 'translate('+x+'px,'+y+'px)';
    }
  }

  function draw() {
    request = 0;
    if (disposed || !host.clientWidth || !host.clientHeight) return;
    width = host.clientWidth;
    height = host.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width*ratio) || canvas.height !== Math.floor(height*ratio)) {
      renderer.setPixelRatio(ratio);
      renderer.setSize(width, height, false);
    }
    const aspect = width/height;
    const half = Math.max(preset.halfHeight,preset.halfWidth/aspect);
    camera.left = -half*aspect;
    camera.right = half*aspect;
    camera.top = half;
    camera.bottom = -half;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    if (comparisonPose) referenceComparison.draw(renderer,camera,pelvis,pose,comparisonPose);
    updateAnnotations();
  }
  function queue() {
    if (!disposed && !request) request = requestAnimationFrame(draw);
  }
  function setView(next = 'oblique') {
    view = Object.hasOwn(VIEWS,next) ? next : 'oblique';
    const direction = VIEWS[view] || preset.direction;
    const socket = view === 'socket';
    controls.target.fromArray(framing?.target ?? preset.target);
    if (socket) controls.target.set(0,0,0);
    controls.target.x *= side;
    camera.position.copy(controls.target).add(new THREE.Vector3(side*direction[0],direction[1],direction[2]).normalize());
    camera.up.set(0,1,0);
    camera.zoom = socket && profile !== 'joint' ? 1.78 : (framing?.zoom ?? 1);
    camera.updateProjectionMatrix();
    controls.update();
    queue();
  }
  function setFrame(next = {}) {
    frame = next;
    const nextSide = /^(l|left)$/i.test(frame.side || '') ? -1 : 1;
    const signChanged = nextSide !== side;
    side = nextSide;
    world.scale.x = side;
    hipRotation(frame,pose);
    const contactResult = contact?.solve(pose);
    if (bursaMechanics) bursaTravelMm = bursaMechanics.excursionMm(pose,contactResult.positions);
    if (selectedBursa) hover.textContent = bursaDetail();
    femur.quaternion.copy(pose);
    if (profile === 'joint' && layers.separate) femur.position.copy(EXPLODED_FEMUR);
    else femur.position.set(0,0,0);
    deformation.setPose(femur.quaternion,femur.position,undefined,contactResult);
    if (profile === 'joint') deformation.setPose(femur.quaternion,SEATED,'pelvic');
    pelvis.rotation.set(angle(frame,'pelvisTiltDeg'), angle(frame,'pelvisRotationDeg'), angle(frame,'pelvisObliquityDeg'), 'ZXY');
    guides.update();
    function scale(value, maximum) {
      return layers.loads && Number.isFinite(value) ? clamp(value/maximum,0,1) : 0;
    }
    uniforms.jointColor.value = scale(frame.jointBw,4);
    uniforms.tendonColor.value = scale(frame.tendonBw,3);
    uniforms.bandColor.value = scale(frame.compressionBw,1);
    if (signChanged) setView(view);
    queue();
  }
  function setLayers(changes = {}) {
    Object.assign(layers,changes);
    if (selectedBursa && (!layers.bursae || !layers.labels)) toggleBursa();
    layerRevision++;
    guides.show(layers.paths && profile !== 'joint');
    for (const mesh of surfaces) {
      mesh.visible = visibleSurface(profile,mesh.userData,layers);
      if (mesh.userData.glutealTendon) mesh.castShadow = layers.muscle && layers.tendons;
    }
    uniforms.showBody.value = Number(layers.muscle);
    uniforms.showTendon.value = Number(profile === 'surface' ? layers.muscle : layers.tendons);
    setFrame(frame);
  }
  function setReferenceFrame(sample) {
    if (profile !== 'deep' || disposed || (!sample && !comparisonPose)) return;
    if (sample) {
      referenceComparison ??= makeHipReference(surfaces,skeleton,deformation,uniforms);
      comparisonPose ??= new THREE.Quaternion();
      hipRotation(sample,comparisonPose);
    } else comparisonPose = null;
    queue();
  }
  function inspect(event, tap = false) {
    if (selectedBursa) return;
    if (event.buttons || (!tap && event.pointerType === 'touch')) { hover.style.display = 'none'; return; }
    const box = canvas.getBoundingClientRect();
    const x = event.clientX-box.left, y = event.clientY-box.top;
    ray.setFromCamera(new THREE.Vector2(2*x/width-1,1-2*y/height),camera);
    let title = '';
    const visible = surfaces.filter(o => o.visible && o.userData.layer !== 'context');
    for (const hit of ray.intersectObjects(visible,false)) {
      const {label,glutealTendon,layer} = hit.object.userData;
      if (layer === 'fascia' && label === 'Fascia lata' && hit.object.material.opacity < .25) continue;
      if (glutealTendon) {
        const face = hit.face;
        const insertion = hit.object.geometry.attributes.hipInsertion;
        const b = hit.barycoord;
        const tendon = insertion && b.x*insertion.getX(face.a)+b.y*insertion.getX(face.b)+b.z*insertion.getX(face.c) >= .5;
        if (tendon ? !uniforms.showTendon.value : !layers.muscle) continue;
        title = hit.object.userData.title + (tendon ? ' tendon region (schematic)' : '');
      } else {
        title = label === bursaMechanics?.label ? bursaDetail() : hit.object.userData.title;
      }
      break;
    }
    hover.style.display = title ? 'block' : 'none';
    if (!title) return;
    hover.textContent = title;
    hover.style.left = Math.max(2, Math.min(width-hover.offsetWidth-5,x+13))+'px';
    hover.style.top = Math.max(2,Math.min(height-hover.offsetHeight-4,y+11))+'px';
  }
  function leave() { if (!selectedBursa) hover.style.display = 'none'; }
  function onPointerLeave(event) { if (event.pointerType !== 'touch') leave(); }
  function onTouchStart(event) {
    if (selectedBursa) toggleBursa();
    if (event.pointerType !== 'touch') return;
    touchStart = new THREE.Vector2(event.clientX, event.clientY);
    clearTimeout(hideTooltip);
    leave();
  }
  function onTouchEnd(event) {
    if (event.pointerType !== 'touch' || !touchStart) return;
    if (Math.hypot(event.clientX - touchStart.x, event.clientY - touchStart.y) < 6) {
      inspect(event, true);
      hideTooltip = setTimeout(leave,2400);
    }
    touchStart = undefined;
  }
  function onKey(event) {
    if (event.key === 'Escape' && selectedBursa) {
      toggleBursa(); return;
    } else if (event.key === '+' || event.key === '-' || event.key === '=') {
      camera.zoom = clamp(camera.zoom*(event.key === '-' ? .85 : 1.18), controls.minZoom,controls.maxZoom);
    } else if (event.key === 'Home') {
      setView(); event.preventDefault(); return;
    } else if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) {
      const displacement = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(displacement);
      if (event.key==='ArrowLeft') spherical.theta -= .10;
      if (event.key==='ArrowRight') spherical.theta += .10;
      if (event.key==='ArrowUp') spherical.phi -= .10;
      if (event.key==='ArrowDown') spherical.phi += .10;
      camera.position.copy(controls.target).add(displacement.setFromSpherical(spherical.makeSafe()));
    } else return;
    camera.updateProjectionMatrix(); controls.update(); queue(); event.preventDefault();
  }

  function cameraChanged() {
    queue();
    onCameraChange?.({position:camera.position.toArray(), target:controls.target.toArray(), zoom:camera.zoom});
  }
  function setCameraPose({position, target, zoom}) {
    camera.position.fromArray(position);
    controls.target.fromArray(target);
    camera.zoom = zoom;
    camera.updateProjectionMatrix();
    controls.update();
    queue();
  }
  controls.addEventListener('change',cameraChanged);
  canvas.addEventListener('pointermove',inspect);
  canvas.addEventListener('pointerdown',onTouchStart);
  canvas.addEventListener('pointerup',onTouchEnd);
  canvas.addEventListener('pointerleave',onPointerLeave);
  canvas.addEventListener('keydown',onKey);
  const observer = new ResizeObserver(queue);
  observer.observe(host);
  setLayers();
  setView();
  return {
    setFrame, setLayers, setView, setCameraPose, setReferenceFrame, reset:() => setView(),
    getBursaTravelMm(sample) {
      if (!bursaMechanics) return null;
      if (!sample) return bursaTravelMm;
      hipRotation(sample,referencePose);
      return bursaMechanics.excursionMm(referencePose,contact.solve(referencePose).positions);
    },
    dispose() {
      disposed = true; cancelAnimationFrame(request);
      observer.disconnect(); controls.dispose();
      canvas.removeEventListener('pointermove',inspect); canvas.removeEventListener('pointerleave',onPointerLeave); canvas.removeEventListener('keydown',onKey);
      canvas.removeEventListener('pointerdown',onTouchStart);
      canvas.removeEventListener('pointerup',onTouchEnd);
      clearTimeout(hideTooltip);
      bursaLabel?.removeEventListener('click',toggleBursa);
      referenceComparison?.dispose();
      deformation.dispose();
      skeleton.dispose();
      radiance.dispose();
      guides.dispose();
      visibility.dispose();
      for (const g of new Set(surfaces.map(m => m.geometry))) g.dispose();
      for (const m of surfaces) m.material.dispose();
      renderer.dispose();
      overlay.remove(); hover.remove(); canvas.remove();
    },
  };
}
