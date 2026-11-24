import { Euler, MathUtils, Quaternion, Vector3 } from 'three';
import { createHipScene, loadHipAtlas } from '../hip-study/hip.js';
import { HIP_PROFILES } from '../hip-study/hip-anatomy.js';
import { prepareHipContact } from '../hip-study/hip-contact.js';
import { createBursaMechanics } from '../hip-study/bursa-mechanics.js';
import reference from '../../hip-study/data/bursa-mechanics.json';
import './tissue.css';

const svgNS = 'http://www.w3.org/2000/svg';
const radians = MathUtils.degToRad;
const degrees = value => Number(value) + '°';
const IDENTITY = { side: 'right', hipFlexionDeg: 0, hipAdductionDeg: 0, hipRotationDeg: 0 };
const bandLandmark = HIP_PROFILES.surface.anchors.find(anchor => anchor.label === reference.band);
const bursaLandmark = {label:reference.bursa, text:'Trochanteric bursa', compact:'Bursa', xyz:reference.landmarkM, side:1, narrow:true};
const landmarks = [bandLandmark, bursaLandmark];

function svg(tag, attributes) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

// Draw one of the three band samples for each lateral-facing bursa triangle.
// Rest and posed band positions are compared in the moving bursa's coordinates.
function materialPlot(plot) {
  const { vertices, samples } = reference.evaluator;
  const vertex = new Vector3();
  const rest = new Vector3();
  const posed = new Vector3();
  const delta = new Vector3();
  const normal = new Vector3();
  const inverse = new Quaternion();
  const paths = samples.filter((_, index) => index % 3 === 0).map(sample => {
    rest.set(0, 0, 0);
    for (let i = 0; i < 3; i++) {
      const [, x, y, z] = vertices[sample[4 + i]];
      rest.addScaledVector(vertex.set(x, y, z), sample[7 + i]);
    }
    return { sample, origin: rest.clone() };
  });
  const center = paths.reduce((sum, p) => sum.add(p.origin), new Vector3()).multiplyScalar(1 / paths.length);
  const pxPerMm = 2.8;
  const x = p => 130 + (p.z - center.z) * 1000 * pxPerMm;
  const y = p => 78 - (p.y - center.y) * 1000 * pxPerMm;
  const traces = svg('path', { fill: 'none', stroke: '#66849a', 'stroke-width': 0.85 });
  const origins = svg('g', { fill: '#9e5138' });
  const tips = svg('g', { fill: '#365e79' });
  for (const item of paths) {
    origins.append(svg('circle', {cx: x(item.origin), cy: y(item.origin), r: 1.65}));
    item.tip = svg('circle', {r: 1.5});
    tips.append(item.tip);
  }
  plot.append(traces, origins, tips);
  const ruler = svg('path', {d: 'M218 114v5m0-2h14m0-3v5', stroke: '#43545d', fill: 'none'});
  const legend = svg('text', {x: 225, y: 108, 'text-anchor': 'middle', fill: '#43545d', 'font-size': 10});
  legend.textContent = '5 mm';
  plot.append(ruler, legend);
  return function update(pose, positions) {
    inverse.copy(pose).invert();
    let d = '';
    for (const {sample, origin, tip} of paths) {
      posed.set(0, 0, 0);
      for (let i = 0; i < 3; i++) {
        const vertexId = vertices[sample[4 + i]][0];
        posed.addScaledVector(vertex.fromArray(positions, 3 * vertexId), sample[7 + i]);
      }
      delta.copy(posed).applyQuaternion(inverse).sub(origin);
      normal.fromArray(sample, 1);
      delta.addScaledVector(normal, -delta.dot(normal));
      posed.copy(origin).add(delta);
      const tx = x(posed).toFixed(2), ty = y(posed).toFixed(2);
      tip.setAttribute('cx', tx); tip.setAttribute('cy', ty);
      d += 'M' + x(origin).toFixed(2) + ' ' + y(origin).toFixed(2) + 'L' + tx + ' ' + ty;
    }
    traces.setAttribute('d', d);
  };
}

// A lateral orthographic projection uses the segmented bone and bursa contours
// and the same solved band positions when a WebGL context is unavailable.
function lateralProjection(host, atlas) {
  const canvas = document.createElement('canvas');
  canvas.className = 'tl-projection';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Lateral projection of the hip, iliotibial band and trochanteric bursa');
  host.append(canvas);
  const ctx = canvas.getContext('2d');
  if (!ctx) { canvas.remove(); return null; }
  const center = new Vector3().fromArray(atlas.parser.json.asset.extras.femoralHeadSphere.centerM);
  const vertex = new Vector3();
  const meshes = new Map();
  const names = ['Hip bone', 'Femur', reference.bursa, 'Iliotibial tract'];
  atlas.scene.traverse(object => {
    const name = object.userData.label;
    if (object.isMesh && names.includes(name)) meshes.set(name, object.geometry);
  });
  const landmarkIndices = new Map();
  for (const {label, xyz} of landmarks) {
    const positions = meshes.get(label)?.attributes.position;
    const target = new Vector3().fromArray(xyz);
    let least = Infinity;
    for (let i = 0; i < (positions?.count ?? 0); i++) {
      const distance = vertex.fromBufferAttribute(positions, i).distanceToSquared(target);
      if (distance < least) { least = distance; landmarkIndices.set(label, i); }
    }
  }
  const colors = ['#a9987b', '#c8b596', '#8eacb5', '#d4c2a6'];
  return function paint(pose, result, layer) {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    const ratio = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    ctx.scale(ratio, ratio);
    const scale = Math.min(w / .35, h / .35);
    const landmark = layer === 'bursa' ? bursaLandmark : bandLandmark;
    let endpoint;
    for (const [number, name] of names.entries()) {
      if (layer === 'bursa' && name === 'Iliotibial tract') continue;
      const geometry = meshes.get(name);
      if (!geometry) continue;
      const position = geometry.attributes.position;
      const projected = new Float32Array(position.count * 2);
      for (let i = 0; i < position.count; i++) {
        if (name === 'Iliotibial tract') {
          vertex.fromArray(result.positions, 3 * result.packedToWelded[i]);
        } else {
          vertex.fromBufferAttribute(position, i).sub(center);
          if (name !== 'Hip bone') vertex.applyQuaternion(pose);
        }
        projected[2 * i] = w * .5 + scale * (vertex.z + .01);
        projected[2 * i + 1] = h * .48 - scale * vertex.y;
      }
      if (name === landmark.label && landmarkIndices.has(name)) {
        const i = 2 * landmarkIndices.get(name);
        endpoint = [projected[i], projected[i + 1]];
      }
      const indices = geometry.index.array;
      ctx.fillStyle = colors[number];
      ctx.beginPath();
      for (let i = 0; i < indices.length; i += 3) {
        const a = 2 * indices[i], b = 2 * indices[i + 1], c = 2 * indices[i + 2];
        const ax = projected[a], ay = projected[a + 1], bx = projected[b], by = projected[b + 1];
        const cx = projected[c], cy = projected[c + 1];
        if ((bx - ax) * (cy - ay) <= (by - ay) * (cx - ax)) continue;
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx, cy); ctx.closePath();
      }
      ctx.fill();
    }
    if (endpoint) {
      const [x, y] = endpoint;
      const title = w < 415 ? landmark.compact : landmark.text;
      ctx.font = '500 12px system-ui, sans-serif';
      const labelWidth = ctx.measureText(title).width;
      const labelX = Math.min(w - labelWidth - 12, x + w * .20);
      const labelY = Math.max(44, Math.min(h - 46, y - h * .08));
      ctx.strokeStyle = '#c6d7d8'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(labelX-5,labelY); ctx.stroke();
      ctx.beginPath(); ctx.arc(x,y,2.5,0,2*Math.PI); ctx.fillStyle = '#eff5ef'; ctx.fill();
      ctx.fillStyle = '#14232d'; ctx.fillRect(labelX-2,labelY-8,labelWidth+5,16);
      ctx.textBaseline = 'middle'; ctx.fillStyle = '#f2f3ee'; ctx.fillText(title,labelX,labelY);
    }
  };
}

export function mountTissue(root) {
  const container = document.createElement('section');
  container.className = 'tissue-lab';
  container.setAttribute('aria-label', 'Iliotibial band motion over the trochanteric bursa');
  container.innerHTML = `
    <figure class="tl-stage">
      <div class="tl-canvas"></div>
      <div class="tl-tag"><span class="tl-marker"></span><span data-view-label>Right hip · dissection</span></div>
      <p class="tl-render-status" role="status">Loading hip dissection…</p>
      <div class="tl-layers" role="group" aria-label="Dissection layer">
        <button type="button" data-layer="band" aria-pressed="true">IT band</button>
        <button type="button" data-layer="bursa" aria-pressed="false">Expose bursa</button>
      </div>
      <div class="tl-orbit">Drag to rotate</div>
    </figure>
    <div class="tl-controls">
      <div class="tl-measure">
        <div class="tl-label">Band travel along the bursa</div>
        <div class="tl-value"><output data-travel>…</output> <span>mm</span></div>
        <p>Movement of the band’s surface relative to its neutral position on the outer hip.</p>
      </div>
      <div class="tl-inputs">
        <label>Hip adduction <output data-add-value>20°</output>
          <input data-angle="add" aria-label="Hip adduction" type="range" min="-20" max="20" step="1" value="20">
        </label>
        <label>Hip flexion <output data-flex-value>0°</output>
          <input data-angle="flex" aria-label="Hip flexion" type="range" min="-10" max="30" step="1" value="0">
        </label>
        <button class="tl-neutral" type="button">Return to neutral</button>
      </div>
      <figure class="tl-footprint">
        <div class="tl-label">Band surface tracks</div>
        <svg viewBox="0 0 260 132" role="img" aria-label="Side projection of modeled material displacement across the bursa"></svg>
        <figcaption><span><i></i>Neutral</span><span><i></i>Current pose</span></figcaption>
      </figure>
      <details class="tl-method"><summary>How travel is calculated</summary>
        <p>At each pose a geometric contact solver moves the band around the underlying tissues. It follows 237 band points against the bursa and calculates their area-weighted root-mean-square tangential displacement. The map shows one track per sampled triangle from the side.</p>
        <p>The readout describes modeled sliding in a shared anatomical atlas. Measuring pressure or predicting inflammation requires additional tissue data.</p>
      </details>
    </div>`;
  root.append(container);
  const viewport = container.querySelector('.tl-canvas');
  const renderStatus = container.querySelector('.tl-render-status');
  const inputs = container.querySelector('.tl-inputs');
  const value = container.querySelector('[data-travel]');
  const adduction = container.querySelector('[data-angle=add]');
  const flexion = container.querySelector('[data-angle=flex]');
  const updateMap = materialPlot(container.querySelector('.tl-footprint svg'));
  const measure = createBursaMechanics(reference);
  const rotation = new Quaternion(), euler = new Euler();
  let scene, contact, projection, frameRequest = 0, disposed = false, active = true, layer = 'band';

  const viewButtons = [...container.querySelectorAll('[data-layer]')];
  function showLayer() {
    const exposed = layer === 'bursa';
    for (const button of viewButtons) button.setAttribute('aria-pressed', String(button.dataset.layer === layer));
    scene?.setLayers({ labels: true, band: !exposed, muscle: !exposed, tendons: false, bursae: true, fascia: false, loads: false });
    queue();
  }
  function draw() {
    frameRequest = 0;
    if (disposed || !active) return;
    const add = Number(adduction.value), flex = Number(flexion.value);
    container.querySelector('[data-add-value]').value = degrees(add);
    container.querySelector('[data-flex-value]').value = degrees(flex);
    container.querySelector('.tl-neutral').disabled = add === 0 && flex === 0;
    scene?.setFrame({...IDENTITY, hipAdductionDeg:add, hipFlexionDeg:flex});
    if (!contact) return;
    rotation.setFromEuler(euler.set(-radians(flex), 0, radians(add), 'ZXY'));
    const result = contact.solve(rotation);
    const { positions } = result;
    value.value = measure.excursionMm(rotation, positions).toFixed(2);
    updateMap(rotation, positions);
    projection?.(rotation, result, layer);
  }
  function queue() {
    if (!disposed && active && !frameRequest) frameRequest = requestAnimationFrame(draw);
  }
  inputs.addEventListener('input', queue);
  container.querySelector('.tl-neutral').addEventListener('click', () => {
    adduction.value = '0'; flexion.value = '0'; queue();
  });
  viewButtons.forEach(button => button.addEventListener('click', () => { layer = button.dataset.layer; showLayer(); }));
  viewport.addEventListener('keydown', event => {
    if (event.defaultPrevented) event.stopPropagation();
  });
  const resize = new ResizeObserver(() => { if (projection) queue(); });
  resize.observe(viewport);
  loadHipAtlas().then(async atlas => {
    if (disposed) return;
    contact = await prepareHipContact(atlas);
    if (disposed) return;
    draw();
    try {
      const view = await createHipScene(viewport, { profile:'overview', landmarks });
      if (disposed) { view.dispose(); return; }
      scene = view;
      scene.setView('lateral');
      renderStatus.hidden = true;
      showLayer();
      queue();
    } catch {
      if (disposed) return;
      container.classList.add('tl-map-only');
      projection = lateralProjection(viewport, atlas);
      container.querySelector('[data-view-label]').textContent = 'Right hip · lateral projection';
      renderStatus.hidden = Boolean(projection);
      if (!projection) renderStatus.textContent = 'See live surface tracks alongside the controls.';
      queue();
    }
  }).catch(() => {
    if (disposed) return;
    renderStatus.textContent = 'The hip contact assets could not load.';
    container.querySelectorAll('input, button').forEach(control => control.disabled = true);
  });
  return {
    setActive(next) { active = Boolean(next); if (active) queue(); },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frameRequest);
      inputs.removeEventListener('input', queue);
      resize.disconnect();
      scene?.dispose();
      container.remove();
    },
  };
}
