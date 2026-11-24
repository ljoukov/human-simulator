import { createBodyView } from './body.js';
import { createHipScene } from './hip.js';
import { createDeepTravelReadout } from './deep-travel.js';

const datasets = import.meta.glob(
  '../../hip-study/data/{motion,motion-public,results,counterfactual,benchmark,benchmark-metal,benchmark-cpu,bootstrap,bootstrap-sensitivity,benchmark-metal-sensitivity,benchmark-pareto,bursa-mechanics}.json',
  { eager: true, import: 'default' },
);
const data = (name) => datasets[`../../hip-study/data/${name}.json`];
const $ = (id) => document.getElementById(id);
const motionEmbed = document.documentElement.classList.contains('motion-embed');
let embedClosed = false;
const colors = { teal: '#187368', coral: '#bd6049', gray: '#808574' };
const fmt = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : 'N/A');
const percent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'N/A');
const fileNumber = (value) => (Number.isFinite(value) ? value.toLocaleString('en-US') : 'N/A');
const sourceLink = (value) => typeof value === 'string' && /^https?:\/\//.test(value);
const svgNS = 'http://www.w3.org/2000/svg';
const chartRegistry = new WeakMap();
const phases = { motion: 0.2, mechanisms: 0.2 };
const playing = { motion: false, mechanisms: false };
let lastTick = 0;
const motionData = data('motion') ?? data('motion-public');
const travelByClip = new Map((data('bursa-mechanics')?.clipChecks ?? []).map(c => [c.clip, c.chartExcursionMm]));
const results = data('results');
const counterfactual = data('counterfactual');
let active = 'motion';
let currentClip, currentLoad, referenceWalk, travelFrames, motionCursor, pairCursor;
let activePairTask = 'walk';

function svg(name, attributes, parent) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, val] of Object.entries(attributes)) node.setAttribute(key, val);
  parent.append(node);
  return node;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawChart(
  host,
  {
    series,
    frames,
    yLabel,
    title,
    band,
    scrub,
    cursorPointColor,
    axisLabel = 'Phase (%)',
    xTicks = [
      [0, '0'],
      [0.25, '25'],
      [0.5, '50'],
      [0.75, '75'],
      [1, '100'],
    ],
  },
) {
  chartRegistry.get(host)?.disconnect();
  let currentPhase;
  let currentValue;
  let cursor;
  let cursorPoint;
  let plot;
  let bounds;
  const rows = frames ?? [];
  function render() {
    host.replaceChildren();
    cursor = null;
    cursorPoint = null;
    const rawValues = rows
      .flatMap((row) => [...series.map((s) => row[s.key]), ...(band ? [row[band[0]], row[band[1]]] : [])])
      .filter(Number.isFinite);
    if (!rawValues.length) {
      host.append(element('span', 'chart-empty', 'No samples for this output.'));
      return;
    }
    const width = Math.max(270, host.clientWidth || 520),
      height = host.clientHeight || 155;
    const left = 42,
      right = width - 13,
      top = 18,
      bottom = height - 29;
    const min = Math.min(0, ...rawValues),
      max = Math.max(0, ...rawValues);
    const span = Math.max(0.1, max - min);
    const size = 10 ** Math.floor(Math.log10(span / 3));
    const step = ([1, 2, 2.5, 5, 10].find((x) => x >= span / (3 * size)) || 10) * size;
    const low = Math.floor(min / step) * step,
      high = Math.max(low + step, Math.ceil(max / step) * step);
    const x = (phase) => left + phase * (right - left);
    const y = (v) => bottom - ((v - low) / (high - low)) * (bottom - top);
    plot = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': title }, host);
    const desc = svg('desc', {}, plot);
    desc.textContent =
      `${title}. ${rows.length} sampled phases. Vertical unit ${yLabel}. ` +
      series
        .map((s) => {
          const vs = rows.map((row) => row[s.key]).filter(Number.isFinite);
          return vs.length ? `${s.label} ranges from ${fmt(Math.min(...vs))} to ${fmt(Math.max(...vs))}.` : '';
        })
        .join(' ');
    for (let v = low; v <= high + step * 0.01; v += step) {
      svg('line', { x1: left, x2: right, y1: y(v), y2: y(v), stroke: '#dee3d9', 'stroke-width': 1 }, plot);
      svg('text', { x: left - 7, y: y(v) + 3, 'text-anchor': 'end' }, plot).textContent = Number(
        v.toFixed(2),
      ).toString();
    }
    svg('text', { x: 0, y: 9 }, plot).textContent = yLabel;
    for (const [position, label] of xTicks) {
      svg('text', { x: x(position), y: height - 13, 'text-anchor': 'middle' }, plot).textContent = label;
    }
    svg('text', { x: (left + right) / 2, y: height - 1, 'text-anchor': 'middle' }, plot).textContent = axisLabel;
    if (band) {
      const valid = rows.filter((row) => band.every((key) => Number.isFinite(row[key])));
      if (valid.length) {
        const coords = [
          ...valid.map((row) => [x(row.phase), y(row[band[0]])]),
          ...valid.toReversed().map((row) => [x(row.phase), y(row[band[1]])]),
        ];
        svg('polygon', { points: coords.map((p) => p.join(',')).join(' '), fill: '#edcec0', opacity: 0.65 }, plot);
      }
    }
    for (const s of series) {
      let start = true;
      const path = rows
        .map((row) => {
          if (!Number.isFinite(row[s.key])) {
            start = true;
            return '';
          }
          const cmd = start ? 'M' : 'L';
          start = false;
          return `${cmd}${fmt(x(row.phase))},${fmt(y(row[s.key]))}`;
        })
        .join(' ');
      svg(
        'path',
        {
          d: path,
          fill: 'none',
          stroke: s.color,
          'stroke-width': s.dashed ? 1.7 : 2.6,
          'stroke-dasharray': s.dashed ? '6 4' : 'none',
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        },
        plot,
      );
    }
    cursor = svg(
      'line',
      { y1: top, y2: bottom, stroke: '#294e43', 'stroke-width': 1, 'stroke-dasharray': '3 3' },
      plot,
    );
    cursor.style.display = 'none';
    if (cursorPointColor) {
      cursorPoint = svg('circle', { r: 3.5, fill: cursorPointColor, stroke: 'white', 'stroke-width': 1.3 }, plot);
      cursorPoint.style.display = 'none';
    }
    bounds = { left, right, y };
    if (currentPhase !== undefined) moveCursor(currentPhase, currentValue);
    if (scrub) {
      plot.style.cursor = 'crosshair';
      plot.addEventListener('pointerdown', (e) => {
        const value = (e.clientX - plot.getBoundingClientRect().left - left) / (right - left);
        scrub(Math.max(0, Math.min(1, value)));
      });
    }
  }
  function moveCursor(phase, value) {
    currentPhase = phase;
    currentValue = value;
    if (!cursor) return;
    cursor.style.display = '';
    const pos = bounds.left + phase * (bounds.right - bounds.left);
    cursor.setAttribute('x1', pos);
    cursor.setAttribute('x2', pos);
    if (cursorPoint) {
      cursorPoint.style.display = Number.isFinite(value) ? '' : 'none';
      if (Number.isFinite(value)) {
        cursorPoint.setAttribute('cx', pos);
        cursorPoint.setAttribute('cy', bounds.y(value));
      }
    }
  }
  const observer = new ResizeObserver(render);
  chartRegistry.set(host, observer);
  observer.observe(host);
  render();
  return moveCursor;
}

function asOption(value, label) {
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label;
  return opt;
}

function sectionError(id, message) {
  $(id).textContent = message;
  $(id).hidden = false;
}

function metric(label, value, foot) {
  const card = element('div', 'metric');
  card.append(element('span', 'eyebrow', label), element('strong', '', value));
  if (foot) card.append(element('small', '', foot));
  return card;
}

function scalarAt(frames, phase, keys) {
  if (!frames?.length) return {};
  const length = frames.length;
  let i = 0,
    j = length - 1;
  while (i < j) {
    const m = (i + j) >> 1;
    if ((frames[m].phase ?? m / Math.max(1, length - 1)) < phase) i = m + 1;
    else j = m;
  }
  const a = frames[Math.max(0, i - 1)],
    b = frames[i],
    t0 = a.phase ?? Math.max(0, i - 1) / (length - 1),
    t1 = b.phase ?? i / (length - 1);
  const mix = t1 > t0 ? Math.min(1, Math.max(0, (phase - t0) / (t1 - t0))) : 0;
  const out = {};
  for (const key of keys) {
    if (Number.isFinite(a[key]) && Number.isFinite(b[key])) out[key] = a[key] + (b[key] - a[key]) * mix;
    else if (Number.isFinite(b[key])) out[key] = b[key];
  }
  return out;
}
const poseKeys = [
  'hipFlexionDeg',
  'hipAdductionDeg',
  'hipRotationDeg',
  'pelvisTiltDeg',
  'pelvisObliquityDeg',
  'pelvisRotationDeg',
  'trunkLeanDeg',
  'abductorForceBw',
];

const panes = [
  { id: 'body', title: 'Full body', controlId: 'body', description: 'Reference skeleton' },
  {
    id: 'surface',
    title: 'Superficial hip',
    controlId: 'hip',
    description: 'Gluteal muscles · fascia · iliotibial band',
    layers: {
      bone: true,
      muscle: true,
      fascia: true,
      envelope: false,
      band: true,
      bursae: false,
      labels: true,
      loads: true,
    },
  },
  {
    id: 'deep',
    title: 'Deep gluteal tissues',
    controlId: 'deep',
    description: 'Gluteus minimus · rotators · tendons',
    layers: {
      bone: true,
      muscle: true,
      medius: false,
      rotators: true,
      tendons: true,
      bursae: true,
      nerves: false,
      vessels: false,
      paths: false,
      labels: true,
      loads: true,
    },
  },
  {
    id: 'joint',
    title: 'Articular surfaces',
    controlId: 'joint',
    description: 'Exploded joint · schematic spacing',
    layers: {
      bone: true,
      cartilage: true,
      labrum: true,
      ligaments: true,
      capsule: false,
      separate: true,
      labels: true,
      loads: true,
    },
  },
];
const deepPane = panes.find(pane => pane.id === 'deep');
const layerChoices = {
  bone: ['Bone', '#b8a991'],
  cartilage: ['Cartilage', '#68adab'],
  muscle: ['Muscle', '#c4796b'],
  tendons: ['Tendon', '#d7b99c', 'Schematic distal tendon regions of the gluteal muscle surfaces.'],
  fascia: ['Aponeurosis', '#d9c5aa'],
  envelope: [
    'Fascia lata',
    '#c5c3af',
    'Outer fascial envelope. Adjacent model surfaces can intersect during motion.',
  ],
  band: ['IT band', '#b6b1a1'],
  bursae: ['Bursae', '#88babc', 'Segmented bursa surfaces show their anatomical positions.'],
  medius: ['Overlying medius', '#b86d66'],
  rotators: ['Short rotators', '#c68a80'],
  nerves: ['Nerves', '#d6b755'],
  vessels: ['Vessels', '#af757d'],
  paths: [
    'Model guides',
    '#5b8c98',
    'Published muscle and iliotibial-band control points joined in the reference frame.',
  ],
  labrum: ['Labrum', '#a0c3ba'],
  ligaments: ['Ligaments', '#cbbd9d'],
  capsule: ['Capsule', '#d9cab4'],
  separate: [
    'Exploded view',
    '#858f99',
    'Separates the femoral head and socket for inspection. Spacing is schematic.',
  ],
  loads: ['Load colors', '#8c9873', 'Color tissue by model-estimated force during instrumented trials.'],
  labels: ['Labels', '#62746d'],
};
const bodyModes = [
  ['all', 'Hip muscles'],
  ['abductors', 'Abductors'],
  ['bone', 'Skeleton'],
];
let bodyMode = 'all';
let inspectedPane;

function positionInspector() {
  if (!inspectedPane) return;
  const panel = $('layer-inspector');
  const trigger = inspectedPane.layersButton.getBoundingClientRect();
  const margin = 8;
  const left = Math.max(
    margin,
    Math.min(window.innerWidth - margin - panel.offsetWidth, trigger.right - panel.offsetWidth),
  );
  const below = trigger.bottom + 5;
  const top =
    below + panel.offsetHeight <= window.innerHeight - margin
      ? below
      : Math.max(margin, trigger.top - panel.offsetHeight - 5);
  panel.style.left = left + 'px';
  panel.style.top = top + 'px';
}

function closeInspector(returnFocus = false) {
  if (!inspectedPane) return;
  const trigger = inspectedPane.layersButton;
  inspectedPane = undefined;
  const strip = document.querySelector('.layer-strip');
  $('layer-inspector').hidden = true;
  strip.insertBefore($('layers'), strip.querySelector('.orbit-help'));
  strip.hidden = false;
  trigger.setAttribute('aria-expanded', 'false');
  if (returnFocus) trigger.focus({ preventScroll: true });
}

function inspectLayers(pane) {
  if (inspectedPane === pane) return closeInspector(true);
  closeInspector();
  inspectedPane = pane;
  $('layer-target').value = pane.id;
  renderLayers();
  const panel = $('layer-inspector');
  $('inspector-title').textContent = pane.title + ' layers';
  panel.append($('layers'));
  document.querySelector('.layer-strip').hidden = true;
  panel.hidden = false;
  pane.layersButton.setAttribute('aria-expanded', 'true');
  positionInspector();
  panel.querySelector('#layers button:not(:disabled)').focus({ preventScroll: true });
}

function renderLayers() {
  const pane = panes.find((p) => p.id === $('layer-target').value);
  const controls = $('layers');
  controls.replaceChildren();
  if (pane.id === 'body') {
    for (const [mode, label] of bodyModes) {
      const b = element('button', '', label);
      b.type = 'button';
      b.dataset.mode = mode;
      b.style.setProperty('--layer-color', mode === 'bone' ? '#b8a991' : '#7aa79b');
      b.setAttribute('aria-pressed', String(mode === bodyMode));
      b.onclick = () => {
        bodyMode = mode;
        pane.view?.setTissueMode?.(mode);
        for (const button of controls.children) button.setAttribute('aria-pressed', String(button === b));
      };
      controls.append(b);
    }
    return;
  }
  for (const [key, selected] of Object.entries(pane.layers)) {
    const [label, color, hint] = layerChoices[key];
    const b = element('button', '', label);
    b.type = 'button';
    b.dataset.layer = key;
    b.title = hint || label;
    b.style.setProperty('--layer-color', color);
    b.disabled = key === 'loads' && !currentLoad;
    b.setAttribute('aria-pressed', String(selected && !b.disabled));
    b.onclick = () => {
      pane.layers[key] = !pane.layers[key];
      pane.view?.setLayers({ [key]: pane.layers[key] });
      b.setAttribute('aria-pressed', String(pane.layers[key]));
      if (key === 'separate')
        pane.caption.textContent = pane.layers.separate
          ? pane.description
          : 'Femoral head seated in the acetabulum';
    };
    controls.append(b);
  }
}

function setLayout(id) {
  closeInspector();
  const focused = panes.find((p) => p.id === id);
  if (gridScreen) gridScreen.layout = focused?.id || 'all';
  $('motion-stages').dataset.focus = focused?.id || 'all';
  $('layout-view').value = focused?.id || 'all';
  for (const pane of panes) {
    pane.node.hidden = Boolean(focused) && pane !== focused;
    pane.expand.setAttribute('aria-pressed', String(pane === focused));
    const label = pane === focused ? 'Restore all four views' : 'Enlarge ' + pane.title.toLowerCase();
    pane.expand.setAttribute('aria-label', label);
    pane.expand.title = label;
  }
  if (focused) {
    $('layer-target').value = focused.id;
    renderLayers();
  }
}

let gridScreen;
const nativeFullscreenElement = () => document.fullscreenElement || document.webkitFullscreenElement;

function restoreGrid() {
  const previous = gridScreen;
  if (!previous) return;
  gridScreen = undefined;
  closeInspector();
  const grid = $('motion-stages');
  grid.classList.remove('grid-fullscreen', 'grid-fallback');
  document.documentElement.classList.remove('grid-open');
  grid.before($('motion-transport'));
  $('exit-grid-fullscreen').hidden = true;
  $('enter-grid-fullscreen').hidden = false;
  document.querySelector('.layer-strip').after($('layer-inspector'));
  setLayout(previous.layout);
  requestAnimationFrame(() => {
    if (active !== 'motion' || gridScreen) return;
    window.scrollTo({ top: previous.scrollY, behavior: 'instant' });
    $('enter-grid-fullscreen').focus({ preventScroll: true });
  });
}

async function exitGridScreen() {
  const exiting = gridScreen;
  if (!exiting) return;
  if (nativeFullscreenElement() === $('motion-stages')) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    try {
      await exit.call(document);
    } catch {
      /* The browser can finish its own exit first. */
    }
  }
  if (gridScreen === exiting) restoreGrid();
}

async function enterGridScreen() {
  if (gridScreen) return;
  const grid = $('motion-stages');
  const session = { layout: $('layout-view').value, scrollY: window.scrollY, native: false };
  setLayout('all');
  gridScreen = session;
  grid.prepend($('motion-transport'));
  grid.append($('layer-inspector'));
  document.documentElement.classList.add('grid-open');
  grid.classList.add('grid-fullscreen', 'grid-fallback');
  $('enter-grid-fullscreen').hidden = true;
  $('exit-grid-fullscreen').hidden = false;
  const request = grid.requestFullscreen || grid.webkitRequestFullscreen;
  if (request) {
    try {
      await request.call(grid);
      if (gridScreen === session && nativeFullscreenElement() === grid) {
        session.native = true;
        grid.classList.remove('grid-fallback');
      } else if (!gridScreen && nativeFullscreenElement() === grid) {
        // The user closed the overlay while the browser was processing the request.
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        await exit.call(document);
      }
    } catch {
      /* Keep the viewport overlay when the embedding browser declines native fullscreen. */
    }
  }
  if (gridScreen === session) $('exit-grid-fullscreen').focus({ preventScroll: true });
}

function onNativeScreenChange() {
  if (!gridScreen) return;
  if (nativeFullscreenElement() === $('motion-stages')) {
    gridScreen.native = true;
    $('motion-stages').classList.remove('grid-fallback');
  } else if (gridScreen.native) restoreGrid();
}

function createMotionPanes() {
  for (const pane of panes) {
    $('layout-view').append(asOption(pane.id, pane.title));
    $('layer-target').append(asOption(pane.id, pane.title));
    const node = element('section', 'scene motion-pane');
    node.dataset.pane = pane.id;
    node.setAttribute('aria-label', pane.title + ' 3D view');
    const header = element('header', 'scene-head');
    const name = element('div');
    name.append(element('span', 'signal'), element('h3', '', pane.title));
    const tools = element('span', 'pane-tools');
    const camera = element('select');
    camera.id = pane.controlId + '-view';
    camera.setAttribute('aria-label', pane.title + ' camera');
    const cameras =
      pane.id === 'body'
        ? [
            ['oblique', 'Oblique'],
            ['front', 'Front'],
            ['side', 'Side'],
            ['back', 'Back'],
          ]
        : [
            ['oblique', 'Dissection view'],
            ['lateral', 'Lateral'],
            ['anterior', 'Anterior'],
            ['posterior', 'Posterior'],
          ];
    for (const [id, title] of cameras) camera.append(asOption(id, title));
    const expand = element('button', 'pane-expand');
    expand.type = 'button';
    const icon = element('i');
    icon.setAttribute('aria-hidden', 'true');
    expand.append(icon);
    const layersButton = element('button', 'pane-layers-button');
    layersButton.type = 'button';
    layersButton.setAttribute('aria-label', 'Layers for ' + pane.title.toLowerCase());
    layersButton.setAttribute('aria-haspopup', 'dialog');
    layersButton.setAttribute('aria-expanded', 'false');
    layersButton.setAttribute('aria-controls', 'layer-inspector');
    layersButton.title = 'Inspect ' + pane.title.toLowerCase() + ' layers';
    const symbol = svg(
      'svg',
      {
        viewBox: '0 0 16 16',
        width: 14,
        height: 14,
        'aria-hidden': 'true',
        fill: 'none',
        stroke: 'currentColor',
        'stroke-width': 1.4,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      },
      layersButton,
    );
    svg(
      'path',
      { d: 'm1.5 5.2 6.5-3.1 6.5 3.1-6.5 3.1z M1.6 8.2 8 11.3l6.4-3.1 M1.6 11.1 8 14.2l6.4-3.1' },
      symbol,
    );
    layersButton.append(element('span', '', 'Layers'));
    tools.append(camera, layersButton, expand);
    header.append(name, tools);
    const host = element('div', 'canvas-host');
    host.id = pane.controlId + '-canvas';
    host.append(element('span', 'waiting', 'Loading anatomy…'));
    const foot = element('div', 'stage-foot');
    const caption = element('span', '', pane.description);
    caption.id = pane.id === 'body' ? 'body-source' : pane.id + '-caption';
    const reset = element('button', '', 'Reset view');
    reset.type = 'button';
    reset.id = 'reset-' + pane.controlId;
    reset.setAttribute('aria-label', 'Reset ' + pane.title.toLowerCase() + ' camera');
    foot.append(caption, reset);
    if (pane === deepPane) {
      const ghost = element('button', 'ghost-toggle', '100% reference ghost');
      ghost.id = 'deep-ghost-toggle';
      ghost.type = 'button';
      ghost.hidden = true;
      ghost.setAttribute('aria-pressed', 'true');
      ghost.setAttribute('aria-controls', host.id);
      ghost.title = 'Overlay the 100% capacity walk at the same stride phase, aligned at the hip center.';
      ghost.onclick = () => {
        ghost.setAttribute('aria-pressed', String(ghost.getAttribute('aria-pressed') !== 'true'));
        redrawMotion();
      };
      foot.insertBefore(ghost, reset);
      deepPane.ghostToggle = ghost;
    }
    node.append(header, host, foot);
    if (pane === deepPane) deepPane.travel = createDeepTravelReadout(node);
    $('motion-stages').append(node);
    Object.assign(pane, { node, host, expand, caption, camera, layersButton });
    layersButton.onclick = () => inspectLayers(pane);
    expand.onclick = () => setLayout($('layout-view').value === pane.id ? 'all' : pane.id);
    camera.onchange = () => pane.view?.setView(camera.value);
    reset.onclick = () => {
      pane.view?.reset();
      camera.value = 'oblique';
    };
  }
  $('layer-target').value = 'surface';
  $('layer-target').onchange = renderLayers;
  $('layout-view').onchange = (e) => setLayout(e.target.value);
  const grid = $('motion-stages');
  function placeOverlays() {
    if (gridScreen) {
      const viewport = window.visualViewport;
      grid.style.setProperty('--screen-top', (viewport?.offsetTop || 0) + 'px');
      grid.style.setProperty('--screen-left', (viewport?.offsetLeft || 0) + 'px');
      grid.style.setProperty('--screen-width', (viewport?.width || window.innerWidth) + 'px');
      grid.style.setProperty('--screen-height', (viewport?.height || window.innerHeight) + 'px');
    }
    positionInspector();
  }
  window.addEventListener('resize', placeOverlays);
  window.addEventListener('scroll', positionInspector, { passive: true });
  window.visualViewport?.addEventListener('resize', placeOverlays);
  window.visualViewport?.addEventListener('scroll', placeOverlays, { passive: true });
  document.addEventListener('fullscreenchange', () => {
    onNativeScreenChange();
    placeOverlays();
  });
  document.addEventListener('webkitfullscreenchange', () => {
    onNativeScreenChange();
    placeOverlays();
  });
  $('enter-grid-fullscreen').onclick = () => {
    enterGridScreen();
    placeOverlays();
  };
  $('exit-grid-fullscreen').onclick = exitGridScreen;
  $('close-inspector').onclick = () => closeInspector(true);
  const dismissOutside = (event) => {
    if (
      inspectedPane &&
      !$('layer-inspector').contains(event.target) &&
      !event.target.closest('.pane-layers-button')
    )
      closeInspector();
  };
  document.addEventListener('pointerdown', dismissOutside);
  document.addEventListener('focusin', dismissOutside);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && inspectedPane) {
      event.preventDefault();
      closeInspector(true);
      return;
    }
    if (gridScreen && event.key === 'Escape') {
      event.preventDefault();
      exitGridScreen();
      return;
    }
    if (gridScreen && event.key === 'Tab') {
      const focusable = [...grid.querySelectorAll('button,select,input,canvas[tabindex]')].filter(
        (el) => !el.disabled && el.getClientRects().length,
      );
      const target = event.shiftKey ? focusable[0] : focusable.at(-1);
      if (document.activeElement === target) {
        event.preventDefault();
        (event.shiftKey ? focusable.at(-1) : focusable[0]).focus({ preventScroll: true });
      }
    }
    if (event.key === 'Escape' && active === 'motion' && $('layout-view').value !== 'all') {
      const previouslyFocused = panes.find((p) => p.id === $('layout-view').value);
      setLayout('all');
      previouslyFocused.expand.focus({ preventScroll: true });
    }
  });
  setLayout('all');
  renderLayers();
}

function updateGhostControl() {
  const available = Boolean(deepPane.view?.setReferenceFrame && referenceWalk && referenceWalk !== currentClip);
  deepPane.ghostToggle.hidden = !available;
  deepPane.caption.hidden = available;
}

function redrawMotion() {
  if (!currentClip) return;
  const phase = phases.motion;
  panes[0].view?.setPhase(phase);
  const frame = scalarAt(currentClip.frames, phase, poseKeys);
  if (currentLoad) {
    const at = scalarAt(currentLoad.frames, phase, ['measuredBw', 'predictedBw', 'tendonBw', 'compressionBw']);
    for (const [id, key] of [
      ['measured', 'measuredBw'],
      ['joint', 'predictedBw'],
      ['tendon', 'tendonBw'],
      ['compression', 'compressionBw'],
    ]) {
      $('live-' + id).textContent = fmt(at[key]) + ' BW';
    }
    if (motionEmbed) {
      const matched = Number.isFinite(at.measuredBw) && Number.isFinite(at.predictedBw);
      $('embed-force').hidden = !matched;
      if (matched) {
        const interval = currentLoad.activity === 'walk' ? 'stance' : 'of the stand';
        $('embed-force-phase').textContent = 'Hip force at ' + Math.round(phase * 100) + '% ' + interval;
        $('embed-implant-value').textContent = fmt(at.measuredBw);
        $('embed-model-value').textContent = fmt(at.predictedBw);
      }
    }
    frame.jointBw = at.predictedBw;
    frame.tendonBw = at.tendonBw;
    frame.compressionBw = at.compressionBw;
  }
  const pose = { ...frame, side: currentClip.side, phase };
  for (const pane of panes) if (pane.layers) pane.view?.setFrame(pose);
  const currentTravel = deepPane.view?.getBursaTravelMm();
  // Before WebGL is ready, both values use the same phase on the archived solver traces.
  const traceAtPhase = clip => {
    const values = travelByClip.get(clip.id);
    return values?.[Math.round(phase * (values.length - 1))];
  };
  const selectedMm = currentTravel ?? traceAtPhase(currentClip);
  let referenceFrame;
  let referenceMm;
  if (referenceWalk === currentClip) referenceMm = selectedMm;
  else if (referenceWalk) {
    referenceFrame = scalarAt(referenceWalk.frames, phase, poseKeys);
    referenceMm = deepPane.view?.getBursaTravelMm(referenceFrame) ?? traceAtPhase(referenceWalk);
  }
  const showGhost = deepPane.ghostToggle.getAttribute('aria-pressed') === 'true';
  deepPane.view?.setReferenceFrame?.(showGhost ? referenceFrame ?? null : null);
  deepPane.travel.setPhase(phase, selectedMm, referenceMm);
  if (travelFrames && $('motion-output').value === 'travel') {
    if (Number.isFinite(currentTravel)) {
      $('travel-value').textContent = fmt(currentTravel, 1) + ' mm at ' + fmt(phase * 100, 1) + '%';
    } else {
      const sample = travelFrames[Math.round(phase * (travelFrames.length - 1))];
      $('travel-value').textContent = fmt(sample.travelMm, 1) + ' mm at nearest solved phase (' + fmt(sample.phase * 100, 1) + '%)';
    }
  }
  motionCursor?.(phase, currentTravel);
}

function selectClip() {
  currentClip = motionData?.clips.find((c) => c.id === $('clip').value);
  if (!currentClip) return;
  currentLoad = results?.trials?.find((t) => t.id === currentClip.trial);
  $('live-loads').hidden = !currentLoad;
  for (const pane of panes)
    if (pane.layers) pane.view?.setLayers({ loads: Boolean(currentLoad) && pane.layers.loads });
  renderLayers();
  // Capacity runs share a speed and a full right stride. Stance-only trials and
  // recordings from a different protocol have no paired reference here.
  referenceWalk = Number.isFinite(currentClip.hipCapacityFraction)
    ? motionData.clips.find(c => c.hipCapacityFraction === 1 && c.source === currentClip.source &&
      c.speedMps === currentClip.speedMps && c.side === currentClip.side && c.phaseLabel === currentClip.phaseLabel)
    : undefined;
  updateGhostControl();
  const values = travelByClip.get(currentClip.id);
  deepPane.travel.setClip(currentClip, values, travelByClip.get(referenceWalk?.id));
  travelFrames = values?.length > 1
    ? values.map((travelMm, i) => ({ phase: i / (values.length - 1), travelMm }))
    : undefined;
  const output = $('motion-output');
  output.querySelector('[value="loads"]').disabled = !currentLoad;
  output.querySelector('[value="travel"]').disabled = !travelFrames;
  if (!(output.value === 'travel' && travelFrames)) output.value = currentLoad ? 'loads' : 'angles';
  panes[0].view?.setMotion(currentClip, motionData.jointNames);
  $('phase-label').textContent = currentClip.phaseLabel || 'Cycle';
  $('clip-detail').textContent = [
    currentClip.phaseLabel || currentClip.label,
    fmt(currentClip.durationS) + ' s',
    'Reference anatomy',
  ].join(' · ');
  $('body-source').textContent = currentClip.kind === 'simulated' ? 'Simulated motion' : 'Recorded motion';
  $('record-title').textContent = currentClip.label;
  $('record-notes').textContent = Array.isArray(currentClip.notes)
    ? currentClip.notes.join(' ')
    : currentClip.notes || currentClip.source;
  if (currentClip.trial) $('record-notes').append(' Trial ' + currentClip.trial + '.');
  $('orthoload-credit').hidden = !currentClip.source?.includes('OrthoLoad');
  $('record-source').hidden = !sourceLink(currentClip.sourceUrl);
  if (sourceLink(currentClip.sourceUrl)) $('record-source').href = currentClip.sourceUrl;
  if (motionEmbed) {
    $('embed-caption').hidden = false;
    $('embed-force').hidden = !currentLoad;
    const source = $('embed-source');
    const label = currentLoad?.split === 'calibration' ? 'Calibration recording from ' :
      currentClip.kind === 'simulated' ? 'Simulated motion from ' : 'Recorded motion from ';
    const instrumented = currentClip.source.includes('OrthoLoad');
    const name = instrumented ? 'OrthoLoad' : currentClip.kind === 'simulated' ? 'Falisse et al.' : 'Bertaux et al.';
    const credit = element('a', '', name);
    credit.href = currentClip.sourceUrl;
    credit.target = '_blank';
    credit.rel = 'noreferrer';
    const note = instrumented ? ' The head and lower arms are reconstructed.' : '';
    source.replaceChildren(label, credit, name.endsWith('.') ? '' : '.', note, ' Hip tissues use reference anatomy.');
    redrawMotion();
  } else {
    renderMotionChart();
  }
}

function renderMotionChart() {
  const travelMode = $('motion-output').value === 'travel' && travelFrames;
  const loadMode = $('motion-output').value === 'loads' && currentLoad;
  const labels = travelMode ? ['Atlas model'] : loadMode ? ['Measured', 'Model'] : ['Flexion', 'Adduction'];
  const keys = travelMode
    ? ['travelMm']
    : loadMode
      ? ['measuredBw', 'predictedBw']
      : ['hipFlexionDeg', 'hipAdductionDeg'];
  const frames = travelMode ? travelFrames : loadMode ? currentLoad.frames : currentClip.frames;
  const title = travelMode
    ? 'IT band travel at the trochanteric bursa'
    : loadMode
      ? 'Resultant hip load'
      : 'Hip angles';
  $('travel-note').hidden = !travelMode;
  $('motion-chart-title').textContent = title;
  $('motion-legend').replaceChildren(...labels.map((label, i) => element('span', i ? 'coral' : 'teal', label)));
  motionCursor = drawChart($('motion-chart'), {
    frames,
    title,
    cursorPointColor: travelMode ? colors.teal : undefined,
    yLabel: travelMode ? 'mm' : loadMode ? 'BW' : 'deg',
    series: labels.map((label, i) => ({ key: keys[i], label, color: i ? colors.coral : colors.teal })),
    scrub: (p) => {
      setPlay('motion', false);
      setPhase('motion', p);
    },
  });
  redrawMotion();
}
$('motion-output').onchange = renderMotionChart;

async function createScene(host, factory) {
  try {
    const view = await factory(host);
    host.querySelector('.waiting')?.remove();
    return view;
  } catch (error) {
    console.error(error);
    host.replaceChildren(
      element('span', 'scene-error', motionEmbed ? 'This view requires WebGL.' : 'The 3D view could not load. The numerical traces remain available below.'),
    );
  }
}
async function setupMotion() {
  if (motionData?.clips?.length) {
    $('clip').replaceChildren(...motionData.clips.map((c) => asOption(c.id, c.label)));
    $('clip').value = motionData.clips.find((c) => c.id === 'h1l-walk')?.id || motionData.clips[0].id;
    $('clip').disabled = false;
    $('clip').onchange = selectClip;
    selectClip();
  } else {
    $('clip').replaceChildren(asOption('', 'No motion file'));
    $('record-notes').textContent = 'The motion dataset is unavailable in this build.';
  }
  await Promise.all(
    panes.map(async (pane) => {
      const body = pane.id === 'body';
      const factory = body ? createBodyView : (host) => createHipScene(host, { profile: pane.id });
      pane.view = await createScene(pane.host, factory);
      if (!pane.view) return;
      if (embedClosed) {
        releasePane(pane);
        return;
      }
      if (pane.camera.value !== 'oblique') pane.view.setView(pane.camera.value);
      if (body) {
        if (currentClip) pane.view.setMotion(currentClip, motionData.jointNames);
        pane.view.setTissueMode?.(bodyMode);
      } else {
        pane.view.setLayers({ ...pane.layers, loads: Boolean(currentLoad) && pane.layers.loads });
      }
      if (pane === deepPane) updateGhostControl();
      redrawMotion();
    }),
  );
}

function setPhase(section, phase) {
  phases[section] = Math.max(0, Math.min(1, phase));
  const prefix = section === 'motion' ? '' : 'pair-';
  const slider = $(prefix + 'phase');
  slider.value = Math.round(phases[section] * 1000);
  slider.style.setProperty('--progress', phases[section] * 100 + '%');
  $(section === 'motion' ? 'phase-value' : 'pair-value').textContent = Math.round(phases[section] * 100) + '%';
  if (section === 'motion') redrawMotion();
  else redrawPair();
}
function setPlay(section, state) {
  if (state && !playing[section]) lastTick = performance.now();
  playing[section] = state;
  const b = $(section === 'motion' ? 'play' : 'pair-play');
  b.setAttribute('aria-pressed', String(state));
  b.setAttribute('aria-label', (state ? 'Pause ' : 'Play ') + (section === 'motion' ? 'motion' : 'comparison'));
  b.querySelector('span').textContent = state ? 'Pause' : 'Play';
}
for (const [section, prefix] of [
  ['motion', ''],
  ['mechanisms', 'pair-'],
]) {
  $(prefix + 'play').onclick = () => setPlay(section, !playing[section]);
  $(prefix + 'phase').oninput = (e) => {
    setPlay(section, false);
    setPhase(section, e.target.valueAsNumber / 1000);
  };
}
function advancePlayback(section, elapsedMs) {
  if (phases[section] === 1) {
    setPhase(section, 0);
    return;
  }
  const duration = section === 'motion'
    ? currentClip?.durationS
    : counterfactual?.tasks?.find((task) => task.id === activePairTask)?.durationS;
  const speed = section === 'motion' ? Number($('speed').value) : 1;
  const nextPhase = phases[section] + (Math.min(elapsedMs, 80) * speed) / (1000 * (duration || 1));
  setPhase(section, nextPhase);
}

function tick(time) {
  if (embedClosed) return;
  const elapsedMs = Math.max(0, time - lastTick);
  lastTick = time;
  if (playing[active]) advancePlayback(active, elapsedMs);
  requestAnimationFrame(tick);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    setPlay('motion', false);
    setPlay('mechanisms', false);
  }
});
let pairSetup;
let settledPairViews = [];
function preparePair() {
  if (pairSetup || !counterfactual?.pair?.cases) return;
  const embed = $('pair-stages');
  const cases = counterfactual.pair.cases;
  const hosts = cases.map((c, i) => {
    const scene = element('section', 'scene');
    const header = element('header', 'scene-head');
    const name = element('div');
    name.append(element('span', i === 0 ? 'signal' : 'signal coral'), element('h3', '', c.label));
    header.append(name);
    const host = element('div', 'canvas-host');
    host.append(element('span', 'waiting', 'Loading anatomy…'));
    const foot = element('div', 'stage-foot');
    const value = element('span', '', '');
    value.id = 'pair-force-' + i;
    foot.append(value);
    scene.append(header, host, foot);
    embed.append(scene);
    return host;
  });
  pairSetup = Promise.all(hosts.map((host) => createScene(host, createHipScene))).then((views) => {
    settledPairViews = views;
    views.forEach((view) => view?.setLayers({ labels: false, loads: false }));
    redrawPair();
  });
}

function renderPairChart() {
  const cases = counterfactual.pair.cases,
    key = $('pair-channel').value;
  const rows = cases[0][activePairTask].frames.map((f, i) => ({
    phase: f.phase,
    a: f[key],
    b: cases[1][activePairTask].frames[i]?.[key],
  }));
  const title = $('pair-channel').selectedOptions[0].textContent;
  pairCursor = drawChart($('pair-chart'), {
    frames: rows,
    title: `${title} for two virtual impairments`,
    yLabel: key.endsWith('Bw') ? 'BW' : 'deg',
    series: [
      { key: 'a', label: cases[0].label, color: colors.teal },
      { key: 'b', label: cases[1].label, color: colors.coral },
    ],
    scrub: (p) => {
      setPlay('mechanisms', false);
      setPhase('mechanisms', p);
    },
  });
  $('pair-legend').replaceChildren(
    element('span', 'teal', cases[0].label),
    element('span', 'coral', cases[1].label),
  );
  const pair = counterfactual.pair;
  const difference = activePairTask === 'walk' ? pair.walkRmsDeg : pair.challengeRmsDeg;
  $('pair-caption').textContent =
    `These two trajectories differ by ${fmt(difference)}° RMS across hip flexion, adduction and trunk lean. ${pair.description}`;
  redrawPair();
}

function redrawPair() {
  const cases = counterfactual?.pair?.cases;
  if (!cases) return;
  const phase = phases.mechanisms;
  cases.forEach((c, i) => {
    const frame = scalarAt(c[activePairTask].frames, phase, poseKeys);
    settledPairViews[i]?.setFrame({ ...frame, side: 'right', phase });
    const output = $('pair-force-' + i);
    if (output) output.textContent = `Abductor force  ${fmt(frame.abductorForceBw)} BW`;
  });
  pairCursor?.(phase);
}

function setupMechanism() {
  if (!counterfactual?.pair?.cases?.length)
    return sectionError('mechanism-status', 'No simulated cohort is available in this build.');
  $('mechanism-status').hidden = true;
  $('mechanism-content').hidden = false;
  $('task').replaceChildren(...counterfactual.tasks.map((t) => asOption(t.id, t.label)));
  $('task').onchange = () => {
    activePairTask = $('task').value;
    renderPairChart();
  };
  $('pair-channel').onchange = renderPairChart;
  const m = counterfactual.metrics;
  const stats = [
    ['Held-out anatomies', fileNumber(m.nTestAnatomies)],
    ['Walking · balanced accuracy', percent(m.walkBalancedAccuracy)],
    ['Walking + challenge · balanced accuracy', percent(m.combinedBalancedAccuracy)],
    ['Ambiguous pairs after both tasks', percent(m.combinedAmbiguousFraction)],
  ];
  for (const [label, value] of stats) {
    const row = element('div');
    row.append(element('span', '', label), element('strong', '', value));
    $('mechanism-metrics').append(row);
  }
  $('assumptions').replaceChildren(
    ...[
      ...counterfactual.assumptions,
      `The simulated camera errors were ${m.noiseRmsDeg.hipFlexionDeg}° in hip flexion, ${m.noiseRmsDeg.hipAdductionDeg}° in adduction and ${m.noiseRmsDeg.trunkLeanDeg}° in trunk lean. A pair counts as ambiguous when each angle’s waveform difference falls within its respective error budget.`,
      `Walking and loaded stance improved balanced accuracy by ${fmt((m.combinedBalancedAccuracy - m.walkBalancedAccuracy) * 100, 1)} percentage points over walking alone (paired 95% bootstrap interval ${m.combinedMinusWalkCi95.map((x) => fmt(x * 100, 1)).join('–')} points). The repeat-walk comparison scored ${percent(m.repeatWalkBalancedAccuracy)}.`,
      'Balanced accuracy measures recovery of the two modeled impairments in a held-out synthetic cohort. Clinical accuracy for tendinopathy and osteoarthritis requires patient labels.',
    ].map((note) => element('li', '', note)),
  );
  const gap = counterfactual.sensitivity?.heldOutClinicalComparison;
  if (gap) {
    $('clinical-gap').hidden = false;
    $('clinical-gap-note').textContent =
      `The modeled impairments differ by ${fmt(gap.simulatedCapacityMinusResistanceDeg, 1)}° of hip flexion–extension excursion during stance. A separate study measured a ${fmt(gap.clinicalGtMinusOaDeg, 1)}° group difference between gluteal tendinopathy and hip osteoarthritis (95% CI ${gap.clinicalCi95Deg.join('–')}°). The virtual task captures only part of that difference. Individual diagnosis from video remains untested.`;
    $('clinical-gap-source').href = gap.source;
  }
  renderPairChart();
}

function renderTrial() {
  const t = results.trials.find((trial) => trial.id === $('trial').value);
  if (!t) return;
  const role = t.split === 'calibration' ? 'Calibration participant' : 'Held-out participant';
  $('trial-kind').textContent = `${t.subject} · ${role}`;
  $('trial-note').textContent =
    `The model’s root mean square error is ${fmt(t.rmseBw, 3)} body weights and the baseline’s is ${fmt(t.baselineRmseBw, 3)}. The shaded region spans the central 90% of retained model predictions.`;
  drawChart($('force-chart'), {
    frames: t.frames,
    yLabel: 'BW',
    title: `Measured and modeled hip force in ${t.id}`,
    series: [
      { key: 'baselineBw', label: 'Baseline', color: colors.gray, dashed: true },
      { key: 'measuredBw', label: 'Implant', color: colors.teal },
      { key: 'predictedBw', label: 'Estimate', color: colors.coral },
    ],
    band: ['loBw', 'hiBw'],
  });
}

function setupValidation() {
  if (!results?.trials?.length)
    return sectionError('validation-status', 'To load the implant comparison, prepare the local data using hip-study/compute/README.md.');
  $('validation-status').hidden = true;
  $('validation-content').hidden = false;
  const m = results.summary;
  const nCalibration = Array.isArray(m.calibrationSubjects) ? m.calibrationSubjects.length : m.calibrationSubjects;
  const nEvaluation = Array.isArray(m.evaluationSubjects) ? m.evaluationSubjects.length : m.evaluationSubjects;
  $('validation-metrics').replaceChildren(
    metric('Held-out model RMSE', `${fmt(m.modelRmseBw, 3)} BW`, `${nEvaluation} evaluation participants`),
    metric('Baseline RMSE', `${fmt(m.baselineRmseBw, 3)} BW`, `${nCalibration} calibration participants`),
    metric('90% band coverage', percent(m.coverage90), 'Held-out measured samples inside the band'),
  );
  const trials = results.trials.toSorted(
    (a, b) => Number(b.split === 'evaluation') - Number(a.split === 'evaluation'),
  );
  $('trial').replaceChildren(
    ...trials.map((t) =>
      asOption(t.id, `${t.subject} · ${t.activity === 'walk' ? 'Walking' : 'Single-leg stance'}`),
    ),
  );
  $('trial').disabled = false;
  $('trial').onchange = renderTrial;
  renderTrial();
  const amb = results.ambiguity;
  if (amb?.low?.frames && amb?.high?.frames) {
    const frames = amb.low.frames.map((f, i) => ({
      phase: f.phase,
      a: f.compressionBw,
      b: amb.high.frames[i]?.compressionBw,
    }));
    const at = amb.frameIndex;
    $('ambiguity-caption').textContent =
      `Both candidates use the motion and ground forces from ${amb.trialId}. At ${Math.round(amb.phase * 100)}% stance they predict ${fmt(frames[at].a, 3)} and ${fmt(frames[at].b, 3)} BW of lateral compression. Across stance their resultant joint loads differ by just ${fmt(amb.contactRmsDifferenceBw, 3)} BW RMS.`;
    const cursor = drawChart($('ambiguity-chart'), {
      frames,
      yLabel: 'BW',
      title: 'Lateral compression estimates for two plausible anatomies',
      series: [
        { key: 'a', label: 'Candidate A', color: colors.teal },
        { key: 'b', label: 'Candidate B', color: colors.coral },
      ],
    });
    cursor(amb.phase);
  } else $('ambiguity-caption').textContent = 'The retained-model comparison is unavailable for these trials.';
}

function renderOffsetStudy(sweep) {
  const metal = data('benchmark-metal-sensitivity');
  const cpuPareto = data('benchmark-pareto')?.cpu;
  const timing = sweep.timing;
  const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
  const warmCloud = median(timing.warmCloudSamplesSeconds);
  $('offset-study').hidden = false;
  const metrics = [
    metric(
      'M4 CPU / warm H100 round trip',
      fmt(timing.warmRpcSpeedup, 2) + '×',
      'Exhaustive search on both backends',
    ),
    metric(
      'Candidate comparisons',
      fmt(sweep.candidateResampleOffsetCombinations / 1e9, 2) + 'B',
      'Exhaustive search across ' + sweep.cases.length + ' offsets',
    ),
  ];
  if (metal?.pareto) {
    metrics.unshift(
      metric(
        'Fastest measured search',
        fmt(metal.pareto.warmEndToEndSeconds, 4) + ' s',
        'M4 GPU with exact pruning',
      ),
    );
  }
  $('bench-metrics').replaceChildren(...metrics);
  const times = [
    ['M4 CPU · exhaustive', median(timing.cpuSamplesSeconds), timing.firstLocalSeconds],
    ['M4 CPU · pruned', cpuPareto?.medianSeconds, cpuPareto?.firstLocalSeconds],
    ['M4 GPU · exhaustive', metal?.exhaustive?.warmEndToEndSeconds, undefined],
    ['M4 GPU · pruned', metal?.pareto?.warmEndToEndSeconds, metal?.firstLocalSeconds],
    ['H100 cloud · exhaustive', warmCloud, timing.firstCloudAppSeconds],
  ];
  $('offset-timing').replaceChildren(
    ...times
      .filter(([, warm]) => Number.isFinite(warm))
      .map(([label, warm, first]) => {
        const row = element('tr');
        row.append(
          element('td', '', label),
          element('td', '', fmt(warm, 4) + ' s'),
          element('td', '', Number.isFinite(first) ? fmt(first, 4) + ' s' : 'N/A'),
        );
        return row;
      }),
  );
  const comparisons = fileNumber(sweep.winnerAgreement);
  const overview = `Each timing starts with preprocessed recordings and ends when the ${comparisons} winning indices reach the laptop. Cloud times include transport. The first H100 job uses a prebuilt image.`;
  $('offset-method').textContent = overview;
  if (metal?.pareto) {
    const factor = warmCloud / metal.pareto.warmEndToEndSeconds;
    const fitCount = fileNumber(metal.pareto.weightedFitsEvaluated);
    const local = cpuPareto?.medianSeconds
      ? ` With the same pruning rule, the M4 GPU finished ${fmt(cpuPareto.medianSeconds / metal.pareto.warmEndToEndSeconds, 2)}× faster than the M4 CPU.`
      : '';
    $('offset-compute').textContent =
      `The pruned M4 GPU finished ${fmt(factor, 2)}× faster than the exhaustive H100 round trip.${local} The exhaustive GPU timing ranges overlapped. The cloud comparison covers exhaustive search only.`;
    $('offset-method').textContent +=
      ` Pruning skips a model only when another fits all six calibration participants better. The pruned GPU evaluated ${fitCount} weighted fits and matched every exhaustive winner.`;
  } else {
    $('offset-compute').textContent = overview;
  }
  const reports = [
    ['bootstrap-sensitivity', 'H100 & CPU'],
    ['benchmark-metal-sensitivity', 'M4 GPU'],
    ['benchmark-pareto', 'Pruned CPU'],
  ];
  for (const [name, label] of reports) {
    const report = data(name);
    if (!report) continue;
    const link = element('a', 'text-link', label + ' JSON ↓');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    link.download = name + '.json';
    $('benchmark-files').append(link);
  }
  const offsets = sweep.cases.map((c) => c.offsetMm);
  const lo = Math.min(...offsets),
    hi = Math.max(...offsets);
  const frames = sweep.cases.map((c) => ({
    phase: (c.offsetMm - lo) / (hi - lo),
    measured: sweep.measuredBw,
    predicted: c.jointForceBwQuantiles[1],
    low: c.jointForceBwQuantiles[0],
    high: c.jointForceBwQuantiles[2],
  }));
  const slider = $('offset-index');
  slider.max = String(sweep.cases.length - 1);
  const cursor = drawChart($('offset-chart'), {
    frames,
    title: 'Held-out joint force by frontal joint-center offset',
    yLabel: 'BW',
    axisLabel: 'Frontal offset (mm)',
    xTicks: offsets
      .filter((x) => Number.isInteger(x) && x % 5 === 0)
      .map((x) => [(x - lo) / (hi - lo), String(x)]),
    series: [
      { key: 'measured', label: 'Implant', color: colors.teal },
      { key: 'predicted', label: 'Median model', color: colors.coral },
    ],
    band: ['low', 'high'],
    scrub: (p) => showOffset(Math.round(p * (sweep.cases.length - 1))),
  });
  function showOffset(index) {
    const c = sweep.cases[index];
    const values = c.jointForceBwQuantiles;
    slider.value = String(index);
    const label = c.offsetMm + ' mm';
    slider.setAttribute('aria-valuetext', label);
    $('offset-value').textContent = label;
    cursor(frames[index].phase);
    $('offset-result').textContent =
      `At ${Math.round(sweep.heldOutPhase * 100)}% stance in ${sweep.heldOutTrialId}, the implant measured ${fmt(sweep.measuredBw, 3)} BW. At ${label} offset the model median is ${fmt(values[1], 3)} BW (5th–95th percentiles ${fmt(values[0], 3)}–${fmt(values[2], 3)} BW). All ${sweep.cases.length} model intervals remain above the measurement.`;
  }
  slider.oninput = () => showOffset(slider.valueAsNumber);
  showOffset(Math.max(0, offsets.indexOf(0)));
}

function setupCompute() {
  const bench = data('benchmark'),
    metal = data('benchmark-metal'),
    cpu = data('benchmark-cpu');
  if (!bench?.rows?.length && !metal?.rows?.length)
    return sectionError('bench-status', 'No benchmark measurements are available in this build.');
  $('bench-status').hidden = true;
  $('compute-content').hidden = false;
  const rows = bench?.rows || [];
  const last = rows.at(-1) || cpu?.rows?.at(-1);
  const gpuName = bench?.gpuName || 'H100';

  $('bench-metrics').replaceChildren(
    metric('CPU / H100 kernel', last?.speedup ? fmt(last.speedup, 1) + '×' : 'N/A', 'Largest matched workload'),
    metric('Largest search', fileNumber(last?.candidates), gpuName),
    metric(
      'CPU reference',
      bench?.cpuName || cpu?.cpuName || 'Laptop',
      cpu?.cpuThreads ? `${cpu.cpuThreads} worker threads` : '',
    ),
  );
  const sweep = data('bootstrap-sensitivity');
  if (sweep?.timing && sweep.cases?.length) renderOffsetStudy(sweep);
  const metalTimes = new Map((metal?.rows || []).map((r) => [r.candidates, r.deviceMs]));
  const cpuRows = new Map((cpu?.rows || []).map((r) => [r.candidates, r]));
  const sizes = [...new Set([...rows.map((r) => r.candidates), ...metalTimes.keys()])].sort((a, b) => a - b);
  $('bench-rows').replaceChildren(
    ...sizes.map((size) => {
      const r = rows.find((row) => row.candidates === size) || {};
      const tr = element('tr');
      const local = cpuRows.get(size),
        m4 = metalTimes.get(size);
      for (const v of [
        fileNumber(size),
        fileNumber(r.frames ?? local?.frames),
        fmt(r.cpuMs ?? local?.cpuMs),
        fmt(m4),
        fmt(r.gpuMs),
        Number.isFinite(r.speedup) ? fmt(r.speedup, 2) + '×' : 'N/A',
        r.gpuMs && m4 ? fmt(m4 / r.gpuMs, 2) + '×' : 'N/A',
      ])
        tr.append(element('td', '', v));
      return tr;
    }),
  );
  const m = metal?.rows?.at(-1);
  $('bench-detail').textContent =
    'Median time in milliseconds. Each candidate is scored against the same calibration frames. GPU columns show device execution time.' +
    (m
      ? ' The largest Metal run recovered ' +
        m.topKAgreement +
        ' of the CPU reference’s top ' +
        m.topK +
        ' candidates.'
      : '') +
    (last?.topK
      ? ' H100 recovered ' +
        last.topKAgreement +
        ' of ' +
        last.topK +
        ', with maximum absolute score difference ' +
        fmt(last.maxAbsoluteError, 6) +
        ' BW².'
      : '');
  $('rpc-note').textContent = bench?.compact
    ? 'At ' +
      fileNumber(bench.compact.candidates) +
      ' candidates, one seed-based cloud request took ' +
      fmt(bench.compact.rpcSeconds, 2) +
      ' s including sampling, transport and computation. Sending the precomputed arrays took ' +
      fmt(bench.singlePassRpcSeconds, 2) +
      ' s end to end. The resident GPU kernel took ' +
      fmt(last.gpuMs, 2) +
      ' ms.'
    : 'Cloud timing is unavailable for this run.';
  const search = results?.convergence;
  if (search) {
    $('yield-note').textContent =
      'In trial ' +
      search.trialId +
      ', each compatible model fits the calibration recordings and stays within ' +
      fmt(search.contactRmsLimitBw, 2) +
      ' BW RMS of the reference model’s joint-load trace. The range gives the estimated lateral compression at the selected frame.';
    $('yield-rows').replaceChildren(
      ...search.rows.map((row) => {
        const tr = element('tr');
        for (const v of [
          fileNumber(row.candidates),
          fileNumber(row.matches),
          row.compressionRangeBw.map((n) => fmt(n, 3)).join(' – '),
        ])
          tr.append(element('td', '', v));
        return tr;
      }),
    );
  }
  const bootstrap = data('bootstrap');
  if (bootstrap?.timing) {
    $('bootstrap-card').hidden = false;
    $('bootstrap-note').textContent =
      `Resampling the ${bootstrap.subjects.length} calibration participants gave ${fileNumber(bootstrap.uniqueResamples)} distinct combinations and selected ${bootstrap.distinctSelectedCandidates} different models. The most frequent model won ${percent(bootstrap.modeProbability)} of bootstrap draws. The H100 evaluated all combinations in ${fmt(bootstrap.timing.gpuMs, 1)} ms and the laptop CPU took ${fmt(bootstrap.timing.cpuMs, 1)} ms. Both selected the same model in all ${bootstrap.timing.winnerAgreement} combinations.`;
  }
}
function activateTab() {
  const wanted = motionEmbed ? 'motion' : location.hash.slice(1);
  const nav = document.querySelectorAll('.tabs a');
  active = [...nav].some((link) => link.hash === '#' + wanted) ? wanted : 'motion';
  if (active !== 'motion') {
    closeInspector();
    if (gridScreen) exitGridScreen();
  }
  for (const section of document.querySelectorAll('main > .section')) section.hidden = section.id !== active;
  for (const link of nav) {
    if (link.hash === '#' + active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  for (const p of Object.keys(playing)) if (p !== active) setPlay(p, false);
  if (active === 'mechanisms') {
    preparePair();
    redrawPair();
  }
}
for (const link of document.querySelectorAll('.tabs a')) {
  link.addEventListener('click', (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (location.hash !== link.hash) history.pushState(null, '', link.hash);
    activateTab();
  });
}
window.addEventListener('hashchange', activateTab);
window.addEventListener('popstate', activateTab);
createMotionPanes();
if (motionEmbed) {
  window.addEventListener('pagehide', event => {
    if (event.persisted) return;
    embedClosed = true;
    for (const pane of panes) releasePane(pane);
  });
} else {
  import('./evidence.js')
    .then(({ mountEvidence }) => mountEvidence($('evidence'), drawChart))
    .catch(() => {
      $('evidence').textContent = 'The gait evidence could not load. Reload the study to try again.';
    });
  setupMechanism();
  setupValidation();
  setupCompute();
}
activateTab();
setupMotion();
requestAnimationFrame(tick);

function releasePane(pane) {
  if (!pane.view) return;
  const canvas = pane.host.querySelector('canvas');
  const gl = canvas?.getContext('webgl2');
  pane.view.dispose();
  gl?.getExtension('WEBGL_lose_context')?.loseContext();
  pane.view = null;
}
