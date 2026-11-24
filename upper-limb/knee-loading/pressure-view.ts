import {
  BoxGeometry, Color, Float32BufferAttribute, GridHelper, InstancedMesh, Matrix4, MeshBasicMaterial, OrthographicCamera,
  Raycaster, Scene, Vector2, Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import './pressure-view.css';

interface PressureCondition {
  flexionDeg: number;
  quadricepsLoadN: number;
  pressureMpa: number[];
  peakMpa: number;
  sampleCount: number;
  samplePeriodSeconds: number;
}

interface PressureStudy {
  specimen: string;
  source: { title: string; url: string; license: string };
  grid: { rows: number; columns: number; rowSpacingMm: number; columnSpacingMm: number; orientation: string };
  pressureRangeMpa: [number, number];
  conditions: PressureCondition[];
}

const dataUrl = new URL('../../knee-loading/data/patellar-pressure.json', import.meta.url).href;
const background = 0x122726;
const neutral = '#25433e';
const colors = [neutral, '#305573', '#398f8f', '#9dc685', '#f0cb72', '#f57955'].map((hex) => new Color(hex));
const format = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
const maxReliefMm = 17;

function toColor(value: number, maximum: number, target = new Color()): Color {
  const index = Math.max(0, Math.min(value / maximum, 1)) * (colors.length - 1);
  const left = Math.floor(index);
  return target.lerpColors(colors[left], colors[Math.min(left + 1, colors.length - 1)], index - left);
}

function find<T extends Element = HTMLElement>(root: HTMLElement, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing pressure view element ${selector}`);
  return element;
}

interface Relief {
  setData(pressure: number[]): void;
  reset(): void;
  dispose(): void;
}

function createRelief(
  canvas: HTMLCanvasElement, stage: HTMLElement, grid: PressureStudy['grid'], maximum: number,
  probe: (cell: number | null) => void,
): Relief {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setClearColor(background);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new Scene();
  const { rows, columns, rowSpacingMm, columnSpacingMm } = grid;
  const widthMm = columns * columnSpacingMm;
  const depthMm = rows * rowSpacingMm;
  const geometry = new BoxGeometry(1, 1, 1);
  const normals = geometry.getAttribute('normal');
  const sides = new Float32Array(normals.count * 3);
  for (let index = 0; index < normals.count; index++) {
    // Keep the top face true to the legend. Shade the four side faces so the
    // cell heights remain legible when adjacent cells have similar pressure.
    const shade = normals.getY(index) > 0 ? 1 : normals.getX(index) > 0 ? 0.65 : 0.82;
    sides.fill(shade, index * 3, index * 3 + 3);
  }
  geometry.setAttribute('color', new Float32BufferAttribute(sides, 3));
  const material = new MeshBasicMaterial({ color: 0xffffff, vertexColors: true });
  const bars = new InstancedMesh(geometry, material, rows * columns);
  bars.frustumCulled = false;
  bars.count = 0;
  scene.add(bars);

  const gridLines = new GridHelper(Math.max(widthMm, depthMm), Math.max(rows, columns), 0x415952, 0x25433e);
  scene.add(gridLines);
  const camera = new OrthographicCamera(-42, 42, 37, -37, 1, 500);
  const controls = new OrbitControls(camera, canvas);
  controls.enablePan = false;
  controls.minZoom = 0.72;
  controls.maxZoom = 2.8;
  controls.minPolarAngle = 0.1;
  controls.maxPolarAngle = Math.PI / 2 - 0.08;

  function draw() {
    if (stage.clientWidth && stage.clientHeight) renderer.render(scene, camera);
  }

  function resize() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (!width || !height) return;
    const aspect = width / height;
    const halfHeight = Math.max(35, 37 / aspect);
    camera.left = -halfHeight * aspect;
    camera.right = halfHeight * aspect;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    draw();
  }

  function reset() {
    camera.position.set(-64, 68, 78);
    camera.zoom = 1;
    controls.target.set(0, 2, 0);
    camera.updateProjectionMatrix();
    controls.update();
    draw();
  }

  controls.addEventListener('change', draw);
  const observer = new ResizeObserver(resize);
  observer.observe(stage);
  reset();
  resize();

  const cells: number[] = [];
  const matrix = new Matrix4();
  const color = new Color();
  function setData(pressure: number[]) {
    cells.length = 0;
    for (const [cell, value] of pressure.entries()) {
      if (!(value > 0)) continue;
      const row = Math.floor(cell / columns);
      const column = cell % columns;
      const x = (column + 0.5) * columnSpacingMm - widthMm / 2;
      const z = (row + 0.5) * rowSpacingMm - depthMm / 2;
      const height = value / maximum * maxReliefMm;
      const instance = cells.length;
      matrix.makeScale(columnSpacingMm * 0.97, height, rowSpacingMm * 0.97);
      matrix.setPosition(x, height / 2, z);
      bars.setMatrixAt(instance, matrix);
      bars.setColorAt(instance, toColor(value, maximum, color));
      cells.push(cell);
    }
    bars.count = cells.length;
    bars.instanceMatrix.needsUpdate = true;
    if (bars.instanceColor) bars.instanceColor.needsUpdate = true;
    bars.computeBoundingSphere();
    draw();
  }

  const mouse = new Vector2();
  const ray = new Raycaster();
  const ground = new Vector3();
  function inspect(event: PointerEvent) {
    if (event.type === 'pointermove' && event.buttons) return;
    const rect = canvas.getBoundingClientRect();
    mouse.set((event.clientX - rect.left) * 2 / rect.width - 1, 1 - (event.clientY - rect.top) * 2 / rect.height);
    ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObject(bars, false)[0]?.instanceId;
    if (hit !== undefined) {
      probe(cells[hit]);
      return;
    }
    const distance = -ray.ray.origin.y / ray.ray.direction.y;
    if (!Number.isFinite(distance) || distance <= 0) {
      probe(null);
      return;
    }
    ray.ray.at(distance, ground);
    const row = Math.floor((ground.z + depthMm / 2) / rowSpacingMm);
    const column = Math.floor((ground.x + widthMm / 2) / columnSpacingMm);
    probe(row >= 0 && row < rows && column >= 0 && column < columns ? row * columns + column : null);
  }
  canvas.addEventListener('pointermove', inspect);
  canvas.addEventListener('pointerdown', inspect);
  canvas.addEventListener('pointerleave', () => probe(null));
  canvas.addEventListener('keydown', (event) => {
    const step = Math.PI / 18;
    const rotation: Record<string, () => void> = {
      ArrowLeft: () => controls.rotateLeft(step),
      ArrowRight: () => controls.rotateLeft(-step),
      ArrowUp: () => controls.rotateUp(step),
      ArrowDown: () => controls.rotateUp(-step),
    };
    if (rotation[event.key]) {
      event.preventDefault();
      rotation[event.key]();
      return;
    }
    if (event.key !== '+' && event.key !== '-' && event.key !== '=') return;
    event.preventDefault();
    camera.zoom = Math.max(controls.minZoom, Math.min(controls.maxZoom, camera.zoom * (event.key === '-' ? 1 / 1.15 : 1.15)));
    camera.updateProjectionMatrix();
    draw();
  });

  return {
    setData,
    reset,
    dispose() {
      observer.disconnect();
      controls.dispose();
      bars.dispose();
      gridLines.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    },
  };
}

function drawGrid(canvas: HTMLCanvasElement, grid: PressureStudy['grid'], values: number[], maximum: number): void {
  const context = canvas.getContext('2d');
  if (!context) return;
  const { rows, columns } = grid;
  const pixelScale = 8;
  canvas.width = columns * pixelScale;
  canvas.height = rows * pixelScale;
  const color = new Color();
  for (const [index, value] of values.entries()) {
    context.fillStyle = value > 0 ? toColor(value, maximum, color).getStyle() : neutral;
    context.fillRect(index % columns * pixelScale, Math.floor(index / columns) * pixelScale, pixelScale - 1, pixelScale - 1);
  }
}

function validDataset(study: PressureStudy): boolean {
  const { rows, columns, rowSpacingMm, columnSpacingMm } = study.grid;
  const maximum = study.pressureRangeMpa[1];
  return Number.isInteger(rows) && rows > 0
    && Number.isInteger(columns) && columns > 0
    && Number.isFinite(rowSpacingMm) && rowSpacingMm > 0
    && Number.isFinite(columnSpacingMm) && columnSpacingMm > 0
    && Number.isFinite(maximum) && maximum > 0
    && study.conditions.length > 0
    && study.conditions.every((condition) => condition.pressureMpa.length === rows * columns
      && Number.isFinite(condition.flexionDeg)
      && Number.isFinite(condition.quadricepsLoadN)
      && Number.isFinite(condition.peakMpa) && condition.peakMpa <= maximum
      && condition.pressureMpa.every((value) => Number.isFinite(value) && value >= 0 && value <= maximum));
}

export async function mountPressureView(container: HTMLElement): Promise<() => void> {
  container.classList.add('card', 'pressure-card');
  container.innerHTML = `
    <div class="pressure-head">
      <div><h3>Measured kneecap pressure</h3><p class="muted" data-pressure-specimen></p></div>
      <div class="pressure-switch" role="group" aria-label="Pressure view">
        <button type="button" data-pressure-relief aria-pressed="true">3D relief</button>
        <button type="button" data-pressure-grid aria-pressed="false">Sensor grid</button>
      </div>
    </div>
    <div class="pressure-layout">
      <div class="pressure-stage">
        <canvas data-pressure-scene role="img" aria-label="3D relief of measured kneecap contact pressure"></canvas>
        <canvas data-pressure-map role="img" aria-label="Kneecap contact pressure map in sensor coordinates" hidden></canvas>
        <div class="pressure-stage-caption"><span>Pressure relief</span><button type="button" data-pressure-reset>Reset view</button></div>
        <div class="pressure-probe" data-pressure-probe>Explore a sensor cell</div>
      </div>
      <div class="pressure-controls">
        <div class="pressure-filters">
          <label>Knee flexion<select data-pressure-angle></select></label>
          <label>Quadriceps load<select data-pressure-load></select></label>
        </div>
        <dl><dt>Peak pressure in median map</dt><dd data-pressure-peak></dd></dl>
        <div class="pressure-legend">
          <span>0</span><i aria-hidden="true"></i><span data-pressure-max></span>
        </div>
        <p class="muted" data-pressure-details></p>
        <p class="muted" data-pressure-scale></p>
        <a class="subtle-link" data-pressure-source>Open Knee(s) measurement archive</a>
      </div>
    </div>`;

  try {
    const response = await fetch(dataUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const study: PressureStudy = await response.json();
    if (!validDataset(study)) throw new Error('Incorrect pressure grid data');
    const { grid } = study;
    const maximum = study.pressureRangeMpa[1];
    const angles = find<HTMLSelectElement>(container, '[data-pressure-angle]');
    const loads = find<HTMLSelectElement>(container, '[data-pressure-load]');
    const reset = find<HTMLButtonElement>(container, '[data-pressure-reset]');
    const switchRelief = find<HTMLButtonElement>(container, '[data-pressure-relief]');
    const switchGrid = find<HTMLButtonElement>(container, '[data-pressure-grid]');
    const stage = find(container, '.pressure-stage');
    const scene = find<HTMLCanvasElement>(container, '[data-pressure-scene]');
    const map = find<HTMLCanvasElement>(container, '[data-pressure-map]');
    const caption = find(container, '.pressure-stage-caption span');
    const specimen = find(container, '[data-pressure-specimen]');
    const details = find(container, '[data-pressure-details]');
    const peak = find(container, '[data-pressure-peak]');
    const probeLabel = find(container, '[data-pressure-probe]');
    const link = find<HTMLAnchorElement>(container, '[data-pressure-source]');
    specimen.textContent = `Open Knee(s) · cadaver specimen ${study.specimen} · controlled load`;
    link.href = study.source.url;
    link.textContent = `${study.source.title} · ${study.source.license}`;
    find(container, '[data-pressure-max]').textContent = `${format.format(maximum)} MPa`;
    find(container, '.pressure-legend i').style.background = `linear-gradient(to right, ${colors.map((color) => color.getStyle()).join(', ')})`;
    const rowPitch = format.format(grid.rowSpacingMm);
    const columnPitch = format.format(grid.columnSpacingMm);
    const spacing = rowPitch === columnPitch ? `${rowPitch} mm` : `${rowPitch} × ${columnPitch} mm`;
    const scale = find(container, '[data-pressure-scale]');
    scale.textContent = `Height and color share an MPa scale across conditions. Tekscan cell spacing is ${spacing}. Column numbers increase medially and row numbers increase inferiorly.`;
    scale.title = grid.orientation;

    const allAngles = Array.from(new Set(study.conditions.map(({ flexionDeg }) => flexionDeg))).sort((a, b) => a - b);
    angles.replaceChildren(...allAngles.map((angle) => new Option(`${angle}°`, String(angle))));
    angles.value = String(allAngles.includes(45) ? 45 : allAngles[0]);
    let current: PressureCondition | undefined;
    let showingGrid = false;
    function probe(cell: number | null) {
      const value = cell === null ? undefined : current?.pressureMpa[cell];
      probeLabel.textContent = value !== undefined && cell !== null
        ? `Row ${Math.floor(cell / grid.columns) + 1} · Column ${cell % grid.columns + 1} · ${format.format(value)} MPa`
        : 'Select a sensor cell';
    }
    let relief: Relief | undefined;
    try {
      relief = createRelief(scene, stage, grid, maximum, probe);
      scene.tabIndex = 0;
      scene.title = 'Drag or use arrows to orbit. Scroll or use +/− to zoom.';
    } catch {
      switchRelief.disabled = true;
    }

    function draw() {
      if (!current) return;
      if (showingGrid) drawGrid(map, grid, current.pressureMpa, maximum);
      else relief?.setData(current.pressureMpa);
      peak.textContent = `${format.format(current.peakMpa)} MPa`;
      details.textContent = `Median of ${current.sampleCount} pressure frames at this held load. ${grid.rows} × ${grid.columns} calibrated sensor cells.`;
      probe(null);
    }

    function setMode(gridMode: boolean) {
      showingGrid = gridMode || !relief;
      scene.hidden = showingGrid;
      map.hidden = !showingGrid;
      reset.hidden = showingGrid;
      caption.textContent = showingGrid ? 'Sensor rows and columns' : 'Drag to orbit · Scroll to zoom';
      switchGrid.setAttribute('aria-pressed', String(showingGrid));
      switchRelief.setAttribute('aria-pressed', String(!showingGrid));
      draw();
    }

    function changeCondition() {
      current = study.conditions.find(({ flexionDeg, quadricepsLoadN }) => flexionDeg === Number(angles.value)
        && quadricepsLoadN === Number(loads.value));
      draw();
    }

    function updateLoads() {
      const previous = loads.value === '' ? 600 : Number(loads.value);
      const available = study.conditions.filter(({ flexionDeg }) => flexionDeg === Number(angles.value))
        .map(({ quadricepsLoadN }) => quadricepsLoadN).sort((a, b) => a - b);
      loads.replaceChildren(...available.map((load) => new Option(`${load} N`, String(load))));
      loads.value = String(available.includes(previous) ? previous : available.at(-1));
      changeCondition();
    }

    angles.addEventListener('change', updateLoads);
    loads.addEventListener('change', changeCondition);
    switchRelief.addEventListener('click', () => setMode(false));
    switchGrid.addEventListener('click', () => setMode(true));
    reset.addEventListener('click', () => relief?.reset());
    function inspectCell(event: PointerEvent) {
      const rect = map.getBoundingClientRect();
      const column = Math.min(grid.columns - 1, Math.floor((event.clientX - rect.left) * grid.columns / rect.width));
      const row = Math.min(grid.rows - 1, Math.floor((event.clientY - rect.top) * grid.rows / rect.height));
      probe(row * grid.columns + column);
    }
    map.addEventListener('pointermove', inspectCell);
    map.addEventListener('pointerdown', inspectCell);
    map.addEventListener('pointerleave', () => probe(null));
    updateLoads();
    setMode(false);
    return () => relief?.dispose();
  } catch (error: unknown) {
    console.warn('Open Knee(s) pressure data could not be opened', error);
    const message = document.createElement('p');
    message.textContent = 'Measured pressure data are unavailable.';
    message.setAttribute('role', 'status');
    message.className = 'pressure-error';
    container.replaceChildren(message);
    return () => {};
  }
}
