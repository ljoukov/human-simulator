import './intervention.css';
import { DEFAULT_SETTINGS, PROFILES, sampleFrame, simulateGait } from './dynamics.js';
import { createBodyStage } from './body-stage.js';

const SIGNALS = {
  trunkLeanDeg: { label: 'Rightward trunk lean', unit: '°', digits: 1 },
  bandForceN: { label: 'IT band tension', unit: 'N', digits: 0 },
  pelvisDropDeg: { label: 'Left pelvic drop', unit: '°', digits: 1 },
  hipAdductionDeg: { label: 'Hip adduction', unit: '°', digits: 1 },
  hipFlexionDeg: { label: 'Hip flexion', unit: '°', digits: 1 },
  abductorForceN: { label: 'Hip abductor force', unit: 'N', digits: 0 },
  abductorUtilizationPct: { label: 'Share of available strength', unit: '%', digits: 1 },
  hipMomentNm: { label: 'Hip adduction moment', unit: 'N·m', digits: 1 },
};
const PRESETS = [
  { label: 'Reference', gluteCapacityPct: 100, bandShorteningMm: 0 },
  { label: 'Glute weakness', gluteCapacityPct: 55, bandShorteningMm: 0, strategy: 'effort', signal: 'trunkLeanDeg', view: 'front' },
  { label: 'Tighter IT band', gluteCapacityPct: 100, bandShorteningMm: 6, strategy: 'effort', signal: 'hipFlexionDeg', view: 'side' },
  { label: 'Combined', gluteCapacityPct: DEFAULT_SETTINGS.gluteCapacityPct, bandShorteningMm: DEFAULT_SETTINGS.bandShorteningMm, strategy: 'effort', signal: 'trunkLeanDeg', view: 'front' },
  { label: 'Less reserve', gluteCapacityPct: 75, bandShorteningMm: 0, strategy: 'reserve', task: 'walk', signal: 'abductorUtilizationPct', view: 'front' },
];
let saved = null;
const format = (value, digits) => (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits).replace('-', '−');
function suffix(unit, change = false) {
  if (unit === '°') return unit;
  if (unit === '%') return change ? ' pp' : '%';
  return ' ' + unit;
}

function options(node, entries) {
  for (const [value, label] of entries) node.add(new Option(label, value));
}

function largestChange(data, key) {
  return data.altered.reduce((peak, frame, i) => {
    const delta = frame[key] - data.baseline[i][key];
    return Math.abs(delta) > Math.abs(peak.delta) ? { phase: frame.phase, delta } : peak;
  }, { phase: 0, delta: 0 });
}

function waveform(root, data, key) {
  const svg = root.querySelector('.iv-chart');
  root.querySelector('[data-strength-note]').hidden = key !== 'abductorUtilizationPct';
  const chart = { w: svg.clientWidth || 320, h: svg.clientHeight || 77, left: 36, right: 5, top: 10, bottom: 15 };
  svg.setAttribute('viewBox', '0 0 ' + chart.w + ' ' + chart.h);
  root.querySelector('[data-cursor]').setAttribute('y2', chart.h - chart.bottom);
  const info = SIGNALS[key];
  const samples = [...data.baseline, ...data.altered].map(frame => frame[key]);
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const padding = Math.max((max - min) * 0.09, info.unit === 'N' ? 1 : 0.1);
  const lo = min - padding;
  const hi = max + padding;
  const plotWidth = chart.w - chart.left - chart.right;
  const px = phase => chart.left + phase * plotWidth;
  const py = value => chart.top + (hi - value) / (hi - lo) * (chart.h - chart.top - chart.bottom);
  const trace = frames => frames.map((frame, i) => (i === 0 ? 'M' : 'L') + px(frame.phase).toFixed(2) + ' ' + py(frame[key]).toFixed(2)).join(' ');
  root.querySelector('[data-base-line]').setAttribute('d', trace(data.baseline));
  root.querySelector('[data-alt-line]').setAttribute('d', trace(data.altered));
  const axes = root.querySelector('[data-guides]');
  axes.replaceChildren();
  const svgNode = (name, attrs, label) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.entries(attrs).forEach(([attr, value]) => element.setAttribute(attr, value));
    if (label !== undefined) element.textContent = label;
    return element;
  };
  const bounds = max > min + 0.01 ? [max, min] : [min];
  for (const y of bounds) {
    axes.append(svgNode('line', { x1: chart.left, x2: chart.w - chart.right, y1: py(y), y2: py(y) }));
    axes.append(svgNode('text', { x: chart.left - 3, y: py(y) + 3, 'text-anchor': 'end' }, format(y, info.digits)));
  }
  axes.append(svgNode('text', { x: chart.left, y: chart.h - 1 }, '0%'));
  axes.append(svgNode('text', { x: chart.w - chart.right, y: chart.h - 1, 'text-anchor': 'end' }, '100%'));
  const best = largestChange(data, key);
  const jump = root.querySelector('[data-peak]');
  jump.disabled = Math.abs(best.delta) < 1e-7;
  const unit = suffix(info.unit, true);
  const sign = best.delta > 0 ? '+' : '';
  jump.textContent = jump.disabled ? 'No difference' : 'Peak Δ ' + sign + format(best.delta, info.digits) + unit + ' ↗';
  jump.setAttribute('aria-label', jump.disabled ? 'No difference throughout movement' : 'Pause at largest ' + info.label.toLowerCase() + ' difference, ' + sign + format(best.delta, info.digits) + unit + ' at ' + Math.round(best.phase * 100) + ' percent');
  return { px, peak: best.phase };
}

export function mountIntervention(root) {
  const state = saved ? { ...saved, settings: { ...saved.settings } } : {
    settings: { ...DEFAULT_SETTINGS }, signal: 'trunkLeanDeg', phase: 0.3,
    playing: !window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    layer: 'muscles', view: 'front', region: 'body', rate: 0.5,
  };
  root.classList.add('int-host');
  root.innerHTML = `
    <div class="iv-room">
      <div class="iv-visual">
        <div class="iv-tools">
          <label>Focus <select data-region aria-label="Anatomical focus"><option value="body">Full body</option><option value="hip">Hip</option><option value="knee">Knee</option></select></label>
          <label>Layer <select data-layer aria-label="Anatomical layer"><option value="muscles">Muscles</option><option value="bones">Bones</option><option value="surface">Surface</option></select></label>
          <label>View <select data-view aria-label="Camera view"><option value="front">Front</option><option value="oblique">Oblique</option><option value="side">Side</option><option value="back">Back</option></select></label>
        </div>
        <div class="iv-stage"><div class="iv-canvas" data-stage></div><div class="iv-status" data-status role="status">Loading anatomy…</div></div>
        <div class="iv-transport">
          <button class="iv-play" data-play type="button" aria-label="Pause movement"></button>
          <label class="iv-phase">Movement phase <output data-phase-label>30%</output><input aria-label="Movement phase" data-phase type="range" min="0" max="1000" step="1"></label>
          <select data-rate aria-label="Playback speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1">1×</option></select>
        </div>
      </div>
      <div class="iv-controls">
        <div class="iv-config">
          <div class="iv-profile">
            <label for="iv-person">Virtual person</label><select id="iv-person" data-person></select>
            <div class="iv-profile-meta"><span class="iv-profile-detail" data-profile-detail></span>
              <details class="iv-methods" data-methods><summary>How the motion is calculated</summary>
                <div>Each profile uses the same recorded reference stride for walking and the same prescribed weight shift for loaded stance. The solver couples 16 body angles. Height scales body lengths, and height and mass scale inertia. Age and sex describe the profiles. A tension-only cable represents the IT band and pulls when stretched beyond its resting length.</div>
              </details>
            </div>
          </div>
          <div class="iv-tasks" role="group" aria-label="Movement">
            <button type="button" data-task="walk">Walking</button><button type="button" data-task="stance">Loaded stance</button>
          </div>
          <div class="iv-strategy" role="group" aria-label="Compensation strategy"><span>Compensation</span><button type="button" data-strategy="effort">Favor less muscle effort</button><button type="button" data-strategy="reserve">Use available strength first</button></div>
        </div>
        <div class="iv-settings">
          <label class="iv-slider"><span>Right glute strength loss</span><output data-loss-value></output><input type="range" data-loss min="0" max="60" step="1" aria-label="Right glute strength loss"></label>
          <label class="iv-slider"><span>Right IT band shortening</span><output data-band-value></output><input type="range" data-band min="0" max="8" step="0.5" aria-label="Right IT band shortening"></label>
          <div class="iv-presets" data-presets role="group" aria-label="Parameter presets"></div>
        </div>
        <div class="iv-effects">
          <span data-effects-title>Largest changes over the stride</span>
          <div><button type="button" data-effect="trunkLeanDeg">Rightward trunk lean <strong></strong></button><button type="button" data-effect="bandForceN" data-tissue-effect><span>IT band tension</span><strong></strong></button></div>
        </div>
        <div class="iv-plot">
          <div class="iv-plothead"><select aria-label="Modeled response" data-signal></select><button type="button" data-peak title="Pause at the largest absolute difference">Peak difference <span aria-hidden="true">↗</span></button></div>
          <p class="iv-strength-note" data-strength-note hidden>Required abductor torque ÷ available torque, capped at 100%.</p>
          <div class="iv-values"><div><span>Reference</span><strong data-base-value></strong></div><div><span>Adjusted</span><strong data-alt-value></strong></div></div>
          <svg viewBox="0 0 320 77" class="iv-chart" role="img" aria-label="Computed response across the movement, reference and adjusted"><g data-guides></g><path class="iv-baseline" data-base-line/><path class="iv-changed" data-alt-line/><line class="iv-cursor" data-cursor y1="5" y2="66"/></svg>
        </div>
      </div>
    </div>`;
  const find = key => root.querySelector('[data-' + key + ']');
  const person = find('person'), signal = find('signal'), loss = find('loss'), band = find('band');
  const phase = find('phase'), play = find('play'), stageHost = find('stage'), status = find('status');
  options(person, PROFILES.map(profile => [profile.id, profile.label]));
  options(signal, Object.entries(SIGNALS).map(([value, info]) => [value, info.label + ' (' + info.unit + ')']));
  for (const [index, preset] of PRESETS.entries()) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.preset = index;
    button.textContent = preset.label;
    find('presets').append(button);
  }
  let data, graph, stage, raf = 0, lastTime, lastTextTime = 0;
  let alive = true, active = true, stageReady = false, runtimeError = false;
  const persist = () => { saved = { ...state, settings: { ...state.settings } }; };
  const controller = new AbortController();
  const listen = (target, event, fn) => target.addEventListener(event, fn, { signal: controller.signal });
  const methods = find('methods');
  listen(document, 'pointerdown', event => { if (methods.open && !methods.contains(event.target)) methods.open = false; });
  listen(methods, 'keydown', event => {
    if (event.key === 'Escape') { methods.open = false; methods.querySelector('summary').focus(); event.stopPropagation(); }
  });
  function draw(text = true) {
    const base = sampleFrame(data.baseline, state.phase);
    const changed = sampleFrame(data.altered, state.phase);
    if (stageReady && !runtimeError) {
      try {
        stage.setPose({ phase: state.phase, signal: state.signal, baseline: base, changed, task: data.settings.task, heightM: data.profile.heightCm / 100 });
      } catch (error) {
        runtimeError = true;
        status.textContent = 'Anatomy unavailable. The phase chart is active.';
        status.hidden = false;
        console.error(error);
      }
    }
    if (!text) return;
    phase.value = Math.round(1000 * state.phase);
    find('phase-label').value = Math.round(100 * state.phase) + '%';
    const { unit, digits } = SIGNALS[state.signal];
    find('base-value').textContent = format(base[state.signal], digits) + suffix(unit);
    find('alt-value').textContent = format(changed[state.signal], digits) + suffix(unit);
    const cursor = find('cursor'), x = graph.px(state.phase);
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
  }
  function tick(now) {
    raf = 0;
    if (!alive || !active || !state.playing || document.hidden) return;
    const elapsed = lastTime === undefined ? 0 : Math.min(100, now - lastTime);
    lastTime = now;
    state.phase = (state.phase + elapsed * state.rate / (1000 * data.durationSeconds)) % 1;
    const updateText = now - lastTextTime >= 40;
    if (updateText) lastTextTime = now;
    draw(updateText);
    raf = requestAnimationFrame(tick);
  }
  function playback() {
    cancelAnimationFrame(raf);
    lastTime = undefined;
    play.textContent = state.playing ? 'Ⅱ' : '▶';
    play.setAttribute('aria-label', state.playing ? 'Pause movement' : 'Play movement');
    play.title = play.getAttribute('aria-label');
    if (active && state.playing && !document.hidden) raf = requestAnimationFrame(tick);
  }
  function update() {
    data = simulateGait(state.settings);
    graph = waveform(root, data, state.signal);
    person.value = data.profile.id;
    for (const node of root.querySelectorAll('[data-strategy]')) node.setAttribute('aria-pressed', String(node.dataset.strategy === state.settings.strategy));
    signal.value = state.signal;
    loss.value = 100 - state.settings.gluteCapacityPct;
    band.value = state.settings.bandShorteningMm;
    loss.setAttribute('aria-valuetext', loss.value + ' percent strength loss');
    band.setAttribute('aria-valuetext', band.value + ' millimeters shorter');
    find('loss-value').value = loss.value + '%';
    find('band-value').value = Number(band.value) + ' mm';
    find('profile-detail').textContent = data.profile.heightCm + ' cm · ' + data.profile.massKg + ' kg';
    find('effects-title').textContent = state.settings.task === 'walk' ? 'Largest changes over the stride' : 'Largest changes during the stance';
    const tissue = find('tissue-effect');
    const isolatedWeakness = state.settings.gluteCapacityPct < 100 && state.settings.bandShorteningMm === 0;
    tissue.dataset.effect = isolatedWeakness ? 'abductorUtilizationPct' : 'bandForceN';
    tissue.querySelector('span').textContent = isolatedWeakness ? 'Available strength used' : 'IT band tension';
    if (isolatedWeakness && state.settings.strategy === 'reserve') {
      const angles = Object.keys(data.baseline[0]).filter(key => key.endsWith('Deg'));
      const sameMotion = data.altered.every((frame, index) =>
        angles.every(key => Math.abs(frame[key] - data.baseline[index][key]) < 1e-7));
      if (sameMotion) find('effects-title').textContent = 'Both move alike. The right hip uses more of its available strength.';
    }
    for (const node of root.querySelectorAll('[data-effect]')) {
      const key = node.dataset.effect, info = SIGNALS[key], effect = largestChange(data, key);
      const unit = suffix(info.unit, true);
      node.querySelector('strong').textContent = (effect.delta > 0 ? '+' : '') + format(effect.delta, info.digits) + unit;
      node.disabled = Math.abs(effect.delta) < 1e-7;
      node.title = 'Show this difference at ' + Math.round(effect.phase * 100) + '% of the movement';
    }
    for (const node of root.querySelectorAll('[data-task]')) node.setAttribute('aria-pressed', String(node.dataset.task === state.settings.task));
    for (const node of root.querySelectorAll('[data-preset]')) {
      const preset = PRESETS[Number(node.dataset.preset)];
      const selected = preset.gluteCapacityPct === state.settings.gluteCapacityPct && preset.bandShorteningMm === state.settings.bandShorteningMm &&
        (!preset.strategy || preset.strategy === state.settings.strategy) && (!preset.task || preset.task === state.settings.task);
      node.setAttribute('aria-pressed', String(selected));
    }
    draw();
    persist();
  }
  listen(person, 'change', () => { state.settings.profileId = person.value; update(); });
  for (const button of root.querySelectorAll('[data-strategy]')) listen(button, 'click', () => {
    state.settings.strategy = button.dataset.strategy;
    update();
  });
  listen(loss, 'input', () => { state.settings.gluteCapacityPct = 100 - Number(loss.value); update(); });
  listen(band, 'input', () => { state.settings.bandShorteningMm = Number(band.value); update(); });
  listen(signal, 'change', () => { state.signal = signal.value; graph = waveform(root, data, state.signal); draw(); persist(); });
  listen(find('presets'), 'click', event => {
    const button = event.target.closest('[data-preset]');
    if (!button) return;
    const preset = PRESETS[Number(button.dataset.preset)];
    Object.assign(state.settings, { gluteCapacityPct: preset.gluteCapacityPct, bandShorteningMm: preset.bandShorteningMm });
    if (preset.strategy) state.settings.strategy = preset.strategy;
    if (preset.task) state.settings.task = preset.task;
    if (preset.signal) {
      state.signal = preset.signal;
      state.view = preset.view;
      find('view').value = preset.view;
      if (stageReady) stage.setView(preset.view);
    }
    update();
    if (preset.signal) pauseAt(graph.peak);
  });
  for (const button of root.querySelectorAll('[data-task]')) listen(button, 'click', () => {
    state.settings.task = button.dataset.task;
    update();
  });
  listen(play, 'click', () => { state.playing = !state.playing; playback(); persist(); });
  const pauseAt = next => { state.playing = false; state.phase = next; playback(); draw(); persist(); };
  for (const button of root.querySelectorAll('[data-effect]')) listen(button, 'click', () => {
    state.signal = button.dataset.effect;
    signal.value = state.signal;
    graph = waveform(root, data, state.signal);
    pauseAt(graph.peak);
  });
  listen(phase, 'input', () => pauseAt(Number(phase.value) / 1000));
  listen(find('peak'), 'click', () => pauseAt(graph.peak));
  for (const [control, method] of [['view', 'setView'], ['layer', 'setLayer'], ['region', 'setRegion']]) {
    const select = find(control);
    select.value = state[control];
    listen(select, 'change', () => { state[control] = select.value; if (stageReady) stage[method](select.value); persist(); });
  }
  const rate = find('rate');
  rate.value = state.rate;
  listen(rate, 'change', () => { state.rate = Number(rate.value); persist(); });
  listen(document, 'visibilitychange', playback);
  update();
  const chartResize = new ResizeObserver(() => {
    if (!alive) return;
    graph = waveform(root, data, state.signal);
    draw();
  });
  chartResize.observe(root.querySelector('.iv-chart'));
  playback();
  Promise.resolve().then(() => createBodyStage(stageHost, { comparison: true, view: state.view })).then(result => {
    if (!alive) { result.dispose(); return; }
    stage = result; stageReady = true;
    stage.setRegion(state.region); stage.setLayer(state.layer); stage.setView(state.view); stage.setActive(active);
    status.hidden = true;
    draw();
  }).catch(error => {
    if (!alive) return;
    status.textContent = 'Anatomy unavailable. The phase chart is active.';
    console.error(error);
  });
  return {
    setActive(value) { active = value; if (stageReady) stage.setActive(value); playback(); },
    dispose() { alive = false; active = false; cancelAnimationFrame(raf); chartResize.disconnect(); persist(); controller.abort(); if (stageReady) stage.dispose(); root.classList.remove('int-host'); },
  };
}
