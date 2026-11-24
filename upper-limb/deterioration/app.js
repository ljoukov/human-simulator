import { DEFAULT_SETTINGS, PROFILES, simulateGait, sampleFrame } from '../presentation/dynamics.js';

const control = name => document.querySelector('[data-' + name + ']');
const elements = name => [...document.querySelectorAll('[data-' + name + ']')];
const presets = {
  reference: { loss: 0, shortening: 0 },
  glutes: { loss: 45, shortening: 0, view: 'front', tissue: 'glutes' },
  band: { loss: 0, shortening: 6, view: 'side', tissue: 'band' },
  combined: { loss: 100 - DEFAULT_SETTINGS.gluteCapacityPct, shortening: DEFAULT_SETTINGS.bandShorteningMm, view: 'front', tissue: 'band' },
};
const signals = {
  trunkLeanDeg: { label: 'Rightward trunk lean', short: 'lean', unit: '°', digits: 1 },
  pelvisDropDeg: { label: 'Opposite pelvic drop', short: 'pelvic drop', unit: '°', digits: 1 },
  hipFlexionDeg: { label: 'Right hip flexion', short: 'flexion', unit: '°', digits: 1 },
  abductorUtilizationPct: { label: 'Abductor capacity used', unit: '%', digits: 1 },
  bandForceN: { label: 'IT band tension', unit: ' N', digits: 0 },
};
function format(value, metric) {
  const rounded = Number(value.toFixed(metric.digits));
  return rounded.toFixed(metric.digits).replace('-', '−') + metric.unit;
}

const settings = { ...DEFAULT_SETTINGS };
const ui = Object.fromEntries([
  'person','task','strategy','loss','loss-value','band','band-value','view','tissue','phase','phase-value','play','speed',
  'jump','peak-text','sweep','reset-camera','fullscreen','profile-size','tint-min','tint-max','tint-name',
  'body-reference-value','body-adjusted-value','hip-reference-value','hip-adjusted-value',
].map(name => [name, control(name)]));
for (const p of PROFILES) ui.person.add(new Option(p.label, p.id));
ui.person.value = settings.profileId;
ui.task.value = settings.task;
ui.strategy.value = settings.strategy;
ui.view.value = 'front';
ui.tissue.value = 'band';

let solution, phase = 0.3, peakPhase = phase, speed = Number(ui.speed.value);
let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
let disposed = false, ready = false, animation = 0, lastTime, sweep;
const stages = { body: null, hip: null };
const bodyKey = () => ({front:'trunkLeanDeg', pelvis:'pelvisDropDeg', side:'hipFlexionDeg', oblique:'trunkLeanDeg'})[ui.view.value];
const cameraView = () => ui.view.value === 'pelvis' ? 'front' : ui.view.value;
const hipKey = () => ui.tissue.value === 'glutes' ? 'abductorUtilizationPct' : 'bandForceN';

function draw() {
  const baseline = sampleFrame(solution.baseline, phase);
  const altered = sampleFrame(solution.altered, phase);
  stages.body?.setFrames(baseline, altered, { phase, task: settings.task, heightM: solution.profile.heightCm / 100 });
  stages.hip?.setFrames(baseline, altered);
  const body = bodyKey(), hip = hipKey();
  for (const [column, frame] of [['reference', baseline], ['adjusted', altered]]) {
    ui['body-' + column + '-value'].value = format(frame[body], signals[body]);
    ui['hip-' + column + '-value'].value = format(frame[hip], signals[hip]);
  }
  const phaseValue = Math.round(phase * 100) + '%';
  ui.phase.value = Math.round(phase * 1000);
  ui['phase-value'].value = phaseValue;
  ui.phase.setAttribute('aria-valuetext', phaseValue);
}

function updatePeak() {
  const key = bodyKey(), metric = signals[key];
  let largest = 0;
  peakPhase = phase;
  for (const [i, frame] of solution.altered.entries()) {
    const delta = frame[key] - solution.baseline[i][key];
    if (Math.abs(delta) > Math.abs(largest)) {
      largest = delta;
      peakPhase = frame.phase;
    }
  }
  ui.jump.disabled = Math.abs(largest) < 0.05 || Boolean(sweep);
  const deltaText = (largest > 0 ? '+' : '') + format(largest, metric);
  ui['peak-text'].textContent = Math.abs(largest) < .05 ? 'No change in ' + metric.short : 'Peak ' + metric.short + ' change ' + deltaText;
  ui.jump.title = 'Pause at ' + Math.round(100 * peakPhase) + '% of the movement';
}

// For combined changes, open where both compared signals differ. The peak button
// always goes to the actual maximum angle difference, which may occur with a slack band.
function combinedPhase() {
  const changes = solution.altered.map((frame, i) => ({
    phase: frame.phase,
    angle: Math.abs(frame.trunkLeanDeg - solution.baseline[i].trunkLeanDeg),
    tension: Math.abs(frame.bandForceN - solution.baseline[i].bandForceN),
  }));
  const maxAngle = Math.max(...changes.map(f => f.angle));
  const maxTension = Math.max(...changes.map(f => f.tension));
  if (!maxAngle || !maxTension) return peakPhase;
  let bestPhase = peakPhase, bestScore = 0;
  for (const frame of changes) {
    const score = Math.min(frame.angle / maxAngle, frame.tension / maxTension);
    if (score > bestScore) { bestScore = score; bestPhase = frame.phase; }
  }
  return bestPhase;
}

function solve() {
  solution = simulateGait(settings);
  const loss = 100 - settings.gluteCapacityPct, mm = settings.bandShorteningMm;
  ui.loss.value = loss;
  ui.band.value = mm;
  ui['loss-value'].value = loss + '%';
  ui['band-value'].value = mm + ' mm';
  ui.loss.setAttribute('aria-valuetext', ui['loss-value'].value);
  ui.band.setAttribute('aria-valuetext', ui['band-value'].value);
  ui['profile-size'].textContent = solution.profile.heightCm + ' cm · ' + solution.profile.massKg + ' kg';
  for (const button of elements('preset')) {
    const preset = presets[button.dataset.preset];
    button.setAttribute('aria-pressed', String(preset.loss === loss && preset.shortening === mm));
  }
  ui.sweep.disabled = !sweep && !loss && !mm;
  updatePeak();
}

function showBodySignal() {
  stages.body?.setView(cameraView());
  stages.body?.setSignal(bodyKey());
  for (const label of elements('body-metric')) label.textContent = signals[bodyKey()].label;
}
function showHipSignal() {
  stages.hip?.setTissue(ui.tissue.value);
  for (const label of elements('hip-metric')) label.textContent = signals[hipKey()].label;
  const band = ui.tissue.value === 'band';
  ui['tint-name'].textContent = 'Modeled ' + (band ? 'IT band tension' : 'abductor capacity used');
  ui['tint-min'].textContent = band ? '0 N' : '0%';
  ui['tint-max'].textContent = band ? '≥500 N' : '100%';
}

function transport() {
  ui.play.textContent = playing ? 'Pause' : 'Play';
  ui.play.setAttribute('aria-label', playing ? 'Pause movement' : 'Play movement');
  ui.sweep.textContent = sweep ? 'Stop sweep' : 'Sweep severity';
  ui.sweep.setAttribute('aria-pressed', String(Boolean(sweep)));
}

function tick(now) {
  animation = 0;
  if (disposed || document.hidden || (!playing && !sweep)) return;
  // One clock drives all four poses. Avoid computing tissue contact between displayed frames.
  if (lastTime === undefined || now - lastTime >= 1000 / 30 - 1) {
    const dt = lastTime === undefined ? 0 : Math.min(100, now - lastTime);
    lastTime = now;
    if (sweep) {
      sweep.elapsed += dt;
      const fraction = Math.min(sweep.elapsed / 2800, 1);
      const loss = Math.round(fraction * sweep.loss);
      const shortening = Math.round(fraction * sweep.shortening * 2) / 2;
      const changed = 100 - loss !== settings.gluteCapacityPct || shortening !== settings.bandShorteningMm;
      if (fraction === 1) { sweep = null; transport(); }
      if (changed || !sweep) {
        settings.gluteCapacityPct = 100 - loss;
        settings.bandShorteningMm = shortening;
        solve();
        draw();
      }
    } else {
      phase = (phase + dt * speed / (1000 * solution.durationSeconds)) % 1;
      draw();
    }
  }
  if (playing || sweep) animation = requestAnimationFrame(tick);
}

function schedule() {
  cancelAnimationFrame(animation);
  animation = 0;
  lastTime = undefined;
  if (ready && !disposed && !document.hidden && (playing || sweep)) animation = requestAnimationFrame(tick);
  transport();
}

function pause() {
  playing = false;
  if (sweep) {
    sweep = null;
    solve();
  }
  schedule();
}

for (const button of elements('preset')) {
  button.addEventListener('click', () => {
    pause();
    const preset = presets[button.dataset.preset];
    settings.gluteCapacityPct = 100 - preset.loss;
    settings.bandShorteningMm = preset.shortening;
    if (preset.view) ui.view.value = preset.view;
    if (preset.tissue) ui.tissue.value = preset.tissue;
    showBodySignal();
    showHipSignal();
    solve();
    if (button.dataset.preset === 'combined') phase = combinedPhase();
    else if (preset.view && !ui.jump.disabled) phase = peakPhase;
    draw();
  });
}
for (const [input, property] of [[ui.person, 'profileId'], [ui.task, 'task'], [ui.strategy, 'strategy']]) {
  input.addEventListener('change', () => {
    pause(); settings[property] = input.value; solve(); draw();
  });
}
for (const range of [ui.loss, ui.band]) {
  range.addEventListener('input', () => {
    // Capture first because interrupting an active severity sweep refreshes the controls.
    const value = Number(range.value);
    pause();
    if (range === ui.loss) settings.gluteCapacityPct = 100 - value;
    else settings.bandShorteningMm = value;
    solve(); draw();
  });
}
ui.view.addEventListener('change', () => {
  showBodySignal();
  updatePeak(); draw();
});
ui.tissue.addEventListener('change', () => {
  showHipSignal();
  draw();
});
ui.sweep.addEventListener('click', () => {
  if (sweep) { pause(); return; }
  pause();
  sweep = { loss: 100 - settings.gluteCapacityPct, shortening: settings.bandShorteningMm, elapsed: 0 };
  settings.gluteCapacityPct = 100; settings.bandShorteningMm = 0;
  solve(); draw(); schedule();
});
ui.play.addEventListener('click', () => {
  const start = !playing;
  pause(); playing = start; schedule();
});
ui.phase.addEventListener('input', () => {
  const requested = Number(ui.phase.value) / 1000;
  pause(); phase = requested; draw();
});
ui.speed.addEventListener('change', () => { speed = Number(ui.speed.value); });
ui.jump.addEventListener('click', () => { pause(); phase = peakPhase; draw(); });
ui['reset-camera'].addEventListener('click', () => { stages.body?.reset(); stages.hip?.reset(); });
document.addEventListener('visibilitychange', schedule);

const full = ui.fullscreen;
if (!document.documentElement.requestFullscreen && !document.documentElement.webkitRequestFullscreen) full.hidden = true;
function refreshFullscreen() {
  const active = Boolean(document.fullscreenElement || document.webkitFullscreenElement);
  full.textContent = active ? 'Exit full screen' : 'Full screen';
  full.setAttribute('aria-label', active ? 'Exit full screen' : 'Enter full screen');
}
full.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
      const host = document.documentElement;
      await (host.requestFullscreen || host.webkitRequestFullscreen).call(host);
    }
  } catch { full.title = 'Full screen is unavailable in this window'; }
});
document.addEventListener('fullscreenchange', refreshFullscreen);
document.addEventListener('webkitfullscreenchange', refreshFullscreen);

async function loadPair(kind) {
  const hosts = [control(kind + '-reference'), control(kind + '-adjusted')];
  const statuses = elements(kind + '-status');
  for (const status of statuses) { status.hidden = false; status.textContent = 'Loading anatomy…'; }
  try {
    const pair = kind === 'body'
      ? await (await import('./body-pair.js')).createBodyPair(hosts)
      : await (await import('./hip-pair.js')).createHipPair(hosts, { tissue: ui.tissue.value });
    if (disposed) { pair.dispose(); return; }
    stages[kind] = pair;
    if (kind === 'body') { pair.setView(cameraView()); pair.setSignal(bodyKey()); }
    else pair.setTissue(ui.tissue.value);
    for (const status of statuses) status.hidden = true;
    draw();
  } catch (error) {
    if (disposed) return;
    console.error('Could not initialize ' + kind + ' anatomy', error);
    for (const status of statuses) {
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = 'Retry';
      retry.addEventListener('click', () => loadPair(kind), { once: true });
      status.replaceChildren('3D view unavailable. ', retry);
    }
  }
}

showBodySignal(); showHipSignal(); solve(); phase = combinedPhase(); draw(); transport();
Promise.all([loadPair('body'), loadPair('hip')]).then(() => { ready = true; schedule(); });
window.addEventListener('pagehide', event => {
  cancelAnimationFrame(animation);
  if (event.persisted) return;
  disposed = true;
  stages.body?.dispose(); stages.hip?.dispose();
});
window.addEventListener('pageshow', event => { if (event.persisted) schedule(); });
