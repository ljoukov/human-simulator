import geometry from '../../hip-study/data/geometry.json';
import cohort from '../../hip-study/data/cohort.json';
import gtps from '../../hip-study/data/gtps.json';
import video from '../../hip-study/data/video-audit.json';
import './evidence.css';

const axes = geometry.axes;
const flex = axes.flexionDeg,
  add = axes.adductionDeg,
  rot = axes.rotationDeg;
const shape = (i, j, k) => (i * add.length + j) * rot.length + k;
const ns = 'http://www.w3.org/2000/svg';
function svg(tag, attributes, label) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (label !== undefined) node.textContent = label;
  return node;
}

function anatomyMap(host) {
  host.innerHTML = `
    <div class="card-head"><div><p class="eyebrow">OpenSim geometry</p><h3>How hip pose changes abductor leverage</h3></div>
      <label class="select-field">Muscle path<select class="ev-muscle"><option value="all">Combined abductors</option></select></label></div>
    <p class="caption">Each cell is a pose of the reference right hip. Moment arm is the leverage a muscle has to turn the joint. Select a cell or use the arrow keys to inspect it.</p>
    <div class="ev-map"></div>
    <div class="ev-map-controls"><label>Internal rotation <input type="range" class="ev-rotation" step="1" min="0"></label><output class="ev-rotation-value"></output><output class="ev-readout" aria-live="polite"></output></div>
    <p class="caption ev-model-note"></p><a class="text-link" target="_blank" rel="noreferrer" href="https://doi.org/10.1371/journal.pcbi.1014073">MRI personalization study ↗</a>`;
  const select = host.querySelector('.ev-muscle');
  for (const [id, label] of [
    ['glmed1_r', 'Medius · anterior'],
    ['glmed2_r', 'Medius · middle'],
    ['glmed3_r', 'Medius · posterior'],
    ['glmin1_r', 'Minimus · anterior'],
    ['glmin2_r', 'Minimus · middle'],
    ['glmin3_r', 'Minimus · posterior'],
  ])
    select.add(new Option(label, id));
  const slider = host.querySelector('.ev-rotation');
  slider.max = rot.length - 1;
  slider.value = rot.indexOf(0);
  const graph = svg('svg', {
    viewBox: '0 0 665 256',
    role: 'group',
    tabindex: '0',
    'aria-label':
      'Abduction moment arm in millimeters by hip flexion and adduction. Use arrow keys to select a pose.',
  });
  host.querySelector('.ev-map').append(graph);
  const x = 66,
    y = 16,
    w = 568,
    h = 171,
    cw = w / flex.length,
    ch = h / add.length;
  let fi = flex.indexOf(0),
    ai = add.indexOf(0);
  function cellValue(i, j) {
    const index = shape(i, j, Number(slider.value));
    return (
      1000 *
      (select.value === 'all'
        ? geometry.effectiveAbductionArmM[index]
        : -geometry.muscles[select.value].adductionArmM[index])
    );
  }
  const marker = svg('rect', {
    width: cw - 1,
    height: ch - 1,
    fill: 'none',
    stroke: '#102d38',
    'stroke-width': 3,
    'pointer-events': 'none',
  });
  const label = (x, y, s, extra = {}) =>
    graph.append(svg('text', { x, y, fill: '#415b57', 'font-size': 11, 'text-anchor': 'middle', ...extra }, s));
  function readout() {
    marker.setAttribute('x', x + fi * cw + 0.5);
    marker.setAttribute('y', y + (add.length - 1 - ai) * ch + 0.5);
    host.querySelector('.ev-readout').textContent =
      `${flex[fi]}° flexion · ${add[ai]}° adduction · ${cellValue(fi, ai).toFixed(1)} mm`;
  }
  function draw() {
    graph.replaceChildren();
    host.querySelector('.ev-rotation-value').textContent = rot[slider.value] + '°';
    for (let i = 0; i < flex.length; i++)
      for (let j = 0; j < add.length; j++) {
        const mm = cellValue(i, j);
        const t = Math.min(1, Math.max(0, Math.abs(mm) / 65));
        const lo = [234, 239, 229],
          hi = mm < 0 ? [157, 66, 47] : [22, 104, 95];
        const color = lo.map((v, c) => Math.round(v * (1 - t) + hi[c] * t)).join(',');
        const cell = svg('rect', {
          x: x + i * cw,
          y: y + (add.length - 1 - j) * ch,
          width: cw - 0.45,
          height: ch - 0.45,
          fill: 'rgb(' + color + ')',
        });
        const pick = () => {
          fi = i;
          ai = j;
          readout();
        };
        cell.addEventListener('pointerover', pick);
        cell.addEventListener('click', pick);
        graph.append(cell);
      }
    for (let i = 0; i < flex.length; i += 3) label(x + (i + 0.5) * cw, y + h + 16, flex[i] + '°');
    for (let j = 1; j < add.length; j += 2)
      label(x - 7, y + (add.length - 0.5 - j) * ch + 4, add[j] + '°', { 'text-anchor': 'end' });
    label(x + w / 2, y + h + 36, 'Hip flexion');
    label(0, 0, 'Hip adduction', { transform: 'translate(12 105) rotate(-90)' });
    label(x, 245, 'Darker teal indicates greater abduction leverage. Red indicates adduction leverage.', {
      'text-anchor': 'start',
    });
    graph.append(marker);
    readout();
  }
  graph.addEventListener('keydown', (e) => {
    const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (!delta) return;
    e.preventDefault();
    fi = Math.min(flex.length - 1, Math.max(0, fi + delta[0]));
    ai = Math.min(add.length - 1, Math.max(0, ai + delta[1]));
    readout();
  });
  slider.addEventListener('input', draw);
  select.addEventListener('change', draw);
  draw();
  const g = geometry.summary;
  host.querySelector('.ev-model-note').textContent =
    `In one participant’s paired models, the neutral effective arm measured ${(1000 * g.genericScaledExample.neutralM).toFixed(1)} mm with generic scaling and ${(1000 * g.MRIPersonalizedExample.neutralM).toFixed(1)} mm with MRI personalization. The force calculation assumes recruitment proportional to each segment’s maximum isometric force. This map uses the MRI-informed 2023 reference model.`;
}

export function mountEvidence(host, drawChart) {
  host.innerHTML = `<div class="section-head"><div><h2>Evidence from measured gait</h2><p class="sub">Clinical motion, model geometry and independent checks on the hip study.</p></div></div>
  <article class="card ev-symptomatic"></article><article class="card ev-video" hidden></article><article class="card ev-clinical"></article><article class="card ev-cohort"></article><article class="card ev-anatomy"></article>`;
  drawSymptomatic(host.querySelector('.ev-symptomatic'), gtps);
  if (cohort) drawClinical(host.querySelector('.ev-clinical'), cohort);
  if (video?.stride?.timeline?.length && drawChart) drawVideo(host.querySelector('.ev-video'), video, drawChart);
  drawCohort(host.querySelector('.ev-cohort'), cohort);
  anatomyMap(host.querySelector('.ev-anatomy'));
}

function drawVideo(host, video, drawChart) {
  host.hidden = false;
  const { stride } = video,
    rows = stride.timeline;
  host.innerHTML = `<div class="card-head"><div><p class="eyebrow">Measured video · one preoperative stride</p><h3>How much of the movement can a camera recover?</h3></div></div>
    <p class="caption">Two estimates of the same right thigh, seen from behind. The teal landmarks come from a pose tracker applied to the actual video. The dashed orange landmarks come from synchronized optical motion capture projected into the camera view.</p>
    <div class="ev-video-grid"><div class="ev-tracking"></div><div><h4>Thigh tilt in the image plane</h4><div class="ev-trace chart"></div><div class="legend"><span class="teal">Video</span><span class="coral">Projected optical reference</span></div>
    <label class="ev-frame-control">Stride phase <input type="range" min="0" step="1" aria-label="Video audit stride phase"><output></output></label></div></div>
    <p class="ev-video-summary"></p><details class="method"><summary>Camera registration &amp; interpretation</summary><p class="ev-calibration"></p></details>
    <p class="ev-links"><a class="text-link" rel="noreferrer" target="_blank">Original recording ↗</a></p>`;
  host.querySelector('.ev-links a').href = video.sourceUrl;
  const range = host.querySelector('input'),
    out = host.querySelector('output');
  range.max = rows.length - 1;
  const graph = svg('svg', { role: 'img', 'aria-label': 'One stride, overlaid video and optical landmarks' });
  host.querySelector('.ev-tracking').append(graph);
  const points = rows
    .flatMap((row) => [...(row.videoXY ?? []), ...(row.referenceXY ?? [])])
    .filter((p) => Array.isArray(p) && p.every(Number.isFinite));
  if (points.length) {
    const xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    const minX = Math.min(...xs) - 12,
      minY = Math.min(...ys) - 12;
    graph.setAttribute(
      'viewBox',
      [minX, minY, Math.max(...xs) + 12 - minX, Math.max(...ys) + 12 - minY].join(' '),
    );
  }
  const names = video.jointNames ?? [],
    connections = [
      ['leftShoulder', 'rightShoulder'],
      ['leftHip', 'rightHip'],
      ['rightShoulder', 'rightHip'],
      ['leftShoulder', 'leftHip'],
      ['leftHip', 'leftKnee'],
      ['leftKnee', 'leftAnkle'],
      ['rightHip', 'rightKnee'],
      ['rightKnee', 'rightAnkle'],
    ].map((pair) => pair.map((id) => names.indexOf(id)));
  const valid = (p) => Array.isArray(p) && p.every(Number.isFinite);
  function select(index) {
    const row = rows[index];
    range.value = index;
    out.textContent = `${Math.round(row.phase * 100)}% · ${row.videoDeg === null ? 'unobserved' : row.videoDeg.toFixed(1) + '° on video'}`;
    moveCursor(row.phase);
    graph.replaceChildren();
    for (const [key, color, dashed] of [
      ['referenceXY', '#e5a17c', true],
      ['videoXY', '#71d6be', false],
    ]) {
      const xy = row[key] ?? [];
      for (const [a, b] of connections) {
        if (!valid(xy[a]) || !valid(xy[b])) continue;
        const isThigh = names[a] === 'rightHip' && names[b] === 'rightKnee';
        graph.append(
          svg('line', {
            x1: xy[a][0],
            y1: xy[a][1],
            x2: xy[b][0],
            y2: xy[b][1],
            stroke: color,
            'stroke-width': isThigh ? 2.8 : 1.7,
            opacity: isThigh ? 1 : 0.78,
            'stroke-dasharray': dashed ? '3 2' : 'none',
          }),
        );
      }
      for (const p of xy.filter(valid))
        graph.append(
          svg('circle', {
            cx: p[0],
            cy: p[1],
            r: dashed ? 2.3 : 1.6,
            fill: dashed ? '#142a2d' : color,
            stroke: color,
            'stroke-width': 0.8,
          }),
        );
    }
  }
  const moveCursor = drawChart(host.querySelector('.ev-trace'), {
    frames: rows,
    yLabel: 'deg',
    title: 'Right thigh tilt in the image plane',
    axisLabel: 'Stride (%)',
    series: [
      { key: 'videoDeg', label: 'Video', color: '#187368' },
      { key: 'referenceDeg', label: 'Optical reference', color: '#bd6049', dashed: true },
    ],
    scrub: (phase) =>
      select(
        rows.reduce(
          (best, row, i) => (Math.abs(row.phase - phase) < Math.abs(rows[best].phase - phase) ? i : best),
          0,
        ),
      ),
  });
  range.oninput = () => select(range.valueAsNumber);
  select(Math.floor(rows.length / 2));
  host.querySelector('.ev-video-summary').textContent =
    `Across ${stride.nValid} of ${stride.nFrames} samples the two signals tracked in time (r = ${stride.pearsonR.toFixed(2)}), with ${stride.maeDeg.toFixed(1)}° mean absolute difference. The video measured ${stride.peakToPeakVideoDeg.toFixed(1)}° of excursion and optical tracking measured ${stride.peakToPeakReferenceDeg.toFixed(1)}° in projection.`;
  host.querySelector('.ev-calibration').textContent =
    `Camera position and timing were estimated from shoulders, hips and ankles outside the scored stride. Both knees were withheld from calibration. Holding the pre-stride video tilt constant gave a mean absolute error of ${stride.constantVideoBaselineMaeDeg.toFixed(1)}°. This comparison measures the angle of a thigh in the image plane of one recording. Anatomical hip angles also depend on the pelvis and on motion in depth.`;
}

function drawClinical(host, data) {
  const published = data.publishedStance;
  host.innerHTML = `<div class="card-head"><div><p class="eyebrow">Clinical comparison</p><h3>Gluteal tendinopathy and hip osteoarthritis</h3></div></div>
    <p class="caption">Gluteal tendinopathy affects tendons on the outer hip and is one contributor to greater trochanteric pain syndrome. Hip osteoarthritis affects the joint. Both can change how a person walks.</p>
    <div class="ev-clinical-grid"><div><h4>Hip excursion in stance</h4><p class="caption">Allison and colleagues measured hip flexion and extension as the foot bore weight. Bars show group means.</p><div class="ev-bars"></div></div>
    <div><h4>What clinicians saw on video</h4><p class="caption ev-rating"></p><div class="ev-ratings"></div></div></div>
    <p class="ev-clinical-summary"></p><p class="ev-links"><a class="text-link" target="_blank" rel="noreferrer" href="https://research-repository.griffith.edu.au/server/api/core/bitstreams/a6d57c6f-31a4-42ef-b226-9dab1fed15dd/content">Clinical comparison, Table 2 ↗</a><a class="text-link" target="_blank" rel="noreferrer" href="https://doi.org/10.1038/s41597-022-01483-3">Bertaux video ratings ↗</a></p>`;
  for (const [key, name] of [
    ['glutealTendinopathy', 'Gluteal tendinopathy'],
    ['osteoarthritis', 'Hip osteoarthritis'],
  ]) {
    const group = published[key],
      row = document.createElement('div');
    const label = document.createElement('span'),
      value = document.createElement('strong'),
      bar = document.createElement('i');
    label.textContent = `${name} · n = ${group.participants}`;
    value.textContent = `${group.meanDeg.toFixed(1)}°`;
    bar.style.setProperty('--width', Math.min(100, (100 * group.meanDeg) / 50) + '%');
    row.append(label, value, bar);
    host.querySelector('.ev-bars').append(row);
  }
  const observed = data.observedGait.groups.filter((g) => g.participants > 0);
  const count = observed.reduce((sum, g) => sum + g.participants, 0);
  host.querySelector('.ev-rating').textContent =
    `In a separate cohort of ${count} people awaiting hip replacement for osteoarthritis, clinicians recorded four walking patterns.`;
  for (const g of observed) {
    const row = document.createElement('div'),
      label = document.createElement('span'),
      n = document.createElement('strong');
    label.textContent = g.label;
    n.textContent = g.participants;
    row.append(label, n);
    host.querySelector('.ev-ratings').append(row);
  }
  host.querySelector('.ev-clinical-summary').textContent =
    `The published groups differed by ${published.reportedDifferenceDeg.toFixed(1)}° (95% CI ${published.difference95CI.join('–')}°). To measure diagnostic accuracy in one person, the workbench would need independently labeled videos covering both conditions and their coexistence.`;
}

function drawCohort(host, data) {
  if (!data) {
    host.textContent = 'The cohort results file is unavailable.';
    return;
  }
  host.innerHTML = `<div class="card-head"><div><p class="eyebrow">Independent clinical recordings</p><h3>Hip flexion and extension during walking</h3></div>
    <label class="select-field">Analysis window<select class="ev-window"><option value="loadedStance">Loaded stance</option><option value="cycle">Full gait cycle</option></select></label></div>
    <p class="caption">The Bertaux archive includes people awaiting hip replacement for osteoarthritis and asymptomatic controls. Each participant contributes one average. Lines show the bootstrap 95% confidence interval for the group mean.</p>
    <div class="ev-distribution"></div><p class="ev-matched"></p>
    <details class="method"><summary>Clinical comparison and measurement details</summary><p class="ev-comparison"></p><ul class="ev-method"></ul></details>
    <p class="ev-links"><a class="text-link" target="_blank" rel="noreferrer" href="https://doi.org/10.1038/s41597-022-01483-3">Bertaux walking archive ↗</a><a class="text-link" target="_blank" rel="noreferrer" href="https://doi.org/10.1016/j.gaitpost.2018.02.011">Allison clinical study ↗</a></p>`;
  const select = host.querySelector('.ev-window');
  function draw() {
    const obs = data[select.value],
      matched = obs.ageSexMatched;
    const field = select.value === 'loadedStance' ? 'loadedStanceDeg' : 'cycleDeg';
    const groups = [
      { name: 'Hip osteoarthritis', code: 'HOA', ...obs.HOA },
      { name: 'Asymptomatic', code: 'HEA', ...obs.HEA },
    ];
    host.querySelector('.ev-distribution').replaceChildren(distributionPlot(groups, data.people, field, 65, 10));
    host.querySelector('.ev-matched').textContent =
      `In ${matched.pairs} age- and sex-matched pairs, controls had ${matched.controlMinusOaDeg.toFixed(1)}° more excursion on average (95% CI ${matched.pairedDifference95CI.map((v) => v.toFixed(1)).join('–')}°).`;
  }
  const pub = data.publishedStance;
  host.querySelector('.ev-comparison').textContent =
    `In a separate study, Allison et al. measured ${pub.glutealTendinopathy.meanDeg}° of stance excursion with symptomatic gluteal tendinopathy (n = ${pub.glutealTendinopathy.participants}) and ${pub.osteoarthritis.meanDeg}° with hip osteoarthritis (n = ${pub.osteoarthritis.participants}). These are group-level measurements from a different protocol.`;
  for (const note of data.method) {
    const li = document.createElement('li');
    li.textContent = note;
    host.querySelector('.ev-method').append(li);
  }
  select.addEventListener('change', draw);
  draw();
}

function distributionPlot(groups, people, field, maxDeg, step, overlap) {
  const graph = svg('svg', {
    viewBox: '0 0 700 186',
    class: 'ev-cohort-plot',
    role: 'img',
    'aria-label': 'Participant hip excursion, group means and 95 percent confidence intervals',
  });
  const x0 = 150,
    w = 470;
  const x = (deg) => x0 + (deg / maxDeg) * w;
  const label = (at, y, text, attr = {}) =>
    graph.append(svg('text', { x: at, y, 'font-size': 12, fill: '#34534c', ...attr }, text));
  if (overlap)
    graph.append(
      svg('rect', { x: x(overlap[0]), width: x(overlap[1]) - x(overlap[0]), y: 12, height: 121, fill: '#e3e9dc' }),
    );
  for (let angle = 0; angle <= maxDeg; angle += step) {
    graph.append(svg('line', { x1: x(angle), x2: x(angle), y1: 12, y2: 133, stroke: '#dfe5db' }));
    label(x(angle), 151, angle + '°', { 'text-anchor': 'middle' });
  }
  label(x0 + w / 2, 176, 'Hip excursion', { 'text-anchor': 'middle' });
  for (const [index, group] of groups.entries()) {
    const y = 46 + index * 66,
      color = index ? '#19766e' : '#ac563e';
    label(x0 - 13, y - 3, group.name, { 'text-anchor': 'end' });
    label(x0 - 13, y + 15, 'n = ' + group.participants, { 'text-anchor': 'end', 'font-size': 11 });
    const binned = new Map();
    for (const person of people.filter((p) => p.group === group.code)) {
      const value = person[field];
      if (!Number.isFinite(value)) continue;
      const bin = Math.round(value / (maxDeg / 55)),
        slot = binned.get(bin) ?? 0;
      binned.set(bin, slot + 1);
      const offset = slot === 0 ? 0 : Math.ceil(slot / 2) * (slot % 2 ? 1 : -1) * 4.4;
      graph.append(svg('circle', { cx: x(value), cy: y + offset, r: 2.3, fill: color, opacity: 0.55 }));
    }
    graph.append(
      svg('line', {
        x1: x(group.mean95CI[0]),
        x2: x(group.mean95CI[1]),
        y1: y,
        y2: y,
        stroke: '#142f32',
        'stroke-width': 3,
      }),
    );
    graph.append(
      svg('circle', { cx: x(group.meanDeg), cy: y, r: 4.7, fill: '#172f2e', stroke: 'white', 'stroke-width': 1 }),
    );
    label(x0 + w + 7, y + 4, group.meanDeg.toFixed(1) + '°', { 'font-weight': 700 });
  }
  return graph;
}

function drawSymptomatic(host, data) {
  const g = data.gtps,
    o = data.oa,
    difference = data.comparison,
    { overlap } = difference;
  host.innerHTML = `<div class="card-head"><div><p class="eyebrow">Participant-level measurements · two clinics</p><h3>How much do the two groups overlap?</h3></div></div>
    <p class="caption ev-intro"></p><div class="ev-symptomatic-plot"></div><p class="ev-overlap"></p>
    <details class="method"><summary>Comparison limits &amp; sensitivity to phase</summary><ul class="ev-limits"></ul></details>
    <p class="ev-links"><a class="text-link" target="_blank" rel="noreferrer" href="https://data.mendeley.com/datasets/66mcwyk854/1">Robinson GTPS gait data ↗</a><a class="text-link" target="_blank" rel="noreferrer" href="https://doi.org/10.1038/s41597-022-01483-3">Bertaux OA gait data ↗</a></p>`;
  host.querySelector('.ev-intro').textContent =
    `Side-to-side hip excursion during the middle 20–80% of stance. Each dot represents one baseline record from women with greater trochanteric pain syndrome (GTPS) or preoperative hip osteoarthritis. The two conditions were recorded at different sites with different motion-capture models.`;
  const groups = [
    { name: 'Hip osteoarthritis', code: 'HOA', ...o },
    { name: 'GTPS', code: 'GTPS', ...g },
  ];
  host
    .querySelector('.ev-symptomatic-plot')
    .append(distributionPlot(groups, data.people, 'excursionDeg', 12, 2, overlap.sharedP10p90Deg));
  host.querySelector('.ev-overlap').textContent =
    `The group means differed by ${difference.gtpsMinusOaDeg.toFixed(2)}° (95% bootstrap CI ${difference.difference95CI.map((v) => v.toFixed(2)).join('–')}°). The shaded region is the intersection of the groups’ middle 80% ranges. It contains ${overlap.gtpsInside} of ${g.participants} GTPS records and ${overlap.oaInside} of ${o.participants} OA records. A value in that range fits either group.`;
  const notes = [
    `Using the entire available stance window changes the difference to ${data.sensitivity['0to100'].comparison.gtpsMinusOaDeg.toFixed(2)}°. The two labs defined the edges of stance differently.`,
    ...data.limitations,
  ];
  for (const note of notes) {
    const li = document.createElement('li');
    li.textContent = note;
    host.querySelector('.ev-limits').append(li);
  }
}
