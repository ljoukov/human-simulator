import './cover.css';

const strideSeconds = 1.09;
const layers = {
  muscles: {label: 'Muscles', detail: '386 muscle and other soft-tissue structures'},
  bones: {label: 'Skeleton', detail: '242 skeletal structures'},
  surface: {label: 'Surface', detail: 'Outer body surface'},
};
const views = ['oblique', 'front', 'side', 'back'];
const posters = import.meta.glob('./media/atlas-*.webp', {eager: true, query: '?url', import: 'default'});
const icons = {
  arrow: '<path d="M4 9h11m-4.5-4.5L15 9l-4.5 4.5"/>',
  rotate: '<path d="M3.3 7.2C5.2 4 12.7 3.6 15.4 6.3c2.2 2.3-.7 5.4-6.3 5.8M3 3.9v3.8h3.8"/>',
  pause: '<path d="M6.5 4v10m5-10v10"/>',
  play: '<path d="m6 3.8 7.9 5.2L6 14.2Z"/>',
};
function icon(name) {
  return '<svg viewBox="0 0 18 18" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round">'+icons[name]+'</svg>';
}

export function mountCover(root) {
  root.innerHTML = `
    <div class="cover-layout">
      <div class="cover-story">
        <h1>The forces<br> behind a step</h1>
        <p>Watch a recorded stride, then change one hip in a virtual walker. Follow the forces through the body and compare the calculations with measurements.</p>
        <div class="cover-links">
          <a href="#whole-body" class="cover-primary">Compare with force plates ${icon('arrow')}</a>
          <a href="../deterioration/" class="cover-secondary">Deterioration effects ${icon('arrow')}</a>
        </div>
      </div>
      <figure class="cover-atlas" aria-label="Interactive anatomy from a recorded walking stride">
        <div class="cover-visual">
          <div class="cover-stage"></div>
          <span class="cover-viewnote"><span class="cover-motion">Recorded stride</span><span class="cover-angle">Oblique view</span></span>
          <button class="cover-rotate" type="button" title="Cycle camera angle" aria-label="Cycle camera angle">${icon('rotate')}</button>
          <div class="cover-layer-menu" role="group" aria-label="Visible anatomy">
            ${Object.entries(layers).map(([key,layer])=>`<button type="button" data-layer="${key}" aria-pressed="${key==='muscles'}">${layer.label}</button>`).join('')}
          </div>
        </div>
        <figcaption>
          <div class="cover-anatomy-label"><a href="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html" target="_blank" rel="noreferrer" title="BodyParts3D, © The Database Center for Life Science. Adapted under CC BY-SA 2.1 Japan.">BodyParts3D atlas</a><strong class="cover-detail"></strong></div>
          <div class="cover-playback">
            <button class="cover-play" type="button" title="Pause walking"></button>
            <div class="cover-track">
              <a href="https://doi.org/10.1038/s41597-023-02767-y" target="_blank" rel="noreferrer" title="Van Criekinge et al., optical recording SUBJ130">Walker 130 · 1.09 s</a>
              <input class="cover-phase" type="range" min="0" max="100" step="1" value="27" aria-label="Recorded stride phase">
            </div>
            <output class="cover-readout">27%</output>
          </div>
        </figcaption>
      </figure>
    </div>`;
  const stageHost = root.querySelector('.cover-stage');
  const motionNote = root.querySelector('.cover-motion');
  const angleNote = root.querySelector('.cover-angle');
  const detail = root.querySelector('.cover-detail');
  const playButton = root.querySelector('.cover-play');
  const rotateButton = root.querySelector('.cover-rotate');
  const phaseInput = root.querySelector('.cover-phase');
  const phaseOutput = root.querySelector('.cover-readout');
  const layerButtons = [...root.querySelectorAll('[data-layer]')];
  const events = new AbortController();
  const options = {signal: events.signal};
  let layer = 'muscles', view = 0, phase = 0.27, stage;
  let disposed = false, active = true, still = false, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let raf = 0, lastDraw = null;

  function updatePlay() {
    const label = playing ? 'Pause walking' : 'Play recorded stride';
    playButton.setAttribute('aria-label', label);
    playButton.title = label;
    playButton.innerHTML = icon(playing ? 'pause' : 'play');
  }
  function updatePose() {
    stage?.setPose({phase});
    const percent = Math.round(phase * 100);
    phaseInput.value = percent;
    phaseInput.style.setProperty('--cover-phase', percent+'%');
    phaseOutput.value = percent+'%';
  }
  function tick(time) {
    if (!active || !playing || disposed || !stage || document.hidden) { raf=0; lastDraw=null; return; }
    if (lastDraw === null || time-lastDraw >= 1000/30) {
      if (lastDraw !== null) phase = (phase + (time-lastDraw)/(1000*strideSeconds)) % 1;
      lastDraw = time;
      updatePose();
    }
    raf = requestAnimationFrame(tick);
  }
  function sync() {
    if (raf) cancelAnimationFrame(raf);
    raf=0; lastDraw=null;
    stage?.setActive(active && !document.hidden);
    if (active && playing && stage && !disposed && !document.hidden) raf = requestAnimationFrame(tick);
  }
  function showStill() {
    const name = views[view];
    stageHost.style.setProperty('--cover-still', 'url("'+posters['./media/atlas-'+layer+'-'+name+'.webp']+'")');
    motionNote.textContent = 'Recorded pose';
    angleNote.textContent = name[0].toUpperCase()+name.slice(1)+' view';
    stageHost.setAttribute('role', 'img');
    stageHost.setAttribute('aria-label', layers[layer].label+' atlas at 27% of the recorded stride, '+name+' view');
  }
  function usePoster() {
    if (disposed) return;
    stage?.dispose(); stage=undefined;
    stageHost.classList.remove('cover-ready');
    still=true; phase=.27; playing=false;
    playButton.disabled=true; phaseInput.disabled=true;
    showStill(); updatePlay(); updatePose(); sync();
  }
  for (const button of layerButtons) button.addEventListener('click', () => {
    layer = button.dataset.layer;
    for (const option of layerButtons) option.setAttribute('aria-pressed', String(option === button));
    detail.textContent=layers[layer].detail;
    stage?.setLayer(layer);
    if (still) showStill();
  }, options);
  rotateButton.addEventListener('click', () => {
    view=(view+1)%views.length;
    const name=views[view];
    angleNote.textContent=name[0].toUpperCase()+name.slice(1)+' view';
    stage?.setView(name);
    if (still) showStill();
  }, options);
  playButton.addEventListener('click', () => { playing=!playing; updatePlay(); sync(); }, options);
  phaseInput.addEventListener('input', () => {
    playing=false; phase=Number(phaseInput.value)/100; updatePlay(); updatePose(); sync();
  }, options);
  document.addEventListener('visibilitychange',sync,options);
  const modeObserver = new MutationObserver(() => {
    if (stage?.mode === 'landmarks') usePoster();
  });
  modeObserver.observe(stageHost, {attributes: true, attributeFilter: ['data-body-renderer']});
  detail.textContent=layers[layer].detail; updatePlay(); updatePose();
  import('./body-stage.js').then(async ({createBodyStage}) => {
    if (disposed) return;
    const result=await createBodyStage(stageHost,{comparison:false,view:views[view],layer,region:'body'});
    if (disposed) { result.dispose(); return; }
    stage=result;
    if (stage.mode === 'landmarks') {usePoster(); return;}
    stage.setLayer(layer); stage.setView(views[view]);
    stageHost.classList.add('cover-ready');
    updatePose(); sync();
  }).catch(usePoster);
  return {
    setActive(value) { active=Boolean(value); sync(); },
    dispose() { disposed=true; events.abort(); modeObserver.disconnect(); if (raf) cancelAnimationFrame(raf); stage?.dispose(); },
  };
}
