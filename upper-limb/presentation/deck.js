const sections = [
  { id: 'start', name: 'The walking body', load: () => import('./cover.js').then(m => m.mountCover) },
  { id: 'whole-body', name: 'The force underfoot', title: 'What does the rest of the body add?', intro: 'One participant walked over force plates. Compare the measured force with estimates from pelvis motion alone and from all 24 body segments.', load: () => import('./wholebody.js').then(m => m.mountWholeBody) },
  { id: 'hip-impact', name: 'Outer-hip pain', title: 'When outer-hip pain limits walking', intro: 'Greater trochanteric pain syndrome causes pain and tenderness on the outside of the hip, where the gluteal tendons attach to the thigh bone.', load: conditions },
  { id: 'anatomy', name: 'Four views of the hip', title: 'One step, four anatomical views', intro: 'Follow a different walker through the same instant in four views. The recording drives the motion. Anatomical meshes show where the tissues sit.', load: () => import('./anatomy.js').then(m => m.mountAnatomy) },
  { id: 'soft-tissue-model', name: 'Soft-tissue mechanics', title: 'Soft-tissue mechanics', intro: 'The model couples a moving skeleton to deformable tissues that contract, stretch and slide against one another.', load: () => import('./modeling.js').then(m => m.mountSoftTissueModel) },
  { id: 'gpu-body-model', name: 'GPU whole-body model', title: 'GPU-accelerated whole-body simulation', intro: 'A high-resolution musculoskeletal model links whole-body motion to local tissue strain and contact pressure.', load: () => import('./modeling.js').then(m => m.mountGpuBodyModel) },
  { id: 'deterioration', name: 'Change a walker', title: 'How does a weaker hip change a stride?', intro: 'Compare the same virtual walker before and after a change to the right hip. Step timing and external loads stay fixed as the body adjusts.', load: () => import('./intervention.js').then(m => m.mountIntervention) },
  { id: 'tissue', name: 'Inside the outer hip', title: 'How far does the band slide?', intro: 'Bring the thigh toward the midline. Track the same points on the iliotibial band over the fluid-filled bursa at the outside of the hip.', load: () => import('./tissue.js').then(m => m.mountTissue) },
  { id: 'challenge', name: 'Choosing a movement', title: 'Which movement reveals a weak hip?', intro: 'In a separate simulation, a weaker hip and a stiffer hip can look similar during walking. Compare the hip and trunk as the virtual person stands on one leg.', load: evidence },
  { id: 'hip-oa', name: 'Hip osteoarthritis', title: 'How common is hip osteoarthritis?', intro: 'The English Longitudinal Study of Ageing provides a symptom-based estimate for adults aged 45 and over.', load: conditions },
  { id: 'knee-impact', name: 'Knee osteoarthritis', title: 'Knee osteoarthritis in middle and later life', intro: 'An England-wide survey estimated the burden of knee symptoms. A separate London study followed changes visible on X-rays in the same women over time.', load: conditions },
  { id: 'knee', name: 'Gait and knee load', title: 'Does changing gait unload the knee?', intro: 'An implant measured force on the inner side of one person’s knee. Compare its readings with the model during ordinary and modified walking.', load: evidence },
  { id: 'symptoms', name: 'Load and symptoms', title: 'Can walking angles predict hip pain?', intro: 'Forty-eight women with outer-hip pain rated their symptoms. We held out each woman in turn and estimated her rating from her walking angles.', load: evidence },
  { id: 'correctness', previousId: 'judging-correctness', name: 'Motion data and model checks', title: 'How we check the whole-body model', intro: 'Recorded motion drives the whole-body model. Force plates measure the push underfoot. Implants measure forces inside joints.', load: () => import('./judging.js').then(m => m.mountCorrectness) },
  { id: 'reshaping', previousId: 'judging-reshaping', name: 'Parallel force calculations', title: 'How the GPU calculates forces during a step', intro: 'As the foot lands, the GPU calculates forces from ground contact and muscle stretch in parallel.', load: () => import('./judging.js').then(m => m.mountReshaping) },
  { id: 'science', previousId: 'judging-science', name: 'Whole-body response', title: 'A hip muscle affects the back and shoulder', load: () => import('./judging.js').then(m => m.mountScience) },
  { id: 'compute', name: 'GPU vs CPU', title: 'How long does a hip model search take?', intro: 'We varied the assumed hip joint centre and how much each participant counted. For each combination, we searched for the model that best matched the measured hip forces.', load: () => import('./compute.js').then(m => m.mountCompute) },
  { id: 'follow-up', name: 'Following a person over time', title: 'Does movement change before pain begins?', intro: 'Answering that question takes repeat measurements in the same people. Choose the hip or knee to see the proposed study.', load: () => import('./followup.js').then(m => m.mountFollowup) },
];

async function conditions() {
  return (await import('./conditions.js')).mountCondition;
}

async function evidence() {
  const mod = await import('./evidence.js');
  return mod.mountEvidence;
}

const el = id => document.getElementById(id);
const dialog = el('index-dialog');
const stage = el('scene');
let activeIndex = -1;
let revision = 0;
let mounted;

function targetIndex() {
  const fragment = window.location.hash.slice(1);
  const index = sections.findIndex(item => item.id === fragment || item.previousId === fragment);
  if (index < 0) return 0;
  const { id } = sections[index];
  if (id !== fragment) window.history.replaceState(window.history.state, '', '#' + id);
  return index;
}

function link(item, index, menu) {
  const a = document.createElement('a');
  a.href = '#' + item.id;
  a.title = item.name;
  a.setAttribute('aria-label', (index + 1) + '. ' + item.name);
  if (menu) {
    const num = document.createElement('span');
    num.textContent = String(index + 1).padStart(2, '0');
    a.append(num, item.name);
    a.addEventListener('click', () => dialog.close());
  }
  return a;
}

sections.forEach((item, index) => {
  el('index-list').append(link(item, index, true));
  el('progress').append(link(item, index, false));
});

function dispose(view) {
  if (typeof view === 'function') view();
  else view?.dispose?.();
}

async function display(force = false) {
  const index = targetIndex();
  if (!force && index === activeIndex) return;
  const token = ++revision;
  const item = sections[index];
  const wasMounted = activeIndex >= 0;
  activeIndex = index;
  dispose(mounted);
  mounted = null;
  el('deck').dataset.scene = item.id;
  el('heading').hidden = !item.title;
  el('headline').textContent = item.title ?? '';
  el('intro').textContent = item.intro ?? '';
  el('position').textContent = String(index+1).padStart(2, '0') + ' / ' + String(sections.length).padStart(2, '0');
  document.title = item.name + ' · Human Simulator';
  el('previous').disabled = index === 0;
  el('next-name').textContent = sections[index+1]?.name ?? 'Back to the body';
  for (const nav of [el('index-list'), el('progress')]) {
    [...nav.children].forEach((a,i) => i === index ? a.setAttribute('aria-current','step') : a.removeAttribute('aria-current'));
  }
  const slot = document.createElement('div');
  slot.className = 'view-slot';
  slot.setAttribute('aria-busy', 'true');
  stage.setAttribute('aria-label', item.name);
  stage.replaceChildren(slot);
  el('deck-main').scrollTop = 0;
  if (wasMounted) el('deck-main').focus({preventScroll:true});
  try {
    const mount = await item.load();
    if (token !== revision) return;
    const view = await mount(slot, item.id);
    if (token !== revision) dispose(view);
    else { mounted = view; mounted?.setActive?.(!document.hidden); }
  } catch (err) {
    if (token !== revision) return;
    console.error('Chapter could not load', item.id, err);
    const msg = document.createElement('div');
    msg.className = 'panel-status';
    msg.textContent = 'This chapter could not load. ';
    const retry = document.createElement('button');
    retry.textContent = 'Try again';
    retry.onclick = () => display(true);
    msg.append(retry);
    slot.replaceChildren(msg);
  } finally {
    slot.removeAttribute('aria-busy');
  }
}

const go = i => { location.hash = sections[Math.max(0,Math.min(i,sections.length - 1))].id; };
el('previous').onclick = () => go(activeIndex - 1);
el('next').onclick = () => go((activeIndex + 1) % sections.length);
el('chapters').onclick = () => { dialog.showModal(); dialog.querySelector('[aria-current=step]')?.focus(); };
el('close-index').onclick = () => dialog.close();
dialog.addEventListener('click', e => {if (e.target === dialog && (e.offsetX < 0 || e.offsetY < 0 || e.offsetX > dialog.clientWidth || e.offsetY > dialog.clientHeight)) dialog.close();});
const fullButton = el('screen');
const isFull = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
if (!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen)) fullButton.hidden = true;
fullButton.onclick = async () => {
  try {
    if (isFull()) await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    else await (document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen).call(document.documentElement);
  } catch { fullButton.title = 'Full screen is unavailable in this window'; }
};
function updateFullscreen() {
  const label = isFull() ? 'Exit full screen' : 'Enter full screen';
  fullButton.title = label; fullButton.setAttribute('aria-label',label);fullButton.setAttribute('aria-pressed',String(isFull()));
}
document.addEventListener('fullscreenchange',updateFullscreen);
document.addEventListener('webkitfullscreenchange',updateFullscreen);
window.addEventListener('hashchange', () => display());
document.addEventListener('visibilitychange', () => mounted?.setActive?.(!document.hidden));
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || event.altKey || event.metaKey || event.ctrlKey || dialog.open || event.target.closest('input,select,textarea,canvas,summary,[role=slider],[contenteditable=true]')) return;
  if (event.key === 'ArrowRight' || event.key === 'PageDown') {event.preventDefault(); go(activeIndex + 1);}
  if (event.key === 'ArrowLeft' || event.key === 'PageUp') {event.preventDefault(); go(activeIndex - 1);}
  if (event.key === 'Home' || event.key === 'End') {event.preventDefault(); go(event.key === 'Home' ? 0 : sections.length - 1);}
});
window.addEventListener('pagehide', e => { if (!e.persisted) dispose(mounted); });
display();
