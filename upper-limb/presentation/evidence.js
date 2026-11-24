import './evidence.css';
import report from '../../knee-loading/data/report.json';
import gait from '../../knee-loading/data/gait.json';
import peaks from '../../knee-loading/data/stance-reference-audit.json';
import observation from '../../hip-study/data/observation-ablation.json';
import perturbations from '../../hip-study/data/counterfactual.json';
import symptoms from '../../hip-study/data/gait-pain-check.json';

const blue = '#365e79';
const terracotta = '#9e5138';
const phaseTicks = [0, 25, 50, 75, 100];
const web = { knee: gait.source.url, hip: perturbations.source, pain: symptoms.sources.paper };
const conditionNames = {normal:'Ordinary', bouncy:'Bouncy', forefoot_strike:'Forefoot strike', fast_treadmill:'Fast treadmill'};
const gaitPhrases = {bouncy:'With a bouncy gait',forefoot_strike:'With a forefoot strike',fast_treadmill:'On the fast treadmill'};
const magnitude = (n, digits = 2) => n.toFixed(digits).replace('-', '−');
const signed = (n, digits = 2) => (n > 0 ? '+' : '') + magnitude(n, digits);
const percent = n => (100 * n).toFixed(1) + '%';
const link = (href, label) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`;
const source = (...parts) => `<div class="ev-credit">${parts.join(' · ')}</div>`;

function choices(key, items, current, label) {
  return `<div class="ev-choices" role="group" aria-label="${label}">${items.map(([value, text]) =>
    `<button type="button" data-choice="${key}" data-value="${value}" aria-pressed="${value === current}">${text}</button>`).join('')}</div>`;
}
function select(key, label, items, current) {
  return `<label class="ev-select"><span>${label}</span><select data-choice="${key}" aria-label="${label}">${items.map(([key, label]) =>
    `<option value="${key}" ${current === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>`;
}
function metric(label, value, unit='', color='') {
  return `<div class="ev-result ${color}"><div>${label}</div><strong>${value}</strong>${unit ? `<span>${unit}</span>` : ''}</div>`;
}
function legend(items) {
  return `<div class="ev-legend">${items.map(([label, variant]) =>
    `<span><i class="${variant || ''}"></i>${label}</span>`).join('')}</div>`;
}
function title(s, tag) {
  return `<div class="ev-plot-heading"><h3>${s}</h3><span>${tag}</span></div>`;
}
function split(plot, detail) {
  return `<div class="ev-body"><div class="ev-figure">${plot}</div><div class="ev-side">${detail}</div></div>`;
}
function rail(inputs, center, output) {
  return `<div class="ev-rail"><span>${inputs}</span><b aria-hidden="true">→</b><span>${center}</span><b aria-hidden="true">→</b><span>${output}</span></div>`;
}
const svgText = (x,y,txt,anchor='middle',cls='') => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${txt}</text>`;
function waveform({phase, series, axis, min, max, step, cursor, window, area=false, phaseLabel='Stance phase (%)', mobile=false}) {
  const w = mobile ? 460 : 710, h = mobile ? 215 : 247, left=43, right=w-18, top=23, bottom=h-39;
  const x = v => left+(v/100)*(right-left), y=v=>bottom-(v-min)/(max-min)*(bottom-top);
  const svg = [];
  if(window) svg.push(`<rect x="${x(window[0])}" y="${top}" width="${x(window[1])-x(window[0])}" height="${bottom-top}" fill="${blue}" opacity=".09"/>`);
  for(let v=min;v<=max+step/100;v+=step) {
    svg.push(`<line class="ev-guide" x1="${left}" y1="${y(v)}" x2="${right}" y2="${y(v)}"/>`,svgText(left-8,y(v)+4,Number(v.toFixed(2)),'end'));
  }
  phaseTicks.forEach(v=>svg.push(svgText(x(v),bottom+18,v)));
  svg.push(svgText(left,13,axis,'start','ev-axis'), svgText((right+left)/2,h-3,phaseLabel,'middle','ev-axis'));
  series.forEach(({values,color,dash}) => {
    const points = phase.map((v,i)=>`${x(v).toFixed(2)},${y(values[i]).toFixed(2)}`);
    if(area) svg.push(`<polygon points="${left},${bottom} ${points.join(' ')} ${right},${bottom}" fill="${color}" opacity=".18"/>`);
    svg.push(`<polyline points="${points.join(' ')}" stroke="${color}" stroke-width="2.8" ${dash ? 'stroke-dasharray="7 4"' : ''} fill="none" stroke-linecap="round" stroke-linejoin="round"/>`);
  });
  if(cursor != null) {
    const i=phase.reduce((best,p,j)=>Math.abs(p-cursor)<Math.abs(phase[best]-cursor)?j:best,0);
    svg.push(`<line stroke="#73766e" stroke-dasharray="2 4" x1="${x(phase[i])}" x2="${x(phase[i])}" y1="${top}" y2="${bottom}"/>`);
    for(const s of series) svg.push(`<circle cx="${x(phase[i])}" cy="${y(s.values[i])}" r="4.6" fill="${s.color}" stroke="#fbf9f4" stroke-width="1.7"/>`);
  }
  return `<svg class="ev-wave" viewBox="0 0 ${w} ${h}" role="img" aria-label="${axis} over ${phaseLabel}">${svg.join('')}</svg>`;
}
function peakPlot(selected, mobile) {
  const keys = Object.keys(conditionNames), w=mobile?460:710, h=226, left=mobile?105:125, right=w-20;
  const x=v=>left+(v-1.30)/0.4*(right-left), y=i=>31+i*37;
  const reference=peaks.stances.find(s=>s.id===selected);
  const markup = [];
  for(let i=0; i<=4; i++){
    const v=1.3+i*.1;
    markup.push(`<line class="ev-guide" x1="${x(v)}" x2="${x(v)}" y1="19" y2="166"/>`,svgText(x(v),186,v.toFixed(1)));
  }
  keys.forEach((k,i)=>markup.push(svgText(left-10,y(i)+4,conditionNames[k],'end','ev-group')));
  markup.push(`<line stroke="${blue}" stroke-width="1.8" stroke-dasharray="4 4" x1="${x(reference.earlyMedialPeakBw)}" x2="${x(reference.earlyMedialPeakBw)}" y1="14" y2="166"/>`);
  peaks.stances.forEach(s=>markup.push(`<circle cx="${x(s.earlyMedialPeakBw)}" cy="${y(keys.indexOf(s.condition))}" r="${s.id===selected?7:5.6}" fill="${s.id===selected?blue:terracotta}" stroke="#fbf9f4" stroke-width="2"><title>${conditionNames[s.condition]}, stance ${s.id.split('-').at(-1)}: ${magnitude(s.earlyMedialPeakBw,3)} BW</title></circle>`));
  markup.push(svgText((left+right)/2,213,'Peak medial knee force (body weights)','middle','ev-axis'));
  return `<svg class="ev-wave" viewBox="0 0 ${w} ${h}" role="img" aria-label="Seven measured early medial knee force peaks in one recipient">${markup.join('')}</svg>`;
}
const sourceLabel = 'Grand Challenge gait data';
const normalStances = peaks.stances.filter(s=>s.condition==='normal');
const kneeStudies = report.study.comparisons.filter(c=>c.groupId==='1-GC4-JW');
function kneePanel(s) {
  const top = choices('kneeMode',[['curves','Force through stance'],['peaks','Compare seven stances']],s.kneeMode,'Knee chart');
  if(s.kneeMode==='peaks'){
    const ordinary = normalStances[Number(s.ref)];
    const sidebar = `<div class="ev-side-label">Peak change from selected ordinary stance</div><div class="ev-changes">${peaks.comparisons.map(c=>{
      const val=c.byReference.find(r=>r.ordinaryStance===ordinary.id).meanDeltaBw;
      return `<div><span>${conditionNames[c.condition]}</span><b>${signed(val,3)} <small>BW</small></b></div>`;
    }).join('')}</div><p>The two ordinary peaks differ by <b>${magnitude(peaks.ordinaryPeakSpreadBw,3)} BW</b> in a single pass. Switching reference reverses the bouncy and forefoot comparisons.</p>`;
    return top+split(title('Seven implant measurements','One recipient')+
      choices('ref',[['0','Ordinary stance 1'],['1','Ordinary stance 2']],s.ref,'Ordinary reference')+
      peakPlot(ordinary.id,s.mobile)+legend([['One early peak per stance','dot'],['Selected ordinary reference','reference']])+
      `<p class="ev-caption">Each peak falls between 5% and 45% of stance. Fast treadmill walking changed speed and surface together.</p>`,sidebar)+
      source(link(web.knee,sourceLabel));
  }
  const study=kneeStudies.find(c=>c.id===s.gait), group=study.intervention, trace=group.compartments.medial, i=Number(s.phase);
  const early = study.metrics.find(m=>m.compartment==='medial'&&m.label==='Early-stance peak');
  const condition=study.id.replace(study.groupId+'-', '');
  const measuredInputs=gait.trials.filter(trial=>trial.role==='evaluation' && trial.condition===condition);
  const meanInput=key=>measuredInputs.reduce((sum,trial)=>sum+trial.frames[i][key],0)/measuredInputs.length;
  const chart = title('Inside the medial compartment',group.trialCount===1?'1 held-out stance':`${group.trialCount} held-out stances · mean`)+
    select('gait','Gait',kneeStudies.map(c=>[c.id,c.label]),s.gait)+
    waveform({phase:group.phase,series:[{values:trace.modeled,color:terracotta,dash:true},{values:trace.measured,color:blue}],axis:'Force (body weights)',min:0,max:Math.max(2,Math.ceil(Math.max(...trace.measured,...trace.modeled)*2)/2),step:.5,cursor:i,window:[5,45],mobile:s.mobile})+
    legend([['Measured by implant',''],['Estimated by mechanics','dash']])+
    `<label class="ev-slider"><span>Stance phase</span><input type="range" min="0" max="100" value="${i}" data-range="phase"><output>${i}%</output></label>`;
  return top+split(chart, `<div class="ev-side-label">At ${i}% of stance</div><p class="ev-inputs">Knee flexion <b>${magnitude(meanInput('kneeFlexionDeg'),1)}°</b> · Adduction moment <b>${magnitude(meanInput('kamNmKg'),2)} N·m/kg</b></p><div class="ev-twins">${metric('Implant',magnitude(trace.measured[i]),'BW','blue')}${metric('Model',magnitude(trace.modeled[i]),'BW','terracotta')}</div>
    <p>${gaitPhrases[condition]}, the implant recorded a ${early.measured<0?'lower':'higher'} early peak than in ordinary walking (<b>${signed(early.measured,3)} BW</b>). The model predicted a ${early.model<0?'lower':'higher'} peak (<b>${signed(early.model,3)} BW</b>).</p>
    <p class="ev-muted">${condition==='fast_treadmill'?'Both peaks rose for bouncy and forefoot walking.':'On the fast treadmill, the implant’s peak fell while the model’s rose.'} A separate ordinary stance calibrated the model.</p>`)+
    rail('Motion + external loading','Muscle and joint balance','Compartment force')+source(link(web.knee,sourceLabel),link('../knee-loading/#forces','Full force study'));
}
function challengePanel(s) {
  const {pair}=perturbations, t=s.task==='walk'?'walk':'challenge';
  const signal=s.signal, traces=pair.cases.map(c=>c[t].frames);
  const a=observation.originalCohort.effort[s.task], r=observation.additionalCohorts.summary[s.task];
  const signals={trunkLeanDeg:['Trunk lean',-2,4,1],hipFlexionDeg:['Hip flexion',-20,40,10],hipAdductionDeg:['Hip adduction',-10,15,5]};
  const [name, min, max, step]=signals[signal];
  const chart=title(name,'Simulated angle')+
    select('signal','Signal',Object.entries(signals).map(([key,[label]])=>[key,label]),s.signal)+
    waveform({phase:traces[0].map(row=>100*row.phase),series:traces.map((rows,i)=>({values:rows.map(row=>row[signal]),color:i?terracotta:blue,dash:!!i})),axis:'Angle (degrees)',min,max,step,phaseLabel:s.task==='walk'?'Walking stance (%)':'Loaded stance task (%)',mobile:s.mobile})+
    legend([['Reduced abductor capacity',''],['Restricted hip extension','dash']])+
    `<p class="ev-caption">One virtual body under two impairments. This pair had the largest loaded-stance separation among walking matches.</p>`;
  const b=a.balancedAccuracy;
  const sidebar=`<div class="ev-side-label">Identification accuracy · 256 unseen virtual bodies</div>
   <div class="ev-twins">${metric('Hip angles',percent(b.hipAngles),'','blue')}${metric('Hip + trunk',percent(b.hipAnglesAndTrunkLean),'','terracotta')}</div>
   <p>In 20 additional virtual groups, observing the trunk improved accuracy by an average of <b>${magnitude(100*r.meanDifference,1)} percentage points</b>.</p>
   <p class="ev-muted">The simulation lets the trunk lean in response to reduced hip strength. Angle observations include simulated camera error. Chance is 50%.</p>`;
  return choices('task',[['walk','Walking'],['loadedStance','Loaded single-leg stance']],s.task,'Movement task')+split(chart,sidebar)+
    source(link(web.hip,'Source gait template'), link('../hip-study/#mechanisms','Hip study'));
}
const ordinaryTrial=gait.trials.find(t=>t.id===normalStances[1].id);
const pulse=ordinaryTrial.frames.map(f=>f.medialBw);
const timing=ordinaryTrial.frames.map(f=>f.stancePct);
function oneStanceImpulse() {
  let integral=0;
  for(let i=1;i<pulse.length;i++)integral+=(pulse[i]+pulse[i-1])*.5*(timing[i]-timing[i-1])/100;
  return integral*ordinaryTrial.stanceSeconds;
}
const stanceImpulse=oneStanceImpulse();
function painPanel(s) {
  const top=choices('symptomMode',[['pain','Pain during the visit'],['exposure','Repeated load']],s.symptomMode,'Choose evidence');
  if(s.symptomMode==='exposure'){
    const cycles=Number(s.steps), total=cycles*stanceImpulse;
    const bar=waveform({phase:timing,series:[{values:pulse,color:blue}],axis:'Medial force (body weights)',min:0,max:2,step:.5,area:true,mobile:s.mobile});
    return top+split(title('One recorded foot contact','Implant force')+bar+
      `<p class="ev-caption">The shaded area, integrated over ${magnitude(ordinaryTrial.stanceSeconds,2)} seconds, gives <b>${magnitude(stanceImpulse,2)} BW·s</b> of medial knee-force impulse.</p>`,
      `<div class="ev-side-label">Illustrative repetition · same measured stance</div>${metric('Accumulated impulse',Math.round(total).toLocaleString('en-US'),'BW·s','blue')}
      <label class="ev-slider ev-stack"><span>Contacts on this leg</span><input type="range" min="100" max="5000" step="100" value="${cycles}" data-range="steps"><output>${cycles.toLocaleString('en-US')}</output></label>
      <p>Repeating the same force record estimates cumulative exposure. A pre-symptom study would pair repeated gait and pain records with later MRI scans.</p>
      <p class="ev-muted">In 144 people with knee osteoarthritis, a separate gait measure, knee adduction-moment impulse, was associated with medial tibial cartilage loss over 12 months.</p>`)+source(link(web.knee,sourceLabel), link('https://pubmed.ncbi.nlm.nih.gov/21742637/','12-month gait and MRI study'));
  }
  const p=symptoms.baselineWalking;
  const entries=[['Group mean','meanOnly'],['Hip excursion','hip'],['Hip + trunk excursion','hipAndTrunk']];
  const chart=title('Error in estimated hip pain','0–10 pain scale')+
    `<div class="ev-bars">${entries.map(([name,k])=>{
      const val=p.leaveOnePersonOut[k].maePainPoints; return `<div><span>${name}</span><div class="ev-bar-track"><i style="width:${100*val/2.5}%"></i></div><b>${magnitude(val,2)}</b></div>`;
    }).join('')}<div class="ev-bar-axis"><span></span><div><span>0</span><span>1.25</span><span>2.50</span></div><span></span></div></div>
    <div class="ev-caption">Mean absolute error in pain points. Lower error is better.</div>`;
  return top+split(chart,`<div class="ev-side-label">Measured in people</div>${metric('Women with outer-hip pain',p.people,'','blue')}
    <p>The group average was closer than either walking estimate. We predicted each woman’s pain using the other women’s records, then compared it with her reported score.</p>
    <p class="ev-muted">A separate trial of 68 people with knee osteoarthritis found 1.2 points greater pain reduction at one year with personalized foot-angle retraining than with sham retraining.</p>`)+
    source(link(web.pain,'Robinson et al. · clinical study'), link(symptoms.sources.dataset,'Gait and pain records'), link('https://pubmed.ncbi.nlm.nih.gov/40816302/','Knee gait-retraining trial'));
}

export function mountEvidence(root, id) {
  const views={knee:kneePanel,symptoms:painPanel,challenge:challengePanel};
  const content=views[id];
  if (!content) throw new Error('Unknown evidence panel: '+id);
  const host=document.createElement('div');
  host.className='ev-panel';
  root.replaceChildren(host);
  const media=window.matchMedia('(max-width: 650px)');
  const state={kneeMode:'curves',ref:'1',gait:'1-GC4-JW-fast_treadmill',phase:'28',task:'loadedStance',signal:'trunkLeanDeg',symptomMode:'pain',steps:'1500',mobile:media.matches};
  function render() { host.innerHTML=content(state); }
  function change(event){
    const el=event.target.closest('[data-choice]');
    if(!el || (event.type==='click') !== (el.tagName==='BUTTON'))return;
    const key=el.dataset.choice, value=el.tagName==='BUTTON'?el.dataset.value:el.value;
    state[key]=value; render();
    const target=el.tagName==='BUTTON' ? `[data-choice="${key}"][data-value="${value}"]` : `select[data-choice="${key}"]`;
    host.querySelector(target)?.focus({preventScroll:true});
  }
  function slide(event){
    const input=event.target, key=input.dataset.range;
    if(!key)return;
    state[key]=input.value;
    // Keep the active slider in place through a drag, replacing only its results.
    const temp=document.createElement('div');
    temp.innerHTML=content(state);
    for(const sel of ['.ev-result','.ev-side-label','.ev-inputs','.ev-wave','output']) {
      const old=host.querySelectorAll(sel), fresh=temp.querySelectorAll(sel);
      old.forEach((node,i)=>node.replaceWith(fresh[i]));
    }
  }
  function adapt(){state.mobile=media.matches;render();}
  host.addEventListener('click',change);
  host.addEventListener('change',change);
  host.addEventListener('input',slide);
  media.addEventListener('change',adapt);
  render();
  return { dispose(){media.removeEventListener('change',adapt);host.remove();} };
}
