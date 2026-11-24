import './wholebody.css';

const strideUrl = new URL('../../whole-body-study/stride.json', import.meta.url).href;
const motionUrl = new URL('../../knee-loading/data/whole-body-motion.json', import.meta.url).href;
const ordinaryClipId = 'ngait_og5-stance-1';
const svgNS = 'http://www.w3.org/2000/svg';
const chartBox = { left: 47, right: 470, top: 14, bottom: 125 };

function svg(tag, attrs, content) {
  const node = document.createElementNS(svgNS, tag);
  for (const [key,value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (content !== undefined) node.textContent = content;
  return node;
}

async function read(url, signal) {
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}

function phaseIndex(phases, phase) {
  let lo = 0, hi = phases.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (phases[mid] < phase) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(phases[lo-1] - phase) < Math.abs(phases[lo] - phase)) return lo-1;
  return lo;
}

function massMarkersAt(markers, phase) {
  if(!markers || phase<markers.phasePct[0] || phase>markers.phasePct.at(-1))return;
  const phases=markers.phasePct;
  const nearest=phaseIndex(phases,phase);
  const lo=phases[nearest]>phase?nearest-1:nearest;
  const hi=phases[nearest]<phase?nearest+1:nearest;
  const blend=lo===hi?0:(phase-phases[lo])/(phases[hi]-phases[lo]);
  const sample=values=>values[lo].map((value,axis)=>value+blend*(values[hi][axis]-value));
  return {
    body:sample(markers.centerOfMassGroundM),
    pelvis:sample(markers.pelvisOriginGroundM),
    clipOriginGroundM:markers.clipOriginGroundM,
  };
}

function fallbackSkeleton(motion, host) {
  // The outline follows the identical segment transforms and local joint centers as the 3D atlas.
  const coords = Object.fromEntries(motion.segmentOrder.map((key,i)=>[key,7*i]));
  const rotate = ([x,y,z],[qx,qy,qz,qw])=>{
    const tx=2*(qy*z-qz*y),ty=2*(qz*x-qx*z),tz=2*(qx*y-qy*x);
    return [x+qw*tx+qy*tz-qz*ty,y+qw*ty+qz*tx-qx*tz,z+qw*tz+qx*ty-qy*tx];
  };
  const landmarks = {...motion.landmarks, head:{segment:'torso',offsetM:[0.015,0.56,0]},
    rightToe:{segment:'toes',offsetM:[0.06,0,0]},leftToe:{segment:'toes_l',offsetM:[0.06,0,0]}};
  const draw = svg('svg', {viewBox:'0 0 300 330',role:'img','aria-label':'Recorded joint centres, projected from the whole-body pose'});
  const chains = [
    ['leftShoulder','leftElbow','leftWrist'],['leftHip','leftKnee','leftAnkle','leftToe'],
    ['leftShoulder','rightShoulder'],['leftHip','rightHip'],['head','lumbar'],
    ['leftShoulder','lumbar','leftHip'],['rightShoulder','lumbar','rightHip'],
    ['rightShoulder','rightElbow','rightWrist'],['rightHip','rightKnee','rightAnkle','rightToe']
  ];
  const strokes = chains.map((names,i)=>{
    const line = svg('polyline',{fill:'none',stroke:i<2?'#81979d':i>6?'#d5b298':'#c5d8d2','stroke-width':i<2?4:5,'stroke-linejoin':'round','stroke-linecap':'round'});
    draw.append(line); return {names,line};
  });
  const head = svg('circle',{r:10,fill:'#d1dad4'}); draw.append(head);host.append(draw);
  return function(clip, phase, fixedArms) {
    const frame = clip.frames[Math.round(phase*(clip.frames.length-1))];
    const out = {};
    for(const [key, landmark] of Object.entries(landmarks)) {
      let segment=landmark.segment, local=landmark.offsetM;
      const fixed=fixedArms?.[segment];
      if(fixed){
        local=rotate(local,fixed.slice(3)).map((v,i)=>v+fixed[i]);
        segment='torso';
      }
      const o=coords[segment];
      const offset=rotate(local,frame.slice(o+3,o+7));
      const X=frame[o]-frame[coords.pelvis]+offset[0];
      const Y=frame[o+1]+offset[1];
      const Z=frame[o+2]-frame[coords.pelvis+2]+offset[2];
      out[key]=[150+145*(0.77*X+0.64*Z),307-151*Y];
    }
    for(const {names,line} of strokes)line.setAttribute('points',names.map(n=>out[n].join(',')).join(' '));
    head.setAttribute('cx',out.head[0]);head.setAttribute('cy',out.head[1]);
  };
}

export function mountWholeBody(root) {
  root.innerHTML = `<div class="wb-layout">
    <div class="wb-left"><div class="wb-plate">
      <div class="wb-fallback" aria-hidden="true"></div><div class="wb-model" aria-label="Whole-body walking anatomy"></div>
      <div class="wb-origin"><span></span><b>GC4-JW</b></div><div class="wb-event"></div>
    </div><div class="wb-transport"><button type="button" class="wb-play" aria-label="Play stride" aria-pressed="false">Play</button><label>Stride <output class="wb-phase">25%</output><input aria-label="Stride phase" class="wb-scrub" type="range" min="0" max="100" step="1" value="25"></label></div></div>
    <div class="wb-analysis"><div class="wb-fields"><label>Calculation<select class="wb-mode"><option value="moments">Arm swing and joint moments</option><option value="muscles">Muscle path lengths</option></select></label><label><span class="wb-coordinate-label">Structure</span><select class="wb-coordinate" aria-label="Structure"></select></label></div>
      <div class="wb-scenario" role="group" aria-label="Arm movement"><button type="button" class="wb-free" aria-pressed="true">Recorded swing</button><button type="button" class="wb-fixed" aria-pressed="false">Hold arms fixed</button></div>
      <div class="wb-readout"><span class="wb-signal-label">Calculated moment</span><div class="wb-value"><strong class="wb-reading">…</strong><span class="wb-unit">N·m</span><span class="wb-observed" hidden></span></div></div>
      <svg class="wb-chart" viewBox="0 0 480 158" role="img" aria-label="Signal through one complete stride"></svg>
      <div class="wb-legend"><span class="wb-ours">Calculated</span><span class="wb-theirs">Fixed arms</span><label class="wb-proxy-choice" hidden><input type="checkbox" class="wb-proxy" checked>Pelvis-height estimate</label></div>
      <p class="wb-caption" aria-live="polite">Loading the motion and force records…</p>
      <div class="wb-evidence" hidden><table><caption>RMS vertical force error in three strides from one participant</caption><thead><tr><th scope="col">Walk</th><th scope="col">Pelvis height</th><th scope="col">Full body</th></tr></thead><tbody></tbody></table><button class="wb-try-arms" type="button">Try holding the arms fixed <span aria-hidden="true">→</span></button></div>
      <div class="wb-meta"></div><details class="wb-source"><summary>Calculation and source</summary><div><span class="wb-method"></span> <span class="wb-error"></span> <a href="https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8" target="_blank" rel="noreferrer">Motion, force plates and model</a>. <a href="https://www.iri.upc.edu/files/scidoc/3107-Reducing-Complexity-in-Muscle-Tendon-Kinematics-Parameterization-Improves-Convergence-Speed-in-Musculoskeletal-Simulations.pdf" target="_blank" rel="noreferrer">Gait protocols</a>.</div></details>
    </div></div>`;
  const find = cls => root.querySelector('.wb-'+cls);
  const scrub=find('scrub'), play=find('play'), mode=find('mode'), coordinate=find('coordinate'), chart=find('chart');
  const abort = new AbortController();
  let study, motion, clip, stage, project, trace, plotPhases, transform, cursor, marker, selectedTrial;
  let momentPhase=56, fixed=false, disposed=false, active=true, playing=false, tickId=0, lastTime=0, phase=56;

  function dispose() {
    disposed=true; abort.abort(); cancelAnimationFrame(tickId); stage?.dispose();
  }
  function setPlaying(value) {
    playing=value;lastTime=0;
    play.textContent=playing?'Pause':'Play'; play.setAttribute('aria-label',playing?'Pause stride':'Play stride'); play.setAttribute('aria-pressed',String(playing));
    cancelAnimationFrame(tickId); if (playing && active) tickId=requestAnimationFrame(tick);
  }
  function tick(now) {
    if (!playing || !active || disposed) return;
    if (lastTime) {
      const min=Number(scrub.min), max=Number(scrub.max);
      phase=min+(phase-min+Math.min(now-lastTime,100)/((clip?.durationSeconds||1.1)*10))%(max-min);
    }
    lastTime=now; displayPhase(); tickId=requestAnimationFrame(tick);
  }
  function displayPhase() {
    const normalized=phase/100;
    const groundTrial=mode.value==='ground'?selectedTrial:undefined;
    const clipId=groundTrial?.clipId??ordinaryClipId;
    if(clip?.id!==clipId)clip=motion?.clips.find(c=>c.id===clipId);
    const walk=groundTrial?.condition??'Ordinary';
    find('origin').lastElementChild.textContent=`GC4-JW · ${walk.toLowerCase()} walk`;
    const fixedSegments=fixed?study?.fixedArmSegmentsInTorso:undefined;
    const massMarkers=mode.value==='ground' && study?massMarkersAt(groundCheck().positionMarkers,phase):undefined;
    if (stage) stage.setPose({phase:normalized,clipId,fixedArmSegmentsInTorso:fixedSegments,massMarkers});
    else if(clip)project?.(clip,normalized,fixedSegments);
    find('phase').value=Math.round(phase)+'%'; scrub.value=String(Math.round(phase));
    if (clip) find('event').textContent=phase<=clip.stanceEndPct?'Right foot in stance':'Right leg in swing';
    if (!study) return;
    if (mode.value==='forward') {drawResponse();return;}
    if (!trace || !transform) return;
    const i=phaseIndex(plotPhases,phase), v=trace[i];
    const valid=Number.isFinite(v);
    find('reading').textContent=valid?v.toFixed(1):'Unavailable';
    if(mode.value==='ground')find('observed').textContent=`Plate reading ${groundCheck()[coordinate.value].measuredN[i].toFixed(1)} N`;
    cursor.setAttribute('x1',transform.x(plotPhases[i]));cursor.setAttribute('x2',transform.x(plotPhases[i]));
    marker.setAttribute('cx',transform.x(plotPhases[i]));marker.setAttribute('cy',valid?transform.y(v):0);marker.style.display=valid?'':'none';
  }
  function groundCheck() {
    return selectedTrial?.forcePlateCheck??study.forcePlateCheck;
  }
  function drawResponse() {
    const samples=study.forwardResponse.samples;
    const sample=samples[phaseIndex(samples.map(s=>s.phasePct),phase)];
    const values=sample.deltaDegPerSecondSquared;
    const key=coordinate.value;
    const chosen=values[key];
    find('signal-label').textContent='Immediate acceleration change';
    find('reading').textContent=(chosen>0?'+':'')+chosen.toFixed(1);
    find('unit').textContent='°/s²';
    const channels=['hip_adduction','knee_flexion','ankle_angle','lumbar_bending','arm_flex_r'].filter(k=>Number.isFinite(values[k]));
    if (!channels.includes(key)) channels[channels.length-1]=key;
    const max=Math.max(1,...samples.flatMap(s=>channels.map(k=>Math.abs(s.deltaDegPerSecondSquared[k]))));
    const left=140, right=417, mid=(left+right)/2;
    chart.replaceChildren(svg('line',{x1:mid,x2:mid,y1:5,y2:135,class:'wb-rule'}));
    for(const [index,k] of channels.entries()){
      const val=values[k], x=mid+val/max*(right-left)*0.5, y=16+index*25;
      chart.append(svg('text',{x:1,y:y+4},study.moments[k]?.label??k),
        svg('line',{x1:mid,x2:x,y1:y,y2:y,stroke:k===key?'#9e5138':'#6f8da0','stroke-width':6}),
        svg('circle',{cx:x,cy:y,r:3.1,fill:k===key?'#9e5138':'#6f8da0'}),
        svg('text',{x:476,y:y+4,'text-anchor':'end'},(val>0?'+':'')+val.toFixed(1)));
    }
    chart.append(svg('text',{x:mid,y:151,'text-anchor':'middle'},'0 °/s²'));
    chart.setAttribute('aria-label','Calculated differences in angular acceleration after reducing right gluteus medius force.');
    find('caption').textContent='From rest in this recorded pose, reduce the right gluteus medius force by 50 N. The right heel pivots at the measured center of pressure. The left foot is free.';
    find('method').textContent='The forward-dynamics calculation starts from rest. A 50 N decrease is shared across three right gluteus medius paths. The foot can rotate around its ground contact.';
    find('error').textContent='The result is an instantaneous acceleration difference. Signs follow the OpenSim model’s coordinate conventions.';
  }
  function drawSignal() {
    const forward=mode.value==='forward', ground=mode.value==='ground', joint=mode.value==='moments';
    find('scenario').hidden=!joint;
    find('legend').hidden=forward;
    find('observed').hidden=!ground;
    find('evidence').hidden=!(ground && coordinate.value==='vertical' && study.forcePlateComparison?.length);
    if(forward){displayPhase();return;}
    const rec=(ground?groundCheck():study[mode.value])[coordinate.value];
    const recorded=ground?rec.measuredN:joint?rec.computedNm:rec.lengthMm;
    const comparison=ground?rec.computedN:joint?rec.fixedArmsNm:undefined;
    const hasProxy=ground&&Array.isArray(rec.pelvisOnlyN);
    find('proxy-choice').hidden=!hasProxy;
    const proxy=hasProxy&&find('proxy').checked?rec.pelvisOnlyN:undefined;
    plotPhases=ground?groundCheck().phasePct:study.phasePct;
    const first=plotPhases[0], last=plotPhases.at(-1);
    trace=ground||fixed?comparison:recorded;
    find('free').setAttribute('aria-pressed',String(!fixed));
    find('fixed').setAttribute('aria-pressed',String(fixed));
    const values=[...recorded,...(comparison??[]),...(proxy??[])].filter(Number.isFinite);
    const bottom=Math.min(...values),top=Math.max(...values);
    const margin=Math.max(1,(top-bottom)*.14);
    const min=bottom-margin, max=top+margin;
    const B=chartBox; const x=v=>B.left+(B.right-B.left)*(v-first)/(last-first), y=v=>B.bottom-(v-min)/(max-min)*(B.bottom-B.top);
    transform={x,y};chart.replaceChildren();
    const unit=ground?'N':joint?'N·m':'mm';find('unit').textContent=unit;find('signal-label').textContent=ground?'Calculated ground force':joint?(fixed?'Fixed-arm moment':'Recorded-swing moment'):'Calculated path length';
    chart.setAttribute('aria-label',`${coordinate.selectedOptions[0].textContent}, ${unit}, from ${first} to ${last} percent of the stride. Use the stride control to read each sample.`);
    const lines=[bottom,(top+bottom)/2,top];
    for(const val of lines){const vertical=y(val);chart.append(svg('line',{x1:B.left,x2:B.right,y1:vertical,y2:vertical,class:'wb-rule'}),svg('text',{x:B.left-8,y:vertical+4,'text-anchor':'end'},Math.abs(val)<10?val.toFixed(1):Math.round(val)));}
    for(let n=0;n<=4;n++){const val=first+n*(last-first)/4;chart.append(svg('text',{x:x(val),y:149,'text-anchor':val===last?'end':'middle'},val+'%'));}
    const line=(series,css)=>{
      let connected=false,path='';
      for(let i=0;i<series.length;i++){
        if(!Number.isFinite(series[i])){connected=false;continue;}
        path+=`${connected?'L':'M'}${x(plotPhases[i]).toFixed(2)} ${y(series[i]).toFixed(2)} `;connected=true;
      }
      return svg('path',{d:path,class:css,fill:'none'});
    };
    if(proxy)chart.append(line(proxy,'wb-pelvis'));
    chart.append(line(recorded,'wb-computed'));
    if(comparison)chart.append(line(comparison,'wb-published'));
    cursor=svg('line',{y1:B.top,y2:B.bottom,class:'wb-cursor'});marker=svg('circle',{r:4.8,class:fixed||ground?'wb-marker wb-isfixed':'wb-marker'});chart.append(cursor,marker);
    find('theirs').hidden=!comparison;
    find('ours').textContent=ground?'Force plates':joint?'Recorded swing':'Muscle–tendon length';
    find('theirs').textContent=ground?'Full-body estimate':'Fixed arms';
    let caption;
    if(ground) {
      const name=selectedTrial?.condition?.toLowerCase()??'ordinary';
      if(hasProxy) {
        const improvement=Math.round(100*(1-rec.rmseN/rec.pelvisOnlyRmseN));
        caption=`Using all ${study.model.bodyCount} body segments lowers error by ${improvement}% on the ${name} stride, from ${rec.pelvisOnlyRmseN.toFixed(1)} to ${rec.rmseN.toFixed(1)} N RMS. Both estimates cover ${first}–${last}% of the stride.`;
      } else {
        caption=`Across ${first}–${last}% of the ${name} stride, the full-body error is ${rec.rmseN.toFixed(1)} N RMS (${rec.rmsePctParticipantBodyWeight.toFixed(1)}% of body weight).`;
      }
    } else if(joint) {
      caption=rec.fixedArmsRmsChangeNm>=0.005?`Holding the arms fixed changes this moment by ${rec.fixedArmsRmsChangeNm.toFixed(2)} N·m RMS over the stride.`:'This leg moment is unchanged when the arms are fixed. Leg motion and ground forces are prescribed in both calculations.';
    } else {
      caption=`The modeled path runs from ${Math.min(...recorded).toFixed(1)} to ${Math.max(...recorded).toFixed(1)} mm over the stride, including the tendon.`;
    }
    find('caption').textContent=caption;
    find('method').textContent=ground
      ?'Three force plates measure the total load under both feet. The comparison starts at 20% because the trailing foot can miss the plates early in the stride. The full-body estimate follows the center of mass of all segments. The simpler estimate follows the height of the pelvis origin. Both use the same model mass and 6 Hz filter to convert motion into force.'
      :joint?'Recorded motion and three force plates drive inverse dynamics. Each fixed arm holds its stride-mean joint angles while leg and trunk motion and external forces stay prescribed.'
      :'The recorded joint coordinates move each attachment and wrapping surface. The summed path length includes muscle and tendon.';
    find('error').textContent=ground?`The two curves correlate at r = ${rec.correlation.toFixed(3)}. The calculation averages ${Math.abs(rec.biasN).toFixed(1)} N ${rec.biasN<0?'below':'above'} the plate. The source model weighs ${study.forcePlateCheck.modelMassKg.toFixed(1)} kg and the participant weighed ${study.forcePlateCheck.participantMassKg} kg.`:joint?`Agreement with the released calculation for this moment is ${rec.rmseNm<0.001?'within 0.001':rec.rmseNm.toFixed(3)} N·m RMS.`:'';
    displayPhase();
  }
  function chooseStructure() {
    const forward=mode.value==='forward', ground=mode.value==='ground';
    const coordinateLabel=ground?'Force direction':'Structure';
    find('coordinate-label').textContent=coordinateLabel;coordinate.setAttribute('aria-label',coordinateLabel);
    const groundAxes={vertical:{label:'Vertical'},laboratoryX:{label:'Horizontal laboratory X'},laboratoryZ:{label:'Horizontal laboratory Z'}};
    const signals=ground?Object.entries(groundAxes).filter(([k])=>groundCheck()[k]):forward?Object.entries(study.moments).filter(([k])=>k in study.forwardResponse.samples[0].deltaDegPerSecondSquared):Object.entries(study[mode.value]);
    if(mode.value==='muscles'){
      const named=document.createElement('optgroup');named.label='Featured muscle paths';
      const remaining=document.createElement('optgroup');remaining.label='Additional OpenSim paths';
      for(const [key,row] of signals)(row.label!==key?named:remaining).append(new Option(row.label,key));
      coordinate.replaceChildren(named,remaining);
    }else coordinate.replaceChildren(...signals.map(([key,row])=>new Option(row.label,key)));
    const preferred=forward?'hip_adduction':mode.value==='moments'?'lumbar_rotation':'glmed2';
    if(signals.some(([key])=>key===preferred))coordinate.value=preferred;
    if(forward){
      setPlaying(false); momentPhase=phase;
      const sampled=study.forwardResponse.samples.map(s=>s.phasePct);
      phase=sampled[phaseIndex(sampled,phase)];scrub.min=sampled[0];scrub.max=sampled.at(-1);scrub.step=sampled[1]-sampled[0];
    }else{
      if(play.disabled)phase=momentPhase;
      const phases=ground?groundCheck().phasePct:study.phasePct;
      scrub.min=phases[0];scrub.max=phases.at(-1);scrub.step=1;
      phase=Math.max(phases[0],Math.min(phase,phases.at(-1)));
    }
    play.disabled=forward;
    drawSignal();
  }
  scrub.addEventListener('input',()=>{setPlaying(false);phase=Number(scrub.value);displayPhase();});
  play.addEventListener('click',()=>setPlaying(!playing));mode.addEventListener('change',()=>{fixed=false;if(study)chooseStructure()});
  find('free').addEventListener('click',()=>{fixed=false;if(study)drawSignal()});
  find('fixed').addEventListener('click',()=>{fixed=true;if(study)drawSignal()});
  coordinate.addEventListener('change',()=>{if(study)drawSignal()});
  find('proxy').addEventListener('change',()=>{if(study)drawSignal()});
  find('try-arms').addEventListener('click',()=>{mode.value='moments';fixed=true;chooseStructure();find('fixed').focus();});
  chart.addEventListener('pointerdown',event=>{
    if(!study||mode.value==='forward')return;
    const matrix=chart.getScreenCTM();if(!matrix)return;
    const pos=new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());
    phase=Math.round(plotPhases[0]+(plotPhases.at(-1)-plotPhases[0])*Math.max(0,Math.min(1,(pos.x-chartBox.left)/(chartBox.right-chartBox.left))));setPlaying(false);displayPhase();
  });
  read(motionUrl,abort.signal).then(data=>{
    if(disposed)return;motion=data;
    project=fallbackSkeleton(motion,find('fallback'));displayPhase();
  }).catch(()=>{});
  read(strideUrl,abort.signal).then(data=>{
    if(disposed)return;
    study=data;
    if(study.forcePlateCheck?.vertical) {
      mode.add(new Option('Ground force','ground'),0);mode.value='ground';
      const body=find('evidence').querySelector('tbody');
      selectedTrial=study.forcePlateComparison?.find(t=>t.recording==='mtpgait3'&&t.clipId&&t.forcePlateCheck);
      for(const trial of study.forcePlateComparison??[]){
        const row=body.insertRow();
        const label=row.insertCell();
        if(trial.clipId && trial.forcePlateCheck) {
          const button=document.createElement('button');button.type='button';button.textContent=trial.condition;
          button.setAttribute('aria-pressed',String(trial===selectedTrial));label.append(button);
          button.addEventListener('click',()=>{
            selectedTrial=trial;
            for(const b of body.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===button));
            drawSignal();
          });
        }else label.textContent=trial.condition;
        for(const error of [trial.pelvisOnlyRmseN,trial.wholeBodyRmseN])row.insertCell().textContent=error.toFixed(1)+' N';
      }
    }
    if(study.forwardResponse?.samples?.length>1)mode.add(new Option('Reduce hip muscle force','forward'));
    chooseStructure();const m=study.model;
    find('meta').textContent=`${m.bodyCount} body segments · ${m.coordinateCount} coordinates · ${m.muscleCount} muscle–tendon paths`;
  }).catch(error=>{if(!disposed)find('caption').textContent='The whole-body calculation could not be loaded.'; if(!abort.signal.aborted)console.warn(error);});
  import('./body-stage.js').then(({createBodyStage})=>{
    if(disposed)return null;
    return createBodyStage(find('model'),{view:'oblique',layer:'muscles',region:'body'});
  }).then(instance=>{
    if(!instance)return;
    if(disposed){instance.dispose();return;}
    stage=instance;stage.setActive(active);displayPhase();find('fallback').hidden=true;
  }).catch(()=>{if(!disposed)find('model').hidden=true;});
  return {dispose,setActive(value){active=value;stage?.setActive(value);if(!value)setPlaying(false);}};
}
