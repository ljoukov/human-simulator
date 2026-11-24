import './followup.css';

const protocols = {
  hip: {
    name:'Outer hip',
    signal:'Trunk lean under load',
    mechanism:'Two people with the same hip strength may move differently. Measuring strength and motion together would tell us which compensation, if any, each person uses.',
    current:'In eight healthy men, temporarily reducing hip abductor strength produced no detectable group change in trunk lean or pelvic drop during walking.',
    visits:[
      ['At entry','Record walking and single-leg stance','Enroll people without current hip pain. Measure abductor strength and muscle activity alongside trunk lean and pelvic drop.'],
      ['At repeat visits','Track strength and movement together','Repeat both tasks and compare each person with their own range across visits.'],
      ['At follow-up','Compare with symptom onset','Collect dated pain ratings and function scores. Check which changes came before the first symptoms.']
    ],
    decision:'Does a change in muscle activity or trunk lean consistently come before a person’s first hip symptoms?',
    link:'https://pubmed.ncbi.nlm.nih.gov/25875071/', label:'Read the walking experiment'
  },
  knee: {
    name:'Inner knee',
    signal:'Repeated medial knee load',
    mechanism:'A change in one stride can also occur during ordinary walking. Repeated recordings establish the usual range for each person.',
    current:'Two ordinary stances in one instrumented knee differed by 0.268 body weights in early peak medial force.',
    visits:[
      ['At entry','Establish usual loading','Verify the force estimates in implant wearers. In a separate cohort without current knee pain, record repeated walks.'],
      ['At repeat visits','Track load and walking volume','Use the same speed and surface. Record stride counts to estimate cumulative load.'],
      ['At follow-up','Compare with symptoms and function','Compare the timing of load changes with dated pain ratings and function scores.']
    ],
    decision:'Could a sustained change in knee load help predict later symptoms beyond walking volume alone?',
    link:'#knee', label:'Revisit the implant measurement'
  }
};

export function mountFollowup(root) {
  root.innerHTML = `<div class="follow-shell"><section class="follow-case">
    <div class="follow-switch" role="group" aria-label="Choose a region"><button data-case="hip">Outer hip</button><button data-case="knee">Inner knee</button></div>
    <p class="follow-lab">Candidate signal</p><h2 class="follow-signal"></h2><p class="follow-mech"></p>
    <div class="follow-observation"><span>Available evidence</span><p></p><a></a></div>
  </section><section class="follow-plan"><div class="follow-plan-head"><p class="follow-lab">Proposed follow-up study</p><h2>A record across visits</h2></div>
    <ol class="follow-visits"></ol><p class="follow-decision"></p>
  </section></div>`;
  const buttons = [...root.querySelectorAll('[data-case]')];
  function show(key) {
    const data = protocols[key];
    buttons.forEach(b=>b.setAttribute('aria-pressed', String(b.dataset.case === key)));
    root.querySelector('.follow-signal').textContent=data.signal;
    root.querySelector('.follow-mech').textContent=data.mechanism;
    root.querySelector('.follow-observation p').textContent=data.current;
    const back=root.querySelector('.follow-observation a');back.textContent=data.label+' ↗';back.href=data.link;
    if (data.link.startsWith('https:')) { back.target = '_blank'; back.rel = 'noopener noreferrer'; }
    else { back.removeAttribute('target'); back.removeAttribute('rel'); }
    const visits=root.querySelector('.follow-visits');visits.replaceChildren();
    for(const [time,action,detail] of data.visits) {
      const li=document.createElement('li');
      const tag=document.createElement('span'), heading=document.createElement('h3'), text=document.createElement('p');
      tag.textContent=time; heading.textContent=action; text.textContent=detail;li.append(tag,heading,text);visits.append(li);
    }
    root.querySelector('.follow-decision').textContent=data.decision;
  }
  buttons.forEach(b=>b.onclick=()=>show(b.dataset.case));
  show('hip');
}
