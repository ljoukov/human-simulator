import './conditions.css';

const source = (url, label) => `<a class="condition-source" href="${url}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`;
const englandReport = 'https://www.arthritis-uk.org/media/lh3fnw1g/technical-report-england-hip-and-knee-oa-prevalence-estimates-2026.pdf';

const hip = {
  region: 'Four UK NHS physiotherapy sites',
  stat: '65.4',
  finding: 'reported difficulty walking',
  sample: '106 of 162 patients seeking physiotherapy for greater trochanteric pain syndrome.',
  source: source('https://doi.org/10.1002/msc.1419', 'Stephens et al., 2019 · Table 1'),
  chartTitle: 'Outer-hip pain in women and men',
  chartLabel: 'Prevalence of greater trochanteric pain syndrome in the US cohort',
  chartDescription: 'US adults aged 50 to 79 who had knee osteoarthritis or were at increased risk of it. Clinicians assessed pain and tenderness at the outer hip.',
  // The unilateral and bilateral groups are mutually exclusive in the source.
  rows: [['Women', 15.0, 8.5], ['Men', 6.6, 1.9]],
  chartSource: source('https://pubmed.ncbi.nlm.nih.gov/17678660/', 'Segal et al., 2007 · US cohort'),
  intervention: 'In an Australian trial of people with confirmed gluteal tendinopathy, 51 of 66 reported improvement after eight weeks of load-management education and exercise, compared with 20 of 68 assigned to wait and see.',
  studyTitle: 'Load management and exercise',
  applicationTitle: 'Hip strength during a step',
  trialSource: source('https://pubmed.ncbi.nlm.nih.gov/29720374/', 'Mellor et al., 2018 · randomised trial'),
  application: 'Change hip abductor strength and see how much the pelvis drops and the trunk leans during a step.',
  href: '../deterioration/',
  action: 'Open the hip simulator',
};
const hipOa = {
  region: 'England · Women aged 45 and over',
  stat: '21',
  finding: 'estimated to have hip osteoarthritis',
  sample: 'The 95% confidence interval was 19.4% to 22.6%. Researchers used frequent hip pain as a proxy for osteoarthritis and used hip replacement history when pain responses were missing.',
  source: source(englandReport + '#page=6', 'Arthritis UK, 2026 · How cases were counted'),
  chartTitle: 'Estimated prevalence by age',
  chartLabel: 'Estimated hip osteoarthritis prevalence by age in England, women and men combined',
  chartDescription: 'Women and men combined, from the 2021 to 2023 survey.',
  rows: [['45 to 64', 11.7], ['65 to 74', 18.9], ['75 and over', 22.8]],
  chartSource: source(englandReport + '#page=14', 'Arthritis UK, 2026 · Table 1'),
  studyTitle: 'Hip motion during walking',
  intervention: 'A gait study found that people with more severe hip osteoarthritis moved their hips through a smaller front-to-back arc as they walked.',
  trialSource: source('https://doi.org/10.1016/j.jelekin.2015.10.010', 'Rutherford et al., 2015 · Gait study'),
  applicationTitle: 'Restricted motion in the simulator',
  application: 'Compare a virtual hip that resists extension with one that has weaker muscles. See how each changes hip motion and trunk lean.',
  href: '#challenge',
  action: 'Compare the two movements',
};
const knee = {
  region: 'England · Women aged 45 and over',
  stat: '25.8',
  finding: 'estimated to have knee osteoarthritis',
  sample: 'The 2021–23 English Longitudinal Study of Ageing used reported knee pain as a proxy for osteoarthritis. The 95% confidence interval was 24.2% to 27.5%.',
  source: source(englandReport + '#page=14', 'Arthritis UK, 2026 · Table 1 and methods'),
  chartTitle: 'X-ray changes over 14 years',
  chartLabel: 'Prevalence of X-ray signs of knee osteoarthritis in the Chingford cohort',
  chartDescription: '561 women in Chingford, London, had X-rays throughout the study. Their median age at entry was 53. The bars show the share with signs of osteoarthritis in at least one knee.',
  rows: [['At entry', 13.7], ['14 years later', 47.8]],
  chartSource: source('https://pubmed.ncbi.nlm.nih.gov/22422507/', 'Leyland et al., 2012 · Chingford Women’s Study'),
  intervention: 'In a US trial of 68 adults with osteoarthritis on the inner side of the knee, personalised foot-angle retraining reduced pain by 1.2 points more than sham retraining at one year on a 0–10 scale.',
  studyTitle: 'Changing foot angle',
  applicationTitle: 'Force inside the knee',
  trialSource: source('https://pubmed.ncbi.nlm.nih.gov/40816302/', 'Uhlrich et al., 2025 · randomised trial'),
  application: 'Compare how different walking patterns load the inner knee. Check each estimate against a force measurement from a knee implant.',
  href: '#knee',
  action: 'Compare knee loads',
};

// The charts share a 0–50% scale. Outer-hip pain totals include either or both sides.
function chart(rows, label, split) {
  const row = ([label, first, second = 0]) => {
    const sum = first + second;
    return `<tr><th scope="row">${label}</th><td class="condition-track-cell">
      <div class="condition-track" aria-hidden="true">
        <span class="condition-bar" style="width:${first * 2}%"></span>${split ? `<span class="condition-bar-both" style="width:${second * 2}%"></span>` : ''}
      </div></td><td class="condition-percent">${sum.toFixed(1)}%${split ? `<span class="condition-sr">. One hip ${first.toFixed(1)}%, both hips ${second.toFixed(1)}%</span>` : ''}</td></tr>`;
  };
  return `<table class="condition-chart" aria-label="${label}">
    <tbody>${rows.map(row).join('')}</tbody>
    <tfoot aria-hidden="true"><tr><td></td><td><div class="condition-ticks"><span>0%</span><span>25%</span><span>50%</span></div></td><td></td></tr></tfoot>
  </table>`;
}

export function mountCondition(root, id) {
  const d = { 'hip-impact': hip, 'hip-oa': hipOa, 'knee-impact': knee }[id];
  const split = d === hip;
  const legend = split ? '<div class="condition-key"><span>One hip</span><span>Both hips</span></div>' : '';
  root.innerHTML = `<article class="condition-page">
    <div class="condition-data">
      <section class="condition-stat" aria-label="${d.region}">
        <p class="condition-location">${d.region}</p>
        <p class="condition-number">${d.stat}<span>%</span></p>
        <h2>${d.finding}</h2>
        <p class="condition-note">${d.sample}</p>${d.source}
      </section>
      <section class="condition-study">
        <h2>${d.chartTitle}</h2>
        <p class="condition-note">${d.chartDescription}</p>
        ${chart(d.rows, d.chartLabel, split)}${legend}${d.chartSource}
      </section>
    </div>
    <div class="condition-application">
      <section><h2>${d.studyTitle}</h2>
        <p>${d.intervention}</p>${d.trialSource}</section>
      <section><h2>${d.applicationTitle}</h2>
        <p>${d.application}</p><a class="condition-action" href="${d.href}">${d.action}<span aria-hidden="true"> →</span></a></section>
    </div>
  </article>`;
}
