import './compute.css';
import { timing, candidates, resamples } from '../../hip-study/data/bootstrap-sensitivity.json';
import { exhaustive, pareto, offsetsMm, device } from '../../hip-study/data/benchmark-metal-sensitivity.json';
import { cpu as screenedCpu } from '../../hip-study/data/benchmark-pareto.json';

function median(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[midpoint] : (sorted[midpoint - 1] + sorted[midpoint]) / 2;
}

const fullCpuSeconds = median(timing.cpuSamplesSeconds);
const totalFits = offsetsMm.length * resamples;
const chartMax = Math.ceil(Math.max(fullCpuSeconds, screenedCpu.medianSeconds,
  exhaustive.warmEndToEndSeconds, pareto.warmEndToEndSeconds) / 5) * 5;
const format = value => value.toLocaleString('en-GB');
const methods = {
  full: {
    cpu: fullCpuSeconds, gpu: exhaustive.warmEndToEndSeconds,
    agreement: exhaustive.winnerAgreement === totalFits,
    explanation: `Each processor compares all ${format(candidates)} candidate models for every fit.`,
  },
  screened: {
    cpu: screenedCpu.medianSeconds, gpu: pareto.warmEndToEndSeconds,
    agreement: screenedCpu.matches === totalFits && pareto.winnerAgreement === totalFits,
    explanation: 'A candidate drops out early if another model already fits every participant better.',
  },
};

function timingRow(name, seconds, color) {
  return `<div class="compute-row"><dt>${name}</dt>
    <dd><span class="compute-track" aria-hidden="true"><span style="width:${100 * seconds / chartMax}%;background:${color}"></span></span>
    <strong>${seconds.toFixed(2)} s</strong></dd></div>`;
}

export function mountCompute(root) {
  root.innerHTML = `<article class="compute-page">
    <div class="compute-methods" role="group" aria-label="Search method">
      <button type="button" data-mode="full">Full search</button>
      <button type="button" data-mode="screened">Screen candidates first</button>
    </div>
    <div class="compute-result" aria-live="polite" aria-atomic="true"></div>
    <p class="compute-footnote">Median time after warm-up, from prepared movement measurements to the returned model IDs.
      <a href="../hip-study/#compute">Timing samples and methods</a></p>
  </article>`;
  const buttons = root.querySelectorAll('[data-mode]');
  const result = root.querySelector('.compute-result');
  function show(mode) {
    const run = methods[mode];
    for (const button of buttons) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    result.innerHTML = `<section class="compute-summary">
        <h2>GPU speedup</h2>
        <p class="compute-number">${(run.cpu / run.gpu).toFixed(1)}<span>×</span></p>
        <p>${run.explanation}</p>
      </section>
      <section class="compute-chart">
        <h2>Time on the same ${device} laptop</h2>
        <dl>${timingRow(`CPU · ${timing.cpuThreads} threads`, run.cpu, '#9d6149')}
        ${timingRow('GPU · Metal', run.gpu, '#466b80')}</dl>
        <p>${run.agreement ? `Both processors selected the same model in all ${format(totalFits)} fits.` : 'See the individual agreement counts in the benchmark report.'}</p>
      </section>`;
  }
  for (const button of buttons) button.onclick = () => show(button.dataset.mode);
  show('full');
}
