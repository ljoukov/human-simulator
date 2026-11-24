import './judging.css';
import {source, model, forcePlateCheck, forcePlateComparison, forwardResponse, moments} from '../../whole-body-study/stride.json';
import {offsetsMm, pareto, resamples} from '../../hip-study/data/benchmark-metal-sensitivity.json';
import {heldOutBodies, originalCohort} from '../../hip-study/data/observation-ablation.json';
import sciencePose from './media/science-walking-pose.webp';

const count = value => value.toLocaleString('en-GB');
const percent = new Intl.NumberFormat('en-GB', {style: 'percent'});
const signedNumber = new Intl.NumberFormat('en-GB', {signDisplay: 'always', maximumFractionDigits: 0});

export function mountCorrectness(root) {
  const {vertical, kinematicsLowpassHz} = forcePlateCheck;
  const [start, end] = forcePlateComparison[0].phasePctRange;
  root.innerHTML = `<article class="judging-slide">
    <section class="judging-feature">
      <h2>Vertical force error against force plates</h2>
      <p class="judging-number">${vertical.rmseN.toFixed(1)} <span>N RMS</span></p>
      <p>${vertical.rmsePctParticipantBodyWeight.toFixed(1)}% of body weight over ${start}–${end}% of one person’s ordinary stride. The model estimates the push underfoot from the acceleration of the body’s centre of mass.</p>
    </section>
    <div class="judging-notes">
      <section>
        <h2>Recorded motion and anatomy</h2>
        <p>Cameras track body markers as a person walks. Marker tracks from the public Grand Challenge archive drive a ${model.bodyCount}-segment OpenSim body scaled to the participant. BodyParts3D supplies the visible anatomy.</p>
      </section>
      <section>
        <h2>Assumptions and numerical checks</h2>
        <p>The force check uses rigid segments with fixed masses and a ${kinematicsLowpassHz} Hz filter to reduce motion noise. With the same recorded inputs, ${Object.keys(moments).length} joint torque traces match the archived calculations within 0.000002 N·m RMS.</p>
      </section>
      <p class="judging-source"><a href="#whole-body">Force plate comparison</a><a href="#knee">Knee implant check</a><a href="${source.url}" target="_blank" rel="noopener">Source archive</a></p>
    </div>
  </article>`;
}

export function mountReshaping(root) {
  root.innerHTML = `<article class="judging-slide">
    <section class="judging-feature">
      <h2>What runs in parallel</h2>
      <dl class="judging-work">
        <div><dt>Forces from tissue changing shape</dt><dd>Divide tissue into small 3D pieces. Each piece computes how much force it generates as it stretches, compresses or contracts.</dd></div>
        <div><dt>Forces where surfaces meet</dt><dd>Each contact pair computes the push between the two surfaces and the friction that resists sliding.</dd></div>
      </dl>
    </section>
    <div class="judging-notes">
      <section>
        <h2>How the body moves together</h2>
        <p>The GPU adds the forces where pieces share a mesh point. Tendon attachments pass those forces to the bones. The model updates the whole body together, then takes the next time step.</p>
      </section>
      <section>
        <h2>Calibration check</h2>
        <p>Candidate models can also be scored independently. In a hip calibration test, the GPU screened out models that fit every participant worse than a rival. It picked the same best model as a full CPU search in <strong>${count(pareto.winnerAgreement)} of ${count(offsetsMm.length * resamples)} fits</strong>.</p>
      </section>
      <p class="judging-source"><a href="#gpu-body-model">Whole-body mesh</a><a href="../hip-study/#compute">Calibration method</a></p>
    </div>
  </article>`;
}

export function mountScience(root) {
  const example = forwardResponse.samples.find(sample => sample.phasePct === 30);
  const response = example.deltaDegPerSecondSquared;
  const stance = originalCohort.effort.loadedStance.balancedAccuracy;
  root.innerHTML = `<article class="judging-slide science-slide">
    <section class="science-result">
      <h2>Response to less hip muscle force</h2>
      <figure class="science-figure">
        <div class="science-anatomy">
          <img src="${sciencePose}" width="1320" height="920" alt="Rear oblique view of the walking model with its right foot on the ground.">
          <div class="science-callout science-callout--input" style="--marker-x:52.2%;--marker-y:47.5%"><span><strong>${signedNumber.format(-forwardResponse.totalForceReductionN)} <small>N</small></strong>Right gluteus medius force</span></div>
          <div class="science-callout science-callout--left" style="--marker-x:48.3%;--marker-y:42.3%"><span><strong>${signedNumber.format(response.lumbar_rotation)} <small>°/s²</small></strong>Lower-back rotation</span></div>
          <div class="science-callout" style="--marker-x:56%;--marker-y:26%"><span><strong>${signedNumber.format(response.arm_flex_r)} <small>°/s²</small></strong>Right shoulder forward lift</span></div>
        </div>
        <figcaption>Simulated change in how quickly each joint starts turning at ${example.phasePct}% of the recorded stride.</figcaption>
      </figure>
      <p class="science-source"><a href="#whole-body">Whole-body experiment</a><a href="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html" target="_blank" rel="noopener">BodyParts3D atlas</a></p>
    </section>
    <section class="science-result">
      <h2>Telling weakness from stiffness</h2>
      <p>Separate simulation of standing on one leg with added weight.</p>
      <dl class="judging-readouts">
        <div><dt>Hip angles alone</dt><dd class="judging-number">${percent.format(stance.hipAngles)}</dd></div>
        <div><dt>Hip angles and trunk lean</dt><dd class="judging-number">${percent.format(stance.hipAnglesAndTrunkLean)}</dd></div>
      </dl>
      <p>Average accuracy across both conditions in ${heldOutBodies} new virtual bodies.</p>
      <p class="science-takeaway"><strong>What to measure in people</strong> Record hip angles and how far the trunk leans during the same one-leg test. Check whether the lean helps tell weakness from stiffness.</p>
      <p class="science-source"><a href="#challenge">Single-leg stance study</a></p>
    </section>
  </article>`;
}
