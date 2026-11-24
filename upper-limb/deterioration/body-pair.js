import { createBodyStage } from '../presentation/body-stage.js';

/** Two copies of the atlas, driven by the same phase and viewed from the same camera angle. */
export async function createBodyPair(hosts) {
  if (hosts.length !== 2) throw new Error('Two whole-body viewports are required.');
  const stages = [];
  const colors = ['#a1c3da', '#f0b593'];
  const results = await Promise.allSettled(hosts.map((host, index) => createBodyStage(host, {
    view: 'front',
    guideColor: colors[index],
    onOrbit: orbit => stages[1 - index]?.setOrbit(orbit),
  })));
  const failure = results.find(result => result.status === 'rejected');
  if (failure) {
    for (const result of results) {
      if (result.status === 'fulfilled') result.value.dispose();
    }
    throw failure.reason;
  }
  stages.push(...results.map(result => result.value));

  let signal = 'trunkLeanDeg';
  let solutions, shared;
  function update() {
    if (!solutions) return;
    stages.forEach((stage, index) => stage.setPose({ ...shared, baseline: solutions[index], signal }));
  }
  return {
    setFrames(baseline, altered, { phase, task, heightM }) {
      solutions = [baseline, altered];
      shared = { phase, task, heightM };
      update();
    },
    setSignal(next) {
      signal = next;
      update();
    },
    setView(view) {
      stages.forEach(stage => stage.setView(view));
    },
    setLayer(layer) {
      stages.forEach(stage => stage.setLayer(layer));
    },
    reset() {
      stages.forEach(stage => stage.reset());
    },
    dispose() {
      stages.forEach(stage => stage.dispose());
    },
  };
}
