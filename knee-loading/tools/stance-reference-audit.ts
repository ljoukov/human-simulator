import { readFile, writeFile } from 'node:fs/promises';
import { validateDataset, type GaitDataset } from '../src/dataset.ts';
import { EARLY_STANCE_WINDOW_PCT, earlyStancePeaks } from '../src/pathway.ts';

const input = new URL('../data/gait.json', import.meta.url);
const output = new URL('../data/stance-reference-audit.json', import.meta.url);
const dataset: GaitDataset = validateDataset(JSON.parse(await readFile(input, 'utf8')));
const participant = 'GC4-JW';
const stances = dataset.trials.filter((trial) => trial.subject === participant).map((trial) => {
  // The input already spans one stance. Only stance phase is used in this audit.
  const peaks = earlyStancePeaks(trial, 100);
  if (!peaks?.medial || !trial.role) throw new Error(`Missing implant peak or study role in ${trial.id}`);
  return {
    id: trial.id, condition: trial.condition, role: trial.role,
    earlyMedialPeakBw: peaks.medial.value, peakStancePct: peaks.medial.stancePct,
  };
});

const ordinary = stances.filter((stance) => stance.condition === 'normal');
const altered = stances.filter((stance) => stance.condition !== 'normal');
if (ordinary.length !== 2 || new Set(ordinary.map((stance) => stance.role)).size !== 2 || !altered.length) {
  throw new Error('Expected two ordinary reference stances with different study roles and at least one modified gait.');
}

const rounded = (value: number) => Number(value.toFixed(7));
const conditions = [...new Set(altered.map((stance) => stance.condition))];
const comparisons = conditions.map((condition) => {
  const observed = altered.filter((stance) => stance.condition === condition);
  const meanPeak = observed.reduce((sum, stance) => sum + stance.earlyMedialPeakBw, 0) / observed.length;
  return {
    condition, meanEarlyMedialPeakBw: rounded(meanPeak),
    byReference: ordinary.map((reference) => ({
      ordinaryStance: reference.id, role: reference.role,
      meanDeltaBw: rounded(meanPeak - reference.earlyMedialPeakBw),
      individualDeltas: observed.map((stance) => ({
        id: stance.id, deltaBw: rounded(stance.earlyMedialPeakBw - reference.earlyMedialPeakBw),
      })),
    })),
  };
});

const audit = {
  source: { file: 'gait.json', upstream: dataset.source.url, participant },
  outcome: 'Peak medial implant force measured in body weights (BW)',
  earlyStanceWindowPctInclusive: EARLY_STANCE_WINDOW_PCT,
  comparison: 'Mean of the individual modified-stance peaks minus the selected ordinary-stance peak. Negative values indicate lower measured load.',
  references: 'The full-stride replay uses the ordinary calibration stance. The formal gait comparison uses the ordinary evaluation stance. Both came from the same pass.',
  scope: 'All stances come from one implant recipient. Repeated overground stances share a recording. Fast treadmill walking changed both speed and surface.',
  ordinaryPeakSpreadBw: rounded(Math.abs(ordinary[0].earlyMedialPeakBw - ordinary[1].earlyMedialPeakBw)),
  stances, comparisons,
};
await writeFile(output, `${JSON.stringify(audit, null, 2)}\n`);
