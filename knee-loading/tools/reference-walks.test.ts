import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const archive = JSON.parse(readFileSync(new URL('../data/reference-walks.json', import.meta.url), 'utf8')) as {
  segmentOrder: string[];
  clips: {
    id: string;
    participant: string;
    ageYears: number;
    sex: string;
    stanceEndPct: number;
    frames: number[][];
  }[];
};

test('optical recordings have distinct participants and continuous pelvis and wrist poses', () => {
  assert.equal(new Set(archive.clips.map(({ participant }) => participant)).size, archive.clips.length);
  assert.ok(archive.clips.length >= 4);
  const pelvis = archive.segmentOrder.indexOf('pelvis') * 7;
  const wrists = ['hand_r', 'hand_l'].map((id) => archive.segmentOrder.indexOf(id) * 7);
  assert.ok(pelvis >= 0 && wrists.every((index) => index >= 0));

  for (const { id, ageYears, sex, stanceEndPct, frames } of archive.clips) {
    assert.ok(ageYears > 0 && sex && stanceEndPct > 0 && stanceEndPct < 100, id);
    assert.equal(frames.length, 101, id);
    const advance = (frames.at(-1)![pelvis] - frames[0][pelvis]) / (frames.length - 1);
    for (let index = 1; index < frames.length; index++) {
      const previous = frames[index - 1];
      const current = frames[index];
      const rootStep = Math.hypot(
        current[pelvis] - previous[pelvis] - advance,
        current[pelvis + 1] - previous[pelvis + 1],
        current[pelvis + 2] - previous[pelvis + 2],
      );
      assert.ok(rootStep < 0.02, `${id} pelvis at ${index}% moves ${rootStep} m in one sample`);
      for (const wrist of wrists) {
        const dot = current.slice(wrist + 3, wrist + 7)
          .reduce((sum, value, axis) => sum + value * previous[wrist + 3 + axis], 0);
        assert.ok(Math.abs(dot) > 0.99, `${id} wrist at ${index}% must keep quaternion continuity`);
      }
    }
  }
});

test('optical leg spacing follows the recorded tracks across each stride', () => {
  // Signed right-minus-left spacing (metres) from the original C3D FEO/TIO
  // trajectories, scaled by the ratio of model leg length to recorded leg length.
  const recordings = [
    { participant: 'SUBJ01', minKnee: 0.14624, minAnkle: 0.09558, ankle70: 0.09993 },
    { participant: 'SUBJ02', minKnee: 0.20868, minAnkle: 0.14363, ankle70: 0.19344 },
    { participant: 'SUBJ03', minKnee: 0.20074, minAnkle: 0.10252, ankle70: 0.14463 },
    { participant: 'SUBJ04', minKnee: 0.18183, minAnkle: 0.08207, ankle70: 0.12436 },
    { participant: 'SUBJ130', minKnee: 0.13870, minAnkle: 0.05251, ankle70: 0.10996 },
  ];
  const { landmarks } = JSON.parse(readFileSync(new URL('../data/whole-body-motion.json', import.meta.url), 'utf8')) as {
    landmarks: Record<string, { segment: string; offsetM: number[] }>;
  };
  function lateralCenter(frame: number[], landmark: string) {
    const { segment, offsetM: [dx, dy, dz] } = landmarks[landmark];
    // Each knee and ankle is on its parent segment's local Y axis.
    assert.equal(dx, 0);
    assert.equal(dz, 0);
    const index = archive.segmentOrder.indexOf(segment) * 7;
    assert.ok(index >= 0);
    const [z, qx, qy, qz, qw] = frame.slice(index + 2, index + 7);
    return z + 2 * (qy * qz + qw * qx) * dy;
  }
  for (const measured of recordings) {
    const clip = archive.clips.find(({ participant }) => participant === measured.participant);
    assert.ok(clip, measured.participant);
    const width = (frame: number[], joint: string) =>
      lateralCenter(frame, 'right' + joint) - lateralCenter(frame, 'left' + joint);
    const ankles = clip.frames.map((frame) => width(frame, 'Ankle'));
    assert.ok(Math.abs(Math.min(...ankles) - measured.minAnkle) < 0.0001, clip.id);
    assert.ok(Math.abs(ankles[70] - measured.ankle70) < 0.0001, clip.id);
    // The fixed reference hips and bony sole constraint allow modest knee deviation.
    const knees = clip.frames.map((frame) => width(frame, 'Knee'));
    assert.ok(Math.abs(Math.min(...knees) - measured.minKnee) < 0.035, clip.id);
    if (measured.participant === 'SUBJ02') assert.ok(Math.abs(knees[70] - 0.23002) < 0.005, clip.id);
  }
});
