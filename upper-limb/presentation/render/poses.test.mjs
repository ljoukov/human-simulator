import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {Euler, Quaternion, Vector3} from 'three';
import { createPoseSampler } from './poses.js';
import { simulateGait } from '../dynamics.js';

const load = name => JSON.parse(fs.readFileSync(new URL('../../../knee-loading/data/' + name + '.json', import.meta.url)));
const sampler = createPoseSampler(load('reference-walks'), load('whole-body-motion'));
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const position = (frame, name) => frame.segments[name].position;
const shoulderCenter = ({segments}) => segments.humerus_r.position.map((v, i) => (v + segments.humerus_l.position[i]) / 2);
const hipCenter = ({segments}) => segments.femur.position.map((v, i) => (v + segments.femur_l.position[i]) / 2);

test('all measured clip IDs retain exact source segment lengths and reverse acquisition heading is normalized', () => {
  const data = load('whole-body-motion');
  for (const clip of data.clips) {
    const frame = sampler.sample({clipId: clip.id, phase: .5});
    const row = clip.frames[50];
    const source = name => row.slice(data.segmentOrder.indexOf(name) * 7, data.segmentOrder.indexOf(name) * 7 + 3);
    const originalSpan = source('femur')[2] - source('femur_l')[2];
    const measuredSpan = position(frame, 'femur')[2] - position(frame, 'femur_l')[2];
    assert.ok(Math.abs(measuredSpan - clip.heading * originalSpan) < 1e-8);
    assert.ok(Math.abs(distance(source('femur'), source('tibia')) - distance(position(frame,'femur'),position(frame,'tibia'))) < 1e-8);
    assert.equal(frame.clipId, clip.id);
  }
});

const degrees = radians => radians * 180 / Math.PI;
const recoveredJoint = (frame, child, parent) => {
  const q = new Quaternion().fromArray(frame.segments[child].rotation);
  return parent ? q.premultiply(new Quaternion().fromArray(frame.segments[parent].rotation).invert()) : q;
};

// Independent readback from serialized segment poses. Ball joint ZXY is the
// source OpenSim sequence; a hinge's signed twist is 2 atan2(q_v · axis, q_w).
function readCoordinates(frame) {
  const reference = recoveredJoint(frame, 'pelvis');
  const shoulders = shoulderCenter(frame), hips = hipCenter(frame);
  const out = { pelvisDropDeg: -degrees(new Euler().setFromQuaternion(reference, 'ZXY').x),
    trunkLeanDeg: degrees(Math.atan2(shoulders[2]-hips[2], shoulders[1]-hips[1])) };
  for (const [side, leg, arm] of [[1, '', '_r'], [-1, '_l', '_l']]) {
    const write = (key, value) => { out[side > 0 ? key : 'left' + key[0].toUpperCase() + key.slice(1)] = value; };
    const hip = new Euler().setFromQuaternion(recoveredJoint(frame, 'femur'+leg, 'pelvis'), 'ZXY');
    const shoulder = new Euler().setFromQuaternion(recoveredJoint(frame, 'humerus'+arm, 'torso'), 'ZXY');
    const hinge = (child, parent, axis) => {
      const q = recoveredJoint(frame,child,parent);
      const unit = new Vector3(...axis).normalize();
      const angle = degrees(2 * Math.atan2(new Vector3(q.x,q.y,q.z).dot(unit), q.w));
      return (angle + 540) % 360 - 180;
    };
    write('hipFlexionDeg', degrees(hip.z)); write('hipAdductionDeg', side * degrees(hip.x));
    write('shoulderFlexionDeg', degrees(shoulder.z)); write('shoulderAdductionDeg', side * degrees(shoulder.x));
    write('kneeFlexionDeg', -hinge('tibia'+leg,'femur'+leg,[0,0,1]));
    write('ankleDorsiflexionDeg', hinge('talus'+leg,'tibia'+leg,[-side*.10501355,-side*.17402245,.97912632]));
    write('elbowFlexionDeg', hinge('ulna'+arm,'humerus'+arm,[side*.22604696,side*.022269,.97386183]));
  }
  return out;
}

test('all 16 solved coordinates round trip in the serialized pose for baseline and changed movement', t => {
  let maximumError = 0;
  for (const task of ['walk','stance']) {
    const result = simulateGait({task});
    for (const frames of [result.baseline,result.altered]) {
      for (const solution of frames) {
        const pose = sampler.sample({task, phase: solution.phase, solution});
        for (const [key,value] of Object.entries(readCoordinates(pose))) {
          maximumError = Math.max(maximumError, Math.abs(value-solution[key]));
          assert.ok(Math.abs(value-solution[key]) < 1e-8, key+' residual '+(value-solution[key]));
        }
      }
    }
  }
  t.diagnostic('Largest angular residual: ' + maximumError.toExponential(2) + ' degrees');
});

test('identical copied and shared solver frames produce the same pose, even when the optical template differs', () => {
  const reference = {trunkLeanDeg: 2.4, pelvisDropDeg: 3, hipAdductionDeg: 8, kneeFlexionDeg: 27};
  for (const phase of [0, .07, .31, .53, .72, 1]) {
    const baseline = sampler.sample({phase, solution: reference, reference});
    const frame = sampler.sample({phase, solution: {...reference}, reference});
    assert.deepEqual(frame.segments, baseline.segments);
    assert.notDeepEqual(frame.segments, sampler.sample({phase}).segments);
  }
});

test('stance-side lean, opposite pelvic drop, hip adduction and knee flexion have consistent signed geometry', () => {
  const phase = .3;
  const zero = {trunkLeanDeg:0, pelvisDropDeg:0, hipAdductionDeg:0, kneeFlexionDeg:0};
  const altered = input => sampler.sample({phase, solution: {...zero,...input}});
  const control = altered({});
  const lean = altered({trunkLeanDeg: 9}), drop = altered({pelvisDropDeg: 6});
  const adduction = altered({hipAdductionDeg: 6}), flex = altered({kneeFlexionDeg: 15});
  const sideLean = frame => shoulderCenter(frame)[2] - hipCenter(frame)[2];
  const tilt = frame => position(frame,'femur')[1] - position(frame,'femur_l')[1];
  const thigh = frame => position(frame,'tibia')[2] - position(frame,'femur')[2];
  const shin = frame => position(frame,'talus')[0] - position(frame,'tibia')[0];
  assert.ok(sideLean(lean) > sideLean(control));
  assert.ok(tilt(drop) > tilt(control));
  assert.ok(thigh(adduction) < thigh(control));
  assert.ok(shin(flex) < shin(control));
  for(const changed of [lean,drop,adduction,flex]) {
    assert.ok(distance(position(changed,'talus'),position(control,'talus')) < 1e-9);
    assert.ok(distance(changed.segments.calcaneus.rotation,control.segments.calcaneus.rotation) < 1e-9);
  }
});

test('designed stance supports one fixed ankle with the opposite foot clear during load', () => {
  const neutral = sampler.sample({phase: .5, task:'stance'});
  assert.ok(position(neutral,'talus_l')[1] > position(neutral,'talus')[1] + .03);
  for (const phase of [0, .2, .5, .9, 1]) {
    const changed = sampler.sample({phase,task:'stance', solution: {trunkLeanDeg: 8, hipAdductionDeg: -4}});
    assert.ok(distance(position(changed,'talus'),position(neutral,'talus')) < 1e-9);
  }
});

test('shoulder, elbow and ankle corrections reach distal atlas segments', () => {
  const phase = .3;
  const control = sampler.sample({phase, solution:{ankleDorsiflexionDeg:0, shoulderFlexionDeg:0, elbowFlexionDeg:0}});
  const moved = sampler.sample({phase, solution:{ankleDorsiflexionDeg:5, shoulderFlexionDeg:10, elbowFlexionDeg:6}});
  assert.ok(position(moved, 'hand_r')[0] > position(control, 'hand_r')[0] + .03);
  assert.ok(position(moved, 'toes')[1] > position(control, 'toes')[1]);
  assert.ok(distance(position(moved,'talus'),position(control,'talus')) < 1e-9);
  assert.ok(distance(position(moved,'hand_l'),position(control,'hand_l')) < 1e-9);
});

test('fixed arms stay rigidly attached to the recorded torso across stride phases', () => {
  const fixed = { humerus_r: [.06, .42, .17, .5, -.5, .5, .5], hand_l: [.1,-.15,-.22,0,0,0,1] };
  for (const phase of [.05, .72]) {
    const options = {phase, clipId:'ngait_og5-stance-1'};
    const frame = sampler.sample({...options, fixedArmSegmentsInTorso:fixed});
    const original = sampler.sample(options);
    assert.deepEqual(frame.segments.pelvis, original.segments.pelvis);
    assert.deepEqual(frame.segments.hand_r, original.segments.hand_r);
    const torso = frame.segments.torso;
    const invertTorso = new Quaternion().fromArray(torso.rotation).invert();
    for (const [name, values] of Object.entries(fixed)) {
      const {position:p,rotation:q} = frame.segments[name];
      const localPoint = new Vector3().fromArray(p).sub(new Vector3().fromArray(torso.position)).applyQuaternion(invertTorso);
      const localRotation = new Quaternion().fromArray(q).premultiply(invertTorso);
      assert.ok(distance(localPoint.toArray(), values.slice(0,3)) < 1e-10);
      assert.ok(distance(localRotation.toArray(),values.slice(3)) < 1e-10);
    }
  }
});

test('retargeted walking retains phase-matched foot registration and segment dimensions', () => {
  const {altered} = simulateGait({gluteCapacityPct:45, bandShorteningMm:6});
  const sole = frame => Math.min(...['calcaneus','calcaneus_l','toes','toes_l'].map(key => position(frame,key)[1]));
  for (const solution of altered) {
    const phase = solution.phase;
    const raw = sampler.sample({phase}), solved = sampler.sample({phase,solution});
    assert.ok(Math.abs(sole(raw)-sole(solved)) < 1e-9);
    for (const [parent,child] of [['femur','tibia'],['tibia','talus'],['humerus_l','ulna_l'],['ulna_r','hand_r']]) {
      const length = frame => distance(position(frame,child),position(frame,parent));
      // Ulna to hand includes a second joint with its source relative pose fixed.
      assert.ok(Math.abs(length(solved)-length(raw)) < 1e-8);
    }
    const supportFoot = phase>.13 && phase<.5 ? 'talus' : phase>.63 ? 'talus_l' : null;
    if (supportFoot) {
      const a=position(raw,supportFoot), b=position(solved,supportFoot);
      assert.ok(Math.hypot(a[0]-b[0],a[2]-b[2]) < 1e-9);
    }
  }
});

test('lab-frame mass markers share subframe pose registration across clip headings and stature scaling', () => {
  const archives = [load('reference-walks'), load('whole-body-motion')];
  const origin = [8.3, 0, -1.7], phase = .3746, profileHeightM = 1.77;
  for (const [archive, clip] of [[archives[0],archives[0].clips.at(-1)],
    [archives[1], archives[1].clips.find(c => c.heading === -1)]]) {
    const t = phase * (clip.frames.length - 1), row = Math.floor(t);
    const pelvis = archive.segmentOrder.indexOf('pelvis') * 7;
    const sourcePoint = (name) => {
      const index = archive.segmentOrder.indexOf(name) * 7;
      return origin.map((value, axis) => value + (1-t+row)*clip.frames[row][index+axis]
        + (t-row)*clip.frames[row+1][index+axis]);
    };
    const massMarkers = { body: sourcePoint('torso'), pelvis: sourcePoint('pelvis'), clipOriginGroundM: origin };
    const frame = sampler.sample({clipId:clip.id, phase, profileHeightM, massMarkers});
    const begin = clip.frames[0], end = clip.frames.at(-1);
    const drift = [(1-phase)*begin[pelvis] + phase*end[pelvis], 0, (begin[pelvis+2]+end[pelvis+2])/2];
    for (const [marker,name] of [['pelvis','pelvis'],['body','torso']]) {
      const expected = massMarkers[marker].map((v,axis) => (v-origin[axis]-drift[axis])
        * (clip.heightM ? profileHeightM/clip.heightM : 1) * (axis === 1 ? 1 : clip.heading));
      assert.ok(distance(frame.massMarkers[marker],expected) < 1e-11);
      assert.ok(distance(frame.massMarkers[marker],position(frame,name)) < 1e-11);
    }
    assert.equal(sampler.sample({clipId:clip.id,phase}).massMarkers, undefined);
  }
});

test('every archived force-trial pelvis marker registers on its atlas pelvis', t => {
  const study = JSON.parse(fs.readFileSync(new URL('../../../whole-body-study/stride.json', import.meta.url)));
  let maximum = 0, checks = 0;
  for (const trial of study.forcePlateComparison) {
    const {phasePct,centerOfMassGroundM,pelvisOriginGroundM,clipOriginGroundM} = trial.forcePlateCheck.positionMarkers;
    for (const [i,pct] of phasePct.entries()) {
      const marker = {body:centerOfMassGroundM[i], pelvis:pelvisOriginGroundM[i], clipOriginGroundM};
      const frame = sampler.sample({clipId:trial.clipId, phase:pct/100, massMarkers:marker});
      maximum = Math.max(maximum,distance(frame.massMarkers.pelvis,frame.segments.pelvis.position));
      const markerGap = distance(frame.massMarkers.body,frame.massMarkers.pelvis);
      assert.ok(Math.abs(markerGap-distance(marker.body,marker.pelvis)) < 1e-12);
      checks++;
    }
  }
  // Archive segment positions are rounded to 0.00001 m on each axis.
  assert.ok(checks >= 303 && maximum < .000009);
  t.diagnostic(checks+' force-trial frames, largest pelvis offset '+(maximum*1e6).toFixed(2)+' µm');
});
