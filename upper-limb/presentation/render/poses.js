import { Euler, MathUtils, Quaternion, Vector3 } from 'three';
import stride from '../model/stride.json' with { type: 'json' };

// Atlas coordinates: +X walking direction, +Y up, +Z anatomical right.
const forward = new Vector3(1, 0, 0);
const right = new Vector3(0, 0, 1);
const reverse = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI);
const deg = (axis, value) => new Quaternion().setFromAxisAngle(axis, MathUtils.degToRad(value));
const sides = [
  { suffix: '', side: 1, prefix: '' },
  { suffix: '_l', side: -1, prefix: 'left' },
];
const legNodes = ['femur', 'tibia', 'patella', 'talus', 'calcaneus', 'toes'];
const upperNodes = ['torso', 'humerus_r', 'ulna_r', 'radius_r', 'hand_r', 'humerus_l', 'ulna_l', 'radius_l', 'hand_l'];

function sourceRegistration(archive, clip, phase, scale) {
  const pelvicIndex = archive.segmentOrder.indexOf('pelvis') * 7;
  const start = clip.frames[0];
  const finish = clip.frames.at(-1);
  const center = new Vector3(
    MathUtils.lerp(start[pelvicIndex], finish[pelvicIndex], phase), 0,
    (start[pelvicIndex + 2] + finish[pelvicIndex + 2]) / 2,
  );
  return point => {
    point.sub(center).multiplyScalar(scale);
    if (clip.heading === -1) point.applyQuaternion(reverse);
    return point;
  };
}

function readPose(archive, clip, phase, registerPoint) {
  const count = clip.frames.length - 1;
  const frame = phase * count;
  const before = Math.floor(frame);
  const first = clip.frames[before];
  const last = clip.frames[Math.min(count, before + 1)];
  const pose = {};
  archive.segmentOrder.forEach((name, i) => {
    const offset = i * 7;
    const position = new Vector3().fromArray(first, offset)
      .lerp(new Vector3().fromArray(last, offset), frame - before);
    const orientation = new Quaternion().fromArray(first, offset + 3).normalize()
      .slerp(new Quaternion().fromArray(last, offset + 3).normalize(), frame - before);
    if (clip.heading === -1) orientation.premultiply(reverse);
    pose[name] = { position: registerPoint(position), orientation };
  });
  return pose;
}

function around(pose, names, pivot, rotation) {
  for (const name of names) {
    const segment = pose[name];
    segment.position.sub(pivot).applyQuaternion(rotation).add(pivot);
    segment.orientation.premultiply(rotation);
  }
}

function translate(pose, shift) {
  for (const { position } of Object.values(pose)) position.add(shift);
}

// Rotations from the pinned Falisse subject1.osim SpatialTransform definitions.
// Positive knee flexion is the negative knee_angle coordinate in that model.
const elbowAxis = new Vector3(.22604696, .022269, .97386183).normalize();
const ankleAxis = new Vector3(-.10501355, -.17402245, .97912632).normalize();
const bodyAxes = ['pelvis', ...upperNodes, ...sides.flatMap(({suffix}) => legNodes.map(name => name + suffix))];
const getRotation = (pose, name) => name ? pose[name].orientation : new Quaternion();

function replaceJointRotation(pose, names, parent, setLocal) {
  const joint = pose[names[0]];
  const parentRotation = getRotation(pose, parent);
  const current = parentRotation.clone().invert().multiply(joint.orientation);
  const target = setLocal(current);
  const correction = parentRotation.clone().multiply(target).multiply(joint.orientation.clone().invert());
  around(pose, names, joint.position.clone(), correction);
}

// OpenSim uses intrinsic (flexion Z, adduction X, axial rotation Y) on hips,
// shoulders and pelvis. Preserve dimensions for which the solver has no channel.
function setBallJoint(pose, names, parent, flexion, adduction) {
  if (flexion === undefined && adduction === undefined) return;
  replaceJointRotation(pose, names, parent, current => {
    const angles = new Euler().setFromQuaternion(current, 'ZXY');
    if (flexion !== undefined) angles.z = MathUtils.degToRad(flexion);
    if (adduction !== undefined) angles.x = MathUtils.degToRad(adduction);
    return new Quaternion().setFromEuler(angles);
  });
}

// The atlas IK uses a different generic skeleton. Preserve its off-axis swing,
// while assigning the solved angle exactly about the source model's hinge axis.
function setHinge(pose, names, parent, axis, angle) {
  if (angle === undefined) return;
  replaceJointRotation(pose, names, parent, current => {
    const component = current.x * axis.x + current.y * axis.y + current.z * axis.z;
    const twist = new Quaternion(axis.x * component, axis.y * component, axis.z * component, current.w).normalize();
    return current.multiply(twist.invert()).multiply(deg(axis, angle));
  });
}

function setTrunkLean(pose, angle) {
  if (angle === undefined) return;
  const hips = pose.femur.position.clone().lerp(pose.femur_l.position, .5);
  const shoulders = pose.humerus_r.position.clone().lerp(pose.humerus_l.position, .5);
  const pivot = pose.torso.position.clone();
  const arm = shoulders.sub(pivot), offset = pivot.clone().sub(hips);
  const target = MathUtils.degToRad(angle);
  // Rotate about the spinal joint, solving the global hip-to-shoulder lean.
  const residual = (offset.y * Math.sin(target) - offset.z * Math.cos(target)) / Math.hypot(arm.y, arm.z);
  const rotation = target - Math.atan2(arm.z, arm.y) + Math.asin(MathUtils.clamp(residual, -1, 1));
  around(pose, upperNodes, pivot, new Quaternion().setFromAxisAngle(forward, rotation));
}

function soleHeight(pose) {
  return Math.min(pose.calcaneus.position.y, pose.calcaneus_l.position.y, pose.toes.position.y, pose.toes_l.position.y);
}

function solvePose(pose, frame, support, registerFloor = false, pelvisPitch) {
  const floor = soleHeight(pose);
  const anchors = sides.map(({suffix}) => pose['talus' + suffix].position.clone());
  const footAxes = sides.map(({suffix}) => pose['calcaneus' + suffix].orientation.clone());
  const drop = frame.pelvisDropDeg;
  setBallJoint(pose, bodyAxes, null, pelvisPitch, drop !== undefined ? -drop : undefined);
  setTrunkLean(pose, frame.trunkLeanDeg);

  sides.forEach(({suffix, side, prefix}, i) => {
    const read = key => frame[prefix ? prefix + key[0].toUpperCase() + key.slice(1) : key];
    const mirror = value => value === undefined ? undefined : side * value;
    const names = legNodes.map(key => key + suffix);
    setBallJoint(pose, names, 'pelvis', read('hipFlexionDeg'), mirror(read('hipAdductionDeg')));
    const knee = read('kneeFlexionDeg');
    setHinge(pose, names.slice(1), names[0], right, knee === undefined ? undefined : -knee);
    const ankle = read('ankleDorsiflexionDeg');
    if (ankle !== undefined) {
      setHinge(pose, names.slice(3), names[1], ankleAxis.clone().multiply(new Vector3(side, side, 1)), ankle);
    } else {
      const contact = i === 0 ? support : 1 - support;
      const restore = footAxes[i].multiply(pose[names[4]].orientation.clone().invert());
      around(pose, names.slice(3), pose[names[3]].position.clone(), new Quaternion().slerp(restore, contact));
    }
    anchors[i].sub(pose[names[3]].position);

    const armSuffix = i === 0 ? '_r' : '_l';
    const arm = ['humerus','ulna','radius','hand'].map(key => key + armSuffix);
    setBallJoint(pose, arm, 'torso', read('shoulderFlexionDeg'), mirror(read('shoulderAdductionDeg')));
    setHinge(pose, arm.slice(1), arm[0], elbowAxis.clone().multiply(new Vector3(side, side, 1)), read('elbowFlexionDeg'));
  });
  // Ankle matches its registered track in single support. Double support
  // crossfades the two constraints, without altering solved joint angles.
  const shift = anchors[1].lerp(anchors[0], support);
  // Retargeted foot orientation can change which heel or forefoot reaches the
  // ground first. Use the original contact envelope for vertical registration.
  if (registerFloor) shift.y = floor - soleHeight(pose);
  translate(pose, shift);
}

function supportFraction(phase, toeOff) {
  const transfer = Math.max(0.025, toeOff - 0.5);
  if (phase < transfer) return MathUtils.smoothstep(phase, 0, transfer);
  return 1 - MathUtils.smoothstep(phase, 0.5, toeOff);
}

function makeNeutralStance(archive, clip, scale) {
  // Use a static anthropometric template for segment lengths and hand orientation;
  // no optical walking time series drives the single-leg hold.
  const pose = readPose(archive, clip, 0.30, sourceRegistration(archive, clip, 0.30, scale));
  const hips = pose.femur.position.clone().lerp(pose.femur_l.position, 0.5);
  const width = pose.femur.position.distanceTo(pose.femur_l.position);
  sides.forEach(({ suffix, side }) => {
    const f = pose['femur' + suffix], t = pose['tibia' + suffix], a = pose['talus' + suffix];
    const thigh = f.position.distanceTo(t.position), shin = t.position.distanceTo(a.position);
    const patella = pose['patella' + suffix];
    const patellaOffset = patella.position.clone().sub(t.position).applyQuaternion(t.orientation.clone().invert());
    const calc = pose['calcaneus' + suffix], toe = pose['toes' + suffix];
    const footInverse = calc.orientation.clone().invert();
    const heelOffset = calc.position.clone().sub(a.position).applyQuaternion(footInverse);
    const toeOffset = toe.position.clone().sub(a.position).applyQuaternion(footInverse);
    f.position.copy(hips).z += side * width / 2;
    t.position.copy(f.position).y -= thigh;
    a.position.copy(t.position).y -= shin;
    patella.position.copy(t.position).add(patellaOffset);
    calc.position.copy(a.position).add(heelOffset);
    toe.position.copy(a.position).add(toeOffset);
    for (const name of legNodes) pose[name + suffix].orientation.identity();
  });
  const shoulders = pose.humerus_r.position.clone().lerp(pose.humerus_l.position, 0.5);
  around(pose, upperNodes, shoulders, pose.torso.orientation.clone().invert());
  pose.pelvis.orientation.identity();
  // Stand in place with the right ankle registered to the modeled support foot.
  translate(pose, new Vector3(0, 0.072 * scale, 0).sub(pose.talus.position));
  return pose;
}

function serialized(pose, clipId, phase) {
  const segments = {};
  for (const [key, segment] of Object.entries(pose)) {
    segments[key] = { position: segment.position.toArray(), rotation: segment.orientation.toArray() };
  }
  return { segments, clipId, phase: phase * 100, heading: 1, side: 'right', cameraOriginM: [0, 0, 0] };
}

function fixArmSegments(pose, localTransforms, scale) {
  if (!localTransforms) return;
  const torso = pose.torso;
  for (const [name, values] of Object.entries(localTransforms)) {
    if (!upperNodes.includes(name) || name === 'torso' || values.length !== 7 || !values.every(Number.isFinite)) {
      throw new RangeError('Invalid torso-relative arm transform for ' + name);
    }
    const segment = pose[name];
    segment.position.fromArray(values).multiplyScalar(scale).applyQuaternion(torso.orientation).add(torso.position);
    segment.orientation.fromArray(values, 3).normalize().premultiply(torso.orientation);
  }
}

export function createPoseSampler(optical, implant) {
  const tracks = new Map();
  for (const archive of [optical, implant]) {
    if (archive) for (const clip of archive.clips) tracks.set(clip.id, { archive, clip });
  }
  const defaultTrack = tracks.get('vicon-subj130-trial-1') ?? tracks.values().next().value;
  if (!defaultTrack) throw new Error('The anatomy stage needs a recorded stride.');

  function sample({ phase = 0, task = 'walk', clipId, profileHeightM, solution, fixedArmSegmentsInTorso, massMarkers } = {}) {
    const track = clipId ? tracks.get(clipId) : defaultTrack;
    if (!track) throw new RangeError('Missing gait recording: ' + clipId);
    const {archive, clip} = track;
    const t = MathUtils.clamp(phase, 0, 1);
    const scale = profileHeightM > 0 && clip.heightM > 0 ? profileHeightM / clip.heightM : 1;
    if (task === 'stance') {
      const pose = makeNeutralStance(archive, clip, scale);
      const effort = Math.sin(t * Math.PI) ** 2;
      const stance = {
        hipFlexionDeg: 9, hipAdductionDeg: 3, kneeFlexionDeg: 10, leftHipFlexionDeg: 8 + 16 * effort,
        leftKneeFlexionDeg: 18 + 25 * effort, ...solution,
      };
      solvePose(pose, stance, 1);
      fixArmSegments(pose, fixedArmSegmentsInTorso, scale);
      return serialized(pose, 'virtual-loaded-stance', t);
    }
    const registerPoint = sourceRegistration(archive, clip, t, scale);
    const pose = readPose(archive, clip, t, registerPoint);
    if (solution) {
      const support = clip.phase === 'stance' ? 1 : supportFraction(t, clip.stanceEndPct / 100);
      const constrainedFeet = solution.ankleDorsiflexionDeg !== undefined && solution.leftAnkleDorsiflexionDeg !== undefined;
      const values = stride.channels.pelvisTiltDeg;
      const time = (t % 1) * values.length, row = Math.floor(time);
      const pelvisPitch = MathUtils.lerp(values[row], values[(row + 1) % values.length], time - row);
      solvePose(pose, solution, support, constrainedFeet, pelvisPitch);
    }
    fixArmSegments(pose, fixedArmSegmentsInTorso, scale);
    const frame = serialized(pose, clip.id, t);
    if (massMarkers) {
      // The OpenSim export is in lab ground coordinates. The gait archive
      // stores the same recording relative to its per-clip origin.
      const origin = new Vector3();
      if (massMarkers.clipOriginGroundM) origin.fromArray(massMarkers.clipOriginGroundM);
      const registerGround = xyz => registerPoint(new Vector3().fromArray(xyz).sub(origin)).toArray();
      frame.massMarkers = {
        body: registerGround(massMarkers.body),
        pelvis: registerGround(massMarkers.pelvis),
      };
    }
    return frame;
  }
  return { sample };
}
