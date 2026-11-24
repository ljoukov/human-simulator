import {
  Bone, Group, Matrix4, MeshBasicMaterial, MeshStandardMaterial, Quaternion, SkinnedMesh, Vector3,
  type Material, type Object3D,
} from 'three';
import { clone } from 'three/addons/utils/SkeletonUtils.js';
import type { BodyFrame } from './body-scene.ts';

// Source landmarks in the BodyParts3D 20110915 atlas, in the exported +X
// anterior, +Y superior, +Z anatomical right coordinate convention.
const vec = (x = 0, y = 0, z = 0) => new Vector3(x, y, z);
const right = vec(0, 0, 1);
const up = vec(0, 1, 0);
const midpoint = (a: Vector3, b: Vector3) => a.clone().lerp(b, .5);
const delta = (a: Vector3, b: Vector3) => a.clone().sub(b);
const sides = ['right', 'left'] as const;
const source: Record<string, Vector3> = {head: vec(-.025, 1.84, 0)};
for (const side of sides) {
  const s = side === 'right' ? 1 : -1;
  source[side+'Hip'] = vec(-.042, .007, s*.203);
  source[side+'Knee'] = vec(-.090, -1.025, s*.202);
  source[side+'Ankle'] = vec(-.110, -1.895, s*.190);
  source[side+'Shoulder'] = vec(-.085, 1.270, s*.400);
  source[side+'Elbow'] = vec(-.105, .575, s*.540);
  source[side+'Wrist'] = vec(.002, .005, s*.623);
  source[side+'Palm'] = vec(.034, -.132, s*.656);
  source[side+'Toe'] = vec(.14, -2.006, s*.213);
}

function basis(vertical: Vector3, transverse: Vector3) {
  const y = vertical.clone().normalize();
  const x = vec().crossVectors(y, transverse);
  if (x.lengthSq() < 1e-8) x.crossVectors(y, vec(1, 0, 0));
  x.normalize();
  return new Matrix4().makeBasis(x, y, vec().crossVectors(x, y).normalize());
}

// Index-minus-little proximal phalanx centroids identify the radial direction;
// the third metacarpal identifies the direction from the wrist into the palm.
// Each axis is measured in its own atlas' coordinate system. The left radial
// axis points opposite the right in rest space.
const sourceRadial = { right: vec(-.00860, -.03697, .16038), left: vec(-.00832, -.03713, -.16055) };
const modelRadial = { right: vec(.00620, -.02042, .05773), left: vec(.00620, -.02042, -.05773) };
const modelPalm = { right: vec(-.00597, -.05450, -.00025), left: vec(-.00597, -.05450, .00025) };

function segmentFrames(
  p: Record<string, Vector3>, pelvisUp: Vector3, torsoUp: Vector3,
  footAxes: Record<string, Matrix4>, toeAxes: Record<string, Matrix4>, radial: Record<string, Vector3>,
  upperRadial: Record<string, Vector3>,
) {
  const frames: Record<string, {anchor: Vector3; axes: Matrix4; length: number; width: number}> = {};
  const hip = midpoint(p.rightHip, p.leftHip);
  const shoulder = midpoint(p.rightShoulder, p.leftShoulder);
  const pelvisRight = delta(p.rightHip, p.leftHip);
  const shouldersRight = delta(p.rightShoulder, p.leftShoulder);
  const trunkLength = shoulder.distanceTo(hip);
  const stature = (p.rightHip.distanceTo(p.rightKnee) + p.rightKnee.distanceTo(p.rightAnkle)
    + p.leftHip.distanceTo(p.leftKnee) + p.leftKnee.distanceTo(p.leftAnkle)) / 2;
  function frame(key: string, anchor: Vector3, axes: Matrix4, length: number, width: number) {
    frames[key] = {anchor, axes, length, width};
  }
  frame('pelvis', hip, basis(pelvisUp, pelvisRight), trunkLength, pelvisRight.length());
  frame('torso', shoulder, basis(torsoUp, shouldersRight), trunkLength, shouldersRight.length());
  frame('head', p.head, basis(torsoUp, shouldersRight), stature, stature);
  for (const side of sides) {
    const point = (key: string) => p[side+key];
    function span(key: string, proximal: string, distal: string, transverse: Vector3, girth: number) {
      const axis = delta(point(proximal), point(distal));
      frame(side+key, point(proximal), basis(axis, transverse), axis.length(), girth);
    }
    // The two atlases have different femur:tibia proportions. Size the
    // transverse knee surfaces from the femur on both sides of that joint.
    // Scaling by the entire leg widened the femoral condyles and thighs.
    const femurLength = point('Hip').distanceTo(point('Knee'));
    span('Thigh', 'Hip', 'Knee', pelvisRight, femurLength);
    span('Shin', 'Knee', 'Ankle', pelvisRight, femurLength);
    span('Arm', 'Shoulder', 'Elbow', shouldersRight, stature);
    const elbowAxis = delta(point('Elbow'), point('Wrist'));
    const forearm = basis(elbowAxis, upperRadial[side]);
    const distal = basis(elbowAxis, radial[side]);
    const halfway = new Quaternion().setFromRotationMatrix(forearm)
      .slerp(new Quaternion().setFromRotationMatrix(distal), .5);
    frame(side+'Forearm', point('Elbow'), forearm, elbowAxis.length(), stature);
    frame(side+'ForearmMid', point('Elbow'), new Matrix4().makeRotationFromQuaternion(halfway), elbowAxis.length(), stature);
    frame(side+'ForearmDistal', point('Elbow'), distal, elbowAxis.length(), stature);
    frame(side+'Hand', point('Wrist'), basis(delta(point('Wrist'), point('Palm')), radial[side]), stature, stature);
    frame(side+'Foot', point('Ankle'), footAxes[side], stature, stature);
    frame(side+'Toe', point('Toe'), toeAxes[side], stature, stature);
  }
  return frames;
}

const identityAxes = {right: new Matrix4(), left: new Matrix4()};
const neutral = segmentFrames(source, up, up, identityAxes, identityAxes, sourceRadial, sourceRadial);
const inverseBind = Object.fromEntries(Object.entries(neutral).map(([name, part]) => [
  name, part.axes.clone().transpose().multiply(new Matrix4().makeTranslation(...part.anchor.clone().negate())),
]));

function makeMuscleMaterial() {
  const material = new MeshStandardMaterial({vertexColors: true, roughness: .72});
  material.onBeforeCompile = shader => {
    shader.vertexShader = 'varying vec2 atlasFiber;\n' + shader.vertexShader
      .replace('#include <uv_vertex>', '#include <uv_vertex>\natlasFiber = uv;');
    shader.fragmentShader = 'varying vec2 atlasFiber;\n' + shader.fragmentShader
      .replace('#include <color_fragment>', `#include <color_fragment>
        float phase = atlasFiber.x * 680.0 + 1.8 * sin(atlasFiber.y * 15.0);
        float detail = 1.0 - smoothstep(0.5, 3.0, fwidth(phase));
        diffuseColor.rgb *= 1.0 + detail * .052 * sin(phase);
      `);
  };
  return material;
}

export function createWalkingAtlas(template: Object3D, {wireframe = false} = {}) {
  const object = clone(template) as Group;
  const bones = new Map<string, Bone>();
  const surfaces: Record<string, SkinnedMesh> = {};
  const materials = new Set<Material>();
  const armatures = new Set<SkinnedMesh['skeleton']>();
  object.traverse(node => {
    if (node instanceof Bone) {
      node.matrixAutoUpdate = false;
      bones.set(node.name, node);
    } else if (node instanceof SkinnedMesh) {
      surfaces[node.name] = node;
      armatures.add(node.skeleton);
      node.frustumCulled = false;
      const material = node.name === 'Muscles' ? makeMuscleMaterial() :
        new MeshStandardMaterial({vertexColors: true, roughness: node.name === 'Skin' ? .80 : .78});
      if (wireframe) {
        material.color.setHex(0x39565b);
        material.vertexColors = false;
        material.polygonOffset = true;
        material.polygonOffsetFactor = 1;
        material.polygonOffsetUnits = 1;
      }
      node.material = material;
      materials.add(material);
    }
  });
  if (!surfaces.Muscles || !surfaces.Bones || !Object.keys(neutral).every(name => bones.has(name))) {
    throw new Error('The walking atlas is missing a required anatomical surface or segment.');
  }
  if (wireframe) {
    // The shaded base occludes back edges. The overlay uses the same skinned triangles.
    const lineMaterial = new MeshBasicMaterial({
      color: 0xa0c0ba, wireframe: true, transparent: true, opacity: .55, depthWrite: false,
    });
    materials.add(lineMaterial);
    for (const surface of Object.values(surfaces)) {
      const edges = new SkinnedMesh(surface.geometry, lineMaterial);
      edges.bindMode = surface.bindMode;
      edges.bind(surface.skeleton, surface.bindMatrix);
      edges.frustumCulled = false;
      surface.add(edges);
    }
  }
  const matrix = new Matrix4();
  const q = new Quaternion();
  const scale = vec();
  const p: Record<string, Vector3> = Object.fromEntries(Object.keys(source).map(k => [k, vec()]));
  const footAxes = {right: new Matrix4(), left: new Matrix4()};
  const toeAxes = {right: new Matrix4(), left: new Matrix4()};
  const radial = {right: vec(), left: vec()};
  const upperRadial = {right: vec(), left: vec()};
  const shoulderCenter = vec();
  const torsoUp = vec();
  const pelvisUp = vec();
  const ankleToToe = vec();
  const ankleRegistration = vec();

  return {
    object,
    setLayers(muscles: boolean, skeleton: boolean, skin: boolean) {
      surfaces.Muscles.visible = muscles && !skin;
      surfaces.Bones.visible = skeleton && !skin;
      if (surfaces.Skin) surfaces.Skin.visible = skin;
    },
    setFrame(frame: BodyFrame) {
      const poses = frame.segments;
      if (!poses.torso || !poses.pelvis) return;
      torsoUp.copy(up).applyQuaternion(q.set(...poses.torso.rotation));
      pelvisUp.copy(up).applyQuaternion(q.set(...poses.pelvis.rotation));
      for (const side of sides) {
        const suffix = side === 'right' ? '' : '_l';
        const arm = side === 'right' ? '_r' : '_l';
        const names = {Hip:'femur'+suffix, Knee:'tibia'+suffix, Ankle:'talus'+suffix,
          Shoulder:'humerus'+arm, Elbow:'ulna'+arm, Wrist:'hand'+arm};
        for (const [joint, segment] of Object.entries(names)) {
          if (!poses[segment]) return;
          p[side+joint].set(...poses[segment].position);
        }
        upperRadial[side].copy(modelRadial[side]).applyQuaternion(q.set(...poses[names.Shoulder].rotation));
        const hand = poses[names.Wrist];
        q.set(...hand.rotation);
        p[side+'Palm'].copy(modelPalm[side]).applyQuaternion(q).add(p[side+'Wrist']);
        radial[side].copy(modelRadial[side]).applyQuaternion(q);
        const foot = poses['calcaneus'+suffix];
        const toe = poses['toes'+suffix];
        if (!foot || !toe) return;
        footAxes[side].makeRotationFromQuaternion(q.set(...foot.rotation));
        // Keep the metatarsophalangeal joint attached to the atlas foot as
        // the source toe frame articulates independently around its own origin.
        const legScale = (p[side+'Hip'].distanceTo(p[side+'Knee']) + p[side+'Knee'].distanceTo(p[side+'Ankle'])) / neutral[side+'Foot'].length;
        ankleToToe.copy(source[side+'Toe']).sub(source[side+'Ankle']).multiplyScalar(legScale).applyQuaternion(q);
        // Fit the atlas heel/sole reference to the OpenSim hindfoot surface.
        // This fixed 17 mm offset rotates with the foot through stance and
        // swing. It leaves the recorded leg and its motion above the floor intact.
        ankleRegistration.set(0, .017, 0).applyQuaternion(q);
        p[side+'Toe'].copy(p[side+'Ankle']).add(ankleToToe).add(ankleRegistration);
        toeAxes[side].makeRotationFromQuaternion(q.set(...toe.rotation));
      }
      shoulderCenter.copy(p.rightShoulder).lerp(p.leftShoulder,.5);
      if (frame.headPosition) p.head.set(...frame.headPosition);
      else {
        const bodyScale = (p.rightHip.distanceTo(p.rightKnee)+p.rightKnee.distanceTo(p.rightAnkle)) / neutral.head.length;
        p.head.copy(source.head).sub(midpoint(source.rightShoulder, source.leftShoulder))
          .multiplyScalar(bodyScale).applyQuaternion(q.set(...poses.torso.rotation)).add(shoulderCenter);
      }
      const target = segmentFrames(p, pelvisUp, torsoUp, footAxes, toeAxes, radial, upperRadial);
      for (const side of sides) {
        const foot = target[side+'Foot'];
        foot.anchor = foot.anchor.clone().add(ankleRegistration.set(0, .017, 0).applyMatrix4(foot.axes));
      }
      for (const [name, bone] of bones) {
        const next = target[name], from = neutral[name];
        if (!next || !from) continue;
        const height = next.length/from.length, width = next.width/from.width;
        scale.set(Math.sqrt(height*width), height, width);
        bone.matrix.makeTranslation(...next.anchor).multiply(next.axes)
          .multiply(matrix.makeScale(...scale)).multiply(inverseBind[name]);
        bone.matrixWorldNeedsUpdate = true;
      }
    },
    dispose() {
      materials.forEach(material => material.dispose());
      armatures.forEach(armature => armature.dispose());
    },
  };
}
