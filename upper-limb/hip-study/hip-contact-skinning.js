import * as THREE from 'three';

// The solver publishes final welded positions after articulation in the hip frame.
// Expand that field back onto atlas UV seams and transport the material normals.
export function createContactSkin(geometry, weights) {
  const position = geometry.attributes.position;
  const normal = geometry.attributes.normal;
  const indices = geometry.index.array;
  const groups = new Uint32Array(position.count);
  const groupKeys = new Map();
  for (let i=0; i<position.count; i++) {
    const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
    // Merge UV duplicates while retaining the atlas's original creases and
    // section normals at the boundary of the cut iliotibial tract.
    const k = [x,y,z,normal.getX(i),normal.getY(i),normal.getZ(i)].join(',');
    if (!groupKeys.has(k)) groupKeys.set(k,groupKeys.size);
    groups[i] = groupKeys.get(k);
  }
  const solvedPosition = new THREE.Float32BufferAttribute(position.array,3).setUsage(THREE.DynamicDrawUsage);
  const rotation = new THREE.BufferAttribute(new Float32Array(4*position.count),4).setUsage(THREE.DynamicDrawUsage);
  for (let i=0; i<position.count; i++) rotation.setW(i,1);
  geometry.setAttribute('hipContactPosition',solvedPosition);
  geometry.setAttribute('hipContactRotation',rotation);
  const before = Array.from({length:position.count},() => new THREE.Vector3());
  const after = Array.from({length:position.count},() => new THREE.Vector3());
  const baseArea = Array.from({length:groupKeys.size},() => new THREE.Vector3());
  const contactArea = Array.from({length:groupKeys.size},() => new THREE.Vector3());
  const groupRotations = Array.from({length:groupKeys.size},() => new THREE.Quaternion());
  const edgeA = new THREE.Vector3(), edgeB = new THREE.Vector3(), cross = new THREE.Vector3();
  let lastResult, radius = 0;

  function accumulate(vertices, sums, i, j, k) {
    edgeA.subVectors(vertices[j],vertices[i]);
    edgeB.subVectors(vertices[k],vertices[i]);
    cross.crossVectors(edgeA,edgeB);
    sums[groups[i]].add(cross);
    sums[groups[j]].add(cross);
    sums[groups[k]].add(cross);
  }

  function update(result, axis, angle) {
    if (result === lastResult) return radius;
    const {positions,packedToWelded} = result;
    if (packedToWelded.length !== position.count) throw new RangeError('Hip contact binding differs from the atlas.');
    radius = 0;
    for (let i=0; i<position.count; i++) {
      before[i].fromBufferAttribute(position,i).applyAxisAngle(axis,weights[i]*angle);
      after[i].fromArray(positions,3*packedToWelded[i]);
      after[i].toArray(solvedPosition.array,3*i);
      radius = Math.max(radius,after[i].length());
    }
    for (let g=0; g<baseArea.length; g++) {
      baseArea[g].set(0,0,0);
      contactArea[g].set(0,0,0);
    }
    for (let f=0; f<indices.length; f+=3) {
      const i = indices[f], j = indices[f+1], k = indices[f+2];
      accumulate(before,baseArea,i,j,k);
      accumulate(after,contactArea,i,j,k);
    }
    for (let g=0; g<baseArea.length; g++) {
      const a = baseArea[g], b = contactArea[g];
      if (a.lengthSq()>1e-25 && b.lengthSq()>1e-25) groupRotations[g].setFromUnitVectors(a.normalize(),b.normalize());
      else groupRotations[g].identity();
    }
    for (let i=0; i<position.count; i++) groupRotations[groups[i]].toArray(rotation.array,4*i);
    solvedPosition.needsUpdate = rotation.needsUpdate = true;
    lastResult = result;
    return radius;
  }
  return {position:solvedPosition,update};
}
