// Measure the same solved IT-band vertices that the anatomy renderer displays.
// The reference stores material coordinates on the contact-corrected neutral band.
const source = new URL('../../hip-study/data/bursa-mechanics.json', import.meta.url);
let cached;

export function createBursaMechanics(data) {
  if (data.version !== 2) throw new Error('Unsupported bursa mechanics data.');
  const {vertices, samples, bandVertexCount} = data.evaluator;
  const delta = new Float64Array(vertices.length * 3);

  function excursionMm(q, positions) {
    if (positions?.length !== 3*bandVertexCount) {
      throw new Error('Bursa measurement requires the full, contact-corrected IT-band position array.');
    }
    for (let i = 0; i < vertices.length; i++) {
      const [id, restX, restY, restZ] = vertices[i];
      const x = positions[3*id], y = positions[3*id + 1], z = positions[3*id + 2];
      // Undo the femur/bursa quaternion to express the deformed band in the
      // neutral bursa frame, then subtract its corrected neutral position.
      const cx = q.y*z-q.z*y, cy = q.z*x-q.x*z, cz = q.x*y-q.y*x;
      delta[3*i]   = x - 2*q.w*cx + 2*(q.y*cz-q.z*cy) - restX;
      delta[3*i+1] = y - 2*q.w*cy + 2*(q.z*cx-q.x*cz) - restY;
      delta[3*i+2] = z - 2*q.w*cz + 2*(q.x*cy-q.y*cx) - restZ;
    }
    let squared = 0;
    for (const [mass,nx,ny,nz,i,j,k,bi,bj,bk] of samples) {
      const dx = bi*delta[3*i]   + bj*delta[3*j]   + bk*delta[3*k];
      const dy = bi*delta[3*i+1] + bj*delta[3*j+1] + bk*delta[3*k+1];
      const dz = bi*delta[3*i+2] + bj*delta[3*j+2] + bk*delta[3*k+2];
      const normal = nx*dx + ny*dy + nz*dz;
      squared += mass*((dx-normal*nx)**2 + (dy-normal*ny)**2 + (dz-normal*nz)**2);
    }
    return 1000*Math.sqrt(squared);
  }
  return { excursionMm, landmarkM:data.landmarkM, label:data.bursa };
}

export function loadBursaMechanics() {
  cached ??= fetch(source).then(async response => {
    if (!response.ok) throw new Error('Bursal reference geometry unavailable.');
    return createBursaMechanics(await response.json());
  }).catch(error => {
    cached = undefined;
    throw error;
  });
  return cached;
}
