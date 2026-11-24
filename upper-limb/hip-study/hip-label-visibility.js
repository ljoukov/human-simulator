import * as THREE from 'three';

// Sample the frontmost atlas structure at low resolution. Picking uses the same
// posed vertices, material cutaways and depth buffer as the displayed anatomy.
export function createLabelVisibility(renderer, scene, camera, surfaces, redraw) {
  const active = {value:false};
  const ids = new Map();
  surfaces.forEach((mesh,i) => {
    ids.set(mesh,i+1);
    const material = mesh.material;
    const before = material.onBeforeCompile.bind(material);
    const key = material.customProgramCacheKey();
    material.customProgramCacheKey = () => key+'/structure-label-pass';
    material.onBeforeCompile = (shader, gl) => {
      before(shader,gl);
      shader.uniforms.labelPass = active;
      shader.uniforms.structureId = {value:(i+1)/255};
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform bool labelPass; uniform float structureId;')
        .replace('#include <alphatest_fragment>', '#include <alphatest_fragment>\nif (labelPass) { gl_FragColor = vec4(structureId,0.,0.,1.); return; }');
    };
  });
  const target = new THREE.WebGLRenderTarget(1,1,{minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter});
  let signature, map, reading = false, disposed = false;
  function refresh(width, height, key) {
    if (reading || key === signature || disposed) return;
    signature = key;
    reading = true;
    const scale = 126 / Math.max(width,height);
    const w = Math.max(2,Math.round(width*scale));
    const h = Math.max(2,Math.round(height*scale));
    const buf = new Uint8Array(w*h*4);
    const previousTarget = renderer.getRenderTarget();
    const shadowUpdate = renderer.shadowMap.autoUpdate;
    const sheets = surfaces.filter(s => s.visible && s.material.opacity < .50);
    target.setSize(w,h);
    try {
      for (const mesh of sheets) mesh.visible = false;
      active.value = true;
      renderer.shadowMap.autoUpdate = false;
      renderer.setRenderTarget(target);
      renderer.render(scene,camera);
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.shadowMap.autoUpdate = shadowUpdate;
      active.value = false;
      for (const mesh of sheets) mesh.visible = true;
    }
    renderer.readRenderTargetPixelsAsync(target,0,0,w,h,buf).then(() => {
      if (!disposed) map = {w,h,buf};
    }).finally(() => {
      reading = false;
      if (disposed) target.dispose();
      else redraw();
    });
  }
  function visiblePoint(mesh, u, v, broaderSearch = false, narrow = false) {
    if (!map || !mesh.visible) return null;
    const {w,h,buf} = map;
    const id = ids.get(mesh);
    const x = Math.round(u*w-.5), y = Math.round((1-v)*h-.5);
    const radius = Math.max(4,Math.round(Math.min(w,h)*(broaderSearch ? .44 : .1)));
    let best, least = (radius+1)**2;
    for (let iy=Math.max(1,y-radius); iy<=Math.min(h-2,y+radius); iy++) {
      for (let ix=Math.max(1,x-radius); ix<=Math.min(w-2,x+radius); ix++) {
        const p = 4*(iy*w+ix);
        if (buf[p] !== id) continue;
        // A thin exposed bursa can occupy a strip less than three pixels wide.
        // Adjacent pixels with its exact structure ID still verify the target.
        const horizontal = buf[p-4] === id && buf[p+4] === id;
        const vertical = buf[p-4*w] === id && buf[p+4*w] === id;
        if (narrow ? !(horizontal || vertical) : !(horizontal && vertical)) continue;
        const d = (x-ix)**2 + (y-iy)**2;
        if (d<least) { least = d; best = [(ix+.5)/w,1-(iy+.5)/h]; }
      }
    }
    return best;
  }
  function rowBounds(v) {
    if (!map) return [1,0];
    const {w,h,buf} = map;
    const y = Math.round((1-v)*h-.5);
    let left = w, right = 0;
    for (let iy=Math.max(0,y-1); iy<=Math.min(h-1,y+1); iy++) {
      for (let ix=0; ix<w; ix++) {
        if (!buf[4*(iy*w+ix)+3]) continue;
        left = Math.min(left,ix);
        right = Math.max(right,ix+1);
      }
    }
    return [left/w,right/w];
  }
  return {refresh,visiblePoint,rowBounds,dispose() { disposed = true; if (!reading) target.dispose(); }};
}
