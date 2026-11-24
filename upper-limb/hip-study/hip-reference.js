import * as THREE from 'three';

const REFERENCE_RIG = 'deep-reference-mask';
const ORIGIN = new THREE.Vector3();

// Each mask contains the frontmost structure and its depth, including static
// anatomy as occluders. Only the reference's exposed moving tissues get ink.
function maskTarget() {
  const target = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  });
  target.depthTexture = new THREE.DepthTexture(1, 1, THREE.UnsignedIntType);
  return target;
}

function maskMaterial(id, gluteal, layers) {
  const material = new THREE.MeshBasicMaterial({side:THREE.DoubleSide, toneMapped:false});
  material.customProgramCacheKey = () => 'hip-reference-mask/' + Number(gluteal);
  material.onBeforeCompile = shader => {
    shader.uniforms.structure = {value:id / 255};
    shader.uniforms.showBody = layers.showBody;
    shader.uniforms.showTendon = layers.showTendon;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float hipInsertion; varying float insertion;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\ninsertion = hipInsertion;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float structure,showBody,showTendon; varying float insertion;')
      .replace('#include <opaque_fragment>', (gluteal ? 'if ((showBody < .5 && insertion < .5) || (showTendon < .5 && insertion >= .5)) discard;\n' : '')
        + 'gl_FragColor = vec4(structure,0.,0.,1.); return;');
  };
  return material;
}

const fragmentShader = `
uniform sampler2D currentId, referenceId, currentDepth, referenceDepth;
uniform vec2 stepUv;
uniform vec3 ink;
uniform float depthRange;
varying vec2 coord;

float shape(sampler2D mask, vec2 at) {
  return floor(texture2D(mask,at).r * 255. + .5);
}
float edge(sampler2D mask, float center) {
  float nearX = abs(center-shape(mask,coord+vec2(stepUv.x,0.))) + abs(center-shape(mask,coord-vec2(stepUv.x,0.)));
  float nearY = abs(center-shape(mask,coord+vec2(0.,stepUv.y))) + abs(center-shape(mask,coord-vec2(0.,stepUv.y)));
  return step(.5,nearX+nearY);
}
void main() {
  float ref = shape(referenceId,coord);
  if (ref < .5) discard;
  vec4 selected = texture2D(currentId,coord);
  float cur = floor(selected.r*255.+.5);
  float different = step(.5,abs(ref-cur));
  float gap = (texture2D(referenceDepth,coord).r-texture2D(currentDepth,coord).r)*depthRange;
  float depthDifference = smoothstep(.0005,.004,abs(gap));
  float contour = edge(referenceId,ref);
  float matchingContour = (1.-different)*edge(currentId,cur);
  float strength = max(different,depthDifference*(1.-.8*matchingContour));
  float fill = mix(.22,.09,selected.a);
  float opacity = strength * (contour*mix(.95,.66,smoothstep(.001,.010,gap)) + fill*different);
  if (opacity < .012) discard;
  gl_FragColor = vec4(ink,min(opacity,.96));
  #include <colorspace_fragment>
}`;

// Allocate the secondary atlas and render targets only on the first request.
export function makeHipReference(surfaces, skeleton, deformation, layerUniforms) {
  const root = new THREE.Group();
  root.matrixAutoUpdate = false;
  const maskScene = new THREE.Scene();
  maskScene.add(root);
  const masks = surfaces.map((source,i) => {
    const {label,layer,regionGroup,glutealTendon} = source.userData;
    const moving = label === 'Femur' || label === 'Art cart of femur head' || glutealTendon ||
      regionGroup === 'rotators' || layer === 'bursae';
    const mask = new THREE.SkinnedMesh(source.geometry.clone(),maskMaterial(moving ? i+1 : 0,glutealTendon,layerUniforms));
    mask.bind(skeleton,source.bindMatrix);
    mask.frustumCulled = false;
    deformation.attach(mask,label,REFERENCE_RIG);
    root.add(mask);
    return mask;
  });
  const current = maskTarget(), reference = maskTarget();
  const uniforms = {
    currentId:{value:current.texture}, referenceId:{value:reference.texture},
    currentDepth:{value:current.depthTexture}, referenceDepth:{value:reference.depthTexture},
    stepUv:{value:new THREE.Vector2()}, depthRange:{value:1}, ink:{value:new THREE.Color('#62f1df')},
  };
  const composite = new THREE.ShaderMaterial({
    uniforms, transparent:true, depthTest:false, depthWrite:false, toneMapped:false,
    vertexShader:'varying vec2 coord; void main(){ coord = uv; gl_Position = vec4(position.xy,0.,1.); }',
    fragmentShader,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2,2),composite);
  quad.frustumCulled = false;
  const overlay = new THREE.Scene();
  overlay.add(quad);
  const size = new THREE.Vector2(), clear = new THREE.Color();

  function draw(renderer, camera, pelvis, currentPose, referencePose) {
    if (currentPose.angleTo(referencePose) < 1e-6) return;
    renderer.getDrawingBufferSize(size);
    const pixelRatio = Math.min(renderer.getPixelRatio(),1.5);
    const w = Math.max(1,Math.round(size.x/renderer.getPixelRatio()*pixelRatio));
    const h = Math.max(1,Math.round(size.y/renderer.getPixelRatio()*pixelRatio));
    current.setSize(w,h);
    reference.setSize(w,h);
    uniforms.stepUv.value.set(1.45*pixelRatio/w,1.45*pixelRatio/h);
    uniforms.depthRange.value = camera.far-camera.near;
    root.matrix.copy(pelvis.matrixWorld);
    masks.forEach((mask,i) => {mask.visible = surfaces[i].visible;});
    const destination = renderer.getRenderTarget();
    const autoClear = renderer.autoClear;
    const clearAlpha = renderer.getClearAlpha();
    renderer.getClearColor(clear);
    try {
      renderer.setClearColor(0,0);
      renderer.autoClear = true;
      for (const [target,pose] of [[current,currentPose],[reference,referencePose]]) {
        deformation.setPose(pose,ORIGIN,REFERENCE_RIG);
        renderer.setRenderTarget(target);
        renderer.render(maskScene,camera);
      }
      renderer.setRenderTarget(destination);
      renderer.autoClear = false;
      renderer.render(overlay,camera);
    } finally {
      renderer.autoClear = autoClear;
      renderer.setClearColor(clear,clearAlpha);
      renderer.setRenderTarget(destination);
    }
  }
  function dispose() {
    for (const mask of masks) {
      mask.material.dispose();
      mask.geometry.dispose();
    }
    current.dispose();
    reference.dispose();
    quad.geometry.dispose();
    composite.dispose();
  }
  return {draw, dispose};
}
