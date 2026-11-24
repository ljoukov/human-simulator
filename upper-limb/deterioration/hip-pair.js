import { Color } from 'three';
import { createHipScene } from '../hip-study/hip.js';
import { HIP_PROFILES } from '../hip-study/hip-anatomy.js';

// The two hips share one physical scale. The gait solver supplies one group
// effort and one cable tension, so each value is uniform on its target tissue.
const lowLoad = new Color('#ecd5b7');
const highLoad = new Color('#883632');
export const BAND_SCALE_N = 500;
// Snap to the tract at trochanter height, inside the close camera's crop.
const bandLandmark = {...HIP_PROFILES.surface.anchors.find(a => a.label === 'Iliotibial tract'),
  xyz: [-.145, .849, -.027]};
const abductorLandmark = HIP_PROFILES.overview.anchors.find(a => a.label === 'Gluteus medius muscle');

function configureTint(material, info, uniforms) {
  const band = info.layer === 'band';
  if (!band && !info.glutealTendon) return;
  const current = band ? uniforms.band : uniforms.glutes;
  const prepare = material.onBeforeCompile.bind(material);
  const key = material.customProgramCacheKey();
  material.customProgramCacheKey = () => key + '/gait-tissue-load-v1';
  material.onBeforeCompile = (shader, renderer) => {
    prepare(shader, renderer);
    shader.uniforms.tissueLoad = current;
    shader.uniforms.tissueLow = {value: lowLoad};
    shader.uniforms.tissueHigh = {value: highLoad};
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float tissueLoad; uniform vec3 tissueLow, tissueHigh;')
      .replace('#include <metalnessmap_fragment>', `
        #include <metalnessmap_fragment>
        if (tissueLoad >= 0.) {
          float grain = .75 + .35 * clamp(dot(diffuseColor.rgb, vec3(.2126, .7152, .0722)), 0., 1.);
          diffuseColor.rgb = grain * mix(tissueLow, tissueHigh, tissueLoad);
        }
      `);
  };
}

function fraction(value, scale) {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value / scale)) : -1;
}

/** Matched right-sided dissections. Hosts are ordered reference, adjusted. */
export async function createHipPair(hosts, {tissue = 'glutes'} = {}) {
  if (hosts.length !== 2) throw new Error('Two hip viewports are required.');
  const tints = hosts.map(() => ({band: {value: -1}, glutes: {value: -1}}));
  let scenes;
  let syncing = false;

  function linkView(index, camera) {
    if (!scenes || syncing) return;
    syncing = true;
    try { scenes[1 - index].setCameraPose(camera); }
    finally { syncing = false; }
  }

  const loaded = await Promise.allSettled(hosts.map((host, i) => createHipScene(host, {
    profile: 'overview',
    landmarks: [bandLandmark, abductorLandmark],
    framing: {target: [.006, .029, -.013], zoom: 1.55},
    configureSurface: (material, info) => configureTint(material, info, tints[i]),
    onCameraChange: camera => linkView(i, camera),
  })));
  const failed = loaded.find(s => s.status === 'rejected');
  if (failed) {
    for (const s of loaded) if (s.status === 'fulfilled') s.value.dispose();
    throw failed.reason;
  }
  scenes = loaded.map(s => s.value);
  const frames = [null, null];

  function updateTint(i) {
    const frame = frames[i];
    tints[i].glutes.value = tissue === 'glutes' ? fraction(frame?.abductorUtilizationPct, 100) : -1;
    tints[i].band.value = tissue === 'band' ? fraction(frame?.bandForceN, BAND_SCALE_N) : -1;
  }

  function setTissue(next) {
    if (next !== 'glutes' && next !== 'band') throw new RangeError('Unknown hip tissue ' + next);
    tissue = next;
    scenes.forEach((scene, i) => {
      updateTint(i);
      scene.setLayers({
        bone: true, muscle: true, tendons: true, band: next === 'band', fascia: false,
        cartilage: false, bursae: false, labels: true, loads: false,
        labelTarget: next === 'band' ? bandLandmark.label : abductorLandmark.label,
      });
    });
  }

  function setFrames(reference, adjusted) {
    frames[0] = reference;
    frames[1] = adjusted;
    scenes.forEach((scene, i) => {
      const frame = frames[i];
      updateTint(i);
      // In this right-sided atlas +X points medially. Dropping the left side
      // is a negative rotation of that transverse axis about +Z (anterior).
      scene.setFrame({
        side: 'right',
        phase: frame.phase,
        hipFlexionDeg: frame.hipFlexionDeg,
        hipAdductionDeg: frame.hipAdductionDeg,
        pelvisObliquityDeg: -frame.pelvisDropDeg,
      });
    });
  }

  function reset() {
    for (const scene of scenes) scene.setView('lateral');
  }
  setTissue(tissue);
  reset();
  return {
    setTissue, setFrames, reset,
    dispose() {
      const previous = scenes;
      scenes = null;
      previous?.forEach(scene => scene.dispose());
    },
  };
}
