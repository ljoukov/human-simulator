// Start the Vite viewer in upper-limb, then run from any working directory:
//   node hip-study/tools/audit_contact_visibility.mjs > /tmp/hip-visibility.json
// Requires Playwright with an installed Chromium browser. For a separate installation
// set PLAYWRIGHT_MODULE to its package path and CHROMIUM_EXECUTABLE to a browser binary.
// HIP_STUDY_URL selects the Vite origin (default http://127.0.0.1:5180).
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const local = relative => new URL(relative, import.meta.url);
const origin = new URL(process.env.HIP_STUDY_URL || 'http://127.0.0.1:5180').origin;
const input = JSON.parse(await readFile(local('../data/motion.json'), 'utf8'));
const size = { width: 800, height: 550 };
const views = [{ azimuthDeg: 26, elevationDeg: 10 }, { azimuthDeg: -25, elevationDeg: 5 }];
const frozen = JSON.parse(await readFile(local('../data/hip-visibility-reference.json'), 'utf8'));
const packedWeights = Buffer.from(frozen.weightsBase64, 'base64');
const legacyWeights = Array.from({ length: frozen.vertices }, (_, i) => packedWeights.readUInt16LE(i * 2) / 65535);
const hashes = { originalAttachments: frozen.originalAttachmentsSha256 };
for (const file of ['../data/hip-visibility-reference.json', '../contact/contact.c', '../data/motion.json', '../data/hip-attachments.bin', '../data/hip-contact.bin', '../../upper-limb/hip-study/assets/hip-atlas.glb', '../../upper-limb/hip-study/assets/hip-contact.wasm']) {
  hashes[file.split('/').at(-1)] = createHash('sha256').update(await readFile(local(file))).digest('hex');
}

if (hashes['hip-atlas.glb'] !== frozen.originalAtlasSha256) {
  throw Error('Reference attachment weights require the pinned hip atlas.');
}
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_EXECUTABLE || undefined,
  args: process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-unsafe-swiftshader'],
});

try {
  const page = await browser.newPage({ viewport: size });
  await page.route('**/__hip_visibility_audit__', route => route.fulfill({
    contentType: 'text/html', body: '<body style="margin:0;height:100vh"><main style="width:100%;height:100%"></main></body>',
  }));
  await page.goto(origin + '/__hip_visibility_audit__');
  await page.evaluate(async ({ legacyWeights, legacyName }) => {
    // Resolve the exact Three.js module used by Vite so its public render hook
    // sees the study scene. Existing materials and GPU skinning run in both passes.
    const source = await (await fetch('/hip-study/hip.js')).text();
    const threePath = source.match(/import\s+\*\s+as\s+THREE\s+from\s+["']([^"']+)/)?.[1];
    if (!threePath) throw Error('Could not resolve Three.js. Run the audit against the Vite dev server.');
    const THREE = await import(threePath);
    const { createHipScene } = await import('/hip-study/hip.js');
    let capture;
    const previous = THREE.SkinnedMesh.prototype.onBeforeRender;
    THREE.SkinnedMesh.prototype.onBeforeRender = function (renderer, scene, camera, ...rest) {
      if (this.userData.layer === 'band') capture = { renderer, scene, camera };
      previous.call(this, renderer, scene, camera, ...rest);
    };
    const hip = await createHipScene(document.querySelector('main'), { profile: 'surface' });
    hip.setLayers({ labels: false });
    await new Promise(requestAnimationFrame);
    THREE.SkinnedMesh.prototype.onBeforeRender = previous;
    if (!capture) throw Error('The iliotibial tract did not render.');
    const { renderer, scene, camera } = capture;
    const meshes = [];
    scene.traverse(node => {
      if (node.isSkinnedMesh && node.userData.layer) meshes.push(node);
    });
    const bandId = meshes.findIndex(mesh => mesh.userData.layer === 'band') + 1;
    if (!bandId || meshes.length >= 255) throw Error('Unexpected surface IDs in this atlas.');
    const reference = { value: false };
    const w = innerWidth, h = innerHeight;
    const target = new THREE.WebGLRenderTarget(w, h);
    const layers = ['', ...meshes.map(mesh => mesh.userData.layer)];
    const labels = ['', ...meshes.map(mesh => mesh.userData.label)];
    for (const [i, mesh] of meshes.entries()) {
      const geometry = mesh.geometry;
      let legacy = geometry.attributes.hipArcWeight;
      // The updated TFL insertion changes its attachment field as well as its
      // relation to the fascia. All other visible muscles retain their weights.
      if (mesh.userData.label === legacyName) {
        if (legacyWeights.length !== geometry.attributes.position.count) {
          throw Error('Mismatched original TFL vertex field.');
        }
        legacy = new THREE.Float32BufferAttribute(legacyWeights, 1);
      }
      geometry.setAttribute('auditLegacyWeight', legacy);
      const material = mesh.material;
      const prepare = material.onBeforeCompile.bind(material);
      const cacheKey = material.customProgramCacheKey();
      material.customProgramCacheKey = () => cacheKey + '/visibility-audit';
      material.onBeforeCompile = (shader, gl) => {
        prepare(shader, gl);
        shader.uniforms.auditReference = reference;
        shader.uniforms.auditId = { value: (i + 1) / 255 };
        // Reproduce the original per-vertex spherical arc using the retained
        // atlas positions, attachment weights and pose uniforms. The other pass
        // uses the active deformation implementation unchanged.
        shader.vertexShader = 'attribute float auditLegacyWeight; uniform bool auditReference;\n' + shader.vertexShader.replace(
          '#include <project_vertex>',
          'if (auditReference) transformed = hipArc(position,auditLegacyWeight) + auditLegacyWeight*hipTranslation;\n#include <project_vertex>',
        );
        shader.fragmentShader = 'uniform float auditId;\n' + shader.fragmentShader.replace(
          '#include <alphatest_fragment>',
          '#include <alphatest_fragment>\n gl_FragColor = vec4(auditId,0.,0.,1.); return;',
        );
      };
      material.needsUpdate = true;
    }
    const initialVisibility = meshes.map(mesh => mesh.visible && mesh.material.opacity >= .5);
    const bytes = new Uint8Array(4 * w * h);
    function readIds(bandOnly) {
      meshes.forEach((mesh, i) => { mesh.visible = initialVisibility[i] && (!bandOnly || i + 1 === bandId); });
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(target, 0, 0, w, h, bytes);
      const ids = new Uint8Array(w * h);
      for (let i = 0; i < ids.length; i++) ids[i] = bytes[i * 4];
      return ids;
    }
    const queue = new Int32Array(w * h);
    function score(useOriginal) {
      reference.value = useOriginal;
      const all = readIds(false), band = readIds(true);
      const muscle = new Uint8Array(w * h);
      let footprint = 0, count = 0;
      const byStructure = {};
      for (let i = 0; i < band.length; i++) {
        if (band[i] !== bandId) continue;
        footprint++;
        muscle[i] = Number(layers[all[i]] === 'muscle');
      }
      // Exclude components touching the footprint boundary. Legitimate adjacent
      // muscles can occlude the edge; interior islands reproduce the reported defect.
      for (let i = 0; i < muscle.length; i++) {
        if (!muscle[i]) continue;
        let touchesEdge = false, length = 1;
        queue[0] = i;
        muscle[i] = 0;
        for (let j = 0; j < length; j++) {
          const pixel = queue[j];
          for (const next of [pixel - 1, pixel + 1, pixel - w, pixel + w]) {
            if (next < 0 || next >= band.length || band[next] !== bandId) {
              touchesEdge = true;
            } else if (muscle[next]) {
              muscle[next] = 0;
              queue[length++] = next;
            }
          }
        }
        if (touchesEdge || length < 2) continue;
        count += length;
        for (let j = 0; j < length; j++) {
          const name = labels[all[queue[j]]];
          byStructure[name] = (byStructure[name] || 0) + 1;
        }
      }
      return { footprint, count, byStructure };
    }
    window.auditClip = (clip, view) => {
      hip.setFrame({ ...clip.frames[0], side: clip.side });
      hip.setView('oblique');
      const lookAt = camera.position.clone().add(camera.getWorldDirection(new THREE.Vector3()));
      const a = view.azimuthDeg * Math.PI / 180, e = view.elevationDeg * Math.PI / 180;
      const x = Math.cos(a) * Math.cos(e) * (clip.side === 'left' ? 1 : -1);
      camera.position.copy(lookAt).add(new THREE.Vector3(x, Math.sin(e), Math.sin(a) * Math.cos(e)));
      camera.lookAt(lookAt);
      const results = [];
      for (const frame of clip.frames) {
        hip.setFrame({ ...frame, side: clip.side });
        results.push({ original: score(true), current: score(false) });
      }
      meshes.forEach((mesh, i) => { mesh.visible = initialVisibility[i]; });
      renderer.setRenderTarget(null);
      return results;
    };
  }, { legacyWeights, legacyName: frozen.surface });
  const result = {
    model: 'Open3D teaching atlas and seven retargeted hip-motion clips',
    method: 'For each pose, render depth-tested opaque anatomical IDs and the projected iliotibial tract alone. Count muscle pixels in four-connected components of at least two pixels fully inside the tract footprint. Repeat with original attachment-based articulation and current surface positions.',
    limitation: 'This view-dependent metric detects enclosed visible intersections. Edge occlusions, subsurface collisions and physiological contact pressures require different measurements.',
    camera: 'Orthographic superficial hip preset. Positive azimuth turns towards anterior from ipsilateral lateral. Elevation is measured above the horizontal. Mirror the transverse axis for left hips.',
    reference: 'Original attachment-based spherical articulation with the uncoupled tensor fasciae latae weights. The active pass uses current rendered positions for all structures.',
    size, hashes, browser: browser.version(), views: [],
  };
  for (const view of views) {
    const clips = [];
    for (const clip of input.clips) {
      const frames = await page.evaluate(([clip, view]) => window.auditClip(clip, view), [clip, view]);
      const summary = { id: clip.id, frames: frames.length };
      for (const mode of ['original', 'current']) {
        const counts = frames.map(frame => frame[mode].count);
        summary[mode] = {
          framesWithIslands: counts.filter(n => n > 0).length,
          sumPixels: counts.reduce((a, b) => a + b, 0),
          maximumPixels: Math.max(...counts),
        };
      }
      if (clip.id === 'bertaux-hoa001') summary.poseAt51Percent = frames[51];
      clips.push(summary);
    }
    result.views.push({ ...view, clips });
  }
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
