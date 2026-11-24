import './modeling.css';
import outerHip from './media/outer-hip-mechanics.webp';
import bodyMesh from './media/gpu-body-mesh.webp';

export function mountSoftTissueModel(root) {
  root.innerHTML = `<article class="modeling-page">
    <div class="modeling-copy">
      <section>
        <h2>Muscle fibres generate force</h2>
        <p>Each muscle element has a local fibre direction. Activation generates tension along those fibres. The muscle bulges as it shortens and retains nearly the same volume.</p>
      </section>
      <section>
        <h2>Tendons transfer load to bone</h2>
        <p>Anatomical attachment sites anchor the tendons. Tendons, ligaments and fascia each have their own stiffness and stretch under load.</p>
      </section>
      <section>
        <h2>Contact governs sliding</h2>
        <p>Contact forces keep surfaces from crossing and friction resists sliding. At the outer hip, the iliotibial band glides over the bursa as the leg moves.</p>
      </section>
      <p class="modeling-links"><a href="#tissue">Explore sliding at the hip</a><a href="https://doi.org/10.1016/j.jbiomech.2004.04.009" target="_blank" rel="noopener">Muscle model reference</a></p>
    </div>
    <figure class="modeling-figure">
      <img src="${outerHip}" width="675" height="498" alt="Outer hip atlas with the iliotibial band over the gluteus maximus.">
      <figcaption>Outer hip anatomy from <a href="https://anatomytool.org/content/open3dmodel-lower-limb-english-labels" target="_blank" rel="noopener">Open3DModel</a>, adapted under <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener">CC BY-SA 4.0</a>.</figcaption>
    </figure>
  </article>`;
}

export function mountGpuBodyModel(root) {
  root.innerHTML = `<article class="modeling-page">
    <div class="modeling-copy">
      <section>
        <h2>Fine mesh around joints</h2>
        <p>Tetrahedral elements fill the tissue volumes. Smaller elements resolve steep changes in load near attachments and contact surfaces. Mesh refinement checks whether strain and pressure estimates converge.</p>
      </section>
      <section>
        <h2>Parallel physics on the GPU</h2>
        <p>GPU kernels compute element forces and detect contacts in parallel. Each time step updates tissue deformation and skeletal motion together.</p>
      </section>
      <section class="modeling-estimate" aria-labelledby="modeling-estimate-title">
        <h2 id="modeling-estimate-title">Estimated time to simulate one second of gait</h2>
        <dl>
          <div><dt>8-core CPU</dt><dd>~1.5 days</dd></div>
          <div><dt>1 NVIDIA B300</dt><dd>~180 seconds</dd></div>
        </dl>
        <p>Extrapolated from a published contact test to 1.9 million elements and 2.5 μs time steps.</p>
        <details class="modeling-assumptions">
          <summary>Estimate assumptions and sources</summary>
          <p>One second contains 400,000 time steps. The published test presses a rigid hemispherical punch into a deformable block. It reports 316.8 ms per step on eight CPU cores in LS-DYNA (FP64) and 3.378 ms on an L40 GPU with a different solver (FP32).</p>
          <p>Based on peak memory bandwidth, the projection assumes 3.75× throughput from L40 to H100 and 2× from H100 to B300. Whole-body timing depends on the time step and the complexity of tissue contacts.</p>
          <p class="modeling-links"><a href="https://www.mdpi.com/2504-4494/10/6/197" target="_blank" rel="noopener">Contact test, section 6</a><a href="https://www.nvidia.com/en-us/data-center/l40/" target="_blank" rel="noopener">L40 specs</a><a href="https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era" target="_blank" rel="noopener">H100 and B300 bandwidth</a></p>
        </details>
      </section>
    </div>
    <figure class="modeling-figure">
      <div class="modeling-mesh">
        <div class="modeling-mesh-stage"></div>
        <img src="${bodyMesh}" width="1142" height="758" alt="Triangular surface mesh of the BodyParts3D walking atlas at 27% of a recorded stride.">
      </div>
      <figcaption>Walking atlas surface mesh in a <a href="https://doi.org/10.1038/s41597-023-02767-y" target="_blank" rel="noopener">recorded stride</a>. Anatomy from <a href="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html" target="_blank" rel="noopener">BodyParts3D</a> © DBCLS, <a href="https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en" target="_blank" rel="noopener">CC BY-SA 2.1 Japan</a>.</figcaption>
    </figure>
  </article>`;

  const host = root.querySelector('.modeling-mesh-stage');
  let stage, disposed = false, active = true;

  import('./body-stage.js').then(async ({createBodyStage}) => {
    if (disposed) return;
    const view = await createBodyStage(host, {
      clipId: 'vicon-subj130-trial-1',
      phase: .27,
      layer: 'surface',
      wireframe: true,
    });
    if (disposed || view.mode !== 'atlas') {
      view.dispose();
      return;
    }
    stage = view;
    stage.setActive(active);
    host.dataset.ready = '';
  }).catch(error => console.warn('Displaying the walking atlas still:', error.message));

  return {
    setActive(value) {
      active = value;
      stage?.setActive(value);
    },
    dispose() {
      disposed = true;
      stage?.dispose();
    },
  };
}
