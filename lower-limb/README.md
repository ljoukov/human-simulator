# Lower-limb simulator

Open the standalone `lower-limb-viewer.html` in Chrome to explore the leg. The viewer has right and mirrored left views, a knee close-up, and independently switchable tissue layers. To run the development viewer, run `npm ci` and `npm run dev -- --port 5178` in [../upper-limb](../upper-limb/) and open `http://127.0.0.1:5178/?region=legs`. Run `npm run build` there to create the hosted bundle and `dist/lower-limb-viewer.html`.

Import [lower-limb.glb](assets/lower-limb.glb) into a glTF 2.0 editor such as Blender. It contains the right leg, named meshes, and a `Knee flexion` clip covering 0–120° over one second. The pelvis and femur are fixed. The tibia and patella use separate motion tracks, and the ankle and foot follow the lower leg.

## Anatomy and motion

The rig adapts the Open3D Project’s right lower limb. It keeps the atlas surfaces, names, and anatomical layers, replaces the source textures with solid colors, and adds joint motion and pose-dependent deformation. The adapted meshes are available under CC BY-SA 4.0. [Source GLB archive](https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip) · [Publisher credits](https://web.archive.org/web/20260728014822id_/https://anatomytool.org/open3dmodel-create) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

The tibial and patellar trajectories are registered to the atlas knee using bony landmarks. A local translation adjustment improves joint alignment after registration. Soft tissues cross the knee using shared material attachments, with local shape and contact corrections baked into the clip. The atlas combines the quadriceps tendon and patellar ligament into one mesh. Its intermediate attachment follows the patella, and its distal insertion follows the tibial tuberosity. [Rajagopal2016 OpenSim model](https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/Rajagopal2016.osim) · [Model study](https://doi.org/10.1109/TBME.2016.2586891) · [Cadaveric measurements of patellar tracking](https://pmc.ncbi.nlm.nih.gov/articles/PMC9483075/)

The mechanics panel shows rectus femoris muscle–tendon path length and signed moment arm sampled from the pinned OpenSim model with its reference hip and ankle pose. Negative moment arms indicate extension. Ankle-load torque is the gravitational moment on an illustrative 40 cm lever. The downloaded geometry and the OpenSim reference calculations have separate sources and assumptions.

## Nerves

Select the sciatic, tibial, common fibular, or femoral nerve in **Reference nerve paths**. The anatomy picker also includes the superficial and deep fibular, saphenous, sural, and plantar nerves. **Isolate** shows the selected structure by itself. To see the common fibular nerve near the fibular head, choose **Lateral** and **Knee detail**.

The nerve layer illustrates anatomical routing during knee motion. Estimating tension would require tracking nerve excursion in an experimental motion, then measuring slack, sliding, and the force–extension relation. Local compression would additionally require validated contact geometry and tissue properties. The viewer’s current outputs are geometry and reference mechanics for education. [In-vivo common fibular nerve course by knee angle](https://pubmed.ncbi.nlm.nih.gov/38510322/)

## Reproduce and audit

From this directory, regenerate the asset with `python tools/build_asset.py path/to/lower-limb-glb.zip assets/lower-limb.glb` using NumPy, SciPy, DracoPy, and libigl. The builder verifies the unpacked source GLB hash before fitting the rig. The pinned reference poses are recorded in [data/knee-motion.json](data/knee-motion.json).

With OpenSim 4.6 installed, run `python tools/sample_mechanics.py path/to/Rajagopal2016.osim` to update [data/leg-mechanics.json](data/leg-mechanics.json). The sampler checks 121 angles from extension to 120° and verifies signed moment arms against the derivative of path length.

After installing the shared viewer dependencies, run `node tools/audit.mjs --step-deg 2.5 --grid-mm 1.5 --self-crossings assets/lower-limb.glb` to sample the exported skinning and morphs. The audit tracks frozen anatomical points across their attachments, checks coincident seam points for separation, and estimates intersections and enclosed volumes. Compare overlap against extension to distinguish atlas overlap from deformation during flexion.

The released GLB has SHA-256 `4c807a0ee9827b38a448f95372e7d1667d2a2245078047c7d2bd0c1daeb94be4`. At 105° in the medial view, all three mid-thigh steps disappeared. On the same sampled material regions, local centerline turns fell from 31° to 4.3° in sartorius, from 38° to 0.6° in gracilis, and from 42° to 2.1° in semitendinosus.

Across 49 poses, mean positive pairwise overlap above neutral fell from 122.84 to 11.18 cm³ across the motion samples compared with the first animated version. These are summed pairwise estimates. The audited material footprints for the extensor mechanism, distal biceps femoris tendon, and Achilles tendon stay in place relative to their insertion bones to 0.001 mm reporting precision. The main tibial and common fibular nerves stay outside the tested bones across 41 poses.

Deep flexion still produces local surface folds near the distal semitendinosus tendon and residual intersections between some tissues. The infrapatellar sensory branch clips the patella in some poses. These measurements characterize the exported reference geometry.
