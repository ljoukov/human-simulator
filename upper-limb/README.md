# Upper-limb simulator

Open the standalone `upper-limb-viewer.html` in Chrome for the interactive model. It embeds the WebGL viewer, anatomy, and reference data. Import [upper-limb.glb](assets/upper-limb.glb) into Blender or another glTF 2.0 editor for the reusable model. The GLB includes the `Elbow flexion` clip. The clip runs from extension to 125° over one second. Mesh names and tissue layers are retained.

From this directory, run `npm ci` and `npm run dev` to work on the viewer. Run `npm run build` to generate both a hosted bundle and `dist/upper-limb-viewer.html`.

The viewer also opens the [lower-limb simulator](../lower-limb/README.md) from the **Legs** tab.

## Anatomy and mechanics

The asset adapts [Open3DModel Upper Limb](https://caskanatomy.info/open3dmodelfiles/upper-limb/upper-limb-glb.zip) by the Open3D Project and university collaborators under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The adaptation retains selected right-arm geometry, adds a fitted elbow axis and pose deformation, and replaces source materials with original solid colors. The adapted GLB uses the same license. [Publisher’s downloads and credits](https://web.archive.org/web/20260728014822id_/https://anatomytool.org/open3dmodel-create)

The biceps readouts come from [OpenSim Arm26](https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Arm26/arm26.osim) under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The path length and moment arm describe the long-head muscle–tendon path in that generic model. External torque comes from gravity on the selected hand load in a static 30 cm lever model. The anatomy rig demonstrates flexion and extension with the shoulder and forearm rotation fixed. Soft-tissue deformation and nerve paths are illustrative.

## Inspect the joint and nerves

Select **Lateral** and **Elbow detail** for a close view of the bending joint. Under **Reference nerve paths**, choose the median, ulnar, radial, or musculocutaneous nerve. The full structure list also includes smaller radial and digital branches. **Isolate** shows the selected structure on its own.

The nerve geometry follows the animation. A mechanical nerve model would track material markers along a sliding, slack-bearing path constrained by surrounding tissue. Marker excursion would be measured in millimetres, segment stretch as strain, and axial tension in newtons after calibrating a force–extension relation. Pressure against a tunnel wall has its own contact parameters and measurements. The ulnar nerve at the cubital tunnel is a tractable calibration case. Clinical symptoms require independent endpoints. [Ulnar nerve marker measurements](https://www.mdpi.com/2076-3417/11/14/6487) · [Direct cadaver tension measurements](https://pubmed.ncbi.nlm.nih.gov/40327207/) · [Cubital tunnel pressure measurements](https://pubmed.ncbi.nlm.nih.gov/9563378/)

## Tendon attachments

The bicipital aponeurosis fans from the distal biceps into the fascia over the proximal forearm flexors. The main distal biceps tendon continues to the radial tuberosity. The distal triceps tendon attaches to the olecranon. The **Medial** and **Elbow detail** controls show the anterior attachments. The **Posterior** view exposes the triceps insertion. [Bicipital aponeurosis dissection](https://pmc.ncbi.nlm.nih.gov/articles/PMC5771790/) · [Radial insertion study](https://doi.org/10.1016/j.jse.2007.05.005) · [Triceps insertion dissection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6340860/)

Run `node tools/audit_attachments.mjs older.glb assets/upper-limb.glb` to check the exported animation. The audit freezes sampled attachment patches at extension, measures direct surface gap, and follows the same material points through flexion. The forearm fascia in the viewer is a local cutaway of the atlas sheet.

In the attachment revision `2cf63894`, 95% of the audited fan-edge points stay within about 0.6 mm of that fascia through the sampled motion. A separate 51-pose scan of source attachment points measured maximum distance change of 0.045 mm for distal biceps–radius and 0.002 mm for triceps–ulna. Contact checks still find deep biceps tendon–brachialis overlap and crossings between the median nerve and the sheet at flexed poses. These checks evaluate the modeled geometry.

## Contact validation

The GLB is a kinematic model with contact corrections baked into the flexion animation. Cross-sections follow the revised muscle routes, anatomical attachment regions follow their named bones, and coincident source vertices move together throughout the pose. An offline projection resolves selected muscle–bone and muscle–muscle penetrations before export.

Inspect intersections in the actual posed GLB. Compare the same anatomical pairs at extension and through flexion to distinguish existing atlas overlap from pose-induced penetration. Keep the bones rigid and report mesh overlap independently of the rendered silhouette. Anatomical attachment interfaces can touch throughout motion.

Run `node tools/audit_contact.mjs --grid-mm 0.75 --angles 0,30,65,90,112.5,125 older.glb assets/upper-limb.glb` after installing the viewer dependencies. The audit evaluates the exported animation and estimates shared volume for named pairs using two orthogonal ray grids in a 130 mm radius around the elbow. Its sums are over pairs, so tissue shared by more than two structures can contribute to more than one pair. Check intermediate poses and whole-mesh volume alongside clearance.

The revision from `7760f83d` to `d3339550` reduced the mean summed excess over neutral by 65.7% across 420 pairs and 151 sampled angles. At 90°, estimated brachialis–humerus overlap went from 16.15 cm³ to about 0.016 cm³ on a 0.5 mm grid. All 4,401 audited coincident rest pairs stayed together to the audit’s 0.001 mm reporting precision. At 125°, a residual flexor carpi radialis–brachialis overlap measures about 2.09 cm³, and the humeral head of pronator teres retains about 59% of its reference mesh volume. These are geometry measurements from the exported rig.

A tissue mechanics model can use volume-preserving muscle elements and finite-thickness tendon geometry with reviewed attachment regions, sliding and deformable contact. The literature includes coupled upper-limb muscle contact and frictional finite element contact. Local validation needs measured tissue shape under the matching motion and load. [Upper-limb volumetric musculature](https://doi.org/10.1109/TVCG.2005.42) · [ArtiSynth rigid and deformable contact](https://artisynth.org/doc/html/modelguide/Ch8.html) · [Continuous contact method](https://ipc-sim.github.io/) · [OpenArm measurements](https://hart.berkeley.edu/datasets.html)

## Regenerate the inputs

With NumPy, DracoPy and libigl installed, run `python tools/build_asset.py path/to/upper-limb-glb.zip assets/upper-limb.glb`. With OpenSim 4.6 installed, run `python tools/sample_arm26.py path/to/arm26.osim` using the pinned Arm26 file above. The sampler checks signed moment arms against finite differences of path length and records the source revision and hash.
