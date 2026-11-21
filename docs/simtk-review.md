# SimTK data for the limb simulator

SimTK’s strongest immediate contribution is measured anatomy and motion for checking the current rigs. The viewer already uses model-derived Arm26 readouts for the elbow and Rajagopal2016 reference motion and readouts for the knee.

## Check knee tracking first

The exported GLB’s contact adjustment translates the kneecap by about 25 mm at 90° and 29 mm at 120°, measured against Rajagopal2016 after registration to the same atlas. These values quantify the effect of the adjustment. Compare both trajectories with measured tibia and patella poses using shared anatomical coordinate systems and report translations and rotations separately. Reserve specimens for evaluation if the fit is adjusted. [Knee fitting code](../lower-limb/tools/build_asset.py)

[Open Knee(s)](https://simtk.org/projects/openknee/) provides MRI and mechanical tests for eight cadaver knees. Its [measurement archive](https://doi.org/10.18735/4e78-1311) includes tibial passive flexion through 90° and patellar tracking under controlled quadriceps loads through 60°. The [testing protocol](https://doi.org/10.1016/j.dib.2021.106824) describes coordinate frames and applied loads. For deeper flexion, [Bologna’s cadaver tibia measurements](https://amsacta.unibo.it/id/eprint/9129/) are CC BY 4.0. [Published in-vivo patellar tracking](https://doi.org/10.1002/jor.20554) adds a weight-bearing comparison. Check the specific Open Knee(s) package’s terms before redistribution.

## Check elbow muscle deformation

[OpenArm 2.0](https://simtk.org/frs/?group_id=1617) supplies CC BY 4.0 volumetric ultrasound at 0°, 30°, 60°, and 90° under different loads. Register scans to the humerus and compare biceps cross-sections at matching angles and loads. Track which labels are manually drawn or model-derived. The scans benchmark the biceps. Deep-flexion tendon–fascia contact needs different measurements. [Study](https://laurahallock.org/publication/nozikhallock2019embc/)

## Add movements

- **Forearm rotation.** The Stanford–VA upper-limb model supplies forearm kinematics and muscle paths under BSD 3-Clause with a citation requirement. Compare biceps leverage across forearm orientations with published measurements. The MoBL-ARMS release requires a separate licence for commercial use. [Stanford–VA](https://simtk.org/frs/?group_id=324) · [MoBL-ARMS](https://simtk.org/frs/?group_id=657) · [Measured elbow leverage](https://pubmed.ncbi.nlm.nih.gov/7775488/)
- **Squat and sit-to-stand.** OpenCap’s video-free lab-validation download is listed under Apache 2.0. It provides optical motion-capture and force-plate trials from ten adults for checking a multi-joint rig with moving hips and ankles. [Download](https://simtk.org/frs/?group_id=2385) · [Study](https://doi.org/10.1371/journal.pcbi.1011462)
- **Nerve glide.** Published ultrasound can check excursion at a defined site and pose. Tension requires additional force–extension measurements, while pain requires a clinical endpoint. [Median nerve excursion](https://pubmed.ncbi.nlm.nih.gov/14580833/) · [Ulnar nerve strain and tension](https://pubmed.ncbi.nlm.nih.gov/40327207/)
