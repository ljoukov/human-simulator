# Patellar tracking in Open Knee(s) specimen oks001

Optical tracking of one right cadaver knee provides a physical check on the patellar motion in the limb viewer. The source is the Cleveland Clinic Open Knee(s) MRI and joint-mechanics dataset by Chokhandre et al. This derived summary uses its CC BY 4.0 data. [Results](patella-oks001.json) · [Dataset](https://doi.org/10.18735/4e78-1311) · [Protocol](https://doi.org/10.1016/j.dib.2021.106824) · [License](https://creativecommons.org/licenses/by/4.0/)

## Passive flexion

In the optimized passive trial, optical tibiofemoral flexion peaked at 57.78°. Compare the increasing branch at measured 5° intervals from 5° to 55°. The table reports the magnitude of femur-relative patella rotation from the pose at approximately 5°. The calculation restores each optical trial's published position offsets before reconstructing rotation and matches the model by the same tibiofemoral flexion component.

| Tracking source | Rotation at approximately 55° | RMS difference over 10–55° |
| --- | ---: | ---: |
| Open Knee(s) optical measurements | 15.59° | — |
| Rajagopal2016 reference model | 38.17° | 11.49° |
| Exported lower-limb GLB | 37.38° | 11.01° |

Rotation magnitude is independent of a fixed choice of coordinate axes. Both models use the same prescribed patella rotations at the GLB keyframes. The slightly different RMS values reflect interpolation at different sampling intervals. Comparison of translation would require registering the same anatomical landmark in each model and the specimen. The measured patella origin is the midpoint of the digitized medial and lateral margins, while the GLB uses the patella apex. The comparison is geometric and applies to one specimen.

## Controlled quadriceps loads

Across the five commanded poses from 0° to 60°, the change from nominal 20 N to 600 N produced 1.47–3.80 mm of patella origin displacement and 1.59–3.74° of patella rotation. Measured tibiofemoral flexion differed by less than 0.18° within each of these load pairs. Each estimate is the median optical pose from a recorded held-load trial. Neither kinematic model has an applied quadriceps-load input.

## Reproduce

Install `nptdms` and `numpy` in a Python environment and obtain the public specimen archive. Run the audit from the repository root. [oks001 mechanical archive](http://archive.simtk.org/oks/MRI_JointMech/joint_mechanics/joint_mechanics-oks001.zip)

```sh
python knee-loading/tools/audit_patella.py /path/to/joint_mechanics-oks001.zip
```

The audit reads processed TDMS channels and the position offsets from each trial configuration. It keeps the last full set at nominal 30° after an earlier interrupted series. The report records the SHA-256 of each measurement and configuration entry and of both model inputs. An independent calculation from the individual bone sensor transforms checked the coordinate conversion in the passive trial and two held-load trials. Its 95th-percentile disagreement with the processed channels was at most 0.016 mm and 0.007°. These figures check numerical agreement between conversion methods. [SimVITRO coordinate specification](http://archive.simtk.org/oks/MRI_JointMech/doc/Knee%20Coordinate%20Systems.pdf) · [Archive transformation code](http://archive.simtk.org/oks/MRI_JointMech/src/resampling_plotting.py)
