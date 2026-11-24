# A whole-body calculation for a recorded stride

The full-body panel follows a participant with an instrumented right knee during an ordinary overground walk. A scaled OpenSim model links 24 bodies through 24 joints. The arms, trunk and both legs contribute mass and inertia. The source contains 59 recorded coordinate channels, including dependent and locked coordinates, and 103 muscle–tendon paths. The arms have articulated rigid segments. The modeled muscle paths cover the lower body and trunk. [Released model and trial archive](https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8) · [Grand Challenge experiment](https://doi.org/10.1002/jor.22023) · [Archive license](../knee-loading/data/HARBA-LICENSE.txt)

## How much force reaches the floor?

Starting from marker-derived joint motion, the solver computes the mass-weighted center of the body at every recorded frame. It filters the center-of-mass position at 6 Hz, differentiates twice and estimates the total ground reaction as mass times acceleration minus the gravity force. The pelvis-only comparison runs the same calculation with pelvis height as a proxy for the vertical center of mass. Both estimates use the source model mass of 62.636 kg with no fitted correction. Three force plates provide the measured comparison. [Calculation](force_plate_check.py)

The two plotted anatomical markers show the whole-body center of mass and the pelvis frame origin used by that height proxy. The display positions come from the same 101 interpolated poses as each gait clip. The archive stores pelvis-relative horizontal position with its origin fixed at the first frame. Subtracting that clip offset registers the markers to the walker. Across 303 poses, a separate reconstruction from every segment's mass and exported transform agrees within 0.008 mm, consistent with coordinate rounding. The centers are 3.1–5.4 cm apart across these walks. [Marker registration check](verify_markers.py)

| Right overground stride | Pelvis-height proxy, RMS error | Whole-body vertical force, RMS error | Mean whole-body error | Correlation with plate |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 37.1 N | 34.3 N | −31.0 N | 0.994 |
| Bouncy | 40.8 N | 35.1 N | −31.1 N | 0.997 |
| Forefoot strike | 43.4 N | 30.9 N | −28.6 N | 0.996 |

The source study identifies forefoot strike among the four archived walks and reports the participant mass as 65 kg. The forefoot-strike recording is `mtpgait3`. The GC4 gait protocol identifies the MTP trial family as forefoot strike and assigns medial thrust its own trial family. [Source study, section 2.1](https://www.iri.upc.edu/files/scidoc/3107-Reducing-Complexity-in-Muscle-Tendon-Kinematics-Parameterization-Improves-Convergence-Speed-in-Musculoskeletal-Simulations.pdf) · [GC4 gait protocol, Table 1](https://onlinelibrary.wiley.com/doi/10.1155/2015/939480)

The common window covers 20–100% of the stride, after the initial double-support interval where the trailing foot may be outside the instrumented walkway. Plate timing identifies heel strike. Force magnitudes enter only when scoring the estimates. These are all three complete overground right strides in the released archive, all from one person. Each RMS averages 81 phase samples within one stride. For the ordinary walk, 34.3 N is 5.4% of the participant's reported body weight at 65 kg. The lighter source model accounts for part of the discrepancy. This check measures external ground reaction accuracy. Internal joint contact forces require different observations. [Results](stride.json) · [Instrumented knee comparison](../knee-loading/README.md)

## Where do the moments and path lengths come from?

For the ordinary walk, the 20 N rising crossings on plates 2 and 3 bracket two right contacts at 2.1771073 and 3.2825914 seconds. Right toe-off occurs at 66.91% stride. The animation uses the matching clip `ngait_og5-stance-1`. [Recorded segment poses](../knee-loading/data/whole-body-motion.json)

OpenSim inverse dynamics uses the archived coordinates and three measured ground-reaction wrenches. Plate 1 acts on the left calcaneus and plates 2 and 3 act on the right. Filtering the full coordinate record at 6 Hz precedes the stride calculation. Internal actuators are removed for net joint moments. Recomputing eleven exported joint moments agrees with the archived inverse-dynamics calculations within 0.000002 N·m RMS. That figure measures numerical reproduction from shared inputs. The released comparison values enter only after computing the moments. [Inverse-dynamics calculation](solve.py) · [OpenSim inverse dynamics](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084195/Part%2BI%2BComputing%2BJoint%2BMoments%2Bwith%2BOpenSim)

Holding fourteen arm coordinates at their stride-mean angles, then repeating inverse dynamics with identical lower-body motion, trunk motion and plate forces, changes the required lumbar rotation moment by 4.73 N·m RMS and the right shoulder flexion moment by 5.03 N·m RMS. The prescribed lower-limb moments coincide in this calculation. Eight rigid arm transforms in torso coordinates let the viewer show the exact geometric intervention.

At each of 101 gait phases, OpenSim also evaluates the attachment- and wrapping-dependent length of every muscle–tendon path. The middle right gluteus medius path spans 113.6–125.6 mm and the right medial gastrocnemius spans 479.9–507.1 mm. These are model lengths. The source's three right patellar ligament paths stay constant to the 0.001 mm export resolution in these prescribed poses, and their force elements are disabled in the source model. Path geometry alone supplies no individual muscle tension, injury threshold or pain outcome.

## What moves first when hip muscle force falls?

The forward dynamics experiment uses three recorded right single-support poses at 20%, 30% and 40% of the ordinary stride. At each pose, it imposes 50 N total across the three right gluteus medius muscle paths, then removes that force and solves for the change in initial acceleration. All other muscle actuations are set to zero for both solves. Both cases start at rest in the same assembled pose, so this paired difference isolates the incremental force response. The 50 N is a chosen mechanical scenario. The source model's constraint settings and locked joints are retained. A point on the right calcaneus at the measured center of pressure is fixed in space, with rotation free. The opposite foot is free. [Forward calculation](force_response.py)

At 30% stride the decrement changes the right hip adduction acceleration by +376.6°/s², lumbar rotation by +439.1°/s² and right shoulder flexion by −28.8°/s². Signs follow the named native model coordinates. Moving the assumed support point by ±10 mm along laboratory X yields +361.2 to +411.6°/s² at the hip and −29.6 to −23.8°/s² at the shoulder. The right ankle is considerably more sensitive, spanning −662.1 to −122.9°/s². Halving the force decrement halves the computed acceleration change within 10⁻¹¹°/s². The anchored-point acceleration stays below 3 × 10⁻¹⁴ m/s².

These values describe the first instant of a specified force change. Following a person through subsequent steps would require a contact transition model, a neuromuscular controller and walking data under the intervention. Clinical prognosis would require longitudinal outcomes as well.

## Reproduce

The export was generated with Python 3.12, OpenSim 4.6, NumPy 2.5.5 and SciPy 1.18.1. Eight source-file SHA-256 values and the OpenSim build identifier are included in the JSON. Run the following from the project root with a compatible Python installation.

```sh
git clone https://github.com/gilserrancoli/ReducePoly_forMSK.git /tmp/gc4-gait
git -C /tmp/gc4-gait checkout 00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8
python3.12 -m venv /tmp/gc4-env
/tmp/gc4-env/bin/pip install opensim==4.6 numpy==2.5.5 scipy==1.18.1
/tmp/gc4-env/bin/python whole-body-study/solve.py /tmp/gc4-gait --output /tmp/gc4-stride.json
/tmp/gc4-env/bin/python whole-body-study/verify_markers.py /tmp/gc4-gait --study /tmp/gc4-stride.json
```
