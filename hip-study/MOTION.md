# Replaying whole-body motion

The viewer retargets 22 landmarks to a shared anatomical atlas. Positions are in metres. The positive axes point forward, upward and toward the participant’s right. The walking clips subtract constant forward translation so the body stays in view. Pelvic sway, limb swing and rotation remain in the fixed frame. The standing clip preserves displacement in all three directions. The time scrubber spans the source interval shown for each clip.

| Clip | Source interval | What drives the pose |
| --- | --- | --- |
| Preoperative hip OA | Participant HOA001, right foot strike to right foot strike, 5.23–6.67 s. Toe off occurs at 59.7% of the cycle. | Optical markers and Plug-in Gait joint centers at 100 Hz. The workbook identifies right-sided Kellgren–Lawrence grade 4 OA. [Motion archive](https://doi.org/10.6084/m9.figshare.14420645.v2) · [Participant metadata](https://doi.org/10.6084/m9.figshare.13655975) |
| Instrumented hip walk | Participant H1L, left foot on plate 1, 6.15729–6.83750 s. | Postoperative markers and a synchronized left hip implant. Measured shoulder and leg landmarks drive the pose. Head and distal arms use trunk-attached display positions. [Walking and stance recordings](https://orthoload.com/hip-joint-force-and-motion-capture-data/) |
| Instrumented single-leg stand | Same participant, left foot on plate 2, 1.79896–7.47604 s. | Same measurement system. The standing direction is the mean pelvic anterior axis. [Instrumented hip benchmark](IMPLANT_METHODS.md) |
| Matched predictive walks | Right stride at 1.33 m/s, with 100%, 50%, 25% or 10% bilateral hip muscle capacity in the source scenarios. | OpenSim forward kinematics of 29 archived coordinates. The four solutions share contact, walking speed and optimization settings. The source reduced isometric force capacity of muscles acting on hip flexion and extension. [Study](https://doi.org/10.1098/rsif.2019.0402) · [Source revision](https://github.com/antoinefalisse/3dpredictsim/tree/5fa3929fa9ba29d5c057c16b18671a017d8f15c1) |

Each clip has 101 phase samples. The OA clip uses the source's hip flexion and adduction outputs. The OrthoLoad clips estimate the angles from the knee and hip centers in pelvic coordinates. Their source hashes and time intervals match the force benchmark, and both angular channels agree with its independent calculation within 0.001°. The predictive simulations supply right hip angles, vertical ground reaction in body weights, and average activation across three right gluteus medius compartments. Their 101st frame closes the periodic stride by repeating the first pose. The neck and head move with the source model's rigid torso.

The atlas gives the same anatomical context to different motion sources. It depicts generic bone and muscle geometry. Surface markers, model joint centers and anatomical atlas geometry each have different sources of uncertainty. These clips record or simulate motion under the stated conditions. Tendon inflammation and pain were outside the measurements.

## A paired camera check

In HOA001's matched rear-view video, MediaPipe detected the right hip and knee in all 72 video frames of the selected stride. A camera projection fitted to shoulders, hips and ankles outside the stride maps the optical joint centers into the video plane. Right-thigh tilt from video correlated with the projected reference (r = 0.918), but video nearly doubled the excursion (12.10° versus 6.54°). Its 5.69° mean absolute error exceeded the 4.74° error obtained by holding the calibration-period median tilt constant. The comparison measures the image-plane inclination of the hip-to-knee segment in one stride. This view supplies little direct information about motion toward or away from the camera. [Paired numerical traces, calibration and timing sensitivity](data/video-audit.json) · [Video reproduction script](tools/audit_video.py)

## Reproduction

Extract `Patients/HOA001/M0/gait/gait03.c3d` from Bertaux et al.’s archive into `/tmp/hip-motion`. The full archive is about 4.74 GB. The extractor checks the selected C3D’s SHA-256. The published gait recording is licensed CC BY 4.0. [Figshare archive](https://doi.org/10.6084/m9.figshare.14420645.v2)

From the repository root, use an environment with NumPy, SciPy, c3d and OpenSim 4.6. The commands below generate the full local `motion.json`. The OrthoLoad preparation step downloads the originals into a local cache. See the force benchmark for the attribution and permitted noncommercial uses. [Source terms](https://orthoload.com/infos/legal-information/)

```sh
python hip-study/tools/prepare_implants.py --cache /tmp/hip-implants
python hip-study/tools/predsim_motion.py /tmp/hip-falisse --fetch --output-dir /tmp/hip-clips
python hip-study/tools/prepare_motion.py \
  --oa /tmp/hip-motion/gait03.c3d \
  --implant /tmp/hip-implants/h1l_060511_1_47_mov.csv \
  --implant /tmp/hip-implants/h1l_060511_1_24_mov.csv \
  --predsim-clip /tmp/hip-clips/predsim-hip-0.json \
  --predsim-clip /tmp/hip-clips/predsim-hip-50.json \
  --predsim-clip /tmp/hip-clips/predsim-hip-75.json \
  --predsim-clip /tmp/hip-clips/predsim-hip-90.json \
  --output hip-study/data/motion.json
```

To regenerate the bundled `motion-public.json`, run the Falisse preparation and motion commands above with both `--implant` lines omitted and `--output hip-study/data/motion-public.json`. The workbench selects the full motion archive when it is available.
