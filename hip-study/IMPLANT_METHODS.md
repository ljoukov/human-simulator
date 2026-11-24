# Instrumented hip benchmark

The OrthoLoad Hip Joint III release pairs instrumented prosthesis forces with whole-body markers and force plates during level walking and ipsilateral single-leg stance. The study uses one recording of each activity from each of ten people. H1 through H6 form the calibration set. H7 through H10 form the evaluation set. All frames and activities from one person stay in one partition. Each trial record includes the original file identifier, public source link, source SHA-256, body weight, plate number and analyzed time interval. [Original trial index](https://orthoload.com/hip-joint-force-and-motion-capture-data/) · [Original validation study](https://doi.org/10.1371/journal.pone.0195376)

## Coordinates, synchronization and contact

The source CSV uses a common 960 Hz time grid. Marker and implant channels occupy a subset of rows on that grid. We interpolate each channel against its own timestamps and preserve gaps exceeding 100 ms. Plate vertical force is expressed in kN despite the CSV header's N label. Multiplying the exported vertical force by 1,000 agrees with the C3D analog signals to 0.50 N maximum across 20 files after transformation to lab axes. Both horizontal force components are in N. The transforms from raw C3D forces to exported positive-up lab components are diagonal with signs (−, +, −) for plate 1 and (+, −, −) for plate 2. The component comparison uses original timestamps with zero time shift. The maximum difference between the published implant resultant and the norm of its three components was 1.05 N.

The center of pressure coordinates use lab directions relative to the center of each plate. The means of the four plate corners in all 20 C3D files give lab origins of (200, 300, 0) mm and (200, 903, 0) mm, respectively. Walking analysis uses contiguous plate contact above 5% body weight. For stationary balance, the stance foot must bear over 80% body weight, the second plate under 10%, and the contralateral foot markers must rise at least 25 mm relative to those of the loaded foot. The algorithm selects the longest eligible bout whose median center of pressure is closer to the implanted-side foot. Across the 20 selected bouts, median center-of-pressure distances were 12–23 mm from the implanted-side foot and 224–501 mm from the opposite foot. Each contact window is resampled to 101 points. The original durations distinguish a walking stance from a several-second balance hold. Trial selection uses only the foot trajectories and force plates.

## Derived model inputs

The estimated hip center uses the Harrington pelvis-only regression with ASIS and PSIS markers. In pelvic axes, the offset from the ASIS midpoint is 0.33 × pelvic width + 7.3 mm laterally, 0.24 × pelvic depth + 9.9 mm posteriorly, and 0.30 × pelvic width + 10.9 mm inferiorly. The pelvis frame is recomputed from each pose. Pelvic width is the distance between ASIS markers and depth is the distance between the ASIS and PSIS midpoints. The oriented thigh runs from this center to the midpoint between the femoral epicondyles. Its projections onto pelvic sagittal and frontal planes give the flexion and adduction angle proxies. Marker placement, soft-tissue movement and joint center estimation affect these projections. [Harrington et al.](https://pubmed.ncbi.nlm.nih.gov/16584737/)

For each frame, project the stance-side lateral pelvis axis onto the horizontal plane and normalize it. Let *d* be the outward projection of the hip-center location minus the center of pressure (m), *h* the hip-center height above the plate (m), *v* the vertical ground force (BW), and *f* the outward ground force (BW). The frontal external adduction moment proxy is *d v − h f* in BW·m. Positive values imply an internal abductor demand. This force-plate contribution omits individual segment acceleration, segment gravity and other terms of full inverse dynamics. The measured response is the implant force resultant divided by the participant's body weight. These measurements characterize replaced joints.

## Reproduction

The preparation step requires NumPy and curl. The independent raw-signal audit also requires the Python c3d package.

```sh
python3 hip-study/tools/prepare_implants.py --cache /tmp/orthoload-hip
python3 hip-study/tools/audit_implants.py --cache /tmp/orthoload-hip
```

Bergmann G. and Damm P. (eds.), Julius Wolff Institute, Berlin Institute of Health at Charité Universitätsmedizin Berlin (2008), OrthoLoad. Downloaded on 22 November 2026. Data use is subject to OrthoLoad's noncommercial terms. Original files are obtained directly from the source by the preparation and audit commands. The study JSON contains the computed normalized inputs and comparison samples. [OrthoLoad](https://orthoload.com/) · [Use and attribution terms](https://orthoload.com/infos/legal-information/)
