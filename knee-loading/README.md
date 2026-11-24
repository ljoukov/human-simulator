# Which gait changes unload the knee?

This experiment estimates medial and lateral tibiofemoral contact force from recorded movement and external loading. It compares those estimates with held-out instrumented knee measurements and tests how a range of plausible muscle and contact parameters changes the conclusion.

## Run the study

From the repository root with Node.js 24:

```sh
npm --prefix upper-limb ci
npm --prefix upper-limb run dev
```

Open [the knee-load study](http://localhost:5173/knee-loading/). The page includes the recorded hardware benchmark and can rerun it on browsers with WebGPU. The original arm and leg pages remain accessible from the same app.

The viewer uses a bundled full-body atlas and the processed gait recordings. Rebuilding the motion archives uses the original calibration model and its local OpenSim geometry. [Source OpenSim model and geometry](assets/body-SOURCES.md)

## Full-body gait and the path to knee loading

[Follow a stride](http://localhost:5173/knee-loading/#gait-path) opens with two of five complete optical recordings from the Van Criekinge et al. reference cohort (ages 21–86). The selectors identify people by anonymized source ID and show the published demographic data and barefoot overground condition. Shoulder, elbow, wrist, and hand motion come from each recording, including measured differences in wrist travel. The surfaces use the same anatomical atlas for everyone. [Recording protocol](https://doi.org/10.1038/s41597-023-02767-y) · [Demographics, Supplementary Table 1](https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41597-023-02767-y/MediaObjects/41597_2023_2767_MOESM1_ESM.pdf)

Switch to **Implant gait study** to compare ordinary walking, bouncy walking, forefoot strike, and fast treadmill walking in GC4-JW. Each comparison shows a complete recorded right stride. The corresponding force plate, inverse dynamics, and implant observations come from that recording’s first right stance. The separate knee-load experiment includes additional and held-out stances. The movement display aligns walking direction across clips and removes average forward travel so the subject stays in view with visible side-to-side and vertical sway.

The same animated atlas has independently selectable muscles and bones and an outer body surface. The wrist frame preserves palm rotation and the fingers follow the reference hand shape. The knee close-up animates cartilage, menisci, ligaments, and other atlas tissues with recorded sagittal flexion and a 0° atlas limit for hyperextension. The implant view also offers geometry paths for muscles and patellar ligaments computed by the source OpenSim model. Surface deformation illustrates anatomy at the recorded poses. [Atlas attribution](assets/walking-atlas-SOURCES.md) · [Muscle and ligament paths](data/muscle-paths.md)

The two views share stride phase and use the first clip’s duration at the selected playback speed. Playback displays the final pose and begins the next stride on the following animation frame. In the implant view, arrows start at the measured center of pressure and readouts follow vertical ground force and net knee moments through to implant forces in the medial and lateral compartments. Implant foot contact uses a 20 N force-plate crossing. Medial knee pain is a separate clinical endpoint, supported here by a randomized trial of individualized foot-angle retraining. Pain ratings are unavailable for these gait clips. [Recorded gait and OpenSim model](https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8) · [Clinical trial](https://doi.org/10.1016/S2665-9913(25)00151-1)

The [implant preparation tool](tools/prepare_whole_body.py) reproduces [the pose and force archive](data/whole-body-motion.json). The [optical preparation tool](tools/prepare_reference_walks.py) fits foot contact and transfers both legs’ lateral motion in the same pelvis coordinate frame. Tests check bilateral knee and ankle spacing over all five optical recordings. After subtracting average forward travel, the largest adjacent pelvis displacement is 6.2 mm per 1% stride sample. Displayed knee flexion differs from the source angle by 1.1–3.8° RMS owing to the contact fit. Numeric knee flexion uses the source signal. [Processed optical recordings](data/reference-walks.json)

## Reproduce the model results

Generate the study report on CPU:

```sh
node knee-loading/tools/run-study.ts knee-loading/data/gait.json knee-loading/data/gc5.json \
  --candidates 65536 --retain 512
```

Run the same experiment and then the timed workload on a hardware WebGPU adapter using Deno:

```sh
deno run --allow-read --allow-write --unstable-webgpu knee-loading/tools/run-study.ts \
  knee-loading/data/gait.json knee-loading/data/gc5.json --gpu --candidates 65536 --retain 512
deno run --allow-read --allow-write --unstable-webgpu knee-loading/tools/benchmark.ts
```

The study writes `data/report.json`. The audit in that file records the calibration and evaluation IDs, nominal anatomy reference, two regressions, and measured accuracy. The benchmark writes `data/benchmark.json`. Rebuild the Vite app after regenerating these artifacts for use through the production bundle.

## Million-candidate CUDA benchmark

The H100 scorer reuses the angle-dependent knee geometry across all candidates. A 128-thread block scores four candidates in parallel, one per warp. Each lane scores a strided set of observed frames. Warp shuffles combine those losses on the GPU, returning one score per candidate. The CUDA baseline uses the original WebGPU layout of one block per candidate and shared-memory reduction. A bounded heap selects the retained candidates with the same ordering and index tie-break as a full sort.

Every backend scores identical prefixes of a fixed 1,048,576-candidate pool against 91 measured stance frames from GC4-JW. The comparison checks the binary inputs by SHA-256, every score against the TypeScript solver within the stated absolute and relative tolerance, and the membership of the top 256. Both GPU batch timings include candidate upload and score readback with recorded frames already resident. The H100 report separates kernel time, backend setup, and Modal invocation time. The timings measure calibration score sweeps for prescribed gait. Held-out RMSE uses the median of the retained ensemble at each frame of six evaluation stances from the same participant.

At **1,048,576 candidates × 91 frames**, the NVIDIA H100 80 GB took **4.73 ms** per warm batch over seven runs, compared with **58.11 ms** on the laptop M4 GPU and **116.35 ms** with the laptop's fastest tested native CPU configuration (eight threads). The H100 warp kernel itself took **0.628 ms** versus **1.138 ms** for the CUDA block baseline. Including transfers, that same-GPU change reduced the batch from **5.24 ms to 4.73 ms**. The one-off Modal call took **45.94 s**, including startup, compilation, and correctness checks.

Expanding the search from 32,768 to 1,048,576 candidates reduced held-out medial force RMSE from **0.305 to 0.293 body weights** and lateral RMSE from **0.507 to 0.481** in the instrumented participant. The same candidate had the best training score at 262,144 and 1,048,576. [Measured comparison](data/gait-ensemble-comparison.json) · [H100 details](data/gait-ensemble-h100.json) · [Native CPU details](data/gait-ensemble-laptop-native.json) · [Laptop WebGPU details](data/gait-ensemble-laptop-webgpu.json)

Run the ephemeral H100 function from the repository root with an authenticated Modal CLI. The laptop baseline uses Clang and Node.js 24, and the laptop GPU run uses Deno with a hardware WebGPU adapter.

```sh
modal run knee-loading/tools/modal_h100.py

clang++ -std=c++17 -O3 -march=native -ffp-contract=off -pthread \
  knee-loading/tools/native_scores.cpp -o /tmp/gait-native-scores
node knee-loading/tools/gait-ensemble.mjs --native /tmp/gait-native-scores --threads 1,4,8,10 \
  --output /tmp/gait-native.json
deno run --allow-read=knee-loading --allow-write=/tmp --allow-sys=cpus --unstable-webgpu \
  knee-loading/tools/gait-ensemble.mjs --webgpu --output /tmp/gait-webgpu.json

node knee-loading/tools/compare-ensemble.mjs knee-loading/data/gait-ensemble-h100.json \
  /tmp/gait-native.json /tmp/gait-webgpu.json knee-loading/data/gait-ensemble-comparison.json
```

## Recordings and holdout

- **GC4-JW.** The public [Harba et al. replication release](https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8/ExampleTrackingSimulations/OpenSimModel/subject_GC) contains synchronized ground-reaction forces, inverse kinematics, inverse dynamics, and four-quadrant implant data from the [Knee Load Grand Challenge](https://simtk.org/projects/kneeloads). One normal overground stance calibrates the model. A second stance from that same pass and five stances from the bouncy, forefoot-strike and fast-treadmill recordings are held out. Stances from the same recording are repeated measurements of one participant. Fast treadmill walking combines a change in speed and a change in surface.
- **GC5-PS.** The [Modenese et al. release](https://github.com/modenaxe/femoral-anteversion-paper) provides five normal-walking passes for a second implanted participant. Three passes calibrate this participant's model, and two are held out. The available input pipeline approximates external moments from the measured force-plate wrench and knee centre, and implant force timing is aligned by normalized phase. Its results are reported separately.
- Both sources use instrumented implants. The experiment conditions on observed movement and external loading. It does not establish responses for native knees or predict a person's gait in a new condition.

The preparation scripts produce 101 samples per stance. Archived experiment inputs include the derived data and their source license notices. Recreate them with Python 3, NumPy, SciPy, and, for GC5, OpenSim:

```sh
python3 knee-loading/tools/prepare_data.py
python3 knee-loading/tools/prepare_gc5.py --help
```

For another dataset, use the field definitions in [dataset.ts](src/dataset.ts). `role` may explicitly reserve calibration and evaluation stances. Without roles, the first half of ordinary-walking stances in each participant's source order calibrates the model. Modified gait never enters calibration. All output uses body weights unless stated otherwise. GC4 uses the signed axial force from the inverse-dynamics release. Force-plate vertical force serves as the axial approximation when that signal is unavailable.

## Mechanics and uncertainty

1. Bounded extensor and flexor groups balance the measured sagittal moment. Effective moment arms vary with knee angle based on the repository's Rajagopal2016 geometry. When angle is unavailable, the solver uses 15°. Paired antagonist recruitment raises compression while preserving net sagittal torque.
2. The external axial force and the two muscle groups contribute to total compression. Medial and lateral contact forces are constrained to be nonnegative. Their forces and contact lever arms balance the frontal-plane moment when the chosen model can achieve equilibrium. Capacity and unilateral contact limits produce explicit residuals.
3. The experiment sweeps 12 geometry and recruitment parameters using a deterministic stratified sample. Calibration ranks candidates by squared force error plus a penalty for moment residuals. Evaluation uses the retained scenarios on each condition with the same parameters. The plotted central 90% bands summarize the retained scenarios. Their coverage is checked against the implant traces.
4. Each participant's moment-only regression is trained on the same calibration stances. The report includes both adduction-only and adduction-plus-absolute-flexion fits. The latter appears in the viewer. Early and late peaks are the largest force between 5–45% and 55–95% stance, respectively. Force changes are calculated within each candidate before summarizing the band. Robust offload counts retained scenarios that reduce load by at least 0.10 BW.

The sensitivity ranges describe assumptions about effective group geometry and capacity. In particular, a single pair of contact arms, fixed axial projections, and a single co-contraction rule are shared across gait conditions. Improving the fit on a new participant would require an independent evaluation split.

## Measured results

On the GC4 holdout, the calibrated model estimated medial waveform force with **0.304 BW RMSE**, versus **0.447 BW** for the two-moment baseline. The baseline had lower errors for lateral and total force. Only **24.9%** of measured medial force samples fell in the model's central 90% scenario band. On the independent GC5 ordinary-walking holdout the two-moment baseline had lower error in all three compartments.

For fast treadmill walking, the peak external adduction moment in the early window rose **106%** relative to the held-out overground normal stance, while measured medial force changed by **−0.026 BW**. The mechanics model predicted **+0.147 BW**. Both time-averaged and peak results are visible by gait condition. These trials reveal a discrepancy between the proxy, the calibrated model, and the implant measurement for this participant.

The benchmark runs the same numerical model and calibration objective on a multicore CPU and a WebGPU backend. The GPU assigns one workgroup per calibration candidate, computes force balance across that candidate's stance frames, and reduces to a single score before readback. This reduces output traffic compared with returning every frame's forces and residuals. On the Apple M4, **262,144 candidates × 91 stance frames** took **28.84 ms GPU vs 58.08 ms CPU**, or **2.01×**, timed over five runs after warm-up. Outputs passed the stated numerical tolerance and the top 256 candidates matched. Waveform readback was faster on the CPU. These are batch timings after backend setup, with transfers included and setup reported separately.

[Harba et al.](https://doi.org/10.1002/cnm.70143) · [Fregly et al.](https://doi.org/10.1002/jor.22023) · [Instrumented-gait proxy study](https://doi.org/10.1002/jor.21142) · [Modenese et al.](https://doi.org/10.1016/j.gaitpost.2021.06.014) · [OpenSim Joint Articular Mechanics](https://github.com/clnsmith/opensim-jam)

## Patellar tracking check

In a separate passive flexion test for Open Knee(s) specimen oks001, net patellar rotation from about 5° to 55° of knee flexion was **15.59° measured**, **38.17°** for the generic Rajagopal reference, and **37.38°** for the current viewer. The released specimen and model use different patellar landmarks and origins, so the audit compares rotation magnitudes. In 35 held-load tests on the same specimen, increasing the nominal quadriceps load from 20 N to 600 N shifted the observed patellar origin by **1.47–3.80 mm** across five flexion angles. These data quantify a load response for the specimen separately from the gait force experiment. [Audit data](data/patella-oks001.json) · [Measurement archive](https://doi.org/10.18735/4e78-1311)

## 3D gait and pressure views

- **Gait comparison.** Both knees follow the same stance cursor. Each uses its own group’s mean sagittal flexion on the generic atlas rig. The readout preserves any recorded hyperextension while the rendered atlas floors at 0°. The display mirrors the atlas for left-knee recordings. Articular colors encode the total medial or lateral force in body weights, from either the implant or the model. Bones, cartilage, menisci, ligaments, and muscles have independent visibility controls. The compact [knee asset](assets/knee.glb) is a CC BY-SA 4.0 subset of the [lower-limb anatomy](../lower-limb/README.md) and can be rebuilt with `python3 knee-loading/tools/export-knee-asset.py`.
- **Measured patellar pressure.** The separate relief shows calibrated Tekscan data from Open Knee(s) specimen oks001, in MPa, across five flexion angles and seven quadriceps loads. Color and height share a fixed pressure scale. Each map takes a per-sensor median over the recorded frames. The relief and 2D grid use the original sensor coordinates and the 1.27 mm sensor spacing. The archive does not register these grids to the gait anatomy. [Pressure data and provenance](data/patellar-pressure.json) · [Preparation script](tools/prepare_pressure.py) · [Open Knee(s) protocol](https://doi.org/10.1016/j.dib.2021.106824)
