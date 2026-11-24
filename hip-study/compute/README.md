# Hip force ensemble

The experiment asks how much the internal distribution of load can vary when different virtual anatomies support the same measured movement. Each virtual case shares a set of geometry and recruitment parameters across all participants and activities. The search scores an entire parameter vector against a training population.

## Reproduce the study

From the repository root, use Python 3 with NumPy, curl and a native C++ compiler. The input recordings follow OrthoLoad's use terms. Download the originals and prepare `implants.json` and the workbench's `results.json` locally. The OpenSim geometry lookup is shipped in `hip-study/data`. [Recording attribution and terms](../IMPLANT_METHODS.md)

```sh
python3 hip-study/tools/prepare_implants.py --cache /tmp/hip-implants
python3 hip-study/compute/study.py --work /tmp/hip-search-work --science-only
```

The full benchmark builds a temporary native library and writes the matched input bundle to the chosen work directory.

```sh
python3 hip-study/compute/study.py --work /tmp/hip-search-work
modal run hip-study/compute/modal_benchmark.py --inputs /tmp/hip-search-work/benchmark-input.npz --compact-seed 20261122
python3 hip-study/compute/study.py --merge
python3 hip-study/compute/bootstrap.py
python3 hip-study/compute/bootstrap.py --offset-sweep
```

For the local GPU and dominance-screen experiments, use the same NumPy and Modal Python environment. The Metal runner requires a Mac with Metal and Xcode command-line tools. Both sensitivity drivers regenerate the exhaustive CPU winners for their correctness checks.

```sh
python3 hip-study/metal/benchmark.py /tmp/hip-search-work/benchmark-input.npz
python3 hip-study/metal/sensitivity.py
python3 hip-study/compute/pareto.py --mode cpu
```

The cloud runs request an H100 and incur cloud compute charges. `--science-only` recomputes scientific outputs while preserving the recorded performance measurements. `--merge` checks the SHA-256 of the shared kernel and input payload before combining laptop and H100 records.

## Mechanical model

All forces are expressed as multiples of participant body weight (BW). The external adduction moment is divided by body weight, so its unit is metres. Each trial contributes synchronized vertical GRF, outward mediolateral GRF, an external frontal-plane moment about the hip estimated from the force plate and markers, and hip flexion/adduction angles. Stance-segment inertia and explicit muscle activation dynamics are omitted in this quasi-static calculation. The inputs have phase, units and sign conventions documented in the source data.

For vertical force V, outward horizontal force H, and external adduction moment M:

1. Antagonist coactivation adds force C = k V, with k varied across cases. The adductor force is max(−M, 0) / rA + C. The moment required of the abductors is D = max(M, 0) + rA C. This construction balances the measured frontal-plane moment for either sign.
2. The IT band takes fraction s of D, giving band tension B = s D / rI. Combined gluteus medius and minimus tension is G = (1−s) D / rG. The effective gluteal abduction arm rG is a scenario-specific multiple of the two-dimensional interpolation of six MRI-informed OpenSim muscle segment moment arms. Axial rotation is fixed at neutral. The adductor and IT-band effective arms are constant in each scenario. The ranges are engineering sensitivity assumptions.
3. Let A be adductor force and t the inward angle of the gluteal force line from vertical. In a frontal-plane approximation the predicted contact magnitude is hypot(V + G cos(t) + B + 0.85 A, H − G sin(t) − sqrt(1−0.85²) A). The estimator compares this planar resultant with the magnitude of the measured three-dimensional implant force. The discrepancy includes the neglected fore–aft force, distal segment weight and acceleration, simplification of muscle geometry, and differences among the implant recipients.
4. The main regional load estimate is B times the inward unit-tension coefficient from the trochanteric guides of three Davis IT-band tracts. The geometry file gives its coordinate convention and off-grid accuracy. It describes force (BW) across an idealized guide. Measuring contact pressure, individual tendon strain, material injury, and symptoms would require other observations. The supplementary cable sensitivity uses a range of bend-angle assumptions and is reported separately. Its two additional parameters enter only that sensitivity calculation.

The full input table records all eight sampled ranges. The recruitment parameters are effective group quantities. They do not specify patient-level tissue diagnoses. The primary lookup uses the MRI-informed 2023 OpenSim model; a matched 65,536-case run repeats training and evaluation with the 2016, example generic-scaled, and example MRI-personalized gluteal arm lookups.

## Evaluation

Participants H1–H6 supply calibration data and H7–H10 supply evaluation data. Training minimizes phase-sample squared error averaged within trial, then across trials within each subject, then across subjects. The primary run samples 1,048,576 parameter sets at a fixed seed by independently stratifying each parameter range. Retain the best 512 training fits. Predict the median contact across this set at each frame. The central 90% band describes scenario spread. Its empirical coverage quantifies the fraction of held-out implant measurements inside it. The baseline fits an intercept, vertical GRF and absolute frontal moment to the same calibration targets with the same sample weights. Report the mean of trial waveform RMSEs within participant, then average over participants. A separate ten-fold leave-one-participant-out analysis repeats the fit with 65,536 scenarios per fold.

The example compares two retained candidates on the same first lexicographically sorted held-out walking trial (H10R). Select the pair with the widest difference in inward ITB force at peak vertical GRF among pairs whose predicted contact waveforms differ by at most 0.05 BW RMS. This extremal example quantifies ambiguity under the stated model. The participant's implant trace has no role in selecting the pair. The nested search independently counts cases whose training RMSE is within 0.02 BW of the full-run minimum and whose contact waveform is within 0.05 BW RMS of the best-fit model on that example.

## Parallel computation

The same C++ expression compiles as a CUDA thread evaluating one parameter set across all calibration frames, or as four-wide native vector math distributed over the laptop CPU cores. Both return one weighted score per case. Candidate parameters and calibration frames are immutable. This layout avoids constructing the candidate × frame × muscle tensor. Winning conditions can later be reconstructed as individual waveforms. The GPU uses fixed arithmetic order and disables fused multiply-add for parity. A separate NumPy float64 solution tests random cases. The Metal implementation benchmarks the same measured inputs on the laptop GPU.

Both host and device are warmed before reporting medians of repeated scoring runs. The laptop measurement includes native thread startup/join and allocating output. The CUDA device measurement covers kernel execution. H100 transfer times, cloud invocation wall time and a second seed-only request are separately available in the hardware reports. For the compact request, Modal reproduces the sampled parameters from the seed and returns the top 256 indices with a parameter hash. The full H100 scoring measurement tests every score against the laptop.

The bootstrap reuses each virtual anatomy's per-participant objective. Six training participants create 46,656 ordered resamples, equivalent to 462 distinct multiplicity patterns. Weight each pattern by its multinomial frequency to calculate selection frequencies and load percentiles. This captures variation in which training participants are sampled, conditional on the fixed reference model and measured movements.

## Measured outcome

On the four evaluation participants the mechanics ensemble reached 0.419 BW mean trial RMSE. The fitted three-coefficient baseline reached 0.274 BW and vertical ground force alone reached 1.404 BW. The mechanics ensemble's central 90% scenario band covered 11.3% of the measured force samples. Leave-one-participant-out evaluation across ten people gave 0.400 BW versus 0.259 BW for the fitted baseline. These errors limit the interpretation of subject-specific force values.

The selected H10R pair differed in contact waveform by 0.0168 BW RMS. At peak ground force their gluteal tension estimates differed by 0.642 BW and their regional compression estimates differed by 0.182 BW. The pose and external load were identical in that comparison. The uncertainty search around the full-run best case found 14 comparable candidates in the first 4096 draws and 2889 in the full ensemble.

At 1,048,576 candidates × 1212 training frames, the Apple M4 CPU with ten threads scored the ensemble in 111.18 ms median. The NVIDIA H100 80 GB took 1.93 ms median with frames and parameters resident, a 57.6× speedup. All float32 scores matched exactly. The array-upload round trip took 7.82 s and the compact seed-only round trip took 1.15 s. These measurements include different portions of the workflow, as recorded in the JSON files. The laptop is faster for this single interactive request.

The exact participant bootstrap's 484,442,112 candidate/resample comparisons took 1279.95 ms on the laptop and 46.77 ms on the H100 with resident inputs. Both selected identical winners for all 462 unique fits. The most frequently selected model won 24.3% of ordered resamples, with 53 distinct winners across all resamples. The matched result and timing arrays are in `data/benchmark.json`, `data/benchmark-metal.json` and `data/bootstrap.json`.

The joint-center sensitivity run varies the frontal moment lever arm from −15 to +15 mm in 2.5 mm steps. This is an assumed positional offset. Each offset refits all 462 bootstrap cases. The workload made 6,297,747,456 candidate/resample/offset comparisons. All 6,006 winners matched on both devices. From prepared frames to returned winners, the median M4 job took 18.12 s and a warm H100 cloud request took 1.27 s (14.25×). The first comparison, including compiling the local kernel or opening a fresh Modal application and waiting for its first response, took 19.11 s and 10.17 s respectively (1.88×). The cloud used a prebuilt image with uncontrolled worker-cache state. Across the three offsets −15, 0 and +15 mm the median predicted H10R walking contact at peak vertical force was 3.086, 3.095 and 3.257 BW. The implant measured 2.107 BW at that frame. This perturbation leaves substantial contact overprediction. See `hip-study/data/bootstrap-sensitivity.json` for all 13 cases.

## Removing dominated candidates

A model with strictly higher loss on every calibration participant cannot minimize any nonnegative bootstrap-weighted sum of these losses. At each lever offset, 52 pilot weight vectors select comparison candidates from the first 4,096 parameter draws. The screen removes only candidates that lose on all six participants to at least one of these models. Among the survivors the algorithm evaluates the exact weighted objectives, preserving ID order for ties. The screen leaves 277 million final candidate-weight comparisons plus the pilot. CPU and Metal reproduced all 6,006 original winner IDs.

| 13-offset search, median warm time from prepared frames to returned IDs | Exhaustive | With dominance screening |
| --- | ---: | ---: |
| Apple M4 CPU | 18.122 s | 2.675 s |
| Apple M4 GPU, local Metal | 1.156 s | 0.645 s |
| NVIDIA H100, Modal round trip | 1.272 s | See current hardware report |

The warm exhaustive laptop-GPU and remote-GPU timing samples overlapped. For a local interactive search the fastest measured laptop method was the screened Metal implementation. [Metal report and samples](../data/benchmark-metal-sensitivity.json) · [Dominance-screen hardware report](../data/benchmark-pareto.json)
