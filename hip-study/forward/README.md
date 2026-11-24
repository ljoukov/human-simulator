# Which mechanical impairments can yield similar motion?

This experiment simulates the stance hip and lateral trunk response to an abductor torque-capacity reduction, additional passive resistance to hip extension, both perturbations together, and the unperturbed body. It predicts departures from a shared movement target. The labels identify interventions made in the equations. Both symptomatic tendon disease and hip osteoarthritis can affect strength and mobility in people. [Direct strength comparison](https://pubmed.ncbi.nlm.nih.gov/27951482/) · [OA passive motion study](https://pubmed.ncbi.nlm.nih.gov/20811166/)

## Inputs

The walking target is the mean of eleven right-leg trials from one adult in the Falisse archive. The first 100-point gait-cycle sample is heel strike. The extractor ends stance immediately before the first sample below 3% body weight after 45% of the cycle. It converts reported vertical force from percent body weight and divides measured net moments by the weight obtained from the accompanying OpenSim body's segment masses. Right hip flexion and adduction, lateral trunk angle, ground force, and two net moment profiles enter the model. Walking stance duration is set to 0.68 s. The source data and published predictive results are pinned to a Git revision and identified by SHA-256 in `reference.json`. [Reference data and MIT license](https://github.com/antoinefalisse/3dpredictsim/tree/5fa3929fa9ba29d5c057c16b18671a017d8f15c1) · [Falisse et al.](https://doi.org/10.1098/rsif.2019.0402)

The second task is a designed 2.2-second supported-to-single-leg weight shift and return. A smooth pulse reaches 1.10 body weights of stance-side vertical ground force. A 0.10-body-weight contralateral mass on a 0.40 m lateral arm adds 0.04 body-weight metres to the model's peak frontal-plane demand. The peak total demand is 0.12 body-weight metres at nominal anatomy. The prescribed force and posture targets approximate the external load challenge. Actual participants performing this motion could choose other foot forces, support strategies and trunk postures.

## Equations and controls

The states are deviations in flexion, adduction and lateral trunk lean from the common target trajectory, plus their angular velocities. The three equations use effective diagonal rotary inertias divided by body weight of 0.009, 0.010 and 0.016 m·s². The nominal inverse-dynamics input absorbs contributions of the rest of the body at the reference movement. Perturbations propagate through the stance hip and trunk. Contact sequence, fore–aft foot placement, contralateral motion, and vertical ground force remain prescribed. The forward model therefore represents a tracking-response experiment on a stance segment.

Let `x` denote the change in flexion, `y` the change in adduction and `z` the change in loading-side trunk lean, all in radians. Torques below are divided by body weight. The sagittal actuator applies the nominal measured moment minus proportional/derivative tracking feedback on `x`. Actuator limits are −0.19 and +0.16 m. A hypothetical extension restriction adds a passive flexion moment equal to its stiffness times the positive angular distance past extension onset. The net torque driving `x` is the change from the nominal sagittal moment plus this passive moment.

Trunk lean shortens the external frontal-plane moment arm. The external adduction moment is `M − s sin(z)`, where `M` is nominal demand scaled for anatomy and `s` is the effective trunk lever times stance vertical ground force. The group abductor moment tracks external demand plus adduction feedback. A capacity limit caps active abduction while a passive frontal spring and damper resist residual adduction. The resulting abductor moment divided by an effective moment arm gives lumped abductor force in body weights. These outputs combine the muscles that generate the net abduction moment. Tendon-specific tension, inflammation and nociception would require additional models and validation.

The default controller finds the small-angle lean target minimizing `((M − s z)/C)² + (z/B)²`, with capacity `C` and individual lean preference `B`. It subtracts the optimum for the nominal body since that strategy is embodied in the reference. A damped trunk actuator approaches the new target over time. An alternate controller tested on held-out bodies initiates leaning when the reference would consume 90% of available abductor torque. Semi-implicit Euler integrates the three degrees of freedom over 256 substeps and records 65 phases. Doubling the resolution changed any angle by at most 0.050° across the first 16 bodies, both tasks and all four interventions.

## Population and observation

With random seed 20261122, each of 1,024 virtual anatomies undergoes all four interventions in both tasks. Effective abductor arms span 40–65 mm. Unperturbed capacities span 0.10–0.17 body-weight metres. The capacity perturbation retains 55–85% of that body's torque capacity. Passive extension resistance adds 0.18–0.68 body-weight metres/radian beyond body-specific extension onsets between −12° and −2°. Anatomy, baseline motion offsets, tracking gains and unperturbed capacities vary across the entire population. These distributions encode scenario ranges. A clinical tendon cohort had an average measured 32% loss of affected-side abduction torque relative to controls. Both diagnosis and pain can change the relation between measured clinical strength and available torque during motion. [Abductor measurements](https://pubmed.ncbi.nlm.nih.gov/26418561/)

Camera-like perturbations have population RMS scales of 5.4° flexion, 3.7° adduction and 3.0° trunk lean. The first two approximate published ordinary-gait two-phone waveform errors. The trunk scale is an engineering assumption. The model uses a Gaussian per-session offset shared across tasks and independent, temporally correlated sinusoidal components. It omits image formation, landmark occlusion and errors that vary systematically with diagnosis. [Horsak et al., Table 1](https://phaidra.ustp.at/api/object/o%3A5498/download)

A fixed 15-nearest-neighbor classifier uses nine evenly spaced samples of each angle per task, each scaled by its camera error. The binary comparison includes the two isolated impairments. All conditions of a given body stay on the same side of the 768/256 training/test split. The single-feature baseline uses just the flexion excursion of the noisy walking observation. A repeat walking observation carries the same calibration offset. The bootstrap resamples 256 held-out body IDs together with their paired counterfactuals and task data. Intervals reflect sampling among those virtual bodies with a fixed trained classifier.

## Outcomes

| Test on held-out virtual bodies | Balanced accuracy |
|---|---:|
| Walking hip flexion excursion with camera error | 54.1% |
| Three walking angles with camera error | 62.1% |
| Two walking observations with camera error | 60.5% |
| Walking and loaded single-leg stance with camera error | 68.4% |
| Exact simulated walking angles | 88.7% |
| Exact simulated angles from both tasks | 87.9% |

Adding the load challenge to walking gained 6.25 percentage points on these draws (paired bootstrap 95% interval 1.56–10.94 points). Using the previously unseen torque-reserve controller for the test bodies gave 60.7% for walking and 69.1% for both noisy tasks. Finite-sample nearest-neighbor discrimination can fall as dimensions are added, as in the repeated-walk and noise-free comparisons.

A second endpoint measures overlap directly. A same-anatomy pair lies inside the camera budget when the waveform RMS difference in *each* angle is at most its respective error scale. All 256 held-out pairs passed that criterion in walking; 227/256 passed it in both the walk and the 0.10-body-weight challenge. With contralateral mass of 0, 0.05 and 0.15 body weights, 100%, 97.3% and 65.2% passed both checks respectively. This is a scenario-dependent tolerance, distinct from a two-sample statistical test or a probability of clinical misdiagnosis.

For the example pair, the application selects the largest standardized challenge difference among walking matches in the test set. Its three-angle RMS differences were 0.63° in walking and 3.83° in the challenge. Median differences across all held-out pairs were 0.82° and 1.16°. The stored cases include the abductor group force, sagittal passive torque and angle trajectories for each task. The main 8,192 simulated trials and the additional 2,560 controller and load-sensitivity trials took place entirely within the model.

The weakened-capacity scenarios averaged 45.08° of sagittal hip excursion in stance versus 43.57° for the added-extension-resistance scenarios. The 1.51° difference is much smaller than the published 11.3° (95% CI 9.1–13.5°) between symptomatic gluteal tendinopathy and hip osteoarthritis groups. Clinical phenotypes, patient heterogeneity, and targets inferred from a single healthy person's repeated walks limit transfer. The aggregate comparison never enters the solver or classifier. [Allison et al., Table 2](https://research-repository.griffith.edu.au/server/api/core/bitstreams/a6d57c6f-31a4-42ef-b226-9dab1fed15dd/content)

Four separate, archived, converged 3D forward solutions supply an additional reference for mechanism sensitivity. At 1.33 m/s, with identical objective weights, contact model, mesh and quasi-random initial guess settings, decreasing bilateral hip actuators' maximum isometric force by 0%, 50%, 75% and 90% produced full-cycle adduction excursions of 7.96°, 5.13°, 13.32° and 21.85°. Those Falisse simulations use a different intervention from isolated abductor weakening. The new study imports the published results for this comparison.

## Reproduce

From the repository root, run with Python, NumPy and SciPy.

```sh
python3 hip-study/forward/experiment.py
```

To regenerate the bundled reference, download `ExperimentalData/ExperimentalData.mat`, `Results/PredSim_all/Results_all.mat` and `OpenSimModel/subject1/subject1.osim` from the pinned repository above. Pass the three paths in that order to `python3 hip-study/forward/reference.py`.
