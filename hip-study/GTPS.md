# Frontal hip motion in symptomatic walking cohorts

Women in the GTPS workbook moved through an average **4.23°** of hip adduction/abduction during the middle 60% of stance. Women awaiting hip replacement moved through **2.52°**. The groups’ central ranges overlap between **1.99° and 4.06°**. That interval contains 24 of 51 GTPS records and 11 of 24 OA participants. A video measure that estimates this excursion accurately would still encounter similar values in both groups. [Robinson et al.](https://doi.org/10.1016/j.gaitpost.2019.02.031) · [Bertaux et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9276684/)

| Normalized stance window | GTPS mean | OA mean | GTPS minus OA, with 95% bootstrap interval |
|---|---:|---:|---:|
| **20–80%**, primary | **4.23°** | **2.52°** | **1.72° (0.95 to 2.50)** |
| 10–90% | 6.72° | 3.89° | 2.83° (1.89 to 3.76) |
| 0–100% of available contact | 11.46° | 5.39° | 6.07° (4.92 to 7.27) |

The difference grows when the analysis includes more of early and late stance. These are also the phases most sensitive to how each lab identifies contact. The primary result uses 20–80% to reduce that sensitivity. The mean difference is a cross-site descriptive result. Each diagnosis comes from one laboratory, so the data cannot distinguish effects of diagnosis from differences in acquisition, recruited participants and modeling. No video keypoint estimator is involved in these measurements.

## Who contributed data

**GTPS.** Robinson and colleagues recruited women with at least three months of lateral hip pain, tenderness over the greater trochanter, and pain reproduced on a clinical provocation test. They excluded women with clinical signs or symptoms of hip joint pathology. Participants walked barefoot at comfortable speed while the team collected six affected-foot strikes. They used an instrumented walkway and the University of Western Australia lower-body model. The untaped baseline sheet contains 51 distinct numeric IDs, each with a unique, complete 101-point frontal hip waveform. The paper reports 50 women, with mean age 57.4 years (SD 13.8). All 51 deposited records enter this analysis. Omitting any one produces a central-stance mean difference between 1.60° and 1.79°. The identity or reason for the count discrepancy remains unresolved. [Methods and recruitment, pp. 2–5](https://www.dynamictapefrance.fr/wp-content/uploads/2019/03/Robinson-2019-Gait-Posture.pdf) · [Public GTPS workbook](https://data.mendeley.com/datasets/66mcwyk854/1)

**Hip OA.** Bertaux and colleagues collected barefoot walking trials before unilateral hip replacement. We selected women and used each person’s recorded affected side. Of 55 eligible women with 861 affected-side trial records, 24 women had at least one complete force-plate contact inside the exported stride and a finite hip curve. These women contributed 68 contacts, with a mean age of 68.25 years (SD 10.69). The force criterion selects a subset. The GTPS gait workbook has no individual ages or walking speeds for matching. Concurrent GTPS status in the OA group is unknown. [OA protocol and kinematic definitions, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9276684/) · [OA participant metadata](https://doi.org/10.6084/m9.figshare.13655975.v10)

## Measurement

1. Read the published `Baseline` sheet’s `RHipAddAbd` curves at 0–100% stance. The protocol describes affected-foot collection. The exported channel prefix alone does not identify each woman's anatomical side.
2. Read the Bertaux `LHipAngles_Y` or `RHipAngles_Y` channel in degrees. This is Plug-in Gait’s second hip-angle component, corresponding to adduction/abduction. Verify the recorded OA side against the metadata. Verify and collapse duplicate 101-point waveforms with multiple cycle labels within a trial.
3. Pair each OA angle trace with the same side, trial and time grid of vertical force. Require a normalized-force peak of at least 60 on the archive scale and one interior, contiguous run of 40–80 samples above 5% of that peak. Resample that loaded interval to 101 points and average contacts within each person.
4. For each participant waveform, compute the maximum minus the minimum in the selected window. This quantity is unchanged by reversing an angle’s sign or adding a constant offset. Different biomechanical models can still give different excursions.
5. Give each participant equal weight. Estimate intervals with 10,000 seeded bootstrap resamples of individuals within each cohort. Compute the shared central interval by intersecting the groups’ empirical 10th–90th percentiles. These uncertainty intervals address sampling within the two selected groups. They omit uncertainty from site differences and contact definitions.

## Reproduce

Use Python 3.10 or newer, NumPy and SciPy. Download the first GTPS workbook and the two files from the Bertaux releases. The OA archive is about 822 MB. The script also accepts a directory containing the extracted preoperative `HOA_M0_` ARFFs for left/right `HipAngles_Y`, `HipAngles_time`, `NormalisedGRF_Z` and `NormalisedGRF_time`. [GTPS workbook](https://data.mendeley.com/public-files/datasets/66mcwyk854/files/b26da87d-3bca-49a9-b7e2-b3655f3dff71/file_downloaded) · [OA subject-ID ARFF archive](https://ndownloader.figshare.com/files/26821445) · [OA workbook](https://ndownloader.figshare.com/files/34177680)

```sh
python hip-study/tools/prepare_gtps.py \
  --gtps /path/to/gtps-baseline.xlsx \
  --oa /path/to/bertaux-subject-id.zip \
  --metadata /path/to/bertaux-metadata.xlsx
```

The output `hip-study/data/gtps.json` contains every derived participant value, inclusion counts, source identifiers and the window-sensitivity analysis. The gait data and metadata are credited to Fearon, Spratford, Welvaert, Gaida and Robinson, and to Bertaux and colleagues. Both deposits use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
