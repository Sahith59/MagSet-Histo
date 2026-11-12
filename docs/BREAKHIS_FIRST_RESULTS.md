# BreakHis: first checked baseline deliverable

Updated November 12, 2026. **BreakHis is the main dataset. TCGA follow-up is deferred.** This milestone packages existing baseline work; it does not require a novel method or a completed paper.

## What the model does

Give it the images of one provider case at available microscope zooms. A fixed Phikon-v2 image reader creates numerical descriptions. The saved mean-pooling and logistic classifier produce eight subtype probabilities. No exact cross-zoom field pairing is assumed.

## First numbers: available now

| Validation split | Mean F1 over 15 view settings | Mean case accuracy over 15 settings | All-view F1 | All-view case accuracy | Distinct validation cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3843 | 45.7% | 0.4083 | 47.5% | 22 |
| 813 | 0.3169 | 38.3% | 0.3083 | 37.5% | 20 |

These are modest exploratory results. F1 is not accuracy. Every number averages five repeated provider trials; each trial has 46 fitting, 8 validation and 28 excluded outer-role cases. Each view setting has 40 case evaluations across trials, with repeated cases. Across both splits there are 31 distinct validation cases, not 1,200 independent patients.

A second split is shown to expose sensitivity, not to select the better split. The full table below prevents one favorable trial or demonstration from standing in for overall performance.

## Every trial

| Split | Trial | Selected C | Mean-subset F1 | Mean-subset accuracy | All-view F1 | All-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 0.01 | 0.6042 | 66.7% | 0.7083 | 75.0% |
| 812 | 2 | 1 | 0.2567 | 26.7% | 0.2500 | 25.0% |
| 812 | 3 | 10 | 0.3092 | 41.7% | 0.2500 | 37.5% |
| 812 | 4 | 0.01 | 0.2597 | 35.0% | 0.2917 | 37.5% |
| 812 | 5 | 10 | 0.4917 | 58.3% | 0.5417 | 62.5% |
| 813 | 1 | 0.01 | 0.3444 | 40.8% | 0.3333 | 37.5% |
| 813 | 2 | 0.1 | 0.3256 | 36.7% | 0.3333 | 37.5% |
| 813 | 3 | 0.1 | 0.4075 | 48.3% | 0.3750 | 50.0% |
| 813 | 4 | 0.01 | 0.2958 | 40.0% | 0.2917 | 37.5% |
| 813 | 5 | 10 | 0.2111 | 25.8% | 0.2083 | 25.0% |

## Class weaknesses

Recall below uses all saved case/subset records within each split; cases recur. These are descriptive counts, not independent clinical estimates.

| Subtype | Split 812 recall | Distinct cases | Split 813 recall | Distinct cases |
| --- | ---: | ---: | ---: | ---: |
| adenosis | 0.600 | 2 | 0.600 | 2 |
| ductal_carcinoma | 0.800 | 4 | 0.893 | 2 |
| fibroadenoma | 0.347 | 3 | 0.307 | 2 |
| lobular_carcinoma | 0.587 | 3 | 0.587 | 3 |
| mucinous_carcinoma | 0.760 | 2 | 0.467 | 2 |
| papillary_carcinoma | 0.133 | 3 | 0.040 | 3 |
| phyllodes_tumor | 0.000 | 2 | 0.000 | 2 |
| tubular_adenoma | 0.427 | 3 | 0.173 | 4 |

## Working prediction demonstration

Verified the existing trial1/seed812 saved head against all 120 archived validation records, then re-encoded every original image of the first sorted validation case and replayed all 15 view settings. The case was chosen by metadata order, not by correctness. This demonstrates execution and parity, not new accuracy or a successful diagnosis.

Images checked: **121**. Maximum archived-probability difference: **4.44e-16** from cached replay and **5.52e-08** from raw images. SLURM job `4959781`; source `9448a7e319167b7417670a882552edb372603fd3`. No new fitting or outer-role scoring.

## What is complete and what is not

- Verified ten baseline run reports and all 1,200 saved validation records against trusted labels, splits and metric arithmetic.
- Data comprise 7,909 images at 40×/100×/200×/400× and eight subtypes. The primary input is a whole provider-case bag, not one independent patient per image.
- All 82 provider cases have entered development somewhere. Clinical identifier relationships remain unresolved; no globally fresh BreakHis confirmation cohort remains.
- These numbers do not establish a solved task, clinical readiness, a novel method or a completed research paper. The historical notebook image accuracies are from a different split/selection procedure and are not comparable targets.
- Next: package this baseline, then run one registered BreakHis improvement comparison. Further TCGA work is off the immediate path.

[Focused plan](BREAKHIS_MILESTONE_PLAN.md) · [Training strategy](TRAINING_STRATEGY.md) · [Registered evaluation](BENCHMARK_PROTOCOL.md)
