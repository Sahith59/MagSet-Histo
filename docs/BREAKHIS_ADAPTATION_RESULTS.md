# BreakHis: fixed last-block adaptation result

November 12, 2026 (America/New_York). **All ten production runs and the complete artifact checker finished successfully. The registered two-split improvement gate failed. Keep the frozen baseline as the default.** BreakHis remains the main project; TCGA is deferred.

## Main result

These are mean case scores over all 15 nonempty view combinations and five overlapping provider trials per inner split. F1 is not accuracy.

| Split | Frozen F1 | Adapted F1 | F1 change | Frozen accuracy | Adapted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3843 | 0.3840 | -0.0003 | 45.7% | 45.7% |
| 813 | 0.3169 | 0.3205 | +0.0036 | 38.3% | 38.8% |

The gain is mixed and tiny: the first split loses primary F1 while its accuracy stays unchanged; the second adds only 3 correct decisions among 600 repeated case/subset records. Phyllodes recall remains 0 on both splits. This recipe does not provide a meaningful accuracy improvement. It is not a novel-method success.

## All ten paired trials

| Split | Trial | Frozen mean F1 | Adapted mean F1 | Frozen mean accuracy | Adapted mean accuracy | Adapted C |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 0.6042 | 0.6153 | 66.7% | 67.5% | 0.01 |
| 812 | 2 | 0.2567 | 0.2567 | 26.7% | 26.7% | 1 |
| 812 | 3 | 0.3092 | 0.3106 | 41.7% | 41.7% | 10 |
| 812 | 4 | 0.2597 | 0.2486 | 35.0% | 34.2% | 0.01 |
| 812 | 5 | 0.4917 | 0.4889 | 58.3% | 58.3% | 10 |
| 813 | 1 | 0.3444 | 0.3550 | 40.8% | 41.7% | 0.01 |
| 813 | 2 | 0.3256 | 0.3311 | 36.7% | 37.5% | 0.1 |
| 813 | 3 | 0.4075 | 0.4103 | 48.3% | 49.2% | 0.1 |
| 813 | 4 | 0.2958 | 0.3000 | 40.0% | 40.8% | 0.01 |
| 813 | 5 | 0.2111 | 0.2061 | 25.8% | 25.0% | 0.1 |

## Full-view and probability results

| Split | Frozen all-view F1 | Adapted all-view F1 | Frozen all-view accuracy | Adapted all-view accuracy | Frozen Brier | Adapted Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.4083 | 0.4125 | 47.5% | 47.5% | 0.8614 | 0.8583 |
| 813 | 0.3083 | 0.3292 | 37.5% | 40.0% | 0.8314 | 0.8034 |

Brier is the mean sum of eight squared probability errors; lower is better. Its descriptive improvement does not change the failed primary gate. Full-view results also remain weak, so missing views alone cannot explain the low accuracy.

## Subtype recall

| Subtype | Frozen812 | Adapted812 | Frozen813 | Adapted813 | Distinct validation cases812 /813 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adenosis | 0.600 | 0.600 | 0.600 | 0.600 | 2 / 2 |
| ductal_carcinoma | 0.800 | 0.800 | 0.893 | 0.907 | 4 / 2 |
| fibroadenoma | 0.347 | 0.347 | 0.307 | 0.320 | 3 / 2 |
| lobular_carcinoma | 0.587 | 0.573 | 0.587 | 0.587 | 3 / 3 |
| mucinous_carcinoma | 0.760 | 0.773 | 0.467 | 0.480 | 2 / 2 |
| papillary_carcinoma | 0.133 | 0.133 | 0.040 | 0.040 | 3 / 3 |
| phyllodes_tumor | 0.000 | 0.000 | 0.000 | 0.000 | 2 / 2 |
| tubular_adenoma | 0.427 | 0.427 | 0.173 | 0.173 | 3 / 4 |

## All view combinations

Each entry averages the corresponding subset F1 across five trials; no subset is selected after seeing results.

| Available views | Frozen812 F1 | Adapted812 F1 | Frozen813 F1 | Adapted813 F1 |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.3625 | 0.3583 | 0.2683 | 0.2700 |
| 100 | 0.3500 | 0.3500 | 0.2792 | 0.2708 |
| 200 | 0.3933 | 0.3683 | 0.3333 | 0.3375 |
| 400 | 0.2767 | 0.2767 | 0.3142 | 0.3142 |
| 40+100 | 0.3833 | 0.3833 | 0.3183 | 0.3350 |
| 40+200 | 0.4292 | 0.4292 | 0.3408 | 0.3408 |
| 40+400 | 0.3892 | 0.3892 | 0.3183 | 0.3458 |
| 100+200 | 0.4267 | 0.4183 | 0.2917 | 0.3083 |
| 100+400 | 0.3433 | 0.3767 | 0.3375 | 0.3125 |
| 200+400 | 0.3683 | 0.3683 | 0.3125 | 0.3125 |
| 40+100+200 | 0.4333 | 0.4333 | 0.3167 | 0.3000 |
| 40+100+400 | 0.3875 | 0.3833 | 0.3517 | 0.3517 |
| 40+200+400 | 0.4083 | 0.4083 | 0.3500 | 0.3667 |
| 100+200+400 | 0.4042 | 0.4042 | 0.3125 | 0.3125 |
| 40+100+200+400 | 0.4083 | 0.4125 | 0.3083 | 0.3292 |

## What was actually tested

One registered supervised recipe: pinned 24-block Phikon-v2, auxiliary Dropout+eight-class image head, one head-only warmup epoch then five epochs updating only the last block/final LayerNorm. AdamW encoder LR0.00001/head LR0.001, batch 16, class/case/view-balanced replacement draws, fitting-only flips. Each run used 46 fit / 8 validation / 28 excluded outer cases. Final encoder was reloaded; only the54 fit/validation cases were encoded. The same fit-only field-moment normalization and mean/mask logistic Cgrid were reused. The auxiliary image head was discarded for the reported case classifier. [Registered recipe](BREAKHIS_ADAPTATION_SPEC.md).

Exactly ten production fits plus one separately declared two-epoch/two-batch fitting-only technical smoke. No outer predictions or extra training configurations were produced.

## Execution and verification

Array 4962271: all 10 tasks COMPLETED0; individual elapsed6m48s–9m22s. First start to last finish24m34s. Collector 4962297 COMPLETED0 in30s; total from first start to checked completion25m11s, including gaps. No nonempty production error log at inspection.

Training source `3a8dde70d809b115928bf773511a0cd6e0f8e30c`; immutable collector source `d5567ee402dd11d75b295b33f16b1eebb5799f61`. The checker validated roles/truths, epochs/draws/freeze evidence, canonical tensor/AdamW schemas and update counters, source/checkpoint/cache/head hashes, exact54-case caches, fitting-only normalization, saved-head/cache probability replay, model selection and all 1,200 adapted validation records.

An independent local review checked every frozen/adapted probability record, trusted labels/splits, all subset/aggregate/class/confusion/Brier values and available artifact hashes. Full feature/checkpoint verification remains evidence from the scheduled collector; those production artifacts are retained on the cluster. [Review](reviews/BREAKHIS_ADAPTATION_REVIEW.md) · [Complete aggregate/hash ledger](evidence/BREAKHIS_ADAPTATION_RESULTS.json).

## Why this is concerning and what remains uncertain

Each fitting set has only one adenosis and one phyllodes provider case. More fields or GPUs do not create additional independent examples. Every validation set has one case per subtype, so its accuracy also equals balanced accuracy; frequent ductal cases cannot dominate that score. Eight cases per trial also make scores sensitive to which cases were selected.

Auxiliary image-training loss fell from0.796–0.894 in epoch1 to0.100–0.163 in epoch6 across the ten runs, while case-level validation barely moved. This confirms optimization on fitting examples and is consistent with limited transfer to unseen cases. The two losses/tasks differ, so it does not alone diagnose overfitting. Mean pooling may also dilute diagnostic fields; feature suitability, grouping/label relationships and pooling require targeted diagnostics. None is a proven sole cause.

Across1,200 paired repeated case/subset decisions, 38 predicted labels changed: 6 wrong-to-right, 3 right-to-wrong and 29 wrong-to-other-wrong. These are dependent records, not1,200 patients or a significance test.

All 82 provider cases have entered the wider development history, clinical identifier relationships remain unresolved, and only 31 distinct validation cases occur across both arrangements. No globally pristine confirmation cohort exists. Performance is exploratory and insufficient for a strong clinical or novelty claim.

## Next bounded work

Keep the frozen baseline. Diagnose saved-model fitting versus validation behavior and image-head versus case-pooling behavior before another training recipe. Audit a public eight-class patient-disjoint comparator and specify a compatible reproduction with identical cases/metrics. Do not automatically extend epochs, launch a model sweep, switch to TCGA, or score outer cases in response to this negative result. [Published benchmark context](BREAKHIS_PUBLIC_BENCHMARKS.md).
