# BreakHis supervised ResNet-50 reference: negative result

November14,2026. **All ten runs and the complete artifact checker passed. The registered improvement gate failed. Keep the frozen Phikon-v2 reference.**

In plain language: teaching this standard picture model did not improve our main score. It learns the fitting cases very well but struggles with different validation cases. This result concerns one fixed pipeline; it does not establish that all supervised models fail or identify one sole cause.

## Main comparison

Scores average all15 input combinations and five overlapping trials per arrangement. Macro F1 treats all eight classes equally and is not accuracy.

| Arrangement | Frozen F1 | ResNet F1 | F1 change | Frozen case accuracy | ResNet case accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3843 | 0.3084 | -0.0759 | 45.7% | 37.0% |
| 813 | 0.3169 | 0.3006 | -0.0163 | 38.3% | 40.0% |

F1 decreases on both arrangements. Accuracy increases on813 but decreases on812; neither this mixed accuracy result nor an individual trial overrides the predeclared F1 gate. Phyllodes validation recall remains0 on both.

## Every paired trial

| Arrangement | Trial | Frozen F1 | ResNet F1 | Frozen accuracy | ResNet accuracy | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 0.6042 | 0.4200 | 66.7% | 48.3% | 10 |
| 812 | 2 | 0.2567 | 0.2500 | 26.7% | 25.0% | 10 |
| 812 | 3 | 0.3092 | 0.3247 | 41.7% | 43.3% | 20 |
| 812 | 4 | 0.2597 | 0.2844 | 35.0% | 32.5% | 5 |
| 812 | 5 | 0.4917 | 0.2628 | 58.3% | 35.8% | 15 |
| 813 | 1 | 0.3444 | 0.4222 | 40.8% | 47.5% | 10 |
| 813 | 2 | 0.3256 | 0.3178 | 36.7% | 40.0% | 15 |
| 813 | 3 | 0.4075 | 0.3414 | 48.3% | 47.5% | 10 |
| 813 | 4 | 0.2958 | 0.2811 | 40.0% | 40.8% | 5 |
| 813 | 5 | 0.2111 | 0.1406 | 25.8% | 24.2% | 10 |

## Fitting versus validation

| Arrangement | Fitting balanced case accuracy | Validation balanced case accuracy | Fitting weighted field accuracy | Validation weighted field accuracy |
| --- | ---: | ---: | ---: | ---: |
| 812 | 99.6% | 37.0% | 97.5% | 31.2% |
| 813 | 99.7% | 40.0% | 97.8% | 32.6% |

Field weights give equal influence to class, case within class and zoom within case; fields within a zoom are uniform. Case and field endpoints differ. Canonical evaluation cross-entropy also differs from augmented label-smoothed training loss. The large fitting–validation gap persists.

## Full-view, worst-view and probability scores

| Arrangement | Pipeline | All-view F1 | All-view accuracy | Mean trial worst-subset F1 | Brier error |
| --- | --- | ---: | ---: | ---: | ---: |
| 812 | frozen | 0.4083 | 47.5% | 0.2583 | 0.8614 |
| 812 | supervised | 0.3533 | 40.0% | 0.1658 | 0.9210 |
| 813 | frozen | 0.3083 | 37.5% | 0.2475 | 0.8314 |
| 813 | supervised | 0.2975 | 40.0% | 0.2200 | 0.8398 |

Brier is the mean sum of eight squared probability errors; lower is better. All-view inputs also remain weak.

## Subtype recall and distinct validation support

| Subtype | Frozen812 | ResNet812 | Frozen813 | ResNet813 | Distinct cases812 /813 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adenosis | 0.600 | 0.533 | 0.600 | 0.600 | 2 / 2 |
| ductal carcinoma | 0.800 | 0.667 | 0.893 | 0.840 | 4 / 2 |
| fibroadenoma | 0.347 | 0.507 | 0.307 | 0.587 | 3 / 2 |
| lobular carcinoma | 0.587 | 0.293 | 0.587 | 0.267 | 3 / 3 |
| mucinous carcinoma | 0.760 | 0.427 | 0.467 | 0.840 | 2 / 2 |
| papillary carcinoma | 0.133 | 0.000 | 0.040 | 0.053 | 3 / 3 |
| phyllodes tumor | 0.000 | 0.000 | 0.000 | 0.000 | 2 / 2 |
| tubular adenoma | 0.427 | 0.533 | 0.173 | 0.013 | 3 / 4 |

Each fitting set has one adenosis and one phyllodes provider case. Repeated fields and augmentation do not add independent cases. Two distinct phyllodes validation cases per arrangement are a very small support, and the same cases recur across settings.

## All15 view combinations

| Available views | Frozen812 F1 | ResNet812 F1 | Frozen813 F1 | ResNet813 F1 |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.3625 | 0.3017 | 0.2683 | 0.2783 |
| 100 | 0.3500 | 0.3142 | 0.2792 | 0.2883 |
| 200 | 0.3933 | 0.2283 | 0.3333 | 0.3167 |
| 400 | 0.2767 | 0.1908 | 0.3142 | 0.2708 |
| 40+100 | 0.3833 | 0.3767 | 0.3183 | 0.3075 |
| 40+200 | 0.4292 | 0.3158 | 0.3408 | 0.3183 |
| 40+400 | 0.3892 | 0.3283 | 0.3183 | 0.3017 |
| 100+200 | 0.4267 | 0.3242 | 0.2917 | 0.2950 |
| 100+400 | 0.3433 | 0.3283 | 0.3375 | 0.2975 |
| 200+400 | 0.3683 | 0.2300 | 0.3125 | 0.3042 |
| 40+100+200 | 0.4333 | 0.3658 | 0.3167 | 0.3433 |
| 40+100+400 | 0.3875 | 0.3533 | 0.3517 | 0.2658 |
| 40+200+400 | 0.4083 | 0.3200 | 0.3500 | 0.3267 |
| 100+200+400 | 0.4042 | 0.2950 | 0.3125 | 0.2975 |
| 40+100+200+400 | 0.4083 | 0.3533 | 0.3083 | 0.2975 |

## Recipe, execution and verification

One ordinary complete-pipeline reference: ImageNet-initialized ResNet-50, two head-only epochs then18 full-backbone epochs, fixed BatchNorm running buffers, class/case/view-balanced fitting draws, mild fitting augmentation, AdamW and label smoothing. Whole fields are resized to224square. Keep the trained image head; average probabilities within zoom, then equally across available zooms. Select the earliest best validation F1 at epochs5/10/15/20; complete all20. [Registered specification](BREAKHIS_SUPERVISED_SPEC.md).

The same46 fitting/eight validation/28 excluded provider roles and all15 view subsets are retained. This is a complete-pipeline comparison, not a same-encoder fusion ablation, identical optimization budget or exact published-method reproduction. No outer images are scored.

Eight original tasks completed in14m16s–17m29s. Two timed out after2h00m18s. Both A40 recovery attempts failed the unchanged selected-field probability replay safeguard; their states were retained. Original pre-A40 committed checkpoints were restored after hash verification. A100 continuations finished in1m59s/3m24s and passed replay. No tolerance or scientific setting was relaxed. The collector completed in58seconds with an empty error log. [Detailed execution record](BREAKHIS_SUPERVISED_STATUS.md).

For the recovered cases, total active scheduled segments are7,406s and7,549s, including original timeout, rejected A40 attempt and final A100 continuation. The119s/204s final invocation is not their total training cost. Repeated partial work is disclosed, not an extra selected model. The model has23,524,424 parameters;16,392 train during head-only warmup.

The full checker verifies exact source/configuration/roles, canonical model tensors and BatchNorm buffers, AdamW moments/update budgets, all four selection archives and earliest-maximum choice, selected/last snapshots, complete field coverage, stable probability arithmetic and every saved case/subset metric. Independent actual-output review is recorded in the [review ledger](reviews/BREAKHIS_SUPERVISED_REVIEW.md); its exact scope and limitations govern any acceptance claim.

Training source `c1b6b0ce24ffc99a00b3eb98cc09130934ffb0bb`; collector source `f4ba03ae1278ac7ecf1897328353abd43444b1f3`; private complete comparison SHA256 `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`. [Public aggregate/hash ledger](evidence/BREAKHIS_SUPERVISED_RESULTS.json) retains confusion matrices, per-subset metrics, field losses and all artifact hashes without provider identifiers.

## Decision and remaining work

The predeclared screen requires at least+0.02 mean-subset F1 on both arrangements and positive phyllodes recall on both. It failed. Keep the frozen reference and retain this negative result. Do not launch an automatic backbone/rate/epoch sweep, resplit to improve scores or score outer cases as a rescue.

These are exploratory provider-case validations:22/20 distinct validation cases,31 in their union, overlapping trials and reused development data. All82 cases entered development somewhere; clinical identifier relationships remain unresolved and no globally untouched BreakHis confirmation cohort remains. Neither reliable clinical performance nor a novel mechanism is established.

Finish the BreakHis project report with the checked baseline, both failed improvement recipes, fitting–validation evidence and protocol limits. Agree on the intended endpoint and credible final-evaluation route with Professor Dong before registering another model experiment. TCGA stays deferred.
