# Supplied Swin notebook results

Updated November 17, 2026, 10:11 AM EST. **Complete and independently accepted.** The default dual model trained for **17 epochs**, selected **epoch 7** using validation only, and then scored all **2,116 test images**. One GPU fit completed in 16m35s; the independent CPU check completed in 12s. No optional model or seed sweep was run.

| Score | Checked result | Correct / denominator |
| --- | ---: | ---: |
| Binary benign/malignant accuracy | 93.38% | 1976 / 2116 |
| Final five-category accuracy | **64.37%** | 1362 / 2116 |
| Five-category macro F1 | 0.441602 | All five classes equally weighted |
| Malignant recall | 96.14% | 1,468 / 1,527 |
| Benign recall | 86.25% | 508 / 589 |
| End-to-end malignant subtype correctness | 55.93% | 854 / 1,527 |
| Subtype accuracy among correctly routed malignant images | 58.17% | 854 / 1,468 |
| Binary ROC AUC | 0.975982 | All 2,116 test images |
| Five-category unweighted kappa | 0.474234 | All 2,116 test images |

The broad benign/malignant decision is much stronger than the cancer-type decision. Binary accuracy does not require the malignant subtype to be correct. The final-five score does. No 94–95% final-five result was obtained.

| Final label | Correct / test images | Recall |
| --- | ---: | ---: |
| Benign | 508 / 589 | 86.25% |
| Ductal carcinoma | 737 / 1017 | 72.47% |
| Lobular carcinoma | 82 / 197 | 41.62% |
| Mucinous carcinoma | 34 / 174 | 19.54% |
| Papillary carcinoma | 1 / 139 | 0.72% |

Papillary typing is the largest measured weakness: only one of 139 images received its correct final label. This is a performance finding, not a diagnosis of the biological cause. Limited fitting cases, the supplied split and representation/training choices are possible contributors requiring a matched study.

| Test magnification | Images | Binary accuracy | Final-five accuracy | Five-class macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 40× | 545 | 89.72% | 58.90% | 0.427414 |
| 100× | 561 | 96.61% | 68.81% | 0.486977 |
| 200× | 536 | 94.78% | 67.35% | 0.445703 |
| 400× | 474 | 92.19% | 62.03% | 0.389638 |

These are per-image scores at each zoom, without multiview fusion. All four magnifications were included. The separate authored ID/true-target voting gives **18/24 = 75% final-five accuracy**, **24/24 = 100% binary accuracy**, and macro F1 .569855. Its 24 groups come from 23 bare IDs because one ID has two targets. It uses the true target to define voting groups and is not a deployable one-answer-per-patient protocol or the main capstone's 15-subset case endpoint. Keep it secondary.

The accepted split is **5,104 training / 689 validation / 2,116 test images**, grouped into **51 / 7 / 23 bare filename IDs**. No bare ID crosses roles; four conservative identifier stems do. Clinical relationships remain unresolved. Validation has no lobular images. Previous project development used all available keys, so this new within-run test is not a globally untouched clinical cohort. Matching published class quotas does not establish the paper's exact folds.

Swin-Tiny is ImageNet-pretrained. Embeddings/stages 0–1 are frozen; stages 2–3 and the binary/malignant-four heads train. A 768-number image description feeds both heads; stopped-gradient binary probabilities condition the malignant head. Benign samples are masked from its subtype loss. Final routing is malignant probability ≥ .5, otherwise benign. Total parameters **27,879,180**; trainable **26,358,078**. [Fixed recipe and declared compatibility repairs](SUPPLIED_SWIN_RUN.md).

The original is privately preserved and unchanged. Execution source `a50aeee39a78b3210a5a8c33905f6566a0563875` records pinned native TensorFlow weights, all data/split/configuration hashes, complete history and class probabilities. GPU fit 4985804 and independent check 4985805 both completed with exit 0. Selected-checkpoint replay on 32 images differed by at most 6.86e-7 for binary and 3.88e-7 for malignant probabilities, below the fixed 1e-5 tolerance. No new fitting or inference was performed by the independent saved-result checker.

Technical preflight passed the masked-loss training/save/reload check and 16 exact TensorFlow inverse-window/output/gradient cases. Earlier failed technical attempts are preserved and produced no research accuracy. The isolated old-window probe did not reproduce the full-model graph failure; the documented optimizer cause remains a supported diagnosis. No learned-weight omission was permitted. [Protocol review](reviews/SUPPLIED_SWIN_REVIEW.md), [independent actual-results review](reviews/SUPPLIED_SWIN_ACTUAL_RESULTS_REVIEW.md), [input audit](evidence/SUPPLIED_SWIN_INPUT_AUDIT.json), [technical evidence](evidence/SUPPLIED_SWIN_TECHNICAL.json), [complete accepted results](evidence/SUPPLIED_SWIN_RESULTS.json).

Both requested notebooks are complete. [Compare them with the retained main approach](SUPPLIED_NOTEBOOKS_COMPARISON.md). Their different targets, units and splits prevent a model-only ranking. This separate notebook execution does not replace the existing five-category provider-case scorecard or establish novelty, clinical validation or a matched published-paper improvement.
