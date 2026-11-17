# Matched BreakHis comparison: completed results

November 17, 2026. **All three selected test evaluations and the independent checker are complete and accepted.** Midnight P has the highest observed average final-five accuracy and macro F1 in this comparison. CNN has higher all-four-view macro F1, Swin has stronger worst-subset performance, and all paired uncertainty intervals include zero; this small experiment does not establish reliable population superiority.

The task is the same for all three: decide benign or malignant, stop for benign, and otherwise choose ductal, lobular, mucinous or papillary carcinoma. These are five final labels. Only BreakHis supplies fitting and evaluation data.

| Method | Mean-15 final-five accuracy | Mean-15 macro F1 | All-four final-five accuracy | All-four macro F1 | Worst-subset accuracy | Worst-subset macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Midnight P (our retained method) | 67.50% | 0.375675 | 68.75% | 0.341176 | 37.50% | 0.228571 |
| CNN–Transformer adaptation | 56.67% | 0.348180 | 56.25% | 0.372857 | 50.00% | 0.245000 |
| Swin adaptation | 59.17% | 0.304947 | 56.25% | 0.278992 | 56.25% | 0.278992 |

Mean-15 gives every one of the 15 nonempty magnification subsets equal importance, using the same 16 test cases each time. The 240 case/subset events are dependent views of 16 cases, not 240 independent test specimens. The all-four and worst-subset columns are different endpoints; do not substitute one for the registered mean-15 primary macro F1.

## What 94.58% means

| Method | Mean-15 benign/malignant accuracy: first step only | Mean-15 final-five accuracy: complete system |
| --- | ---: | ---: |
| Midnight P (our retained method) | 94.58% | 67.50% |
| CNN–Transformer adaptation | 86.25% | 56.67% |
| Swin adaptation | 82.50% | 59.17% |

**94.58% is Midnight P’s first-step binary accuracy, not its overall five-category result. The complete-system result is 67.50%; 94–95% final accuracy remains unachieved.** Naming the malignant subtype remains a major weakness. With all four zooms, P routes all 16 cases correctly as benign/malignant, but gives the correct final type for only 11 of 16: five benign and six ductal cases. Five malignant cases are mistyped.

Across all 15 subsets, P makes 162 correct final answers out of 240 dependent case/subset events. Its 78 errors comprise 13 routing errors and 65 malignant-subtype errors after correct routing. These are descriptive counts on reused cases, not new independent samples or evidence of a particular morphological cause.

## Shared data, inputs and selection

| Role | Provider cases | Conservative groups | Original images | Case supports: benign / ductal / lobular / mucinous / papillary |
| --- | ---: | ---: | ---: | --- |
| Training | 49 | 42 | 4,621 | 14 / 24 / 2 / 5 / 4 |
| Validation | 17 | 15 | 1,584 | 5 / 7 / 2 / 2 / 1 |
| Test | 16 | 13 | 1,704 | 5 / 7 / 1 / 2 / 1 |

All related identifier stems, images, magnifications and derived crops stay in one role. These are conservative provider-identifier groups, not verified clinical patients. Every case already entered earlier project development, so this is a new matched reused-development holdout rather than a globally fresh clinical test or five-fold cross-validation.

All three share the same original fields, Pillow RGB short-side-224 bilinear resize and three left/center/right crops. Each reader averages crop descriptors before its field heads. Joint five-label field probabilities are averaged within each zoom, then equally over the available zooms. The pooled malignant probability is routed at 0.5; malignant takes the largest malignant component. No true-label voting group or equal-field-number cross-zoom correspondence is used.

Midnight P uses frozen Midnight-12k pathology descriptors with fitting-only head-specific centers and balanced cosine support densities. CNN uses the supplied ImageNet EfficientNetV2B0 plus four Transformer blocks, adapted to binary-two plus malignant-four. Swin uses the supplied ImageNet Swin-Tiny with the same output task. Both neural models start from original pretraining and complete 40 epochs, 11,560 updates each, with the registered class/case/view/field weighting and fixed learning-rate schedules. Pretraining, parameter counts, augmentation and learning algorithms differ intentionally: this compares whole pipelines, not isolated backbones with equal compute.

Each method receives four visible validation choices. First-maximum validation mean-15 macro F1 selects P temperature **0.1**, CNN epoch **40**, and Swin epoch **20**. All three selection hashes were sealed before selected test evaluation. There is no train-plus-validation refit, test-driven threshold fitting, extra candidate or selection change. [Fixed protocol](MATCHED_THREE_WAY_SPEC.md) · [configuration](../configs/breakhis-matched-three-way-v1.json).

The selected validation means were P 79.61% / F1 .577663, CNN 59.22% / .484162, and Swin 59.61% / .393336. Those remain validation-selection results and are not the test table above. [Preserved validation evidence](evidence/MATCHED_THREE_WAY_VALIDATION.json) · [dated partial review](reviews/MATCHED_THREE_WAY_PARTIAL_RESULTS_REVIEW.md).

## Which categories were missed?

This table gives final-five class recall averaged over the 15 subsets, with the same cases repeated in each condition. A recall of zero refers to the listed held-out cases, not a demonstrated inability on every possible case of that disease.

| True category | Test cases | Midnight P recall | CNN–Transformer recall | Swin recall |
| --- | ---: | ---: | ---: | ---: |
| Benign | 5 | 89.33% | 62.67% | 45.33% |
| Ductal | 7 | 84.76% | 75.24% | 100.00% |
| Lobular | 1 | 0.00% | 0.00% | 0.00% |
| Mucinous | 2 | 20.00% | 33.33% | 10.00% |
| Papillary | 1 | 0.00% | 0.00% | 0.00% |

All three miss the single test lobular case and single test papillary case across the evaluated subsets. Mucinous typing is also weak. The small class supports limit generalization claims; these observations do not diagnose label errors, staining problems or a biological mechanism without examining the original fields and appropriate expert review.

## Uncertainty and ranking

The registered paired descriptive bootstrap resamples the same 13 conservative groups for all methods, preserving all cases and all 15 masks per group. It uses 2,000 seed-812 draws, fixed five labels and linear 2.5/97.5 percentiles. In 1,533 draws at least one true class is absent, reflecting the sparse class supports. These are exploratory intervals without an independent clinical cohort or multiplicity adjustment.

| Difference: first method minus second | Mean macro-F1 difference [95% descriptive interval] | Accuracy difference in percentage points [95% interval] |
| --- | --- | --- |
| Midnight P − CNN | +0.0275 [-0.0687, +0.0987] | +10.83 [-10.42, +30.00] |
| Midnight P − Swin | +0.0707 [-0.0448, +0.1504] | +8.33 [-15.24, +31.93] |
| CNN − Swin | +0.0432 [-0.0664, +0.1080] | -2.50 [-26.67, +12.59] |

Every pairwise interval includes zero. P leads the observed averages, while Swin has the best worst-subset accuracy and macro F1. A universal or statistically established winner is not supported. Do not compare these raw percentages directly with the earlier image-random eight-type CNN notebook or differently grouped Swin notebook scores.

## Execution, failures and verification

Both complete neural trajectories ran on A40: CNN 54m32s, Swin 2h07m59s, job array 4986295. The common selection seal 4986296 and selected P test 4986298 completed. An attempted move of unstarted selected neural test inference to available A30 GPUs failed the strict first-16-validation-field probability replay guard before test access. A validation-only diagnostic measured the discrepancies; compiling did not remove them. The failed runs, empty output directories, diagnostic arrays, logs and fail-closed checker are retained. The original 1e-5 guard was never relaxed.

Selected-only A40 tests **4987130_0/_1 completed in 1m24s/1m28s**, with **zero binary and subtype reload error** against the saved validation outputs for both models. Thus the selected weights reproduce the registered outputs on the original hardware; the observed A30 discrepancy does not identify its exact numerical cause. Independent CPU checker **4987131 completed in 1m20s** and reports `complete accepted matched three-way comparison`, with an empty private error record. Scheduler exit 0 alone was not treated as acceptance.

The checker replays split membership, field weights, source/cache/checkpoint hashes, all candidate selections, joint pooling/routing, test metrics and paired bootstrap. P’s fitting-only cached-descriptor geometry replay has maximum field-probability difference **0.0**. A separate reviewer independently replays all 12 validation candidates, all three test NPZs and the bootstrap using separate arithmetic. Neither independent calculation is a new raw-image backbone execution. [Final independent review](reviews/MATCHED_THREE_WAY_RESULTS_REVIEW.md).

Immutable scientific producer: `a6e6a54e4fac090051837df17b0dea14efca3bb3`. Configuration SHA-256: `1cdc7f143941a25be7b595c3b058e545c57c19a3a367df68123d977f757d5d5c`. The accepted [complete evidence JSON](evidence/MATCHED_THREE_WAY_RESULTS.json) is copied byte-for-byte from the checker, SHA-256 `3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62`. [Execution history](MATCHED_THREE_WAY_STATUS.md).

The earlier interrupted four-epoch CNN fitting attempt remains archived and disclosed in the execution history. No negative outcome was removed. The earlier 46/8/28 trial scorecards and supplied original notebook results remain separate historical evidence.

The immediate capstone work is to report these endpoints clearly and review the malignant-subtype errors before specifying another intervention. A training modification would need a new fixed comparison and appropriate evaluation; this completed holdout must not become an unacknowledged tuning target. TCGA expansion and novelty claims remain deferred.
