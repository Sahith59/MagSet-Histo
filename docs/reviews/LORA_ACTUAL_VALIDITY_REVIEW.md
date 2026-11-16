# Actual LoRA validity and failure diagnosis

Reviewed November 16, 2026. Producer `7daff480a44b35f57ca859390a08d447cdd9df46`; complete comparison SHA256 `570892f048f3b8bdbf0095ed6c5dcdd4c5e6736ac2af86fe48d0db089e28072e`. This is a read-only review of the accepted aggregate, its 80 candidate summaries, the earlier frozen comparison and the hash-bound implementation. No new fitting, image/model inference, jobs, downloads or split changes were performed.

**Finding: no invalidating label, normalization, stage, loss or probability-pooling mismatch was found in the examined contract. The negative result is real within this development protocol.** The most informative qualification is that the deliberately fixed C=1 initialization was already substantially weaker than the validation-selected frozen field classifier. LoRA mostly preserved that weaker starting decision rule; it did not destroy a previously 70–74% selected model.

## Measured outcome and the C=1 bridge

All values below average the five trials and all 15 nonempty magnification subsets. Accuracy is distinct from macro F1.

| Arrangement/arm | Frozen fixed C=1 accuracy / F1 | After head-only epoch 2 | Selected LoRA checkpoint |
| --- | --- | --- | --- |
| 812 direct | 71.17% / .5107 | 71.00% / .5010 | 71.33% / .5101 |
| 812 two-step | 68.17% / .4870 | 68.17% / .4904 | 68.33% / .4944 |
| 813 direct | 66.00% / .4038 | 66.67% / .4079 | 67.50% / .4147 |
| 813 two-step | 59.50% / .3682 | 59.83% / .3754 | 59.83% / .3754 |

The earlier **validation-selected** frozen field two-step model scored 73.67%/.5732 and 70.00%/.5017. That comparison had four C choices. C=1 was intentionally fixed before LoRA fitting to avoid initializing from validation-selected heads; this is a registered statistical choice, not an implementation error. In the earlier field comparison, 15 of the 20 arm/arrangement/trial selections preferred C below 1. The strong starting-head regularization effect must be separated from encoder adaptation.

Selected epochs across the 20 LoRA runs were **12 at epoch 2, three at epoch 6, four at epoch 12, and one at epoch 20**. All five arrangement-813 two-step models selected epoch 2, before LoRA was enabled. Their later-epoch aggregate F1 was .3665/.3673/.3673 at epochs 6/12/20, versus .3754 at epoch 2. Arrangement-812 two-step aggregate F1 moved only from .4904 at epoch 2 to .4931 at epoch 20. These patterns do not support “train longer” as an evidence-backed remedy.

Fitting accuracy is 99.13% for every arm/arrangement aggregate. The per-trial fitting accuracies are 100%, 97.83%, 100%, 97.83%, 100%; this exact pattern already exists in the corresponding frozen C=1 heads. Its repetition is therefore not, by itself, evidence of copied results or an adaptation bug. Validation cross-entropy and predictions vary; the saved fitting decisions simply do not improve meaningfully from their already near-perfect starting point.

## The principal remaining errors are malignant typing

| Two-step result | Arrangement 812 | Arrangement 813 |
| --- | ---: | ---: |
| Final five-category accuracy | 68.33% | 59.83% |
| Macro F1 / balanced accuracy | .4944 / 57.93% | .3754 / 47.13% |
| All-four-view accuracy | 70.00% | 57.50% |
| Binary routing accuracy | 92.67% | 90.17% |
| Typing accuracy on all true malignant cases | 51.00% | 39.00% |
| Benign / DC / LC / MC / PC recall | 85.67 / 98.67 / 21.33 / 64.00 / 20.00% | 81.00 / 81.33 / 20.00 / 41.33 / 12.00% |

The conditional typing denominator includes **all true malignant cases**, not merely successfully routed cases. It is obtained from the pooled joint distribution's malignant conditional probabilities, as registered. The five-category confusion-matrix diagonals independently reproduce the reported 68.33%/59.83% final accuracies.

With four benign and four malignant validation cases per run, a counterfactual **perfect router while holding current malignant argmax decisions fixed** could reach at most `0.5 + 0.5*typing_accuracy`: **75.50% and 69.50%**. This is an arithmetic upper bound for these unchanged subtype predictions, not a new model result. Conversely, perfect typing with the existing router cannot exceed its 92.67%/90.17% routing accuracy. Binary improvement alone cannot recover the five-category target from the present subtype errors.

## Implementation and numeric contract examined

All 18 implementation-file hashes in the accepted comparison match the reviewed source; relevant files/config also have no Git diff against producer `7daff48`. The aggregate covers 20 runs, 80 candidate pipelines, 9,600 candidate validation records and 13,800 selected fitting records. These repeated subset/trial records are dependent; they are not independent patient counts.

- **Targets and expert labels:** original-eight-to-five mapping is `(0,1,0,2,3,4,0,0)`. Pooled benign stops at the router. The malignant head uses labels 1–4 mapped to CE indices 0–3. Saved class orders and the scalar positive-malignant binary coefficient agree with those targets. No benign subtype head is present.
- **Stages and frozen base:** epochs 1–2 train heads only; epochs 3–20 enable only the eight final-four-layer Q/V A/B pairs and heads. The base remains in evaluation mode. Adapter dropout is explicitly enabled for adaptation training and disabled during scoring; the residual itself remains active at evaluation. All 20 aggregate receipts report equal initial/final frozen-base digests. The actual fitting-only smoke additionally verified A/B gradient phase transitions; final tensor changes alone are not treated as gradient proof.
- **Head normalization:** fixed-C1 per-view fitting-only mean/std and four one-hot view flags are preserved. Raw 3072-dimensional features use the same official deterministic transform at validation. Initial-probe feature error is zero, and the largest probability error across initial/reload receipts is approximately 1.28e-7. Thus no observed head-schema or standardization discontinuity explains the large accuracy gap. Fixed original normalization during adaptation is a recipe choice, not evidence that feature-statistic drift is impossible.
- **Loss mass:** every case supplies eight sampled fields, with two per magnification, and each epoch includes all 46 fitting cases. The expert receives true-malignant CE only. Its per-case weights for DC/LC/MC/PC are `.46/5.75/2.30/3.8333`; lobular and papillary cases are not accidentally unweighted. An independent full-epoch scalar calculation gives direct class coefficients 2.3 each; binary broad-class coefficients 2.875 each; expert type coefficients 1.4375 each after the registered half-dual/fixed-four denominator. Binary and expert terms therefore have equal total coefficient mass, with no unintended 35/46 dilution. Benign-only batches leave expert gradients absent. Four accumulated one-case microbatches and the final two-case update retain the fixed denominator four.
- **Pooling and decisions:** field joint-five probabilities are composed first, then averaged within a magnification and equally over available magnifications. The case router uses summed malignant mass at threshold .5, followed by malignant-only argmax. The CPU scorer repeats this rule from archived adapted features; it does not substitute five-way argmax for the two-step decision or independently average router/expert marginals.
- **Selection:** candidates are epochs 2/6/12/20, selected by complete-pipeline mean-15-subset five-category macro F1; chronological ordering implements earliest ties. Both fitted and excluded roles retain their fixed assignments. No outer outcomes select checkpoints.

The center crop retains approximately two thirds of each 700×460 image's area. That could omit peripheral diagnostic tissue, but it is the official pinned preprocessing and is shared by the frozen and LoRA paths. It does not by itself explain LoRA's gap relative to its selected frozen control. A crop/field-evidence hypothesis remains distinct from a verified parsing error.

## Interpretation and practical consequence

The fitting/validation gap, 12 head-only selections and persistently low rare-type recall support **poor cross-case generalization of this recipe**, not insufficient ability to fit the assigned cases. Only two lobular and three papillary fitting provider keys are available per run; many correlated fields and augmented copies do not increase that independent support. The provider identity relationships remain unresolved, the 31 distinct validation keys have been reused, and all 82 keys entered development somewhere. These limits affect both reliability and the interpretation of comparisons.

This evidence does **not** isolate one sole cause. It cannot distinguish inadequate rare-type coverage, nuisance/stain dependence, field-label ambiguity, mean pooling, fixed-head regularization, insufficient useful adapter movement or their interaction. The aggregate does not measure adapted-versus-frozen feature movement directly: a zero reload parity error means reconstruction is consistent, not that no adaptation occurred. The earlier stronger kernel/frozen readouts show that readout choice already matters substantially; a larger trainable model is not automatically the solution.

Retain the strongest accepted two-step model and publish the complete negative LoRA comparison. Do not characterize this result as a failed execution, relabel binary accuracy as final five-category accuracy, or claim that another longer GPU run is likely to yield 95%. Any additional work should address a diagnosed, prospectively registered question with matched controls; this review supplies no new 95% result or validated route to one.

**Review limit:** this review independently recomputes scalar mass, candidate/initialization summaries and confusion arithmetic from accepted artifacts and inspects the bound source. It does not independently re-execute original images through adapted encoders. The accepted collector's saved-feature replay and producer reconstruction probes retain their stated scope. Private numerical worksheets preserve the exact source hashes, C=1 bridge and arithmetic calculations.
