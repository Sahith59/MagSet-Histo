# BreakHis improvement: data, pooling and generalization diagnosis

Reviewed November 15, 2026 against the accepted five-category Phikon producer `0471cd4ab8fb1da71eba8870f2ea0fa59ce0e8f3` and EfficientNet producer `b5f0df18a47c1607be492bdaab7ab6cd2171226b`. Final labels remain pooled benign, ductal, lobular, mucinous and papillary carcinoma. Binary routing comes first; predicted benign stops and predicted malignant receives one of four malignant labels.

**Recommendation:** test one bounded field-supervised, strongly regularized classifier on the existing frozen Phikon descriptors, with class/case/magnification-balanced fitting and equal probability pooling within each magnification and then across available magnifications. Include a matched direct-five arm. This is a different training objective and readout, not an established accuracy improvement. It directly tests whether training on bag means hides useful field variation while avoiding another encoder adaptation with only 46 fitting cases. The strongest measured issue is generalization of rare malignant types, especially papillary; pooling is a plausible contributor whose causal effect remains unproved. There is no evidence supporting a 94–95% forecast.

## Evidence scope

The public [Phikon comparison](../evidence/BREAKHIS_FIVE_CLASS_RESULTS.json) and [EfficientNet comparison](../evidence/BREAKHIS_CASEBAG_RESULTS.json) supply all registered trial, subset, fitting and validation outcomes. This audit additionally read the 40 selected Phikon fitting/validation prediction archives, their run metadata and saved head arrays, and the four original frozen feature archives. It checked their recorded hashes, roles, record counts and reconstructed aggregate confusion matrices: 48,732 checks passed. All four feature NPZ hashes match the accepted Phikon comparison. Original images were not decoded again.

New calculations use accepted fitting/validation outputs and fitting-only descriptor geometry. No classifier was fitted, normalization refitted, encoder run, image scored, threshold searched, split changed, excluded outcome inspected, model selected, remote state changed or scheduler job submitted. Applying existing saved affine heads to cached **fitting** field vectors below is diagnostic inference, not a new trained candidate or new validation result. Private calculations and identifiers remain under `.local/improvement-data/`; the aggregate diagnosis JSON SHA256 is `4e97643f462591403d000f7c13bdb466e2a3716d6ed08e9e50a7b74c551e9479`.

## Measured failure concentration

All scores below are five-category provider-case results averaged over five overlapping trials and all 15 nonempty magnification subsets. Trial/subset repetitions are dependent. Fitting and validation scores are separate.

| Two-step model | Arrangement | Fitting accuracy | Validation accuracy | Fitting LC / MC / PC recall | Validation LC / MC / PC recall |
| --- | --- | ---: | ---: | --- | --- |
| Frozen Phikon | 812 | 96.29% | 65.83% | 100 / 94.93 / 95.11% | 56.00 / 57.33 / 9.33% |
| Frozen Phikon | 813 | 96.35% | 68.83% | 100 / 94.93 / 93.78% | 54.67 / 56.00 / 17.33% |
| EfficientNet | 812 | 87.36% | 46.50% | 61.33 / 69.33 / 64.00% | 1.33 / 1.33 / 2.67% |
| EfficientNet | 813 | 85.13% | 53.33% | 41.33 / 58.13 / 69.33% | 1.33 / 12.00 / 28.00% |

LC, MC and PC together account for 64.88% / 68.98% of Phikon two-step final errors while representing 37.5% of validation events. They account for 68.85% / 69.29% of EfficientNet two-step errors. This is a concentration of failures on rare malignant types, not merely a weak global benign/malignant separator. The CNN also fails to fit rare types well: its 85–87% raw fitting accuracy is dominated by the 36 benign/ductal fitting cases out of 46. Its rare-type weakness cannot be described solely as overfitting after successful rare-type learning.

For Phikon, the pooled 600 dependent validation events per arrangement contain 132 / 117 malignant wrong-subtype errors and 73 / 70 binary routing errors. A hypothetical perfect specialist with the current fixed router would still be capped at 87.83% / 88.33% final accuracy. Reaching 95% would require better routing and better malignant typing. This arithmetic ceiling is descriptive; it is not a fitted oracle model or a forecast.

The matched direct-five Phikon control validates at 66.67% / 64.67% accuracy, with papillary recall 10.67% / 17.33%. It has the same weakness without a binary gate. Therefore the hierarchy alone is not the identified cause of the papillary collapse.

## Trials and magnification subsets

| Two-step model | Arrangement | Validation accuracy in trials 1–5 |
| --- | --- | --- |
| Phikon | 812 | 66.67, 50.83, 59.17, 71.67, 80.83% |
| Phikon | 813 | 69.17, 75.00, 72.50, 73.33, 54.17% |
| EfficientNet | 812 | 50.83, 37.50, 42.50, 51.67, 50.00% |
| EfficientNet | 813 | 50.00, 70.00, 60.00, 50.83, 35.83% |

The CNN is lower than Phikon on all ten matched two-step trial comparisons. The underperformance is not confined to one unfavorable trial. These are overlapping trials, not independent folds.

With Phikon, single-view 40× / 100× / 200× / 400× accuracies are 55.0 / 70.0 / 62.5 / 60.0% in arrangement 812 and 65.0 / 65.0 / 67.5 / 70.0% in 813. All-four accuracy is 67.5% / 72.5%. All 15 subsets remain required; these descriptive singleton differences do not authorize choosing a favorable magnification.

Papillary recall is 0% / 20% with all four views versus 20% / 40% at 100× alone. Averaging additional views can therefore remove a correct rare-type decision in these saved outputs. That observation supports inspecting pooling, but does not prove that feature averaging caused the generalization gap. Four of the five papillary validation trial appearances in 812 are wrong under **every** subset; three of five in 813 are wrong under every subset. The same lobular trial appearances fail across every subset in both arrangements. Many errors persist after removing any single zoom, which argues against treating missing magnifications or one bad zoom as a sufficient explanation.

EfficientNet all-four rare-type recall is LC/MC/PC = 0/0/0% in 812 and 0/20/20% in 813. Single 40× has 20/0/20% and 20/40/40%, respectively. Its failure also persists with all four views available.

## Data support and field geometry

The unchanged original-eight-derived roles contain 46 fitting, eight validation and 28 excluded provider cases per run. Five-category fitting support is `[11,25,2,5,3]`; validation support is `[4,1,1,1,1]`. The specialist has 35 truly malignant fitting cases, only two lobular and three papillary. Fields, magnifications and 15 masks increase observations of those same cases; they do not create additional independent subjects. Class balancing is already present in both accepted training pipelines, so adding an inverse-class weight is not by itself a new remedy. [Existing independent data audit](FIVE_CLASS_DATA_DIAGNOSIS.md).

Across fitting cases, individual case/magnification bags range from 1–66 fields. Class-specific ranges are benign 11–66, ductal 1–62, lobular 20–55, mucinous 9–50 and papillary 13–38. An unweighted field classifier would consequently give some cases and magnifications much greater influence. Any field-supervised recipe needs explicit case and magnification weighting as well as class weighting.

A fitting-only descriptor calculation uses each run's saved per-view normalizer. Within-case trace is the mean squared normalized distance of fields to their own case/view mean; between-case trace is the mean squared distance of those case/view means to their fitting-class centroid. The median within/between ratios across ten runs and four views are benign 1.19, ductal 1.49, lobular 4.70, mucinous 1.40 and papillary 1.51. Fields contain substantial within-case variation that a mean omits. The lobular ratio also has an unstable denominator from only two fitting cases. These are unlabeled geometric dispersions within the fitting class, not measures of diagnostic information, discriminative feature rank or proof that covariance will generalize.

The saved per-view normalizer includes within-case field variation, and the inference representation averages standardized case means across available views. The official frozen processor preserves its recorded central crop. Existing regional/encoder adaptation comparisons already failed their improvement gates; the current evidence does not isolate crop, stain, field heterogeneity or encoder pretraining as the sole cause. [Earlier model diagnosis](FIVE_CLASS_MODEL_DIAGNOSIS.md).

## Why the next candidate is specific and bounded

For an affine head, mean-feature logits equal mean field logits. Moving the same affine operation before the mean while retaining logit pooling cannot add information. The candidate must change supervision, probability pooling or both. Probability averaging is nonlinear relative to feature averaging, so it can behave differently; it can also dilute focal evidence and need not improve calibration or rare-class recall.

As a fitting-only arithmetic check, applying the existing selected binary heads to individual fields and averaging their probabilities instead of their logits changes probabilities by a mean absolute 0.0238 across 1,840 repeated case/view appearances, but changes zero binary decisions. For malignant heads on truly malignant fitting cases, the mean maximum absolute probability difference is 0.0592 across 1,400 appearances; only three decisions change. These are repeated fitting appearances of already selected models. A readout swap alone has little fitting decision effect here. They provide no candidate validation accuracy and do not test a head trained with field supervision.

**One next intervention:** retain the accepted frozen Phikon cache and fitting-only normalization, and train regularized affine field heads on 1,024 normalized field channels plus four one-hot magnification indicators. Each field inherits its case label, so this is weak supervision. Give each head equal total class mass, each fitting case equal mass within its class, each of the four magnifications one quarter of its case mass, and each field `1 / fields_in_that_case_view` of that view mass. A convenient normalized weight is `1 / (K * N_class * 4 * N_fields_in_case_view)`, with each head's own class count K and fitting support N_class. Lock the global weight scale too: multiplying these normalized weights by the head's independent fitting-case count (46 for direct/router, 35 for the specialist) preserves that total objective mass rather than silently changing the meaning of C. Apply binary weights to all 46 fitting cases, malignant-four weights only to the 35 truly malignant cases, and direct-five weights to all 46. Do not multiply by obsolete eight-class weights. Retain all fields and avoid pseudo-replication in reporting.

For a declared available subset, average field binary probabilities within each available view and then equally across available views to obtain router probability r. Do the same separately for the four-output malignant head to obtain s. Route malignant at `r >= 0.5`, benign otherwise; malignant predicts `1 + argmax(s)`. Save composed probabilities `[1-r, r*s]` and the explicit hard decision separately. Probability-vector argmax is not interchangeable with this route. No correspondence between fields at different zooms is assumed. The same field representation, weighting rules and probability pooling feed a direct-five control.

Register a fixed small regularization budget and equal complete-pipeline selection opportunities before fitting, retaining both arrangements, five trials, all 15 subsets and all candidates. Select only by the declared end-to-end validation metric. Keep original mean-feature Phikon results as a historical matched reference; the field change tests a complete objective/readout package and cannot by itself attribute any gain specifically to probability pooling. If a mechanistic attribution becomes necessary later, it needs a separately declared ablation.

A nonlinear case head or covariance head is a weaker first choice from this evidence. The current 1,028-dimensional affine case head already fits about 96% of cases with almost perfect rare-type fitting recall. More case-head capacity does not directly address its validation collapse. A diagonal variance descriptor adds another 1,024 dimensions; full covariance has over half a million unique entries per view. Neither acquires new independent rare cases. Shrinkage and dimension reduction could make such candidates measurable later, but a broad capacity search is not justified as the immediate response.

## Acceptance and limits

Before any new fitting, require immutable source/config/cache/split bindings, exact original role reconstruction before label pooling, field-weight mass checks, class order, malignant-only specialist supervision, convergence, nonempty-subset coverage, numerical stability, permutation/missing-view invariance, explicit hard routing and model save/reload checks. Report fitting and validation separately, with mean-subset and all-four accuracy, macro F1, balanced accuracy, five recalls, binary routing and conditional malignant typing. Include failed outcomes. Review selected outputs independently before changing the best-model claim.

Only 31 distinct provider cases appear in the union of reused validation arrangements. Clinical patient identity remains unresolved, and all 82 provider keys have entered development somewhere. The 28 excluded cases per run are not a new global holdout. Any improvement from the proposed candidate remains exploratory provider-case validation. Frozen Phikon two-step 65.83% / 68.83% is still the best verified two-step option; 94–95% remains unachieved. TCGA remains deferred.
