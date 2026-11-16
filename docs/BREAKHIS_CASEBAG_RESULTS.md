# Completed EfficientNet five-category comparison

Verified November 15, 2026 at 21:42 EST. All 20 A40 training runs and both verification jobs completed successfully. The EfficientNet comparison scored below the earlier frozen-Phikon model on both validation arrangements. The earlier two-step model remains the best verified two-step option at 65.83% and 68.83% accuracy. The 94–95% target remains unachieved.

| EfficientNet model | Validation arrangement | Accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct five-category | 812 | 48.83% | 0.3018 | 38.33% | 50.00% |
| Two-step | 812 | 46.50% | 0.2346 | 30.40% | 47.50% |
| Direct five-category | 813 | 49.00% | 0.3016 | 40.00% | 50.00% |
| Two-step | 813 | 53.33% | 0.3108 | 40.73% | 55.00% |

Each score averages the five overlapping trials and all 15 nonempty magnification combinations. The all-four-view column uses the complete set of views. Each run used 46 fitting cases, eight validation cases and 28 excluded cases. Validation contains four benign cases and one case per malignant type. All 82 provider keys have entered development, and clinical patient identity remains unresolved. These scores describe reused provider-case development validation.

## What the result shows

The EfficientNet two-step model scored 87.36% and 85.13% on fitting cases, alongside 46.50% and 53.33% on validation cases. The gap shows weak generalization to validation cases. The model selected from this recipe is less accurate than the earlier frozen-Phikon two-step model.

Rare malignant types remain a major weakness. End-to-end lobular recall is 1.33% on both arrangements. Papillary recall is 2.67% and 28.00%, and mucinous recall is 1.33% and 12.00%. The fitting set has two lobular and three papillary cases per run. These findings support a focused error review. The cause of the performance gap still needs investigation.

Binary accuracy for the new two-step models is 75.17% and 82.17%. Conditional four-type accuracy on true malignant cases is 38.33% and 37.33%. Overall five-category accuracy counts both decisions. The prospectively recorded improvement screen failed on both arrangements, and every outcome is retained.

## Model and training

The chosen candidate was ImageNet-pretrained EfficientNetV2B0. Two sampled fields per magnification produce case features, followed by equal averaging of available views and four availability indicators. The direct model uses a five-output linear head. The two-step model uses a binary head and a four-type malignant head. The first five epochs train the heads. The next 45 epochs adapt the last two backbone stages and feature head while preserving BatchNorm running statistics. Both arms share case assignments, stateless input sampling and four checkpoint-selection opportunities at epochs 10, 20, 35 and 50.

Training and validation images come from BreakHis. All four magnifications are used. Data parsing, routing and score calculations passed their source and output reviews. The saved outputs and class-level gaps provide the evidence for the next diagnosis.

## Execution and verification

All 20 worker tasks in array 4979622 completed with exit code zero. Most took about seven to eight minutes, with three slower tasks taking about 19 or 29 minutes. All worker error logs were empty. Full collector 4979632 completed in 2 minutes 34 seconds, and independent numerical reviewer 4979664 completed in 46 seconds, both with exit code zero and accepted output receipts.

The checks cover 20 fits, 80 candidate checkpoints and 120 saved score archives containing 25,800 dependent fitting/validation score records. The independently written reviewer passed 3,852,210 assertions. Maximum metric replay error was 1.18e-14. The selected feature and probability replay errors were zero. The local suite passed 187 tests, and the code revisions passed GitHub CI.

Training source is `b5f0df18a47c1607be492bdaab7ab6cd2171226b`, collector source is `650c14cb05e75d9b4c1846e86150b372c5faf4a3`, and independent reviewer source is `343829aac54d83ff3c33b80af2bf9ce6f6d2196b`. Comparison SHA256 is `2a904fac355a21c8229cfef24f4a1579a27edf3986c51f828c538d11a1db493a`.

[Complete results](evidence/BREAKHIS_CASEBAG_RESULTS.json) · [Checkpoint and results verification](evidence/BREAKHIS_CASEBAG_VERIFICATION.json) · [Independent numerical verification](evidence/BREAKHIS_CASEBAG_INDEPENDENT_REVIEW.json) · [Earlier five-category baseline](BREAKHIS_FIVE_CLASS_RESULTS.md).

## Current project work

Use the earlier frozen-Phikon two-step result for the current capstone scorecard. Review the rare-subtype errors and fitting/validation gap using the saved predictions. Update the five-category presentation and report for Professor Dong with the full comparison and its limitations. A targeted follow-up experiment can then address the diagnosed weakness within the capstone deadline.
