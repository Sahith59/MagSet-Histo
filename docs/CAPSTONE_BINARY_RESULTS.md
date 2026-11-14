# Benign/malignant secondary result

November14,2026. **The retained frozen model gives87.0%/80.7% mean binary case accuracy on the two existing validation arrangements.** Its exact eight-subtype accuracy remains45.7%/38.3%. These are different label questions, not an improvement of the eight-class score.

This fixed retrospective analysis collapses existing eight-class probabilities into the dataset’s benign/malignant groups. It performs no fitting, model execution, calibration or threshold search. The original paper’s86.22% is binary image accuracy under another protocol; we do not claim to beat it with these case results.

## All three accepted pipelines

Each entry is a five-trial mean of scores averaged equally over all15 input-view subsets. AUROC measures ranking and is not accuracy.

| Pipeline | Arrangement | Binary accuracy | Binary macro F1 | Malignant sensitivity | Benign specificity | Mean AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 812 | 87.00% | 0.8666 | 98.67% | 75.33% | 0.9617 |
| frozen | 813 | 80.67% | 0.7877 | 91.33% | 70.00% | 0.9408 |
| adapted | 812 | 86.50% | 0.8617 | 98.67% | 74.33% | 0.9658 |
| adapted | 813 | 82.00% | 0.8047 | 93.33% | 70.67% | 0.9558 |
| supervised | 812 | 84.67% | 0.8450 | 94.33% | 75.00% | 0.8875 |
| supervised | 813 | 82.17% | 0.8131 | 87.67% | 76.67% | 0.9217 |

The broad group is often correct even when the exact subtype is confused with another subtype in that group. This describes the saved outputs; it does not establish a biological cause or reliable clinical performance. The frozen reference remains the retained eight-class model; no binary winner is selected after inspecting this table.

## Full-view result

| Pipeline | Arrangement | All-view binary accuracy | All-view macro F1 | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen | 812 | 87.5% | 0.8730 | 100.0% | 75.0% |
| frozen | 813 | 77.5% | 0.7504 | 90.0% | 65.0% |
| adapted | 812 | 87.5% | 0.8730 | 100.0% | 75.0% |
| adapted | 813 | 82.5% | 0.8086 | 95.0% | 70.0% |
| supervised | 812 | 85.0% | 0.8484 | 95.0% | 75.0% |
| supervised | 813 | 85.0% | 0.8425 | 90.0% | 80.0% |

## All trials and both arrangements

| Pipeline | Arrangement | Trial | Mean-subset binary accuracy | Macro F1 | AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen | 812 | 1 | 84.17% | 0.8402 | 0.9292 |
| frozen | 812 | 2 | 87.50% | 0.8730 | 0.9500 |
| frozen | 812 | 3 | 81.67% | 0.8078 | 0.9375 |
| frozen | 812 | 4 | 91.67% | 0.9153 | 0.9958 |
| frozen | 812 | 5 | 90.00% | 0.8967 | 0.9958 |
| frozen | 813 | 1 | 78.33% | 0.7706 | 0.9792 |
| frozen | 813 | 2 | 87.50% | 0.8730 | 0.9958 |
| frozen | 813 | 3 | 95.83% | 0.9577 | 0.9917 |
| frozen | 813 | 4 | 85.83% | 0.8544 | 1.0000 |
| frozen | 813 | 5 | 55.83% | 0.4831 | 0.7375 |
| adapted | 812 | 1 | 84.17% | 0.8402 | 0.9375 |
| adapted | 812 | 2 | 87.50% | 0.8730 | 0.9583 |
| adapted | 812 | 3 | 82.50% | 0.8171 | 0.9375 |
| adapted | 812 | 4 | 90.83% | 0.9069 | 1.0000 |
| adapted | 812 | 5 | 87.50% | 0.8713 | 0.9958 |
| adapted | 813 | 1 | 77.50% | 0.7613 | 0.9792 |
| adapted | 813 | 2 | 87.50% | 0.8730 | 0.9958 |
| adapted | 813 | 3 | 96.67% | 0.9661 | 0.9917 |
| adapted | 813 | 4 | 85.83% | 0.8544 | 1.0000 |
| adapted | 813 | 5 | 62.50% | 0.5685 | 0.8125 |
| supervised | 812 | 1 | 74.17% | 0.7413 | 0.8500 |
| supervised | 812 | 2 | 87.50% | 0.8730 | 0.8292 |
| supervised | 812 | 3 | 87.50% | 0.8730 | 0.8833 |
| supervised | 812 | 4 | 86.67% | 0.8648 | 0.8750 |
| supervised | 812 | 5 | 87.50% | 0.8730 | 1.0000 |
| supervised | 813 | 1 | 79.17% | 0.7883 | 0.8750 |
| supervised | 813 | 2 | 79.17% | 0.7799 | 0.9417 |
| supervised | 813 | 3 | 98.33% | 0.9831 | 1.0000 |
| supervised | 813 | 4 | 82.50% | 0.8171 | 0.9250 |
| supervised | 813 | 5 | 71.67% | 0.6972 | 0.8667 |

## Rule, verification and limitations

Malignant score is the sum of ductal, lobular, mucinous and papillary class probabilities. At score≥0.5 the fixed prediction is malignant, including a tie; otherwise benign. The provider broad labels were checked against every subtype. This grouping describes BreakHis labels, not a universal clinical classification of all tumors with these names.

Each validation trial has four benign/four malignant cases. The always-malignant analytic reference is50% accuracy. Arrangements812/813 cover22/20 distinct provider cases; their union is31 (14 benign/17 malignant). There are3,600 repeated case/subset events across three pipelines, not3,600 patients. Sensitivity and specificity must accompany accuracy; the retained model still misclassifies a material fraction of benign validation cases.

All30 reports/prediction archives matched trusted roles/truths and accepted parent hashes. A separate offline scikit-learn1.6.1 audit recomputed all15 subsets, ROC-AUC/average precision, threshold metrics, confusion/support, Brier, full/worst/mean summaries and privacy:7,508 numerical checks, maximum difference2.22e-16. [Independent review](reviews/CAPSTONE_BINARY_REVIEW.md).

Analysis SHA256 `7502fe88b95869e964861d36004d7a46ebd7b342ad042ca25d47626612450ae1`. [Complete aggregate ledger](evidence/CAPSTONE_BINARY_RESULTS.json) retains every subset/trial, including negative outcomes. [Fixed retrospective specification](CAPSTONE_BINARY_SPEC.md).

The eight-class outcomes were already known before this secondary analysis. It reuses exploratory validation cases and models selected on eight-class F1; it is not a new test or newly trained binary classifier. Exact clinical-patient identity remains unverified and no globally fresh cohort remains. No eight-class primary gate is changed, no clinical claim or novelty is established, and no claim of beating the earlier paper follows.
