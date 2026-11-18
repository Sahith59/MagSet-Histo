# BreakHis capstone benchmark

## Numbers for the capstone

**The frozen single model reached 60.00% five-category accuracy and 79.17% benign/malignant accuracy on the retrospective 16-case benchmark.** All 240 decisions were reproduced. The validation result used to select this checkpoint was 80.39% and 99.22%. The 16-case result is the figure to use for the separate follow-up. A fresh-cohort final accuracy is still unknown.

| Evaluation of UNI2-h | What was kept out of fitting? | Correct five-category answers | Correct binary answers | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Validation during model development | 17 cases used to choose checkpoint 10 | 205/255 (80.39%) | 253/255 (99.22%) | 0.6337 |
| Group check within training | Each of 42 groups left out once, covering 49 cases | 469/735 (63.81%) | 657/735 (89.39%) | 0.5107 |
| Frozen retrospective follow-up | 16 different cases excluded from training and checkpoint selection | 144/240 (60.00%) | 190/240 (79.17%) | 0.4731 |

**What is a case?** Picture a folder containing the microscope photos that BreakHis assigns to one specimen identifier. We keep every photo in a folder together and keep folders with related identifiers in the same split. Clinical patient identity has not been independently verified. The 49 training folders contain 4,621 pictures, the 17 validation folders contain 1,584, and the 16 follow-up folders contain 1,704. The three roles contain 42, 15 and 13 conservative groups respectively.

The validation folders comprise five benign, seven ductal, two lobular, two mucinous and one papillary case. For each folder, we give the model different combinations of 40×, 100×, 200× and 400× pictures. There are 15 possible nonempty combinations. For example, the one papillary validation case produced 15 answers to the same diagnosis. The model named it correctly once. With all four zooms together, there is just one answer per folder. Under that condition the model correctly named 14/17 validation cases and 10/16 follow-up cases. It made 13/16 correct binary calls on the follow-up with all four zooms.

### Fair comparison on the same 16 cases

| Method | Five-category accuracy | Benign/malignant accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| UNI2-h, single five-category output | 144/240 (60.00%) | 190/240 (79.17%) | 0.4731 |
| CNN–Transformer | 136/240 (56.67%) | 207/240 (86.25%) | 0.3482 |
| Swin | 142/240 (59.17%) | 198/240 (82.50%) | 0.3049 |
| Midnight P, earlier reference | 162/240 (67.50%) | 227/240 (94.58%) | 0.3757 |

The single-output method had the highest observed macro F1 of these four, a measure that gives each diagnosis equal importance. Midnight P had the highest accuracies. The 95% group-bootstrap intervals for the new method span 40.78–83.33% for five categories and 55.00–100.00% for binary. These are descriptive intervals on 13 reused groups, with 15 related zoom answers per case. Every paired accuracy-difference interval includes zero. A ranking on these cases does not establish superiority on future cases.

### What changed from validation?

| Diagnosis | Validation correct | Follow-up correct |
| --- | ---: | ---: |
| Benign | 73/75 | 25/75 |
| Ductal | 84/105 | 89/105 |
| Lobular | 17/30 | 3/15 |
| Mucinous | 30/30 | 13/30 |
| Papillary | 1/15 | 14/15 |

All 50 binary mistakes on the follow-up were benign answers called malignant. Of the 96 five-category mistakes, the remaining 46 were malignant answers assigned the wrong subtype. Binary detection of malignant cases was 165/165, while correct benign recognition was 25/75. Only one papillary case and one lobular case were in the follow-up. The different case mix and the failures across entire folders make a single overall validation percentage unstable.

**Next experiment.** Start with the blinded review of the full training and validation fields and center crops. Include benign tissue alongside the rare cancers when checking coverage and subtype confusion. Any change to the one-output model or its decision rule should be selected using development groups and fixed before assessment on newly obtained independent cases. Keep this benchmark frozen for the capstone.

Slurm job 4989151 completed in 10 seconds from the registered source using cached frozen UNI2-h descriptors. The independent field check rebuilt all 64 case-by-zoom inputs from 1,704 hash-verified field descriptors and matched the original inputs exactly. It replayed the selected weights, reproduced every hard decision and checked the reported metrics. [Machine-readable benchmark](evidence/SINGLE_OUTPUT_CAPSTONE_BENCHMARK.json) · [Field-level replay](evidence/SINGLE_OUTPUT_CAPSTONE_BENCHMARK_REPLAY.json).

## Locked evaluation plan

Score the selected five-category UNI2-h checkpoint once on the same 16 provider cases used in the earlier matched benchmark. These cases were excluded from its 49-case training set, its 17-case checkpoint selection and its previous training-group check. Earlier project experiments have already used all 82 provider cases. This additional run measures retrospective performance within BreakHis. Independent confirmation requires a genuinely new cohort.

Before opening this model's follow-up predictions, pin the existing weights, training configuration, selected update 10 and previous model predictions. Keep the original normalization, five scores, binary threshold 0.5, subtype rule and all 15 subsets of the four zooms. The primary numbers are the proportion of 240 case-and-zoom answers correct for the five diagnoses and for benign/malignant, plus mean per-subset five-class macro F1. Report the all-four-zoom result as one decision per case. Strictly above 90% and 95% requires at least 217/240 and 229/240 respectively at the same checkpoint.

Verify the selected validation performance before scoring the 16 cases. Reconstruct all follow-up logits independently with NumPy and require agreement on every decision with the saved PyTorch classifier. Compare the same cases, truth and zoom conditions with the three original matched controls. Estimate descriptive 95% percentile intervals for the two accuracies and their paired differences by resampling the 13 conservative groups 10,000 times, preserving all fields, cases and zoom conditions in each sampled group. Describe the coverage and limitations as well as the measured differences. [Frozen evaluation settings](../configs/breakhis-single-output-retrospective-v1.json).
