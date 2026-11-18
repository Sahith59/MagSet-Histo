# BreakHis capstone benchmark

## Locked evaluation plan

Score the selected five-category UNI2-h checkpoint once on the same 16 provider cases used in the earlier matched benchmark. These cases were excluded from its 49-case training set, its 17-case checkpoint selection and its previous training-group check. Earlier project experiments have already used all 82 provider cases. This additional run measures retrospective performance within BreakHis. Independent confirmation requires a genuinely new cohort.

Before opening this model's follow-up predictions, pin the existing weights, training configuration, selected update 10 and previous model predictions. Keep the original normalization, five scores, binary threshold 0.5, subtype rule and all 15 subsets of the four zooms. The primary numbers are the proportion of 240 case-and-zoom answers correct for the five diagnoses and for benign/malignant, plus mean per-subset five-class macro F1. Report the all-four-zoom result as one decision per case. Strictly above 90% and 95% requires at least 217/240 and 229/240 respectively at the same checkpoint.

Verify the selected validation performance before scoring the 16 cases. Reconstruct all follow-up logits independently with NumPy and require agreement on every decision with the saved PyTorch classifier. Compare the same cases, truth and zoom conditions with the three original matched controls. Estimate descriptive 95% percentile intervals for the two accuracies and their paired differences by resampling the 13 conservative groups 10,000 times, preserving all fields, cases and zoom conditions in each sampled group. Describe the coverage and limitations as well as the measured differences. [Frozen evaluation settings](../configs/breakhis-single-output-retrospective-v1.json).
