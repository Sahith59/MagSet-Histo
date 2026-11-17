# UNI2-h case-level SVC: complete results

November 17, 2026. **The bounded comparison is complete and independently accepted, but does not meet either requested follow-up accuracy objective.** RBF fails its registered validation progression screen. No additional C, gamma, kernel, threshold, seed or split was tried after these outcomes. [Registered plan](BREAKHIS_UNI_CASE_SVM_PLAN.md) · [complete accepted evidence](evidence/BREAKHIS_UNI_CASE_SVM_RESULTS.json).

The input is a set of BreakHis pictures from one provider case. A frozen UNI2-h reader supplies 1,536-number descriptions; original fields are averaged within each zoom. Training-only per-view normalization, equal averaging of the available zooms, L2 normalization and four availability flags produce 1,540 inputs. A binary SVC decides benign/malignant. Malignant proceeds to a four-type SVC; benign stops. Sigmoid/softmax reporting values are uncalibrated scores, not fitted probabilities.

Both kernels use the same 49 training / 17 validation / 16 already-viewed follow-up cases, in 42 / 15 / 13 conservative groups, and all 15 nonempty zoom subsets. This is the actual case/mask prediction unit. Related images stay together. Clinical identity remains unresolved; this reused-development follow-up supplies no fresh confirmation.

## Selected outcomes

| Arm | Selected C | Validation final-five accuracy | Validation macro F1 | Validation binary accuracy | Reused follow-up final-five accuracy | Follow-up macro F1 | Follow-up binary accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 10 | 76.47% | 0.476316 | 96.86% | 74.58% | 0.513007 | 92.08% |
| rbf | 10 | 76.47% | 0.486149 | 99.22% | 69.17% | 0.380729 | 92.08% |

The validation F1 difference is **+0.009833**, below the fixed +.02 requirement; accuracy is tied. The screen **fails**. The 99.22% RBF binary score belongs to validation; the unchanged follow-up gives 92.08%. It must not be substituted for the final-five score or called an achieved >97% follow-up result.

Each follow-up has 16 cases × 15 subsets = 240 dependent events: linear gets **179/240** final labels and **221/240** binary decisions correct; RBF gets **166/240** final labels and **221/240** binary decisions correct. No method is promoted from these retrospective outcomes. The contextual fixed-router specialist’s 199/240 (82.92%) remains a mixed observation that failed its own validation screen.

## Every validation candidate

| Kernel | Shared C | Final-five accuracy | Macro F1 | Binary accuracy |
| --- | ---: | ---: | ---: | ---: |
| linear | 0.1 | 41.18% | 0.257143 | 100.00% |
| linear | 1 | 76.47% | 0.464540 | 95.29% |
| linear | 10 | 76.47% | 0.476316 | 96.86% |
| linear | 100 | 76.47% | 0.471438 | 95.69% |
| rbf | 0.1 | 29.41% | 0.090909 | 29.41% |
| rbf | 1 | 76.47% | 0.474379 | 100.00% |
| rbf | 10 | 76.47% | 0.486149 | 99.22% |
| rbf | 100 | 76.47% | 0.484613 | 99.22% |

Each arm received four opportunities. First-maximum mean15 final-five macro F1 selected C=10 for both before either selected follow-up query. All eight validation archives and both selected follow-ups are retained.

## Class performance and secondary endpoints

| Arm / role | Benign recall | Ductal recall | Lobular recall | Mucinous recall | Papillary recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear / validation | 100.00% | 100.00% | 0.00% | 50.00% | 0.00% |
| linear / reused follow-up | 81.33% | 100.00% | 0.00% | 0.00% | 86.67% |
| rbf / validation | 100.00% | 100.00% | 0.00% | 50.00% | 0.00% |
| rbf / reused follow-up | 76.00% | 100.00% | 0.00% | 0.00% | 26.67% |

Recall means the fraction of each true category answered correctly, averaged over the same 15 subsets. Both selected models still miss all lobular validation cases and all papillary validation cases. There are only two lobular and four papillary training cases; duplicated masks do not add independent cases. This is a measured subtype-transfer limitation, not proof of a morphological or labelling defect.

| Arm | Follow-up all-four accuracy / macro F1 | Worst-subset accuracy / macro F1 |
| --- | ---: | ---: |
| linear | 75.00% / 0.542484 | 68.75% / 0.346524 |
| rbf | 68.75% / 0.333333 | 62.50% / 0.297368 |

## Execution proof

Scientific source: `13fe4ae1dc7f377a9d36f78f0e005eb2f6da2b0a`; configuration SHA256 `66fc988ce6452a30552eb3191885238b88b4df901d28a24a3007f2e52b2d4bbc`. CPU producer **4987465** completed with exit 0 in **21 seconds**; independent collector **4987466** completed with exit 0 in **6 seconds**. Eight pipelines fitted 16 SVC heads (56 internal binary optimization problems). No image decoding, encoder update or encoder forward occurred.

The independent collector rebuilt case descriptors and support-vector kernel margins, including six pairwise malignant decisions and their OVR transformation. It checked TRAIN-only support ownership/statistics/weights, all validation outputs, both first-max selections, save/reload checks, the common seal, both follow-ups and score arithmetic. A separate read-only cluster pass rehashed all **61** files in the outer producer inventory without an error. [Implementation review](reviews/UNI_CASE_SVM_IMPLEMENTATION_REVIEW.md) · [independent actual-result audit](reviews/UNI_CASE_SVM_ACTUAL_RESULTS_REVIEW.md).

Accepted result SHA256: `2ee058a20cab04ba16efece2e111c7fd2d56ea9eb55edf15133e3dddf682a355`.

The comparison supplies a completed negative result. More nonlinear capacity did not solve the case-generalization weakness in this family. Finish the capstone evidence package and use the fixed training/validation inspection index for appropriate expert review of rare-subtype labels and tissue coverage. Further performance claims need suitable additional independent cases; do not keep optimizing the already-viewed follow-up. TCGA and novel-paper expansion remain deferred.
