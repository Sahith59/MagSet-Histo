# Independent review of actual UNI2-h case-SVC results

**The saved results reproduce correctly. The registered RBF progression screen fails, and neither final accuracy objective is achieved on the retrospective follow-up.** Both arms select C=10 by the first maximum validation mean15 five-class macro F1. The higher RBF validation binary accuracy does not establish >97% follow-up binary accuracy.

This review covers the actual family from frozen scientific source `13fe4ae1dc7f377a9d36f78f0e005eb2f6da2b0a`. The accepted checker JSON SHA256 is `2ee058a20cab04ba16efece2e111c7fd2d56ea9eb55edf15133e3dddf682a355`. Its hash-bound private review has no errors. A separate reviewer script, without producer or collector math imports, independently reconstructed every saved margin-to-route calculation, normalized reporting score, case/mask record, per-subset confusion matrix, macro F1, recall, ROC AUC and strict/conditional malignant result.

## All registered validation choices

Each row covers the same 17 validation cases and all 15 subsets. Accuracy and binary accuracy are means over those subsets; macro F1 is the mean of 15 separately calculated fixed-five-label F1 values.

| Arm | C | Five-class accuracy | Macro F1 | Binary accuracy |
| --- | ---: | ---: | ---: | ---: |
| Linear | 0.1 | 41.1765% | .257142857 | 100.0000% |
| Linear | 1 | 76.4706% | .464539513 | 95.2941% |
| **Linear** | **10** | **76.4706%** | **.476316102** | **96.8627%** |
| Linear | 100 | 76.4706% | .471437908 | 95.6863% |
| RBF | 0.1 | 29.4118% | .090909091 | 29.4118% |
| RBF | 1 | 76.4706% | .474379085 | 100.0000% |
| **RBF** | **10** | **76.4706%** | **.486148742** | **99.2157%** |
| RBF | 100 | 76.4706% | .484612795 | 99.2157% |

RBF minus linear validation macro F1 is **+.009832640127**, below the registered +.02 minimum. Accuracy difference is zero. The screen therefore fails. Choosing a candidate for its binary score, a favorable subset or its later follow-up outcome would change the registered selection rule.

## Selected retrospective follow-up

The exact original 16 cases remain the follow-up cohort. These 240 case/subset events per arm are dependent observations of those same cases.

| Selected arm | Five-class correct / events | Mean15 macro F1 | Binary correct / events | All-four accuracy |
| --- | ---: | ---: | ---: | ---: |
| Linear C=10 | **179/240 = 74.5833%** | **.513007130** | **221/240 = 92.0833%** | 12/16 = 75.0000% |
| RBF C=10 | **166/240 = 69.1667%** | **.380728586** | **221/240 = 92.0833%** | 11/16 = 68.7500% |

For linear, 61 final errors comprise 14 benign-to-malignant routes, five malignant-to-benign routes and 42 subtype mistakes after a correct malignant route. For RBF, 74 errors comprise 18, one and 55 respectively. Equal binary accuracies thus conceal different errors. Neither arm correctly types the single lobular case or either mucinous case on any subset. Correct papillary calls are 13/15 for linear and 4/15 for RBF. Ductal calls are 105/105 for both. These counts describe this cohort; they do not identify a biological or computational cause.

The primary mean15 F1 must not be replaced by F1 calculated after stacking all masks into one confusion matrix: those are different summaries. The stacked matrices were independently checked as an additional count audit, while the published primary scores above retain the registered mean15 definition.

## Provenance and scope of proof

The reviewer rehashed **all 61 producer-registered artifacts**, including all eight mirrored joblib files, without loading any estimator. All implementation receipts match the corresponding Git files at the frozen source. The review also verified the exact audit hash and disjoint conservative groups, recomputed TRAIN-only saved-case normalizer arrays, checked all eight validation reload receipts, verified both first-maximum selections and their common hash seal, and verified selected model/state/gate bindings for both follow-up archives. No local artifact omission remains. The outer inventory limitation identified in the pre-fit implementation review was closed for these actual artifacts by this additional review.

The immutable producer's call order places both completed validation selections and the common gate before selected test model calls. Hash receipts and source establish that declared execution path; they are not a separate hardware trace of every operation. The review added **zero fits, zero estimator or encoder queries and zero jobs**. It did not independently reproduce libsvm fitting or fresh raw UNI2-h extraction. Dense support-vector margin reconstruction is supplied by the separately accepted checker; this additional review starts from saved head margins and independently verifies their downstream arithmetic.

The private reviewer receipt SHA256 is `a8a9c8d90a3e1a42ac1d7c7f64093922266d7510608abf3ad62be6c48c3b4ac3`. These remain reused development results, not a new independent cohort or clinical confirmation. The bounded family provides a negative RBF progression result and a fully reported linear comparison; it supplies no basis for an automatic expansion of the search or a claim that >90% final-five / >97% binary has been attained.
