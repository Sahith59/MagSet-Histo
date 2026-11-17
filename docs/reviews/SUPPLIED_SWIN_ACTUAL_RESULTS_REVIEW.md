# Supplied Swin execution: actual-result review

November 17, 2026. **Accepted as a separately reported supplied-notebook result, with the limitations below.** This review independently checked the [accepted aggregate evidence](../evidence/SUPPLIED_SWIN_RESULTS.json), downloaded saved probabilities and execution receipts, and the [collector](../../scripts/collect_supplied_swin.py). It performed no fitting, image inference or TensorFlow checkpoint reload. The earlier [protocol review](SUPPLIED_SWIN_REVIEW.md) remains applicable.

The measured model source is `a50aeee39a78b3210a5a8c33905f6566a0563875`. The aggregate evidence SHA-256 is `c43c0a2b8875f62d6df61ff3cc51803b9a02157b81157e7f5da554d77048b787`; saved prediction SHA-256 is `4325b43b173266b2670f665d72bd2f089ea133991337d4197ef6d7ffef54fb50`. Current adapter, collector and configuration hashes match that evidence, and the reviewed model-related files match the recorded source commit.

## Independently observed results

Expanding the reported confusion matrices into label pairs and using separate NumPy/scikit-learn calculations passed 84 scalar comparisons, together with support, correct-count, binary-collapse and malignant-denominator checks. The four magnification confusion matrices sum exactly to the complete test confusion matrix.

| Quantity | Independently checked result |
| --- | ---: |
| Final five-category accuracy | 1,362 / 2,116 = **64.3667%** |
| Five-category macro F1, explicit labels 0–4 | **0.4416019710** |
| Unweighted five-category Cohen kappa | 0.4742342612 |
| Binary accuracy | 1,976 / 2,116 = **93.3837%** |
| Malignant recall | 1,468 / 1,527 = 96.1362% |
| Benign recall | 508 / 589 = 86.2479% |
| End-to-end malignant-four correctness | 854 / 1,527 = **55.9267%** |
| Subtype accuracy conditional on correct malignant routing | 854 / 1,468 = **58.1744%** |
| Malignant-four macro recall | 33.5880% |
| Binary probability AUC | 0.9759818457 |

The raw NPZ contains exactly 2,116 saved test records and the expected six arrays. Its integer indices are unique and ascending; binary and malignant-head arrays have shapes 2,116×2 and 2,116×4, respectively, with finite probabilities in [0,1] and normalized rows. Saved malignancy probabilities exactly equal the second binary-head column. Applying the original `p(malignant) >= 0.5` route and malignant-head argmax reproduces every saved final prediction. Independent raw-array calculations reproduce the complete confusion matrix, macro F1, nominal kappa and AUC. This local probability review did not independently reconstruct original image membership from the private full audit; that binding is covered by the accepted cluster collector and earlier manifest review.

| True label | Correct / support | Recall |
| --- | ---: | ---: |
| Benign | 508 / 589 | 86.2479% |
| Ductal | 737 / 1,017 | 72.4680% |
| Lobular | 82 / 197 | 41.6244% |
| Mucinous | 34 / 174 | 19.5402% |
| Papillary | 1 / 139 | **0.7194%** |

The measurements show strong binary separation alongside weak malignant typing. Among the 1,527 true malignant images, 59 were routed to benign and 614 were routed malignant but assigned the wrong subtype. This arithmetic identifies malignant typing as the larger observed error source; it does not identify its biological or causal explanation.

| Magnification | Images | Final-five correct | Final-five accuracy | Binary correct | Binary accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40× | 545 | 321 | 58.8991% | 489 | 89.7248% |
| 100× | 561 | 386 | 68.8057% | 542 | 96.6132% |
| 200× | 536 | 361 | 67.3507% | 508 | 94.7761% |
| 400× | 474 | 294 | 62.0253% | 437 | 92.1941% |

Secondary majority voting gives 18/24 = 75% final-five accuracy, 24/24 binary accuracy and macro F1 0.5698550725. Its support is 7/11/2/2/2. These are **24 bare-ID/true-target groups from 23 bare IDs**, because the notebook groups by both ID and true label. They are not 24 verified independent patients. This review independently checked voting aggregate arithmetic; the accepted collector reconstructs the group membership and original `value_counts().idxmax()` tie rule.

## Execution and source checks

Twelve downloaded execution artifacts independently match their accepted hashes. The 17-epoch saved history has its first maximum at epoch 7: validation binary accuracy 95.5007256894%, malignant-subtype correctness 63.8%, and their mean 79.6503628447%. The remaining ten epochs agree with the fixed early-stopping patience. No test score enters that selection formula.

All 15 original/adapted/executed cell sources and saved source digests match the declared reconstruction at the recorded archive location. The exact original setup source remains skipped; all 13 experimental code cells executed without saved error outputs. Source inspection preserves the original split, losses, input normalization, augmentation, model heads, validation callbacks and hard routing, with the declared native-TensorFlow loading and integer window-reversal compatibility repairs.

The accepted producer receipt reports 32-example checkpoint replay errors of 6.8545341492e-7 for the binary head and 3.8743019104e-7 for the subtype head, below 1e-5, bound to the recorded checkpoint hash. **This reviewer checked that receipt and its hash; it did not reload the model or independently hash the large checkpoint file locally.** The loading receipt permits only the exact 12 deterministic, nontrainable relative-position index buffers, with their exact matrix hashes; no learned missing weight is exempt. The saved 16-case window-reversal proof reports zero reconstruction, eager/graph and gradient errors. Its `original_graph_reproduced_failure` is **false**, so this proof does not itself demonstrate reproduction of the earlier optimizer failure.

No remaining numerical or source-recipe discrepancy was found. The checker validates saved evidence and provenance; it cannot establish that every execution event occurred solely from hashes and logs, and it relies on the immutable source snapshot and accepted input audit. Its source recipe uses the producer's pure patch-construction functions, while split, routing and metric replay are separately implemented. The additional calculations above independently cross-check the reported arithmetic and raw probabilities.

## Interpretation limits

The supplied split retains 5,104 training, 689 validation and 2,116 test images. Bare IDs do not cross roles, but conservative stems overlap in two training/test, one validation/test and one training/validation intersections. Clinical identity remains unresolved. Validation has **zero lobular images**, so selection cannot assess that subtype. Earlier project development involved all provider keys; this is not a globally fresh cohort.

This is a single-image classifier using all four magnifications, not the main provider-case, 15-subset capstone benchmark. The 93.38% binary number must not be presented as final-five accuracy. The run neither reaches 94–95% final-five accuracy nor establishes a consistent replacement for the main system. Hardcoded paper comparisons are not a matched reproduction or a superiority test. The results justify reporting this bounded experiment and its typing weakness, without a score-driven resplit, new threshold or additional model search.
