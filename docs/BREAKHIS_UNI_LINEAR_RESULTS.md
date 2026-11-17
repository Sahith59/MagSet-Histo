# UNI2-h linear field classifier: completed negative result

November 17, 2026. **The standard supervised linear classifier does not improve the final five-category task.** Its primary two-step arm scores **73.83% / 73.67% accuracy** and **.575342 / .535412 macro F1** in arrangements 812 / 813. Both the primary same-UNI2-h KDE screen and secondary retained-P screen fail. The binary decision improves, but malignant typing generalizes worse. **94–95% final accuracy and novelty remain unachieved.**

This page reports the exact [complete result ledger](evidence/BREAKHIS_UNI_LINEAR_RESULTS.json), [registration](BREAKHIS_UNI_LINEAR_SPEC.md) and [configuration](../configs/breakhis-uni-linear-v1.json). All ten original roles, both architectures and all four C candidates completed and were replayed. The separate [head-metric study](BREAKHIS_SPECIALIST_METRIC_RESULTS.md) is a different family; its factors or predictions were not used here.

## What the classifier learned

The reader remains frozen UNI2-h, using the same 7,909 original images and cached 1,536-dimensional field descriptors. Instead of comparing fields with labeled support examples through cosine KDE, the new classifier optimizes standard L2-regularized logistic decision boundaries from fitting labels. Direct-five learns one multinomial classifier. The requested two-step arm learns benign/malignant plus a separate malignant-four multinomial expert; benign terminates at step one.

Reuse the existing field learner and its exact manual class/case/view/field weights. Each field is standardized using fitting-only equal-case per-view first/second moments, saved in FP32, and receives four one-hot magnification indicators: **1,540 inputs**. There is no L2 feature normalization or learned metric factor. KDE instead uses support centering and cosine normalization, so this evaluates a **changed normalization and supervised readout package**, not an isolated change to the decision-boundary loss.

Binary and direct-five fitting use 46 cases and total sample-weight mass 46. Malignant-four uses 35 malignant fitting cases and mass 35. Automatic class weighting is disabled because manual weights already give equal class mass. `lbfgs` uses L2, tolerance 1e-4, maximum 2,000 iterations and seed 812. The four fixed C values are .01/.1/1/10; the dual heads share C. Each arm/role selects the earliest maximum validation mean-fifteen-subset five-class macro F1.

## Same labels, roles and endpoint

Final labels are pooled benign, ductal, lobular, mucinous and papillary carcinoma. Each original role has **46 fitting/eight validation/28 excluded provider cases**. Fitting counts are 11/25/2/5/3; validation counts are 4/1/1/1/1. No excluded-role case contributes to normalization, fitting, selection or predictions.

Both arrangements contain five overlapping trials, all four magnifications and all fifteen nonempty subsets. Each arrangement therefore has 600 dependent validation case/subset events; the all-four-view endpoint has 40. Macro F1 is averaged within trial/subset rather than calculated from a pooled confusion matrix. No exact cross-view field correspondence is assumed. Inference composes field-level joint probabilities before equal field/view averaging and the fixed .5 malignant route. The direct control uses a pooled five-class argmax and remains a separate one-stage endpoint.

## Complete validation comparison

| Classifier / endpoint | Accuracy 812 | Macro F1 812 | Accuracy 813 | Macro F1 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNI2-h linear direct-five control | 73.00% | .566610 | 73.50% | .528383 | 75.00% / 72.50% |
| **UNI2-h linear two-step: primary** | **73.83%** | **.575342** | **73.67%** | **.535412** | **75.00% / 72.50%** |
| Accepted UNI2-h KDE two-step: primary reference | 75.83% | .664180 | 75.17% | .575175 | 77.50% / 75.00% |
| Retained P two-step: secondary reference | 76.33% | .602715 | 75.67% | .602512 | 85.00% / 85.00% |

The linear two-step arm gets **443/600 and 442/600** events correct, versus same-reader KDE's 455/600 and 451/600. Accuracy drops **2.00 / 1.50 percentage points**, and macro F1 drops **.088838 / .039763**. Against P, accuracy drops **2.50 / 2.00 points** and F1 **.027374 / .067101**. Both declared screens fail on both arrangements, although every final label retains positive recall. The result supplies no replacement of either reference.

Within the linear family, two-step exceeds direct-five by +.83/+.17 accuracy points and +.008732/+.007029 F1. This is an architecture comparison with different head balancing, probability composition and independently selected C; it does not isolate a threshold or prove a hierarchy bug. The new direct-five result is also below the earlier same-reader direct-five KDE result, which scored 75.17%/77.67% and .648631/.634493 F1. A one-stage result is not substituted for the required two-step endpoint.

Selected C counts are shown in the fixed order .01/.1/1/10:

| Arm | Counts 812 | Counts 813 |
| --- | --- | --- |
| Direct-five | `[4, 0, 0, 1]` | `[2, 1, 1, 1]` |
| Two-step | `[4, 0, 0, 1]` | `[2, 0, 2, 1]` |

No C range was widened or selected after viewing these results. The head metrics of the separate metric-learning study were not applied to this classifier.

## Better binary separation, weaker malignant generalization

These component diagnostics refer to the **deployed two-step arm**:

| Diagnostic | UNI2-h KDE 812 / 813 | UNI2-h linear 812 / 813 |
| --- | ---: | ---: |
| Binary router accuracy | 83.33% / 90.33% | 91.83% / 96.17% |
| Malignant sensitivity | 82.00% / 85.33% | 99.33% / 99.67% |
| Benign specificity | 84.67% / 95.33% | 84.33% / 92.67% |
| True-malignant subtype accuracy | 72.00% / 60.67% | 63.67% / 54.67% |

The primary arm has **157 final errors = 49 routing errors + 108 wrong types after correct routing** on 812, and **158 = 23 + 135** on 813. Same-reader KDE had 145 = 100 + 45 and 149 = 58 + 91. Thus improved routing is more than offset by weaker typing, rather than revealing a hidden near-95% expert. With these case-conditional type predictions held fixed, even perfect routing would yield only **81.83% / 77.33%** final accuracy; this counterfactual does not bound a different future classifier.

Final two-step recalls:

| Classifier / arrangement | Benign | Ductal | Lobular | Mucinous | Papillary |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNI2-h KDE / 812 | 84.67% | 64.00% | 100.00% | 77.33% | 26.67% |
| Linear / 812 | 84.33% | 97.33% | 54.67% | 80.00% | 21.33% |
| UNI2-h KDE / 813 | 95.33% | 56.00% | 93.33% | 42.67% | 28.00% |
| Linear / 813 | 92.67% | 88.00% | 54.67% | 61.33% | 14.67% |

The linear classifier strongly favors correct ductal recognition at the expense of lobular and papillary generalization. In each arrangement, **34/75 lobular events are called ductal**. On 813, 49/75 papillary events are called mucinous; only 11/75 are correct and one is sent to benign. On 812, only 16/75 papillary events are correct, with 29 called ductal, 14 lobular and 14 mucinous. These are measured confusion locations, not proof of label defects, stain effects or another biological cause.

The direct arm's summed-malignancy diagnostic is not its deployed argmax router. Do not interpret its binary-score accuracy or binary confusion decomposition as though the direct endpoint used the two-step decision rule.

## Fitting, probabilities and optimization

| Arm | Fitting accuracy 812 / 813 | Fitting macro F1 812 / 813 | Validation accuracy 812 / 813 |
| --- | ---: | ---: | ---: |
| Direct-five | 98.67% / 98.70% | .975943 / .976613 | 73.00% / 73.50% |
| Two-step | 98.58% / 98.43% | .972687 / .970708 | 73.83% / 73.67% |

Fitting scores are **in-sample resubstitution**: both optimizer and normalizer saw those cases and fields. They are neither leave-case-out model checks nor clinical performance. The substantial fitting/validation gap supports a generalization limitation of this package; it does not indicate failure to fit its training objective. Related fields and four views do not turn two lobular or three papillary fitting cases into a large independent subtype sample.

All **120 production heads** satisfy the fixed convergence acceptance: observed iterations range **14–121**, below 2,000, and the maximum independently reconstructed normalized absolute gradient is **9.975666e-5**, below 1e-4 + 1e-8. The maximum saved-coefficient/native-scikit probability replay difference is **1.8874e-15**, below 1e-12. The full collector verifies coefficient, normalizer, weight-mass, objective, prediction and metric replay; it performs no optimization.

Proper losses are mixed versus same-reader KDE. Primary two-step cross-entropy is **.872262 / .884087**, compared with .957088/.836066: improved on 812 and worse on 813. Brier loss is **.404797 / .401783**, below KDE's .472184/.409558 in both arrangements. Both losses are below contextual P, despite worse hard-label accuracy/F1. Probability losses and thresholded classification measure different properties; no empirical calibration conclusion follows from logistic probabilities alone. Mean trial worst-subset F1 is .422190/.437905, also a mixed comparison to KDE's .480000/.377556.

The shared field-level linear model can use view indicators as view-specific intercept shifts, but it has no learned feature-by-view interactions or attention to a few diagnostic fields. Together with the small independent case counts, field-level inherited labels and averaging, these are plausible limitations, not isolated measured causes. This negative result does not authorize another learner search or automatic combination with the metric study.

## Complete budget and fitting-only acceptance

The production matrix contains **ten roles, 80 candidate pipelines, 120 optimized heads, 160 prediction archives and 64,800 dependent case/subset records**. Each complete arm has four C candidates. One C is shared by both dual heads; there is no head-wise C tuning, hidden calibration cross-validation, repeated seed or threshold selection. Zero new encoder forwards and zero excluded-role predictions occurred.

An additional **12 technical head fits** cover all four C values and all three heads on the complete 812/trial-1 fitting role. The accepted smoke scores one fitting query from each final label across fifteen subsets, giving eight archives and **600 fitting-only records**, with zero validation records. Its independent acceptance reconstructs fitting-only normalization, weights, objectives, convergence, saved coefficients and probabilities. These twelve technical fits are separate from the 120 production fits, giving 132 actual optimized heads overall for this family.

Input audit **4983955**, technical smoke **4983956**, independent acceptance **4983957**, ten-role production **4983958** and collector **4983959** belong to source `3369de0280ffbdfa6080cc10699d748a2a75cb3e`. The exact completed collection verifies all original roles and every candidate/archive. The registered CPU allocation was four CPUs, sixteen GiB and forty-five minutes per role, at most four concurrent tasks, with thirty minutes for collection. Saved candidate elapsed times range **1.32–7.83 seconds**, summing to **294.66 seconds**; these are candidate program times, not scheduler queue time or hardware utilization.

## Selection history and scope

The family was separately registered before its fitting or validation outcomes and before inspection of the metric-study results. It adds **four C choices per arm/role**, or eight arm/C combinations if choosing between architectures. Earlier adaptive choice of UNI2-h, prior reader/temperature choices and the separate metric family remain part of the cumulative development history. This report retains the requested dual primary comparison even though both architectures are reported.

There are **31 distinct reused validation provider keys**, dependent overlapping trials and magnification subsets, and unresolved clinical identity. All 82 qualified provider keys have entered development. The 600 events per arrangement must not be treated as 600 independent patients. This is exploratory provider-case validation, not a globally fresh final cohort or clinical confirmation. The family is standard transfer learning and supplies no novelty evidence. Retain P under the established comparison decision while reporting UNI2-h and metric-family F1 tradeoffs honestly; no universal F1 leader or 94–95% five-category result is established.

## Exact provenance

The linked public result is the exact collector output; trained coefficients, raw feature arrays, detailed smoke artifacts and private job records remain private.

| Artifact | Revision / SHA256 |
| --- | --- |
| Numerical producer source | `3369de0280ffbdfa6080cc10699d748a2a75cb3e` |
| Producer source archive | `7f23d7235eaa7d239444125833f0f4ab0159473e0446ab176a85c3af0bb2d5c7` |
| [Complete result ledger](evidence/BREAKHIS_UNI_LINEAR_RESULTS.json) | `465f89c6ea8431ee0c9e37a835f85913602673a32a6288ae209e60f5becab9ed` |
| [Registered configuration](../configs/breakhis-uni-linear-v1.json) | `39ac36c3ade6693c0a3fb49937e64b1c105a42a348be3124dc4d996089ca77b0` |
| Accepted fitting-only smoke | `461bb6fc70e7f04f5ed450ad65af856e445138973677b62475059ff16e9cc360` |
| Input audit | `e0a540490c5bf4ad0f56ed13c315493c3956726c91b944bf29734255891a61fc` |
| Dataset manifest | `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31` |
| Original role protocol | `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134` |
| UNI2-h feature producer source | `375bcbc499770635dde6c9586c5ce49e2c44300d` |
| Accepted full feature audit | `9f5895589bad5fc29a9d2c77378ca72e98ea26da9a98ee1512dc25ae166051ff` |
| [Same-reader KDE reference](evidence/BREAKHIS_UNI2_H_RESULTS.json) | `f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230` |
| [Retained-P reference](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) | `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c` |

The exact ledger records Python 3.11.18, NumPy 1.26.4, SciPy 1.15.2 and scikit-learn 1.6.1, plus encoder/configuration revisions, all four cache hashes, implementation hashes, saved models, candidate selections and all per-role outcomes. Actual cohort predictions support the numbers on this page; synthetic checks establish only source mechanics. The complete result is a failed improvement screen with improved binary separation and unresolved malignant-type generalization.
