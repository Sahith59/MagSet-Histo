# Independent review: actual approved-encoder results

November 16, 2026. **Accept all three complete result ledgers' saved arithmetic, selections and provenance bindings. None passes its registered two-arrangement improvement screens.** UNI2-h nevertheless establishes the highest observed **812 two-step macro F1, .664180**; P retains the highest observed **813 two-step F1, .602512**. The earlier statement that P leads on both arrangements is now stale. These separate observations must not be combined into one pipeline result. [Registered recipe](../BREAKHIS_APPROVED_ENCODERS_SPEC.md), [prospective validity and technical review](APPROVED_ENCODER_VALIDITY_REVIEW.md).

This review independently reads accepted collector JSONs, existing feature-audit/technical receipts and immutable source/configuration archives. It does **not** independently replay private raw prediction JSONL, feature NPZ arrays, original images or model weights. The cluster collector's complete cached-feature prediction replay is separate execution evidence. No new fitting, encoder inference, download, scheduler action or production-code change was performed here.

Independent verification across all three ledgers passed **968,317 arithmetic/provenance assertions**, including **23,040 candidate confusion matrices** and **60 earliest-maximum F1 selections**. Each reader contributes 7,680 matrices: 7,200 five-class/router/conditional-type subset matrices and 480 pooled matrices. All 240 candidate pipelines, 480 prediction archives and 194,400 dependent candidate records remain represented. All three feature producers are source `375bcbc499770635dde6c9586c5ce49e2c44300d`; immutable classifier sources/configuration hashes and every recorded implementation hash were independently matched to their source/archive bytes.

| Accepted reader | Classifier producer | Result-ledger SHA256 | Independent assertions |
| --- | --- | --- | ---: |
| UNI2-h | `a981baf38df116c3d5cfd2381a76630b078b11bf` | `f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230` | 322,781 |
| Virchow2 | `a981baf38df116c3d5cfd2381a76630b078b11bf` | `dbd0085b03fde377d8e9ec2b3359a232a3740847abaf72e58f05eb54907c5cc6` | 322,773 |
| H-optimus-0 | `20b3463467901efd370025d93b1465c7a101dc97` | `9e316a67c805dffeb71ce4a73de29f5861d719945398cb7f38772ff1132e9659` | 322,763 |

## All three readers: actual input-receipt acceptance

An additional **192 independent assertions** passed across the three preparation/smoke/full-feature audit receipt sets. Each matches source `375bcbc499770635dde6c9586c5ce49e2c44300d`, its own configuration/reader/readout, original manifest, artifact pins and the exact 1,995/2,081/2,013/1,820 image counts across the four magnifications. Each full audit reports 7,909 complete fields, canonical one-image model calls and zero new audit-time encoder forwards/image decodes. Every smoke reports strict FP32 full-state copying, all three parity errors exactly zero, the same four predeclared fitting-only fields and the registered geometry. These are acceptance of recorded lineage/coverage checks, without an independent NPZ-byte or image replay in this review.

| Reader | Output dimensions | Recorded FP32 parameters | Smoke peak allocated GPU bytes | Steady four-image host group, seconds |
| --- | ---: | ---: | ---: | ---: |
| UNI2-h | 1,536 | 681,394,176 | 2,820,907,008 | .132957 |
| Virchow2 | 2,560 | 631,239,424 | 2,603,693,568 | .124799 |
| H-optimus-0 | 1,536 | 1,134,774,272 | 4,650,800,128 | .220849 |

The four-image timing is technical smoke evidence on A40, comprising four sequential canonical model calls, rather than a full-dataset throughput benchmark. Feature-audit SHA256 bindings:

- UNI2-h: `9f5895589bad5fc29a9d2c77378ca72e98ea26da9a98ee1512dc25ae166051ff`.
- Virchow2: `f63926930e0ab7c9542f3c47758850e7184efbe52d8a55ea1cd1b21d0783c5a3`.
- H-optimus-0: `7165f9eda9a16790c6a4ec3be95e2454c9fe687a17a5099d2d55ed9ca7be4423`.

## UNI2-h and Virchow2: complete result acceptance

All values below use the same final-five endpoint and role/selection contract, ordered **812 / 813**. Correct-record counts each have denominator 600; all-four-view scores each have denominator 40.

| Reader and arm | Final accuracy | Correct records, each /600 | Macro F1 | All-four-view accuracy |
| --- | ---: | ---: | ---: | ---: |
| UNI2-h two-step | 75.83% / 75.17% | 455 / 451 | .664180 / .575175 | 77.50% / 75.00% |
| UNI2-h direct-five | 75.17% / 77.67% | 451 / 466 | .648631 / .634493 | 75.00% / 80.00% |
| Virchow2 two-step | 72.50% / 73.00% | 435 / 438 | .575767 / .538743 | 72.50% / 77.50% |
| Virchow2 direct-five | 72.83% / 75.00% | 437 / 450 | .590080 / .587676 | 75.00% / 77.50% |

UNI2-h's two-step F1 gain over C is **+.074489 / +.010883**; raw accuracy changes by **+0.17 / +0.17 percentage points**. Its primary screen fails because the 813 F1 gain is below +.02. Its secondary F screen also fails. Against P, its 812 F1 is higher, its 813 F1 is lower, and accuracy is lower by .50 points on both. Its two-step all-four-view scores are below P's separate 85%/85% endpoint. The new 812 observation is a measured improvement in one arrangement, without a consistent two-arrangement winner or a 95% result. UNI2-h's direct arm is stronger than its dual arm on 813 but weaker on 812; hierarchy is not uniformly beneficial.

Virchow2's two-step final accuracy/F1 are below C/F/P on both arrangements, and both registered screens fail. Its direct-five control exceeds its own two-step accuracy/F1 on both arrangements. Proper losses nevertheless improve against C/F/P on both arrangements, as quantified below; calling every metric worse would be incorrect.

| Two-step diagnostic | UNI2-h 812 / 813 | Virchow2 812 / 813 |
| --- | ---: | ---: |
| Binary routing accuracy | 83.33% / 90.33% | 89.00% / 89.50% |
| Accuracy on all truly malignant cases | 72.00% / 60.67% | 59.33% / 56.00% |

End-to-end recalls retain class order benign/ductal/lobular/mucinous/papillary:

- UNI2-h 812: **84.67/64.00/100.00/77.33/26.67%**; 813: **95.33/56.00/93.33/42.67/28.00%**.
- Virchow2 812: **88.00/66.67/98.67/44.00/18.67%**; 813: **94.67/42.67/97.33/36.00/29.33%**.

The table gives candidate-minus-fixed-reference proper-loss deltas for the two-step arms; lower is better:

| Reader / reference | 812 cross-entropy delta | 812 Brier delta | 813 cross-entropy delta | 813 Brier delta |
| --- | ---: | ---: | ---: | ---: |
| UNI2-h / C | +.014885 | +.033103 | −.089240 | −.040848 |
| UNI2-h / F | +.036031 | +.041451 | −.046103 | −.023813 |
| UNI2-h / P | −.020989 | +.014372 | −.233086 | −.121569 |
| Virchow2 / C | −.051815 | −.014231 | −.103452 | −.043840 |
| Virchow2 / F | −.030669 | −.005883 | −.060315 | −.026804 |
| Virchow2 / P | −.087688 | −.032962 | −.247298 | −.124561 |

UNI2-h improves C/F probability errors on 813 and worsens them on 812. Against P, 812 cross-entropy improves while Brier worsens; both improve on 813. Probability errors and classification therefore support different comparisons. All these proper-loss aggregate/decomposition checks have the same raw-probability reconstruction limitation described below for H-optimus-0.

## H-optimus-0: complete result acceptance

Result SHA256: `9e316a67c805dffeb71ce4a73de29f5861d719945398cb7f38772ff1132e9659`. Classifier producer: `20b3463467901efd370025d93b1465c7a101dc97`. Feature producer: `375bcbc499770635dde6c9586c5ce49e2c44300d`. Complete feature-audit SHA256: `7165f9eda9a16790c6a4ec3be95e2454c9fe687a17a5099d2d55ed9ca7be4423`.

Independent verification passed **322,763 arithmetic/provenance assertions**, including **7,680 candidate confusion matrices** (7,200 subset matrices plus 480 pooled matrices) and **20 earliest-maximum F1 selections**. These cover all 80 direct/dual temperature candidates, both fitting and validation roles, every magnification subset, five-class decisions, binary routing and true-malignant conditional typing.

Verified checks include original role/class supports, nonnegative integer matrices, accuracy/macro F1/balanced accuracy/recall from those matrices, pooled totals, equal subset/trial means, full-view identity, worst-subset scores, selected-candidate identity and all reported reference deltas/screens. For two-step candidates, collapsing the final five-class matrix reproduces the binary routing matrix. Conditional typing matrices consistently account for malignant predictions suppressed by benign routing. Joint cross-entropy equals binary cross-entropy plus the true-malignant fraction times conditional cross-entropy. Probability loss aggregates and this decomposition were checked; losses and AUROC/AP were not independently reconstructed from raw probabilities here.

| Arm | Arrangement | Final accuracy | Correct / 600 | Macro F1 | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Requested two-step | 812 | 73.50% | 441/600 | .509710 | 77.50% = 31/40 |
| Requested two-step | 813 | 71.17% | 427/600 | .506235 | 75.00% = 30/40 |
| Direct-five control | 812 | 77.33% | 464/600 | .603560 | 87.50% = 35/40 |
| Direct-five control | 813 | 75.17% | 451/600 | .606677 | 82.50% = 33/40 |

Each arm/arrangement averages 15 subsets and five overlapping trials, with eight validation cases per trial. The 600 records are dependent case/subset observations; the all-four-view endpoint has 40 dependent observations. The full candidate ledger has 160 fitting/validation archives and 64,800 dependent records, with 30 full-support heads shared over temperatures and zero optimized classifier fits or new encoder forwards during classification/collection.

The direct control exceeds this reader's two-step F1 by .093850/.100443 and accuracy by 3.83/4.00 percentage points. Its F1 slightly exceeds retained P's two-step F1, while its 813 accuracy is lower. This is a separate one-stage observation. The requested two-step remains below C/F/P in both final accuracy and F1 and fails the C-primary and F-secondary screens.

## H-optimus-0: routing and probability qualifications

Two-step binary accuracy is **92.83% / 84.33%**; conditional typing accuracy on all truly malignant validation cases is **58.33% / 60.33%**. Neither equals final-five accuracy. End-to-end recalls, ordered benign/ductal/lobular/mucinous/papillary, are **97.67/89.33/60.00/46.67/1.33%** on 812 and **96.67/25.33/93.33/30.67/33.33%** on 813. The papillary 812 observation is one correct dependent record out of 75, rather than evidence from 75 independent papillary patients.

Probability errors have mixed comparison outcomes; lower is better. The following are H-optimus two-step minus each fixed two-step reference:

| Reference | 812 cross-entropy delta | 812 Brier delta | 813 cross-entropy delta | 813 Brier delta |
| --- | ---: | ---: | ---: | ---: |
| C: single-center Midnight | −.021196 | −.024970 | +.039854 | +.035741 |
| F: three-crop Midnight | −.000051 | −.016622 | +.082990 | +.052777 |
| P: decision-specific centering | −.057070 | −.043700 | −.103993 | −.044980 |

Thus proper losses improve versus C/F on 812 and worsen on 813; they improve versus P on both arrangements. Stronger final classification and better probability errors remain separate observations. No universal improvement or universal deterioration is established.

## Scope carried forward

All original provider-key roles remain unchanged: 46 fitting/eight validation/28 excluded cases per trial; 22/20 distinct validation keys and 31 in their union. All 82 keys entered earlier development, clinical identities remain unresolved and no globally fresh final cohort exists. Reader selection across this three-reader family entails 12 reader/temperature settings per arm/role. All outcomes must remain visible. Technical/result-ledger acceptance establishes neither 95% final accuracy, clinical validation nor novelty.
