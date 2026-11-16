# Approved encoders: diagnosis of the actual model outcomes

November 16, 2026. **None of the three approved readers passes its paired two-step improvement screen. Retain P as the current reference under the registered criteria, while reporting UNI2-h's substantial 812 F1 gain.** UNI2-h has the highest observed two-step F1 on 812 and the highest simple average of the two arrangement F1 values; P has higher 813 F1 and higher accuracy on both arrangements. There is no universal F1 leader. No 94–95% final-five result is established.

I authored the shared approved-encoder feature producer. This is a read-only diagnosis of actual saved classifier outcomes and related preprocessing assumptions, **not an independent certification of that feature implementation or raw-image reproduction**. No new fit, prediction, encoder execution, setting search or scheduler action was performed. I independently recalculated 60 earliest-maximum selections, 60 selected subset-mean metric aggregates and 12 pooled confusion matrices/counts from the three complete ledgers. The separate validity review addresses a different acceptance question.

## Complete outcome and retained decision

Headline scores average all 15 nonempty magnification subsets and five overlapping trials for arrangements 812 / 813. Accuracy uses 600 dependent validation case/subset events per arrangement; all-four-view accuracy uses 40. Macro F1 is averaged within each trial/subset, rather than calculated from one pooled matrix. The 31 distinct validation provider keys are reused, all 82 provider keys have entered development somewhere, and clinical identities remain unresolved. These are exploratory development outcomes, not independent clinical confirmation.

| Pipeline | Final accuracy 812 / 813 | Macro F1 | All-four-view accuracy |
| --- | ---: | ---: | ---: |
| C, single-center Midnight / dual | 75.67% / 75.00% | .589691 / .564292 | 82.50% / 80.00% |
| F, three-crop Midnight / dual | 75.50% / 76.83% | .592510 / .579947 | 82.50% / 82.50% |
| P, decision-specific centers / dual | 76.33% / 75.67% | .602715 / .602512 | 85.00% / 85.00% |
| UNI2-h / dual | 75.83% / 75.17% | .664180 / .575175 | 77.50% / 75.00% |
| Virchow2 / dual | 72.50% / 73.00% | .575767 / .538743 | 72.50% / 77.50% |
| H-optimus-0 / dual | 73.50% / 71.17% | .509710 / .506235 | 77.50% / 75.00% |
| UNI2-h / direct-five | 75.17% / 77.67% | .648631 / .634493 | 75.00% / 80.00% |
| Virchow2 / direct-five | 72.83% / 75.00% | .590080 / .587676 | 75.00% / 77.50% |
| H-optimus-0 / direct-five | 77.33% / 75.17% | .603560 / .606677 | 87.50% / 82.50% |

UNI2-h's dual F1 gains over primary C are +.074489/+.010883. Its accuracy gains are +0.17/+0.17 percentage points, but the 813 F1 gain does not reach the required .02. Against F, UNI2-h loses 813 accuracy and F1. Against contextual P, UNI2-h gains .061465 F1 on 812, loses .027338 on 813, and loses .50 accuracy points on both. Its simple two-arrangement F1 average is .619677 versus P's .602614; this descriptive average does not replace the prospectively specified paired screen. Virchow2 and H-optimus-0 do not improve dual accuracy/F1 over C on either arrangement.

Direct-five controls remain one-stage results. Direct versus dual changes class-density/group weighting, joint-probability aggregation, final decision rule and separately selected temperature. The source uses the same support-only per-view center for these arms. The comparison does not isolate a threshold change, and a stronger direct score does not identify a hierarchy implementation bug. In particular, the direct arm's summed-malignancy binary diagnostic is not its final argmax behavior and must not be called its deployed router.

## Routing and malignant typing identify different bottlenecks

| Selected dual reader / arrangement | Router accuracy | True-malignant type accuracy | Final errors | Routing errors | Wrong type after correct route | Perfect-router accuracy, current conditional predictions fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNI2-h / 812 | 83.33% | 72.00% | 145 | 100 | 45 | 86.00% |
| UNI2-h / 813 | 90.33% | 60.67% | 149 | 58 | 91 | 80.33% |
| Virchow2 / 812 | 89.00% | 59.33% | 165 | 66 | 99 | 79.67% |
| Virchow2 / 813 | 89.50% | 56.00% | 162 | 63 | 99 | 78.00% |
| H-optimus-0 / 812 | 92.83% | 58.33% | 159 | 43 | 116 | 79.17% |
| H-optimus-0 / 813 | 84.33% | 60.33% | 173 | 94 | 79 | 80.17% |

Each error decomposition uses 600 dependent events. An incorrect broad route is always a final error; subtracting routing errors from final errors gives wrong malignant types after correct routing. The ideal-router calculation is `0.5 + 0.5 × current true-malignant type accuracy`, because validation has equal benign/malignant support. It holds only while those case-conditional subtype predictions remain fixed. It does not bound another representation or field-pooling intervention that changes them.

**UNI2-h improves specialist discrimination strongly on 812, but loses broad separation.** Relative to C, true-malignant accuracy rises from 56.67% to 72.00%, while routing drops from 92.00% to 83.33%. Of its 100 routing mistakes, 46 are benign events called malignant and 54 are malignant events sent to benign. Those include 44 papillary events. On 813, routing remains below C and typing improves much less; 91/149 final errors arise after a correct route. A router repair alone cannot address that arrangement's specialist limitations.

Virchow2's true-malignant accuracy is 59.33%/56.00%, with 99 wrong types after correct routing in each arrangement. This is chiefly a weak subtype-transfer package alongside a weaker router than C/F/P, rather than a hidden high-performing expert awaiting only a new threshold. H-optimus-0 has different failure balances: 116/159 errors occur in post-route typing on 812, while routing accounts for 94/173 errors on 813.

Router ranking and fixed decisions also differ. H-optimus-0's mean ROC AUC is .993333/.990000 despite malignant sensitivity .88/.72 at the fixed .5 threshold. UNI2-h's mean AUC is .928333/.985833 with sensitivity .82/.853333 and benign specificity .846667/.953333. These suggest probability/operating-point issues as well as discrimination differences, without identifying a unique calibration cause. AUC is averaged over small trial/subset samples, so it does not prove one global alternative threshold would repair the mistakes. No threshold was searched.

## Class tradeoffs remain substantial

Final recalls preserve benign, ductal, lobular, mucinous and papillary order:

| Dual reader / arrangement | Benign | Ductal | Lobular | Mucinous | Papillary |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNI2-h / 812 | 84.67% | 64.00% | 100.00% | 77.33% | 26.67% |
| UNI2-h / 813 | 95.33% | 56.00% | 93.33% | 42.67% | 28.00% |
| Virchow2 / 812 | 88.00% | 66.67% | 98.67% | 44.00% | 18.67% |
| Virchow2 / 813 | 94.67% | 42.67% | 97.33% | 36.00% | 29.33% |
| H-optimus-0 / 812 | 97.67% | 89.33% | 60.00% | 46.67% | 1.33% |
| H-optimus-0 / 813 | 96.67% | 25.33% | 93.33% | 30.67% | 33.33% |

UNI2-h's 812 macro-F1 gain is compatible with better class balance: lobular/mucinous recalls reach 100%/77.33% while benign recall falls to 84.67%. Since benign has half the validation events, these changes can increase F1 and balanced accuracy substantially while barely changing raw accuracy. The gain is measured and should remain visible, without relabeling it as a paired replacement of P.

For UNI2-h on 813, ductal and mucinous conditional type recalls are only 56.00%/54.67%; 30 ductal events are typed as mucinous and 33 mucinous events as ductal. Papillary conditional recall is 38.67%, before the 33 events routed to benign. Virchow2 preserves high lobular recall but does not recover mucinous/papillary performance broadly. A strong result for one rare type is not universal rare-class recovery.

H-optimus-0 on 812 gets only one of 75 papillary events finally correct: 18 are sent to benign and 56 to a wrong carcinoma type. Even with correct routing, its current expert identifies only 13/75 papillary events. On 813, ductal typing is correct on only 19/75 events even before routing. Mucinous/papillary conditional recalls are 62.67%/60.00%, but 34/45 events respectively are routed to benign, reducing final recalls to 30.67%/33.33%. These are error locations, not evidence of label defects or a biological cause.

## Leave-case-out support and selection

| Dual reader | Fitting leave-case-out accuracy 812 / 813 | Fitting leave-case-out F1 |
| --- | ---: | ---: |
| UNI2-h | 64.43% / 67.16% | .476545 / .498092 |
| Virchow2 | 65.59% / 65.36% | .461511 / .431735 |
| H-optimus-0 | 69.91% / 63.39% | .477417 / .449436 |

This is not the earlier MIL model's near-perfect in-sample fitting pattern. There are no optimized classifier coefficients or training loss trajectories here to diagnose as failed optimization. Every fitting query is removed from support and centering. Two lobular and three papillary fitting cases per full role leave only one other lobular or two other papillary cases for that query class. The fitting class mixture also differs from validation. Thus validation exceeding these diagnostic scores does not establish leakage or contradict sparse cross-case support.

All four temperatures remain .01/.03/.1/.3. Selected dual counts in that order are UNI2-h `[0,1,2,2] / [0,1,4,0]`, Virchow2 `[0,2,3,0] / [1,1,3,0]`, and H-optimus-0 `[0,4,1,0] / [0,1,2,2]`. Selection varies by reader and arrangement; it does not reveal an untested optimum or justify widening the grid. Every named reader/arm has four opportunities per role. Choosing among this family uses **12 reader/temperature settings per arm/role**, and reporting an average across arrangements does not create independent replication.

## Proper losses do not follow the F1 ranking

| Dual pipeline | Validation cross-entropy 812 / 813 | Five-class Brier loss |
| --- | ---: | ---: |
| C | .942203 / .925306 | .439081 / .450406 |
| F | .921057 / .882170 | .430733 / .433371 |
| P, decision-specific | .978076 / 1.069153 | .457811 / .531127 |
| UNI2-h | .957088 / .836066 | .472184 / .409558 |
| Virchow2 | .890388 / .821855 | .424850 / .406567 |
| H-optimus-0 | .921007 / .965160 | .414111 / .486148 |

Virchow2 improves both proper losses over C/F in both arrangements despite weaker final accuracy/F1. UNI2-h's probability losses worsen against C/F on 812 and improve on 813. H-optimus-0 improves against C/F on 812 and worsens on 813. Each approved reader has better proper losses than P on both arrangements except UNI2-h's 812 Brier, which is worse. Direct arms have worse CE/Brier than their corresponding dual arms in both arrangements, despite several stronger F1 scores. Proper losses assess probability quality, not calibration alone; no calibration curves were analyzed. Neither a universal failure nor a universal gain across metrics is supported.

The comparison changes each frozen reader together with its declared transform/readout. H-optimus-0 uses the registered BreakHis resize/center-crop adaptation followed by official normalization; its native physical-scale assumption is not established. No controlled experiment isolates physical scale, stain, morphology, crop information loss or another biological cause. Successful loading and numerical parity establish execution, not favorable task geometry.

## One bounded next step, conditional on continuing model work

First freeze these outcomes and finish the five-category capstone packet/demo with P as current reference, UNI2-h's 812 gain visible, and all negative/mixed rows retained. This is the immediate priority. The results do not justify automatic cross-reader fusion, another backbone search or an adapter sweep.

For a separately authorized model study after that deliverable, prioritize **cross-case malignant subtype generalization**. A bounded candidate is one strongly identity-regularized diagonal metric on frozen UNI2-h features, compared with an exactly matched fixed-identity metric control. Keep the same UNI2-h binary router, support weighting, pooling, .5 rule, roles, masks and original four-temperature budget. Fit the malignant metric only from the 35 fitting malignant cases, using different cases of the same subtype as positive examples; never use two fields from the same case as evidence of cross-case generalization. Use a fixed training duration/regularization and report the identity control and candidate with their full combined selection budget. No P-router swap, second reader, new crops, adapter or checkpoint sweep belongs in that experiment.

This is a proposed standard metric-learning question, **not a registered recipe or permission to launch it**. Its motivation is the remaining subtype mistakes in every reader and UNI2-h's measured 72%/60.67% conditional accuracy. Its principal counterevidence is sparse support: only two lobular and three papillary fitting cases, plus prior negative neural/adaptation studies. A complete registration must fix the case representation, metric bounds/identity penalty, cross-case objective, optimizer, training duration and control before execution. Any improvement would remain exploratory reused validation, not established novelty or clinical generalization.

A fit-only two-parameter calibration of UNI2-h's case binary log odds, with an identity-calibration control, is a narrower optional diagnostic for its 100 routing errors on 812. It could preserve the specialist and final .5 rule without a validation threshold sweep. It is not the primary next improvement recommendation: with current conditional predictions fixed, even perfect routing reaches only 86.00%/80.33%. Calibration alone cannot reach 95% here or repair specialist discrimination. Neither proposal is an automatic next job.

## Source receipts

All three ledgers contain ten roles, 80 candidates, 160 prediction archives and 64,800 dependent candidate records per reader. Feature producer is `375bcbc499770635dde6c9586c5ce49e2c44300d`. UNI2-h/Virchow2 classifier producer is `a981baf38df116c3d5cfd2381a76630b078b11bf`; H-optimus-0 producer is `20b3463467901efd370025d93b1465c7a101dc97`.

| Complete ledger | SHA256 |
| --- | --- |
| [UNI2-h](../evidence/BREAKHIS_UNI2_H_RESULTS.json) | `f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230` |
| [Virchow2](../evidence/BREAKHIS_VIRCHOW2_RESULTS.json) | `dbd0085b03fde377d8e9ec2b3359a232a3740847abaf72e58f05eb54907c5cc6` |
| [H-optimus-0](../evidence/BREAKHIS_H_OPTIMUS_0_RESULTS.json) | `9e316a67c805dffeb71ce4a73de29f5861d719945398cb7f38772ff1132e9659` |

Related evidence: [registered comparison](../BREAKHIS_APPROVED_ENCODERS_SPEC.md), [C](../evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), [F](../evidence/BREAKHIS_COVERAGE_RESULTS.json), [P](../evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json), and [separate validity review](APPROVED_ENCODER_VALIDITY_REVIEW.md).
