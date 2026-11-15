# Five-category target: saved-error and metric diagnosis

November 15, 2026. Independent retrospective audit of the accepted frozen, adapted and supervised **eight-class** validation predictions. The corrected target is pooled benign plus ductal, lobular, mucinous and papillary carcinoma; benign stops at the binary gate. **No trained direct-five control or binary-plus-malignant-four model is evaluated here.** The descriptive readouts below reuse already known validation outcomes; they are not a new test, an architecture improvement or evidence of 94–95% performance.

## What the existing errors establish

The retained frozen model has two substantial remaining problems: benign cases being called malignant and malignant subtypes being confused with one another. Pooling the benign labels removes a large category of historical errors automatically, including many phyllodes errors, but cannot solve malignant typing.

Each row below contains all five trials and all 15 nonempty magnification subsets: 600 repeated case/subset events per arrangement. Categories use the original eight-way top-one prediction. They therefore need not equal the errors from the separate summed-probability binary rule.

| Historical pipeline | Arrangement | Correct eight-type | Wrong benign type, still benign | Wrong malignant type, still malignant | Benign → malignant | Malignant → benign |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 812 | 274 | 115 | 127 | 82 | 2 |
| Frozen | 813 | 230 | 129 | 115 | 90 | 36 |
| Adapted | 812 | 274 | 117 | 128 | 80 | 1 |
| Adapted | 813 | 233 | 127 | 115 | 91 | 34 |
| Supervised | 812 | 222 | 107 | 177 | 75 | 19 |
| Supervised | 813 | 240 | 142 | 102 | 68 | 48 |

For frozen 812, the largest individual wrong pair is papillary → ductal, 50 of 75 papillary events. Other large pairs include phyllodes → fibroadenoma (31), fibroadenoma → ductal (30), tubular adenoma → fibroadenoma (25) and adenosis → mucinous (23). For frozen 813, tubular adenoma → fibroadenoma (41), phyllodes → fibroadenoma (35), fibroadenoma → tubular adenoma (28), papillary → mucinous (26), papillary → fibroadenoma (25) and lobular → ductal (24) dominate. These are correlated appearances of a small number of cases, not independent disease examples.

Phyllodes has zero original eight-type recall. Nevertheless, 64/75 and 54/75 phyllodes events already receive another benign subtype on 812/813; those cease to be errors under benign pooling. This is a task change. Papillary remains malignant and continues to require its own correct final label.

## Routing and malignant typing are separate failures

The existing secondary binary rule sums the four malignant probabilities and calls malignant at score ≥ 0.5, including a tie. Recomputing that exact fixed rule gives:

| Historical pipeline | Arrangement | Benign routed malignant / 300 | Malignant routed benign / 300 | Binary accuracy | Malignant sensitivity | Benign specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 812 | 74 | 4 | 87.00% | 98.67% | 75.33% |
| Frozen | 813 | 90 | 26 | 80.67% | 91.33% | 70.00% |
| Adapted | 812 | 77 | 4 | 86.50% | 98.67% | 74.33% |
| Adapted | 813 | 88 | 20 | 82.00% | 93.33% | 70.67% |
| Supervised | 812 | 75 | 17 | 84.67% | 94.33% | 75.00% |
| Supervised | 813 | 70 | 37 | 82.17% | 87.67% | 76.67% |

For the frozen reference, most binary errors are benign false positives. On 813, 25 of its 26 malignant false negatives are papillary; the remaining one is ductal. On 812, all four malignant false negatives are mucinous. The 813 papillary issue therefore involves both routing and typing.

Restricting the same saved eight-way scores to the four malignant labels, **while supplying the true malignant group**, gives 57.67%/54.00% malignant typing accuracy. Per-class recall is:

| True malignant subtype | Frozen 812 | Frozen 813 |
| --- | ---: | ---: |
| Ductal | 80.00% | 94.67% |
| Lobular | 58.67% | 58.67% |
| Mucinous | 78.67% | 48.00% |
| Papillary | 13.33% | 14.67% |

Each cell has 75 repeated events, with only one validation case per trial for that subtype. This is an oracle-group diagnostic of an old head, not the performance of a trained malignant specialist. Even perfect routing would give only 78.83%/77.00% final-five accuracy if this particular malignant readout stayed unchanged. Conversely, perfect malignant typing with the old binary decisions unchanged cannot exceed 87.00%/80.67%. Neither bound applies to newly trained heads.

## Three retrospective readouts must not be conflated

All three rules below use the identical saved eight-class probability vector:

1. **Map old top-one:** take its original eight-way winner, then rename any benign winner as benign. This removes only within-benign errors.
2. **Pool probabilities, five-way winner:** sum the four benign probabilities into one score, retain the four individual malignant scores and choose the largest of these five scores.
3. **Hard binary gate, then malignant winner:** call malignant when the sum of malignant scores is ≥ 0.5; if malignant, choose the largest malignant subtype score. This imitates the decision rule with old scores, without training either requested head.

These rules can disagree. In particular, choosing the largest five-way probability is not equivalent to a binary gate: pooled benign can exceed every individual malignant subtype even when total malignant probability exceeds 0.5. A future hard-routing implementation must save and score its actual hard final decision separately from any hierarchical probability vector.

Frozen reference, five-trial means of per-subset metrics:

| Retrospective rule | Arrangement | Five-label accuracy | Five-label balanced accuracy | Five-label macro F1 |
| --- | --- | ---: | ---: | ---: |
| Map old top-one | 812 | 64.83% | 60.13% | 0.5267 |
| Map old top-one | 813 | 59.83% | 53.73% | 0.4670 |
| Pool probabilities, five-way winner | 812 | 66.50% | 57.20% | 0.5017 |
| Pool probabilities, five-way winner | 813 | 67.00% | 56.00% | 0.4904 |
| Hard old binary gate, then malignant winner | 812 | 65.83% | 60.13% | 0.5310 |
| Hard old binary gate, then malignant winner | 813 | 61.00% | 55.60% | 0.4781 |

The top-one mapping increases raw accuracy by 115/600 = 19.17 and 129/600 = 21.50 percentage points through changed label accounting alone. The pooled-probability rule also changes decisions. None is a newly trained direct-five control. No rule or pipeline is selected as a winner from this posthoc comparison.

For the hard old-score readout, final accuracy equals binary accuracy minus the fraction of all events that are correctly routed malignant but given the wrong malignant type. Frozen 812 has 127 such errors: 87.00% − 127/600 = 65.83%. Frozen 813 has 118: 80.67% − 118/600 = 61.00%. This arithmetic explains why binary accuracy cannot stand in for the requested endpoint.

All three historical pipelines, using that same hard old-score rule:

| Historical pipeline | Arrangement | Mean-15 accuracy | Balanced accuracy | Macro F1 | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen | 812 | 65.83% | 60.13% | 0.5310 | 67.50% |
| Frozen | 813 | 61.00% | 55.60% | 0.4781 | 57.50% |
| Adapted | 812 | 65.17% | 59.67% | 0.5242 | 67.50% |
| Adapted | 813 | 61.83% | 56.53% | 0.4889 | 62.50% |
| Supervised | 812 | 54.83% | 42.73% | 0.3723 | 55.00% |
| Supervised | 813 | 64.33% | 56.93% | 0.4658 | 67.50% |

Frozen hard-readout final-label recall is benign 75.33%/70.00%, ductal 80.00%/93.33%, lobular 58.67%/58.67%, mucinous 73.33%/48.00% and papillary 13.33%/8.00%. High pooled-benign support must not hide low papillary recall.

## Complete retained-reference trial and subset summaries

Every row remains a retrospective hard readout of the old frozen scores.

| Arrangement | Trial | Mean-15 accuracy | Balanced accuracy | Macro F1 | All-four-view accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 78.33% | 80.33% | 0.7546 | 87.50% |
| 812 | 2 | 51.67% | 37.67% | 0.3821 | 50.00% |
| 812 | 3 | 56.67% | 52.67% | 0.3839 | 62.50% |
| 812 | 4 | 65.00% | 54.00% | 0.4743 | 62.50% |
| 812 | 5 | 77.50% | 76.00% | 0.6603 | 75.00% |
| 813 | 1 | 56.67% | 56.67% | 0.4933 | 50.00% |
| 813 | 2 | 75.00% | 60.00% | 0.5111 | 75.00% |
| 813 | 3 | 69.17% | 51.67% | 0.4388 | 75.00% |
| 813 | 4 | 63.33% | 58.33% | 0.5241 | 62.50% |
| 813 | 5 | 40.83% | 51.33% | 0.4229 | 25.00% |

The following includes all 15 subsets, with no view selection. Each cell is a five-trial mean.

| Available magnifications | 812 accuracy | 812 macro F1 | 813 accuracy | 813 macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 40 | 60.0% | 0.4595 | 52.5% | 0.3676 |
| 100 | 65.0% | 0.4819 | 57.5% | 0.4544 |
| 200 | 70.0% | 0.5646 | 70.0% | 0.5597 |
| 400 | 60.0% | 0.4598 | 67.5% | 0.4937 |
| 40+100 | 60.0% | 0.4662 | 57.5% | 0.4658 |
| 40+200 | 65.0% | 0.5505 | 62.5% | 0.5192 |
| 40+400 | 62.5% | 0.4965 | 57.5% | 0.4342 |
| 100+200 | 72.5% | 0.5912 | 60.0% | 0.4747 |
| 100+400 | 72.5% | 0.5855 | 62.5% | 0.4794 |
| 200+400 | 65.0% | 0.5531 | 67.5% | 0.5225 |
| 40+100+200 | 65.0% | 0.5505 | 57.5% | 0.4632 |
| 40+100+400 | 62.5% | 0.4938 | 62.5% | 0.5125 |
| 40+200+400 | 67.5% | 0.5581 | 60.0% | 0.4858 |
| 100+200+400 | 72.5% | 0.5895 | 62.5% | 0.4816 |
| 40+100+200+400 | 67.5% | 0.5648 | 57.5% | 0.4565 |

Mean trial worst-subset macro F1 is 0.4019/0.3582 for this readout. A favorable subset or trial does not replace the registered all-subset endpoint.

## Metric, label, mask and weighting audit

- Canonical old order is adenosis, ductal, fibroadenoma, lobular, mucinous, papillary, phyllodes and tubular adenoma. Benign indices are 0/2/6/7; malignant indices are 1/3/4/5. New final-label and malignant-head orders must be explicit; integer labels cannot be reused without mapping.
- Each accepted archive has exactly eight validation cases × 15 subsets, one case per original subtype, with consistent truths, finite normalized eight-way probabilities and validation-only roles. Prediction and report hashes for all 30 runs matched the accepted binary-analysis ledger. Its existing binary accuracy, balanced accuracy and F1 were reproduced exactly for all 450 run/subset cells.
- After pooling, per-subset final-label support is `[4,1,1,1,1]`. Raw accuracy is `0.5 × benign recall + 0.125 × sum of four malignant recalls`; balanced accuracy is the mean of all five recalls. Always benign gives 50% accuracy, 20% balanced accuracy and 0.1333 macro F1. The original eight-class raw accuracy and balanced accuracy happened to agree because its validation support was balanced.
- Macro F1 is calculated separately on each eight-case subset, then averaged equally over all 15 subsets and five trials. Computing one F1 from a pooled 600-event confusion matrix is generally different. Pooled event counts here diagnose mistakes; they are not substituted for the registered mean F1.
- Existing mean vectorization includes only available views and appends four availability flags. The 15-entry coverage check confirms the evaluated masks. Nothing in these saved errors identifies an absent-view leak, a missing-magnification bug or a physical field correspondence. Feature/processor causes require the separate model audit; no new encoder execution was used here.
- Old case-head fitting uses class balancing and subset sample weight 1/15, so reproducing every case 15 times does not give it 15 times the nominal fitting weight. Existing field-level diagnosis separately weights class, case within class, view and field count. Raw field accuracy is not the case endpoint. New five-class balancing, binary balancing and malignant-four balancing define different losses and must be locked explicitly; do not retain an extra eight-class inverse-weight multiplier after relabeling.
- Preserve original eight-subtype split membership before pooling. Existing split generation and several metric/head checks hardcode eight classes; the new five-label and binary paths need their own declared class counts and mappings. Binary logistic coefficients have one row for two probability columns. A serialized-head replay must respect that shape and the saved class order.

There is no discovered arithmetic or canonical-label blocker in the accepted historical scores. There are required implementation safeguards for the new experiment, especially hard-decision versus probability argmax, dynamic metric dimensions and unchanged roles.

## Interpretation and verification limits

The [earlier saved-model diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md) measured roughly 98% fitting case balanced accuracy versus 38–46% validation, with a similar field-level gap. These support a generalization problem under the provider-case protocol. The errors above localize which decisions fail; they do not prove that a particular encoder, averaging operation, label defect, stain difference or biological feature caused the failure. Pooling benign labels reduces the requested task, and a true malignant specialist is an unmeasured proposed change.

The existing arrangements cover 22/20 distinct validation provider cases, 31 in their union. Cases recur across trials, arrangements and subsets; the three pipelines evaluate the same memberships. Clinical identity relationships remain unresolved, all 82 provider cases entered development somewhere and no globally fresh confirmation cohort remains. No confidence interval or significance claim treats the repeated events as independent patients.

The accepted binary-analysis SHA256 is `7502fe88b95869e964861d36004d7a46ebd7b342ad042ca25d47626612450ae1`. This audit performed 3,150 arithmetic, support and routing-bound checks; its maximum deviation from the stored binary subset metrics was zero. The complete retrospective calculations retain all three rules, all 30 runs and every subset in a private aggregate archive. No fitting, normalization refit, threshold search, encoder forward, new image scoring, resplit, outer-role prediction or cluster job occurred.

The next scientific result must come from the bounded, registered direct-five versus trained binary-plus-malignant-four comparison described in [the confirmed scope](../DUAL_STEP_SCOPE.md), with matching cases, representation, selection opportunities and all-view reporting. Existing eight-class and binary results retain their original meanings.
