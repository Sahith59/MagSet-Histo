# Actual case-MIL outcome: fits the cases, transfers poorly

November 16, 2026. **The registered case-supervised neural readouts do not improve on F or P.** Mean-pooling dual accuracy is 70.17%/71.83%, macro F1 .519915/.509386; attention dual is 71.67%/70.50%, F1 .545323/.523416. Attention helps some class metrics but does not provide a consistent accuracy or proper-loss improvement over mean pooling. Both remain below F/P, and 94–95% final accuracy is unachieved.

I authored the trainer and collector. This is an analysis of their accepted learning outcome, **not an independent validity certification**. The separate implementation review and independently assigned actual-output review have different roles. Sources here are the [accepted complete JSON](../evidence/BREAKHIS_CASE_MIL_RESULTS.json), SHA256 `a7c00e08a9279e79311d24547cf5df06d5c7cbfe913b1a880e8c6c1e345ae2b5`, plus read-only aggregates of the 40 actual training histories. No new predictions, fits, encoder execution, selection search or source changes were made for this diagnosis.

## What was actually tested

Use the same accepted F field features, fitting-only mean/std, shared 3072→128 projection, ReLU/dropout, all fields, all 15 view subsets and direct/dual heads. Only field pooling differs between the two modes. Each mode/arm has four checkpoint opportunities, epochs 10/25/50/100. Thus mean versus attention is a matched pooling comparison; either neural mode versus F/P is a complete classifier-package comparison, including nonlinear projection and feature-level pooling in place of the previous field-probability density readout. It does not isolate a loss-only effect.

All 40 production trajectories finished in 32–47 seconds. Complete collection finished successfully in 3 minutes 6 seconds, checking 160 candidate checkpoints, 320 prediction archives and 129,600 dependent fitting/validation case/subset records. The speed and successful verification establish execution, not scientific improvement. [Registered recipe](../BREAKHIS_NEXT_ROUND_SPEC.md) · [Implementation review](CASE_MIL_IMPLEMENTATION_REVIEW.md).

## The model is not broadly failing to fit

| Dual mode | Fitting accuracy 812 / 813 | Fitting macro F1 | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Mean | 99.10% / 99.13% | .981955 / .982367 | 70.17% / 71.83% | .519915 / .509386 |
| Attention | 99.13% / 99.13% | .982367 / .982367 | 71.67% / 70.50% | .545323 / .523416 |

These selected fitting scores are in-sample neural diagnostics. They must not be ranked directly against KDE leave-case-out fitting scores, which remove the query before support/centering.

At the first available checkpoint, epoch 10, fitting macro F1 is already about .982 for every mode/arm/arrangement. Selected attention models have 100% fitting recall for lobular, mucinous and papillary cases, but validation recall is much lower. The direct arms show the same large gap: selected fitting F1 is .982367, versus validation .522538/.502834 for mean and .551384/.495257 for attention. Changing the hierarchy is therefore not the sole source of weak transfer.

Read-only history checks found all 4,000 epochs visited 46 unique fitting indices, with the same fitting membership retained throughout each trajectory. That is 184,000 case visits and 48,000 optimizer updates. Every recorded gradient value is finite. Projection and head groups have positive gradient maxima on all 48,000 updates. Attention has positive gradient maxima on all 24,000 updates in attention models; its diagnostic group is zero in mean models as expected. Each dual pool family records 11,931 malignant-supervised updates out of 12,000; the other 69 updates are benign-only batches where expert supervision is skipped. The accepted collector separately checked optimizer step counts and original role coverage.

The logged weighted objective, normalized as `4 × epoch loss_sum / 46`, falls from .907427 to .008116 for mean dual and .907311 to .007550 for attention dual between epochs 1 and 100. Direct losses fall from about 1.437 to .0123/.0117. These are training-objective diagnostics, not unweighted validation cross-entropy. They provide no evidence of missed case visits, dead attention or a general inability to optimize the registered objective.

## Checkpoint selection and later training

| Readout/arm, ten roles each | Epoch 10 | Epoch 25 | Epoch 50 | Epoch 100 |
| --- | ---: | ---: | ---: | ---: |
| Mean direct | 7 | 2 | 1 | 0 |
| Mean dual | 8 | 2 | 0 | 0 |
| Attention direct | 4 | 3 | 2 | 1 |
| Attention dual | 4 | 1 | 3 | 2 |
| All 40 trajectories | 23 | 8 | 6 | 3 |

Mean dual's fixed-epoch validation F1 falls from .519915/.502212 at epoch 10 to .483143/.495079 at epoch 100, while fitting remains near ceiling. Its validation cross-entropy rises from .921358/.940516 to 1.309634/1.317333. This supports overconfident poor generalization with additional fitting, rather than a need to train longer.

Attention is less uniform: its fixed-epoch dual F1 rises slightly from .537812/.492025 to .545175/.507386 between epochs 10 and 100, while cross-entropy worsens from .901718/1.037791 to 1.257099/1.537459. Some later decision changes improve F1 even as probability quality deteriorates. It would be inaccurate to say every additional epoch hurts every metric. Conversely, an epoch-100 choice in a few roles does not rescue the aggregate comparison.

Epoch 10 is the earliest registered checkpoint. The boundary selections do not identify an unobserved optimum before epoch 10, and no earlier-epoch search was performed. Already near-perfect fitting at epoch 10 argues against broad optimization underfitting; it does not prove what any untested early-stopping or regularization recipe would achieve.

## What attention changes

Final class recalls below include routing errors and preserve benign/DC/LC/MC/PC order.

| Mode/arrangement | Benign | Ductal | Lobular | Mucinous | Papillary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean dual 812 | 87.33% | 86.67% | 54.67% | 48.00% | 22.67% |
| Attention dual 812 | 86.33% | 90.67% | 54.67% | 46.67% | 36.00% |
| Mean dual 813 | 94.33% | 72.00% | 45.33% | 37.33% | 42.67% |
| Attention dual 813 | 91.00% | 64.00% | 44.00% | 36.00% | 56.00% |

Papillary recall rises 13.33 percentage points in both arrangements. In 812 this yields nine more correct final decisions per 600 repeated case/subset events, with +1.50 accuracy points and +.025407 F1. In 813 it is offset by ten fewer correct benign decisions, six fewer ductal and one fewer each lobular/mucinous decision; overall accuracy falls 1.33 points while F1 rises .014030. Balanced accuracy changes from 59.87%/58.33% to 62.87%/58.20%, also showing the mixed effect. These are aggregate decision counts, not additional independent patients.

Both learned poolers are much weaker on lobular cases than F/P, despite perfect lobular fitting recall. Attention's PC gain therefore is not broad rare-type recovery. Its direct arm also changes inconsistently: accuracy/F1 improve on 812, but both decline on 813 relative to direct mean pooling. Learned field weighting is not a uniformly beneficial operation here.

Attention dual's binary accuracy is 92.17%/94.67%, versus mean's 91.17%/96.33%. True-malignant typing is only 58.33%/51.00%, versus 56.00%/50.33%. The 96.33% mean-router value is not 96.33% final accuracy. Of mean's 179/169 final errors, 126/147 occur in subtype prediction after correct routing; attention has 123/145 such errors among 170/177 total. Typing remains limiting.

## Proper losses and the strongest supported diagnosis

| Dual mode | Validation cross-entropy 812 / 813 | Validation Brier loss |
| --- | ---: | ---: |
| Mean | .921358 / 1.030620 | .423482 / .432134 |
| Attention | 1.078371 / 1.201040 | .432555 / .480721 |

Attention worsens both proper losses relative to mean in both arrangements, despite its higher selected macro F1. It also has lower fitting CE/Brier than mean, reinforcing the gap between fitting confidence and held-case probability quality. Mean's Brier loss is slightly lower than F's .430733/.433371, so it would be wrong to claim every proper-loss comparison favors the density reference. Accuracy, F1 and proper losses need their separate scorecards.

The strongest direct evidence is a **large generalization gap, including rare cases**: near-perfect fitting labels and low training losses, active finite gradients, complete case coverage, and much weaker validation subtype recognition. Approximately 0.4 million fitted parameters and only two lobular/three papillary fitting cases per role make case-specific overfitting plausible. Many fields from those same cases do not add independent case diversity. The results are consistent with memorizing case-associated patterns, but no attention-map or morphology analysis establishes whether stain, tissue composition, patient-specific appearance or another factor causes the gap.

This outcome rejects the registered mean/attention neural package as an improvement over F/P. It does not prove bag-level supervision, attention or discriminative learning universally fail. In particular, the hypothesis that replacing field supervision would help is not supported by this complete package comparison; the objective, projection and classifier family changed together relative to LoRA/KDE.

Keep the accepted C/F/P references and all negative MIL rows. The complementary encoder study remains separate until its complete collector is accepted; no result from that lane is inferred here. No extra epochs, earlier checkpoints, threshold search, new split, combined encoder or source tweak follows automatically from this diagnosis. Future work needs a distinct fixed hypothesis and matched controls; none of these measurements establishes 95%, novelty, independent clinical generalization or a solution to the sparse-case limitation.
