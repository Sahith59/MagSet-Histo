# Actual case-MIL validity review

Reviewed November 16, 2026. The accepted [case-MIL artifact](../evidence/BREAKHIS_CASE_MIL_RESULTS.json) has SHA256 `a7c00e08a9279e79311d24547cf5df06d5c7cbfe913b1a880e8c6c1e345ae2b5`, producer revision `6861fbfe8d847f16fa09b4f839193716c1e8d501`. This review inspected accepted summaries/candidates and current source, without new fitting, prediction generation, threshold changes or jobs. GenBio results were not inspected while its collector remained unaccepted.

**Conclusion:** attention supplies a limited paired macro-F1 improvement over learned mean pooling, but neither case-supervised recipe improves on the specified frozen F reference. No numeric or source mismatch was found that explains away the result. The large in-sample fitting/validation gap and weak malignant typing are measured limitations; their biological or data-level cause is not established.

## What completed and what the comparison isolates

All 40 trajectories completed 100 epochs, with 160 fixed candidates, 160 checkpoint archives, 320 prediction archives and 129,600 verified case/subset records. Current `case_mil.py`, `case_mil_five.py` and `case_mil_collect.py` hashes match the accepted artifact. Independent checks of all 40 earliest-max selections, selected fitting/validation copies, 16 aggregate summaries, confusion-derived supports/accuracy/recall and 20 exact shared-initialization pairs passed: **268 checks**. These supplement the accepted collector; they do not independently reconstruct all GPU training updates.

The implemented objective matches the registration: whole-case supervision, all 15 subsets averaged equally, exact class/case weights and fixed batch denominator 4, with true-malignant-only expert loss. The absence of expert gradients on benign-only batches was independently checked before execution. No raw encoder is trained or executed. Models have 394,009 parameters for mean and 402,297 for attention, so the latter adds 8,288 attention parameters to the otherwise paired design.

Mean-versus-attention is the pooling ablation. Relative to KDE, both also change standardization, projection, objective and readout. These results do not uniquely establish whether inherited field labels, learned pooling, classifier capacity or another factor caused earlier failures.

## Paired validation result

Values average five trials and all 15 subsets per arrangement:

| Arrangement | Pool | Direct accuracy / macro F1 | Dual accuracy / macro F1 | Dual all-four accuracy |
|---|---|---:|---:|---:|
| 812 | mean | 71.17% / .522538 | 70.17% / .519915 | 70.00% |
| 812 | attention | 73.00% / .551384 | 71.67% / .545323 | 72.50% |
| 813 | mean | 71.50% / .502834 | 71.83% / .509386 | 72.50% |
| 813 | attention | 70.83% / .495257 | 70.50% / .523416 | 70.00% |

Attention increases dual macro F1 by .025407/.014030 over mean, but its accuracy changes are +1.50/−1.33 percentage points. Direct attention also loses accuracy/F1 in 813. Thus the paired result is mixed, not uniform superiority. Against primary F, attention-dual loses 3.83/6.33 accuracy points and .047187/.056531 macro F1. It also remains below head-specific P in both accuracy and macro F1. Both registered improvement screens are false; all-four scores do not rescue that conclusion.

Selection favors relatively early checkpoints: across 40 trajectories, epochs 10/25/50/100 are selected 23/8/6/3 times. Every mean 812 direct and dual trial selects 10. This shows that later registered checkpoints did not improve the selection criterion for those trials; it does not authorize searching earlier epochs or another learning rate afterward. There are four checkpoint opportunities per named pool/arm, and eight if choosing between pools.

## Generalization and class tradeoffs

Selected fitting accuracy is 99.10–99.13% across dual models, with macro F1 approximately .982. Dual validation accuracy is 70.17–71.83%, with F1 .509386–.545323. All rare-class fitting recalls are essentially perfect, while validation recalls remain weak. These are **in-sample neural fitting predictions**, unlike KDE's leave-query-case-out diagnostics. Their high level establishes capacity to fit the assigned cases, not reliable unseen-case discrimination.

Attention-dual validation recalls in benign/DC/LC/MC/PC order are `[86.33,90.67,54.67,46.67,36.00]%` for 812 and `[91.00,64.00,44.00,36.00,56.00]%` for 813. Attention improves PC recall over mean by 13.33 points in both arrangements, while LC remains 54.67%/44%, versus F's 92%/96%. In attention 813, 42 of 75 repeated LC records and 43 of 75 MC records become DC; 26 of 75 DC records become MC. The PC benefit coexists with other errors rather than removing the malignant bottleneck.

Mean 813 binary accuracy is **96.33%**, but true-malignant typing is **50.33%** and final accuracy **71.83%**. Attention routing is 92.17%/94.67%, with typing 58.33%/51%. With the mean 813 conditional predictions fixed, even perfect routing would yield only 75.17% final accuracy under the four-benign/four-malignant validation supports. No binary submetric establishes 95% final five-category accuracy.

The aggregate fitting residual includes the same 30 DC-to-LC repeated records in each selected dual recipe. This is a recurring observed confusion, not evidence sufficient to relabel cases or diagnose its cause. All quoted confusion counts repeat provider cases across roles/subsets; they are not additional independent patients.

## Probability quality and decision

Attention worsens dual cross-entropy relative to mean by .157013/.170420 and Brier score by .009073/.048587, despite the higher macro F1. Its true-malignant cross-entropy also rises. These proper losses assess the full probabilities; the result cannot be reduced to whether a hard routing threshold was favorable. Their change alone does not distinguish calibration from discrimination.

Avoid claiming that every metric deteriorates: learned mean Brier is .423482/.432134, slightly better than F's .430733/.433371. Nevertheless, mean's final accuracy/F1 are lower and its 813 cross-entropy is worse. The registered scorecard must remain intact rather than choosing whichever metric favors a recipe.

Retain F/P and the earlier accepted baselines alongside this negative case-supervision study. The same 46 fitting cases, LC 2/PC 3 scarcity, 31 reused validation provider keys and unresolved clinical identity remain. These results support reporting a limited attention tradeoff and a substantial fitting/validation gap, without a novelty claim, a new validation search or a 94–95% promise.
