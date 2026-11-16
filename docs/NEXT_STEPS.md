# Next tasks: finish the BreakHis capstone

Updated November 16, 2026. BreakHis remains the priority for Professor Dong's capstone deliverable. The target is **pooled benign plus four malignant types**, using every available magnification and evaluating all 15 nonempty subsets. TCGA and novel-paper development remain deferred. The working materials deadline is November 17–18; presentation around November 20 remains a planning assumption, with the exact day and rubric unconfirmed.

## Completed user-requested improvement round

Both new studies are complete and accepted: frozen GenBio with the unchanged case-centered KDE, and learned mean versus gated field attention under whole-case supervision on F features. All source/input/technical checks and complete saved-candidate collectors finished; no project run is active. Neither study establishes an improvement over the retained two-step F/P references. The next deliverable is the five-class presentation and demo. [Execution](BREAKHIS_NEXT_ROUND_STATUS.md) · [GenBio result](BREAKHIS_GENBIO_RESULTS.md) · [MIL result](BREAKHIS_CASE_MIL_RESULTS.md) · [Recipe](BREAKHIS_NEXT_ROUND_SPEC.md).

## Accepted position

All experiments and collectors are complete and accepted. The leading results remain a scorecard with tradeoffs. P's head-specific geometry has the highest measured **two-step** macro F1 on both arrangements; its 813 accuracy remains below F. Both P modes fail the registered screen against F. **94–95% final accuracy remains unachieved.**

| Two-step method | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| Single-center similarity, C | 75.67% / 75.00% | .589691 / .564292 |
| Three-crop similarity, F | 75.50% / 76.83% | .592510 / .579947 |
| Class-balanced shared center, P | 76.17% / 75.33% | .599196 / .577788 |
| Head-specific centers, P | 76.33% / 75.67% | .602715 / .602512 |
| Native-pixel similarity, N | 60.00% / 61.00% | .439405 / .501395 |
| GenBio similarity, G | 70.50% / 75.17% | .583661 / .581348 |
| Whole-case MIL, mean | 70.17% / 71.83% | .519915 / .509386 |
| Whole-case MIL, gated attention | 71.67% / 70.50% | .545323 / .523416 |

These average all 15 subsets and five trials per arrangement. P's head-specific all-four-view accuracy is 85%/85%, a separate endpoint. [C](BREAKHIS_KERNEL_RESULTS.md) · [F](BREAKHIS_COVERAGE_RESULTS.md) · [P accepted JSON](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) · [N](BREAKHIS_NATIVE_RESULTS.md).

G's one-stage direct-five control reaches 77.83% and F1 .640603 on 813, but only 71.33%/.595998 on 812. It is a distinct decision system, not the desired two-step score or a uniform improvement. Keep both arms and arrangements; do not construct a headline from different winning arms.

The LoRA comparison and fixed equal-probability vote are complete, not running. LoRA dual accuracy is 68.33%/59.83%; its fixed C=1 initialization was already 68.17%/59.50%, and eight of ten dual runs select the head-only epoch 2 checkpoint. The fixed vote scores 74.17%/72.17%, below C. Its 95% binary score on 813 is not final five-category accuracy. Preserve these negative outcomes. [LoRA](BREAKHIS_LORA_RESULTS.md) · [Fixed vote](BREAKHIS_FIXED_ENSEMBLE_RESULTS.md).

The main remaining weakness is malignant typing. Head-specific P achieves 58.33%/62.33% true-malignant subtype accuracy despite 94.83%/93.00% binary accuracy. Mucinous and papillary recall remains weak and uneven. F's typing accuracy was 57.00%/59.33%. Stronger routing alone cannot close the gap to 95% final accuracy.

G illustrates the tradeoff: its 4,608-feature separate-R/G/B CLS representation improves typing to 60.00%/67.67% but weakens routing. An ideal router with those fixed case-conditional subtype predictions would still yield only 80.00%/83.83% final accuracy. Whole-case MIL instead learns roughly 400,000 parameters from F's 3,072-feature fields, using mean or gated attention. It fits the training cases at about 99.13% accuracy/.982 F1, yet transfers poorly. Active gradients and complete visits argue against broad optimization underfitting; sparse-case overfitting is plausible without proving a unique cause. [MIL diagnosis](reviews/CASE_MIL_ACTUAL_MODEL_DIAGNOSIS.md).

## Close the evidence and prioritize the capstone

1. **Freeze the complete verified package:** preserve accepted caches, model/support metadata, complete candidate ledgers, prediction archives, source/configuration hashes and independent reviews. Keep C/F/P tradeoffs and every negative/mixed result, including G and MIL, visible. No comparison remains awaiting execution or collection.
2. **Finish the five-class report, deck and demonstration:** use the accepted scorecard and representative failure cases; distinguish final five-category, binary, malignant-typing and all-four-view endpoints. Historical eight-class presentation/demo artifacts do not constitute a completed five-class packet. Demonstrate the retained frozen workflow without converting reused validation into a fresh test claim.
3. **Rehearse and confirm delivery details:** prepare a short explanation of the task correction, controlled changes, measured gains and remaining subtype errors. Confirm the actual presentation day and rubric before treating the tentative schedule as fixed.

N's larger native input did not improve this fixed recipe; it changes resolution, context and coverage together, so its loss does not isolate a single cause. P reuses accepted F features and changes fitting-label-based centering only. Its geometry result is descriptive: the original center remains valid, and sparse class centroids may increase variance. Each named P mode has four temperatures; direct controls repeat exactly, leaving four structurally unique direct and eight dual choices across the family. Report both modes. [P registration](BREAKHIS_PRIOR_CENTERED_SPEC.md).

Any next research intervention should separately register a malignant-specialist objective or a complementary encoder with confirmed access, retaining the same controls and complete reporting. Access to gated alternatives is unconfirmed; no gated model is assumed available. [Model/source search](reviews/NEXT_ROUND_MODEL_SEARCH.md) · [Model/training diagnosis](reviews/NEXT_ROUND_MODEL_DIAGNOSIS.md).

No automatic router/expert hybrid, N+P combination, threshold search, resplit or sweep follows. The MIL result does not justify repeating the same case-supervision package unchanged. A new hypothesis is not a promised route to 95%.

## Complete the deliverable

Update the [five-category report](../paper/breakhis-five-class-report.md), [professor brief](evidence/PROFESSOR_BRIEF.md), presentation and demonstration with accepted results and their failure cases. Explain the binary-to-malignant-four decision chain, direct-five control, class supports, all-view versus 15-subset endpoints and computation costs. Retain the original eight-class comparisons as historical evidence.

The supplied notebooks' saved 93.73%/94.99% values concern eight-type, 40× image-random reused validation. They are neither reproduced results for this system nor comparable case-level five-category scores. The current workflow's standard components and evaluation safeguards do not establish a novel algorithm or clinical readiness.

Use the completed evidence to rehearse the practical question: what does the model answer, where does it fail, and what did each controlled change teach us? The next report should state an achieved number, not infer 95% from a binary diagnostic or promise it from a larger model. [Plain-language training guide](TRAINING_STRATEGY.md) · [Delivery plan](CAPSTONE_DELIVERY_PLAN.md).

## Evaluation limits that remain fixed

Each role has 46 fitting/eight validation/28 excluded provider cases. Fitting support is 11/25/2/5/3 across benign/ductal/lobular/mucinous/papillary; validation support is 4/1/1/1/1. Images, crops and all magnifications stay together, and excluded cases supply neither classifier predictions nor fitting statistics. Fitting similarity diagnostics remove the whole query case before support and centering.

Arrangements 812/813 cover 22/20 distinct validation keys, 31 in their union. All 82 provider keys entered development somewhere; clinical identity remains unresolved. Repeated images, masks and overlapping trials are dependent, and reused validation is not a fresh confirmation cohort. Final accuracy, macro F1, balanced accuracy, all-five recalls and routing/type diagnostics must be reported together.

Earlier baseline execution, adaptation, CNN, image-replay and eight-class results remain preserved in their dated result pages. Their old running-job statements do not describe current execution. No message to Professor Dong or conference submission has been sent.
