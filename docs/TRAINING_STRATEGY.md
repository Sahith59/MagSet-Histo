# How the current BreakHis system learns

The current single-model pilot uses one frozen UNI2-h image reader and a shared 32-unit neural classifier. The two decisions learn together from 49 training cases. On 17 validation cases the selected model gives 75.69% final-five accuracy. [Single-model training and results](SINGLE_MODEL_DUAL_STEP.md).

## Earlier training strategies

Updated November 17, 2026. The system first asks whether tissue is benign or malignant. A benign answer stops there; a malignant answer proceeds to ductal, lobular, mucinous or papillary carcinoma. These are **five final categories**, with no benign subtype expert. Only BreakHis supplies current fitting and evaluation images and labels; external pretrained reader weights are transfer learning. TCGA work remains deferred.

## Images, cases and available zooms

One prediction uses a provider case's tissue photographs at the available 40×, 100×, 200× and 400× magnifications. Equally numbered fields at different zooms are not assumed to show the same physical spot. We evaluate all 15 nonempty zoom subsets and keep related images, derived crops and magnifications in the same role.

Each role contains 46 fitting, eight validation and 28 excluded provider cases. Fitting support is 11 benign, 25 ductal, two lobular, five mucinous and three papillary cases. The malignant expert uses only the 35 truly malignant fitting cases. Validation has four benign cases and one of each malignant type. These roles preserve the original eight-label assignments before benign labels are pooled.

There are five overlapping trials in each arrangement, 812 and 813. They are arrangements of one dataset, not separate datasets. All 82 provider keys entered development somewhere; validation covers 31 distinct keys, with unresolved clinical identity relationships. The reused validation results are exploratory, not a fresh clinical test. More fields, masks or trials do not create more independent patients.

## The leading frozen-reader workflow

1. **Describe each photograph.** Frozen Midnight-12k produces 3,072 numbers per field: its CLS representation and mean patch representation. The original single-center reader and the completed three-crop reader use the same pretrained weights. Three-crop input averages left, cached official-center and right crop embeddings before classification; it still produces one description per original field.
2. **Center and normalize.** For each zoom, C and F subtract a raw-feature center calculated from fitting cases with equal case influence, then normalize vector length. The completed P study also tests centers balanced across each decision's classes. Validation and excluded examples supply no centering statistics.
3. **Compare fitting examples.** Same-zoom cosine similarities form class-balanced density estimates. Cases have equal influence within their head class, and fields have equal influence within a case/view. A temperature controls how strongly nearby examples dominate. Both the binary router and malignant-four specialist use uniform head-class priors.
4. **Combine probabilities.** Compose joint five-category probabilities for each field, average fields within a zoom, then average the available zooms equally. Derive the case malignant probability and conditional malignant-four probabilities from that pooled joint distribution.
5. **Route explicitly.** A malignant probability below 0.5 returns benign. At or above 0.5, return the highest-probability malignant subtype. Wrong routing and wrong subtype both count as final five-category errors.

This classifier learns its reference distributions from fitting features and labels; it optimizes no new classifier coefficients and keeps the reader frozen. Direct-five controls use the same features and roles. Four shared temperatures, 0.01/0.03/0.1/0.3, are selected by earliest maximum final macro F1 averaged over all 15 subsets. Fitting diagnostics remove the whole query case before both support construction and centering. Excluded cases receive no classifier predictions.

## Completed comparisons

All accuracy/F1 pairs below are arrangement 812 / 813, averaged over five trials and all 15 zoom subsets. They are final five-category results.

| Two-step method | Accuracy | Macro F1 | Interpretation |
| --- | ---: | ---: | --- |
| First frozen Phikon case heads | 65.83% / 68.83% | .4975 / .5325 | Historical five-category reference |
| Frozen Midnight case heads | 72.00% / 71.33% | .5558 / .5350 | Larger frozen-reader comparison |
| Single-center Midnight similarity, C | 75.67% / 75.00% | .589691 / .564292 | Retained reference |
| LoRA adapter recipe, D | 68.33% / 59.83% | .4944 / .3754 | Completed; no retained gain |
| Fixed equal-probability vote, E | 74.17% / 72.17% | .5781 / .5326 | Completed; below C |
| Three-crop Midnight similarity, F | 75.50% / 76.83% | .592510 / .579947 | Small F1 gains; mixed raw accuracy |
| Native-pixel Midnight similarity, N | 60.00% / 61.00% | .439405 / .501395 | Completed; no improvement |
| Class-balanced shared center, P | 76.17% / 75.33% | .599196 / .577788 | Completed; mixed against F |
| Head-specific centers, P | 76.33% / 75.67% | .602715 / .602512 | Retained reference; earlier F1 lead superseded on 812; screen fails |
| GenBio similarity, G | 70.50% / 75.17% | .583661 / .581348 | Better typing, weaker routing; mixed result |
| Whole-case MIL, mean | 70.17% / 71.83% | .519915 / .509386 | Strong fitting, weak validation |
| Whole-case MIL, gated attention | 71.67% / 70.50% | .545323 / .523416 | Mixed versus learned mean; below F/P |
| UNI2-h frozen similarity | 75.83% / 75.17% | .664180 / .575175 | Highest observed 812 two-step F1; paired screen fails |
| Virchow2 frozen similarity | 72.50% / 73.00% | .575767 / .538743 | Paired screen fails |
| H-optimus-0 frozen similarity | 73.50% / 71.17% | .509710 / .506235 | Paired screen fails |
| UNI2-h binary metric only | 75.67% / 75.33% | .660624 / .578434 | Mixed ablation; screen fails |
| UNI2-h malignant metric only | 76.33% / 75.50% | .672180 / .581397 | Highest observed812F1; ablation |
| UNI2-h both metrics, primary | 76.17% / 75.83% | .668624 / .587323 | Small gain; screen fails |
| UNI2-h supervised field heads | 73.83% / 73.67% | .575342 / .535412 | Strong fitting; weak transfer |

F improves macro F1 slightly over C on both arrangements, but accuracy falls 0.17 percentage points on 812 and rises 1.83 on 813. The newer UNI2-h malignant-only metric ablation has the highest measured two-step F1 on 812. Head-specific P retains the 813 F1 lead, while its 813 accuracy is below F's 76.83%. Neither P mode passes the registered screen against F, which requires at least +.02 macro F1, nondecreasing accuracy and positive recall for every final class in both arrangements. Keep C, F and both P modes visible. Head-specific P's all-four-zoom accuracy is 85% on both arrangements; this is a separate endpoint from the 15-subset average. [C evidence](BREAKHIS_KERNEL_RESULTS.md) · [F evidence](BREAKHIS_COVERAGE_RESULTS.md) · [Accepted P comparison](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json).

Head-specific P's binary decision is 94.83%/93.00% correct, while true-malignant typing is only 58.33%/62.33%. Final mucinous recall is 49.33%/36.00% and papillary recall 20.00%/45.33%, showing uneven subtype effects. Malignant typing remains the main weakness. F's corresponding typing accuracy is 57.00%/59.33%. E reaches 95% binary accuracy on 813 while final accuracy is only 72.17%; binary accuracy is not five-category accuracy. [Fixed-vote evidence](BREAKHIS_FIXED_ENSEMBLE_RESULTS.md).

The completed LoRA study trained heads for two epochs and then adapters/heads for 18, selecting complete pipelines at epochs 2/6/12/20. Eight of ten dual runs selected epoch 2, before adapter updates. Its fixed C=1 initialization already scored 68.17%/59.50%, so the selected recipe gains only 0.17/0.33 points over that starting point. The much larger gap to C cannot be attributed entirely to adapter damage. Selected fitting accuracy is 99.13% on both arrangements, indicating a large generalization gap. [Adapter evidence](BREAKHIS_LORA_RESULTS.md).

## What the completed input and centering studies learned

**Native-pixel reading, N:** the complete extraction, feature audit, ten-role classifier comparison and collector are accepted. Reading 700×448 fields without resizing retains over 97% of original pixels and uses 1,601 tokens, but scores only 60.00%/61.00% final accuracy. This fixed change of resolution, coverage and global context did not help. Its successful technical execution does not establish a benefit from native input, and the experiment does not isolate one cause of the loss. [Accepted result](BREAKHIS_NATIVE_RESULTS.md).

**Class-balanced centers, P:** all ten paired runs and the full collector are accepted. Reuse F features and compare one five-class center shared by all heads against centers specific to direct-five, binary and malignant-four decisions. Centers use fitting-only case means followed by class means; fitting diagnostics remove the query first. This is supervised preprocessing, and the original case-balanced center remains valid. Head-specific geometry gives a descriptive F1 gain, with raw-accuracy tradeoffs against F. Both declared modes and negative results remain in the scorecard. Each mode has four temperatures; direct predictions repeat exactly across modes, leaving four structurally unique direct choices and eight dual choices across the family. [Accepted comparison](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) · [Registered method](BREAKHIS_PRIOR_CENTERED_SPEC.md).

## What the completed complementary-reader and MIL studies learned

**GenBio, G:** replace Midnight with a different frozen pathology reader and its prescribed whole-image 224×224 transform. It processes normalized R/G/B as separate single-channel examples through one shared backbone; concatenating three CLS outputs gives 4,608 features, versus Midnight F's 3,072. Keep the case-centered KDE and four temperatures unchanged. This changes encoder and preprocessing together. True-malignant typing improves to 60.00%/67.67%, but binary routing falls to 82.00%/89.83%, so final dual accuracy is only 70.50%/75.17%. Its direct-five control scores 71.33%/77.83%, F1 .595998/.640603. The higher one-stage 813 result does not replace the requested two-step score or establish a uniform improvement. With G's case-conditional subtype predictions fixed, even a perfect router would reach only 80.00%/83.83%; routing alone does not recover 95%. [Complete result](BREAKHIS_GENBIO_RESULTS.md) · [Model/source search](reviews/NEXT_ROUND_MODEL_SEARCH.md).

**Whole-case MIL:** keep F features frozen and train about 400,000 readout parameters: a shared 3072→128 projection, mean or gated field attention within each view, equal available-view means and direct/dual heads. Use every field and all 15 case losses, rather than assigning a case label independently to sampled fields. Four checkpoints at epochs 10/25/50/100 provide four choices per pool/arm; choosing between pools inherits eight. All 40 trajectories and every saved candidate are accepted. Selected fitting accuracy is about 99.13%, F1 .982, while validation remains around 70–72% and F1 .51–.55. Complete case visits, finite active gradients and low losses support a generalization-gap diagnosis rather than a broad failure to fit. Rare-case memorization is plausible, but the exact cause is unproven. Attention raises papillary recall while shifting other errors and worsening proper losses against mean; neither pooler beats F/P. Mean's 96.33% binary score on 813 is not its 71.83% final accuracy. [Complete result](BREAKHIS_CASE_MIL_RESULTS.md) · [Actual learning diagnosis](reviews/CASE_MIL_ACTUAL_MODEL_DIAGNOSIS.md).

All registered jobs and collectors are complete and accepted; no project run is active. Next freeze the verified evidence and finish the **five-class** deck/demo and report. Any further malignant-specialist objective or complementary-encoder intervention needs a separate registration and matched controls. Access to UNI2-h, Virchow2 and H-optimus-0 is confirmed, and their complete fixed-reader comparison is accepted. Each reader stays frozen and supplies its registered numerical description to the same case-centered density classifier. UNI2-h improves 812 F1, but all three overall paired screens fail. [Complete approved-reader results](BREAKHIS_APPROVED_ENCODERS_RESULTS.md). No automatic router/expert hybrid, N+P combination, resplit or tuning sweep follows.

Standard transfer learning, density classification, masking, probability pooling, LoRA and supervised centering are existing methods. The capstone contribution is the implemented, audited comparison under the corrected task and its clearly reported limits. The highest descriptive F1 does not establish novelty or fresh generalization. **94–95% final accuracy remains unachieved.**

## November 17: what the two new training methods did

**Bounded feature learning:** keep UNI2-h frozen and learn one list of 1,536 positive feature scales per head. The class-balanced different-case density objective uses fitting-case means; inference uses field densities. Both heads train for 200 fixed Adam steps, with an identity penalty and inherited temperatures. Four prespecified arms isolate binary, malignant and combined effects. Twenty production metric fits and four technical fits completed, including the preserved first reporting failure. Primary both-head accuracy is 76.17% / 75.83%, F1 .668624 / .587323. The objective decreases and the validation gain is small; the paired screen fails. [Complete method/results](BREAKHIS_SPECIALIST_METRIC_RESULTS.md).

**Standard supervised UNI2-h readout:** keep the same frozen features and learn L2 logistic field heads using fitting-only field-moment standardization plus four zoom flags. Manual class/case/view/field weights keep total head weights at 46 / 46 / 35. Compare direct-five with binary-plus-malignant-four using four C values (.01 / .1 / 1 / 10), fixed L-BFGS tolerance 1e-4 and maximum 2,000 iterations. The dual heads share C; select the earliest maximum mean-subset validation macro F1. All 120 production head fits and 12 fitting-only technical fits completed. Independent coefficient, objective, gradient and prediction replay accepts every candidate. Dual validation accuracy is 73.83% / 73.67%, versus about 98.5% on fitting cases. Both reference screens fail. [Complete results](BREAKHIS_UNI_LINEAR_RESULTS.md).

These recipes were fixed before their outcomes. The linear family fills a missing conventional UNI2-h baseline and changes normalization/classifier as a complete package versus KDE. Its 96.17% binary score on 813 remains separate from 73.67% final accuracy. Neither family reaches 94–95% final accuracy. Training is complete. The completed [read-only audit](BREAKHIS_SPECIALIST_ERROR_AUDIT.md) traces errors to distinct provider cases before another intervention.
