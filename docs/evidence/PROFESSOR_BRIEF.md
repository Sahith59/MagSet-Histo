# BreakHis five-category capstone: current evidence

## Current capstone result

One model makes both decisions on BreakHis. Of 255 answers across 17 validation cases and 15 zoom combinations, our shared UNI2-h network got 229 benign/malignant decisions right (89.80%) and 193 complete five-category answers right (75.69%, macro F1 0.4982). Swin scored 59.61% and CNN–Transformer 59.22% on the same validation exercise. The new network called 25 of 30 lobular views of two cases ductal, and correctly named the single papillary case under zero of 15 zoom conditions. There were only two lobular and four papillary cases in training. The greater-than-90% objective remains open. We verified every selected validation answer from the saved weights. [Evidence and matched comparison](../SINGLE_MODEL_DUAL_STEP.md).

## Historical experiments

The subsequent deep audit reproduced selected-checkpoint inference and reported metrics for the original three models from raw images, including every Midnight fitting-support field. Neural retraining was not independently repeated. A frozen UNI2-h malignant specialist gives **82.92% final-five accuracy on the already-viewed follow-up**, with the binary score fixed at 94.58%. Its validation accuracy is 77.25% versus 79.61% for the matched P control, so its predefined progression screen fails. This is a mixed development result, not a confirmed upgrade or a fresh test. The bounded case-level SVC comparison is now complete: linear gives 74.58% final-five accuracy / .513007 F1 and RBF 69.17% / .380729, with 92.08% binary accuracy for both on the same reused follow-up. The RBF validation progression screen also fails. Both requested follow-up accuracy targets remain unmet. [Complete SVC outcomes](../BREAKHIS_UNI_CASE_SVM_RESULTS.md). [Proof and diagnosis](../DEEP_DIAGNOSIS_AND_PROOF.md).

The latest matched comparison, completed November 17, is **independently accepted**. On the same 16 test provider cases, 13 conservative groups and all 15 zoom subsets, final-five accuracy / macro F1 is **67.50% / .375675 for Midnight P**, **56.67% / .348180 for CNN–Transformer**, and **59.17% / .304947 for Swin**. P’s first-step benign/malignant accuracy is **94.58%**, separate from its final-five score. P leads the mean-15 point estimates; CNN leads all-four-view macro F1 and Swin leads worst-subset performance. Every paired uncertainty interval includes zero. The rare malignant types remain weak, and 94–95% final accuracy and clinical superiority are unestablished. [Complete matched results and evidence](../MATCHED_THREE_WAY_RESULTS.md).

The shared roles contain 49 training, 17 validation and 16 test cases. All cases appeared in earlier project development, so this is not a globally fresh clinical cohort. The accepted independent checker verifies all three selected models and saved prediction arithmetic. Both original-hardware A40 replay errors are zero; the failed A30 attempt and all earlier outcomes are preserved. This new holdout is separate from the historical 46/8/28 trial arrangements below.

## Earlier paired development studies


Updated November 17, 2026. The new metric primary gives **76.17% / 75.83% final accuracy**, macro F1 **.668624 / .587323**, with a small gain over its exact UNI2-h control. The malignant-only ablation has the highest observed 812 two-step F1, **.672180**; retained P leads 813 at **.602512** and scores **76.33% / 75.67%** accuracy. F retains higher 813 raw accuracy at **76.83%**. Standard supervised UNI2-h field heads score **73.83% / 73.67%**, a completed negative result. Both new complete studies fail their declared larger-improvement screens. **94–95% final five-category accuracy remains unachieved.**

Both metric-learning and standard-linear production arrays, fitting-only technical checks and complete prediction collectors have finished. The metric study verifies 40 pipelines, 20 production metric fits and 32,400 dependent records; the linear study verifies 80 pipelines, 120 production head fits and 64,800 records. These counts are repeated computation, not independent cases. [Metric results](../BREAKHIS_SPECIALIST_METRIC_RESULTS.md) · [Linear results](../BREAKHIS_UNI_LINEAR_RESULTS.md).

We predict pooled benign or ductal, lobular, mucinous or papillary carcinoma. The two-step system first decides benign/malignant; only malignant predictions receive a subtype. Only BreakHis images and labels are fitted/evaluated. TCGA follow-up remains deferred.

| Completed two-step method | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| Original frozen Phikon case means | 65.83% / 68.83% | 0.4975 / 0.5325 |
| EfficientNet case supervision | 46.50% / 53.33% | 0.2346 / 0.3108 |
| Frozen Phikon field supervision | 66.67% / 69.50% | 0.4975 / 0.5207 |
| Frozen Midnight case means | 72.00% / 71.33% | 0.5558 / 0.5350 |
| Frozen Midnight field supervision | 73.67% / 70.00% | 0.5732 / 0.5017 |
| C: frozen Midnight single-center similarity | 75.67% / 75.00% | 0.5897 / 0.5643 |
| D: Midnight LoRA recipe | 68.33% / 59.83% | 0.4944 / 0.3754 |
| E: fixed equal-probability ensemble | 74.17% / 72.17% | 0.5781 / 0.5326 |
| F: frozen Midnight three-crop similarity | 75.50% / 76.83% | 0.5925 / 0.5799 |
| N: frozen Midnight native global input | 60.00% / 61.00% | 0.4394 / 0.5014 |
| P: class-shared balanced centering | 76.17% / 75.33% | 0.5992 / 0.5778 |
| P: head-specific centering | 76.33% / 75.67% | 0.6027 / 0.6025 |
| GenBio frozen similarity | 70.50% / 75.17% | 0.5837 / 0.5813 |
| Whole-case learned mean | 70.17% / 71.83% | 0.5199 / 0.5094 |
| Whole-case gated attention | 71.67% / 70.50% | 0.5453 / 0.5234 |
| UNI2-h frozen similarity | 75.83% / 75.17% | 0.6642 / 0.5752 |
| Virchow2 frozen similarity | 72.50% / 73.00% | 0.5758 / 0.5387 |
| H-optimus-0 frozen similarity | 73.50% / 71.17% | 0.5097 / 0.5062 |
| UNI2-h binary metric only | 75.67% / 75.33% | 0.6606 / 0.5784 |
| UNI2-h malignant metric only | 76.33% / 75.50% | 0.6722 / 0.5814 |
| UNI2-h both metrics, primary | 76.17% / 75.83% | 0.6686 / 0.5873 |
| UNI2-h supervised field heads | 73.83% / 73.67% | 0.5753 / 0.5354 |

The latest one-stage controls predict the same five labels directly:

| Direct-five control | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| GenBio frozen similarity | 71.33% / 77.83% | 0.5960 / 0.6406 |
| Whole-case learned mean | 71.17% / 71.50% | 0.5225 / 0.5028 |
| Whole-case gated attention | 73.00% / 70.83% | 0.5514 / 0.4953 |
| UNI2-h frozen similarity | 75.17% / 77.67% | 0.6486 / 0.6345 |
| Virchow2 frozen similarity | 72.83% / 75.00% | 0.5901 / 0.5877 |
| H-optimus-0 frozen similarity | 77.33% / 75.17% | 0.6036 / 0.6067 |
| UNI2-h supervised field heads | 73.00% / 73.50% | 0.5666 / 0.5284 |

These headline scores average five overlapping trials and all **15 nonempty magnification subsets**. With all four views present, head-specific centering retains the highest observed two-step accuracy, **85.00% / 85.00%**, versus F's 82.50% / 82.50%; these are separate endpoints. H-optimus-0's 87.50% / 82.50% full-view result uses the one-stage control. P's binary router scores 94.83% / 93.00%, while true-malignant typing scores 58.33% / 62.33%. The ensemble's 95.00% binary score on 813 is also separate from its 72.17% final five-category score.

UNI2-h changes the 812 tradeoff: malignant typing improves to 72.00% and mucinous recall to 77.33%, while router accuracy falls to 83.33% and benign recall to 84.67%. Its 813 typing is 60.67%, with papillary recall 28.00%. Virchow2 improves both probability losses versus C in both arrangements despite lower accuracy/F1. UNI2-h's losses worsen on 812 and improve on 813; H-optimus-0 shows the opposite pattern. Every new direct-five arm has worse probability losses than its own two-step arm, even where hard-label metrics improve. No universally better reader or confidence estimate is established.

The LoRA study is complete. Its fixed C=1 initialization already scored 68.17% / 59.50%, so the selected recipe adds only 0.17 / 0.33 percentage points. Eight of ten two-step runs select epoch 2, before adapter training begins. The result does not support blaming the entire gap from the stronger frozen models on adapter damage.

GenBio improves true-malignant typing to 60.00% / 67.67%, but its router falls to 82.00% / 89.83%. Its probability losses are mixed, improving on the single-center reference in 813 but worsening in 812. The one-stage 813 gain cannot be relabeled as a two-step gain. Attention improves macro F1 over its matched learned-mean control by .0254 / .0140, with mixed accuracy, but both remain below F and P. Learned mean slightly improves Brier probability error over F while losing accuracy/F1; attention worsens both probability losses relative to learned mean.

BreakHis has 7,909 images at 40×/100×/200×/400× and 82 qualified provider keys. Each run keeps related images and views together: 46 fitting, eight validation and 28 excluded cases. Fitting support is 11 benign, 25 ductal, two lobular, five mucinous and three papillary. Trials and view subsets reuse 31 distinct validation keys. Clinical identifier relationships remain unresolved, and all 82 keys have entered development. These are exploratory provider-case validation results, without a globally fresh clinical test or a novelty claim.

Native global-input classification scored below the three-crop reference. UNI2-h leads 812 two-step F1 and P leads 813; neither supplies a new consistent winner. Neither centering mode meets its full descriptive screen. Their direct controls are identical across modes; choosing between modes introduces eight structural two-step settings, with four distinct direct settings. GenBio uses four temperatures per arm. Whole-case learning uses four checkpoints per pool/arm, or eight choices if selecting between pooling modes. Selecting across the three approved readers supplies 12 reader/temperature choices per arm/role. All failed screens and controls remain visible.

Source CI passed all 370 neural tests. GenBio's ten paired runs verify 80 candidates and 64,800 dependent records. The 40 whole-case GPU trajectories verify 160 checkpoints and 129,600 dependent records; their approximately 99% in-sample fitting accuracy does not establish validation generalization. No new encoder passes were used for whole-case learning. These counts are not independent patients.

The approved-reader feature source passed all 390 neural tests. Every reader completed its original 7,909 fields and four-image fitting-only consistency/reload checks; all differences were zero under canonical one-image calls. The three complete collectors cover 240 candidates, 480 archives and 194,400 dependent records. Independent review checked saved arithmetic/provenance and 60 selections; raw-image and feature-array re-execution were outside that review. Headline denominators are 600 repeated validation decisions per arrangement and 40 for all-four-view, reusing 31 validation keys.

Head-specific centering also has worse cross-entropy and Brier probability errors than F on both arrangements. Its arrangement-813 two-step F1 lead does not establish uniformly better predictions or confidence estimates.

The supplied notebooks' 93.73% / 94.99% concerned historical eight-type, single-40× images with image-random, reused validation. They do not establish 95% accuracy for this five-category case task.

Next finish the five-category presentation/demo and explain the specialist's generalization limits and the complete metric/linear outcomes. Access to the three named readers is confirmed, and all three comparisons are complete. No automatic reader fusion or additional setting search follows from these results.

[Approved-reader results](../BREAKHIS_APPROVED_ENCODERS_RESULTS.md) · [Independent result review](../reviews/APPROVED_ENCODER_ACTUAL_RESULTS_REVIEW.md) · [GenBio results](../BREAKHIS_GENBIO_RESULTS.md) · [Whole-case mean/attention results](../BREAKHIS_CASE_MIL_RESULTS.md) · [Centering results](../BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Native-input results](../BREAKHIS_NATIVE_RESULTS.md) · [Three-crop results](../BREAKHIS_COVERAGE_RESULTS.md) · [Single-center similarity](../BREAKHIS_KERNEL_RESULTS.md) · [Completed LoRA](../BREAKHIS_LORA_RESULTS.md) · [Fixed ensemble](../BREAKHIS_FIXED_ENSEMBLE_RESULTS.md) · [Project report](../../paper/breakhis-five-class-report.md).

## Latest specialist diagnosis

The metric study gives small descriptive gains: primary both-head learning adds two/four correct dependent checks per 600, with both F1 gains below +.02. Its learned-arm fitting diagnostics remain in-sample even with whole-query support exclusion. All four arms are reported and none passes the paired screen.

Supervised UNI2-h field heads fit 98.58% / 98.43% but validate 73.83% / 73.67%. The router is 91.83% / 96.17% correct, while true-malignant typing is 63.67% / 54.67%, weaker than the same-reader similarity control's 72.00% / 60.67%. Final lobular recall is 54.67% in both arrangements and papillary recall 21.33% / 14.67%. Correct convergence and strong fitting leave a substantial transfer gap. There are only two lobular and three papillary fitting cases; fields are correlated, and validation has an adaptive reuse history. None is proven the sole cause.

The metric family was registered first. The separate four-C standard linear family was chosen before metric outcomes and registered before fitting. All failed outcomes and the direct-linear control remain visible. No new encoder calls, excluded-role predictions, threshold changes, split changes or automatic combinations occurred. No 95% final score, clinical validation or novelty is established.

The descriptive case-error audit is complete: five pipelines, 6,000 saved events and 31 reused provider keys. The metric primary corrects eight events and regresses on two, with 288 final errors partitioned into 156 routing and 132 post-route subtype errors across both arrangements. Papillary accounts for 106/150 errors, including 75 routed to benign. Original-field paths and fixed inspection examples remain private and uninspected. The audit locates cases for appropriate review; it adds no fitting, predictions, independent cohort or morphological diagnosis.

The complete linear source passed all **415** pinned neural tests in CI, without skips, and the lightweight suite also passed. These checks and actual fitted-model replay support execution correctness; the registered predictive screens still fail.
