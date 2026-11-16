# BreakHis five-category capstone: current evidence

Updated November 16, 2026. **UNI2-h now has the highest observed arrangement-812 two-step macro F1, 0.6642, at 75.83% accuracy**, but scores 0.5752 / 75.17% on 813. Head-specific centering, P, retains the 813 two-step F1 lead at 0.6025 / 75.67%; three-crop Midnight, F, retains higher 813 two-step raw accuracy at 76.83%. GenBio's one-stage 813 result is 77.83% / F1 .6406. These are uneven benefits across endpoints, without a uniform winner. **94–95% final five-category accuracy remains unachieved.**

The newly approved UNI2-h, Virchow2 and H-optimus-0 comparisons are complete and independently reviewed, alongside GenBio and whole-case mean/attention. UNI2-h passes the descriptive improvement screen on 812 but not 813; none of the three new readers passes the complete two-arrangement screen. All classification runs and collectors in these studies have finished. [Approved-reader comparison](../BREAKHIS_APPROVED_ENCODERS_RESULTS.md).

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

The latest one-stage controls predict the same five labels directly:

| Direct-five control | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| GenBio frozen similarity | 71.33% / 77.83% | 0.5960 / 0.6406 |
| Whole-case learned mean | 71.17% / 71.50% | 0.5225 / 0.5028 |
| Whole-case gated attention | 73.00% / 70.83% | 0.5514 / 0.4953 |
| UNI2-h frozen similarity | 75.17% / 77.67% | 0.6486 / 0.6345 |
| Virchow2 frozen similarity | 72.83% / 75.00% | 0.5901 / 0.5877 |
| H-optimus-0 frozen similarity | 77.33% / 75.17% | 0.6036 / 0.6067 |

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

Next finish the five-category presentation/demo and explain the specialist's generalization limits and UNI2-h's uneven benefit. Access to the three named readers is confirmed, and all three comparisons are complete. No automatic reader fusion or additional setting search follows from these results.

[Approved-reader results](../BREAKHIS_APPROVED_ENCODERS_RESULTS.md) · [Independent result review](../reviews/APPROVED_ENCODER_ACTUAL_RESULTS_REVIEW.md) · [GenBio results](../BREAKHIS_GENBIO_RESULTS.md) · [Whole-case mean/attention results](../BREAKHIS_CASE_MIL_RESULTS.md) · [Centering results](../BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Native-input results](../BREAKHIS_NATIVE_RESULTS.md) · [Three-crop results](../BREAKHIS_COVERAGE_RESULTS.md) · [Single-center similarity](../BREAKHIS_KERNEL_RESULTS.md) · [Completed LoRA](../BREAKHIS_LORA_RESULTS.md) · [Fixed ensemble](../BREAKHIS_FIXED_ENSEMBLE_RESULTS.md) · [Project report](../../paper/breakhis-five-class-report.md).
