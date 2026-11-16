# BreakHis five-category capstone: current evidence

Updated November 16, 2026. The completed **head-specific centering model has the highest macro F1 on both arrangements: 0.6027 / 0.6025**, with **76.33% / 75.67% final five-category validation accuracy**. The earlier three-crop model retains higher arrangement-813 accuracy at 76.83% (75.50% on 812). These are tradeoffs, not an unqualified accuracy winner. **94–95% final accuracy remains unachieved.**

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

These headline scores average five overlapping trials and all **15 nonempty magnification subsets**. With all four views present, head-specific centering scores **85.00% / 85.00%**, versus F's 82.50% / 82.50%; these are separate endpoints. Its binary router scores 94.83% / 93.00%, while true-malignant typing scores 58.33% / 62.33%. Mucinous and papillary recognition remains weak. The ensemble's 95.00% binary score on 813 is also separate from its 72.17% final five-category score.

The LoRA study is complete. Its fixed C=1 initialization already scored 68.17% / 59.50%, so the selected recipe adds only 0.17 / 0.33 percentage points. Eight of ten two-step runs select epoch 2, before adapter training begins. The result does not support blaming the entire gap from the stronger frozen models on adapter damage.

BreakHis has 7,909 images at 40×/100×/200×/400× and 82 qualified provider keys. Each run keeps related images and views together: 46 fitting, eight validation and 28 excluded cases. Fitting support is 11 benign, 25 ductal, two lobular, five mucinous and three papillary. Trials and view subsets reuse 31 distinct validation keys. Clinical identifier relationships remain unresolved, and all 82 keys have entered development. These are exploratory provider-case validation results, without a globally fresh clinical test or a novelty claim.

Native global-input classification is complete and scored below the three-crop reference. Both centering modes are also complete. Head-specific centering gives the strongest F1 result, but neither mode meets the full descriptive improvement screen. Their direct controls are identical across modes; choosing between modes introduces eight structural two-step settings, with four distinct direct settings. No jobs remain pending.

Head-specific centering also has worse cross-entropy and Brier probability errors than F on both arrangements. Its F1 lead does not establish uniformly better predictions or confidence estimates.

The supplied notebooks' 93.73% / 94.99% concerned historical eight-type, single-40× images with image-random, reused validation. They do not establish 95% accuracy for this five-category case task.

[Centering results](../BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Native-input results](../BREAKHIS_NATIVE_RESULTS.md) · [Three-crop results](../BREAKHIS_COVERAGE_RESULTS.md) · [Single-center similarity](../BREAKHIS_KERNEL_RESULTS.md) · [Completed LoRA](../BREAKHIS_LORA_RESULTS.md) · [Fixed ensemble](../BREAKHIS_FIXED_ENSEMBLE_RESULTS.md) · [Project report](../../paper/breakhis-five-class-report.md).
