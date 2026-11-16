# BreakHis five-category capstone: current evidence

Updated November16,2026. We classify a provider case as benign or one of four malignant types: ductal, lobular, mucinous or papillary carcinoma. The two-step system first decides benign/malignant, then gives a malignant subtype when needed. Only BreakHis images/labels are fitted/evaluated; TCGA follow-up is deferred.

**The best current paired result is75.67%/75.00% final accuracy**, compared with65.83%/68.83% for our first Phikon baseline. Both arrangements improve in accuracy and macro F1 under the same cases, splits and15-view scoring. **94–95% final accuracy remains unachieved.**

| Two-step method | Accuracy812 /813 | Macro F1812 /813 |
| --- | ---: | ---: |
| Frozen Phikon case means |65.83% /68.83%|.4975 /.5325|
| EfficientNet case-supervised |46.50% /53.33%|.2346 /.3108|
| Frozen Phikon field supervision |66.67% /69.50%|.4975 /.5207|
| Frozen Midnight case means |72.00% /71.33%|.5558 /.5350|
| Frozen Midnight field supervision |73.67% /70.00%|.5732 /.5017|
| **Frozen Midnight similarity two-step** |**75.67% /75.00%**|**.5897 /.5643**|

The current system uses frozen Midnight-12k descriptions, fitting-only per-view centering/L2 normalization and class/case-balanced similarity densities. Joint field probabilities are averaged within zoom and across available zooms, then explicitly routed at malignant probability.5. All ten paired runs,80 candidates and160 prediction archives were verified. The all-four-view accuracies are82.5%/80.0%, a separate endpoint from the15-subset average above.

The current actual router scores92.0%/92.7%; true-malignant typing remains56.7%/56.7%. Benign and lobular recognition improve, while ductal/mucinous/papillary errors remain substantial. One fixed standard-LoRA last-layer encoder adjustment is registered and undergoing actual GPU technical checks. This is an improvement candidate, without a95% guarantee or novelty claim.

BreakHis has7,909 images at40×/100×/200×/400×. Each run keeps related images/zooms together under a provider key:46 fitting,8 validation,28 excluded. Five overlapping trials and two arrangements reuse31 distinct validation keys. Fitting support is11 benign/25 ductal/2 lobular/5 mucinous/3 papillary. Clinical identifier relationships remain unresolved, and all82 provider keys have entered development somewhere. These are exploratory provider-case validation scores, not a fresh clinical patient test. Repeated zoom outputs do not create independent patients.

The supplied notebooks saved93.73%/94.99% for eight-type40× individual images with image-random, reused validation. That historical score uses a different task, prediction unit and split. It cannot establish that this new case-based system has95% accuracy.

[Latest similarity results](../BREAKHIS_KERNEL_RESULTS.md) · [Earlier improvement studies](../BREAKHIS_IMPROVEMENT_RESULTS.md) · [Five-category report](../../paper/breakhis-five-class-report.md) · [Training explained](../TRAINING_STRATEGY.md) · [Adapter recipe](../BREAKHIS_LORA_SPEC.md) · [Historical before/now](../CAPSTONE_BEFORE_AND_NOW.md).

Immediate deliverable: verify the remaining bounded improvement study, then finish the five-category presentation and report using all measured outcomes. Novelty and conference claims follow stronger evidence; the capstone comes first.
