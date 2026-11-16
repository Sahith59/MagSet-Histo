Latest November16 result: **75.67%/75.00%** final two-step accuracy and macro F1.5897/.5643 for the completed Midnight similarity classifier, on the same splits and15-subset scoring. This supersedes72.00%/71.33% as the best current paired accuracy result. [Verified similarity comparison](../BREAKHIS_KERNEL_RESULTS.md). One bounded last-layer adapter comparison is being implemented;94–95% remains unachieved. Earlier execution notes below retain their original scope.

# BreakHis five-category capstone: current evidence

Updated November16,2026. We classify breast tissue as benign or one of four malignant types: ductal, lobular, mucinous or papillary carcinoma. The two-step system first predicts benign/malignant, then gives a malignant subtype. Only BreakHis images/labels are fitted/evaluated. TCGA follow-up is deferred.

**The best consistently improved two-step model now scores72.0%/71.3% accuracy**, versus65.8%/68.8% for our first Phikon baseline. Both validation arrangements improve, with identical cases, roles and15-magnification-subset scoring. The new image reader is frozen Midnight-12k, followed by fitting-only normalization, average available-view features, four view flags and balanced logistic router/expert heads.

| Two-step method | Accuracy812 /813 | Macro F1812 /813 |
| --- | ---: | ---: |
| Frozen Phikon case means |65.8% /68.8%|.4975 /.5325|
| EfficientNet case-supervised |46.5% /53.3%|.2346 /.3108|
| Frozen Phikon field supervision |66.7% /69.5%|.4975 /.5207|
| **Frozen Midnight case means** |**72.0% /71.3%**|**.5558 /.5350**|
| Frozen Midnight field supervision |73.7% /70.0%|.5732 /.5017|

All20 Midnight paired direct/dual case/field runs and the complete240-head prediction matrix were verified. Benign/malignant decisions improve to91.2%/94.2%; malignant subtype accuracy remains60.0%/54.3%. That is the main improvement target. **94–95% final five-category accuracy has not been reached.** A same-cache support-example classifier is executing, and one fixed last-layer adapter recipe is registered for further improvement.

BreakHis has7,909 images at40×/100×/200×/400×. Each run keeps related images/zooms together under a provider key:46 fitting,8 validation,28 excluded. Five overlapping trials and two arrangements reuse31 distinct validation keys. Fitting support is11 benign/25 ductal/2 lobular/5 mucinous/3 papillary. Clinical identifier relationships remain unresolved, and all82 provider keys have entered development somewhere. These are exploratory provider-case validation scores, not a fresh clinical patient test. Repeated zoom outputs do not create independent patients.

The supplied notebooks saved93.73%/94.99% for eight-type40× individual images with image-random, reused validation. That historical score uses a different task, prediction unit and split. It cannot establish that this new case-based system has95% accuracy.

[Detailed improvement results](../BREAKHIS_IMPROVEMENT_RESULTS.md) · [Five-category report](../../paper/breakhis-five-class-report.md) · [Original comparison](../BREAKHIS_FIVE_CLASS_RESULTS.md) · [CNN result](../BREAKHIS_CASEBAG_RESULTS.md) · [Historical before/now](../CAPSTONE_BEFORE_AND_NOW.md) · [Next tasks](../NEXT_STEPS.md).

Immediate deliverable: verify the remaining bounded improvement study, then revise the five-category presentation and report around all measured results. Novelty and conference claims follow stronger evidence; the capstone comes first.
