# Capstone: received results, current results and the next test

November14,2026. The master's capstone is the immediate priority; a novel research paper comes afterward. The submission date and formal task requirements have been requested and are not yet supplied. Until clarified, retain eight tumor types, all four magnifications and the existing case roles.

## The numbers we received

| Original notebook | Saved accuracy | Correct evaluation images | Saved macro F1 |
| --- | ---: | ---: | ---: |
| EfficientNetV2B0 with non-local attention |93.7343%|374/399|0.934478|
| EfficientNetV2B0 with spatial feature gate |94.9875%|379/399|0.942221|

These are real saved outputs in the supplied notebooks, independently checked from their printed confusion matrices. They are **historical results, not results we have retrained or independently reproduced**. The original trained models, exact image memberships and probability arrays were not supplied. [Independent original-artifact review](reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md).

The notebooks use one40× image, eight subtypes and an80/20 image-random split:1,596 fitting images and399 evaluation images. They do not enforce separation between provider cases. The same evaluation partition chooses the checkpoint and changes the learning rate, then supplies the reported result. Exact historical case overlap and its effect on accuracy cannot be recovered from the supplied artifacts. The reported result is reused validation, not a fresh new-patient test.

The supplied earlier paper is a third experiment:86.22% is **binary benign/malignant accuracy**, not eight-class accuracy. Its final five-label pathway and malignant-subtype recall must not be confused with that binary headline. It is prior published work, not a new result of this project.

## Our checked numbers now

| Current eight-class case pipeline | Arrangement812 accuracy /macro F1 | Arrangement813 accuracy /macro F1 | Decision |
| --- | ---: | ---: | --- |
| Frozen Phikon-v2 mean/logistic reference |45.7% /0.3843|38.3% /0.3169|Retained reference |
| Last-block Phikon adaptation |45.7% /0.3840|38.8% /0.3205|Improvement gate failed |
| Fully supervised ResNet-50 reference |37.0% /0.3084|40.0% /0.3006|Improvement gate failed |

These are five-trial means across all15 available-view settings. Each model fits46 provider cases, validates on eight cases (one per subtype), and excludes28 outer-role cases. Cases recur across trials; the two arrangements contain22/20 distinct validation cases and31 in their union. They are exploratory provider-case validation results, not independently reverified clinical-patient test performance.

**We did not measure a fall from95% to40% on the same task.** The project expanded from a40× image classifier into a case-level four-view/missing-view study. The architecture, preprocessing, fitting objective, selection, class weighting, prediction unit and splitting protocol also changed. We cannot isolate the causal contribution of those changes from these headline scores.

## Why the current scores are low

The strongest measured finding is a fitting–validation gap: ResNet answers fitting cases with99.6–99.7% balanced accuracy but validates at37–40%. Its image classifier is weak on validation too, so the case aggregation alone is not an established explanation.

Each fitting set has **one adenosis case and one phyllodes case**. Hundreds of images and augmentations from one case still represent one independent example. Phyllodes validation recall remains zero. Validation gives each subtype equal importance, whereas the original399-image distribution gives common subtypes more influence. Eight validation cases also make model selection fragile.

Small independent case coverage, dataset/feature transfer, image information lost during processing and objective alignment are plausible contributors. A bug in the checked metric arithmetic, label order or role filtering has not been found. Audits establish implementation consistency; they do not prove optimal training, one sole cause, reliable clinical performance or a fundamental performance ceiling.

## What we are doing for the capstone

1. Preserve the original saved results with their exact meaning and show our changed protocol separately. No original figure is relabeled as a new result.
2. Run a [fixed saved-prediction bridge](CAPSTONE_BRIDGE_SPEC.md): report individual-image scores at all four zooms, including40×, and compare40× image versus case aggregation using the same accepted weights and images. This requires no new fitting or model execution. It is endpoint alignment, not an exact EfficientNet reproduction.
3. Test one [registered local-plus-global representation](BREAKHIS_MULTICROP_SPEC.md). Keep the pathology image reader frozen, keep existing roles, and compare its original central view with a fixed blend of that view and four regional crops. Both arms use one fixed classifier regularization. This is a conventional preprocessing experiment; improvement is unproven.
4. Finish the capstone report and working prediction evidence regardless of that test's outcome. Retain every result. A corrected EfficientNet reproduction is a separately scoped option if the formal capstone requires it; it is not an automatic additional model search.

The three independent GPT-6.1 Sol reviews cover [original evidence](reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md), [protocol](reviews/CAPSTONE_PROTOCOL_REVIEW.md) and [modelling](reviews/CAPSTONE_MODEL_REVIEW.md). Capstone acceptance depends on the university's requirements, which we do not yet have. We can seek stronger measured performance; we cannot promise95%, change the target to manufacture it, or present training accuracy as held-out accuracy.

The research-paper route stays separate: novelty, stronger baselines and credible independent evaluation need evidence beyond a high capstone score.
