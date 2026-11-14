# Capstone: received results and checked current results

November14,2026. The master's capstone is the immediate priority; a novel research paper comes afterward. The user needs the package in3–4days, by November17–18, with a presentation around November20. Exact day and formal task/rubric are unspecified. Retain eight tumor types, all four magnifications and existing case roles. [Delivery plan](CAPSTONE_DELIVERY_PLAN.md).

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

## Separate benign/malignant result

Using the same saved eight-class outputs, the retained model answers the broader question with **87.0%/80.7% accuracy**, binary macro F1.8666/.7877, malignant sensitivity98.7%/91.3% and benign specificity75.3%/70.0%. These average all15 view settings and five trials. The broad group is often correct while subtypes within it are confused. This is a retrospective secondary analysis of reused validation cases, not a newly trained binary model, fresh test or improvement of eight-class accuracy. [All pipelines, trials and checks](CAPSTONE_BINARY_RESULTS.md).

## Completed work and next steps

1. Preserve the original saved results with their exact meaning and show our changed protocol separately.
2. Completed the [fixed image-endpoint bridge](CAPSTONE_BRIDGE_RESULTS.md), independently checked on all ten runs. ResNet40× raw-image accuracy is27.13%/29.94%; same-view case accuracy is37.5%/35.0%. No new fitting or model execution. This aligns endpoints; it does not reproduce EfficientNet.
3. Completed the [registered regional-view comparison](BREAKHIS_MULTICROP_RESULTS.md). All four feature shards and20 heads passed independent checks. F1 changes against the matched fixed-C control are−.0058/+.0059, with zero phyllodes recall. The gate failed. Keep the historical frozen reference; no follow-on sweep.
4. Completed the [binary secondary analysis](CAPSTONE_BINARY_RESULTS.md) and a fresh121-image saved-model replay in39seconds. All15 view combinations matched archived outputs within6.05e-08. The [interactive viewer](evidence/CAPSTONE_PREDICTION_REPLAY.html) displays saved outputs rather than running live image uploads.
5. Freeze the model and finish the report, presentation and rehearsal by November17–18. Retain every result. A corrected EfficientNet reproduction is separately scoped if the formal capstone requires it; no automatic model search follows.

The three independent GPT-6.1 Sol reviews cover [original evidence](reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md), [protocol](reviews/CAPSTONE_PROTOCOL_REVIEW.md) and [modelling](reviews/CAPSTONE_MODEL_REVIEW.md). Capstone acceptance depends on the university's requirements, which we do not yet have. We can seek stronger measured performance; we cannot promise95%, change the target to manufacture it, or present training accuracy as held-out accuracy.

The research-paper route stays separate: novelty, stronger baselines and credible independent evaluation need evidence beyond a high capstone score.
