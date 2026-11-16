# Native-pixel input: completed negative result

November 16, 2026. Reading nearly the complete field at its original pixel scale **did not improve this frozen encoder/classifier**. Final two-step accuracy is **60.00% / 61.00%**, with macro F1 **.4394 / .5014**, compared with the fixed three-crop reference at 75.50% / 76.83%, F1 .5925 / .5799. Both registered improvement screens fail. Retain the earlier reference; successful extraction is not predictive improvement.

| Frozen-input dual-step method | Accuracy 812 / 813 | Macro F1 812 / 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: |
| Single center, C |75.67% /75.00%|.5897 /.5643|82.50% /80.00%|
| Three-crop mean, F |75.50% / 76.83%|.5925 /.5799|82.50% /82.50%|
| Native 700×448, N |60.00% /61.00%|.4394 /.5014|60.00% /65.00%|

Headline accuracy/F1 averages all 15 nonempty magnification subsets and five overlapping trials in each arrangement. All-four-view accuracy is a different endpoint. The direct-five native control scores 59.50% /56.83%, F1 .4456 /.5038. A larger input is therefore not automatically a stronger representation.

Binary accuracy is 84.83% /71.67%; true-malignant typing is 46.67% /65.00%. Final benign/ductal/lobular/mucinous/papillary recall is 75.00/80.00/58.67/28.00/13.33% on 812 and 67.33/74.67/98.67/40.00/5.33% on 813. Even where the conditional specialist improves, routing and end-to-end errors can worsen. The selected fitting leave-case-out accuracy is 64.99% /56.12%; it uses a different fitting class mix and is not directly comparable with validation accuracy.

## What the comparison tested

Keep the original frozen Midnight weights, 3072-dimensional output, fitting roles, four temperature choices, class/case/field weighting, probability pooling and explicit binary→malignant-four routing. Replace the resized 224-square reader with a direct 700×448 crop retaining 97.39%/98.25% of the original two image heights. The official positional interpolation expands 37×37 learned patch positions to32×50; CLS plus the mean of 1600 patch tokens remains the field descriptor.

This changes resolution, coverage, global context and positional interpolation together. The result rejects the claim that this particular larger-input intervention improves this benchmark. It does not isolate which change caused the decline, prove a clinical mechanism, or establish that higher-resolution pathology models cannot work. A numerical or metadata failure was not found by the declared checks; that does not make predictive failure an implementation success claim.

## Actual execution and verification

Four-image technical check 4982934 completed in 82seconds, including 16 native forwards. Repeat and fresh-loader feature differences were zero; single-versus-batch maximum difference was 5.96e-8. Peak allocated memory was 6.78GB (6.31GiB), and the measured steady four-image batch took 1.583seconds.

All four A40 extraction tasks 4982936 completed in 12m50–14m33 across four GPUs on two nodes. The independent CPU audit 4982937 completed in 5 seconds, accepting all 7909 original paths, finite FP32 features, geometry and hashes. This audit performed no encoder forwards; production extraction performed 7909, in addition to the 16 technical forwards. Native inputs have 6.25 times the 224-square pixels and roughly 38.8 times the attention-matrix terms, so forward counts alone do not compare compute cost.

Classifier input audit 4982980 completed in 5 seconds. All ten paired classifier tasks 4982985 completed in 27–46seconds. Collector 4982986 completed in 4 minutes with empty error output, verifying 80 candidates, 160 prediction archives and 64,800 dependent candidate records. Root independently checked 68 selection, aggregate and class-recall calculations. CPU collection replays cached-feature classifier arithmetic; it does not independently re-encode every raw image.

Feature producer `9eeb4edfad0822cb59518de86376efb687b240f3`; classifier producer `6c216c5c3184650127a9ac4e9e8a9bc377ba0bb3`. Bound classifier config SHA256 `24c78a449a2b8da09ef0bb90e1492e94c2b956a302eeb0bdc609b0b1b508fac0`. Accepted comparison SHA256 `417ec2a4a3806d63154f2c76300db378568e19a29b0d5d1531d745ea891fd6e4`.

[Complete accepted JSON](evidence/BREAKHIS_NATIVE_RESULTS.json) · [Registered recipe](BREAKHIS_NATIVE_SPEC.md) · [Independent review](reviews/NATIVE_AND_CENTERING_BENCHMARK_REVIEW.md) · [Three-crop reference](BREAKHIS_COVERAGE_RESULTS.md).

BreakHis-only fitting/evaluation, 46 fitting/eight validation/28 excluded provider cases, reused 31-key validation union and unresolved clinical identities are unchanged. These are exploratory development results. 94–95% final accuracy and novelty are not established.
