# Wider-image coverage: a small, mixed improvement

November16,2026. Reading left/center/right regions with the frozen Midnight reader produces **75.50%/76.83%** final two-step accuracy, versus single-center75.67%/75.00%. Macro F1 improves slightly on both arrangements. Accuracy falls0.17points on812 and rises1.83points on813. **94–95% remains unachieved.**

| Two-step input | Arrangement | Mean15 accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Single center |812|75.67%|.5897|62.87%|82.50%|
| Three-crop mean |812|75.50%|.5925|62.80%|82.50%|
| Single center |813|75.00%|.5643|62.40%|80.00%|
| Three-crop mean |813|76.83%|.5799|64.93%|82.50%|

The15-subset average and all-four-view columns answer different questions. With every zoom present, the new model scores82.5% on both arrangements. That is not95%, and it does not replace the registered15-subset endpoint. The existing+.02F1/nondecreasing-accuracy screen fails; the measured small gains and tradeoffs remain reportable.

The direct-five control scores75.00%/73.33%, F1.5852/.5758. Two-step binary accuracy is92.17%/93.17%, while true-malignant typing is57.00%/59.33%. Final benign/ductal/lobular/mucinous/papillary recall is96.67/62.67/92.00/44.00/18.67% on812 and96.67/56.00/96.00/41.33/34.67% on813. Mucinous and papillary recognition remains weak.

## What changed

Keep the original frozen weights, dataset roles, classifier math and four temperature settings. For every original image, combine left, cached official-center and right224-square crop embeddings using a float64 arithmetic mean, then cast once toFP32. Images still contribute one3072-dimensional field description each. The two new crops cover omitted horizontal edges, with overlap that gives central areas more weight. The separate normalization/four-type density method remains unchanged.

Wider coverage provides modest benefit but each region is still resized to224pixels. The next registered native-input test retains over97% of original pixels in a700×448 global field, with the same frozen weights and classifier. This changes resolution, context and coverage together; it remains an unmeasured hypothesis.

## Complete evidence

All four A40 extraction tasks4982893 completed successfully in212–248seconds, followed by independent CPU feature audit4982894 in12seconds. It verified all7909 original paths, cached-center lineage,15,818 new crop features, component shapes/hashes and exact fused-vector reconstruction. The four-image technical check4982889 had already passed both source heights, center/repeat/batch/fresh-loader parity and actual save/reload.

The bound classifier input audit4982916 completed in6seconds. All ten paired tasks4982918 completed in28–30seconds; collector4982919 completed in237seconds with no error output. It verified80 candidates,160 archives and64,800 dependent records, including source/roles/supports/probabilities/selection. Root independently checked20 selections and12 aggregate means. The input-only configuration binding passed its three focused tests; the source framework had299 passing tests and successful CI.

Feature producer `0ceb91c4036333d5043aa60f769a082e277f1eb6`; classifier producer `f82eb8e725d2ce87f06285fb532acb7502c441c2`. Feature audit SHA256 `6547bf186f06b96065d60a87e5eeddbad1a60f784ca3fa140a661891098bfd75`; classifier config SHA256 `a18a4bfe210265d7ca9a59a75cd8cd797e562c275f665b690cddfaebbdfd0c9b`.

[Accepted complete JSON](evidence/BREAKHIS_COVERAGE_RESULTS.json), SHA256 `39d15d949e9fdbf0606b407f1eb364b456cbbec36d896d3fb263e68a9f35699c` · [Registered recipe](BREAKHIS_COVERAGE_SPEC.md) · [Implementation review](reviews/COVERAGE_KDE_IMPLEMENTATION_REVIEW.md) · [Single-center reference](BREAKHIS_KERNEL_RESULTS.md).

Only BreakHis is fitted/evaluated.46 fitting/eight validation/28 excluded provider cases per role,31 distinct reused validation keys and unresolved clinical identity are unchanged. Fields, masks and overlapping trials are dependent. These are exploratory development results, without fresh clinical confirmation or a novelty claim. GPU replay verified frozen-reader consistency; CPU collection verifies cached-feature/classifier arithmetic, not independent raw-image model execution.
