# Approved encoder comparison

Registered November 16, 2026, after the user reported access approval and the cluster's existing authenticated account successfully checked UNI2-h, Virchow2 and H-optimus-0. H-optimus-1 access remains unavailable. No model scores from this round existed at registration.

## Question and scope

Test whether replacing the frozen picture reader improves five-category BreakHis classification under the existing evaluation. Run all three named readers and report every result. Preserve the requested binary benign/malignant decision followed by four malignant categories. Include a direct-five control for each reader.

| Reader | Feature definition | Pinned model revision |
| --- | --- | --- |
| UNI2-h | Final 1,536-dimensional CLS token | `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` |
| Virchow2 | CLS concatenated with the mean of patch tokens, excluding four registers, 2,560 dimensions | `3158645804b69e3f3bc4439d4116edddf0840a72` |
| H-optimus-0 | Final 1,536-dimensional CLS token | `b145cc1e6c6b30d3251aa8b1f844e6974188a743` |

Use each pinned model's documented architecture and normalization. UNI2-h and Virchow2 use the fixed 224-pixel evaluation transform resolved from the published timm configuration, with shorter-side resize and center crop. H-optimus-0 uses a registered shorter-side 224 bicubic resize and center crop before its official normalization. Its native example expects 224-pixel tissue tiles at 0.5 microns per pixel. BreakHis physical scale equivalence has not been established. These packages change both reader and its documented input/readout, so their comparison measures transfer of the complete reader package.

## Fixed evaluation

- Original 7,909 images, four magnifications and manifest/protocol byte hashes. BreakHis supplies all current fitting and evaluation images.
- Original provider-key roles, arrangements 812 and 813, five overlapping trials each, 46 fitting, eight validation and 28 excluded keys per role. No excluded predictions. Case identity limitations and validation reuse carry forward.
- Reuse the single-center Midnight comparison's case-centered cosine KDE implementation, class/case/field weighting, fitting-only centers and whole-query-case-out fitting diagnostics. No class-specific centering, learned attention, adapter, fusion or new crop search.
- Four shared temperatures, 0.01, 0.03, 0.1 and 0.3. Select the earliest maximum validation mean-15-subset macro F1 separately for each named reader and direct/dual arm. Binary routing uses 0.5 with malignant ties.
- Compose field joint probabilities, average within each magnification, then equally across available magnifications. Evaluate all 15 nonempty magnification subsets.
- Ten paired roles, 80 candidate pipelines, 30 full-support head instances and 160 prediction archives per reader. Across three readers there are 240 candidate pipelines. Choosing across readers supplies 12 reader/temperature settings per arm and role, from three readers and four temperatures. These dependent validation opportunities are disclosed.
- Primary comparator is the accepted single-center Midnight KDE. Three-crop Midnight is secondary. Head-specific centering supplies current retained context. A descriptive improvement screen requires at least 0.02 macro F1 gain, nondecreasing raw accuracy and positive recall for every final class in both arrangements. Report raw deltas and calibration even when the screen fails.

Report five-category accuracy, macro F1, balanced accuracy, class recall/confusion, binary routing metrics, malignant conditional typing, full-four-view and worst-subset scores, log loss, Brier error and computation. Aggregate each arrangement separately. The 600 selected validation decisions per arrangement are repeated case/subset records, not 600 independent patients. All 82 provider keys entered earlier development, so this round supplies exploratory validation evidence.

## Execution and acceptance

Artifact preparation runs in scheduler CPU allocations and checks revision, size and SHA256. Model weights, extracted features, credentials and raw identifiers stay private. No remote model Python is executed. Use strict complete checkpoint loading with FP32 frozen parameters, eager attention, no autocast and TF32 disabled. The official model cards and their terms are pinned alongside weights.

First run one technical four-image check per model, selected from fitting arrangement 812/trial 1, with two examples per audited source geometry. Public configs store path digests. Repeat, outer host-batch partition and fresh-loader checks use fixed absolute and relative tolerances of 1e-4. Canonical model calls process one original image. Technical checks produce no validation scores.

After checks pass, run four magnification shards per model, at most four A40 tasks concurrently, one GPU and at most 45 minutes per shard. Audit exact 7,909-field coverage, finite arrays, shapes, frozen model lineage and source/config/archive checksums before classifier use. Bind actual accepted feature hashes in a later source revision, without changing the registered numerical recipe. Independently replay every candidate and selection before accepting results.

The target remains 94–95% end-to-end accuracy. It is an experimental objective. Access approval, a larger backbone and successful technical execution establish no performance guarantee or novelty claim.
