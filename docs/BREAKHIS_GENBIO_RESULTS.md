# Complementary GenBio encoder: completed comparison

November 16, 2026. The new frozen encoder gives a **mixed result rather than a consistent two-step improvement**. Two-step final five-category accuracy is **70.50% / 75.17%**, macro F1 **.5837 / .5813**. Both registered improvement screens fail. The retained P model remains the strongest observed two-step macro-F1 reference; 95% final accuracy remains unachieved.

| Method | Accuracy 812 / 813 | Macro F1 812 / 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: |
| Single-center Midnight, C / dual |75.67% / 75.00%|.5897 /.5643|82.50% /80.00%|
| GenBio / dual |70.50% / 75.17%|.5837 /.5813|72.50% / 75.00%|
| GenBio / direct-five control |71.33% /77.83%|.5960 /.6406|75.00% /80.00%|
| Head-specific Midnight, P / dual |76.33% / 75.67%|.6027 /.6025|85.00% /85.00%|

Headline metrics average all 15 nonempty zoom combinations and five overlapping trials per arrangement. Full-four-view accuracy is separate. The one-stage control reaches 77.83% and F1 .6406 on 813, while its 812 accuracy is lower. That score is neither the requested two-step result nor a consistent raw-accuracy improvement. Do not select the better arm separately for each arrangement and present a hybrid headline.

## What changed

[GenBio-PathFM](https://huggingface.co/genbio-ai/genbio-pathfm) is a publicly accessible complementary pathology encoder. Its prescribed transform resizes the complete image to 224×224, normalizes RGB, then processes R/G/B as separate single-channel examples through one shared frozen backbone. Concatenating three 1536-dimensional CLS outputs gives 4608 features. This differs from Midnight's CLS-plus-mean-patch representation and preprocessing. The comparison changes the encoder together with its prescribed input/representation, not one isolated biological factor.

Keep the original case-centered cosine density classifier, four temperatures, class/case/field weighting, probability pooling, explicit binary→malignant-four routing and 46/8/28 provider roles. C is the prospectively pinned primary reference; F is secondary and P remains context. No model fusion, threshold search or combined P-centering intervention was performed.

The official model benchmark's BreakHis headline uses a different four-type 40× configuration that excludes lobular and papillary carcinoma. It is not a 95% target for our five-category case benchmark. The official pretraining sources do not name BreakHis, but upstream image overlap has not been independently ruled out. We fit/evaluate only BreakHis images and labels; externally pretrained weights remain transfer learning. Community-license attribution and research-use restrictions apply; weights, upstream source and embeddings remain private.

## The important tradeoff

Compared with C, GenBio's true-malignant typing improves from 56.67%/56.67% to 60.00%/67.67%. Its binary router drops from 92.00%/92.67% to 82.00%/89.83%. Final benign/ductal/lobular/mucinous/papillary recall is 85.00/49.33/100.00/48.00/26.67% on 812 and 95.00/50.67/100.00/41.33/29.33% on 813. Fifty of 75 papillary events on 812 and 40 of 75 on 813 are routed to benign. This supports a representation tradeoff; it does not establish a label error or unique biological cause.

An ideal router with the **current fixed case-conditional type predictions** would reach only 80.00%/83.83%. Those bounds do not apply to a new field-level routing/pooling intervention, which could also change the specialist distribution. A P-router/GenBio-expert cascade would be a separately registered hypothesis, not an automatically validated improvement or route to 95%.

The two-step balanced accuracies are 61.80%/63.27%; cross-entropy .9467/.8369 and Brier .4617/.4254. Metrics are mixed; do not describe all probability-quality measures as worse. Fitting leave-case-out accuracy is 64.87%/65.39%, F1 .4812/.4942, using a different fitting class mixture and support-exclusion condition from validation. These are also different from the neural model's in-sample fitting diagnostics.

## Complete execution evidence

The first technical GPU check 4983050 exposed native batch-size sensitivity before any feature archive was published. The corrected reader uses one original image per official call (three internal channel examples), with host batches of four only for I/O. It retains the official parameters, BF16 rotary buffer and unchanged 1e-4 tolerance. Technical check 4983063 completed in 66seconds: repeat/host-partition/fresh-strict-loader differences were all zero. Peak allocated memory was 4.65GB and the measured steady four-image host batch took .404seconds. [Technical record](reviews/GENBIO_BATCH_EXECUTION_FIX.md).

All four A40 magnification tasks 4983078 completed in 3m54–4m28. They processed 7909 original images, equivalent to 23,727 channel examples. Independent audit 4983079 completed in 7 seconds and verified complete ordered paths, hashes, finite FP32 shapes and geometry/runtime declarations. Its zero encoder-forward count describes the CPU audit itself. The successful technical check used 16 additional image calls/48 channel examples; the earlier failed technical attempt is retained separately.

Classifier input audit 4983121 completed in 9 seconds. All ten paired CPU runs 4983141 completed in 35–38seconds; collector 4983142 completed in 4m46 with empty error output. It replayed 80 candidates, 160 prediction archives and 64,800 dependent records, including original roles, class supports, probability arithmetic and earliest selection. Root independently checked 68 selections/aggregates/recalls. Source CI passed 370 neural tests; the input-hash-only binding passed three focused checks. GPU smoke checks reader consistency on its sample; CPU collection does not independently re-encode all original images.

Feature producer `6861fbfe8d847f16fa09b4f839193716c1e8d501`; classifier producer `dfaa6c190f2ae8a32028aea6ba70ca82154ee6f6`. Classifier config SHA256 `196608c55ab8e3418f03b052f712fc77dcfe8155d3b38293d69c51610e435737`. Accepted comparison SHA256 `92e2ff976b1be20a69a0ccee2147537b2905293926675d2ff3be85283cc589d1`.

[Complete accepted JSON](evidence/BREAKHIS_GENBIO_RESULTS.json) · [Registered round](BREAKHIS_NEXT_ROUND_SPEC.md) · [Model/source search](reviews/NEXT_ROUND_MODEL_SEARCH.md) · [Independent actual-result review](reviews/GENBIO_ACTUAL_VALIDITY_REVIEW.md).

All 82 provider keys entered development; 31 validation keys are reused and clinical identities remain unresolved. Fields, masks and trials are dependent. No globally fresh clinical confirmation, novelty or 95% five-category result is established.
