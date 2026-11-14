# Capstone endpoint bridge: independent actual-output review

November 14, 2026. **Accepted within the saved-validation scope.** Independently inspected the actual bridge artifact and all ten local supervised validation field archives, prediction files and metrics reports. No new model fitting, image decoding, encoder/classifier forward pass, cluster job or outer-role prediction was performed. The review used a standalone NumPy calculation; it did not call `capstone_bridge.metrics`, its collector, or its metric comparison helpers.

Reviewed bridge SHA256: `51c71fd8122f6e1684eb969f5c465d7279b53b61eb5c5718c6855ede382d1811`. Bridge implementation SHA256: `4ed823925415a78465c173194f1cf37024eab576c46ce7cb0029975d9bd2061c`. Accepted supervised comparison SHA256: `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`.

## Independent checks

- Rehashed all ten metrics, predictions and validation-field archives against the accepted comparison and the bridge's artifact ledger. Checked the accepted training-source/configuration identities, actual configuration file hash, manifest/protocol hashes, analysis configuration hash and current bridge implementation hash.
- Independently reconstructed each seed/trial's eight validation identities from the trusted outer roles and per-class seeded inner choice. Confirmed one case per subtype and disjoint fitting/validation/excluded roles. Reconstructed each validation archive's exact path order and complete original-manifest field coverage.
- Recomputed stable log-softmax directly from saved logits, then probabilities, without executing a model. For all five slices (`all_fields`,40×,100×,200×,400×), independently used NumPy argmax and flattened-label weighted bincount to reconstruct eight-class confusion matrices, accuracy, macro F1, balanced accuracy, per-class recall/support, canonical cross-entropy, Brier score, weight sums, record counts and zero true-probability counts.
- Independently assigned equal mass to each class, each case within class, each available view within case, and each field within case/view. Checked each subtype's weight mass is1/8 in every slice. Raw metrics give each original field unit weight. Weighted and raw endpoints are different descriptive measures.
- Reconstructed every40× case probability as the arithmetic mean of that case's40× field probabilities. Checked truth and validation role, saved probabilities/log probabilities, bridge metrics and the corresponding accepted supervised40× case metrics.
- Recomputed both five-trial metric-array summaries and every distinct-case/image/repeated-evaluation count, including union counts across arrangements. Averages of nonlinear metrics were recomputed as averages of the five trial metrics, rather than substituted with metrics from pooled predictions.

All1,613 checked numeric scalars/arrays agreed within a maximum absolute difference of `2.2648549702353193e-14`. The maximum40× case probability arithmetic-mean difference was `4.440892098500626e-16`. There were no arithmetic, coverage or provenance blockers.

## Checked endpoints

These are means over five overlapping validation trials per arrangement, from the accepted ResNet image classifier.

| Endpoint | Arrangement812 accuracy / macro F1 | Arrangement813 accuracy / macro F1 |
| --- | --- | --- |
|40× raw original-image predictions|27.13% /0.2536|29.94% /0.2398|
|40× class/case/view-balanced image predictions|32.89% /0.2827|31.82% /0.2593|
|40× probability-mean case predictions|37.50% /0.3017|35.00% /0.2783|

Arrangement812 includes22 distinct validation cases and2,267 distinct validation images; arrangement813 includes20 cases and2,004 images. Their union is31 cases and2,914 images. There are7,529 repeated field evaluations across the ten runs, not7,529 independent images or patients. Checked distinct-view image counts are566/606/565/530 for812 and514/525/499/466 for813, in40/100/200/400 order. Union counts are729/775/729/681.

The public interpretation must state that this bridge retains the same **eight subtype labels**, while changing the prediction unit. It does not reproduce the historical EfficientNetV2B0 experiment: this ResNet trained on all magnifications, used the registered case/subset selection criterion and provider-case validation roles, and differs in architecture and training. The historical approximately95% saved notebook image accuracy is not directly reproduced or contradicted by one matched reproduction here. The image-to-case differences are descriptive; they do not isolate a causal pooling benefit.

## Aggregate privacy and claim limits

Inspected the entire reviewed bridge JSON against all manifest provider-case identifiers and original image paths. None occur. No local machine paths, cluster paths or individual probability rows occur. The artifact contains aggregate metrics, counts and cryptographic provenance; this reviewed content is suitable for an aggregate evidence page. This is an inspection of this exact hash, not approval of arbitrary later exports or the private underlying per-case artifacts.

All82 cases have entered development somewhere; these are reused provider-case validations with unresolved clinical identifier relationships. The bridge creates no fresh cohort, final clinical evidence, novel mechanism or improved model. None of its five slices were used to choose a new model.

## Independent review of the prospective head evaluator

Reviewed `magset/multicrop_compare.py` SHA256 `bf75a9cd8a32e7abb25804a2166017df4c064a91ad30606ead593215c56f1fe8`, including the CLI source-binding path. This is a source review of prospective code, not acceptance of new head outputs or measured performance.

No blocking issue found in the reviewed path:

- CLI requires the immutable source marker to equal the declared full source SHA, a valid archive checksum marker, and configuration bytes equal to the immutable registered configuration. It supplies an expected extractor source contract containing the same source SHA, archive checksum and actual hashes of the six extractor dependency files. Every candidate shard must match this contract and each other.
- Both arms verify the registered original thumbnail NPZ/JSON pins and exact encoder/processor identity. Candidate archives must retain exact cached thumbnail vectors, finite float32 quadrant vectors and the fixed float64-accumulation-to-float32 blend, with exact raw image identity/dimensions/quadrant bounds. Full-run metadata excludes smoke artifacts and requires the declared forward budget and exact blend replay.
- Each head uses the same retained46/8/28 roles, fitting-only `field-moments-v2` normalization, fixedC=0.01, class balancing, uniform subset contribution and1000-iteration limit. Convergence warnings fail the fit. Saved-array probability replay checks fitted estimator output. The fitting and validation prediction paths exclude outer-role indices.
- Head metrics retain raw and class-balanced eight-class confusion/recall/CE/Brier endpoints, all15 subset scores, mean/worst/all-view results, distinct support and explicit fitting-minus-validation gaps. Class-balanced weights give each subtype equal mass for both per-subset and pooled repeated-subset records; pooled subsets remain dependent observations. Stable saved log probabilities preserve CE even if a softmax probability underflows to zero.
- The improvement screen compares the two fixed arms on both arrangements; no head regularization/checkpoint search is introduced. The original historically selected frozen reference remains separately identified.

The immutable archive installation and scheduler invocation remain part of the execution trust boundary: archive markers are not independently reconstructed from a missing archive inside this numerical review. New extraction/head artifacts still require actual-output checks after execution. No scientific improvement claim is authorized by this source review.

## Independent actual fitting-only smoke acceptance — November 14, 2026

Independently accepted the **technical consistency** of smoke job4971319's four fitting-only40× fields. Recomputed the local NPZ/metadata hashes against `smoke-acceptance.json`, exact registered configuration hash `42e8e98e1310c7d9daf339c2f0376d2abb6fcc270dbb350def389318402498af`, all six recorded dependency hashes against Git source `47ad41657979c3e154cc9db7b89f1fec4b1f40ac`, and the actual source archive hash `73454a6ba398560d6ca54abedc550aff9130bce8b24cc4c9a38fe1962e129118` and its dependency contents. The smoke NPZ SHA256 is `ab2c0e2da857365070b0afe754102c14dcc3954a55ec47cb489e867e95bfdd16`; metadata SHA256 is `5e662a1b20c9e814bc1445a5afe47192a3dcf13ee559176060fb240278511884`.

Reconstructed trial1/seed812 from the trusted manifest/provider protocol and confirmed the exact first four sorted fitting40× paths. Independently checked raw-image hash/dimension metadata, quadrant bounds/order, finite float32 `(4,1024)` thumbnail/blend vectors and `(4,4,1024)` quadrant vectors, the16-quadrant/four-technical-thumbnail forward budget, and exact float64-accumulation-to-float32 blend replay. Saved re-encoded thumbnails reproduce the independently calculated maximum absolute discrepancy `1.385807991027832e-6`, below the prospectively fixed `1e-4` tolerance. The processor metadata matches the pinned original shortest-edge resize224/center-crop224 processor; no square-warp assumption was introduced.

A bounded read-only SSH check additionally hashed the actual original40× NPZ against its registered `b5daddf7c8539851dadce9be88c093aa3bc866efffc9dccffd3d6f1e02ad614d` pin and compared the four original cached rows directly: saved thumbnails are exactly equal by path. Their shared raw-vector SHA256 is `5b1173bfac8139e629bfa881e2e34dcdb94f06149f7338eec68b26d8019096be`. Remote smoke stderr is empty; stdout SHA256 is `2c1c5c63f4b22bb262f3d2b1bdea306014b35e203f95edec6cd8786ba90f7e23`. Scheduler completion/time and hardware timing in the acceptance record were not independently queried here.

This audit performed no fits, forwards, label predictions, validation/outer scoring or job submissions. Acceptance concerns provenance, allowed-role membership, tensor/geometry consistency and replay only. It does not establish semantic correctness of quadrant features, improved classification, clinical generalization, or acceptance of the pending full-cache/head outputs. Four technical fields are not an evaluation cohort.
