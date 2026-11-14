# BreakHis fixed multicrop: independent actual-output review

November 14, 2026. **Accepted as a complete, correctly replayed exploratory comparison. The registered improvement gate failed. Keep the previously retained frozen Phikon reference.** This review checked all20 actual saved heads and their underlying four original/candidate feature shards. It performed no model fitting, raw-image decoding, encoder forwards, scheduler jobs or outer-role scoring.

Reviewed comparison SHA256: `13e547269d9c5dc9d0ab3b867b6cba36e068501d14f80109f9b8e4fa17c7164b`. Both extraction and head execution identify source `47ad41657979c3e154cc9db7b89f1fec4b1f40ac`. The actual source archive SHA256 is `73454a6ba398560d6ca54abedc550aff9130bce8b24cc4c9a38fe1962e129118`; its embedded Git commit, extractor dependency hashes, evaluator implementation hash and exact configuration bytes were independently verified. A later source change to avoid repeated NPZ reads was not the source of these results and was not used for scientific replay.

## Checked outcome

Scores below average each run's15 nonempty magnification subsets, then its five overlapping trials. The matched control and candidate both fixC=0.01 without validation selection.

| Arrangement | Fixed-C original F1 | Fixed-C multicrop F1 | Candidate minus control | Original / candidate accuracy | Phyllodes recall, original / candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
|812|0.346333|0.340500|-0.005833|43.00% /42.00%|0 /0|
|813|0.280833|0.286722|+0.005889|34.33% /34.67%|0 /0|

The candidate decreases F1 on812 and increases it slightly on813. Neither arrangement reaches the required+0.02 F1 improvement, and neither obtains positive phyllodes recall. Independent recomputation confirms `gate_passed=false`; the small positive813 change does not establish a successful improvement.

The existing retained frozen reference uses its earlier validation-selected regularization, with F1 approximately0.3843/0.3169. It is a separately identified historical comparator, not the fixed-C control above. Its selectedC values across trials1–5 are `[0.01,1,10,0.01,10]` for812 and `[0.01,0.1,0.1,0.01,10]` for813. Each saved historical-reference report/prediction hash, selectedC, validation summary, normalization policy and original feature-cache binding was checked. The new fixed-C heads must not overwrite or be described as exact replicas of those historically selected heads.

## Independent numerical verification

The review used standalone NumPy calculations rather than importing the production bag loader, normalizer, vectorizer, prediction replay, collector or metric functions. Each compressed NPZ array was materialized once per archive and reused in memory. No fit was repeated to check coefficients.

1. **Trusted identities and role reconstruction.** Parsed the original manifest and hash-bound protocol; reconstructed subtype-qualified case identities and canonical labels. Recreated the seed812/813, trial1–5 inner choices directly from per-class sorted outer-training cases. Checked exact role lists, their46/8/28 counts, validation class coverage and every prediction record's role/identity/truth/subset. Recomputed22/20 distinct validation cases and31 in their union. No prediction records for excluded outer-role cases were used.
2. **All four original caches.** Independently hashed the actual local original NPZ and JSON files, including400×, against the registered pins. Verified their dimensions, finite float32 vectors, unique full magnification path coverage and metadata. The400× original finished copying before the completed review; acceptance does not rely on an unavailable original cache or on a remote-only equality assertion.
3. **All four candidate caches.** Checked actual NPZ hashes against saved metadata and all20 reports' provenance; exact encoder/official processor identity; immutable source/configuration/archive contracts; full-run status; zero technical thumbnail forwards; four quadrant forwards per original field; and every recorded raw-image checksum, width/height and integer-midpoint quadrant bound against the manifest. Checked all7,909 fields, with finite float32 original and quadrant vectors. Original vectors stored in candidate archives equal the actual accepted cached originals bitwise. The saved candidate equals `float32(0.5*float64(g)+0.5*mean(q,float64))` bitwise for every field. These are cache/manifest checks; raw image bytes were not reopened in this review.
4. **All20 normalizers and models.** Independently reconstructed each arm's raw within-case/view field means in float32. Independently recomputed fitting-only case-balanced first/second field moments in float64, the variance clamp and standard-deviation fallback, then the saved float32 normalization arrays. Every saved mean/std matched bitwise. Checked finite positive scales, model coefficient/intercept shape/dtype, exact eight-class order, fixedC policy, reported convergence status/iteration bounds and hashes of all model/metrics/fitting-prediction/validation-prediction artifacts.
5. **All16,200 saved probability records.** Reconstructed normalized masked mean features for each fitting/validation case and all15 subsets, with the four availability indicators. Replayed the saved affine coefficients using an explicit NumPy contraction and stable log-softmax. Checked every probability and log probability, exact record coverage/order and labels. There are13,800 fitting and2,400 validation case/subset records; they are repeated observations, not16,200 independent patients.
6. **Every saved metric and comparison decision.** Independently used argmax and weighted bincount confusion matrices to recalculate all eight-class raw and class-balanced metrics: F1, accuracy, balanced accuracy, recall/support, CE, Brier, record/weight totals and zero true-probability counts. Reconstructed all15 per-subset metrics, pooled metrics, all-view metrics, mean/worst-subset summaries, distinct class/case counts, phyllodes recall and fitting-minus-validation gaps for every head. Rechecked report summaries against comparison entries, both five-trial primary summaries, paired deltas and the registered gate.

The completed explicit-contraction replay compared30,850 metric scalars/arrays. Maximum metric absolute discrepancy was `6.661338147750939e-16`; maximum probability discrepancy was `1.2212453270876722e-15`; maximum log-probability discrepancy was `5.773159728050814e-15`. No metric, role, arithmetic, source, cache or saved-model provenance blocker remained.

## Generalization evidence

| Arrangement / arm | Fitting mean-subset balanced accuracy | Validation mean-subset balanced accuracy | Validation CE | Validation Brier |
| --- | ---: | ---: | ---: | ---: |
|812 original|96.91%|43.00%|1.745136|0.756092|
|812 multicrop|97.32%|42.00%|1.746794|0.757216|
|813 original|96.75%|34.33%|1.752068|0.770326|
|813 multicrop|97.15%|34.67%|1.752504|0.767037|

The generalization gap persists. The candidate fits slightly better but does not consistently improve validation. The fixed representation experiment changes locality, effective physical resolution and peripheral coverage together. The original Phikon feature uses official shortest-edge resizing and a center crop; each raw quadrant also undergoes that same processor. Raw quadrant boxes cover every original pixel once, while the subsequently encoded center crops do not retain all raw pixels. These results neither isolate a compression effect nor prove that higher-resolution/context-preserving approaches cannot help.

## Scope and limits

This accepts the saved numerical comparison and provenance at the reviewed hash. It does not independently rerun raw encoding, optimization, convergence trajectories or scheduler allocation, and does not establish a new clinical performance claim. Reported iteration counts and checked execution source support the declared fixed training policy; this audit did not refit to prove an optimization optimum. Separate cache/raw-input and execution records retain their own scope.

The20 heads constitute the declared two-arm, two-arrangement, five-trial matrix. The reviewed artifacts identify the original immutable execution source; no later efficiency fix, retry, extra fit or selected alternative is substituted in this review. Scheduler completion and elapsed-time claims should cite the execution record rather than infer timing from this numerical audit.

The task remains eight-subtype provider-case classification with all four magnifications and15 available-view subsets. Rare classes still have only one fitting case per run, and the reused validation cases are not a fresh confirmation cohort. These results do not reproduce the historical40× EfficientNet notebook or promise approximately95% accuracy under a grouped case protocol. Finish the capstone report with this bounded negative result, preserve the retained frozen reference and distinguish exploratory validation from final clinical/research claims. TCGA and novel-method work remain deferred.
