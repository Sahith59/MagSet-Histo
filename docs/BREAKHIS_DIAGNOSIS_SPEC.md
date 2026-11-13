# Saved-model diagnosis, fixed before execution

November 13, 2026. Diagnose the completed negative adaptation experiment. Use all ten existing runs; no fitting, encoder execution, new image processing, calibration, thresholds, parameter search or outer-role scoring. Keep the frozen baseline as the reference. TCGA remains deferred.

Bind the input to the complete comparison SHA256 `c26398dcfbeac6ac76f737ab686a987f82c2ea6874a45132fe9a0a7fc8b5637b`, trusted manifest/protocol, immutable training source, and each verified checkpoint/cache/head. Recheck adaptation-specific artifact gates and archived validation probabilities before computing diagnostics. Only the existing 46 fitting and eight validation cases may be scored in each run.

## Fixed probes

1. Replay saved frozen and adapted logistic case heads on fitting and validation cases, for all 15 view subsets. Report raw case accuracy, balanced accuracy, macroF1, class recall, confusion matrices, full-view and worst-subset metrics. Include class-balanced case cross-entropy to compare role distributions; ordinary per-subset CE is case-uniform. Fitting metrics describe apparent fit, not generalization. No classifier or normalization is refitted.
2. Evaluate the saved auxiliary image head on raw adapted CLS features, with dropout off and canonical unaugmented caches. Report raw image accuracy and class/case/view-balanced accuracy, CE and subtype recall. Within each role, give each class mass1/8, each case equal mass within its class, each of four views mass1/4, then uniform fields within a view. This matches expected training sampling mass, not realized minibatches; do not add another inverse-class multiplier. Field labels are inherited case labels, not independent field annotations.
3. Evaluate two prespecified diagnostic case readouts of that same auxiliary head: average logits within each view then equally across available views, and average field probabilities within each view then equally across available views. Report both on all cases/subsets/runs; do not select a winner or describe a new method success. The affine-logit path must agree with applying the head after equal-view mean feature pooling. A head-to-head gap alone cannot prove loss from mean pooling.
4. Measure frozen/adapted feature cosine similarity and relative L2 drift for raw fields and separately normalized per-view case means, by role with class/case/view-balanced mass. This describes how much the representation changes; it does not establish a causal mechanism or clinical tissue correctness.

Saved auxiliary weights are float32. NumPy uses float64 for diagnostic linear algebra/probabilities and checks a fixed first64-field probe against the actual dropout-off Torch float32 head within1e-5 probability difference. Affine-pooling identity tolerance1e-10. Existing logistic validation replay remains within1e-8.

Carry stable log probabilities for every case and field score. Reconstruct the saved logistic logits and cross-check their probabilities against the original scorer; use log-sum-exp for mean-logit scores and log-mean-exp within/across views for probability averages. Do not cap an underflowed true-class loss by clipping its probability. Store the log probabilities for independent CE verification.

For feature geometry, norms at or below1e-12 are treated as undefined. Exclude their weight from that cosine/relative-L2 average and report valid mass and near-zero counts; return null if no valid mass remains. Relative L2 uses the frozen norm. This does not silently replace undefined geometry with agreement.

## Interpretation and outputs

Save private case predictions, field probability arrays, role/truth/coverage and artifact hashes so an independent reviewer can recompute the diagnostics. Public outputs contain aggregate counts/scores only. Require the complete ten-run matrix before cross-run summaries.

High fitting scores with poor validation support an observed generalization gap. Stronger auxiliary held-out case scores than the logistic refit would implicate readout/objective differences; they would not alone establish an early-pooling defect. Weak auxiliary and logistic validation together would support investigating representation/data diversity. Similar normalized case features describe limited geometric change; small changes can still cross margins, so geometry alone does not explain unchanged boundaries. None of these observations proves a sole cause, reliable clinical accuracy, novelty, or a fresh confirmation result.

All diagnostic probes reuse development data and are posthoc relative to the completed experiment. They may guide one newly registered experiment; they cannot rescue the failed primary improvement gate or certify whichever probe scores highest. No new training is part of this task.
