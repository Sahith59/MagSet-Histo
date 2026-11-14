# Registered frozen regional-view comparison

November14,2026, before smoke extraction, new region features or fitting outcomes. This is one bounded exploratory preprocessing comparison for the BreakHis capstone. It is not a novel mechanism, an exact notebook reproduction or a promised performance improvement.

## Fixed question and data

Does adding four deterministic regional views to the existing frozen Phikon representation improve the same eight-class case/subset task? Retain all four magnifications, all15 nonempty view subsets, both inner seeds812/813 and all five existing provider trials. Each fit uses46 fitting/eight validation/28 excluded outer-role cases. Do not change roles, labels or clinical-identity claims.

All82 cases appear somewhere in the union of existing fitting/validation roles. Frozen label-blind feature preparation may therefore cover the release; a run's fitted classifier and normalizer still use only its own fitting cases and predict only its fitting/validation cases. No outer-role prediction or globally pristine confirmation claim is permitted. Derived regions retain their original field/case identity.

## Exact input processing

Use the accepted Phikon-v2 model revision `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`,1,024D CLS embeddings, frozen evaluation mode, float32 and TF32 disabled. No new pretrained weights, gradient updates, augmentation, stain template, region selection or learned pooling.

The accepted processor is **shortest-edge resize to224 followed by a224×224 center crop**, fixed ImageNet normalization and RGB conversion. It is not the ResNet reference's whole-field square warp. Keep the official processor unchanged for both the original field and each region. Record and compare its actual configuration with the accepted cache metadata before opening raw images.

Control feature g is the exact accepted cached embedding of the original field. Divide the native field at integer width/height midpoints into top-left, top-right, bottom-left and bottom-right quadrants. Process each independently with the same official processor to get q1…q4. Candidate is `float32(0.5*float64(g) + 0.5*mean(q1…q4,float64))`. Save g, every q, the blended vector, raw hashes, dimensions and exact crop boxes for independent arithmetic replay.

The four raw quadrant boxes cover the input without overlap, including odd dimensions. The processor subsequently center-crops each region, so **the encoded regions do not preserve every source pixel**. This experiment changes local detail, context and peripheral coverage together; it cannot isolate compression as a cause.

Pin original cache NPZ and JSON hashes for all four views in `configs/breakhis-multicrop-v1.json`. Verify all raw hashes/dimensions/paths, feature order/coverage/dtype/finiteness, model/processor identity, immutable source/archive/configuration, and exact blend arithmetic. Smoke and partial caches cannot enter fitting.

## Budget and technical gate

One fitting-only GPU smoke uses the first four40× fields belonging to trusted trial1/seed812 fitting cases. It computes16 region embeddings and re-encodes four original fields solely to verify accepted-cache parity. Require original-feature maximum absolute difference≤0.0001, prospectively matching the previous accepted initial-cache feature gate, plus exact saved-vector blend replay. Report the observed value and a clean error log. It must not fit a model, score labels, inspect validation outcomes or select processing settings.

After tests, independent source review and real smoke pass, extract four regional vectors per unique field: at most31,636 production encoder forwards, one shard per magnification and at most four simultaneous scheduler GPUs. No production re-encoding of g. Allocate based on actual smoke throughput plus model-loading allowance; no exact wall-time promise. Distinct immutable shard directories prevent overwrites; verify and assemble all four before fitting.

Exactly20 CPU logistic fits: two arms × ten existing arrangements. Both arms pool raw field vectors within view, fit equal-case field first/second-moment normalization from fitting cases only (`field-moments-v2`), normalize pooled view means, then use mean available views plus four availability indicators. Fixed C=0.01, class balancing, equal total case/subset mass, LBFGS/max_iter1000; convergence is mandatory. No C grid, validation model selection, calibration or follow-on region/weight/encoder search. Save/reload all heads and independently replay probabilities from arrays without pickle.

Both arms have1,024 feature dimensions plus the same four flags. The original retained frozen reference used its historical four-C selection, so show it separately; the matched fixed-C control is the causal comparison for this new representation. Fit each arm's normalization separately using the same declared rule.

## Reporting and decision

Require all four complete audited caches and all20 unique converged fits. Report every trial and both arrangements, all15 subsets, full/worst-subset F1, accuracy, balanced accuracy, recall/confusion, probability errors, fitting/validation gaps, distinct case support, source/input/model hashes and compute costs. An independent numerical review must pass before adopting or promoting a result. Public files contain aggregate data only.

Prospective descriptive screen: candidate-minus-matched-control mean-subset macro F1≥0.02 on both arrangements and positive phyllodes recall on both. This is not significance or clinical sufficiency. Failure ends the experiment and retains the original reference; it does not authorize another crop weight, backbone or split. Passing remains exploratory and does not repair scarce independent cases, adaptive validation reuse, uncertain clinical grouping or the absence of a globally fresh cohort.

Finish the capstone report regardless of outcome. The research-paper route remains deferred.
