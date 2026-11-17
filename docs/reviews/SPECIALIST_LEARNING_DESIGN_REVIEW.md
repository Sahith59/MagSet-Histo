# Fixed-head metric-learning design review

November 17, 2026. The final decision is the [registered head-metric experiment](../BREAKHIS_SPECIALIST_METRIC_SPEC.md), not the earlier considered PCA or free-factor alternatives. This note explains that design and its limits. Its author implemented `magset/specialist_metric.py`; it is a design/self-review, **not independent implementation certification**. Only synthetic unit fixtures have been fitted by this author; no cohort job, validation outcome or new performance result is reported here.

## Selected family

Select one supervised diagonal metric-learning family on frozen 1,536-dimensional UNI2-h descriptors. Fit binary and malignant-four metrics independently from original fitting cases. Evaluate identity, binary-only, malignant-only and both, with **both versus identity as the primary comparison**. The same two fitted metrics are shared across these arms. There are twenty production fits across ten unchanged roles plus two fitting-only technical fits. All four zooms, fifteen nonempty subsets and original case roles remain fixed.

Same-class attraction and other-class separation are standard metric-learning ideas; [large-margin nearest-neighbor learning](https://www.jmlr.org/papers/volume10/weinberger09a/weinberger09a.pdf) and [supervised contrastive learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) establish this general direction. The case-mean density objective here is an adapted engineering hypothesis, not a reproduction of either algorithm or established novelty.

Each role inherits its previously selected UNI2-h dual temperature, fixed for both heads and all arms. The historical four-temperature selection and adaptive reader choice remain part of development history. The new family adds no seed, temperature, penalty, architecture or checkpoint selection. Choosing among its four reported arms would still add four arm choices; the primary comparison is fixed beforehand. No encoder, threshold, split, fusion or crop search belongs here.

## Representation, capacity and objective

Calculate every fitting case/view centroid as the **float64 mean of original frozen field vectors**. Both heads use the original equal-case per-view center from all 46 fitting centroids. Subtract and L2 normalize in float64 with epsilon 1e-12. Cast only these normalized training vectors to FP32, preserving the old KDE center definition.

Each head learns one length-1536 vector theta shared across views, initialized to zero: `a = .5*tanh(theta)`, `d = exp(a)`, and `z = normalize(d * normalize(case_mean - full46_center))`. Factor bounds are exp(-.5) to exp(.5), approximately .607 to 1.649. The corresponding pre-normalization diagonal squared metric ranges from exp(-1) to exp(1). Each head optimizes 1,536 coefficients without hidden layers, biases, attention or encoder updates. This is more constrained than a dense projection, but still many coefficients relative to independent fitting cases. Bounds and an identity penalty mitigate rather than eliminate overfitting.

The binary objective uses 46 fitting cases (11 benign/35 malignant). The malignant objective uses 35 cases (25 ductal/two lobular/five mucinous/three papillary). For each anchor and view, compute one class density by averaging exp(cosine/inherited_T) over **other cases** of that class. The correct-class denominator is n_class-1; every other denominator is n_class. Normalize densities across classes. Minimize the true-class negative log probability, with per-anchor weight `1/(K*n_class)`, averaged across four views, plus **10*mean(a**2)**.

A positive therefore always comes from another case, not another field of the same case. Logsumexp gives the most confusable negative densities the strongest softmax gradients. There is no separate discrete hard-negative mining rule or second training loss.

Use exactly 200 full-batch FP32 Adam updates, learning rate .01, betas .9/.999, Adam epsilon 1e-8, zero weight decay, seed 812, deterministic execution and disabled TF32. No restart, early stopping or checkpoint selection. Each history row records loss **before** that update; the final-step objective is separately saved. Preserve actual FP32 theta, actual FP32 log factors and factors cast to float64, plus state/history/source/configuration hashes. A float64 NumPy theta-to-factor map is an approximate independent rounding check; stored FP32 factors are authoritative.

## Inference and factorial interpretation

Apply the selected head factor to normalized original **field descriptors**, normalize again and retain same-view class/case/field-balanced KDE. Identity bypasses scaling and re-normalization exactly. Training on case means while predicting from fields is a deliberate mismatch that can limit transfer; the trained loss is not exactly the deployed field-density loss.

For each field form `[1-b, b*q_ductal, b*q_lobular, b*q_mucinous, b*q_papillary]`, average equally over fields within a view and available views, then use the original .5 hard router and malignant argmax. Benign terminates after step one. No benign subtype expert is fitted.

Malignant-only must preserve binary probabilities and routes. Binary-only preserves field-level subtype probabilities, but changes field malignancy weights. Pooling joint probabilities means it can consequently change a pooled conditional subtype prediction. It would be incorrect to characterize binary-only as changing only the final route. Both changes the two geometries without reader fusion.

The identity arm must replay accepted selected UNI2-h predictions before any gain is credited. P is a separately named contextual reference, not a substitute matched control or a router to insert into the model.

## Honest fitting diagnostic

Omitting the query from density support does not remove its labels or inputs from learned theta. Label these scores **in-sample learned metric with query case omitted from density support**. Recomputing support centers without that query does not undo its metric-training contribution. These are optimization diagnostics, not leave-case-out model generalization.

An actual leave-case-out model diagnostic would refit the center and metric without each query. Holding out one lobular case leaves only one lobular fitting case, so the exact cross-case-positive recipe cannot train that class. Duplicating that case as its positive violates the premise. Do not report an incomplete subset of possible refits as an overall fitting-generalization score. No additional 350 diagnostic fits are required.

Separate fitting loss/gradient/factor summaries from eight-case-role validation outcomes. Validation does not train these metrics, but repeated role reuse across rounds makes this exploratory development. All 82 qualified provider keys have entered development; clinical identities remain unresolved. Subset events, overlapping trials and repeated views are dependent, so 600 events per arrangement are not 600 independent cases.

## Motivation and failure conditions

The [accepted diagnosis](APPROVED_ENCODER_ACTUAL_MODEL_DIAGNOSIS.md) reports UNI2-h conditional malignant accuracy 72.00%/60.67% and router accuracy 83.33%/90.33%. Separate subtype and binary geometries can address both documented errors instead of merely changing the reader again.

Counterevidence remains sparse independent subtype support, prior negative neural/MIL studies, stain/acquisition confounding, and case-mean versus field-inference mismatch. Diagonal weighting can learn nuisance directions shared by the few fitting cases. Independent head fitting may help one part while harming the final joint endpoint. A large fitting gain alone does not establish transfer.

The **malignant-only** final-accuracy ceilings are fixed-router accuracies 83.33%/90.33%, even with perfect typing. The older **86.00%/80.33%** ceilings instead fix old case-conditional type predictions and grant a perfect router. Neither bounds the new both-head model, where both geometries and pooled conditionals change. These counterfactuals still explain why a routing-only promise of 95% was unjustified. **94–95% remains unachieved and is not an expected result of this study.**

Report all arms, both arrangements, final accuracy/F1, routing and true-malignant typing, class recalls/confusions, all-four-view and worst-subset outcomes, proper losses and negative results. Preserve the fixed primary comparison even if an ablation looks better. No automatic combination, threshold adjustment or new training recipe follows failure.

## Standard alternative considered

A fixed class-balanced linear SVM on mean UNI2-h case descriptors is a credible inexpensive later comparison. [Official LinearSVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) describes its linear one-versus-rest classifier and inverse-frequency class weighting. It replaces the density mechanism and introduces score/probability choices for the current field-joint output. It is deferred rather than added to this searched family. XGBoost is also outside registration.

Synthetic source checks cover explicit NumPy cross-case/class-balanced oracles for both heads, finite analytic/numerical gradient parity, identity field geometry, bounded factors, zero-vector stability, case/view permutations, deterministic fixed-budget fits, artifact replay and restoration of caller RNG/determinism settings. These certify mechanics, not cohort performance. The separate runner/collector must certify role/support isolation, missing-view and field permutations, identity replay, immutable source/configuration bindings and every actual aggregate.
