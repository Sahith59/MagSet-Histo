# Five-category model, feature and training diagnosis

November 15, 2026. Independent model-focused review of the corrected BreakHis capstone target. The final labels are pooled benign, ductal carcinoma, lobular carcinoma, mucinous carcinoma and papillary carcinoma. The requested hierarchy contains a benign/malignant router and a malignant-four-class specialist. Benign predictions terminate at the router. [Confirmed scope](../DUAL_STEP_SCOPE.md).

This review inspected source, accepted cache metadata, saved diagnostic aggregates and the completed adaptation, supervised and regional comparisons. It introduced zero experiment fits, encoder forwards, raw-image decoding, scheduler submissions or excluded-role predictions. Newly trained five-category outcomes remain pending. The subsequent metric implementation has separate synthetic unit checks.

## Historical and current scores measure different targets

The supplied EfficientNetV2B0 notebooks report 374/399 and 379/399 correct predictions on single 40× images, giving 93.73% and 94.99%. Their image-random split and reused selection/evaluation images differ from the current provider-case task. The current frozen reference measures eight-class case predictions over 15 magnification subsets and five overlapping trials per arrangement, giving 45.67%/38.33% accuracy. The protocol differences prevent interpreting their numerical difference as a same-task accuracy decrease. [Original-result audit](CAPSTONE_ORIGINAL_RESULTS_REVIEW.md), [endpoint explanation](../CAPSTONE_BEFORE_AND_NOW.md).

Pooling benign labels changes both the prediction task and validation support. Each retained validation set becomes benign 4, ductal 1, lobular 1, mucinous 1 and papillary 1. Raw accuracy consequently weights benign at 50%, while five-class balanced accuracy weights every final class at 20%. Previously measured eight-class accuracy, retrospective binary accuracy and future end-to-end five-class accuracy need their own labels. [Confirmed scorecard](../DUAL_STEP_SCOPE.md).

## Exact accepted feature pipeline

The encoder is `owkin/phikon-v2` at revision `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`. `load_encoder` loads the official processor with `use_fast=False`, loads the safetensors checkpoint, sets evaluation mode, disables gradients and reads `last_hidden_state[:, 0, :]`. The saved cache uses finite float32 1,024-dimensional CLS vectors with TF32 disabled. This matches the author's demonstrated feature readout. [Extraction source](../../magset/features.py), [encoder pin](../../configs/encoders.json), [pinned author model card](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/README.md).

All four accepted metadata records describe RGB conversion, bicubic shortest-edge resize to 224, a central 224×224 crop, rescaling by 1/255, and channel normalization using means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. These settings agree with the pinned official `BitImageProcessor` configuration. [Pinned processor configuration](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/preprocessor_config.json).

The pinned backbone configuration has 24 transformer blocks, 16 attention heads, patch size 16 and final LayerNorm. A processed image supplies a 14×14 patch grid and a CLS token. The cache keeps the final CLS vector. Subsequent pooling operates on those vectors. [Pinned backbone configuration](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/config.json), [extraction source](../../magset/features.py).

The manifest contains 7,835 fields sized 700×460 and 74 sized 700×456. Shortest-edge resize followed by center crop therefore retains a central square region in source coordinates, approximately 65–66% of the original field area. The official processor can discard diagnostically relevant periphery or compress fine texture. This is a plausible representation limitation. Its contribution to classification error has no isolated measurement. [Manifest audit](../DATA_AUDIT.md), [regional cache review](BREAKHIS_MULTICROP_CACHE_REVIEW.md).

The retained case representation uses every field in each case/magnification bag. It first averages raw vectors within each zoom. `field-moments-v2` computes per-zoom mean and variance from fitting fields with equal case mass. The calculation averages each fitting case's first and second field moments in float64, clamps negative round-off variance to zero, substitutes scale 1 for standard deviations at most `1e-6`, and saves float32 mean/scale arrays. It then standardizes the raw per-zoom case means. Thus its scale reflects within-case field variation as well as between-case variation. [Bag loader and normalizer](../../magset/bags.py), [retained baseline](../../magset/baseline.py).

For an available subset, `vectorize` averages its standardized zoom vectors and appends four availability flags, yielding 1,028 inputs. Each case contributes all 15 nonempty subsets with fitting sample weight `1/15` per subset. The same cases recur across those rows, so 690 fitting rows still represent 46 provider cases. The source uses class-balanced logistic regression and the four-value C grid `[0.01, 0.1, 1, 10]`, selecting mean-subset validation macro F1 and retaining the first C on ties. [Retained baseline source](../../magset/baseline.py).

## What the completed experiments establish

| Eight-class pipeline | Fitting balanced case accuracy, 812 / 813 | Validation case accuracy, 812 / 813 | Validation macro F1, 812 / 813 |
| --- | --- | --- | --- |
| Frozen Phikon plus selected logistic head | 98.26% / 98.35% | 45.67% / 38.33% | 0.3843 / 0.3169 |
| Last-block Phikon adaptation plus selected logistic head | 98.47% / 98.67% | 45.67% / 38.83% | 0.3840 / 0.3205 |
| Supervised ResNet-50 reference | 99.6% / 99.7% | 37.00% / 40.00% | 0.3084 / 0.3006 |

These are mean-15-subset development results. The frozen/adapted fitting values come from the saved-model diagnosis. The ResNet fit values are rounded as published in its complete result table. The common finding is a large fitting–validation generalization gap across distinct training pipelines. [Saved-model diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md), [supervised results](../BREAKHIS_SUPERVISED_RESULTS.md).

The adapted auxiliary image head also reaches about 98% fitting accuracy under class/case/view weights, with 35.0%/31.8% validation accuracy. Its validation weighted cross-entropy is 3.022/2.865. Reusing this head through fixed mean-logit or mean-probability case readouts produces lower F1 than the retained frozen head. The training signal is each image's inherited case subtype. Independently annotated diagnostic-region targets are unavailable. [Saved-model diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md), [adaptation recipe](../BREAKHIS_ADAPTATION_SPEC.md).

The adaptation changes validation raw-field direction only modestly, with cosine 0.99840/0.99839 and relative L2 distance 0.08232/0.08244. Standardized case/view-mean cosine is 0.99659/0.99613. These summaries compare frozen and adapted representations. Interclass separability, feature rank and individual classifier margins require separate measurements. [Feature-geometry diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md).

For an affine classifier, averaging features and averaging logits commute. The saved diagnosis verifies that identity numerically. A simple change of this averaging order therefore cannot explain a hidden improvement. Learned nonlinear pooling remains an unmeasured hypothesis for the corrected target. [Saved-model diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md).

The regional experiment blended the original CLS vector with four quadrant CLS vectors using fixed 50/50 weights. Against its matched C=0.01 control, F1 changed by −0.00583/+0.00589 and phyllodes recall remained zero. Its fit balanced accuracy stayed about 97%. The experiment changed locality, effective resolution and peripheral coverage together. Its mixed validation deltas fail the registered improvement gate. [Regional results](../BREAKHIS_MULTICROP_RESULTS.md), [independent numerical review](BREAKHIS_MULTICROP_RESULTS_REVIEW.md).

The five-category fitting support is benign 11, ductal 25, lobular 2, mucinous 5 and papillary 3. Pooling benign removes the requirement to distinguish its four original subtypes. The malignant specialist still has only 35 fitting cases, including two lobular and three papillary cases. The retained eight-class papillary recall is 13.33%/4.00%, so malignant typing already has a documented weakness. Additional fields and view masks increase observations of those same cases. Limited case diversity, weak field supervision, representation mismatch and model capacity are plausible contributors. Their causal contributions remain unresolved. [Coverage audit](../BREAKHIS_REFERENCE_AUDIT.md), [saved diagnostic recalls](../BREAKHIS_DIAGNOSIS_RESULTS.md).

## Recommended first trained comparison

Use the original accepted frozen Phikon cache and the existing 46-fitting/8-validation/28-excluded roles for both arrangements and all five trials. Reconstruct roles using the original eight-class identities and split procedure, then apply the five-label mapping. This preserves the accepted role membership. The same fitting-only normalizer, raw within-zoom means and available-zoom mean-plus-mask representation feed both arms. [Split and representation source](../../magset/baseline.py), [normalization source](../../magset/bags.py).

Use `LogisticRegression` through the existing `fit_models` mean branch. Pin scikit-learn 1.6.1 and record the effective estimator settings, including L2 penalty, `solver='lbfgs'`, `tol=1e-4`, `max_iter=1000`, intercept fitting and `class_weight='balanced'`. Multiclass heads use multinomial log loss and the binary head uses logistic loss. Check convergence for every fit. The library multiplies class weights by supplied sample weights. [Existing fitter](../../magset/baseline.py), [scikit-learn 1.6.1 documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html).

| Arm or head | Fitting cases | Class order | Class weight before subset multiplier |
| --- | --- | --- | --- |
| Direct five-class head | All 46 | Benign, ductal, lobular, mucinous, papillary | `46 / (5 * class_count)` |
| Dual-step router | All 46 | Benign, malignant | `46 / (2 * group_count)` |
| Dual-step malignant specialist | True-malignant 35 | Ductal, lobular, mucinous, papillary | `35 / (4 * subtype_count)` |

Each case's 15 subset sample weights sum to 1 before class balancing. Head-specific balancing gives equal total mass to each decision class. It changes the weighting and factorized training loss across the two arms, so the comparison measures the complete declared pipelines. Its interpretation must include both the decision structure and those objective choices.

For each role, fit four complete candidates per arm at C `[0.01, 0.1, 1, 10]`. Each hierarchy candidate uses the same C for its router and specialist. Select one C per complete arm by end-to-end five-class mean-15-subset validation macro F1, using increasing grid order for ties. This gives 40 direct heads plus 80 hierarchy heads, or 120 individual head fits. Both arms receive 40 complete-system candidates across the ten roles. Keep all candidate validation scores and the selected arrays with source, configuration, data and feature hashes.

Fix the router decision to malignant when `p(malignant) >= 0.5`. The malignant head learns from true-malignant fitting cases. A predicted malignant case receives `1 + argmax(specialist_probability)` in the canonical final-label order. Report end-to-end predictions using that hard route. Count every routing error as a final-label error.

Save the router probability `r`, specialist probabilities `s`, explicit hard prediction and composed five-class probabilities `[1-r, r*s_ductal, r*s_lobular, r*s_mucinous, r*s_papillary]`. The composed vector's argmax can differ from the hard-routed label. Selection, accuracy, macro F1, balanced accuracy and confusion matrices must consume the explicit hard-routed labels. CE and Brier consume the composed probability vector. Stable log probabilities follow `log p(benign) = logsigmoid(-router_logit)` and `log p(malignant subtype j) = logsigmoid(router_logit) + logsoftmax(specialist_logits)[j]`.

Report both arrangements and every trial, mean-15 and all-four-view results, worst-subset F1, five-class recall/confusion, raw accuracy, macro F1, balanced accuracy, CE/Brier, binary sensitivity/specificity and malignant typing conditional on the true malignant group. Save fitting diagnostics separately. The hard-routing accuracy bound applies to the same records and weights. Any final five-class success requires correct routing and correct malignant typing. [Score definitions](../DUAL_STEP_SCOPE.md).

## Logistic regression and XGBoost

Logistic regression is the appropriate first bounded comparison because the accepted source already supports the representation, regularization search and coefficient replay. Its one-dimensional C budget is transparent. The current feature dimension exceeds the number of independent fitting cases and the specialist has two or three cases for important classes, making regularization and case-level diagnostics material.

The original reference paper used binary and malignant-four-class XGBoost heads with ResNet features. XGBoost is a reasonable future reproduction choice if that becomes the declared objective. Tree depth, boosting rounds, shrinkage, row/column subsampling and regularization introduce further settings that would need their own bounded recipe. Official documentation links deeper trees to increased model complexity and overfitting potential. XGBoost performance on the accepted roles/features remains unmeasured. The first requested hierarchy comparison can proceed through the existing logistic fitter. [Reference-paper audit](CAPSTONE_ORIGINAL_RESULTS_REVIEW.md), [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).

## Execution acceptance

Before accepting actual outputs, independently verify original role reconstruction followed by label pooling, fitting-only normalizer arrays, head-specific weights, canonical class orders, all 15 subset coverage, missing-view/permutation invariance, convergence, stable composed probabilities and save/reload replay. Include a test where the hard route and composed-vector argmax disagree. Independently replay every candidate's hard-label selection score and every selected model's fitting/validation metrics from saved arrays.

This recommendation supplies one executable experiment budget. It retains exploratory provider-case validation as the evidence scope. All 82 provider keys have entered development somewhere, clinical identity remains unresolved and 31 distinct cases recur in the validation union. A measured result near the user's 94–95% goal would still require those protocol qualifications. [Current evaluation scope](../DUAL_STEP_SCOPE.md).
