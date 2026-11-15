# Independent five-category protocol review

Reviewed November 15, 2026 before five-category fitting. Scope: the registered [specification](../BREAKHIS_FIVE_CLASS_SPEC.md), [canonical configuration](../../configs/breakhis-five-class-v1.json), existing source contracts and independently audited data roles. This review is separate from runner/metric implementation. No runner existed at the initial review, so protocol acceptance is not executable-source or output acceptance.

**Decision: accept the scientific design for registration. No protocol-level leakage blocker was found.** Production remains conditional on source review, meaningful tests, immutable source/configuration binding and the specified zero-classifier-fit input audit. No five-category performance has been measured by this review.

| Reviewed input | SHA-256 |
| --- | --- |
| Canonical configuration | `d3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b` |
| Specification at initial review | `8ce92bd2c615ec2018fca33f03ad15757ad4d1604e735ce3361ce79fa32d757d` |

Independent arithmetic checks confirmed the ten runs, four complete candidates per arm/run, 120 production classifier-head fits, role supports, fixed routing rule and zero technical head-fit/encoder-forward/outer-prediction allocations. All eight registered feature NPZ/JSON hashes match their local accepted originals. Any subsequent amendment must be recorded before new outcomes and must not silently inherit this exact artifact binding.

## Selection opportunities and weighting

The direct arm fits one five-class head at each of four C values. The dual arm fits one router and one malignant specialist at each of the same four C values, using a single shared C for both heads. Thus each arm has four complete-pipeline validation selection opportunities per run, rather than a 4 × 4 specialist/router search. The disclosed count is 40 direct head fits plus 80 hierarchy head fits. Equal selection opportunities do not imply equal parameter capacity or computation; the specification correctly describes two complete pipelines rather than attributing any difference solely to hierarchy.

The prescribed sample weight 1/15 spreads each original case's base mass over all 15 masking rows. Head-specific balanced class weights then give equal total mass to that head's classes. With 46 fitting cases, each direct class has total mass 46/5 and each router group 46/2; with 35 specialist cases, each malignant subtype has mass 35/4. Cases within a head/class receive equal mass regardless of their field count. These are intentionally different factorized objectives and should remain disclosed. Do not apply another inverse-class multiplier after `class_weight='balanced'`.

Each arm independently chooses the first/smallest C maximizing mean-subset **end-to-end five-class macro F1**, retaining all four candidate outcomes. The proposed gate uses the selected complete pipelines on both arrangements and is an exploratory development screen. It supplies no independent test of the same selection decisions, significance guarantee or permission for further unregistered search.

## Roles and leakage controls

Original eight-class identities must reconstruct the existing roles before mapping labels to five categories. The audited fitting support is 11 benign / 25 ductal / 2 lobular / 5 mucinous / 3 papillary, validation 4 / 1 / 1 / 1 / 1, and excluded 9 / 12 / 2 / 3 / 2. Reinvoking a splitter over pooled classes would alter the validation selection and is unacceptable. The implementation must compare exact memberships and supports, rather than checking only total case counts.

The router fits all 46 fitting cases. The specialist fits only the 35 cases with **true malignant fitting labels**, never cases selected through validation routing, predicted labels, validation truths or excluded outcomes. It has no benign expert. The common representation and field-moments normalizer use the same 46 fitting cases for both arms and both hierarchy heads. Validation influences only the declared C choice; excluded cases influence neither fitting, normalization, C selection nor predictions.

Reusing all original frozen caches is compatible with this comparison because the encoder and feature extraction are fixed. No new stain processing, adaptation, feature selection, crop alternative, threshold choice or representation search is authorized. The implementation must verify exact per-view paths, hashes, encoder revision and array values before constructing case bags; deriving 15 rows from a case never changes its role.

The earlier [data diagnosis](FIVE_CLASS_DATA_DIAGNOSIS.md) remains a material limit: clinical identity is not independently reconciled, some conservative stems cross roles, all 82 provider keys entered development somewhere and only 31 distinct cases occur in validation across both arrangements. This comparison preserves the existing exploratory provider-case protocol. Neither a new model nor pooled labels create a fresh confirmation cohort or establish patient independence.

## Decisions, diagnostics and metric contract

The hierarchy's final class must use router malignant probability ≥0.5, with ties malignant, followed by the specialist's fixed-order argmax; benign routing stops with benign. Its composed probability vector is `[1-r, r*s_DC, r*s_LC, r*s_MC, r*s_PC]`. The global argmax of that vector can disagree with the required hard route. Explicit routed labels must therefore drive accuracy, macro F1, balanced accuracy, recalls, confusion matrices and C selection. A test must deliberately construct this disagreement.

Use stable log probabilities for cross-entropy and composed probabilities for Brier. Do not recover supposedly stable cross-entropy by taking `log` of probabilities after numerical underflow. Save enough candidate head/normalizer data to replay probabilities, log probabilities, explicit predictions and selection independently.

One useful clarification before source acceptance is to state the direct arm's binary diagnostic explicitly: sum its four malignant probabilities and compare with 0.5, including a malignant tie. This is a derived diagnostic; it need not equal the broad group of its five-way argmax and cannot redefine its final five-class decision. Direct malignant typing should likewise be a fixed conditional diagnostic. For the dual arm, binary diagnostics must measure the actual router; specialist typing on all truly malignant validation cases is diagnostic and must not conceal routing failures.

A wrong broad route and a wrong malignant subtype are both final errors. Hard-route five-class accuracy cannot exceed the router's binary accuracy on the same examples. The historical derived binary results are neither the new router score nor measured five-category performance.

Pooled benign constitutes half of each eight-case validation set. Raw accuracy and balanced accuracy must remain separate, alongside macro F1 and every class recall. Always benign gives 50% raw accuracy and 20% balanced accuracy. Retain all views/subsets, full-view and worst-subset outputs, both arrangements and all trials; do not select a favorable view setting for the headline. Repeated cases/subsets are correlated and cannot be presented as a large independent validation cohort.

## Required execution acceptance

Before production, independently verify that the runner has no outer prediction path, no independent head-C search, no hidden threshold tuning or extra real-data fits, and no role reconstruction after pooling labels. Tests must exercise a poisoned excluded role, true-malignant-only specialist fitting, original-role preservation, 15-subset weighting, candidate ties, hard-route/probability-argmax disagreement, binary reload, stable extreme-score losses and incomplete/tampered artifacts. These fixtures verify software behavior, not research accuracy.

Reject convergence warnings and incomplete matrices. A failed gate or technical failure does not authorize silently enlarging the C/iteration/threshold/representation budget. Independently replay the complete saved results before promotion or public claims. A measured result may be described only with its exact task, reused validation roles and view endpoint; 94–95% remains unachieved until such evidence exists.

This reviewer performed no fitting, encoder forwards, new predictions, scheduler submissions, resplitting or modifications to source/configuration/original data. Changes are confined to this review document. TCGA and novel-method work remain deferred.
