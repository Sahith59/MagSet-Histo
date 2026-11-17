# UNI2-h supervised linear field readout

Registered November 17, 2026, before this family's cohort fitting or validation outcomes. This is one standard frozen-feature probabilistic readout family addressing a baseline gap: the approved UNI2-h round evaluated case-centered cosine KDE, while the earlier Phikon/Midnight field-supervised logistic results do not establish the same learner's performance on UNI2-h. The separate [metric-learning family](BREAKHIS_SPECIALIST_METRIC_SPEC.md) remains a different experiment with its own accounting. This registration does not combine their outputs or choose a learner retrospectively.

## Fixed hypothesis and comparisons

Test whether regularized linear decision boundaries on frozen UNI2-h fields improve cross-case malignant typing and the final five-category task relative to that reader's accepted KDE. This is transfer learning with a standard supervised classifier, not a new encoder, a novel mechanism or an expected 94–95% result. The hypothesis is plausible because nearest-example geometry and discriminatively fitted class boundaries need not rank cases identically. Its main counterevidence is only two lobular and three papillary fitting cases, previous negative neural experiments and substantial reused-validation uncertainty.

The primary arm is the requested two-step classifier: benign/malignant first, then malignant-four when needed. The primary comparison is **UNI2-h linear dual versus accepted UNI2-h KDE dual**, separately in arrangements 812 and 813. Retained P is a separately named secondary reference. The direct-five arm is a prespecified one-stage control and stays a separate endpoint; a better direct result cannot be presented as a better two-step result. Report both arms and every candidate, including failures.

The classifier and input scaling change together relative to KDE. This comparison assesses the supervised linear readout package; it does not isolate a single optimization loss while holding geometry fixed. There is no metric-factor application, reader fusion, new crop, router threshold sweep or split adjustment.

## Immutable input and roles

The [configuration](../configs/breakhis-uni-linear-v1.json) binds the accepted UNI2-h input configuration and its exact SHA256, the complete accepted UNI2-h result ledger and P ledger. Delegate input loading to the existing approved-reader loader so its feature producer, encoder weights/configuration, feature audit, four cache shards, original image membership, dataset manifest and role protocol remain bound to the accepted bytes. No new image encoding is needed.

Use all ten original roles: five overlapping trials for each arrangement, with 46 fitting/eight validation/28 excluded provider cases. Fitting final-five counts remain 11 benign/25 ductal/two lobular/five mucinous/three papillary; validation counts are four/one/one/one/one. Excluded-role cases supply no normalization, optimization, selection or predictions. Preserve all four magnifications and all fifteen nonempty available subsets. No filename-based cross-view field correspondence is assumed.

## Exact existing normalization and field weighting

Reuse `bags.fit_normalization` and `field_five.field_training_rows`, rather than inventing a new scaler. For each view, calculate the equal-case mean of fitting cases' within-view raw-field means, and the equal-case mean of within-view second moments. Compute variance as the nonnegative difference between the second moment and squared mean. Use standard deviation one whenever it is at most 1e-6. Moment calculation uses float64; save means and standard deviations in float32. This is the accepted **field-moments-v2** policy.

Standardize each original field in float32 and append four one-hot magnification indicators. Each input has 1,540 components: 1,536 frozen UNI2-h features and four indicators. There is no L2 normalization, KDE support center or learned metric factor in this family. The availability subset controls which views are pooled at inference; it does not create new field-level training examples or change the normalizer.

For each head, field weight is

    N_head / (K_head * n_cases_in_head_class * 4 * n_fields_in_case_view).

Every class receives equal total head weight, every eligible case in a class receives equal mass, and every view within a case receives equal mass despite variable field counts. Direct-five and binary use all 46 fitting cases, with total weight 46 each. Malignant-four uses only 35 malignant fitting cases and has total weight 35. The malignant targets retain canonical labels 1/2/3/4, while binary targets are 0/1. Set `class_weight=None` because manual sample weights already perform the balancing. Do not apply another automatic class-balance factor.

## Optimizer, candidates and prediction

Use the existing `field_five.head_fit`: scikit-learn 1.6.1 `LogisticRegression`, L2 regularization, `lbfgs`, tolerance 1e-4, maximum 2,000 iterations and training seed 812. Direct-five and malignant-four use the estimator's multinomial probabilistic readout; binary uses its logistic readout. This estimator supports sample weights and probability prediction directly; no hidden calibration cross-validation is required. [Official version-matched documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html).

Fit exactly four ordered C values, **.01/.1/1/10**, per role and complete arm. The two dual heads share the same C. Choose the earliest maximum validation end-to-end mean-fifteen-subset five-class macro F1 separately for each arm and role. There is no separate head-wise C selection, repeated seed, probability calibration, threshold search, early-stopping selection or larger C grid. A convergence warning is an error under the existing helper and must be reported rather than silently changing optimizer settings after viewing validation outcomes.

Before accepting any fitted head, require **n_iter < 2,000** and an independently reconstructed **normalized maximum absolute objective gradient at most 1e-4 + 1e-8**, including intercept gradients and the L2 coefficient contribution. These checks verify convergence at the fixed registered tolerance; they do not change the optimizer, iteration budget or candidate selection. A head failing either check fails acceptance and is reported explicitly.

For dual inference, compose the field-level joint five-vector `[1-b, b*q_ductal, b*q_lobular, b*q_mucinous, b*q_papillary]`. Average joint probabilities across fields within a magnification, then equally across available magnifications. Derive the case binary and conditional malignant probabilities from that pooled joint distribution. Route malignant probability at least .5 to the highest conditional malignant subtype; otherwise output pooled benign. Benign stops after the binary decision. The direct arm takes the pooled five-class argmax instead. Use the existing logsumexp pooling and finite saved log probabilities.

## Budget, fitting-only technical acceptance and replay

Ten roles times four C values times three optimized heads produce **120 production head fits**. Four C values times two arms times ten roles produce **80 candidate pipelines**, **160 fitting/validation prediction archives**, and **64,800 dependent case/subset records**. Each candidate has 46*15 fitting plus 8*15 validation records. These are computational counts, not independent sample sizes.

Before production, use arrangement 812/trial 1's complete 46-case fitting role for a technical smoke: all four C values and all three heads, giving **12 additional technical head fits**. Score one fitting query for each of the five final labels across all fifteen subsets. This yields eight fitting-only archives and 600 dependent records. Cover all five labels explicitly to exercise the complete output mapping. No validation predictions or outcomes are allowed in that smoke. Preserve the smoke's source/configuration/normalization/model hashes, convergence metadata, head weight receipts, finite objectives/gradients and save/reload probability replay. These technical fits are additional to the 120 production fits.

Use scheduler jobs, bounded to four concurrent tasks, four CPUs and sixteen GiB host memory per task, forty-five minutes per production role and thirty minutes for the collector. The root deployment lane chooses the permitted cluster allocation; there is no encoder GPU work. Do not launch another unregistered learner or combine linear and metric outputs automatically.

The collector must independently reconstruct every candidate's field inputs/weights, saved coefficients, probabilities, joint pooling, metrics and earliest-maximum selection. Verify original roles and class labels, fixed C values, exact fitting-only normalizers, total head masses 46/46/35, all 120 production fits, all 160 archives, no extra excluded-role predictions and immutable source/input bindings. Save all models and actual prediction archives privately; public results retain safe provenance and aggregates.

## Interpretation and decision

Fitting scores are **in-sample resubstitution; optimizer and normalizer use fitting cases only**. They are not leave-case-out model evaluation or evidence of transfer. High fitting accuracy with poor validation indicates a generalization gap rather than the claimed success of a strong reader.

Report final accuracy, mean-subset macro F1, balanced accuracy, five class recalls/confusions, router and conditional true-malignant typing, proper losses, all-four-view and worst-subset performance, convergence and time/memory. The descriptive paired improvement screen requires at least +.02 macro F1 in each arrangement, nondecreasing accuracy and positive recall for all five labels versus the accepted same-reader dual reference. Evaluate the separately declared P comparison without renaming a failed primary screen as success.

This family contains four C choices per arm/role, with eight arm/C combinations if choosing between architectures. It is additional to the independently registered metric family and the full earlier reader/temperature development history. Do not hide that cumulative choice history or describe either family as fresh confirmation. There are 31 distinct reused validation provider keys, dependent trials/subsets, all 82 qualified provider keys have entered development, and clinical identities remain unresolved. Neither ordinary event-level binomial intervals nor a high pooled accuracy establishes independent clinical validation. No performance, novelty or 94–95% claim is made before the complete saved-result replay.
