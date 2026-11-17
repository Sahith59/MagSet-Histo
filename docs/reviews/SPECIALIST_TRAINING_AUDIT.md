# Malignant specialist training audit

November 16, 2026. **The approved-reader comparison does not test a learned classifier on UNI2-h. A tightly constrained, malignant-only metric is a defensible next controlled study; its identity control and fitting-diagnostic definition need particular care.** The current evidence identifies weak cross-case subtype transfer, without proving a representation defect, a data-label defect, or a hierarchy implementation bug. No new fit, prediction, encoder execution, scheduler action or result selection was performed for this audit.

Subsequent implementation decision, before any new outcome: the [registered specialist study](../BREAKHIS_SPECIALIST_METRIC_SPEC.md) uses separate binary and malignant metrics in a four-arm factorial: identity, binary-only, malignant-only and both. Both heads inherit the previously selected UNI2-h temperature for each role; there is no new temperature or checkpoint selection. The main comparison is both versus identity, with the two single-head arms as ablations. The malignant-only recommendation below records this audit's initial proposal; the linked registration governs the selected experiment. A separate synthetic implementation test later exercised fitting and saved-state replay, and supplies no research performance.

This review inspected the current handoff, project instructions, accepted reader/MIL/LoRA diagnoses, registrations, and the ordinary entry points in `approved_five.py`, `kernel_five.py`, `kernel_model.py`, `field_five.py`, `five_class.py`, `midnight_five.py` and `case_mil.py`. It is a source/evidence diagnosis, not an independent raw-image or numerical reproduction of the accepted experiments.

## What was actually learned

| Completed experiment | Task-specific fitting | What its failure does and does not show |
| --- | --- | --- |
| Approved UNI2-h, Virchow2 and H-optimus-0 | No optimized head coefficients. Labels define balanced support densities; four KDE temperatures are validation-selected. | Tests frozen reader packages under this particular density rule. Does not reject a trained linear, SVM or tree readout on these features. |
| Phikon/Midnight case and field heads | L2 logistic coefficients, four shared C settings | Tests those representations, normalization and readout packages. Does not establish that a UNI2-h linear probe fails. |
| Midnight LoRA | Fixed-C1 initialization, head warmup, adapters | Much of the weakness already exists before adapter updates; eight of ten selected dual checkpoints precede adapter learning. Does not establish that every pretrained representation is damaged by fitting. |
| Midnight whole-case mean/attention MIL | Approximately 0.4 million fitted parameters, projection and head learning | Near-perfect fitting with poor subtype transfer. Opposes adding unconstrained capacity or training longer as an evidence-based solution; does not reject all low-capacity discriminative fitting. |

The three approved configs explicitly record `optimized_head_fits: 0`. Their run path enters `approved_five.run` → `kernel_five.run` → `kernel_model.prepare_case` / `density_logs`; there is no hidden optimizer in that chain. A repository search of active source/configurations found logistic implementations, but no SVC/LinearSVC, RBF SVM, XGBoost, CatBoost, LightGBM or sklearn boosted-tree implementation for the new approved features. The original supplied paper describes XGBoost, but that historical source is not an executed new-UNI comparison.

This is an experimentally relevant gap, not proof that one of those untested models will win. A broad model-family search would add substantial validation-selection opportunity to an already reused cohort.

## Bottleneck supported by the results

Each full fitting role has 35 malignant cases: 25 ductal, two lobular, five mucinous and three papillary. Every additional field and magnification from one provider key remains dependent evidence. For cross-case metric learning, the two lobular keys supply only one unordered within-class positive pair, and three papillary keys supply three. Numerous field-pair combinations do not create additional independent positive cases.

UNI2-h has the strongest observed first-arrangement specialist result, but its conditional malignant accuracy is 72.00% / 60.67%. Its 813 ductal/mucinous confusion and weak papillary recognition remain substantial. An ideal router holding those existing conditional predictions fixed reaches only 86.00% / 80.33% final accuracy. That arithmetic motivates a specialist intervention but does not bound a newly learned expert. The accepted [reader diagnosis](APPROVED_ENCODER_ACTUAL_MODEL_DIAGNOSIS.md) supplies the class/error decomposition.

The [case-MIL diagnosis](CASE_MIL_ACTUAL_MODEL_DIAGNOSIS.md) finds about 99.1% fitting accuracy and .982 fitting F1, active gradients and complete case visits, followed by much weaker validation subtype recognition. The [LoRA diagnosis](LORA_ACTUAL_RESULTS_DIAGNOSIS.md) also reports 100% rare-type fitting recall with poor held-case transfer. These support a generalization problem in those trained packages. They do not identify stain, morphology, acquisition, annotation or clinical identity as its cause.

KDE's leave-case-out fitting values are a different diagnostic: UNI2-h's .4765 / .4981 final-five fitting F1 excludes the whole query case from support and center. They are not comparable to neural resubstitution fitting F1 as if both measured the same generalization gap.

## Ordinary entry-point audit

No actionable implementation defect was found in the inspected ordinary approved-reader prediction path:

- `load_inputs` binds the accepted extraction/feature hashes, dimension, exact 7,909-image membership, original case order and registered roles before the kernel runner scores anything.
- `class_weights` gives every head-class unit total field mass. A case contributes equal total mass within its class, divided over its same-view fields; ductal image count therefore does not directly overwhelm each rare-class density.
- `_centers` uses equal support-case raw means, independently for each view. `prepare_case` removes a fitting query from both support and centering and rejects validation queries in support.
- Similarities compare fields only within the same magnification. No equal field-number correspondence across views is assumed.
- Kernel computation stabilizes exponentials and normalizes each head separately. All-zero geometry gives uniform probabilities rather than NaN.
- Dual field probabilities are `[1-r, r*q_DC, r*q_LC, r*q_MC, r*q_PC]`. `pool_log_probabilities` averages that joint vector over fields and available views, then routes at .5 and derives conditional subtype probabilities.

That last operation makes case-level subtype evidence proportional to each field's predicted malignancy. In symbols, pooled conditional type probability is `mean(r*q_k) / mean(r)`, with the declared within-view/equal-view means. It is intentionally different from independently averaging `q_k`. The source has an explicit fixture demonstrating that difference. This is a modeling choice worth controlling if changed, not an overlooked multiplication bug. Better direct-five outcomes also do not isolate a routing defect: priors/group densities, final rule and selected temperature differ together.

Existing source fixtures cover whole-case exclusion, support/field order, equal-case centering, same-view behavior, zero/extreme features, missing classes, joint-before-pooling and the .5 tie. This audit read those fixtures but did not rerun the suite or claim new numerical verification.

## Recommended bounded study

Use the accepted frozen UNI2-h fields and the existing centered-cosine KDE, learning **only one shared positive diagonal metric for the malignant expert**, strongly constrained around identity. Do not train the reader, introduce a projection/attention pool, combine readers or replace the router in the same experiment. The aim is to test whether a modest task-specific reweighting can improve cross-case malignant geometry while retaining the effective nonparametric classifier.

Before any production fit, registration must fix the following details; these are design requirements, not a completed registration:

1. One global diagonal shared across magnifications, identity initialization, a bounded positive parameterization, an explicit identity penalty, deterministic optimizer/stopping rule and one fixed final checkpoint. A full learned matrix would be unnecessary capacity here. Centering and the order of diagonal transformation/L2 normalization must be identical in training, scoring, serialization and replay.
2. A class-balanced **cross-case** malignant objective. If using positive-pair or neighbor classification loss, remove every field from the queried case from its positive and negative support. Give cases/classes explicit mass; do not permit ductal's many pairs to dominate the loss or treat same-case fields as independent positive-case evidence. Fix the training temperature independently of validation outcomes.
3. An exact identity-metric control through the same new executable path, using the same fields, centers, class/case/field weights, router, pooling, four inference temperatures, masks and numerical dtype. At every equal temperature its probabilities should replay the accepted UNI2-h density calculation within a predeclared tolerance. This is a necessary mechanism control, not another benchmark family.
4. All original ten roles and 15 subsets; save every candidate, including losses, metric bounds/distance from identity, convergence/gradient receipts and reload probabilities. Report original UNI2-h, identity control and learned metric separately, plus contextual C/F/P. Disclose the combined family selection budget. Do not select a favorable reader/arrangement combination or add a fallback search after an unfavorable result.

Two easy-to-miss distinctions need to be explicit in the registration and report:

**Learned-metric fitting scores are no longer ordinary KDE leave-case-out scores.** If a fitting case's label helped learn the diagonal, removing it only from density support/centering does not make its prediction fully held-case-out. Either refit the metric without each queried case for a genuinely cross-fitted diagnostic, or call the reported number an in-sample learned-metric diagnostic with query-omitted density support. Validation keys must never affect the metric, support or center. The cross-case training objective is not itself independent generalization evidence.

**An unchanged router per temperature need not mean unchanged selected router decisions.** If control and learned candidates separately select among the original four shared temperatures, their binary probabilities match at each equal temperature, but different selected temperatures can change their final routing. Report this fact and its error decomposition. Alternatively freeze the originally accepted UNI2-h router and vary only expert temperature; that requires a matched identity control with the same newly decoupled expert selection policy. Do not silently compare those two policies or describe a selection-induced router difference as metric-only typing improvement.

A negative or mixed result closes this fixed study. It does not authorize another regularization range, checkpoint schedule, resplit, class rule, threshold adjustment or automatically combined predictor. No present evidence justifies a forecast of 94–95%.

## Fast conventional alternative, if a classifier rather than metric is selected

A **single class/case/view-balanced L2 multinomial logistic UNI2-h specialist**, retaining the accepted UNI2-h KDE router and field joint-probability pooling, is the fastest conventional trained-probability alternative to implement from existing components. Its objective must keep total malignant sample-weight mass at 35, as in the reviewed field-head formulation, so C retains a meaningful scale. A predeclared four-C budget can replace the four-temperature expert budget; do not multiply them into a hidden 16-setting search. Its comparison to KDE is a classifier-package comparison, not a geometry-only ablation. An untrained/identity replay control cannot isolate the effect of optimized logistic coefficients because it changes the classifier family.

There is no repository evidence that SVM/RBF or boosted trees on the new UNI2-h features have already failed. Nevertheless, a naive probability-enabled SVM introduces a specific additional issue: [sklearn 1.6.1 SVC](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html) fits probability estimates using internal five-fold cross-validation. At field level, default row splitting can put fields of one case on both sides; at case level, two lobular keys cannot supply five distinct class-stratified held-case folds. This is a reason to explicitly design grouped fitting/calibration, not a claim that SVM classification cannot work. [L2 multinomial logistic regression](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html) supplies probabilities directly and avoids that extra probability-calibration fit.

The [official UNI repository](https://github.com/mahmoodlab/UNI) and the [original UNI study](https://www.nature.com/articles/s41591-024-02857-3) establish linear/few-shot evaluation as conventional ways to read frozen pathology representations. They do not establish which readout wins for this five-category, provider-case, 15-subset endpoint. The current next-step choice should therefore remain one fixed controlled question, not a claim of novelty or evidence that the reader is already optimally trained.

## Scope of conclusion

The rare-class support and reused validation remain the hard limitations. All 82 qualified provider keys have entered development somewhere; the 600 selected records per arrangement are dependent repetitions, and clinical identity remains unresolved. A successful new learner would be another exploratory development result. It would not supply an untouched clinical cohort, establish novelty or make the capstone presentation/demo complete.
