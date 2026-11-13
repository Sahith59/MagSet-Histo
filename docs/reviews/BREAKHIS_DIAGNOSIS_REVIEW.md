# BreakHis saved-artifact diagnosis: independent review

Prepared November 13, 2026. This review concerns a fixed diagnosis of the ten completed adaptation runs and their frozen references. It does not authorize new fitting, images, outer-role predictions, checkpoint selection or secondary-dataset work. Reviewer changes are confined to this file.

## Design assessment

The proposed analysis is useful and appropriately bounded: replay the existing frozen/adapted case-head validation records first; then examine fitting-versus-validation behavior of those heads, the saved auxiliary image head, two fixed auxiliary case readouts, feature changes and subtype confusions. All outputs must retain every trial, both inner splits and all 15 subsets where a case prediction is involved. This can locate descriptive inconsistencies between tasks/readouts and expose fit–validation gaps. It cannot establish one causal reason for poor generalization or turn a diagnostic readout into a newly selected model.

The trusted scope is exactly 46 fitting and eight validation provider cases per run. The 28 own-run outer cases remain excluded. Hash-bind the complete previously accepted paired comparison, trusted manifest/protocol, immutable training source/configuration, original frozen reference, adapted checkpoint/cache and each saved head. Reject incomplete matrices, wrong trial/split cache reuse, label/membership changes, duplicate/missing paths or nonfinite values before diagnosis.

## Required weighting and endpoint definitions

### Canonical auxiliary field evaluation

For each role separately, compute a deterministic expectation under:

`P(class)=1/8; P(case|class)=1/n_cases(class); P(view|case)=1/4; P(field|case,view)=1/n_fields(case,view)`.

Thus each observed field receives weight `1/[8*n_cases(class)*4*n_fields(case,view)]`. Weights must sum to one within the role. Report weighted accuracy and cross-entropy without another inverse-class multiplier. This matches the intended adaptation sampling distribution in expectation and prevents field-rich cases/views or the 25 fitting ductal cases from dominating the result. The eight validation cases have one case per class, but their field counts still differ.

Use the saved auxiliary head's raw adapted CLS features. Its target was the case-derived subtype. Do not feed it the case classifier's standardized features or imply independent diagnostic region/cell annotations. Dropout is disabled; applying saved affine coefficients directly is equivalent to the original head in evaluation mode. Validate the exact two saved head tensors, canonical output order, shapes, dtype and finite values.

Use stable log-softmax/log-sum-exp cross-entropy, rather than clipping zero probabilities or taking unstable exponentials. Canonical unaugmented/dropout-off fitting CE is a new diagnostic evaluation of the saved head; it is different from the stochastic augmented/dropout-on minibatch loss logged during training. Label these quantities separately.

### Saved case classifiers

Each selected logistic classifier should be evaluated on its own fitting and validation cases using its saved normalization, coefficients and availability-mask vectorizer. Select no new C, threshold, epoch, calibration transform or head. Validation probabilities must reproduce the accepted archive before proceeding to any diagnostic calculation.

Case fitting sets are class imbalanced. Report raw case accuracy together with macro F1 and balanced accuracy. If fitting-versus-validation CE is reported, state whether it is ordinary case-uniform or class-balanced; providing class-balanced CE makes the role distributions comparable. Report the same defined weighting on both roles. Unregularized evaluation CE is not the full penalized logistic training objective.

Report each subset separately and equal-weight averages over all 15 subsets. Fitting data contain 46 original cases, not 690 independent cases after masking. Validation contains eight original cases, not 120 independent cases. Confusion matrices must identify whether they contain ordinary counts, weighted mass, or conditional class proportions.

## Linear pooling commutation is a required control

With dropout off, the auxiliary head is affine: `h(z)=Wz+b`. Equal-field averaging within a view, followed by equal-view averaging over an observed subset, obeys:

`mean_views(mean_fields(h(z))) = W*mean_views(mean_fields(z)) + b`.

Test this identity numerically with unequal field counts and the same dtype/weights on both sides. A linear mean-logit readout therefore uses exactly the information in that mean feature vector under the same head. It is not an independent nonlinear pooling mechanism. A difference between auxiliary mean-logit predictions and the selected logistic mean head can arise from their objectives, parameters, normalization or mask terms; it does not prove that early averaging discarded information that the linear auxiliary readout recovered.

The second fixed readout averages field **probabilities**, using equal field mass within each view and equal mass across available views. Softmax is nonlinear, so mean probabilities need not equal softmax of mean logits. Both readouts must use the same fields, subsets and class order, and both must be reported for all ten runs. A favorable mean-probability diagnostic is a hypothesis-generating observation, not a fair newly trained-method comparison or permission to adopt the better readout post hoc.

Avoid a flat mean over all observed fields: that changes the view weights when field counts differ. No sequence-number pairing across magnifications is required or permitted by these calculations.

## Feature-change diagnostics

Match each frozen/adapted field by its exact original path within the allowed fitting/validation list. Compare raw CLS vectors and, separately, vectors standardized by each saved head's own fitting-only per-view mean/std. Keep these interpretations separate: different fitted standardizers can absorb global shifts/scales, so standardized similarity does not prove that raw features stayed unchanged.

Define relative L2 with the frozen vector norm as denominator and specify the treatment of zero/near-zero vectors. Cosine similarly needs an explicit norm guard. Reject or explicitly count undefined/nonfinite comparisons; do not silently turn them into perfect agreement. Do not append availability-mask coordinates to these field-vector comparisons.

Aggregate field changes using a stated case/class/view weighting, ideally the same role-specific distribution as auxiliary field metrics. A simple image-uniform mean or unweighted quantile answers a different question and may overweight field-rich cases; label it if retained. High cosine or small L2 is descriptive geometry, not proof of no learning, no biological change, or an explanation of the classification outcome.

## Interpretation limits and required controls

- A high fitting score and poor validation score is evidence of a fit–validation gap under these roles. It is compatible with limited generalization, weak field labels, case variation and selection effects; it does not identify one cause by itself.
- The auxiliary image objective and logistic case objective differ. Their losses and accuracies cannot be compared as if only pooling changed. The two fixed auxiliary readouts help describe this mismatch but do not isolate a causal remedy.
- Both adenosis and phyllodes have one fitting provider case per run. No number of field-level predictions or masks adds independent subtype examples. The 31 distinct validation cases across the two arrangements recur across trials.
- Existing validation already selected C and influenced the wider development program. Report all readouts and negative results; do not claim fresh confirmation, significance, calibrated clinical performance or novelty from a diagnostic winner.
- Published binary/image/patient benchmarks remain a separate compatibility question. This analysis cannot explain a gap to a paper with different cases, prediction units, labels or model-selection rules.

Tests should cover exact forbidden-role/path rejection, archive replay failure, checkpoint/cache/head/hash binding, role-specific weighting, unequal-field-count view weighting, affine-pooling commutation, nonlinear mean-probability differences, stable extreme-logit CE, zero-vector feature behavior, canonical class order, missing/permutation invariances, and saved aggregate arithmetic. The implementation must contain no optimizer, fit call, raw-image decoding or new encoder execution. Aggregate public reports should exclude private provider identifiers.

**Current status:** the fixed no-refit design is accepted with these controls. Implementation and measured diagnostic outputs remain pending review. No new scoring, fitting, images or cluster work has been performed by this reviewer.

## Specification and first source review

The registered specification/configuration fixes the accepted comparison hash, both roles, all ten runs, all four readouts, expected field weighting, Torch auxiliary-head parity and affine identity tolerance. It distinguishes raw-field drift from drift of each saved head's own standardized per-view case means. Those are different units, and are accepted when labeled separately. Near-zero norms at or below `1e-12` are excluded from the corresponding geometry mean; excluded mass and null results when nothing remains are explicit.

Source inspection of `magset/diagnose_breakhis.py` confirms fit/validation-only case construction, unchanged saved normalization/heads, raw auxiliary inputs, class/case/view field weighting and equal-view auxiliary pooling. It binds the exact accepted comparison and training snapshot, rechecks the existing adaptation-specific artifact gates, and replays both archived validation case heads before auxiliary diagnosis. The saved Torch auxiliary head is replayed with dropout disabled; no new encoder is instantiated or executed. Existing normalizer arithmetic is recomputed only by the imported artifact-verification oracle to check equality with saved arrays; diagnostic predictions use the saved normalizers and no new fitted component is created.

All six initial focused tests pass independently. They check unequal field counts, affine commutation, a real decision difference between logit/probability pooling, excluded-view and field-order invariance, stable field loss, zero-norm behavior, forbidden outer roles, archive tampering and incomplete case coverage. The saved-readout test prevents sklearn fitting and AdamW construction during its inference path. These are synthetic software checks, not additional research-performance evidence.

**One loss-reporting defect was reproduced before execution acceptance.** Field CE correctly receives stable log probabilities, but the initial case-summary path used only probabilities and replaced zeros with `float64.tiny`. A synthetic true-class logit loss of 1,000 was consequently reported as approximately 708.396. This matters because class-balanced case CE is included in the top-line diagnosis. Propagate stable log probabilities through all case readouts, including log-mean-exp for probability averaging, or omit/reject/explicitly label clipped CE where underflow occurs. Do not label a clipped finite value as exact stable CE. No real fitting, new image processing or outer scoring was used to demonstrate this defect.

At this stage the split/provenance/readout design is accepted, while final loss handling and actual saved diagnostic outputs remain pending review.

## Stable-loss repair and execution gate

Committed diagnostic source `23a8132cca1d2068ae76eee5f6812234ab235cd0` repairs the case-loss defect. Saved logistic logits are reconstructed with the original float32 pooling/normalization/vectorizer and checked against its probability scorer. Auxiliary logit pooling retains log-softmax values, and auxiliary probability pooling retains a stable log-mean-exp across fields and views. Every case record carries finite log probabilities; case summaries use those values for both ordinary and class-balanced CE. Probability-only input with a zero true-class probability is rejected rather than given a falsely capped loss.

All seven focused tests pass independently, including a new extreme-logit fixture requiring CE 1,000 for each of the four case readouts. The source, tests, specification and configuration match the committed Git objects. The main agent reports the full 77-test suite passing; that broader suite was not independently rerun by this reviewer.

| Reviewed artifact | SHA-256 |
| --- | --- |
| Diagnostic module | `f470dd583685dfd6936c3c502627d73055db585daa4c8f60c8553d0eb37d5af6` |
| Focused tests | `688752d15dc30210c445544f8bd1bb3e145aea3b6b08ce18650e74b3348e17d3` |
| Final registered specification | `67bedc8feceebbd5ca59d3bb85e2174a2cceeaa15bf4c91c45469266e7a34606` |
| Fixed configuration | `38ee5d7177e2a2b0b468abb6ce73f66298f3bfb3a49c51a145f2f71ad34c77df` |

**Execution acceptance:** no remaining blocker was found for one bounded two-CPU/12-GB job applying these fixed probes to the ten existing fitting/validation artifact sets. The job performs no fitting, optimizer updates, encoder execution or raw-image processing. Its source/configuration/comparison and per-run artifact bindings, early archive replay, forbidden-role checks, exact weighting, pooling identities and stable-loss path are accepted. Actual output acceptance remains pending the saved complete diagnostic artifacts.

Private diagnostic JSON contains case assignments, including excluded-role metadata needed for provenance. Public evidence must recursively remove provider identifiers and field paths, retaining only aggregate scores/counts and hashes. The failed primary adaptation gate and frozen-baseline choice remain unchanged by any diagnostic readout score.
