# Separate training reference centers: a fixed geometry ablation

Registered November16,2026, before new predictions. Hypothesis: a reference center balanced for a decision's classes may improve cosine comparisons on this imbalanced cohort. The original case-balanced center is mathematically valid. Uniform density priors do not require a matching center; this is supervised preprocessing, not a prior correction, bug fix or certified novel mechanism.

Use the accepted three-crop fields and unchangedKDE density weights, head priors, temperature grid, probability pooling and original provider roles. The shared case mean currently weights class centroids by11/25/2/5/3 fitting counts. That geometry may help or hurt particular classes afterL2 normalization. Equal class centering may also amplify noise from two lobular and three papillary fitting cases.

## Two named alternatives

Calculate raw field means within each support case, then equal case means within each head class, then an equal mean over that head's classes, separately for each zoom. Use float64 and the unchanged L2 epsilon1e-12.

- **class_shared:** the equal mean of five final class centroids is used for direct, binary and malignant-four comparisons.
- **head_specific:** direct uses the same five-class center; binary uses equal benign-case and malignant-case group means; the malignant expert uses equal means of its four type centroids, excluding benign cases.

These moments equal the population means under each head's existing class/case/field weighting, but this identity does not imply better classification. No density prior, support weight, field budget or coefficient optimization changes. The direct control is numerically identical across the two modes by construction.

For fitting diagnostics, remove the entire query case before all centers, weights and counts. Validation/excluded labels or features must not enter centers. Changing query truth cannot change predictor output. In head_specific mode, changing benign support features cannot change malignant expert centering/probabilities; class_shared intentionally does not have that property. Both properties have independent scalar checks.

Each named mode/arm selects among the same four shared temperatures.01/.03/.1/.3 by earliest maximum end-to-end mean15-subset macroF1. There are eight logical mode-temperature slots per arm if choosing between modes afterward. Direct controls repeat, leaving four structurally unique direct choices and eight unique dual choices. Report both modes, including negative outcomes.

Ten paired role jobs produce160 logical candidate definitions,120 structurally nonduplicated definitions,320 logical prediction archives and129,600 logical records. Removing the intentionally repeated direct controls leaves240 archive definitions and97,200 records; these remain dependent repeated-case outputs, not independent patients. Sixty logical full-fit head definitions represent50 structural definitions. Numerical outcomes/hashes can coincide beyond these known duplicates.

## Verification and resource budget

Bind the original three-crop cache/source/audit/configuration and both accepted references before execution. Audit all ten role memberships and actual full-support center hashes. Save per-mode/arm/candidate private support metadata, center-use hashes, train-LOO/validation predictions and metrics. Verify the complete matrix, scalar density math, earliest selection, unchanged original split/metric/pool source hashes and exact duplicated direct controls. Classifier coefficient fits and encoder forwards are zero; fitting labels determine centers/support distributions.

Primary comparison is three-crop75.50%/76.83%, F1.5925/.5799; secondary is single-center75.67%/75.00%, F1.5897/.5643. Four CPUcores/16GB/30minutes per paired role, at most four concurrent,45minutes for full collection. Three cosine preparations per query increase CPU work; no new GPU work is needed.

[Canonical registration](../configs/breakhis-prior-centered-v1.json) · [Feasibility critique](reviews/CONDITIONAL_CENTERING_FEASIBILITY.md) · [Independent implementation review](reviews/PRIOR_CENTERED_IMPLEMENTATION_REVIEW.md). Only BreakHis is fitted/evaluated;46/8/28 provider roles,31 reused validation keys and unresolved clinical identity remain unchanged. No fresh clinical test,95% result or novelty is established by this registration.
