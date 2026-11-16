# Two-mode supervised centering: independent implementation review

November 16, 2026. Registration SHA256 `d6b770641c6cae6e336ce8e50f822fd176fc965256a8793dd48a9fbd284eceab`. Scope: `prior_centered_model.py`, coordinator and collector, before actual cohort evaluation. No cohort predictions, fitted coefficients, encoder execution, model construction or scheduler jobs were produced by this review.

**Verdict:** no remaining mathematical or implementation blocker was found in the reviewed path. Independent synthetic scalar oracles agree with the implemented centers, density weights and joint probabilities. Actual source/input audit and complete-matrix acceptance remain separate prerequisites; this is not a performance result.

## Center and density correctness

The core computes float64 raw field means within each eligible support case, then case means within each head class, then an equal mean over head classes. `class_shared` uses the five-class center for every head. `head_specific` uses that same center for direct-five, an equal benign/all-malignant group center for routing, and four equally weighted malignant-type centers for the expert. The binary malignant group retains its actual case mixture; it is not silently rebalanced over subtypes.

An unequal-class/unequal-field-count synthetic fixture has exact five-class, binary and expert centers `[.4,.2]`, `[1.9,.2]`, `[.25,.25]`. Independently computed nested class/case/field log-sum-exp densities were compared with 96 mode/head/view/temperature outputs; maximum log-probability error was **2.84e-14**. Joint-five log composition matched exactly. These are tiny arithmetic checks, not reproduced research metrics.

The new core wraps three center-specific instances of the old prepared-case structure and reuses the unchanged normalization, class weights and density implementation. It changes only the origin before L2 normalization. Density classes retain unit total support mass; fields and cases receive the same weights as before. EPS and the four temperatures are unchanged. Different head geometries still form a coherent joint distribution `[1-r, r*q4]`; they do not require a shared metric to define valid two-step probabilities.

## Leakage and invariance

Whole fitting-query exclusion happens before every center and support count. Synthetic leave-case-out centers matched an independently recomputed support set, and removing the sole remaining case of a class was rejected. Changing a validation query's true label, or the excluded fitting query's true label, left all prediction arrays **exactly unchanged**. Query diagnosis therefore does not select its normalizer.

Changing benign support features left the conditional expert center exactly unchanged and its probabilities equal within floating-point tolerance. The same perturbation changed the shared-center expert, as intended: shared mu5 includes benign geometry. Requiring benign invariance in both modes would incorrectly remove the control intervention. The direct outputs in both modes are the same cached arrays and match exactly; density caches are bound to the particular query bundle and temperature.

## Coordinator, references and duplicate accounting

The coordinator loads only accepted three-crop F features through the bound `a18a4bfe…` coverage registration. Native-input artifacts or future native outcomes do not select this study's inputs. Original image/provider roles, labels, magnifications and all 15 subsets remain inherited. The audit binds source/cache/config provenance and the actual support-center summaries for ten roles. Models record the center hashes selected by their declared mode/head mapping.

The collector reconstructs every candidate from trusted inputs, verifies model/prediction hashes, metrics and earliest maximum-F1 temperature selection, and rejects missing/extra matrices. It explicitly requires duplicated direct prediction-file hashes, fitting/validation summaries and selections to agree across modes. Primary reference is F `39d15d94…`; C `e6e98fc…` is secondary. Six unchanged data/split/metric/kernel/pooling hashes match the accepted reference.

There are **160 logical / 120 structurally unique candidate definitions**, **320 logical / 240 structurally nonduplicated prediction archives**, and **129,600 / 97,200 records** under those respective counts. Full-support head definitions are 60 logical / 50 structurally unique. These counts remove duplicated direct controls; they are neither independent observations nor guarantees that all remaining numerical outcomes have distinct hashes.

Each named mode has four temperature opportunities. Choosing between the modes after examining results inherits eight opportunities per arm. The original case-balanced center remains a valid baseline; uniform density priors never required class-balanced centering. This supervised geometry may amplify sparse-class centroid noise or near-center nuisance variation. No optimized coefficients does not mean label-free normalization, novelty, reliable clinical generalization or a 95% guarantee.
