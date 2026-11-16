# Independent review of the next frozen-reader and case-supervision lanes

Reviewed November 16, 2026, before technical GPU checks or new cohort scores. Read-only examination of `case_mil.py`, its trainer/collector interfaces, `genbio_five.py`, `genbio_collect.py`, registrations and associated fixtures. No source/config edits, real fitting, encoder construction or jobs were performed. The independent arithmetic checks below use synthetic arrays only.

**Assessment: no implementation or benchmark-lineage blocker found in the reviewed version.** Proceed through the registered full-suite and immutable technical checks. Successful source review does not establish that either hypothesis will improve five-category validation.

## Whole-case MIL mathematics

The core projects actual fields before pooling; padded values never enter the projection, and `forward(..., subset=...)` hides requested-out views before any learned operations. The gated score implements the registered tanh/sigmoid product with softmax only over observed fields in one view. An absent view avoids an all-negative-infinity softmax and contributes zero mass. Available views receive equal weight, with canonical four-position availability flags appended. This assumes no numbered-field correspondence between magnifications.

Mean and attention models construct shared projection/head parameters before attention parameters. The trainer resets Torch/CUDA randomness after construction, so extra attention initialization does not shift the registered paired training dropout stream. All15 subset losses share one projected-field/dropout draw. Separate evaluation uses deterministic pooling; training dropout does not imply exact realization-by-realization field-permutation invariance.

The loss averages the15 subset terms per case before weighting. Direct weights are46/(5*n_class); binary weights are46/(2*n_broad); malignant expert weights are46/(4*n_type), with true-malignant labels shifted from1..4 to0..3 for cross-entropy. The last two cases retain denominator4. Expert loss is disconnected from malignant logits in an all-benign batch, rather than multiplied by zero; with `zero_grad(set_to_none=True)`, AdamW consequently skips those expert parameters. Shared projection parameters still receive the benign binary loss, as intended.

Independent closed-form checks used all46 synthetic labels with support11/25/2/5/3 and zero logits. Summed fixed-batch losses matched `(46/4)*log(5)` for direct and `(46/8)*(log(2)+log(4))` for dual: absolute errors0 and1.78e-15. A uniform-attention fixture produced exactly the matched mean output in both arms. The benign-only expert-gradient oracle also passed. These checks validate arithmetic, not learned performance.

## Saved predictions and fit diagnostics

The MIL direct output uses five-way argmax; its binary diagnostic sums malignant probability. Dual output composes `[1-r,r*q4]` but uses the specified hard `r>=.5` gate for final classification. An independent zero-logit fixture verified the distinction: direct predicts benign with derived malignant mass.8, while dual has malignant mass.5 and predicts the first malignant class under the tie rule. This difference reflects the two heads' defined priors; it is not a class-map or saved-prediction defect.

Candidate inference uses CPUFP64 arithmetic on the original fitting-normalizedFP32 fields and savedFP32 learned parameters. The collector separately implements affine/ReLU/gated attention/view pooling and heads in NumPyFP64. This protects near-boundary saved-label replay; it does not promise bitwise identity to GPUFP32 training inference. Fitting neural predictions are in-sample diagnostics, whereas KDE fitting predictions exclude the query case from support and centering. The collector labels this difference; do not compare their training accuracies as equivalent generalization estimates.

The mean-versus-attention comparison isolates pooling within the same case-supervised model. Comparing either with field KDE also changes projection, objective and readout. A gain cannot alone prove that inherited field labels were the unique cause of earlier errors.

## GenBio cache delegation and reference integrity

The controller first verifies the original accepted Midnight inputs solely to recover trusted cases/labels/roles. It then requires the new bound extraction config, accepted full GenBio audit and four actual GenBio NPZ/JSON byte bindings. Individual shard verification checks the4608-dimensional RGB-channel contract, precision/provenance and ordered original paths. Case membership must equal the original trusted ordering. It explicitly passes the new bags into `kernel.run(inputs=...)`; no default old-reader fallback occurs.

The unbound feature producer/audit/cache registration intentionally rejects scoring. Actual audit bytes and current producer source must be bound after extraction. The collector replays all80 candidates/160 archives/64,800 records with unchanged kernel mathematics, query-case exclusion, field-joint pooling and hard routing. Original role membership and each saved prediction are checked before publishing aggregate results.

Primary reference is **C**, secondary **F**. Independent calls to the reference helper against the actual accepted public artifacts passed and returned the expected single-center and three-crop encoder descriptions. The six split/weight/pooling method-file hashes must match both references, while the encoder/cache may change. One nonblocking stale module docstring says native/primary-three-crop; actual configuration, comparison keys and gate scope correctly identify GenBio/C/F. No scientific result is mislabeled by that string.

GenBio feature production is7,909 image-level outputs from23,727 shared-backbone channel examples. CPU classification/collection adds no encoder forwards. Whole-image square resizing and the official channel-concatenated feature definition are part of this reader package, so comparison with C is not an architecture-only causal test. The separate MIL lane uses F features and does not inherit GenBio preprocessing or P centers.

Four candidate checkpoints/temperatures are available per named arm/mode; choosing among pooling modes afterward inherits eight opportunities. The repeated roles/subsets still reuse31 validation provider keys, not new independent patients. Retain unsuccessful results, distinguish binary routing from the final five labels, and preserve the registered improvement screen. There is no source-backed94–95% prediction from either lane.
