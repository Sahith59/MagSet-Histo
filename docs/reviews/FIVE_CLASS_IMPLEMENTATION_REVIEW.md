# Independent five-class implementation review

Reviewed November 15, 2026, before production outcomes. Scope: the registered [specification](../BREAKHIS_FIVE_CLASS_SPEC.md), [configuration](../../configs/breakhis-five-class-v1.json), [runner](../../magset/five_class.py), [metrics](../../magset/five_metrics.py), [collector](../../magset/five_collect.py), and their existing data, splitting, vectorization and normalization dependencies.

**Decision: accept the reviewed implementation for the bounded registered comparison, conditional on the complete runner/collector fixture suite, immutable launch snapshot and production zero-fit input audit. No blocking source or numerical-contract defect was found. Actual-output acceptance remains pending.** This is implementation acceptance, not a five-category performance result.

| Reviewed artifact | SHA-256 |
| --- | --- |
| Configuration | `d3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b` |
| Specification at this review | `aad9912d77ed0e3568545ad738d2892fd598f4b80b10ed2fe639e5cc7a2e13c5` |
| Runner | `99416f28087304199290059d08a5d0ab87e071c01d73d1d80020f940991f0574` |
| Metrics | `50738aefd433a0c0b54b55fe1065e20883c53df31942ea47ef908d97956756a7` |
| Collector | `01b2750d96e4644d7afd1be4d75c6f3b88dd035484e1bc19e1f255d95be5718f` |

The launch commit/archive is finalized separately after the remaining fixture work. These byte hashes identify the source inspected here; later amendments require review of their differences rather than silently inheriting this binding.

## Checked implementation contracts

- The loader binds exact configuration, manifest, protocol, encoder revision, all four feature NPZ/JSON hashes and numerical dependency versions. It reconstructs the original eight-label memberships before pooling targets, checks all ten role assignments and supports, and preserves 46 fitting / 8 validation / 28 excluded cases.
- Both arms and both hierarchy heads use the same fitting-only, equal-case field-moments normalizer and the same 1,028-input available-view mean plus flags. No encoder training, stain fitting, crop search or feature selection is present. The runner fits only the fitting rows. The malignant head receives only true-malignant fitting cases.
- The registered four C values are completed in order for both arms. The hierarchy shares one C between its two heads. Each head uses balanced class weights multiplied by subset weight 1/15, with convergence warnings rejected. The budget is 80 complete candidate pipelines and 120 individual head fits across ten runs, with no separate technical real-data fit.
- Each arm selects the earliest maximum of validation **end-to-end five-class mean-15 macro F1**. No independent router/expert grid, threshold tuning, excluded-case selection or partial-matrix winner is present. The two arms have equal declared selection opportunities, while their head counts, capacity and factorized objectives differ as disclosed.
- Binary reload correctly handles one coefficient row and two probability columns. Direct and malignant multinomial heads have five and four rows in explicit canonical orders. Stable log-sigmoid/log-softmax arithmetic retains finite log losses when ordinary probabilities underflow.
- The dual final label follows the fixed malignant probability ≥0.5 gate and then malignant argmax. Its global composed-probability argmax is not substituted for that label. Classification metrics and C selection consume explicit predictions; CE/Brier consume the coherent probability vector. Direct binary/conditional malignant readouts remain diagnostics.
- Metrics retain five-class raw accuracy, balanced accuracy, macro F1, all-five confusion/recall, stable CE/Brier, binary routing and true-malignant typing separately. Per-subset F1 is averaged rather than replaced by F1 from pooled repeated events. Validation support is 4/1/1/1/1, so accuracy and balanced accuracy differ.
- The collector checks the complete ten-run, two-arm, four-candidate matrix; original roles/truths; hashes; fitting-only normalizers; serialized coefficient replay; hard decisions; every saved metric; and earliest-maximum selection. Unexpected model/prediction archives, incomplete coverage and private identifiers in public aggregates are rejected.

The collector explicitly distinguishes the minimum across trial/subset cells from the mean of each trial's worst-subset F1. It also distinguishes the 120 production head fits from the collector's zero new fits. These reviewed metadata clarifications change no scientific setting or outcome.

## Independent zero-fit numerical preparation

A separate private reviewer program was written without importing any `magset` implementation or metric helper. It reconstructs qualified identities and original role assignments directly from the manifest/protocol; verifies accepted cache bytes and exact per-view path coverage; rebuilds case means and all ten fitting-only normalizers; and uses its own NumPy/SciPy calculations plus scikit-learn metric functions.

Executed under Python 3.11 with NumPy 1.26.4, SciPy 1.15.2 and scikit-learn 1.6.1, this independent **actual-input reconstruction passed 15,897 assertions**. It performed no classifier fitting or encoder forward. Its synthetic reviewer checks also passed for complete support, duplicate rejection, hard-route/global-argmax disagreement, fixed ties, stable extreme-score loss and binary coefficient replay. Synthetic checks establish software behavior only.

The prepared actual-output reviewer will independently inspect all **80 candidate model archives, 120 head arrays and 64,800 fitting/validation case-subset records**. It checks immutable source/archive contents, configuration/cache/data hashes, exact labels and roles, normalizers, coefficients, probability/log-probability replay, explicit decisions, every metric, candidate selection and the two-arrangement comparison gate. It also recomputes the prescribed weighted training objective and gradient descriptively without refitting. This output review has **not yet run** because production outputs are not yet available.

Acceptance after execution requires the complete registered matrix and both the production collector and independent actual-output review. A technical failure does not authorize changing scientific settings or widening the budget. An unfavorable result remains reportable. Provider-case identity uncertainty, 31 distinct reused validation cases and the absence of a globally fresh cohort remain unchanged. No 94–95% claim, clinical validity or causal hierarchy-only improvement follows from this review.

This review performed no classifier fits, encoder forwards, raw-image inference, outer-role predictions, resplits or scheduler submissions. Public changes by this reviewer are confined to this document; the independent reviewer and its intermediate evidence remain private.
