# Centered-cosine support classifier study

Registered before scoring this method. The completed Midnight case-level two-step model scored72.00% and71.33% accuracy, with macro F1.555818/.534982. The field-level variant scored73.67%/70.00%. These methods improved the earlier reference's accuracy, while malignant subtype recognition remains weak. This next study uses the same frozen3,072-dimensional Midnight descriptors and original46 fitting/eight validation/28 excluded case roles.

For each magnification, compute the raw mean field vector for every support case and average those case means equally. Subtract that support-only center, then divide each centered field by max(L2norm,1e-12) in float64. Compare a query image only with fitting images at the same magnification. The class density is the weighted mean of exp(cosine_similarity/temperature), with equal total mass for each class, equal case mass within its class and equal field mass within a case/view. Four fixed temperatures[.01,.03,.1,.3] are shared across the two hierarchy heads.

The direct arm forms a five-class posterior. The two-step arm forms a benign/malignant posterior from all fitting cases and a four-type posterior from malignant fitting cases. Uniform priors apply within each head. Compose each field's joint five-class probabilities, average within each view, then average available views. Case malignant probability at least0.5 routes to the malignant subtype with largest conditional probability. All15nonempty view combinations are scored.

The fitting diagnostic removes the entire query case from both support and centering. Validation queries use the46fitting cases, with validation/excluded cases absent from support and fitted statistics. Each fitting class has at least two cases, so every class remains represented after a query case is removed. This leave-case-out diagnostic differs from the self-included fitting scores of the earlier optimized classifiers.

There are ten paired runs,80candidate pipeline evaluations and four selection opportunities per arm. The same support data define30base full-fit head distributions across the ten roles, with temperature variants and additional leave-case-out diagnostic evaluations. Training labels define these distributions. The method optimizes zero classifier coefficients and performs zero new encoder forwards. It is a standard few-shot baseline.

Select the earliest maximum validation mean15macroF1. Verify the full matrix, support descriptors, coefficient-free replay, original cases/labels, all15probability/label metrics and selected temperatures before publication. Compare with the fixed Midnight case-level two-step reference. The descriptive improvement screen uses+.02macroF1, no lower accuracy and positive recall for all five labels on both arrangements. All outcomes and both arms are reported.

Allocations are four CPU cores,16GB,15minutes per role, with four tasks at most running concurrently. Clinical identity remains unresolved and all82provider keys have entered development. The validation measurements describe the current development cohort.

[Configuration](../configs/breakhis-midnight-kernel-v1.json) · [Primary-source assessment](reviews/FEW_SHOT_KERNEL_ASSESSMENT.md) · [Midnight comparison](evidence/BREAKHIS_MIDNIGHT_RESULTS.json).
