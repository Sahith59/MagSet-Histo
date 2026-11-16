# Similarity classifier: another measured improvement

November16,2026. The completed two-step similarity classifier scores **75.67%/75.00%** final five-category accuracy, up from frozen Midnight logistic case heads at72.00%/71.33%. The cases, roles, cached image reader and scoring rules are unchanged. Macro F1 also improves on both arrangements. **94–95% has not been reached.**

In plain language: turn each tissue image into a numerical description, compare it with fitting examples from the same microscope zoom, combine the evidence, decide benign/malignant, then choose a malignant type when needed. Every fitting case has equal influence inside its class regardless of how many images it contains. This is a standard similarity/density classifier, not a newly established research mechanism.

## Complete validation comparison

Scores average all15 available-view combinations and five trials per arrangement. All-four-view accuracy is a separate score, shown explicitly.

| Method | Arrangement | Final accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Midnight case logistic dual reference |812|72.00%|.5558|63.80%|75.00%|
| Similarity direct-five control |812|75.00%|.5847|62.80%|77.50%|
| **Similarity two-step** |**812**|**75.67%**|**.5897**|**62.87%**|**82.50%**|
| Midnight case logistic dual reference |813|71.33%|.5350|60.53%|72.50%|
| Similarity direct-five control |813|73.33%|.5713|64.13%|80.00%|
| **Similarity two-step** |**813**|**75.00%**|**.5643**|**62.40%**|**80.00%**|

Against the fixed Midnight case-dual reference, raw accuracy gains are3.67/3.67 percentage points and F1 gains.0339/.0293. Against the first frozen-Phikon dual reference65.83%/68.83%, raw gains are9.83/6.17 points. The registered descriptive improvement screen passes on both arrangements. Balanced accuracy falls slightly on812, and the hierarchy's F1 is lower than the direct control on813. The evidence does not show an improvement in every metric or class.

## Where it still struggles

| Two-step diagnostic |812|813|
| --- | ---: | ---: |
| Actual benign/malignant router accuracy |92.00%|92.67%|
| True-malignant conditional subtype accuracy |56.67%|56.67%|
| Benign final recall |97.00%|96.00%|
| Ductal final recall |66.67%|44.00%|
| Lobular final recall |90.67%|93.33%|
| Mucinous final recall |41.33%|41.33%|
| Papillary final recall |18.67%|37.33%|

The higher overall score partly reflects stronger benign recognition; benign represents half of validation cases. Lobular recall improves substantially, while other malignant classes remain weak. Conditional typing excludes router errors and is not whole-system accuracy. The next representation-learning study addresses this remaining limitation; similarity alone does not solve it.

## Method and execution

Use the accepted3072-dimensional Midnight features. Center each zoom using the equal mean of fitting-case image means, then L2-normalize. Compute same-view cosine kernel densities with uniform head-class priors and equal case/field mass within each head class. The four shared temperatures are.01/.03/.1/.3. Compose field joint-five probabilities before equal field/view means. The binary decision uses malignant probability>=.5, then the four-type argmax. Direct-five and dual each select among four complete candidates by validation mean15-subset macro F1.

There are zero newly optimized classifier coefficients and zero new encoder forwards. Fitting labels define the support distributions, so the classifier still learns its reference examples from fitting data. Fitting diagnostics leave the complete query case out of support and centering; validation uses the46 fitting cases only.

All ten paired CPU tasks completed successfully, followed by full collector replay. The collector verified80 candidates,30 full-fit support-head definitions,160 prediction archives and64,800 dependent case/subset records, including support weights, probabilities, final labels, metrics and selection. An independent scalar oracle had earlier checked251 numerical cases; the full pinned suite passed252 tests. Source CI also passed. Root separately checked20 selections and12 aggregate means. Cached-feature arithmetic replay is not an independent raw-image encoder execution.

## Provenance and limits

- Immutable producer: `05baa0e80bcc4fa4270646334bf2223fb1c4d7eb`.
- [Registered configuration](../configs/breakhis-midnight-kernel-v1.json), SHA256 `65f56e4151cd0add1dbfd8cf2a92fa051fcd0af498eae635fb242e550c51d02e`.
- [Complete accepted comparison](evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), SHA256 `e6e98fc26b3579261baf39b677e75551f7bbfbc5f3a7fe399db59c7169511274`.
- [Method specification](BREAKHIS_KERNEL_SPEC.md) · [Independent implementation review](reviews/KERNEL_IMPLEMENTATION_REVIEW.md) · [Prior improvements](BREAKHIS_IMPROVEMENT_RESULTS.md).

Only BreakHis is fitted/evaluated. Roles remain46 fitting/eight validation/28 excluded; five overlapping provider trials at seeds812/813 reuse31 distinct validation keys. All82 provider keys have entered development somewhere and clinical identifier relationships remain unresolved. The records and masks are dependent, not thousands of independent patients. These are reused exploratory development numbers; they do not establish fresh-test95%, clinical validity or novelty. Preserve every model/candidate and report the exact prediction unit and endpoint.
