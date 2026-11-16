# One bounded cosine-kernel follow-up

November 15, 2026. Assessment only: no implementation, new fitting, model download, job submission or current-result inspection. The running four-C linear comparison remains unchanged.

**Recommendation: yes, one separately registered cache-only study is reasonable if the current frozen Midnight linear readouts fall short.** It tests a materially different decision rule without fitting thousands of coefficients from 46 independent fitting cases. Use one predeclared encoder, four shared temperatures and the existing direct-five/two-step controls. It is a standard nonparametric baseline, neither a novel method nor a reproduction of SimpleShot or Tip-Adapter. There is no evidence that it will reach 94–95%.

## Primary-source support and limits

[SimpleShot, Wang et al.](https://arxiv.org/abs/1911.04623v2) shows that simple nearest-neighbor/nearest-centroid rules on frozen representations can be competitive in few-shot learning, and that normalization choices materially affect results. Its [official implementation](https://github.com/mileyan/simple_shot/blob/91c262718fe66837c6f6e90b831fca96123654b4/src/test_inatural.py) implements raw, L2-normalized and centered-plus-L2 variants. This supports a simple geometry-based control, not a claim about breast histology or our few-provider protocol.

[Tip-Adapter, Zhang et al.](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf) retrieves labels from normalized frozen image-feature keys using exponential cosine affinities. Its [official cache/search code](https://github.com/gaopengcuhk/Tip-Adapter/blob/d0e2d6f8c5feb8b6ce937b757810761f7155d4d5/utils.py) uses an affinity kernel equivalent up to a common factor to `exp(cosine/T)`. However, the original also incorporates CLIP text logits, augmentation and additional hyperparameters; the proposed pathology density rule does not. Its headline results therefore cannot be transferred here.

Those papers motivate testing a cheap established alternative. They do not establish that normalized pathology embeddings separate lobular/papillary cases, that every field has discriminative subtype morphology, or that avoiding an optimizer prevents overfitting to reused validation.

## Exact proposed rule

Normalize each original frozen field feature independently: `u = x / ||x||₂`. Reject nonfinite or near-zero-norm embeddings. No cohort mean, per-coordinate standard deviation, learned projection, PCA, encoder update or query-derived statistics are used. Compare queries only with **same-magnification fitting fields**.

For head `h`, class `c`, magnification `m`, and query `q`, define:

```text
log D_h,c(q,m,T) = logsumexp over fitting cases i in class c and their fields j:
    cosine(u_q,u_i,m,j)/T - log(N_h,c) - log(n_i,m)
```

Here `N_h,c` is the number of eligible fitting cases in that head/class and `n_i,m` is the field count for that case at that magnification. Thus each fitting case has equal total mass inside its class, irrespective of field count. Normalize these log densities across the head's classes using log-softmax: this specifies a uniform **head-class prior**, rather than using empirical class frequency. A common spherical-kernel normalizing constant cancels because temperature and embedding dimension are shared across classes.

Direct-five uses all five final labels. The binary router uses benign/malignant fitting labels; the four-type head uses **true malignant fitting cases only**. For every query field, first form the dual joint distribution `[1-r, r*s_DC, r*s_LC, r*s_MC, r*s_PC]`. Average these joint probabilities within each available magnification, then average available magnifications equally. Let `R` be the sum of the four pooled malignant probabilities. Predict benign when `R < .5`; otherwise predict the largest malignant component, including a malignant route at equality. Do not substitute joint argmax across all five outputs or compose independently pooled head probabilities.

Use exactly `T = [.01, .03, .1, .3]`, **the same T for both dual heads**, giving four complete candidates per arm/run. Selecting binary and subtype temperatures independently would create 16 combinations and break the four-opportunity budget. Preserve the original five trials, arrangements 812/813, 46/8/28 roles, all four magnifications, all 15 nonempty subsets, and end-to-end macro-F1 selection/tie rules. Validation cases and excluded-role cases never enter a run's fitting key cache.

For fitting diagnostics, remove the query's **entire provider case**, not only its own field, from every head before recomputing class denominators and scores. Two lobular fitting cases leave one other lobular case; three papillary cases leave two. These leave-case-out diagnostics are deliberately different from the existing heads' resubstitution fitting accuracy and must be labeled accordingly. Provider-key exclusion does not resolve the existing uncertainty about clinical identity.

## Risks that should be recorded before execution

- **No coefficient fitting is not no selection.** Four temperatures selected on eight reused validation cases can still overfit; each malignant validation type has only one case per run. Retain every candidate/result and require the existing both-arrangement improvement screen. Do not change temperature range after results appear.
- **The two arms have different hierarchical priors.** Uniform direct-five priors assign each leaf .2. Uniform binary and malignant-four priors imply benign .5 and each malignant leaf .125 before evidence. Also, the binary malignant density averages its 35 fitting cases, so ductal cases still dominate that coarse-class mixture. Case balancing removes unequal field counts; it does not make the binary malignant mixture uniform over subtypes. Report this explicitly; it is not architecture-only evidence.
- **L2 does not solve embedding anisotropy.** Raw L2 preserves angles of the supplied representation and avoids estimating a noisy diagonal metric from this cohort. Fit-only standardization instead changes those angles, potentially amplifying weak-variance directions; it can also suppress nuisance scales. SimpleShot's centering results are counterevidence to any universal claim that raw L2 is best. Fix raw L2 for this study; do not add normalization variants after selection. For Midnight's concatenated CLS/mean-patch feature, global L2 also retains the original relative norms of those two blocks.
- **Field density can retrieve nuisance similarity.** Stain, scanner, preparation and abundant normal/background morphology may outweigh subtype evidence. Small T behaves more like nearest retrieval; large T washes classes toward similar densities. Rare types still have only two/three independent fitting cases. More correlated fields do not create more independent patients.
- **Stable arithmetic is essential.** At T=.01, direct `exp(cosine/T)` can overflow FP32. Use logsumexp/log-softmax, finite masks and updated leave-case-out denominators. Test against a tiny float64 oracle, plus invariance when all fields of a case/view are duplicated uniformly, field/view permutation invariance, absent-view invariance, split exclusion, hard routing and save/reload. Duplicating only one distinct field changes its empirical mass and is not expected to be invariant.

## Compute and stopping budget

The audited magnification counts are 1,995/2,081/2,013/1,820 ([dataset audit](../DATA_AUDIT.md)). Across the entire release, all same-magnification pairwise similarities total **15,675,155 entries**, about **59.8 MiB FP32**. Forming all of them would require about **96.3 GFLOP at 3072 dimensions**, or **32.1 GFLOP at 1024 dimensions**; these are arithmetic counts, not measured runtimes. Actual per-run fitting/validation blocks are smaller. Query batches of 256 keep temporary similarity/log-density arrays small, and temperatures/roles can reuse dot products without additional encoder forwards. Do not score excluded-role outputs merely because their embeddings exist.

One scheduler CPU job with **8 cores, 16 GiB RAM, a 30-minute wall-time ceiling**, zero GPU requests and zero encoder forwards is a reasonable proposed cap. Choose the encoder before kernel outcomes; default to the already accepted Midnight cache if testing the frozen Midnight follow-up, and do not automatically sweep both encoders. Ten matched role arrangements × two arms × four temperatures gives **80 complete candidate evaluations**, not 80 optimizer fits. Collect all outputs and independently replay selection/metrics before accepting any result. If this one study fails, retain the best verified two-step model and finish the report/deck/demo; do not turn the capstone deadline into an open kernel/normalization/encoder search.
