# Conditional centering: bounded mechanistic assessment

**Verdict: reasonable for one fixed CPU-only study**, with the old case-balanced center retained as reference. This is a supervised-normalization hypothesis, not correction of an implementation bug, a novel mechanism or evidence for 95%.

The proposed center is mathematically well-defined: if W is the existing field-by-class `class_weights` matrix, then `mu_h = sum_fields(x * row_sum(W))/K_h`. Each class contributes equally, each eligible case equally within class, and each field equally within case. No density weights or head priors change. Translation would cancel in ordinary Euclidean differences; it does **not** cancel after L2 normalization, so this intervention changes cosine geometry legitimately.

Uniform density priors do not require uniform-class centering. The existing equal-case center is a standard baseline. DC's 25 fitting cases influence it heavily, but that observation does not establish that centering erased useful DC information or caused its measured recall. Equal-class centers also give noisy two-case LC and three-case PC means substantial influence; generalization could worsen. Near-center vectors can amplify small nuisance differences after normalization.

The shared-mu5 control is essential. Direct-five outputs must be identical in shared and head-conditional modes. In the conditional mode, changing benign support features must leave **expert mu4 and expert probabilities** unchanged; this invariance is **not expected** under shared mu5, which intentionally includes benign information. Query-label invariance must hold in both modes: all applicable heads normalize every query with support-derived centers, never a center chosen using its true label. For fitting diagnostics, remove the whole query case before both centers and density weights. Check empty-class guards and unchanged zero-norm handling.

Compare each fixed mode separately with four shared temperatures and unchanged roles/subsets. Choosing the better mode afterward inherits **eight** candidate opportunities per arm, not four; direct duplicates are not independent evidence. Preserve every outcome. The study adds neither encoder adaptation nor independent rare-type cases, and repeated development validation remains exploratory.

Design review only: no implementation, cohort predictions, model construction, fitting or jobs.
