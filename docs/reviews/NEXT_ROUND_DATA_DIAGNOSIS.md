# Next-round data and supervision diagnosis

November 16, 2026. The strongest bottleneck is **transfer of malignant morphology between cases**, especially mucinous and papillary cases. The current evidence supports one bounded test of training on complete case bags with the frozen three-crop features, with a matched mean-pooling control. It does not support a 94–95% forecast, another routing-threshold search or treating hundreds of minority fields as hundreds of independent examples.

## What the accepted results establish

All entries are final five-category development accuracy / macro F1, averaged over five overlapping trials and all 15 nonempty magnification subsets.

| Two-step pipeline | Arrangement 812 | Arrangement 813 |
| --- | ---: | ---: |
| C: single-center Midnight KDE |75.67% / .589691|75.00% / .564292|
| F: three-crop Midnight KDE |75.50% / .592510|76.83% / .579947|
| P: head-specific centers, F features |76.33% / .602715|75.67% / .602512|
| N: native global pixels, same KDE |60.00% / .439405|61.00% / .501395|
| LoRA, selected checkpoints |68.33% / .494406|59.83% / .375418|

P has the highest observed F1 in both arrangements, but F retains higher 813 raw accuracy. P also worsens cross-entropy and Brier loss relative to F. More pixels, adaptation and modest geometric improvements have not established uniformly better predictions.

P routes benign/malignant at **94.83% / 93.00%**, while true-malignant typing is **58.33% / 62.33%**. Its 600 dependent validation events per arrangement contain 142/146 final errors: **111/104 are wrong malignant subtypes**, and 31/42 are routing errors. With its conditional subtype predictions unchanged, even perfect routing yields only **79.17% / 81.17%** final accuracy. A perfect specialist with the existing router would still be bounded by binary accuracy. Both decisions must improve to reach 95%.

The available-view problem also matters. P averages **71.25% / 69.38%** with one view, versus **85% / 85%** with all four. Ten of the 15 equally weighted masks contain only one or two views. Full-view accuracy is a different endpoint, and concentrating on full-view performance would conceal the main scoring burden.

## Independent cases, not image counts

Every role has fitting support **11 benign, 25 ductal, 2 lobular, 5 mucinous and 3 papillary cases**. Validation has **4/1/1/1/1**. The original eight-label split precedes benign pooling; these are deliberate retained roles, not evidence of a five-label stratification bug. Fitting fields range from 226–324 for lobular and 226–343 for papillary across roles, but they still come from two and three cases. Whole-case exclusion leaves only one lobular and two papillary support cases for those fitting queries.

The most revealing negative evidence is P's **leave-case-out fitting recall**:

| Class | Fitting diagnostic 812 / 813 | Validation 812 / 813 |
| --- | ---: | ---: |
| Lobular |27.33% / 56.00%|93.33% / 98.67%|
| Mucinous |18.67% / 20.80%|49.33% / 36.00%|
| Papillary |11.11% / 4.44%|20.00% / 45.33%|

Strong validation lobular recall therefore does not establish broadly reliable lobular morphology. Rare-case appearance and support composition matter. These fitting diagnostics are harder than self-included fitting predictions and have different class proportions; their lower overall score does not imply validation leakage.

KDE already assigns equal prior mass to each head class, equal mass to its cases and equal mass to fields within a case/view. Repeating minority fields or adding the same inverse-frequency weighting cannot create missing case diversity. By contrast, LoRA's 99.13% fitting accuracy is resubstitution performance. Comparing it directly with KDE's leave-case-out fitting accuracy would be misleading.

## Implementation and supervision checks

I found no invalidating defect in the inspected paths. Original-label roles remain disjoint; feature paths bind to the manifest; labels are constant within provider keys. KDE excludes the whole fitting query before support weights and centers. P's expert center excludes benign support, and query truth does not determine prediction. Joint five-category field probabilities are composed before pooling; no artificial cross-magnification field pairing occurs. Requested-out and padded values must remain excluded before learned operations in the new implementation.

This review reconstructed ten role supports from the manifest/protocol, checked five accepted ledger hashes and independently recomputed confusion-based error factorization. There were 7,969 deterministic assertions, mostly manifest-row consistency checks. An additional **245 synthetic assertions** checked query-label invariance, case/class/field mass, joint probability pooling and unavailable-view invariance. These are source/artifact checks, not new training performance or independent raw-image encoder reproduction.

The remaining supervision concern is a hypothesis: field-supervised heads and LoRA demand the case label from every sampled field. With heterogeneous tissue, that may punish useful localized evidence. For example, true-class field probabilities .9 and .1 give mean field NLL 1.204, but the pooled case NLL is .693. The objectives differ even with correct code. This toy calculation does not prove that the actual fields contain mislabeled or nondiagnostic tissue.

## One bounded intervention and its refutation

**Train on complete case bags of the accepted frozen F features, comparing arithmetic field means with gated attention under the same case-level objective.** Keep the 46/8/28 roles, all fields, all 15 masks, four fixed checkpoints, direct-five control, two-step routing and threshold unchanged. No new encoder passes or raw-image augmentation are needed. This is the registered [case-supervision lane](../BREAKHIS_NEXT_ROUND_SPEC.md), not an attention novelty claim.

The falsifiable hypothesis is that selecting and combining fields under a case objective preserves subtype evidence that independent field losses dilute. The matched mean model is essential: if both improve similarly, the result supports the training/readout package rather than attention. If attention fails against its matched mean control, or improvements disappear in one arrangement or rare-class recall, reject attention as the explanation. Training loss improvements alone do not establish transfer.

The negative evidence is substantial: Midnight case heads were not consistently stronger than field heads; the case-supervised CNN failed; earlier attention experiments were negative. Roughly 0.4 million learned parameters for 46 cases create a serious overfitting risk, despite freezing the encoder. LoRA also selected head-only epoch 2 in eight of ten dual runs, and its fixed C=1 initialization was already weak. Longer adaptation is not justified by those outcomes alone.

Before production, a fitting-only technical smoke must demonstrate actual updates, exact class weights and last-batch scaling, absent expert gradients on benign-only batches, finite extreme-logit losses, padding/missing-view and permutation checks, paired shared initialization, saved-state replay and measured peak memory/time. It must not inspect validation or excluded outcomes. The registered bound is 40 trajectories, 160 checkpoint candidates, at most four A40s for this lane and 15 minutes per trajectory. Runtime is a budget to verify, not a measured forecast. This introduces real optimized classifier fits, despite zero new encoder forwards.

My lower-capacity diagonal-metric suggestion remains a future hypothesis, outside this registered search. Neither a new split nor additional labeling/clinical identity claims are part of this intervention. All 82 qualified provider keys entered development, only 31 distinct keys appear in validation across arrangements, and their clinical independence remains unresolved. Any gain remains reused development evidence.

Sources: accepted [C ledger](../evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), [F ledger](../evidence/BREAKHIS_COVERAGE_RESULTS.json), [P ledger](../evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json), [N ledger](../evidence/BREAKHIS_NATIVE_RESULTS.json), [LoRA ledger](../evidence/BREAKHIS_LORA_RESULTS.json); [KDE implementation](../../magset/kernel_model.py), [centering implementation](../../magset/prior_centered_model.py), [LoRA objective](../../magset/midnight_lora.py). Private calculations retain the exact public ledger hashes and contain no new cohort fits.
