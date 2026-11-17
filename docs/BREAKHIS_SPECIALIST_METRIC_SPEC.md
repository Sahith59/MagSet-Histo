# Fitting-case metric learning for the two-step classifier

Registered November 17, 2026, before any new fitting or validation outcomes. The user requests active work on malignant-subtype generalization and improved final five-category accuracy. The 94–95% goal remains an objective.

## Question and fixed comparison

Test whether supervised weighting of frozen UNI2-h feature dimensions improves the requested benign/malignant decision and malignant-four decision. The prior reader study constructed labeled density supports without optimizing classifier coefficients. This study learns task-specific distance weights from fitting cases.

The primary comparison is **both learned heads versus exact identity replay**. Binary-only and malignant-only learning are mechanistic ablations. Report all four arms. The same two trained head metrics are shared across these arms, giving twenty production optimizations across ten original roles, forty named pipelines, eighty prediction archives and 32,400 dependent case/subset records. No new validation-based hyperparameter, checkpoint or temperature selection occurs. Choosing among the four reported arms would add four arm choices. Earlier adaptive choice of UNI2-h and inherited temperature selection remain part of the validation-reuse history.

The unchanged two-step structure has one binary density head and one four-type malignant density head. The existing one-stage UNI2-h results remain contextual evidence. This experiment estimates learned geometry within the fixed two-step architecture.

## Trainable mechanism

For each head, learn one 1,536-component vector shared across all four magnifications. Initialize theta to zero. Its log scale is `a = 0.5*tanh(theta)` and its positive feature scale is `d = exp(a)`. Each scale lies between approximately 0.607 and 1.649. Identity uses an exact bypass with unit scales.

Construct one mean descriptor per fitting case and magnification. Both heads use the equal-case raw center from all 46 fitting cases, followed by L2 normalization with epsilon 1e-12. The binary objective includes all 46 fitting cases, grouped 11 benign and 35 malignant. The malignant objective includes 35 cases with subtype counts 25 ductal, two lobular, five mucinous and three papillary.

Within each magnification, apply the bounded diagonal scale and normalize again. Each anchor compares only with different fitting cases at that magnification. For each head class, average exponentiated cosine similarities over the remaining support cases. The anchor's own class denominator is its class count minus one. Other class denominators are their full counts. Normalize those class densities to obtain class probabilities and minimize the anchor's negative log probability for its true class. Each class receives equal total anchor weight, with per-anchor weight `1/(K*n_class)`. Average the four view losses and add `10*mean(a**2)` to favor the identity geometry.

Train each head independently for exactly 200 full-batch Adam steps with learning rate 0.01, weight decay zero and seed 812. Use FP32, deterministic execution and disabled TF32. Save the final step only. There is one fixed configuration for each head. The binary and malignant objectives are separate training losses.

Compute fitting-case field means, the full fitting center and initial normalization in FP64 before casting normalized training vectors to FP32. Store final theta in FP32 and the actual final FP32 Torch log scales and scales as exactly representable FP64 arrays. Canonical inference uses these stored scales in FP64. An independent FP64 formula check uses fixed absolute and relative tolerances of 1e-6 for FP32 activation rounding. Identity bypasses scaling and re-normalization.

This is an adapted neighborhood metric-learning objective. The original NCA work establishes supervised distance learning as prior art. The case-mean training objective and field-level inference definition are deliberate implementation choices whose effect requires measurement. [Neighbourhood Components Analysis](https://proceedings.neurips.cc/paper_files/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html).

## Prediction and diagnostic contract

Each role inherits its already selected UNI2-h two-step temperature from the accepted reference ledger. That same fixed temperature is used by every arm and both heads. The final classifier applies learned scales to the head's centered, normalized original field descriptors, normalizes again and builds class/case/field-balanced kernel densities. Identity preserves the previous numerical path.

Compose field-level joint five-category probabilities, average within each zoom and equally across available zooms, then route malignant probability at least 0.5 to its highest-probability malignant subtype. Malignant-only learning preserves the original binary outputs exactly. Binary-only learning changes field malignancy weights and can consequently change the pooled conditional subtype distribution even with the original field-level subtype head.

Fitting diagnostics are titled **in-sample learned metric with whole query omitted from density support**. Query-case labels trained the metric, so this is distinct from the earlier fully fixed-feature leave-case-out diagnostic. Each fitting query is still omitted from its density support and support center. Validation cases contribute no training labels, centers or density support.

## Data, evidence and acceptance

Use the exact accepted UNI2-h feature cache, manifest and original 46 fitting/eight validation/28 excluded roles. Preserve both arrangements, five overlapping trials and all fifteen nonempty magnification subsets. BreakHis supplies all task images and labels. There are 31 distinct reused validation keys, unresolved clinical identity relationships and no globally fresh final cohort.

Before production, run two complete 200-step fitting-only head checks on arrangement 812/trial 1 and score the first fitting case from each final class. This produces 300 fitting-only records across four arms. An initial two-case check completed both optimizations but failed when the complete-role summary required all five final classes. Preserve that attempt and account for its two optimizations, giving four technical optimizations across the initial attempt and retry. The training objective, inherited settings and twenty production optimizations retain their registered definitions. Verify finite losses, active gradients, factor bounds, exact artifact reload and native inference parity. The identity arm must reproduce the reference's selected validation labels and metrics exactly and probabilities within 1e-10. Save all final factors, training histories and source/configuration/cache hashes privately.

The full collector independently replays all stored predictions and verifies the complete role/arm matrix, constant inherited temperatures and learned artifacts. Report final accuracy, mean-subset macro F1, balanced accuracy, class recalls/confusions, routing, conditional malignant typing, proper losses, all-four-view scores and time/memory. The primary descriptive improvement screen requires at least +0.02 macro F1 on both arrangements, nondecreasing accuracy and positive recall for all five final labels versus identity. Compare P as a separately named contextual reference.

Bounded execution uses ten one-GPU A40 jobs, at most four concurrently, with fifteen minutes and sixteen GiB host memory each. The CPU collector has thirty minutes. Model weights remain frozen, and no new image encoding is required. All outcomes will be retained, including a negative result. Novelty and 94–95% final accuracy require additional measured evidence.
