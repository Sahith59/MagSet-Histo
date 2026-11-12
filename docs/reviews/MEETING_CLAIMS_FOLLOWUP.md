# Adversarial professor review: meeting claims and answers

Prepared November 12, 2026. This is a second-round review of the planned evidence package, not an endorsement of a finished paper. The internal package target is **November 18, 2026**; the meeting date has not been fixed. No professor contact, account registration, model job, or final-test scoring was performed for this review.

Read together with the [validity critique](VALIDITY_CRITIQUE.md), [data/provenance critique](DATA_PROVENANCE_CRITIQUE.md), and [novelty critique](NOVELTY_CRITIQUE.md). Claims about future work below are proposals, not completion statements.

## Verdict

The team can present a credible **progress and decision package**: an audited pipeline, retained negative results, a substantial development ledger, and a candidate that can be disproved by simple controls. The team cannot yet present a demonstrated novel method, a globally untouched BreakHis confirmation result, contamination-free TCGA generalization with Phikon, or a submission-ready paper.

The strongest current message is: **we checked the original work, rebuilt the evaluation, found that a normalization-matched simple baseline remains ahead of the tested neural controls, and are narrowing the next experiment to a small, controlled evidence-routing hypothesis.**

## Four claims that must be withdrawn or replaced

| Wording to withdraw | Evidence-supported replacement | Why it matters |
| --- | --- | --- |
| “The final BreakHis test patients are globally untouched.” | “No outer-role predictions have been scored. Each run excludes its own outer-role cases from fitting and selection, but all 82 provider cases appeared in development somewhere across the repeated trials.” | Run-level exclusion is valid; global confirmation independence is a different requirement. Later BreakHis scores must disclose the adaptive development reuse. |
| “TCGA-MS is independent external validation with Phikon-v2.” | “TCGA-MS is a secondary multi-organ morphology task. Phikon has confirmed TCGA-source pretraining exposure; exact overlap with these slides is unresolved. A frozen ImageNet control and an explicit exposure statement are needed.” | No supervised head result exists there yet. Source exposure is not proof of label leakage, but it prevents a clean unseen-pretraining-cohort claim. |
| “We tested on 175 independent samples/experiments.” | “We completed 175 development configurations, with repeated cases, view subsets, trials, and tuning. The original dataset has 82 provider cases, not 175 new patients.” | Repeated computation improves debugging evidence, not patient diversity or inferential sample size. |
| “Our OT/shared–unique mechanism is novel.” | “The candidate is an untested hypothesis. Cross-scale OT, missing-view gates, shared/specific representations, and reliability-dependent rejection all have prior work.” | A distinct operation and a controlled benefit must be demonstrated; combining existing modules or changing the task is insufficient by itself. |

Also avoid “the attention approach is generally worse,” “clinical robustness,” “biologically unique evidence,” “registered fields,” or “a paper is guaranteed next week.” The evidence supports narrower, conditional statements.

## Likely questions and concise answers

### 1. What exactly is the research problem?

We want one subtype prediction from several sets of tissue images at different magnifications, even when some magnifications are absent. The first task is eight-class BreakHis classification over all 15 nonempty combinations of its four scales. We treat fields as unpaired sets, not registered image pairs.

### 2. What did you improve over the original supplied notebooks?

We preserved them as historical material and rebuilt an explicit provider-case development protocol, frozen features, missing-view handling, saved predictions, provenance, and verification. The old reported 93.73%/94.99% image accuracies use a different splitting/model-selection procedure and are not targets for the current case-set benchmark.

### 3. What evidence exists today?

The repository contains verified data audits, fixed feature caches, linear and learned internal controls, split/training-seed checks, and normalization/capacity controls. The [matched report](../CAPACITY_RESULTS.md) gives linear mean macro F1 of 0.3843/0.3169 versus checked neural maxima of 0.3636/0.2744 for the two inner splits. Those are selected development summaries, not final test accuracy or significance estimates.

### 4. Why do those scores look much lower than published headline accuracies?

Task labels, prediction unit, splitting, and metric differ. Eight-class case-set macro F1 is not binary image accuracy. The small and rare-class development cohorts are difficult and unstable. We should not claim one universal cause: representation/crop policy, limited diversity, grouping uncertainty, and optimization can all matter.

### 5. Are these clinical patients or identifiers from filenames?

BreakHis currently uses the provider's 82 subtype-qualified case identities. There are 70 conservative numeric stems, with unresolved relationships and some mixed subtype groups. These are not independently reverified clinical identities. A documented prediction/grouping decision is needed before claiming clinical-patient generalization.

### 6. Why not simply enforce the stricter grouping now?

The [validity review](VALIDITY_CRITIQUE.md) finds only two adenosis stems. A class-covered fitting/development/test split needs at least three independent groups for that class. A simple embargo on the current assignments removes whole fitting subtypes in every checked split. That is a feasibility limit, not a random-seed problem; the protocol/task must be decided explicitly.

### 7. Have the test data been used?

No outer-role predictions have been scored. Within a run, its outer cases do not fit normalization or classifiers or select settings. Across the rotating development trials, however, the same cases can have other roles. All 82 cases participated in development somewhere. We will not describe a later adaptive BreakHis evaluation as a globally pristine confirmation cohort.

### 8. Is the accumulated validation evidence sufficient to select a final architecture?

It is sufficient for debugging and hypothesis formation, not confirmatory selection. The validity review counts 700 candidate-validation evaluations and only 31 distinct validation provider cases across both inner seeds. A frozen recipe and genuinely protected confirmation population are needed for a stronger claim.

### 9. Do you already have apples-to-apples baselines?

We have useful internal controls, including a recent comparison with exact saved normalization-array equality and case/cache agreement. The [comparison contract](../COMPARISON_PLAN.md) records the common inputs and budgets. Different fitting algorithms are not identical compute/capacity experiments; candidate ablations must match the changed component more tightly.

### 10. Have MagViT and M2OTCA been reproduced?

No. The contextual gate and small attention model are generic controls, not exact reproductions. MagViT's source task is binary. M2OTCA's full equations and official implementation remain unverified. A clearly documented compatible adaptation can be a comparison, but must not be labeled an exact reproduction.

### 11. What is the proposed new idea?

At present it is a hypothesis: keep the strong mean representation and add a small fixed-dimensional description of how evidence agreement varies across observed views. Test whether that correction contributes beyond generic within-bag variation, a duplicate-mean block, or a random partition. No performance improvement or distinct novelty is claimed yet.

### 12. Why not implement the full transport architecture immediately?

The current evidence does not establish that transport is the missing ingredient. It would add choices and complexity to a tiny reused development set. A cheap agreement-contrast pilot can first test the specific rationale. OT becomes justified only if simpler controls leave an identifiable problem and a relevant prior-art distinction remains.

### 13. Does retaining matched and unmatched evidence automatically add information?

No. For original masses a_i and matched masses r_i, the pooled branches satisfy `Σ a_i h_i = Σ r_i h_i + Σ (a_i − r_i) h_i`. Their sum recovers the original mean exactly. A benefit may arise from reweighting, additional dimensions, or changed regularization. The contrast/residual controls are designed to expose those alternatives.

### 14. Is the candidate guaranteed not to harm the mean baseline?

No. Setting correction coefficients to zero represents the baseline within the model family, but fitting or validation selection may still choose a worse model. That is baseline **recoverability**, not a guarantee of improved accuracy, calibration, monotonic view addition, or generalization.

### 15. How is the fixed 32-dimensional contrast controlled?

Projection and routing definitions must be specified before outcomes. Use identical feature normalization, case/field budgets, masks, projection dimension, classifier penalty, and selection count in every arm. Random partitions must be stable under field reordering, not depend on array positions or labels. A one-view case gets zero cross-view correction and an explicit supported baseline path.

### 16. What outcome would make you reject the proposed explanation?

If a duplicate-mean or random/within-bag contrast matches the gain, the agreement-specific story is not supported. If an inexpensive affinity rule explains the effect, OT-specific novelty is not supported. If gains depend on one selected split, disappear under acceptable grouping sensitivity, or trade away the rare-class/worst-subset endpoint, revise or stop the claim.

### 17. Is TCGA-MS another breast dataset?

No. The audited 128 supplied patient IDs cover 16 morphology labels across several organs; 32 are breast cases in the supplied tissue field. It is a changed secondary task with a separate head and native mpp scale definitions. Its 65,312 image fields are not 65,312 independent patients.

### 18. How will TCGA give stronger evidence?

A new patient-level development/confirmation protocol can be frozen before supervised modeling outcomes are examined. The proposed four-native-resolution track must preserve related fields/scales together and fix eligibility. It can test the mechanism on another task, with explicit pretraining exposure. It will not erase limited cohort size or automatically validate the identical breast classifier.

### 19. Is Phikon pretraining overlap a fatal error?

It is a serious interpretation constraint, not proof that the supervised experiment is automatically invalid. The [pinned model card](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/README.md) and [model paper](https://arxiv.org/html/2409.09173v1) document TCGA-source exposure. Exact slide membership remains unresolved. Use ImageNet-pretrained ResNet first as a control without known TCGA pathology-pretraining exposure; label any Phikon result accordingly.

### 20. Could more GPUs solve the present limitation?

They can accelerate encoding and bounded repeats. They cannot create independent patients, settle identity ambiguity, establish novelty, or protect a reused holdout. The next resource allocation should follow the agreed hypothesis and protocol, rather than another unrestricted architecture sweep.

### 21. Do you have BRACS access as a fallback?

No. Access has not been obtained, no registration has been completed, and no data has been downloaded. The current plan must not depend on BRACS being immediately available. It remains an optional later route subject to access and a separate protocol.

### 22. Is the code completely validated?

Existing tests cover important parsing, split, masking, permutation, numerical, reload, and metric checks. The validity critique also identified a missing trusted label/role oracle and mutable queued-source risk. Those findings need explicit closure evidence before we call them resolved. Passing software tests does not validate clinical utility or scientific novelty.

### 23. What will be ready by November 18?

The target package is a short narrative, a verified data/provenance table, baseline and failure-analysis figures, the critique/closure ledger, a frozen next-experiment specification, and a candid decision page. A small candidate feasibility pilot is conditional on protocol and implementation gates. Positive results, a final test score, and a submission-ready paper are not promised.

### 24. What decision do you need from Professor Dong?

Agree on the defensible prediction/grouping unit, the protected confirmation strategy, and whether the best contribution is currently a benchmark/analysis study or a mechanism study requiring further evidence. Review the candidate's exact distinction and stop criteria before expanding compute. Venue and authorship remain decisions for the eventual completed work.

## Candidate proposal: what can be claimed before performance exists

The proposed feature family is a mean-and-mask baseline block plus a **fixed 32-dimensional agreement contrast**. The projection seed and dimension are fixed without task-outcome selection. The classifier remains small and regularized. Jointly fitting a correction is not evidence that the baseline was preserved as an actual learned prediction; it only makes the baseline representable at zero correction.

The mechanism question should be: **does agreement-derived reweighting add useful information beyond an equal-size generic correction under the same observed inputs?** It should not be: “can another set of features raise a selected development score?”

| Required arm | What it isolates |
| --- | --- |
| Matched mean-and-mask baseline | Existing strong starting point |
| Baseline plus projected duplicate mean | Reparameterization and regularization changes |
| Baseline plus random within-bag contrast | Extra descriptor dimensions and arbitrary partitioning |
| Baseline plus within-bag variation contrast | Generic field heterogeneity without cross-view agreement |
| Baseline plus fixed agreement contrast | The proposed application-specific hypothesis |

These are design specifications only. They are not implemented or measured by this review. A partial-OT variant is optional later, after the simpler comparison and prior-art gate. No source-code or model result should be inferred from this table.

Before fitting, tests should establish exact algebraic identities, stable field permutations, missing-view independence, one-view fallback, deterministic descriptor construction, finite values, and training-only normalization. The random-control definition must not accidentally use labels or a different number of fields. Every plotted task result must trace to saved probability records and an audited trusted label/role context.

## Minimum evidence board for the target package

| Item | Acceptable evidence | Unacceptable substitute |
| --- | --- | --- |
| Reproducible progress | Data/cache/source/config hashes, saved scores, rerunnable table commands | A claim based on GPU hours or screenshots alone |
| Current result | All specified development outcomes with case support and limitations | Only the highest row or favorable inner seed |
| Protected confirmation | Frozen patient assignments and an outcome-access policy | “No test prediction file” while every patient guides global discovery |
| Mechanism rationale | Exact operations, closest prior art, an algebra/control argument, and a falsifiable endpoint | A new model name or a list of familiar components |
| Critique closure | Test/commit/artifact proving each fix, with unresolved issues retained | Saying “agents reviewed it” as proof that it is correct |
| Meeting readiness | Honest summary and a concrete next decision by the internal target date | An invented acceptance, novelty, external-generalization, or clinical claim |

The reviewer verdict remains: **ready to develop a candid progress presentation, conditional on accurate claim wording and visible unresolved issues; not ready to present a novel-method success.**
