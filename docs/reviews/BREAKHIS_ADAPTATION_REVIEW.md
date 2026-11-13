# BreakHis supervised adaptation: independent review

Prepared November 12, 2026. Scope is the next bounded BreakHis improvement after the accepted baseline/demo milestone. Secondary-dataset work and fusion novelty are outside this experiment.

## Proposed experiment and verdict

The proposed comparison adapts the existing pinned Phikon-v2 image encoder on each run's 46 fitting provider cases, using an auxiliary eight-class image head. The first epoch trains only that head; the following five epochs also train the final encoder block and final output LayerNorm. The fixed final encoder then encodes only the 46 fitting and eight validation cases. Fitting-only field-moment normalization and the same available-view mean plus mask logistic classifier are refitted; the existing four-value C grid selects that classifier on validation. All 15 nonempty view subsets are reported.

The proposed budget is ten adaptation fits: five provider trials at each of inner split seeds 812 and 813, with training seed 812 and one fixed training recipe. This is a reasonable **exploratory representation-adaptation comparison**. Its purpose is to test whether the existing image descriptions improve when a small part of the reader learns from fitting labels. It does not propose a new fusion mechanism, an independent clinical test, or a paper-ready novelty claim. No improvement is assumed.

At the initial review, the detailed specification and new implementation were not yet available. The later registered specification/configuration review is recorded below. Execution acceptance remains conditional on the implementation checks; no new fitting, scoring, image encoding, or cluster work has been performed by this reviewer.

## Requirements before launch

### High priority: exact split and feature provenance

- Regenerate the trusted provider-case assignments from the pinned manifest/protocol and each declared trial/inner seed. Confirm exactly 46 fitting, eight validation, and 28 excluded outer-role cases. Retain the existing subtype-qualified prediction unit and its clinical-identity limitation.
- Build the fitting image list before raw-file access. Training labels, augmentation, optimizer updates, normalization, and supervised checkpoint construction must use fitting cases only. Construct the separate 54-case encoding allowlist before reading any extraction image. Own-run outer cases must not enter adapted encoding or prediction, even though old externally frozen caches cover the full dataset.
- Give every adapted checkpoint/cache a distinct trial/inner-seed identity. Before reloading, compare manifest, protocol, base encoder revision, exact fitting IDs, training recipe, code revision, and checkpoint hash against the current run. Reject cross-run cache/checkpoint reuse, role changes, missing/duplicate paths, and inconsistent class order.
- Record image checksums, canonical label order, trainable parameter names/counts, actual update counts, sampler configuration, checkpoint/cache hashes, source/configuration revisions, environment, and precision. Preserve all ten outcomes, including failed or worse runs.

### High priority: define the weighting and update budget precisely

For fitting subtype `y`, fitting case `c`, observed magnification `m`, and field `i`, the proposed sampling interpretation should be:

`P(y)=1/8; P(c|y)=1/n_cases(y); P(m|c)=1/4; P(i|c,m)=1/n_fields(c,m)`.

This gives equal mass to each subtype, then equal case mass within a subtype and equal magnification mass within a case. Cases in different subtypes do not necessarily have equal unconditional mass. All current fitting cases have four views. If an implementation generalizes to incomplete cases, it must divide by the number of actually observed views.

Use ordinary cross-entropy with this class-balanced sampling distribution; adding inverse-class loss weights as well would change the intended objective through double balancing. Specify replacement behavior, exact draws per epoch, batch/last-batch policy, number of batches, and expected total updates. “Six epochs” alone is ambiguous under weighted sampling with replacement. Record realized sampled class/case/view counts for diagnosis, while recognizing that finite random draws need not give exactly equal counts.

Deterministic flips should derive from stable path/epoch/training-seed hashing, rather than Python's process-dependent `hash()`. No random training augmentation should occur during canonical feature extraction. Epoch-based flips may repeat when an oversampled image is drawn several times in that epoch; document this fixed policy.

### High priority: actually freeze and then adapt the intended modules

Check the actual pinned model's names, rather than guessing its module structure. Only the last encoder block, final output LayerNorm, and auxiliary head may become trainable after warmup. During the head-only epoch, the encoder block and final LayerNorm must both remain unchanged. The frozen prefix should remain in evaluation mode, including any stochastic modules; extraction uses the entire encoder in evaluation mode.

An optimizer created by filtering `requires_grad` during head-only warmup can accidentally omit the later-unfrozen encoder parameters forever. Either include the intended parameter groups before warmup or explicitly construct/extend the groups when unfreezing, with the registered learning rates. Define the policy in advance. Require tests/evidence that the prefix hash is unchanged, the intended block changes after encoder updates, all loss/gradient values remain finite, and optimizer groups/learning rates match the specification. Fail on nonfinite gradients or parameters rather than silently skipping unstable steps.

The auxiliary image head is a training aid. It must not quietly replace the registered case-level mean/mask logistic head in the final comparison. Save and reload the fixed final encoder checkpoint before canonical extraction; bind any partial checkpoint to the exact base revision. Retain the trained auxiliary head separately if needed for resumption/provenance.

The auxiliary loss assigns the provider-case diagnosis to its constituent fields. The manifest supplies no region/cell annotation demonstrating which tissue supports that diagnosis. This ordinary case-derived field supervision is part of the tested recipe; it does not establish that individual diagnostic regions were identified correctly.

## Fair comparison and statistics

The saved frozen matched mean baselines are the correct direct controls: same source images, provider trials, inner seeds, labels, field coverage, case pooling, normalization algorithm, mean/mask vectorizer, class-balanced logistic objective, C grid, tie rule, metrics, and 15 view subsets. Supervised encoder training and its fixed augmentation are the intentional difference. Describe the result as the effect of the **whole registered adaptation recipe**, rather than isolating the last block or any one ingredient without an ablation.

Use the existing `field-moments-v2` algorithm with fitting cases only. Adapted features legitimately change the numerical means and standard deviations; requiring equality to frozen-encoder normalization arrays would be the wrong fairness test. Fairness requires the same fitting identities and computation policy, not identical arrays after a representation change.

No validation-based epoch selection is planned. Preserve that choice: save epoch six regardless of validation outcomes, and do not inspect validation loss/accuracy to choose checkpoints, learning rates, training duration, or failed-run replacements. Selecting C on validation remains the same four-candidate procedure used by the frozen baseline. A technical retry should preserve the intended recipe and disclose its reason; completed unfavorable runs must not be replaced.

Report paired adapted-minus-frozen deltas for every trial, plus separate five-trial summaries for both inner seeds. Keep mean-subset macro F1 as the declared primary endpoint and show accuracy, full-view/worst-subset values and subtype recall distinctly. Do not present one favorable trial, one split, or image-level training accuracy as the project result.

The ten runs are overlapping repeated holdouts, not ten independent clinical replications; the 15 subsets do not create new cases. A t-test treating these runs/subsets as independent would not establish significance. Existing validation cases already influenced the larger development program, and one training seed does not measure training randomness. This experiment can provide descriptive improvement evidence, not an unbiased final-performance estimate. Clinical-identifier uncertainty and the absence of a globally fresh BreakHis confirmation cohort remain unchanged.

An independent trusted-membership count finds the same fitting class-case counts in every one of the ten runs: `[1,25,6,2,5,3,1,3]` in canonical order (adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor, tubular adenoma). **Adenosis and phyllodes each have only one fitting provider case per run.** Under the intended sampler, that single case receives one eighth of all draws. Repeated fields/flips cannot supply additional subtype diversity, so rare-class memorization is a substantial risk even with selective unfreezing. This does not prohibit an exploratory comparison, but it limits any interpretation of high training accuracy or apparent class recovery.

Fitting image counts range from 4,071 to 4,724 across these runs. If each epoch draws exactly the number of fitting images with replacement and retains the final batch, batch size 16 gives 255–296 updates per epoch. Alternatively a fixed draw count per epoch can make update budgets identical across runs. Choose and register one policy before launch; these counts are metadata audits, not model outcomes.

Before scores are produced, state how the fixed experiment will be interpreted. A transparent rule can require improvement on the primary five-trial average for both inner splits before calling the recipe promising, while still reporting every regression and class weakness. A gain on one split and loss on the other is mixed exploratory evidence, not permission for another unregistered sweep.

## Minimum verification and release gate

Tests should cover trusted split reconstruction and tampering, fitting-only sampling mass, stable augmentation, warmup/unfreeze optimizer membership, frozen-prefix preservation, final-checkpoint reload, exact allowed encoding coverage, class order and finite probabilities, fitting-only normalization, existing missing/permutation invariances, and independent metric arithmetic. Synthetic checks verify implementation, not real classification performance.

Before comparing scores, require saved adapted-feature provenance and exact validation case/truth/subset equality with the frozen control. Recompute metrics from saved probabilities. Verify final logistic coefficients/normalization reload and reproduce saved validation probabilities. Do not score excluded outer cases to rescue an ambiguous development result.

## Registered specification/configuration review

The subsequently written `docs/BREAKHIS_ADAPTATION_SPEC.md` and `configs/breakhis-adaptation-v1.json` implement the proposed safeguards at the design level: exact fitting-field draws with replacement and no dropped final batch, one head-only plus five encoder-update epochs, fixed learning rates/decay/dropout/precision, plain cross-entropy, fitting-only deterministic flips, final encoder reload, exact 54-case encoding, and matched downstream head selection. The specification explicitly records the single-fitting-case limitation of both rare subtypes and retains all production outcomes.

The budget additionally permits **one separate technical smoke fit**, limited to two fitting-only epochs and at most two batches per epoch, with no validation encoding/classification/score. This is a predeclared software execution gate, not one of the ten ranked production models; report its existence separately rather than calling the entire execution history exactly ten training fits. A completed smoke must not become a model-selection result.

Reviewed specification SHA-256: `6d5dd5b751e781a05ad54e179b491b3fcde2ff2a8ed83f4a9d570073d876d95a`. Reviewed configuration SHA-256: `224c0de10715bdc2248653e4d4c0803aa510823e2efbfba44e2a2229f3b156e6`.

## First implementation review

The first implementation pass inspected `magset/adapt_breakhis.py` at SHA-256 `2b4b5473c4e269eb69141b7ecf0fc52ae042069fca1f52716d1cd32d4b012798` and its accompanying tests. Its data path filters fitting images before adaptation and constructs the exact fitting-plus-validation list before canonical extraction. Checkpoint context binds roles, base/config/source identity and reference artifacts. The current optimizer includes intended encoder groups from the beginning; phase changes set the full reader to evaluation mode and enable only its last block/final normalization. The downstream head uses the same four C choices, fitting-only field moments, full-field pooling and mean/mask representation as the frozen control.

Independent helper checks, without any optimizer step or encoder forward, confirmed that actual trial 1/seed 812 fitting weights sum to one and each subtype has exactly 0.125 total mass. Structural toy checks confirmed the intended phase-specific gradient whitelist and evaluation/training mode boundaries. These are implementation checks, not performance results.

Two repairs are needed before execution acceptance:

1. **Nonfinite reload/probe values:** restored fragments are checked for keys/shapes but not finite values; a temporary structural fixture confirmed that a NaN encoder fragment is accepted. Initial-base and final-reload probe comparisons use only `error > tolerance`, which also accepts NaN. Require finite, shape/dtype-valid fragments and finite probe outputs/differences before any successful smoke claim. Production extraction has a later finite-feature check, but the fitting-only smoke returns before that stage.
2. **Strength of freeze/adaptation proof:** the final combined allowed-fragment hash proves that some allowed tensor changed. Add explicit head-only-epoch encoder invariance evidence and separate last-block/final-LayerNorm change checks. This makes the actual execution proof match the specification rather than relying only on a combined hash and source-level phase logic.

The inspected tests cover sampler mass/forbidden rows, whitelist changes after warmup, context binding, deterministic flips, epoch-boundary resume, and case-head reload. The reviewer did not run their optimizer/fitting paths under the no-refit task restriction. The planned real fitting-only smoke remains a distinct execution gate.

## Repair review and scoped acceptance

The revised implementation at SHA-256 `f9134b6c4d166123e77f630f73dacf1367671bf49c45a42a1ee3ec51d2a0bac7` repairs the demonstrated nonfinite-fragment/parity gaps and strengthens the freeze proof. Restored fragments reject nonfinite values. The shared parity helper checks shapes and finite values before subtraction, and is used for starting feature parity, final encoder reload, and case-head reload. Each epoch records last-block and final-LayerNorm hashes; head-only warmup checks that both remain at their original hashes, and final acceptance requires both components to have changed while the prefix hash remains fixed. Resume also binds original frozen/allowed-state fingerprints.

Three non-fitting tests passed independently: exact class/case sampling mass and forbidden rows, fragment/context/NaN guards, and stable hashed augmentation. The reviewed test file has SHA-256 `7696975aaee33687136659f59d6e4aebde7bf716d9b3a15dfd8a72dbb0655eff`. The optimizer/warmup, synthetic fitting, and resume tests were inspected but were left for the main agent to execute under this reviewer's no-refit restriction. No real-image training result is implied by any of these checks.

The latest reviewed specification SHA-256 is `1c99d17441b5a7f10e684aa7a11634b3059bc9a80b4ccc8b642ed81eb8c328e4`; its additions explicitly retain the weak field-supervision assumption and the four fitting-only starting feature probes. The registered configuration hash remains unchanged.

Final precision hardening is present in committed source `3a8dde70d809b115928bf773511a0cd6e0f8e30c`; the adaptation module SHA-256 is `0673f9d90483a71a49d3f6b3f0a021c6a800054821fee1e64191c2ab6a6a6c00`. It requires restored tensor dtypes to match their target parameters and asserts that the real base reader's parameters are float32. An independent no-forward/no-optimizer fixture confirms rejection of a wrong-dtype fragment. The reviewed scheduler script sets deterministic CUDA workspace configuration before importing Torch, pins the source/config/protocol context, separates smoke and production output, and caps the ten-task array at four workers.

The main agent reports six focused tests passing and 66 full-suite tests passing before the final precision guards. These broader optimizer/resume checks were not independently rerun by this reviewer; independent checks are identified above. Runtime remains pending at this handoff.

**Scoped acceptance:** the registered protocol and revised split/weighting/freeze/reload/downstream source are suitable for the planned bounded fitting-only technical smoke. No scientific or split-leakage blocker remains in this source review. Production launch requires passing the complete relevant tests, successful real GPU smoke, and scheduler/source safeguards. Production-result acceptance additionally requires adaptation-specific artifact verification. The generic score verifier already checks labels/splits/metric arithmetic; a complete ten-run collector must additionally verify the fixed training budget, component hashes, cache/checkpoint/head provenance and pairing to the correct frozen controls before any adapted result is accepted or ranked.

Actual adaptation quality, all ten production outcomes, clinical grouping, independent confirmation, and publication novelty remain unresolved. The reviewer has made no source edits, performed no optimizer updates or additional scoring, and launched no cluster jobs.

## Technical smoke artifact audit

The fitting-only technical smoke from SLURM job `4962114` completed and its saved artifacts are independently checked. Using the trusted manifest/protocol, original frozen reference, and training-source files verified against both the source archive and Git revision `3a8dde70d809b115928bf773511a0cd6e0f8e30c`, the new collector's `verify_one` and budget/context gates pass on the actual local artifact.

The run has two epochs with 32 fitting-image draws each, head-only then encoder adaptation. The starting four-field feature difference is `1.3113021850585938e-06`, below the registered `1e-4` tolerance, and final encoder reload difference is exactly zero. Recorded component hashes show the last block/final LayerNorm unchanged through warmup and changed afterward, with the prefix preserved. The saved fragment contains the canonical 20 last-block/final-normalization tensors, totaling 12,600,320 float32 parameters. The folder contains no validation predictions, production metrics, or adapted fitting/validation cache. This is execution evidence, not a validation score.

The reviewer also inspected the actual saved AdamW state independently of the collector. Encoder/head rates are `1e-5`/`0.001`, weight decay is `0.01`, and the saved optimizer counters show exactly two updates for every encoder parameter and four for each auxiliary-head parameter. Parameter groups, moment shapes, and fragment identities are consistent with the intended two-batch-per-epoch smoke.

| Artifact | SHA-256 |
| --- | --- |
| Private smoke JSON | `561ff9e6480458b9ad64858ff16d60fac9af88080670bc2570052ef7572188ae` |
| Actual smoke checkpoint | `21ee668fc45f1eaba2ea0b1aebc6f93846cfa5b76d7c8f9b07cd9927eb14055d` |
| First reviewed adaptation collector | `fce26fbb9d81b2ecff81c117ebcc007a412530db92e26b722db942f3663d89d3` |

Two non-fitting collector tests pass independently: budget/warmup/fragment corruption guards and exact feature coverage/finite-value guards. The third test includes synthetic case-head fitting and was inspected without execution by this reviewer. The main agent reports all three collector tests and the full 69-test suite passing.

## Collector hardening before accepting production results

The first collector correctly checks trusted roles/truths, registered source/configuration, complete ten-run coverage, epoch/draw/phase histories, finite losses, class/case/view consistency, checkpoint/hash bindings, changed encoder components, initial/final feature parity, exact allowed adapted-cache paths, fitting-only normalization, case-head reload, and all saved validation metrics. It also requires the smoke and rejects a partial ranking.

Two remaining integrity gaps were reproduced with temporary in-memory mutations, without changing the actual artifact or fitting any model:

1. **Optimizer state is ignored.** The budget verifier accepted a checkpoint whose optimizer had no state and a learning rate/weight decay of 99, because it relied on reported epoch/draw histories. Require the exact two AdamW groups, registered rates/decay/defaults, complete parameter/state mapping, finite correctly shaped moments, and exact update counters. For production with `B=ceil(N_fitting_fields/16)`, the auxiliary head must have `6B` updates and the adapted encoder `5B`; the smoke requires four/two respectively.
2. **Fragment schema is only prefix checked.** The verifier accepted a coherently rehashed fragment retaining only one last-block tensor and one final-normalization tensor, with its own reduced claimed parameter count. Require the complete canonical 20-name schema, shapes derived from the fixed 24-block/1,024-width/4×MLP architecture, float32 dtype and exact count. Prefix membership alone does not prove that a fragment can reload the specified model.

These defects do not invalidate the genuine smoke, whose complete fragment and optimizer values were independently confirmed. They concern acceptance of malformed future artifacts, and can be repaired in the collector while the unchanged fixed production jobs run. They do not justify changing the training recipe, adding fits, or selecting outcomes. Final production-result acceptance remains conditional on repairing these collector gates and verifying the complete ten-run matrix.

### Collector repair verification

The revised collector at SHA-256 `46f8e18339008e9b33b3b7d363698db2d9fc0b8d14cccc5dfb74aab669887aa8` repairs both findings. Its schema independently derives the exact 20 encoder tensor names, insertion order and shapes from the audited architecture. It also checks the two auxiliary-head tensors, exact AdamW parameter groups/defaults/learning rates/decay, all 22 optimizer-state entries, exact encoder/head update counts, finite correctly shaped float32 moments, and nonnegative second moments. The comparison ledger now retains manifest/protocol/configuration and smoke hashes, paired frozen-reference hashes, and distinct validation case counts without publishing provider identifiers.

The actual smoke passes the repaired collector again against trusted local metadata and Git/archive-bound source. Three non-fitting regression tests pass independently. Nine additional in-memory mutations of the actual loaded artifact were rejected: wrong learning rate, missing optimizer state, wrong encoder update count, wrong head update count, nonfinite first moment, negative second moment, reordered parameter IDs, coherently rehashed truncated encoder fragment, and coherently rehashed wrong tensor shape. No original artifact, source file, training job, optimizer state on disk, or validation outcome was changed by these checks. The reviewed collection-test SHA-256 is `a7fcf520a02c77e1ea5d33fffcc901cd8abc7bf1851a7340d0c1df64310c5e40`.

**Current acceptance:** the fitting-only technical smoke is accepted as an execution/reload/freeze-boundary check, and the repaired collector is accepted for verifying the complete registered production comparison. No remaining collector blocker was found in this review. The ten production runs and any improvement claim remain unaccepted until the full matrix passes saved-artifact verification; the fixed recipe and modest exploratory claim scope remain unchanged.

## Complete production-result audit

The complete ten-run comparison is now available. The main agent reports all tasks of production array `4962271` completed successfully, followed by the repaired collector job `4962297` completing successfully. The reviewer independently checked the copied ten reports, their started contexts, validation predictions, saved linear heads, frozen references and paired comparison against the trusted manifest/protocol.

Every adapted run has the exact 46 fitting/eight validation/28 excluded memberships of its frozen reference, the same validation truths, and all 15 subsets with eight cases each. All 1,200 adapted records and the corresponding 1,200 frozen records were independently recomputed using scalar confusion matrices, canonical eight-class probabilities and trusted labels. All per-subset F1/accuracy/balanced-accuracy values, per-trial and split summaries, class recalls, confusion matrices, Brier scores and paired deltas agree with the comparison JSON within `1e-12`. The two splits contain 22 and 20 distinct validation provider cases, with a union of 31.

The audit also verified source/configuration/manifest/protocol and frozen-reference bindings, exact six-epoch histories and fitting-field draw counts, warmup/final component-hash metadata, saved head hashes/shapes/class order/finite values, selected-C grid/tie behavior, and all locally available ledger references. Production adapted caches and encoder checkpoints remain on the cluster; their full-content shape, optimizer-state, normalization and probability replay checks are recorded by the already reviewed remote collector rather than freshly repeated in this local audit. This distinction does not alter the independently reproduced saved-score arithmetic.

| Inner seed | Frozen mean-subset macro F1 | Adapted mean-subset macro F1 | Paired mean F1 change | Frozen mean-subset accuracy | Adapted mean-subset accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3842777778 | 0.3840000000 | −0.0002777778 | 45.6667% | 45.6667% |
| 813 | 0.3168888889 | 0.3205000000 | +0.0036111111 | 38.3333% | 38.8333% |

Full-view accuracy remains 47.5% for seed 812 and changes from 37.5% to 40.0% for seed 813. The mean of per-trial worst-subset F1 decreases on both arrangements, from 0.258333 to 0.245833 and from 0.247500 to 0.239167 respectively. Both methods retain zero phyllodes recall on these validation records. These additional endpoints are descriptive and do not override the primary decision rule.

Only 11 of 600 saved case/subset decisions change for seed 812, with 274 correct before and after. For seed 813, 27 decisions change and correct records rise from 230 to 233 of 600. These repeated masked records are dependent observations, not hundreds of new patients. The small net gain does not justify a claim of a substantial accuracy advance.

Private paired-comparison SHA-256: `c26398dcfbeac6ac76f737ab686a987f82c2ea6874a45132fe9a0a7fc8b5637b`. Its source, configuration, collector and smoke hashes match the reviewed execution artifacts. No additional fitting, image encoding, outer-role prediction, or new model selection was performed by this reviewer.

**Result acceptance and decision:** the complete fixed adaptation experiment is accepted as a reproducible exploratory comparison. It **fails the predeclared requirement to improve the primary mean-subset macro F1 on both split arrangements**. Retaining the frozen mean baseline as the project reference is supported. This is evidence against adopting this particular fixed adaptation package; it does not prove that all adaptation methods fail or that a numerical performance ceiling has been established.

The project still lacks strong demonstrated eight-subtype generalization: rare fitting subtypes each have one provider case, clinical identifier relationships are uncertain, validation is tiny and adaptively reused, and no globally fresh BreakHis confirmation cohort remains. More GPU work has no supported fixed ETA to 80% or 90% accuracy. Historical image-split or published binary-task headline scores cannot provide that ETA or an apples-to-apples success threshold. Novelty, reliable clinical performance and publication readiness remain open.
