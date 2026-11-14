# BreakHis supervised CNN reference: independent review

Prepared November 13, 2026. Scope is the fixed supervised ResNet-50 reference in `configs/breakhis-supervised-v1.json` and `docs/BREAKHIS_SUPERVISED_SPEC.md`, followed by implementation review before training. The reviewer may edit only this file; no fitting, cluster jobs or outer-role scoring are performed here.

## Protocol assessment

The proposal is an ordinary ImageNet-initialized, fully supervised image classifier with one fixed probability-averaging case readout. It uses the same ten provider-case development arrangements as the accepted frozen reference: five trials at inner seeds 812 and 813, with 46 fitting/eight validation/28 excluded cases per run. All 15 nonempty view subsets are evaluated. Exactly ten production fits and one separately identified fitting-only technical smoke are planned.

This is a defensible bounded reference experiment after the observed fit–validation gap. It tests a **complete alternative pipeline**. Backbone, trainable parameter scope, pretraining, whole-field preprocessing, augmentation, regularization, duration, checkpoint selection and aggregation all differ from the prior Phikon pipeline. A result cannot isolate the benefit of any single ingredient or establish a new fusion method. “Smaller encoder” refers to total architecture size; the number of trainable weights must be reported separately and is not necessarily below the earlier last-block adaptation.

The specification correctly acknowledges that four checkpoint choices and four C choices are different optimization/search budgets. These are matched task/role comparisons, not an identical-budget fusion ablation or an exact published-method reproduction. Public author CSV/fold incompatibilities remain a separate audit; no author implementation should be executed merely to create this standard reference.

## Required model and BatchNorm behavior

Verify the full cached pretrained checkpoint checksum, rather than only a filename prefix. Build the declared ResNet/head architecture, verify canonical class order and float32 parameter dtype, and bind the torchvision/environment revision. Do not substitute another pretrained checkpoint or transform when an artifact is unavailable.

During epochs 1–2, only Dropout/Linear head parameters may update. From epoch 3, all backbone parameters, including BatchNorm affine scale/bias, may update. Put the model in training mode, then explicitly reset every BatchNorm module to evaluation mode on each phase/epoch. Calling `model.train()` alone reactivates BN statistics; calling BN `.eval()` alone does not freeze its affine gradients. Both controls are needed.

Hash and compare all BN `running_mean`, `running_var` and `num_batches_tracked` buffers across warmup, unfreeze and final reload. Require unchanged early/late backbone parameters through warmup and changed early/late parameters after unfreezing. The full backbone must actually be included in optimizer groups even while its gradients are disabled during warmup. Full model-state validation must allow integer BN counters while requiring floating model tensors to retain their declared dtype and finite values.

The fitting-only technical smoke has three epochs and at most two batches each: two head-only phases and one actual backbone phase. It must show finite losses/gradients, the intended group membership, unchanged BN buffers, genuine early/late weight updates and saved-model reload parity. It supplies software/runtime evidence only, with no validation decoding or model ranking.

## Training distribution, transforms and schedule

Preserve the audited sampling distribution: each class has mass 1/8, each fitting case equal mass within its class, four views equal mass within a case, and fields uniform within a view. Use label-smoothed cross-entropy without an additional inverse-class multiplier. Draw exactly the number of fitting fields with replacement each epoch, keep the final partial batch, and record class/case/view counts and exact optimizer updates. Repeated fields remain dependent case-derived labels, not new independent subtype examples.

Per-draw augmentation needs an actual draw index, not just the underlying image index returned by an ordinary weighted sampler. Use deterministic sampled-index/draw-index pairs or a precomputed epoch draw list. Hash seed/epoch/draw/path independently of worker RNG, so repeated draws can differ while worker count and epoch-boundary resume preserve results. Verify the declared H/V flip, quarter rotation and fixed brightness/contrast/saturation order, including factor bounds. Validation and fitting evaluation use canonical unaugmented fields.

Whole-field bicubic resizing to 224×224 deliberately differs from the old cropped ResNet cache. It retains image extent but changes the original rectangular aspect ratio; record that geometric warp rather than implying native-shape preservation. No crop/stain-template/data-derived normalization may be silently added, and the old frozen ResNet cache cannot substitute for raw canonical images under this transform.

Specify the cosine multiplier as an exact epoch function. Epoch 3 must have factor 1 and epoch 20 factor 0.1; warmup head LR stays at 0.001. Log backbone/head rates for every epoch and verify them on resume. The last checkpoint's head optimizer counter should equal `20*ceil(N_fit/32)` and backbone counters `18*ceil(N_fit/32)`; the three-epoch/two-batch smoke should show six head and two backbone updates. Loss/gradient nonfiniteness should fail explicitly.

Label-smoothed minibatch training loss differs from ordinary canonical evaluation NLL. Save stable log probabilities and label the evaluation CE convention separately; do not compare the two quantities as if they were identical objectives.

## Selection, checkpoint storage and resume

Validation is permitted only at the four predeclared epochs 5/10/15/20. Run all 20 epochs despite interim scores. Use only equal-subset mean eight-class macro F1 for selection, retaining the earliest exact tie. No temperature/threshold/calibration/readout choice is allowed. Probability averaging must give equal mass to each field within a view, then equal mass to available views; a flat mean over all fields changes the task when counts differ.

Save all four candidate score/prediction sets so the selected epoch and tie rule can be checked without accepting a reported scalar. Candidate opportunities do not add patients or clinical replications. Keep the selected model as a real copied snapshot: a shallow `state_dict()` mapping can share tensors with later training and silently turn an earlier selected epoch into the final model.

Keep one complete last resumable state and one selected model-state artifact. The last state must preserve current model/optimizer/RNG/history plus enough selected-state information for exact recovery. Handle interruptions around selection writes explicitly: a newer selected file paired with an older last checkpoint must not change selection history or lose the previously chosen model. An atomic authoritative checkpoint containing the selected snapshot, with a separately exported selected artifact, is one valid implementation. State the resulting bounded storage budget.

At final output, load the chosen model into a fresh declared architecture and check four fixed fitting-only probe fields. Preserve the true epoch-20 resumable state separately from a selected earlier model. Bind both artifacts to source/configuration/pretrained hash/manifest/protocol/training seed/exact roles and selection metadata. Resume may repeat a failed partial epoch from a completed checkpoint but may not extend the scientific 20-epoch budget or replace an unfavorable completed run.

## Data, results and acceptance limits

Build fitting and validation image allowlists from trusted regenerated memberships before opening images. Own-run outer cases must never be decoded/scored by the fitted model. Recheck image hashes, dimensions, labels, duplicate/missing paths, all four views and exact final field-score coverage. The all-view/missing-view case records must be reproducible from saved field probabilities/log probabilities, with no refitting. Public results should contain aggregates and hashes, not private provider identifiers or image paths.

The complete ten-run matrix must pass artifact verification before ranking. Report every trial, both split means, all 15 subsets, selected epochs, fitting–validation gaps, raw and class/case/view-balanced field metrics, subtype recall/confusion, Brier/CE, runtime/model size and all negative outcomes. Do not choose a favorable split or cite an unrelated binary/image-level author accuracy as this method's target.

The predeclared screen is at least +0.02 mean-subset macro F1 on both arrangements and positive phyllodes case recall on both. This is an operational descriptive adoption rule, not a significance test or clinical threshold. Even one correct repeated case/subset event can produce nonzero rare-class recall; require the associated distinct case/support counts in the report. One fitting adenosis and one fitting phyllodes case per run, adaptive validation reuse, overlapping trials, uncertain clinical identity and no globally fresh confirmation cohort remain unchanged. Failure does not justify an automatic backbone/rate/epoch sweep.

**Current status:** the registered protocol is acceptable as a bounded standard reference, subject to the implementation, meaningful tests and real fitting-only smoke gates above. Source and actual outcomes remain pending review. No performance gain, reliable clinical generalization or novelty is asserted.

## Implementation review and repair

The initial implementation of `magset/supervised_breakhis.py` correctly implements the principal scientific controls: fitting-only weighted draws with real draw indices; deterministic fitting augmentation; canonical full-field image processing; optimizer groups present before unfreezing; head-only warmup followed by full parameter gradients; BN modules explicitly in evaluation mode; fixed BN buffer and warmup-backbone hashes; exact epoch-dependent rates; one stable probability-mean readout; four fixed selection opportunities; and no outer input list. The technical-smoke branch has no validation decoding or checkpoint selection.

One storage/resume defect was identified before training acceptance. The first last checkpoint retained only best-candidate metadata while selected weights lived in a separately overwritten file. An interruption between the two writes, or resuming from the last file alone, could lose the earlier selected state or leave incoherent epoch provenance. This needed repair even though final score replay would catch many inconsistent outputs.

The revised source makes the last checkpoint authoritative: it contains a CPU-cloned best-model snapshot and its recorded state hash, commits atomically before exporting the selected file, restores/reconstructs that file on resume, and checks selected epoch plus state digest before final scoring. It also verifies complete epoch and candidate coverage and the earliest-maximum rule. The specification records roughly 5 GB steady storage for the ten last/selected artifact pairs; temporary atomic-save copies require some additional space.

Three non-fitting tests pass independently: deterministic draw-index augmentation, cosine schedule endpoints, and equal-view probability aggregation with forbidden outer roles/invalid coverage. Separate no-forward/no-update checks confirm head-only versus full parameter-gradient flags, BN evaluation mode with head dropout in training mode, and the actual 20-epoch schedule's correct endpoints/monotonicity. The reviewer inspected the optimizer/warmup and interruption-resume tests without executing their fitting paths under this task's no-fitting restriction. The latter test explicitly corrupts an orphan selected file and requires reconstruction from the last checkpoint plus bitwise equality with uninterrupted toy training.

**Scoped source acceptance:** the registered recipe and repaired source are suitable for the bounded fitting-only GPU smoke. No remaining scientific/split-leakage/source blocker was found in this pass. Production still requires successful complete tests, genuine smoke evidence, immutable scheduler/source/weights bindings and the declared concurrency limit. Production-result acceptance additionally requires the complete ten-run artifact check, including BN and optimizer histories, all four candidate field sets, selected epoch/state identity and field-to-case score replay. Source/model/transform/environment revisions and both total/trainable parameter counts should accompany the actual outputs. No model fitting, optimizer updates, raw research-image processing or cluster job has been performed by this reviewer.

### Committed source and provenance

The accepted source snapshot is `6b418598b3334dce2e81af02a8f035278e08ef12`. The reviewer checked that the working module, tests, configuration and specification match their committed Git objects. The revision additionally pins torchvision 0.21.0 at model construction and records Python/Torch/torchvision/Pillow/NumPy and GPU provenance. The scheduler pins the immutable source, external weight path and deterministic CUDA workspace, uses one GPU per task and caps the ten-task production array at four workers; its fitting-only smoke uses a separate output.

| Reviewed artifact | SHA-256 |
| --- | --- |
| Supervised module | `a611fd078501ad89dc69ad00f6b5b3e7d36b62897f59df50696336c8fb669f97` |
| Tests | `6bb96b1cecf93239b2455c4d3a64693259bc4b08f74760c9631f87158fc352fc` |
| Fixed configuration | `d239cff3df4fcca16e64b9bf4f36309a5c8ea923c931198cd1a4b34f914b5839` |
| Final specification | `64fee4b449dc6f0ca32e951290d68b4ae90727b206e56baad7697fc1ac0a0172` |

Per-epoch BN checks cover normal runs. The final artifact checker must also require last/selected BN buffers to match the original pretrained buffers, including an already-completed-checkpoint resume where the training loop is skipped. Runtime/selection evidence remains pending; these source checks do not claim that real training or predictive improvement has occurred.

## First technical smoke and bounded cleanup repeat

The main agent reports that technical smoke `4969999` under the original `6b41859` source completed with valid six-update/reload evidence but printed worker-shutdown messages, including “terminate called without an active exception.” It was explicitly **not accepted as the production gate**. Its source/logs/checkpoint are retained, and a single fitting-only technical repeat is disclosed separately from the ten production fits. This changes no model setting, validation opportunity or production fitting budget; both technical fits must remain in the execution record.

The cleanup revision limits the technical sampler itself to exactly 64 draws per epoch and exhausts it normally, rather than constructing an oversized loader and breaking its iterator after two batches. Production still uses all N fitting-field draws. Independent no-training checks confirm that the default production draw sequence matches the prior N-draw implementation and that the bounded sampler yields exactly 64 sequential draw IDs before exhaustion. Source inspection also confirms an explicit final BN equality check after a completed-checkpoint resume.

The revised training source and original collector are committed as `c1b6b0ce24ffc99a00b3eb98cc09130934ffb0bb`. The training module SHA-256 is `dc6f13bc4133ce88cb159b04a8a2fa34ca86d5f41bee1e1e2ec3e74cc4af6318`; the recipe configuration remains `d239cff3df4fcca16e64b9bf4f36309a5c8ea923c931198cd1a4b34f914b5839`. The updated specification SHA-256 is `ca01532a213e04451bd0e34c521a25659b8b723131e687639fbf9050eee999bf`.

## Supervised collector review

The collector derives the complete expected model schema/dtypes from the checksum-verified original pretrained state, replacing only the declared eight-class head. It checks unchanged BN buffers in final/selected states; exact optimizer parameter order, default settings, scheduled final rates, finite matching moments and update counts; every epoch's draw/class/case/view/phase/rate history; warmup and final component hashes; authoritative best-snapshot recovery; all four candidate field sets and earliest-maximum selection; exact role/path/truth coverage; stable log/probability arithmetic; final field-to-case probability means; all stored metrics and prediction records; and paired frozen references. The full comparison requires all ten unique runs, keeps distinct case/class support counts, and records both technical fits, with the first rejected for production cleanup.

Three collector tests pass independently without fitting: BN/optimizer/schema/trace corruption rejection, field path/log-probability integrity, and nonfinite/metric-key validation. The reviewed collector SHA-256 is `93d8880da09b5ac1f6d2122bb0eb52e3ef3deebfe2146aec6cc3605b3ef89860`; its tests have SHA-256 `13d0f1e5c2e233fdc2fbbbfc1e617f7542c3b023d07c2ab1647ac89c1cea7941`. This is the reviewed working collector with subsequent production parameter-count and comparison-lineage additions. The immutable `c1b6b0c` collector used for the smoke has SHA-256 `0b4326e3caf602d880d0a84eb185d7817b02c919d5ace543d08d11974a124487`; those later changes do not alter its smoke-verification path.

**Current gate:** no remaining source/collector blocker was found for the bounded cleanup repeat. Its actual saved state and optimizer evidence must pass verification, and its error log must be inspected separately and recorded as clean before production. Artifact-state verification alone cannot establish worker cleanup; the first smoke demonstrates why these are distinct gates. Complete production-result acceptance remains pending all ten outputs and their checked candidate/selected artifacts. No training settings were changed or model fitting/jobs performed by this reviewer.

## Clean retry artifact acceptance

The actual fitting-only retry `4970065_0` under `c1b6b0ce24ffc99a00b3eb98cc09130934ffb0bb` is now independently checked. Its local started context and checkpoint match the trusted manifest/protocol, exact trial 1/seed 812 fitting/validation/outer assignments, fixed recipe, original frozen reference and Git/archive source. No selected checkpoint, candidate score set, validation prediction or final field-score artifact exists in the technical folder.

Using torchvision 0.21's architecture on the meta device solely as a shape oracle, the reviewer verified all 320 saved state tensors, their order, shapes, dtypes and finite values. The model has 23,524,424 parameters, of which 16,392 belong to the head trained during warmup. All 159 BN buffers retain the recorded initial fingerprint. A bounded read-only artifact query independently verified the original pretrained file's complete SHA-256 and all four original component fingerprints, so the BN/initialization comparison is tied to the actual pinned input rather than only repeated report strings.

The checkpoint contains exactly three completed epochs with 64 draws each, no selection candidates, and no best state. Reconstructing the deterministic sampler from trusted fitting rows reproduces every recorded class/case/view count exactly. Phases are head-only/head-only/full-backbone; early and late hashes remain original during warmup and change afterward. All 161 AdamW parameter states have the declared group/order/settings, finite matching float32 moments, nonnegative second moments and exact counters: two updates for every backbone parameter, six for each head parameter. The recorded four-field reload logit difference is zero; this was not rerun as a new image forward by the reviewer.

The reviewer separately inspected the small remote log/hash evidence. The retry error log is empty (zero bytes), with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The remote immutable checker reports the smoke verified, and its smoke/checkpoint hashes match the copied files.

| Accepted retry artifact | SHA-256 |
| --- | --- |
| Private smoke JSON | `d0a6844545e6737eb251a662f1daf1fa4b387bf17d15452046f0569a87cf9880` |
| Actual last checkpoint | `66858ad0db81e1df297efd7af27dc5aad8510d3b48c7e17e8215b91f30ea8486` |
| Empty retry stderr | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

**Smoke acceptance:** the bounded retry clears the fitting/unfreeze/BN/reload and clean-worker-shutdown gates. The earlier warned smoke remains a separately recorded technical fit, not a production model or selected result. The production array may proceed under the registered ten-run recipe and concurrency policy; no predictive gain or rare-class improvement is accepted until the complete matrix passes result verification. This review used existing tensors, metadata and read-only artifact hashes only: no training, image decoding, encoder/model forward, optimizer update, job submission or unrelated-job action was performed.

## November 14: technical recovery audit

Eight original production tasks completed, but tasks 0/1 of array `4970079` timed out at 7,218 seconds each. Their committed checkpoints were at epochs 20 and 19. Operations wrapper `d7cbe25be36d0137bc6166f47f29569a4b5f50e4` preserved and hashed the interrupted checkpoints, selected states, candidate arrays and orphan temporary files before A40 continuation `4971209`. Both A40 tasks failed the unchanged selected-validation-field probability replay safeguard. Their recorded primary scalar F1 replay passed; that alone does not prove equality of every hard decision, subset metric or probability. No magnitude beyond the fixed tolerance failure was inferred from the logs.

The rollback wrapper `752441c0fbe00dca90f184888ce8ee2ad15d6e38` preserves the rejected A40 artifacts first, validates the saved original receipt bindings, verifies each original preserved file digest, and restores last/selected/candidate files through checked staging copies. It excludes orphan `last.tmp` from resume authority. The original immutable C1 training source then reconstructs selected state from authoritative `last.pt`; settings, roles, 20-epoch limit and four candidate opportunities are unchanged. The A100-family constraint is an attempted compatibility remedy, not proof of the sole numerical cause of the A40 failure.

All four copied receipts match the remote receipt hashes. For both trials, every restored A100 last/selected/candidate digest exactly matches the pre-A40 receipt, and `started.json` remains unchanged. Trial 1 remains at committed epoch 20: its recorded `next_epoch=21` starts an empty training loop and is not an additional epoch. Trial 2 restores original epoch 19 and repeats only the uncommitted final epoch. The A40 epoch-20 state and its changed candidate file are preserved separately rather than becoming the final model's initialization. Both original selected epochs were 10; no performance-driven replacement or altered checkpoint rule was found in this recovery chain.

| Receipt | SHA-256 |
| --- | --- |
| Trial 1 before A40 | `cd25bcb2ba67f4b8f3a97ad373e7f1fde57c4c0ce6db50386504a4f1859e9579` |
| Trial 1 A100 preservation/restoration | `0977ce0c008fa6911b755359daf11cb2120b3252e710a410e96ca60624bc1032` |
| Trial 2 before A40 | `34ea767874bc2020ee886ee3285c9ee2c367e6f42a994d65f640c184e0d0a607` |
| Trial 2 A100 preservation/restoration | `b7dff4e52d2f4f9eb59a13fe6f93dca78bb497c22bfe15700399e27886f68534` |

The live scheduler command points to the immutable `752441c` wrapper, whose remote bytes match the Git object (file SHA-256 `6fd260fbed27e38a91071b576c49cd2de3ef115c53fa16279f8218e6258ef5d7`). Read-only accounting/log inspection subsequently confirms A100 tasks `4971227_0/1` completed with zero exit status in 119/204 seconds and empty stderr. Both reached final-artifact output after replay. Full collector `4970104` then completed with zero exit status in 58 seconds. No checkpoint download, fitting, forward pass or job mutation was used in this operations review.

The recovery is accepted as a transparent technical continuation/rollback, conditional on the unchanged complete-result gates below. The wrappers are operational changes, not changes to the trained scientific recipe. Their generic receipt reason mentions the original timeout; the execution record must also retain the rejected A40 replay failure and restoration lineage so the second continuation is not mistaken for an ordinary first resume.

### Runtime accounting

Trial 1 consumed 7,218 + 69 + 119 = **7,406 allocated GPU-task seconds** across original/A40/A100 segments. Trial 2 consumed 7,218 + 127 + 204 = **7,549 seconds**. This includes discarded/repeated work, including repeated epoch-20 processing for trial 2. The final training report's `elapsed_seconds` measures only its final invocation and must not replace these totals.

All ten original segments plus four recovery segments total **22,747 single-GPU allocated task-seconds** (about 6.32 allocated GPU-hours). Both disclosed technical smokes add 73 seconds, giving 22,820 seconds overall. These are allocation durations, not measured GPU-kernel utilization, and do not include queue/idle gaps or the separate CPU collector. The eight uninterrupted models took 856–1,049 seconds each. Report active resource cost and elapsed calendar time separately; no new scientific epoch allowance is created by repeating failed/uncommitted work.

## November 14: complete numerical result audit

The ten-run paired comparison is now available with SHA-256 `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`. The reviewer independently checked the copied reports, final fitting/validation field arrays and case records, all four candidate field sets per run, recovery receipts, accepted frozen references and comparison ledger against the trusted manifest/protocol. No partial model ranking was used.

The audit covers **51,760 final field records**, **30,116 candidate-validation field records**, **8,100 final case/subset records**, and **4,800 candidate case/subset records**. These are repeated derived observations across the same cases and trials. Exact path coverage, fitting/validation membership and subtype truths pass; no scored record belongs to its own run's excluded outer group. Per-role class/case/view field weights were independently constructed from original manifest counts.

An independent scalar confusion-matrix calculation and stable log-mean-exp reconstruction reproduce every case/field F1, accuracy, balanced accuracy, CE, Brier score, subtype recall and confusion matrix, every one of the 15 subsets, all 40 candidate score sets, and all paired/cross-trial summaries within `2e-12`. Equal-field-within-view/equal-available-view probability averaging reproduces saved case probabilities/log probabilities within `1.7763568394002505e-15`. Candidate selection obeys the registered earliest maximum. Selected epochs in seed/trial order are `[10,10,20,5,15]` and `[10,15,10,5,10]`.

Available report/prediction/field hashes, source/configuration/data identities, frozen-reference hashes, exact 20-epoch history/draw/phase/rate traces, component-hash metadata, selected epoch and support counts agree with the ledger. The production model and full optimizer/checkpoint tensors remain remote; their complete-schema, BN, optimizer, selected-state and reload checks are the reviewed remote collector's evidence rather than a fresh local tensor recheck in this numerical audit. There are 22 and 20 distinct validation provider cases, union 31; phyllodes has two distinct validation cases per arrangement.

| Inner seed | Frozen mean-subset macro F1 | Supervised mean-subset macro F1 | Mean F1 change | Frozen case accuracy | Supervised case accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3842777778 | 0.3083888889 | −0.0758888889 | 45.6667% | 37.0000% |
| 813 | 0.3168888889 | 0.3006111111 | −0.0162777778 | 38.3333% | 40.0000% |

Full-view supervised accuracy is 40% for both arrangements, versus frozen 47.5%/37.5%. Phyllodes case recall remains zero on both. Mean per-trial worst-subset F1 and Brier score are also worse for this recipe on both arrangements. The seed-813 accuracy increase does not override its lower primary macro F1 or the failed rare-class condition.

The generalization gap persists: selected supervised case fitting balanced accuracy averages approximately 99.55%/99.70%, versus validation 37%/40%; class/case/view-balanced field accuracy is 97.49%/97.84% fitting versus 31.23%/32.63% validation. These are descriptive fitting–validation differences, not a causal diagnosis or an independent clinical performance estimate.

**Final result acceptance:** the complete fixed experiment and its disclosed technical recoveries are accepted as reproducible exploratory evidence. The registered +0.02-F1-on-both-arrangements and positive-phyllodes-recall gate is **false**. Retain the frozen reference; do not adopt this recipe or select a favorable split/secondary endpoint. This does not prove that every ResNet/CNN, longer training or alternative aggregation must fail, nor authorize an automatic sweep. Clinical identity, independent confirmation, reliable predictive performance and novelty remain unresolved. The reviewer performed no fitting, model forward, new image processing, outer scoring, checkpoint download or cluster job action in this result/recovery audit.

The generated complete-results Markdown and public JSON were also checked. Public values exactly retain the accepted comparison plus its correct input hash and execution metadata; all per-run runtime segments agree with the independent scheduler accounting. The report preserves every trial, all 15 subsets, distinct subtype support, the failed gate and the complete-pipeline/clinical limitations. A recursive scan found no provider identifiers or private image/machine paths. No public numerical or identifier-redaction blocker remains in this check.
