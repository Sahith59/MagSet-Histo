# MAGSET-Histo execution plan

## Current user priority: BreakHis quick deliverable first

User clarified on November12,2026 that BreakHis is the main dataset and its project deliverable with Professor Dong must be finished first. This section supersedes earlier next-step language that prioritized TCGA or novelty. Preserve completed TCGA work as deferred background; do not expand it now.

Immediate acceptance contract: `docs/BREAKHIS_MILESTONE_PLAN.md`. First checked baseline results, raw-image replay and short professor handoff are complete. The next fixed experiment is registered in `docs/BREAKHIS_ADAPTATION_SPEC.md`: ten supervised last-block adaptation fits, same cases and downstream mean classifier. Smoke4962114, all ten production fits4962271 and collector4962297 passed. Mean accuracy45.7%/38.8%, primary F1.3840/.3205: the two-split improvement gate failed, so retain the frozen baseline. See `docs/BREAKHIS_ADAPTATION_RESULTS.md`. Saved-model diagnosis4968029 is complete: adapted case heads fit98.5–98.7% balanced accuracy but validate38.8–45.7%; auxiliary image heads fit~98% but validate32–35% with class/case/view weighting. Fixed auxiliary readouts do not rescue the result. See `docs/BREAKHIS_DIAGNOSIS_RESULTS.md`. The coverage/reference audit is complete (docs/BREAKHIS_REFERENCE_AUDIT.md). The fixed supervised ResNet-50 reference completed all ten fits and checker4970104 (58s), after disclosed wall-time and cross-GPU replay recovery. F1.3084/.3006 and accuracy37.0%/40.0%, zero phyllodes recall: the improvement gate failed. Keep the frozen reference. See docs/BREAKHIS_SUPERVISED_RESULTS.md and execution record; independent numerical review passed and the BreakHis project report/professor brief are updated. Agree on endpoint/grouping/credible final evaluation before another registered experiment; no automatic model sweep, resplit or outer rescue. Distinguish baseline execution from reliable final performance and method novelty. No globally pristine BreakHis confirmation cohort remains after development; clinical grouping is unresolved.

The current baseline uses frozen Phikon-v2 fields, fitting-only saved normalization, mean pooling plus availability indicators and a logistic head. Do not imply a novel transformer is the implemented model. Both inner splits and all trials must be reported. The historical roadmap below is retained as research context, not the immediate work queue.

Updated November 12, 2026. This is a living execution plan, not a report of completed experiments. Public research status is recorded here; private working copies also maintain a separate running-state handoff. Checkboxes are completion claims and must be updated only with evidence.

## Objective and scope

Develop and evaluate breast histopathology classification using all available magnifications, including incomplete inputs, and prepare a conference manuscript with Professor Dong. Deliver reproducible code, patient-separated splits, fair benchmarks, evidence for or against the candidate mechanism, and a manuscript whose claims follow from saved runs.

The working project/repository name is **MAGSET-Histo**, meaning Magnification-Aware Set Transformer for Histopathology. Naming does not imply established novelty. The first study is BreakHis eight-subtype classification. A second WSI dataset is needed to support a broader methodological claim. We will not relabel a small-dataset implementation extension as a generally novel method without evidence.

### User requirements

- Use 40X, 100X, 200X, and 400X; do not commit to 40X only.
- Use university GPUs as appropriate; H100/A100/L40/A40/DGX access has been verified at the scheduler level.
- Work toward a strong novel method and conference submission promptly.
- Keep explanations brief and understandable. Put extensive mechanisms and operating detail in files.
- Persist running state, long-term project context, decisions, and handoffs locally and on the cluster.
- Preserve supplied reference materials privately and cite their publications. Do not redistribute them automatically.

## Prediction and data contract

A sample is a verified patient/slide group g, with an observed magnification subset S contained in {40, 100, 200, 400}. For each m in S, input X[g,m] is a set of tissue image fields, not a falsely paired image index. A separate image-level track makes one prediction per original image.

Output is eight scores in a saved canonical label order and the highest-scoring subtype. A binary secondary score is the sum of the four malignant-class scores. Calibration parameters and binary thresholds are fitted on validation data only. Outputs are research predictions, not validated clinical decisions.

Manifest fields: relative image path, original filename, raw case/slide ID, conservative grouping key, grouping-rule version, biopsy method, broad class, subtype, magnification, sequence, file SHA-256, width, height, and byte size. Patient identity must be reconciled with the original protocol before any split freezes. Counts of raw cases and conservative groups must be reported separately.

Important ambiguities to resolve during data audit:

- Does a trailing letter in a case identifier indicate a separate patient, slide, or specimen section? Preserve raw IDs. Do not silently count them as independent patients.
- Does the same numeric patient identifier occur across several raw folders or suffixes? Keep potentially related groups together until verified otherwise.
- Which groups have all four magnifications? Which truly have missing views, and which apparent gaps are identifier or file problems?
- Is there evidence for corresponding tissue fields? Filename sequence alone does not establish registration.
- Do published counts and actual archive contents agree? Generated tables are authoritative for the obtained release; explain discrepancies.

## Evidence and literature gates

- [x] Read every original source cell and all supplied research documents; save `docs/PROJECT_REVIEW.md` and `docs/NOTEBOOK_WALKTHROUGH.md`.
- [x] Verify that the historical paper was published at SPIE Medical Imaging 2021 and names Dong Hye Ye as a coauthor.
- [x] Identify MagViT, M2OTCA, patient-aware BreakHis benchmarking, and recent missing-input methods as necessary comparisons.
- [ ] Read full methods, supplementary material, available code, data grouping, and model-selection details for the closest comparisons.
- [ ] Create a claim-to-prior-art matrix. For every proposed contribution, name the closest existing mechanism and the exact distinction.
- [ ] Determine code availability and reproduction costs; mark reimplementations and adaptations explicitly.
- [ ] Agree on the task, contribution, and venue with Professor Dong before making them manuscript claims.

A targeted search is already enough to reject broad novelty statements such as first multiscale transformer, first missing-token model, first use of cross-scale OT, or first contribution-aware multimodal gating. Novelty is still open. If a recent method already does the proposed mechanism, change or narrow the hypothesis before expensive sweeps.

## Candidate architecture and the underlying mechanisms

### Stage A: strong pretrained image encoding

Initial candidate is `owkin/phikon-v2`, a pathology-pretrained ViT-L encoder. Compare an ImageNet-pretrained encoder to distinguish domain pretraining from fusion improvements. Add `MahmoodLab/UNI2-h` if model access is approved/available. Model terms and exact revisions must be recorded; access approval is not assumed.

Use a shared image encoder initially so the small cohort is not split across four independently trained large backbones. Each original field becomes a feature vector or compact token set. Add explicit magnification identity after image encoding. Preserve native aspect ratio with documented padding/crops and compare original-resolution tiling where relevant. Never confuse camera magnification labels with nominal WSI objective magnification across datasets; use microns-per-pixel when available.

Frozen feature extraction is the fast first phase: encode once, cache immutable features, then compare many fusion methods without repeatedly loading a large backbone. If the encoder is frozen and uses no cohort-specific fitting, features for all splits may be cached; labels/test outcomes must never train preprocessing or adaptation. Any self-supervised adaptation, stain-template estimation, prototype fitting, or fine-tuning is restricted to training patients within each fold.

### Stage B: sets within each magnification

Each magnification supplies several tissue fields. Use a permutation-invariant within-scale aggregator, first mean/attention MIL. Cap image fields per patient/scale consistently during training and define a fixed evaluation policy. Large numbers of views must not make a patient contribute more independent weight than others. Bootstrap and statistical analyses use patients, not generated view combinations.

A learned scale token represents the summary of one magnification. Existing attention over sets is a baseline, not a novelty claim. Match field-count budgets across methods and report full-bag results separately if computationally different.

### Stage C: baseline cross-scale fusion

Implement masked mean pooling, late prediction averaging, learned masked gating, and a small set transformer. The availability mask must exclude missing views, including from keys/values and from any normalization/aggregation. Padded image contents must not affect prediction. At least one observed view is required; all-missing inputs return an explicit error/abstention.

A learned missing token and exclusion masking are different methods. Test them as alternatives. Do not claim missing content has been recovered when a placeholder is used.

### Stage D: candidate availability-conditioned partial evidence transport

Provisional research mechanism, pending the full novelty audit:

1. Represent each observed magnification by a small set of learned morphological evidence prototypes obtained from its image-field embeddings.
2. For observed scale pairs, construct a feature-similarity cost matrix in a common projected space.
3. Use partial/unbalanced transport to permit some evidence to match and some to remain unmatched. Do not force every coarse tissue pattern to have a corresponding cellular detail.
4. Preserve both transported/shared evidence and unmatched residual evidence rather than discarding the residual or forcing entire scale vectors to be orthogonal.
5. Condition fusion on availability and matched mass/reliability summaries. When one view remains, bypass cross-scale transport and classify its available evidence through a supported single-view path.
6. Train under a documented distribution of observed subsets and test every nonempty subset.

This differs in intended scope from naive paired-view fusion, but OT and cross-magnification matching are already established, notably M2OTCA. Any paper must identify what, beyond those components, is actually new and useful. The candidate claim to test is whether explicit preservation of unmatched evidence under changing availability improves unpaired-field fusion. Semantic feature matching is not physical cell registration and is not a causal biological explanation.

Provisional loss:

`L = mean patient-balanced CE over sampled nonempty subsets + lambda * optional auxiliary term`

Start with lambda=0. Add only one auxiliary mechanism at a time. Candidate additions include subset robustness objectives or a consistency term for observed evidence, but generic consistency/robust optimization already has prior art. Do not constrain every missing-view prediction to equal a full-view teacher with unjustified confidence: some missing views genuinely remove necessary evidence.

Implement log-domain/stable transport with fixed validation-selected iteration budgets and regularization. Test empty pair sets, unequal token counts, collapsed embeddings, nonfinite costs, gradients, one-view bypass, field permutations, and missing-view invariance. Compare against ordinary cross-attention and balanced OT to isolate the contribution. If partial transport hurts, publish the outcome only if it supports a meaningful analysis; do not hide negative ablations.

### Stage E: adaptation only after fusion evidence

Use adapters/LoRA or selective layer unfreezing after the frozen-feature comparisons. Keep backbone identity and adaptation policy fixed when comparing fusion mechanisms. H100/A100 resources enable larger-resolution inference, parallel independent seeds, and adaptation; they do not justify training a foundation model from scratch on 82 patients.

## Benchmark and split specification

Two tracks must not be mixed:

- Track I: image-level eight-subtype classification, trained/evaluated on disjoint patient groups, separately at four magnifications and with pooled training. This reconnects to the historical notebooks.
- Track II (primary): patient/slide-set eight-subtype classification under the 15 nonempty magnification subsets, with fixed eligible evaluation patients and fixed field budgets.

Primary metric: equal-subset mean patient-level macro F1. Also report the worst-subset macro F1, all-view macro F1, balanced accuracy, per-class recall, binary malignant sensitivity/specificity, Brier score or another specified calibration score, model size, extraction/fusion cost, and patient-bootstrap uncertainty.

For each subset, restrict inputs, not labels. Use a fixed complete-view cohort for artificial missingness when feasible. Report naturally incomplete cases separately. Masks, field subsampling, and data augmentations do not create new independent patients. If evaluation eligibility changes, disclose patient counts and do not attribute population changes to the model.

Provisional split choice (superseded for the initial pilot by the provider-case track described below): three outer patient-group folds subject to the audited count of independent groups in the rarest subtype. The supplied tables contain subtype/count inconsistencies, so they do not determine the final protocol. Verify counts before freezing. Use class-covered validation from outer training patients where feasible; if too few cases remain, use a prespecified conservative tuning scheme and explain the limitation. Use three training seeds per selected final configuration. Outer folds are never used for model/epoch/hyperparameter selection.

Keep raw and conservative identifiers, exact membership lists, split-generation seed, dataset hash, and group-disjointness reports. Freeze membership/config before final evaluation. The original five 70/30 trials, ordinary five-fold CV, and new three-fold grouped evaluation are different protocols; name each accurately. If reproducing a published binary result, use its exact protocol in a separate comparison rather than compare its score directly to an eight-class score.

Do not search many configurations on 82 patients and repeatedly view the test outcomes. Reserve a small fixed hyperparameter grid and use validation only. Confidence intervals must preserve patient dependence; overlapping repeated trials require care and are not independent statistical replicates.

## Experiment matrix

| ID | Configuration | Purpose |
| --- | --- | --- |
| B0 | Majority/prior predictor | Lower-bound sanity check |
| B1 | ResNet/EfficientNet corrected single-view | Historical continuity under a valid split |
| B2 | Frozen Phikon-v2 plus linear/shallow head at each scale | Strong visual representation baseline |
| B3 | Same encoder, average available scores | Simple late fusion |
| B4 | Same encoder, masked mean feature fusion | Simple early fusion |
| B5 | Same encoder, learned masked scale gating | MagViT-style adapted baseline |
| B6 | Same encoder, small set transformer | Generic attention baseline |
| C1 | Same backbone, ordinary cross-attention / published OT adaptation | Close mechanism baseline |
| C2 | C1 with partial transport only | Isolate relaxed matching |
| C3 | C2 preserving unmatched residual evidence | Test proposed distinction |
| A1 | Best candidate without subset-dropout training | Test missingness training effect |
| A2 | Candidate with alternative subset distribution / robust objective | Sensitivity and robustness |
| A3 | Matched-capacity candidate versus best baseline | Isolate size/compute |
| A4 | Second encoder and controlled adapter tuning | Check dependence on encoder |

All selected configurations receive the same folds, seeds, augmentation/field budgets, and validation opportunity. First debug on synthetic tensors and a small training-only slice. Cached feature runs should establish promising mechanisms before full end-to-end jobs.

## Second dataset and broader evidence

Investigate BRACS and/or the multiscale benchmarks associated with Mind the Gap. BRACS has different seven-class labels, so a seven-class task would be a separate validation of the mechanism, not direct validation of BreakHis's eight-label classifier. WSI pyramids allow geometrically verified fields when extracted consistently, but downsampling a single image does not create newly acquired fine-scale information.

For the second dataset define one setting matching its native structure and distinguish registered pyramid views from unpaired-field bags. This is an opportunity to test whether the mechanism works when correspondence assumptions change. Follow release/access rules; no controlled access or author contact is assumed. A second dataset must not become an unbounded detour: select based on available data, reproducible protocols, and deadline feasibility.

## Cluster execution

Configure `MAGSET_ROOT`, `MAGSET_ACCOUNT`, and the appropriate CPU/GPU partitions for the available cluster; see `docs/CLUSTER_SETUP.md`. Keep source, data, cache, environments, runs, and logs in separate subdirectories under shared project storage.

Use SLURM for download/extraction, environment installation, validation, feature extraction, training, and sweeps. Login nodes are for lightweight inspection and submission. Check existing jobs before launching work. Start with one runtime/bootstrap job and separate image-audit jobs; then shard frozen feature extraction across magnifications or encoders. Cache features once and parallelize independent folds/seeds. Begin with a measured concurrency cap and expand within allocation policy and demonstrated throughput. One unnecessarily large distributed job is not a substitute for independent experimental evidence.

Store best/last checkpoints within a retention budget and keep caches/TMPDIR off home quotas. Record job IDs in private running state and reproducible run metadata. Synchronize source without deleting remote artifacts or overwriting divergent changes.

## End-to-end milestones and completion gates

### M0: orientation and working state

- [x] Audit original six files and produce detailed guides.
- [x] Connect to the cluster and verify storage, queue, and GPU inventory.
- [x] Create/synchronize project workspace and verify hashes. All synchronized source, memory, and original reference file hashes verified.
- [x] Persist project context, current state, decisions, and a validated ECC handoff. ECC doctor passed with one handoff and no diagnostics.
- [x] Add data preparation code and bounded SLURM tasks; validate with synthetic fixtures. Five tests passed locally and on the cluster; data preparation and GPU readiness jobs completed.

### M1: dataset and benchmark foundations

- [x] Download the official archive on a compute node; record source URL, bytes, and SHA-256.
- [x] Safely extract it; index all four magnifications; count every subtype/group. Identifier conflicts require further reconciliation.
- [x] Fully decode all 7,909 images; verify magnification coverage and byte-level duplicates.
- [ ] Independently reconcile clinical patient relationships and identifier sensitivity.
- [ ] Reconcile the original patient-group protocol and freeze the split manifest.
- [ ] Complete the closest-method novelty matrix and an explicit claim proposal.

Completion gate: data tables and grouping rules are reproducible, no patient overlap, and every final metric is mathematically defined.

### M2: baseline pipeline

- [x] Create an isolated pinned PyTorch environment on shared storage.
- [x] Record a GPU framework/precision smoke test and environment lock. CUDA gradient and both real-image encoder pilots passed.
- [x] Fetch permitted pinned encoders; complete all eight full frozen-feature caches without cohort fitting.
- [ ] Implement B0–B6 and image-level Track I with saved predictions/class mapping.
- [x] Run real-data execution pilots; test save/reload, split/mask/permutation invariants and finite gradients. These checks are not final performance evidence.

Completion gate: one full patient-separated baseline finishes cleanly, predictions reproduce its reported metrics, and cost estimates determine the sweep budget.

### M3: mechanism pilot

- [ ] Implement partial-transport and unmatched-residual components independently.
- [ ] Validate transport stability and invariances; compare against simple and close baselines.
- [ ] Run a fixed small validation grid; inspect error clusters and negative results.
- [ ] Decide whether evidence supports continuing, simplifying, or replacing the candidate.

Completion gate: the candidate addresses a measured baseline weakness and has a defensible distinction from prior art. A new name or a small score increase is insufficient.

### M4: final experiments and external evidence

- [ ] Freeze method/configurations and perform all outer-fold evaluations.
- [ ] Evaluate all 15 availability subsets on the same eligible patients.
- [ ] Run selected ablations, seeds, parameter/compute controls, corruption tests, and calibration.
- [ ] Validate on a compatible second dataset/task with a separate head and protocol where necessary.
- [ ] Produce patient-aware intervals and audit explanations without selecting only successful examples.

Completion gate: all main tables trace to manifests, code revisions, configuration hashes, jobs, and saved predictions. No claim is based on an untracked notebook state.

### M5: manuscript and submission

- [x] Draft the implemented baseline methods and contribution/evidence gates while experiments run.
- [ ] Complete introduction, full related work, and final candidate method specification.
- [ ] Generate plots/tables directly from frozen result files.
- [ ] Write limitations, data/model provenance, compute, ethics/access, and reproducibility statements.
- [ ] Professor Dong reviews the contribution, authorship, venue, and final claims.
- [ ] Check current venue requirements and deadlines; format and validate the manuscript.
- [ ] Prepare a reviewable submission package. Obtain the user's explicit instruction before submitting or communicating with others.

Submission readiness requires verified results and an approved manuscript. No results or publication success are promised in advance.

## Timing and venue

Working schedule, contingent on queue and data/model access: first few working days for data/protocol/feature cache; the following week for baseline and mechanism pilots; another one to two weeks for controlled final experiments and a second dataset; then manuscript completion and review. Writing proceeds alongside experiments. These are planning estimates, not a booked deadline or an assurance that a competitive paper can be completed in a fixed number of days.

As of November 11, 2026, ISBI 2027's regular paper deadline was November 2, 2026; it has passed. MICCAI 2027 is an ambitious working target and the society currently lists its paper deadline as to be announced. SPIE Medical Imaging 2027 lists an August 5, 2026 abstract deadline; January manuscript deadlines are for the existing submission pipeline, not an assumed new entry point. Recheck official calls before planning an actual submission. Suitable workshops can provide an earlier route if a current call and the resulting evidence fit. No particular open workshop call has yet been selected.

## Session memory and resumption

Read `AGENTS.md`, the research plan, related work, and the latest audit before resuming. Private workspaces additionally keep running state, confirmed context, decisions, and evidence logs, plus optional ECC handoffs. These operating records and machine-specific paths remain outside public Git history.

Refresh job/storage state, resume the first unfinished milestone, and record new evidence separately from hypotheses. Synchronize source and memory in the private working copies after material changes. Do not rerun completed work or duplicate scheduler submissions simply because a new session began.

## Stop and reconsider conditions

Change the design if grouping cannot be verified, the candidate overlaps substantially with existing methods, the primary metric fails on rare subtypes, most gains disappear with capacity controls, or the second dataset contradicts the claim. Record negative evidence. An honest, narrowed contribution is preferable to unsupported novelty or cherry-picked benchmarks.

## November 11 implementation checkpoint

The official archive is downloaded and indexed. Its provider trial files establish 82 subtype-qualified case identities, with 54 train/28 test cases per trial and no case crossing roles. Numeric-stem grouping differs and can combine subtypes; it is a separate sensitivity question. The primary pilot uses the five provider repeated holdouts with inner validation from the training cases only. It is labeled a provider-case benchmark, without asserting independently reverified clinical patient identity. Three-fold stem stratification is infeasible for adenosis's two stems.

Implemented tools: provider-protocol audit; full image-decode shards; pinned Phikon-v2/ResNet-50 frozen extraction; mean, masked concatenation, and late-fusion logistic baselines; model selection across a fixed four-value regularization grid on eight inner-validation cases; explicit outer-evaluation opt-in; score/model/provenance records. Existing labels and images are never used to adapt the frozen pretrained encoder.

Execution dependency: isolated CPU runtime/bootstrap and pinned checkpoint download; one-GPU real-feature pilot; up to four GPU feature workers over eight encoder/magnification shards; up to six CPU workers for 30 validation-only configurations (two encoders, three fusion methods, five provider trials, seed 812). Each run records source/manifest/model revisions. Final statistical analysis and learned fusion/candidate mechanisms remain unfinished.

## Work that proceeds alongside cluster experiments

- Automatically verify saved validation probabilities against reported metrics and collect the complete 30-run matrix; incomplete runs must not produce a misleading partial ranking.
- Implement and test the required learned gating/set-attention baselines independently of the running linear jobs.
- Complete the close-method mechanism comparison before defining the candidate's novelty claim.
- Draft the manuscript structure, data/methods specification, and contribution/evidence map without inventing results; the scaffold is in `paper/outline.md`.
- Select a compatible second dataset based on actual access and reproducible protocols.

At the latest check the environment, full image decoding, and GPU pilot succeeded. Seven of eight feature shards succeeded; one process could not access CUDA. Only that missing shard was resubmitted on an A40, and the existing validation array dependency was updated to the retry. Completed caches were preserved.

## First validation round completed

The runtime, full PNG validation, real GPU pilot, and all eight frozen-feature shards are complete. One CUDA-unavailable process was retried successfully without changing the encoder/preprocessing. All 30 linear baseline configurations completed, and saved probabilities independently reproduce their reported metrics. No convergence or numerical warnings were found in the validation logs. Outer-test scoring remains disabled.

The initial scores are in `docs/VALIDATION_PILOT.md`. Before adding the candidate mechanism, prioritize error analysis and controlled comparisons of image-level heads/case aggregation, learned within-scale pooling, gating, and set attention. The small validation samples make score stability and rare-class behavior important unresolved questions. A manuscript scaffold and score-verification collector were developed alongside the cluster work.

## Controlled fusion round and current decision

Completed the error analysis, 20 field-linear controls, six real-data GPU execution paths, and 30 full hierarchical neural configurations. Every final probability set independently reproduces its metrics; all saved selected models passed reload checks. The predeclared protocol is `docs/FUSION_PILOT.md`, and generated results are in `docs/FUSION_RESULTS.md`.

The original Phikon mean baseline remains higher on the primary exploratory average (0.3868 versus 0.3562 for the highest neural row). Phikon's matched early/late field controls have similar averages; early averaging is not established as the main failure cause. Neural fitting losses approach zero while validation varies widely. The immediate priority is split/training-seed stability and lower-capacity/regularization controls, not an unsupported claim that more complexity helps. The candidate matching method remains pending.

Prespecified next sensitivity round: repeat all six neural methods across five provider trials using inner split seed 813 and training seed 812 (30 new runs). Separately repeat the simplest mean/mean and attention/attention neural controls at split seed 812 with training seed 813 (10 runs). Repeat the original Phikon mean linear control on split seed 813 (five CPU runs). Keep every other training/grid parameter fixed, preserve all outcomes, and collect each seed matrix separately. These are exploratory stability checks with overlapping cases, not independent clinical replications or final test performance. No additional completed run is resubmitted.

In parallel, the pinned public TCGA-MS archive downloaded through SLURM and matched its provider checksum. Its archive/label audit is complete: 65,312 candidate images after sidecar exclusion, 128 labeled patients, and 122 complete seven-resolution slides. Some fold roles omit classes and the release differs from the paper's image count. This does not establish dataset compatibility: full decoding, native protocol/eligibility, correspondence assumptions, and encoder-pretraining overlap remain prerequisites to fitting. See `docs/TCGA_MS_AUDIT.md`. BRACS access requires an account. Primary sources and the remaining novelty questions are recorded in `docs/RESEARCH_DIRECTION.md`; baseline methods are drafted in `paper/methods-draft.md`.

The 45 prespecified sensitivity configurations are now complete and independently verified, with no convergence/numerical warnings or tracebacks in their logs. Results are generated in `docs/SENSITIVITY_RESULTS.md`. Changing the inner split changes both score levels and the strongest neural row. The original linear mean control remains higher than neural rows for both checked splits. Preserve these negative outcomes; do not select only the favorable validation split. Next data-readiness jobs safely extract the pinned TCGA-MS archive and fully decode seven resolution shards (maximum four workers), with no second-dataset fitting or test scoring. The next model work should test reduced capacity/stronger regularization, reconcile grouping sensitivity, and specify a native second-dataset protocol before candidate matching.

## November 12 checkpoint and comparison registration

All prior MagSet jobs are complete. TCGA-MS extraction succeeded and all seven decoding reports match the pinned inventory/archive: 65,312 RGB images, 224×224, zero errors. The metadata audit now includes verified image readiness. There is still no second-dataset fitting or final outer-test result.

Registered the next bounded round in `docs/COMPARISON_PLAN.md`: ten mean-linear fits using exactly the neural field normalization, plus forty one-factor capacity/dropout controls across the two existing inner splits and two neural architectures. Reference the twenty already completed original neural configurations; do not repeat them. Every comparison must pass an actual saved-normalization-array equality check, exact case/cache identity, independent score recomputation, and configuration-difference checks. Use at most four GPU and four CPU workers. Keep all outcomes, including regressions.

The next paper benchmark requires exact compatible reproductions or clearly labeled adaptations of MagViT and M2OTCA, with same-task baselines and capacity controls. The existing contextual gate and one-block attention model are useful internal controls, not exact published-method reproductions. Native TCGA eligibility/labels/resolutions and pretraining overlap remain to define before training there.

## Matched comparison complete: current research position

All 50 registered configurations and the dependent collector completed successfully. The collector verified 60 linear/neural context comparisons, including exact saved normalization-array equality, case/cache agreement, and the intended configuration differences. All 50 model error logs were inspected without listed convergence/numerical warnings or tracebacks. Saved probabilities were independently recomputed locally. There are 175 completed exploratory validation configurations in total; the 20 referenced original neural runs are included in the earlier count and are not added again.

Generated results are in `docs/CAPACITY_RESULTS.md`. Matched linear mean scores are 0.3843/0.3169 for inner split seeds 812/813; the highest neural-control averages in this round are 0.3636/0.2744. These are validation macro F1 values, not accuracy or final test performance. Changing hidden width or dropout did not close the descriptive gap. No MagSet job was active at the verification checkpoint.

The baseline pilot phase now has verified data, repeatable runs, controlled internal comparisons, and negative evidence. The novel-method phase is not complete. Prioritize the native second-dataset protocol and encoder-overlap audit, clinical-identifier sensitivity, compatible published-method comparisons, and a precise candidate/ablation specification before further broad architecture sweeps. Final outer evaluation, independent external results, uncertainty/calibration, and the submission-ready manuscript remain unfinished.

## Critique-driven hardening and Professor Dong package

The user requested several critique agents and evidence for the coming week. Three independent reviewers completed two rounds, producing six artifacts under `docs/reviews/`; the closure ledger separates repaired software gaps from open scientific limitations. November 18 is an internal preparation target, not a confirmed meeting. BRACS access is not approved.

The reviews identified two important interpretation limits: all 82 BreakHis provider cases entered fitting or validation somewhere across repeated splits, so there is no globally pristine confirmation cohort; Phikon-v2's pinned card confirms TCGA source pretraining, so TCGA cannot be called a clean Phikon external breast evaluation. No outer-role predictions were scored. Clinical identity of shared numeric stems remains unresolved, and strict three-way class coverage is infeasible for two adenosis stems.

Trusted verification now checks original labels, exact declared roles, regenerated inner split assignments, and score arithmetic. All 175 saved sets passed. Consistent truth-label permutations and same-class membership swaps are rejected by end-to-end regression checks. Published report/collection CLIs and scheduler jobs require the trusted inputs. The evidence package rebuilds and hash-links this audit rather than trusting old summary numbers alone.

Registered a native TCGA supervised quarantine before any classifier training: 80 fit, 16 development, 32 sealed patients, five/one/two per class. Fixed salted hash assignment is not searched; six release-incomplete cases remain fit-only. Source metadata, exact label/class counts, view coverage, patient/slide/path identity, and job-supplied protocol hash are checked. New jobs use read-only Git-archive snapshots bound to embedded commit and archive digest. Details and hashes are in `docs/TCGA_QUARANTINE_PROTOCOL.md`.

Frozen ImageNet ResNet features for all 37,024 selected native-resolution images have been created after a real GPU smoke test. These caches are label-blind preparation, including permitted frozen encoding of reserved images; no native classifier has fitted, tuned, or scored reserved patients. Independent cache auditing checks every key, checksum, shape, dtype, and finite value. This is a secondary 16-class multi-organ task, not the original breast task or source-paper reproduction.

The current method hypothesis is narrowed to a small agreement-contrast block beside the strong mean representation, with a regularized linear head. Shared/specific representations and partial OT are existing work. Preserve equal-dimension duplicate-mean, within-bag-variation, random-partition, and simple-affinity controls; partial OT is optional only after simpler explanations are tested. No candidate benefit or novelty is claimed. The next task-level pilot must have a minimal prespecified budget, not another adaptive architecture sweep on 16 development cases.

Ready-to-review artifacts: `docs/evidence/PROFESSOR_BRIEF.md`, `docs/evidence/PROFESSOR_DECK.html`, the generated matched-result SVG/PNG, and the hash-linked evidence ledger. The deck was rendered and visually checked across all eight slides. The package shows verified results, negative evidence, corrected claims, open protocol/data decisions, and go/no-go criteria. No messages to Professor Dong or conference submission have been sent.

## Fixed native development pilot implementation

The user requested a plain-language explanation and continuation. `docs/PROJECT_EXPLAINED.md` now maps the actual problem, datasets, roles, encoders, fusion architecture, weighted training, outputs, metrics, and limits. It distinguishes exploratory BreakHis evidence from the secondary 16-class native task.

Implemented the prespecified four-fit pilot in `magset/native_pilot.py` and `slurm/native_pilot.sbatch`: mean, duplicate mean, random contrast, affinity contrast. It verifies submission-pinned cache/protocol hashes, excludes sealed patients before supervised construction, uses fitting-only equal-case moments and inverse-subset weights, fixes all settings, and audits exact development memberships/truths/probabilities. Models and preprocessing reload without pickle. Nine new tests cover role/label tampering, invariances, exact 16-class metrics, subset weighting and a full synthetic pipeline. Real-data outcomes remain pending until the scheduler run completes and records are independently checked.

### Native pilot measured result and stop rule

SLURM job4958291 completed successfully in30seconds from immutable source c1364c6202c2a5de7f9111b927ef0f41c94e0664. Exactly four fixed fits converged; all960 saved development records and seven artifact hashes passed trusted verification. Mean-subset macro F1: mean0.334028, duplicate0.339583, random0.334028, affinity0.338194. Affinity changes only one of240 development decisions relative to mean and loses to the duplicate control. This fails the prespecified agreement-specific improvement gate. No reserved predictions were produced, and no extra parameter/seed/width/temperature search is authorized by this outcome.

Current next work is to consolidate this negative evidence, finish compatible published-method and source-label construction audits, and discuss the contribution/data direction with Professor Dong before registering another mechanism experiment. The main eight-subtype breast task remains distinct from the16-class secondary task. See `docs/NATIVE_DEVELOPMENT_RESULTS.md`, `docs/NATIVE_LABEL_SCHEMA.md`, and `docs/reviews/NATIVE_PILOT_REVIEW.md`. The plain-language guide and evidence brief incorporate the result; the eight-slide deck is explicitly retained as the earlier pre-native-pilot snapshot.

## Post-pilot source audits and next training strategy

The user requested next tasks and an easy training explanation. Published `docs/TRAINING_STRATEGY.md` and `docs/NEXT_STEPS.md`: frozen picture reader plus trained classifier; equal-case missing-view practice; development-only selection; reserved evaluation after a defensible freeze. Fine-tuning is a later separate experiment, not completed work.

Completed a targeted source/code comparison of the three closest references. Pinned the official continuous-magnification repository at83f05d8eed96e805b737b9ea935efa1c4bedb8a1 and inspected its reader, evaluation and token paths without running author code. Confirmed `label_id` targets match our mapping. Six source groups have variable descriptions and two cases have blank text; this does not establish labeling error. A clinical-name/curation crosswalk remains incomplete. Full M2OTCA methods/supplement and official MagViT/M2OTCA implementations remain unverified. Exact reproduction is not claimed. See the method and label audit documents.

Added a trusted-hash, development-only analysis of all28 one-view-addition edges using existing960 pilot predictions. No refits or reserved predictions. The mean control has26 wrong-to-right and22 right-to-wrong transitions across448 repeated patient/edge comparisons; five of16 patients are harmed on at least one edge. These are dependent descriptive events, not448 independent patients, significance or a proven remedy. Every edge/arm is retained publicly as aggregates. Four additional tests cover known help/harm/confidence counts, invalid coverage and pinned label/source guards; all54 tests passed.

Next: one narrow hypothesis and decisive control, compatible-method specification, a newly registered small fit budget, then controlled execution. The failed agreement-specific gate remains in force. No automatic parameter/seed/temperature sweep, fine-tuning, original rotating-fold training, relabeling or reserved evaluation was launched. Writing and breast-data/grouping readiness proceed alongside design.

## BreakHis-first baseline milestone completed

User clarified that the quick BreakHis project with Professor Dong is the immediate deliverable, ahead of TCGA and novel-paper work. The existing ten matched Phikon mean runs were independently rechecked against trusted labels/splits and all1,200 saved validation records. Split812 mean-subset F1/accuracy0.3843/45.7%, all-view F1/accuracy0.4083/47.5%; split8130.3169/38.3%, all-view0.3083/37.5%. Both splits and all trials are retained. These involve31 distinct validation provider cases, not1,200 independent patients.

Implemented a checksum-guarded original-image prediction replay. GPU job4959781 completed0 in44seconds on immutable source9448a7e319167b7417670a882552edb372603fd3. It reused the actual saved trial1/seed812 head, reconstructed all120 archived validation probabilities from the cache, then read121 original images of the first sorted validation case and replayed all15 availability settings. Maximum differences4.44e-16 cached and5.517742757e-08 raw. No new fits or outer-role predictions. An independent reviewer verified the actual artifact, original-case counts, truth/class order, source/head/archive hashes and all15 raw probability vectors. A discovered report-integrity gap was repaired and regression-tested before success was reported. All60 tests pass.

Current deliverables are `docs/BREAKHIS_FIRST_RESULTS.md`, the BreakHis-first professor brief, `docs/BREAKHIS_MILESTONE_PLAN.md`, prediction/report commands, and `docs/reviews/BREAKHIS_MILESTONE_REVIEW.md`. They establish a working baseline milestone only. Predictive quality is modest (including zero phyllodes recall in these selected validation records); final clinical grouping, clean confirmation, method novelty and a paper remain unfinished.

Continue on BreakHis: specify one bounded standard improvement, run it under the fixed provider-case development protocol, then package its evaluation and project report. Do not return to TCGA after baseline completion. First numbers are available now; a next changed-model table is targeted for the next one or two work sessions as a conditional implementation/queue estimate, with no gain guaranteed. No new training experiment has been launched by the replay job.
