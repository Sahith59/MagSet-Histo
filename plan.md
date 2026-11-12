# MAGSET-Histo execution plan

Updated November 11, 2026. This is a living execution plan, not a report of completed experiments. Public research status is recorded here; private working copies also maintain a separate running-state handoff. Checkboxes are completion claims and must be updated only with evidence.

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
- [ ] Verify representative image decoding, duplicates, identifiers, case/patient relationships, and magnification coverage.
- [ ] Reconcile the original patient-group protocol and freeze the split manifest.
- [ ] Complete the closest-method novelty matrix and an explicit claim proposal.

Completion gate: data tables and grouping rules are reproducible, no patient overlap, and every final metric is mathematically defined.

### M2: baseline pipeline

- [x] Create an isolated pinned PyTorch environment on shared storage.
- [x] Record a GPU framework/precision smoke test and environment lock. CUDA gradient and both real-image encoder pilots passed.
- [ ] Fetch a permitted encoder at a fixed revision; cache train/evaluation features without cohort fitting.
- [ ] Implement B0–B6 and image-level Track I with saved predictions/class mapping.
- [ ] Run small training-only pilots; test save/reload and split/mask invariants.

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

- [ ] Draft introduction/related work and method specification while experiments run.
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
