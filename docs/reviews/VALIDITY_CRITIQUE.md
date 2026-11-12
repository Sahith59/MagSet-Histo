# Independent validity critique

Prepared November 12, 2026. Scope: the research proposal, development protocol, and saved validation evidence. This is a targeted scientific-validity review, not a clinical endorsement or a completed-paper review. The principal source snapshot reviewed was `e1177c0`; subsequent fixes should be recorded in a closure ledger rather than silently erasing the original findings.

## Verdict

**Ready for a candid progress meeting; not ready for a claim of a new method's superiority or independent patient generalization.** The engineering foundation is substantially stronger than the starting notebooks: fixed data/model provenance, explicit missing-view masks, training-only normalization, saved probabilities, reload checks, and retention of unfavorable results are useful achievements. The main unresolved problems are the independent evaluation unit and the very small, repeatedly reused development sample. More GPU runs do not repair either problem.

The defensible present claim is: *we built a reproducible provider-case development benchmark and found that the checked attention/capacity/dropout controls did not exceed a normalization-matched linear mean baseline on the reported average.* This is descriptive validation evidence. It does not establish population superiority, general failure of attention, clinical effectiveness, or novelty.

## Review rubric

Scores describe the current proposal/pilot, not the quality of a future completed paper.

| Dimension | Score / 5 | Evidence and priority |
| --- | ---: | --- |
| Problem and research question | 3 | Eight-class classification from incomplete magnification sets is clear; clinical unit and intended contribution need tighter definition. |
| Methodology | 3 | Good shared-input controls and missingness checks; independent grouping and final selection/evaluation design remain unresolved. |
| Data and independent evidence | 2 | Only 82 provider cases and two adenosis numeric stems; eight validation cases per trial. |
| Analysis | 2 | Split sensitivity is measured and acknowledged, but repeated adaptive validation cannot support confirmatory comparisons. |
| Results and interpretation | 4 | Negative outcomes are preserved, metrics are distinguished from accuracy, and uncertainty is acknowledged. Some holdout wording needs correction. |
| Limitations | 4 | Most threats are documented; several require operational restrictions rather than another caveat. |
| Writing and evidence organization | 3 | Clear records exist across files; the professor packet should consolidate claims, evidence, unresolved questions, and a decision. |
| Novelty and citation completeness | N/A | Assigned to the separate novelty reviewer; no novelty finding is inferred here. |

## P0: restrictions needed before confirmatory claims

### 1. Provider cases are not independently verified clinical patients

**Confirmed facts.** The recorded manifest has 82 subtype-qualified provider cases, 70 stripped numeric stems, and three stems containing multiple subtype labels. The provider trial files keep each subtype-qualified case in a single role. However, the conservative numeric stems can cross those roles. The relationship between suffixes/specimens and clinical patient identity has not been independently verified.

**Inference.** A shared stem may indicate related material; it is not proof of a common clinical patient. Conversely, using label-qualified case identifiers does not prove clinical independence. Therefore neither silently merging stems nor ignoring them is a justified final patient-identity solution.

**Required action.** Keep the current results labeled **provider-case development results**. Present the identifier map and feasibility limits to Professor Dong. Obtain a documented decision on the prediction unit and grouping rule before reporting clinical-patient generalization. Do not assign one subtype to a mixed-label stem; grouping for splitting and aggregation into one labeled prediction are different operations.

### 2. A naive strict three-way stem split is infeasible for the current eight-class task

**Measured metadata audit.** Distinct conservative stems per subtype are:

| Subtype | Stem groups |
| --- | ---: |
| Adenosis | 2 |
| Ductal carcinoma | 38 |
| Fibroadenoma | 7 |
| Lobular carcinoma | 4 |
| Mucinous carcinoma | 7 |
| Papillary carcinoma | 6 |
| Phyllodes tumor | 3 |
| Tubular adenoma | 6 |

If every class must appear in fitting, validation, and test, at least three independent groups per class are necessary. Two adenosis stems cannot satisfy that condition. This is a feasibility issue, not something additional random seeds can solve.

The following metadata-only sensitivity calculation retained each existing inner-validation and outer-test assignment, then removed fitting cases sharing a stem with either held-out role. It did not score any outer predictions.

| Inner seed | Trial | Fit–validation shared stems | Fit–outer shared stems | Validation–outer shared stems | Fitting cases after embargo | Entire missing fitting subtypes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 812 | 1 | 2 | 7 | 2 | 38 | Adenosis |
| 812 | 2 | 1 | 4 | 3 | 42 | Adenosis |
| 812 | 3 | 1 | 6 | 2 | 40 | Adenosis |
| 812 | 4 | 0 | 6 | 1 | 39 | Adenosis; phyllodes |
| 812 | 5 | 1 | 5 | 1 | 40 | Lobular |
| 813 | 1 | 1 | 6 | 3 | 40 | Adenosis |
| 813 | 2 | 2 | 5 | 2 | 40 | Adenosis |
| 813 | 3 | 2 | 5 | 3 | 40 | Adenosis |
| 813 | 4 | 1 | 5 | 2 | 39 | Adenosis; phyllodes |
| 813 | 5 | 2 | 3 | 3 | 41 | Adenosis; lobular |

**Consequence.** Simply adding a stem embargo to the present pilot would make every checked fitting partition lose at least one subtype. Do not silently apply this patch and report a comparable eight-class experiment. Professor Dong should help decide whether to verify finer identity, keep an explicitly provider-case track, use a separately specified group sensitivity with fixed hyperparameters, or change the task/cohort. Any changed task must be named and reported separately.

### 3. The existing development sample is highly reused

**Measured facts.** The 175 locally available full reports contain 700 candidate-validation evaluations, counting four settings per run. All report 46 fitting, eight validation, and 28 outer-role cases. The two inner seeds cover 22 and 20 distinct validation provider cases, with only 31 distinct validation cases in their union. All 175 reports are labeled validation-only.

**Inference.** The accumulated comparisons are useful for debugging and hypothesis formation, but the selected validation scores are increasingly optimistic development statistics. Subsets, images, trials, and optimizer seeds do not create new independent patients. The two inner splits also do not estimate a stable variance distribution.

**Required action.** Stop broad architecture searches on these same cases. Prespecify a small candidate/control set and selection budget, retain all results, and use a genuinely untouched label-outcome evaluation for confirmation after the recipe freezes. No confidence interval should treat 175 runs, 15 masks, or five provider holdouts as independent clinical samples.

### 4. No globally pristine BreakHis test cohort remains across all development trials

**Measured metadata facts.** Across the completed development reports, 79 provider cases were fitted somewhere and 31 were used for validation; their union is all 82 cases. Among the 69 cases assigned an outer-test role in at least one provider trial, 66 were fitted in another trial and 27 were used as inner validation in another trial. For each nominal 28-case outer split, 10–11 cases have been validation cases elsewhere and 26–28 have been fitted elsewhere.

This is expected role reuse under repeated holdouts and is **not** evidence that an individual run trained on its own assigned outer cases. The code explicitly excludes outer-role cases from fitting/normalization in that run, and no outer-role predictions were inspected in this review. However, adaptive architecture decisions made from all trials' development summaries prevent describing a later BreakHis evaluation as a globally pristine, never-seen patient test.

**Required wording.** Say: **“No outer-role predictions have been scored; each run excludes its own outer cases from fitting and selection.”** Do not say **“all final test patients remain globally untouched.”** Preserve an untouched patient-level confirmation split in a second dataset before examining modeling outcomes there. If later reporting repeated-split BreakHis scores, disclose global development reuse and avoid claiming unbiased confirmation of an adaptively selected architecture.

## P1: concrete engineering and measurement hardening

### 5. Score verification needs a trusted label and role oracle

At the reviewed snapshot, `magset.collect.verify_run` verifies probability validity, unique case/subset records, internal truth consistency, case counts/disjointness, and reported metric arithmetic. It does not independently compare each saved truth or case-role assignment with the trusted manifest, provider protocol, and split-generation rule. Internally consistent label or membership mistakes could therefore pass the collector.

**Fix recommended to the parent agent:** require trusted manifest/protocol inputs for audited collections, recompute case labels and exact split memberships from those inputs and the recorded trial/seed, verify hashes, and reject a changed label, role, omitted/extra case, altered seed, or incompatible class order. Add negative tests for those cases. Retain the existing arithmetic checks. This is a validation gap, not evidence that current labels are wrong.

### 6. Source revisions are claimed rather than tied to immutable queued-job inputs

The SLURM scripts execute from a synchronized mutable `repo/` and receive a caller-supplied source SHA. Later source synchronization can change files used by a queued task or collector while the job continues to report the earlier SHA. The examined history does not establish that this corrupted the completed metrics, but the mechanism permits future provenance drift.

**Recommended action:** stage a versioned immutable source bundle for each experimental revision, execute the job from that bundle, and record a bundle/file hash at startup. Keep model/cache revisions and the full effective configuration beside predictions. Metadata-only documentation changes need not rerun numerical experiments.

### 7. Interpret mask and field aggregation correctly

- Equal weighting of all 15 subsets is a deliberate benchmark distribution, not a measured clinical missingness distribution. It assigns 4/15 weight to single-view, 6/15 to two-view, 4/15 to three-view, and 1/15 to full-view settings. Report every subset and view-count summary; do not call its mean clinical expected utility.
- All-field bags have unequal field counts. Case-balanced fitting weights prevent an image-rich case from multiplying its training influence, but they do not make its information content equal to a small bag. Add a prespecified fixed-field-budget sensitivity before attributing a candidate's gain to cross-scale evidence reasoning.
- Keep naturally incomplete TCGA cases separate from artificial withholding on an eligible complete-view cohort. Patient counts and class coverage must remain fixed within each method comparison.

### 8. Do not overinterpret small score differences or repeated-case intervals

On the present balanced eight-case validation set, one case changes accuracy/balanced accuracy by 0.125; macro F1 can change nonlinearly. The matched linear averages 0.3843/0.3169 and checked neural maxima 0.3636/0.2744 are descriptive summaries, not significant effect estimates. The strongest row can change with validation-case choice.

For the professor packet, plot paired trial outcomes and subtype recall with distinct-case support, and label them development diagnostics. For final uncertainty, prespecify the independent clustering unit and carry all repeated predictions/masks from a cluster together; do not bootstrap individual images or treat the repeated trial estimates as five independent folds. Exact final inference requires the grouping and evaluation design to be settled first.

## What is already well hardened

The reviewed code fits normalization only on fitting cases, excludes each run's outer cases from its fitting/selection, uses the same manifest/features in controlled comparisons, verifies selected-model reload predictions, tests missing-placeholder and permutation behavior, and retains failed-to-improve architectural controls. The matched-capacity round verifies actual normalization arrays rather than relying solely on setting names. Those are meaningful reproducibility strengths. They should be shown as engineering and benchmark progress, with their scientific limits attached.

## Realistic evidence package for the week of November 16

| Time | Work | Completion evidence |
| --- | --- | --- |
| Days 1–2 | Harden the collector against trusted manifest/protocol; publish the grouping-feasibility table; consolidate the claim ledger. | Negative tests reject changed truth/roles; all retained runs pass trusted checks or discrepancies are disclosed. |
| Days 1–2, parallel | Audit second-dataset pretraining overlap, native labels/groups, incomplete cases, and missing-class folds. Choose and freeze a confirmation holdout before modeling outcomes. | Written native protocol and split hash; explicit statement of pretraining exposure and what “external” means. |
| Days 2–3 | Finalize one candidate hypothesis and no more than a few compatible controls, including a simple linear baseline and a capacity-matched ordinary attention/residual control. | Preregistered input/field/mask/tuning budget; prior-art matrix; a result that could falsify the hypothesis. |
| Days 3–5 | If the data/claim gates pass, run a bounded validation-only pilot. Avoid another broad architecture sweep. | Saved predictions/provenance; all outcomes reported; no confirmation-holdout-driven design change. |
| Days 5–6 | Build one-page summary, paired diagnostic plots, class-support table, reproduction command, and a clear limitations/decision page. | Every number traces to a saved artifact; independent reviewers inspect the exact packet. |
| Day 7 / meeting | Present what is established, what failed, the candidate hypothesis, and the grouping/confirmation choices requiring professor input. | A defensible go/no-go decision; no invented novelty, clinical benefit, final significance, or conference-readiness claim. |

If the candidate does not improve the fixed controls, the correct one-week deliverable is a transparent negative pilot and a focused revision decision. Positive results are not a prerequisite for an honest progress meeting; unsupported superiority claims would weaken it.

## Audit provenance

The quantitative checks above used the recorded manifest/provider-role metadata and saved **development** reports only. No cluster mutation or outer-model scoring was performed. Manifest SHA-256: `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31`. Provider-protocol JSON SHA-256: `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134`.

Relevant reviewed code: `magset/data.py` (identity construction), `magset/baseline.py` (inner splitting and fitting), `magset/bags.py` (normalization), `magset/train_fusion.py` (fixed final-epoch selection), `magset/collect.py` (artifact verification), and the benchmark/comparison documents linked from the README. Other reviewers cover the full novelty and encoder-provenance audits.
