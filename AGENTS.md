# Project working instructions

Current checkpoint, November 17, 2026. The bounded UNI2-h metric-learning family is complete and independently checked. Primary both-head accuracy is **76.17% / 75.83%**, macro F1 **.668624 / .587323**, compared with the exact identity control **75.83% / 75.17%**. Its registered larger-improvement screen fails. The malignant-only ablation reaches 812 F1 .672180; P retains 813 F1 .602512 and separate all-four-view accuracy85%/85%. These observations do not establish a consistent replacement or95%. [Complete result](docs/BREAKHIS_SPECIALIST_METRIC_RESULTS.md), [independent review](docs/reviews/SPECIALIST_LEARNING_VALIDITY_REVIEW.md).

The separately registered standard supervised UNI2-h field-logistic baseline is now executing under the same original roles. Registration/source/synthetic tests and actual fitting-only numerical checks have passed; final scores require its complete collector. [Fixed recipe](docs/BREAKHIS_UNI_LINEAR_SPEC.md). The new family was chosen before metric outcomes. Do not expand settings after seeing results.

**Current priority: BreakHis capstone for Professor Dong.** The user explicitly prioritizes legitimate accuracy improvement while finishing the report, five-category presentation and demonstration. Working materials Nov17–18/presentationaroundNov20, exactday/rubricunconfirmed. TCGA expansion and novel-paper work remain deferred.94–95% is an unmet objective, not a guarantee. [Current tasks](docs/NEXT_STEPS.md), [full scorecard](paper/breakhis-five-class-report.md).

Five final labels are pooled benign plus ductal, lobular, mucinous and papillary carcinoma. Step1 benign/malignant; benign stops and malignant proceeds to the four-type expert. No benign-subtype expert. All four magnifications and all15nonempty subsets stay in scope. [Task contract](docs/DUAL_STEP_SCOPE.md).

The original six supplied files and historical eight-class results remain evidence. Savednotebook93.73%/94.99%image-randomsingle40× results have a different target, prediction unit and split. All82qualifiedproviderkeys entereddevelopment; clinicalidentity is unresolved, and excluded roles are not globallyfresh finalcohort. Each role46fit/8val/28excluded; fields/masks/overlappingtrials aredependent. Never call these clinicalvalidation or established novelty.

Stay on BreakHis through model improvement, evaluation and the project report. Completing the first baseline milestone is not permission to switch back to TCGA; that work remains deferred until the BreakHis project deliverable is finished or the user changes priority.

Use `scripts/report_breakhis_milestone.py` for the historical first-results page. The current professor brief and BreakHis project report also include completed adaptation, diagnosis and the supervised reference; do not overwrite them with an older baseline-only builder. `scripts/report_breakhis_supervised.py` renders the hash-bound complete supervised comparison. The older mixed-dataset packet/deck builder is historical.

Read `README.md`, `plan.md`, and `docs/RELATED_WORK.md` before changing the research direction. When present in the private workspace, also read `memory/STATE.md`, `memory/PROJECT.md`, and `.local/AGENTS.private.md` for the current handoff and cluster operating rules.

- Use all four BreakHis magnifications and support any nonempty available subset. The original 40X notebooks are historical experiments, not the proposed architecture.
- Keep facts, hypotheses, decisions, and measured results distinct. Do not invent novelty, metrics, completed work, citations, or publication acceptance.
- Do not assume exact correspondence between equally numbered fields at different magnifications. Preserve raw identifiers and audit grouping before selecting a protocol.
- Keep all related images, magnifications, and derived examples in the same split. Validation controls model selection; outer test outcomes do not.
- Measure strong masked mean/gating/attention baselines with the same encoder, folds, seeds, and tuning budget as candidate mechanisms.
- Test data parsing, split disjointness, missing-view invariance, permutation invariance, numerical stability, and save/reload behavior. Use `python3 -m unittest discover -s tests -v` for lightweight checks.
- Save dataset, split, encoder, code, and configuration revisions with every experiment. Never describe synthetic checks as reproduced research performance.
- Use scheduler jobs for heavy work on shared clusters. Inspect the queue and allocation policy before launching bounded arrays; leave unrelated jobs alone.
- Keep datasets, model weights, private reference documents, session memory, credentials, and machine-specific paths out of public Git history.
- Do not change or redistribute the original supplied reference files automatically. Cite their papers; maintain their local provenance.
- Update the plan and measured research status after material work. Private workspaces should also update running memory before a handoff.
