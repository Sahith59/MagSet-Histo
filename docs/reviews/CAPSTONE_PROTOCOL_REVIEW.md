# Independent capstone protocol review

Review date: November 14, 2026. Scope: protocol validity, interpretation of low BreakHis scores, and the fastest honest master's capstone deliverable. This review inspected the current instructions, README/plan/related-work record, private handoff, original-material review, coverage/reference audit, supervised/diagnosis reports, and local manifest/provider protocol. No model fitting, jobs, outer-role scoring, or source changes were performed. This file is the reviewer's only written change.

## Decision

Finish the capstone around an audited, reproducible BreakHis classification study with a documented negative generalization result. The present scores are credible measurements of the chosen exploratory provider-case task; they are not evidence that the original notebook task has been reproduced and failed. A high target accuracy is not a valid acceptance criterion across different prediction units and split protocols.

A secondary **single-image, eight-subtype endpoint**, including a separately reported 40× slice, is justified by the supplied notebooks. It should restore a clear bridge to those artifacts without replacing the four-magnification/missing-view primary endpoint or claiming that image-random evaluation measures unseen-case performance. First report available saved validation image predictions under existing roles. If the exact notebook is subsequently reconstructed, label that exercise a historical procedural reproduction with known case-overlap and evaluation-reuse limitations. Its historical accuracy must never be placed beside the current case score as an apples-to-apples ranking.

The fastest defensible path is report completion and saved-artifact endpoint alignment, not another architecture sweep or a favorable resplit. Nothing in these results requires a novel method to complete the capstone. Institutional grading requirements themselves have not been supplied, so this review cannot guarantee capstone acceptance.

## What changed relative to the received project

The received materials already contained three distinct tasks; there was never one single completed starting pipeline.

| Material | Input and endpoint | Relationship to current work |
| --- | --- | --- |
| Two supplied notebooks | One 40× image, eight subtypes; image-random 80/20 split; same evaluation set selects checkpoint/rate | Historical experiment and legitimate reproduction target, not an unseen-case benchmark |
| Supplied paper/poster/slides | One image from any magnification, ResNet features and two XGBoost stages; five final categories | Prior work, different labels and absent implementation |
| Proposal | Multiple magnifications, eight classes, missing inputs and transformer fusion | Motivation for the present research extension; supplied notebooks do not implement it |
| Current baseline study | Sets of fields per provider case, four magnifications, 15 nonempty available-view subsets; frozen/adapted/supervised references | New operationalization, with stronger grouping safeguards and a materially different endpoint |

Thus the current study is a research extension and validity repair, not an exact notebook reproduction. The move to case-level prediction and equal treatment of all 15 subsets was a consequential scope expansion. It is consistent with the proposal and explicit four-view instructions, but it substantially changes what the headline score means. The capstone should say this plainly. Current aggregate figures also mix many incomplete-view settings; they do not mean every single full-input image was classified with that accuracy.

The notebook's 93.73%/94.99% are embedded historical outputs from 399 evaluation images. Patient/case separation was not enforced, and those same images influenced model selection and learning-rate adaptation. Existing outputs do not quantify actual historical case overlap or the causal inflation caused by it. The supplied paper's 86.22% binary figure is neither an eight-class accuracy nor the present endpoint. Copying the recent graph reference's released CSVs is also unsuitable: its training support, calibration split, repeated case membership, and cross-view role inconsistencies differ from this joint-view study.

## Why scores are low: evidence versus explanation

**Directly evidenced:** both neural references fit their fitting cases extremely well and transfer poorly to validation cases. The supervised ResNet case balanced accuracy is approximately 99.6%/99.7% fitting versus 37.0%/40.0% validation. The adapted case classifier likewise fits approximately 98.5%/98.7% and validates at 45.7%/38.8%. Poor transfer is present in saved image classifiers too; it is not confined to the final case readout. Frozen mean-subset macro F1 remains 0.3843/0.3169; ResNet is 0.3084/0.3006. The fixed improvement gate failed.

**Directly evidenced structural constraint:** every run has 46 fitting, eight validation and 28 excluded provider cases. Fitting support by subtype is A=1, DC=25, F=6, LC=2, MC=5, PC=3, PT=1, TA=3. One adenosis and one phyllodes case cannot supply independent within-class diversity, regardless of field count or augmentation. Zero phyllodes validation recall alongside perfect fitting case recall is consistent with this limitation. It does not prove that adding a case alone would fix performance.

**Directly evidenced uncertainty:** the local manifest has 7,909 images but only 82 subtype-qualified provider cases (81 bare identifiers, 70 stripped stems). All cases contain all four magnifications. Provider metadata contain five repeated holdouts, each 54 training/28 test cases, not disjoint five-fold CV. Current inner validation has one case per class; one prediction change in one full-view trial changes accuracy by 12.5 percentage points. Across arrangements only 31 distinct validation cases are involved (22/20 separately). Fifteen masks do not multiply biological sample size, and trial/seed repetition does not provide independent clinical replications.

**Plausible, not isolated causes:** limited case diversity, overfitting, stain/site or specimen shifts, loss of fine morphology on resizing, inherited field labels that do not identify diagnostic regions, encoder/task mismatch, optimization/regularization choices, and unmodeled bag heterogeneity. ResNet's failure under one registered recipe does not show that every supervised image model fails. Phikon adaptation's small geometric change does not prove its margins are unchanged. Excellent fitting accuracy does not by itself exclude label/preprocessing mistakes that consistently affect both fitting inputs and their labels; the independent parser/label/replay checks are the relevant safeguards.

**Explanations weakened by existing evidence:** simple readout choice alone is insufficient. Saved auxiliary mean-logit and mean-probability readouts remain weak, and mean-feature/mean-logit equivalence holds for the linear head. More views also do not guarantee better performance: all-view frozen accuracy is only 47.5%/37.5%. Neither fact excludes more sophisticated pooling; neither supports launching a fusion sweep urgently.

There is no evidence here that low numbers are a broken collector, missing rare-class rows, or failed artifact replay. Those possibilities were specifically audited. Clinical relationships between filename-derived groups remain unresolved. Such ambiguity limits interpretation; it cannot be asserted as the demonstrated cause of the low score.

## Weak assumptions to remove from the report

1. Replace “patient-disjoint” with “provider-case-disjoint” unless clinical identity is verified. Stem grouping can merge different subtype labels, so it cannot silently redefine the prediction label.
2. Do not call 812 and 813 independent studies, the five trials disjoint folds, or the 1,200 case/subset records independent patients.
3. Do not call excluded roles a globally pristine final cohort. All 82 cases entered development somewhere. Per-model exclusion is useful provenance, but does not erase study-level adaptive exposure.
4. Do not compare field-weighted accuracy, case accuracy, mean-subset macro F1 and historical image-random accuracy as the same endpoint.
5. Do not infer genuine missingness robustness from naturally incomplete cases: all provider cases have four views; masking is an imposed evaluation condition.
6. Do not convert successful code/artifact verification into predictive success, clinical validity or novelty. Preserve failed gates and both arrangements.

## Fastest concrete completion sequence

1. Finalize the capstone report now with the frozen reference, failed adaptation, saved-model diagnosis and failed supervised reference. Use separate tables for artifact execution, exploratory predictive results and unsupported future claims. Include supplied-material/task mapping, exact class support, confusion matrices and the central fitting–validation gap. Retain TCGA only as deferred background.
2. From existing permitted validation field archives, produce a prespecified descriptive table for the supervised image head: eight-class field accuracy, macro F1 and per-class recall for each of 40×/100×/200×/400× and pooled views. Report natural image-weighted metrics separately from current class/case/view-weighted metrics. Preserve both arrangements and all trials; do not select the most favorable slice. This requires no new fitting and does not score excluded cases. State that current ResNet predictions are a new supervised reference, not an EfficientNet notebook reproduction.
3. Present 40× image performance as the bridge endpoint and all four magnifications plus all 15 case subsets as the proposal-extension endpoint. The single-view case score is not a substitute for a single-image score. No task or split is promoted to primary because its observed score is larger.
4. If a new training experiment is needed for the capstone's concrete requirements, register one bounded EfficientNetV2B0 standard supervised reference before execution, with existing case roles and the image endpoint fixed in advance. Use a simple head before costly custom attention, separate validation from any final reporting claims, retain preprocessing expected by the pretrained implementation, and report the failed current recipes too. Reusing current cases makes this further development evidence. A notebook-style random-image procedural reproduction is optional and subordinate; audit overlap before interpreting its score, and never use it as evidence of clinical generalization.
5. Package an inference example, immutable split/configuration/label maps and concise reproducibility instructions. The capstone can demonstrate a functioning reproducible classifier and a rigorous finding about evaluation difficulty even when measured generalization is weak. Future novel-method work requires a fresh evaluation strategy and matched-encoder/budget controls; it is not the urgent completion gate.

Minimum controls for any new fit are the existing parsing/disjointness/save-reload checks, fitting-only preprocessing and training, fixed image/case readouts, all four view slices, per-class support, and complete paired reporting. Changes to cropping, stain handling, loss, backbone, split and pooling together would prevent attribution. A clinically verified external cohort could eventually strengthen generalization evidence; merely repartitioning already explored BreakHis cases cannot create an untouched confirmation study.

## Review boundary

This review does not certify historical notebook metrics, establish a causal source of poor generalization, prescribe a new favorable partition, or authorize a clinical claim. It recommends a complete capstone whose scope and conclusions match the evidence. Root implementation/registration work may proceed within the user's existing authorization; another generic permission checkpoint is unnecessary.
