# Next tasks: BreakHis first

Priority reset November 12,2026. **BreakHis is the main dataset. Finish its quick project deliverable for Professor Dong first.** TCGA work and the broader novel-paper route are deferred; completed artifacts are preserved.

## One small milestone at a time

| Order | Task | Timing / evidence |
| --- | --- | --- |
|1|Publish checked BreakHis baseline numbers|Available now: both validation splits and all ten trial results are in BREAKHIS_FIRST_RESULTS.md|
|2|Verify prediction from original images|Completed: job 4959781 replayed 121 original images and 15 view settings with its saved model; maximum probability difference 0.0000000552|
|3|Package the baseline for Professor Dong|Complete baseline package: results, verified replay, model/source hashes, class weaknesses and limits; professor brief updated|
|4|Run the registered BreakHis improvement|Completed: all ten fits and checker passed; primary two-split improvement gate failed; frozen baseline retained|
|5|Resolve final evaluation and research claims|After an accepted grouping/claim scope and method freeze; no claim of a pristine clinical test cohort or novelty now|

The [completed adaptation](BREAKHIS_ADAPTATION_RESULTS.md) failed its improvement gate. The [saved-model diagnosis](BREAKHIS_DIAGNOSIS_RESULTS.md) found a large fitting/validation gap; the [coverage/reference audit](BREAKHIS_REFERENCE_AUDIT.md) is complete. The [fixed supervised ResNet-50 reference](BREAKHIS_SUPERVISED_RESULTS.md) also completed all ten fits and full verification: F1.3084/.3006, accuracy37.0%/40.0%, phyllodes recall0. Its gate failed. Keep the frozen reference; no automatic model sweep follows. [Execution record](BREAKHIS_SUPERVISED_STATUS.md).

## What is already done

7,909 images verified; all four magnifications indexed; frozen features extracted;175 exploratory BreakHis configurations completed; ten matched-mean runs independently rechecked against trusted metadata and 1,200 saved validation records. Mean remains ahead of the checked matched neural controls. The baseline report and raw-image replay are both checked. The remaining project work is to improve performance and settle final evaluation, not to claim novelty.

## What “done” means

The first project milestone is complete only when the stated baseline, report and raw-image replay checks pass. It does not mean that all eight classes are classified reliably, new-patient clinical performance is established, a novel mechanism works, or a paper is ready for submission.

Use “provider-case exploratory validation”: each trial has 46 fitting,8 validation and 28 excluded outer-role cases. The trials overlap, all 82 cases have entered development somewhere, and clinical identifier relationships are unresolved. Show both validation splits and all trials, not one favorable example.

## Work kept off the critical path

No new TCGA experiments, TCGA label investigation, extra datasets, broad literature expansion or novel architecture search on this project's immediate path. Stay on BreakHis through improvement, evaluation and its project report; completing the baseline milestone alone does not switch the priority back to TCGA. No emails, access applications or conference submissions are sent automatically.

Every status update should explain plainly what we just did, what the evidence says, what remains unchecked and the next concrete action. [Focused acceptance plan](BREAKHIS_MILESTONE_PLAN.md) · [First results](BREAKHIS_FIRST_RESULTS.md) · [Training strategy](TRAINING_STRATEGY.md).

## Diagnosis completed November 13

Job4968029 completed all ten fixed probes in70seconds, with no new fits, encoder forwards or outer predictions. Independent auditing recomputed32,400 case-score records and51,760 field-score records (dependent repetitions, not new cases). Mean fitting/validation balanced case accuracy is98.5/45.7% and98.7/38.8%; auxiliary weighted field accuracy is98.1/35.0% and97.9/31.8%. The affine-pooling identity and saved archive replays passed.

Keep the frozen baseline while the registered comparison runs. Do not infer that a bigger model or longer training follows automatically from this diagnosis.

## Current next actions

1. The supervised comparison and independent numerical review are complete. Preserve the full negative result and all original/recovery execution evidence; keep the frozen reference.
2. Consolidate the [BreakHis report](../paper/breakhis-project-report.md) and professor brief: verified baseline, two failed improvement recipes, scarce independent rare-class examples and unresolved grouping/confirmation limits.
3. Agree on the intended prediction unit, acceptable provider/clinical claim scope and credible final-evaluation route with Professor Dong before registering another model experiment. No outer-case rescue, score-driven resplit, automatic architecture/rate/epoch sweep or switch to TCGA.

Training execution is complete. Predictive quality and final evaluation remain unresolved; finishing jobs is not evidence that the problem is solved.

## November14 capstone priority update

The user emphasized the approaching master's capstone and requested three independent GPT-6.1 Sol reviewers. Original saved numbers93.73%/94.99% are verified historical40× image validation, not the same endpoint as today's case/subset score. [Before/now explanation](CAPSTONE_BEFORE_AND_NOW.md). Deadline and formal task have been requested; until clarified, preserve eight subtypes and all four magnifications.

The no-fit [image endpoint bridge](CAPSTONE_BRIDGE_RESULTS.md) is complete and independently checked. One prospective [regional-view representation](BREAKHIS_MULTICROP_SPEC.md), with unchanged labels/roles and fixedC0.01 in both arms, is now launched. Smoke4971319 passed; four GPU tasks4971322 run frozen extraction, and CPU job4971323 waits for complete audited caches before exactly20 fits. [Execution checkpoint](BREAKHIS_MULTICROP_STATUS.md). This is the only new fitting budget; no follow-on sweep is implied.

Next: verify all four caches and all20 outputs, independent numerical review, then update the capstone report with every result. Continue report/demo preparation while those bounded jobs run. Any later EfficientNet reproduction needs an explicit image-task contract; do not copy a case-overlapping split or treat original95% as a new-patient target. Novelty and the paper remain later milestones.
