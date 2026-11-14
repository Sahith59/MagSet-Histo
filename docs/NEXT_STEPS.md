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

The [completed improvement](BREAKHIS_ADAPTATION_RESULTS.md) did not meaningfully improve accuracy and failed the two-split F1 gate. Keep the frozen baseline. The [saved-model diagnosis](BREAKHIS_DIAGNOSIS_RESULTS.md) found a large fitting/validation gap. The [coverage and reference audit](BREAKHIS_REFERENCE_AUDIT.md) is now complete, and one [fixed supervised ResNet-50 recipe](BREAKHIS_SUPERVISED_SPEC.md) was registered before fitting. Its ten full runs are submitted, with at most four simultaneous GPUs; the complete checker is queued after the array. No improved scores are established. [Dated execution checkpoint](BREAKHIS_SUPERVISED_STATUS.md).

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

1. Finish all ten fixed ResNet-50 fits and the automatic artifact checker. Require the complete matrix before ranking; do not use interim trial scores to change settings.
2. Independently inspect actual outputs, then publish every trial, both arrangements, all15 subsets, subtype recall/support, fitting–validation gaps and runtime/model size. Apply the already registered descriptive gate.
3. Update the [BreakHis project-report draft](../paper/breakhis-project-report.md) and professor brief with checked results and limitations. Confirm acceptable grouping/claim scope and final evaluation with Professor Dong before stronger claims.

Coverage auditing, the technical smoke, implementation review, report structure and the first-results package are done. The remaining bottleneck is the measured supervised comparison and a credible final-evaluation route. A failed reference does not authorize an automatic architecture/rate/epoch sweep.
