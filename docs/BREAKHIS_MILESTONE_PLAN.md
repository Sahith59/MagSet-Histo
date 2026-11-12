# BreakHis first: quick project deliverable

Priority reset November 12, 2026 at the user's request. **BreakHis is the main dataset. Finish its first project deliverable before pursuing a novel paper or further TCGA work.** Completed TCGA artifacts are retained as deferred work; they do not determine this milestone.

Baseline completion is only the first milestone. Continue BreakHis improvement, evaluation and the project report afterward; do not resume TCGA automatically after this checkpoint.

## What the first deliverable means

A repeatable eight-subtype BreakHis baseline using all four magnifications, a checked development-results table, and a raw-image prediction demonstration. This is a working project baseline, not a claim that the classification problem is solved, clinical generalization is verified, or a new mechanism has been invented.

| Item | Fixed choice |
| --- | --- |
| Dataset | Official obtained BreakHis release: 7,909 images |
| Inputs | Unpaired image fields from one provider case at available 40×/100×/200×/400× views |
| Output | Eight class probabilities and one predicted subtype |
| Image reader | Pinned frozen Phikon-v2; 1,024D CLS image embeddings |
| Case model | Mean fields per view, fitting-only saved normalization, mean available views plus four availability indicators, saved logistic head |
| Evaluation | Existing five provider repeated holdouts, each 46 fit /8 validation /28 excluded outer-role cases |
| Checks | Inner split seeds812 and813, every one of15 nonempty view subsets |
| Selection | Existing declared C grid; no new tuning or selection from outer predictions |

## Milestones and timing

1. **First numbers: available now, November12.** Recheck the ten existing matched mean runs against trusted labels/splits and saved probabilities. Report both splits, including accuracy, F1, all-view results and rare-class weaknesses. These are development numbers.
2. **Working baseline demonstration: this work session, subject to cluster access/allocation.** Use the already saved trial1/seed812 checkpoint. Select the lexicographically first validation case before looking at its prediction; use all original fields and compare all15 subset probabilities with its archived predictions. Do not choose a successful example after seeing answers.
3. **Professor-ready baseline handoff: after demonstration checks pass.** Include one main results page, prediction command/output, reproducibility hashes and explicit unfinished items. Keep the explanation brief and child-readable. This completes the baseline milestone only.
4. **Next improvement: one bounded BreakHis comparison.** After the baseline is packaged, specify one ordinary improvement such as limited supervised adaptation of the image reader versus the frozen baseline. Fix its settings and fitting budget first. Target the next one or two work sessions for the registered experiment and initial results, conditional on implementation/queue; this is an estimate, not a promised score/date. No broad architecture sweep or new novelty claim is required for the quick project.
5. **Final evaluation and paper: separate gates.** Resolve acceptable grouping/claim scope and freeze the method before a final evaluation. The first project report can honestly present exploratory provider-case performance while these research questions remain open.

## Demonstration acceptance checks fixed before execution

- Reuse the actual saved head and normalization; verify the run against the original manifest/protocol.
- Accept a nonempty subset of the four views. Excluded view pixels/features must not affect a prediction.
- Encode raw images with the same pinned encoder/transforms and float32/TF32-off policy as the cache.
- Check original image SHA256 values, safe paths, image counts and finite feature/probability shapes.
- Compare the chosen case's complete-field raw-image predictions with every archived subset. Maximum probability difference must be at most0.0001. Record the observed difference and actual source/model/data hashes.
- Do not fit anything or score an outer-role case in this demonstration. Stop on a mismatch rather than silently claiming parity.

## Limits we must keep visible

The five trials overlap. Eight validation cases per trial are a tiny sample. All82 provider cases have entered fitting or validation in some earlier trial, so there is no globally fresh BreakHis confirmation group. Some clinical identifier relationships remain uncertain. State “provider-case exploratory validation,” not independently verified patient-level testing.

The historical notebook accuracies are from a different image-based split/selection procedure. They are not directly comparable with these eight-class case-level results. A higher number from an easier or less protected split is not evidence of improvement.

The model may make wrong predictions in the demonstration. An honestly selected incorrect example is still valid execution evidence; it must not be described as successful classification.

No further TCGA training, label research, external-dataset expansion or novelty work is on the immediate critical path. Update this plan from verified evidence, not anticipated success.

## Verified checkpoint

The first baseline package is ready. Ten matched runs/1,200 saved validation records passed trusted checks. Job4959781 completed0 in44seconds, reading121 original images and replaying all15 view settings of the fixed first validation case. Maximum raw probability difference5.517742757e-08; cached120-record replay difference4.44e-16. No new fitting or outer-role prediction. All60 software tests passed. This establishes baseline execution and reported exploratory scores, not reliable clinical performance, method novelty or a completed paper. The next project task is a registered standard BreakHis improvement.
