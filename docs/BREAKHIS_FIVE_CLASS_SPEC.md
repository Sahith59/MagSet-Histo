# Registered first five-category comparison

November15,2026. Registered before any new five-category fitting/outcomes. This is the first trained five-class control/hierarchy study; old eight-class/binary predictions were already inspected. It is exploratory development under reused provider-case validation, not a fresh test.94–95% is a goal, not an assumed result. [Canonical config](../configs/breakhis-five-class-v1.json).

## Task, input and roles

BreakHis only: final labels benign/DC/LC/MC/PC, with original eight-to-five map[0,1,0,2,3,4,0,0]. Keep all four zooms and evaluate all15 nonempty availability subsets. One prediction unit is a provider case, aggregating its fields; no physical alignment across magnifications is invented. All existing related fields/zooms/derivations retain the same case role.

Reconstruct the existing ORIGINAL-EIGHT-LABEL role memberships first, then pool labels. Five trials × inner arrangements812/813 give exactly ten role assignments, each46fitting/eightvalidation/28excluded. Fitting support in final-five order is[11,25,2,5,3]; validation[4,1,1,1,1]; excluded[9,12,2,3,2]. Stage1fits46cases; malignant expert fits35true-malignant cases. There is no benign expert. Both arms use identical fitting/validation cases; excluded cases are never predicted by these models.

The28excluded cases are excluded per run, not globally untouched. There are22/20 distinct validation cases,31 in their union; all82 provider keys entered development somewhere and clinical-patient relationships remain uncertain. A new configuration or resplit would not restore a fresh cohort.

## Shared representation and normalization

Reuse all four accepted original Phikon-v2 caches pinned by NPZ/JSON byte hashes, exact manifest coverage and pinned encoder revision. No new image encoding, feature adaptation, stain fitting, quadrant alternative or feature selection. The official processor uses shortest-edge resize plus224center crop; cached vectors are1,024-dimensional.

Use the existing within-zoom float32 field mean, equal-case fitting-only field first/second moments (`field-moments-v2`), standardized view means, available-view mean and four binary availability flags:1,028classifier inputs. The same normalizer fitted on all46fitting cases is shared by both arms and both hierarchy heads. Expert fitting must not introduce a different validation-informed normalizer.

## Two complete pipelines

**Direct control:** class-balanced five-class multinomial logistic classifier.

**Dual step:** class-balanced binary logistic router, followed only for predicted malignant cases by a four-class multinomial logistic specialist. The router is fitted on all46cases; the specialist only on35cases whose true fitting label is malignant. Fixed malignant probability≥0.5routes to the expert, including ties. Benign routing returns benign. Expert argmax follows fixed DC/LC/MC/PC order on ties.

Each head uses L2/lbfgs, tolerance1e-4, max_iter1000. Every fitting case's base mass is spread over all15subsets with sample_weight1/15. Head-specific class_weight='balanced' gives equal total weight to that head's classes: five for direct, two for router and four for specialist. Within a head/class cases have equal influence. These differing head-balanced/factorized objectives mean this is a comparison of two declared complete pipelines, not proof that hierarchy alone causes an effect.

For both arms test C=[0.01,0.1,1,10]. For the hierarchy, both heads share ONE C; do not search their settings independently. Select the earliest/smallest C attaining maximum end-to-end five-class mean15-subset validation macro F1. Complete all four candidates in both arms for all ten roles. This is four complete-pipeline selection opportunities per arm/run; direct uses40individualheadfits and dual80, for120total. Costs/capacity are not identical. No new technical real-data fits or encoder forwards are allocated. Unit fixtures plus zero-fit actual input/role/normalization audit gate production.

## Explicit decisions and probability scores

Save explicit final labels. Hierarchy labels use the hard router then expert argmax; they must NOT be replaced by global argmax of the five composed probabilities. The coherent probability vector is[1-r,r*s_DC,r*s_LC,r*s_MC,r*s_PC]. It may have a different global argmax. Use explicit decisions for accuracy/F1/recall/confusion and selection; use stable composed log probabilities for cross-entropy and probabilities for Brier. Keep this distinction in every artifact/check. Reported scalar binary scores may be clipped to[0,1]only for floating-point endpoint roundoff; preserve stable log probabilities and the fixed threshold.

Direct binary and malignant-conditional readouts are derived diagnostics; its final decision is five-way argmax. Its binary diagnostic explicitly sums the four malignant probabilities and applies≥0.5with ties malignant; this may differ from the broad group of its five-way argmax. For the hierarchy, binary diagnostics measure the actual router. Score malignant typing on true-malignant cases as a diagnostic; it cannot replace end-to-end performance. Wrong routing is always a final error. With unchanged old binary decisions, five-class accuracy is bounded by old87.0%/80.7%; that is not a forecast for the new router.

## Evaluation and descriptive decision gate

Report every candidate, every trial, both arrangements, all15subsets and full/worst-view performance. Primary selection/comparison metric is mean15-subset five-class macro F1. Also report final accuracy/balanced accuracy, all-five recall/confusion, stable CE/Brier, binary sensitivity/specificity/AUC with correct names, malignant typing and fitting–validation gaps. Raw accuracy and balanced accuracy now differ because benign occupies half of validation. Always-benign gives50%raw accuracy and20%balanced accuracy.

A dual-step improvement screen requires mean-subset macro F1 at least0.02above direct on BOTH arrangements, raw mean-subset accuracy not lower on either, and positive pooled recall for each final label on both. Passing is exploratory, not clinical/novelty evidence. Failure does not trigger a widened C/threshold/encoder/crop/resplit sweep.94–95% achievement may be claimed only if actually measured on the declared end-to-end task, with exact view/role/selection meaning.

## Artifacts, verification and execution budget

Each run retains all four candidates for each arm, their normalization/coefficients/class order, fitting/validation probabilities/log probabilities, explicit decisions, original and final truths, route/expert scores, convergence/iterations and metadata hashes. Independent reload must reproduce sklearn probabilities and hard decisions. Store no outer-role predictions. Private records retain IDs; public aggregate reports remove identifiers and machine paths.

Before fitting verify manifest/protocol hashes, exact cache NPZ/JSON hashes and coverage, immutable Git archive source, canonical config/dependency hashes, all ten original role assignments and supports. Reject convergence warnings. No partial matrix ranking. After all ten runs complete, reconstruct selections and arithmetic from saved artifacts and require an independent actual-output review before promoting a result.

Use SLURM CPU jobs with a zero-fit audit gate, then a ten-task array capped at four concurrent two-CPU workers,12GB/request and30minutes/task; a separate complete-matrix collector follows successful completion. CPU heads reuse accepted GPU features, so GPU allocation would add no useful work here. Leave unrelated cluster jobs untouched. Preserve immutable sources and refuse existing output paths. No external messages/submissions.

Parallel review lanes: three independent GPT-6.1Sol extra-effort diagnosis agents (data/split, model/features, errors/metrics); metrics implementation and runner separated by file ownership; independent research/critique next; actual-output arithmetic replay after production. Scheduling overlap is not a scientific sample-size increase or proof of speedup.
