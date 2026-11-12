# Minimal native development pilot: fixed four-fit budget

Registered November 12, 2026 before any native classifier scores. This specification does not claim the candidate is implemented, novel, or effective. It is the next small experiment after the trusted-data and provenance gates, not another architecture sweep.

## Immutable inputs and quarantine

Use the registered TCGA protocol and pinned frozen ImageNet ResNet caches only. Fit 80 patients; development 16; sealed 32. The manifest/protocol hashes in `TCGA_QUARANTINE_PROTOCOL.md` are fixed. No source-fold training, prototype/statistic fitting, calibration, or prediction may involve sealed patients. Feature caching is allowed because the encoder is frozen and labels never enter it.

Use all observed fitting fields at native 0.25/0.5/1.0/2.0mpp. Six incomplete patients remain fit-only. For each fitting patient enumerate nonempty subsets of its actually available views and weight each by the inverse number of those subsets. Thus every patient contributes total mass one regardless of field/view count. Canonical labels 0..15 have five fitting patients each. Compute per-view feature moments using fitting patients only, with equal case weights over fields.

## Four prespecified arms

1. Standardized available-view mean plus four availability indicators.
2. The same baseline block plus a 32D projected duplicate-mean block (feature duplication/penalty control).
3. The same baseline block plus a 32D fixed random-partition field contrast (same-dimension extra-feature control).
4. The same baseline block plus a 32D cross-view affinity-agreement contrast (candidate hypothesis).

Use one fixed label-blind orthogonal random projection, seed 812, shared across arms 2–4. Random field partition weights must use a fixed hash of each field's immutable path so permutation of fields does not change a control. Fit exactly one case-weighted multinomial logistic model per arm, `C=0.1`, L2, max_iter 1000. Do not search learning rates, C values, widths, temperatures, seeds, or view budgets in this pilot. Optimization failure is a software/solver issue to report, not a reason to select a more favorable setting after scores.

For affinity, standardize fields with fitting-only moments, project and L2-normalize for cosine comparison, and use temperature 0.2. For each field average peer-wise log-mean-exp affinities, then softmax within its view. Average peer contributions so view count does not multiply a signal. The contrast is the weighted projected mean minus the original projected mean. Use zero cross-view contrast for a one-view input. This preserves the baseline feature block; it does not guarantee identical fitted predictions or nondegradation.

These are fixed design choices for a falsifiable feasibility check, not proven optimal hyperparameters or a novel mathematical formulation. Ordinary attention/shared-specific modelling and partial OT already have prior art. No OT solver is needed in this first pilot.

## Required correctness checks before execution

Verify labels and per-patient roles against registered metadata; refuse sealed scoring in the development CLI. Test field/view permutation, missing-content invariance, one-view zero contrast, finite probabilities, case-balanced subset weights, save/reload, and conservation of matched/unmatched branches. Equal added dimensions do not alone establish matched capacity; report coefficient counts and fit/feature compute.

## Development outcomes and decision

Compute all 15 subsets on the same 16 development patients. Report fixed 16-class macro F1, full/worst subset, class recall, balanced accuracy, Brier score, and per-patient predictions. These are development results with only one patient per class, not a final significance test. Preserve every arm even if it loses. Do not inspect reserved predictions to choose a hypothesis or revise these choices.

If affinity does not exceed the mean and same-dimension controls, stop the agreement-specific claim and inspect the documented failure pattern. If it does exceed them, that is only a signal to run prespecified stronger heterogeneity/association-disruption controls; it is not permission to label the method novel or reveal sealed outcomes. Additional comparisons and a final freeze are required before one confirmation evaluation on 32 reserved patients.

Professor Dong's review should determine whether this narrow hypothesis, a benchmark/robustness study, or a different contribution is the best scientific direction. The coming-week package already includes honest baseline and protocol evidence even if this pilot is not yet executed.
