# BreakHis improvement: execution checkpoint

November 12, 2026. BreakHis is the main task. The first baseline is checked; the project is still being improved. TCGA remains deferred.

## What we are trying

The old picture reader stays fixed. The new experiment lets its last block learn from BreakHis training pictures. We then use the same case classifier and scoring rules, so we can compare the two fairly on the same cases. This is standard supervised adaptation, not a new-method claim.

One fixed recipe: six epochs, with one head-only warmup epoch and five last-block/final-LayerNorm epochs. Ten production models cover the existing five trials and both split arrangements. Each uses 46 fitting, eight validation and 28 excluded outer-role provider cases. All 15 view combinations remain in the score. [Registered settings](BREAKHIS_ADAPTATION_SPEC.md).

## Evidence so far

- Training implementation, checkpoint/resume, split/sampling guards and numerical checks passed. The complete local suite has 70 passing tests; synthetic checks do not measure BreakHis performance.
- The separate fitting-only technical smoke, SLURM4962114, completed successfully in59 seconds. It made four minibatch updates total, produced no validation or outer predictions, and is never ranked as a model result.
- Its frozen prefix stayed unchanged. Head-only warmup preserved the encoder; subsequent updates changed both the last block and final LayerNorm.
- Starting features matched the original frozen cache within0.000001312; the final encoder reloaded with maximum feature difference0. Actual checkpoint and source bindings passed the separate artifact checker.
- Ten production models were submitted as SLURM array4962271, capped at four simultaneous GPU tasks. The launch checkpoint showed four running and six waiting for the array limit. Full results are pending; this document is a checkpoint, not a live monitor.

Immutable training source: `3a8dde70d809b115928bf773511a0cd6e0f8e30c`. Configuration SHA256: `224c0de10715bdc2248653e4d4c0803aa510823e2efbfba44e2a2229f3b156e6`. Technical checkpoint SHA256: `21ee668fc45f1eaba2ea0b1aebc6f93846cfa5b76d7c8f9b07cd9927eb14055d`.

## What happens after training

The collector requires all ten runs before comparing methods. It checks the exact epoch/draw budget, phase/frozen-fragment evidence, source/configuration/role bindings, canonical tensor shapes, actual AdamW settings/update counts, checkpoint/cache/head checksums, exact54-case feature coverage, fitting-only normalization and all saved validation probabilities. It recomputes scores from trusted labels. No partial ranking or outer scoring is allowed.

Report both split averages and every paired trial, with macroF1, accuracy, balanced accuracy, all-view/worst-view scores, subtype recall, confusion matrices and Brier error. A promising descriptive result should improve mean-subset macroF1 on both split arrangements; a split gain and split loss is mixed evidence. Keep every regression. Subsequent validation-driven experimentation must be separately justified and registered.

## How far from strong numbers?

The checked baseline has mean case accuracy45.7% and38.3% on the two split arrangements, and phyllodes recall0 on these validation records. The new experiment has no verified performance result yet. We cannot give an honest time estimate to80% or90% accuracy.

Both adenosis and phyllodes have only one fitting provider case per run. Repeated pictures of that case do not create more disease variety. All82 cases have also entered the wider development history; clinical identifier relationships remain unresolved. Better development numbers alone will not establish fresh clinical generalization or novelty. [Independent review](reviews/BREAKHIS_ADAPTATION_REVIEW.md).
