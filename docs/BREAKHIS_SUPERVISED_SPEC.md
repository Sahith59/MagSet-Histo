# Controlled supervised image-model reference

Registered November13,2026 before fitting or outcomes. BreakHis remains the main task. The previous frozen baseline remains the reference; this is one standard transfer-learning pipeline, not a novel method or exact reproduction of an author paper.

## Motivation, scope and budget

The saved-model diagnosis found high fitting scores and weak validation in both Phikon-based heads. No tested readout rescued F1. Test whether a smaller, fully supervised convolutional reference with regularization transfers better under the same difficult case protocol. This compares complete pipelines; it does not isolate the causal effects of backbone, preprocessing, augmentation, regularization or aggregation.

Exactly ten production fits: five existing provider trials × inner seeds812/813, training seed812. One separate fitting-only technical smoke is permitted: three epochs, at most two batches each, with no validation decoding/scoring. Two head-only epochs plus one backbone epoch make the smoke exercise the actual unfreeze. No parameter/architecture sweep, extra training seed, calibration fit or outer prediction.

Retain each run's46 fitting, eight validation and28 excluded outer-role provider cases. Reconstruct and bind membership to the trusted manifest/protocol and accepted frozen reference. Construct the fitting/validation image allowlists before opening raw files. The numeric-stem clinical uncertainty and lack of a globally pristine cohort remain unchanged. The coverage audit confirms A/PT each have one fitting provider case; augmentation does not add independent cases.

## Fixed model and training

Use torchvision0.21 ResNet50 initialized from the cached ImageNet1K_V2 checkpoint SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. Its2048D output feeds Dropout(.3)+Linear(2048,8). Train the head for epochs1–2; epochs3–20 also train all backbone parameters, including BatchNorm affine parameters. Keep pretrained BN running statistics/counters fixed in evaluation mode throughout; verify their hashes. Verify early/late backbone weights stay unchanged in warmup and change after unfreezing.

AdamW: backbone LR1e-4, head LR1e-3, weight decay1e-4, gradient norm cap1. Head rate stays constant during warmup. From epoch3 onward, both rates follow the same fixed cosine multiplier from1 at epoch3 to.1 at epoch20. Plain cross-entropy with label smoothing.1; no additional inverse-class loss weighting. Float32, deterministic algorithms, TF32 disabled.

Batch32, no dropped last batch. Each epoch makes exactly N fitting-field draws with replacement; each class has expected mass1/8, cases equal within class, four views equal within case, fields uniform within view. Record realized class/case/view counts and exact optimizer updates. Sampling uses a fixed epoch generator. Augmentation depends on training seed, epoch, draw index and path, so repeated draws can differ and worker count does not alter transforms.

Only fitting images receive H/V flips, quarter-turn rotations and mild brightness/contrast/saturation factors [.9,1.1] in that fixed order. Resize the entire RGB field to224×224 using bicubic interpolation and ImageNet normalization. No crop, stain template or dataset-derived image normalization. Canonical evaluation has no augmentation. This differs from the old frozen ResNet cache's232-short-side/224-center-crop processor; do not reuse that cache or claim processor identity.

## Fixed selection and scoring

Evaluate validation only at epochs5,10,15,20. Select the maximum mean macroF1 across all15 subsets; ties keep the earliest epoch. These are four prespecified checkpoint opportunities, versus four C choices in the saved linear reference; search spaces/optimization budgets differ, so this is not an identical-tuning-budget fusion ablation. Run all20 epochs without validation-driven stopping.

Keep the trained image head. Average field probabilities within each magnification, then equally across available views. This is the sole primary readout; do not choose between logit/probability/max/attention pooling. Save all four checkpoint-selection score sets and every selected fitting/validation field probability/log probability. Verify source/config/data/checkpoint identity, exact paths/truths, stable losses, and replay every case/subset score from saved field arrays. No outer image is opened by the fitted model.

Retain one full last resumable checkpoint with optimizer/RNG/history and a cloned best-state snapshot for atomic recovery, plus one selected model-state file. The duplicate best state bounds storage at roughly5GB for ten runs; do not retain every model epoch. Commit the last checkpoint atomically before deriving the selected file from its best snapshot, so interruption cannot silently replace an earlier selected model. Resume only the same source/config/roles/budget; never extend20 epochs. Technical failures must be documented and fixed without a hidden replacement recipe.

## Acceptance and interpretation

Before production: meaningful tests, a real fitting-only GPU smoke, save/reload parity on four fixed fitting-only probe fields, warmup/unfreeze/BN-stat proof, and independent review. At most four production GPU tasks run concurrently through SLURM. Leave unrelated jobs alone.

Require the complete ten-run matrix before comparison. Report all trials, both split means, all15 subsets, fitting-vs-validation gaps, per-class recall/confusion, raw and balanced field metrics, Brier/CE, selected epochs, model size and runtime. Preserve every negative result.

A working descriptive improvement gate is at least+.02 mean-subset macroF1 versus frozen Phikon on BOTH inner arrangements, plus nonzero phyllodes case recall on both. This is a project screening rule, not statistical significance or a clinical sufficiency threshold. Tiny dependent validation samples cannot certify reliable disease performance. A result that fails the gate does not authorize an automatic longer run, new backbone or new search.

The public graph-code audit is a compatibility reference, not a direct claim that its headline score should be reproduced here. Released CSV identities/fold coverage need their own audit. Keep original author scores separate from our implementation results. TCGA/novelty work stays deferred.

## Technical cleanup before production

The first technical smoke4969999 completed0 in43seconds and passed fitting/reload checks, but early termination of its oversized loader printed worker-shutdown warnings. Preserve its original source6b41859/logs/checkpoint; it does not clear the production gate. Before production, limit the smoke sampler itself to exactly64 draws per epoch, exhaust that loader normally, and repeat the same six-update fitting-only check once under a new immutable source in `technical-smoke-retry1`. No model setting, validation selection or production draw budget changes. Count both technical fits separately from the ten production models. The repeated smoke must pass artifact verification with a clean error log. Also reassert the final BN-buffer invariant when resuming an already-complete checkpoint.
