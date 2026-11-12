# Baselines and apples-to-apples comparison contract

Updated November 12, 2026. This is an exploratory development benchmark, not a completed novel-method result.

## Baselines already measured

| Family | Implemented controls | Completed evidence |
| --- | --- | --- |
| Frozen image encoders | Phikon-v2 and ImageNet ResNet-50 | All four BreakHis magnification caches verified |
| Case-feature linear fusion | Masked mean, masked concatenation, average available scale probabilities | 30 initial configurations |
| Field aggregation | Classify fields then average probabilities; normalization-matched early-mean control | 20 configurations |
| Hierarchical learned fusion | Mean or gated-attention field pooling crossed with mean, contextual gating, or self-attention view fusion | 30 configurations |
| Stability checks | Alternate inner split; alternate training seed; repeated linear control | 45 configurations |

The 125 configurations include repeated trials/seeds, not 125 independent datasets or patients. The highest initial descriptive Phikon mean score is 0.3868 mean validation macro F1; the best initial neural row is 0.3562. Both use only eight validation cases per trial. The linear and neural rounds used different normalization, so their headline difference does not isolate architecture. [Saved results](FUSION_RESULTS.md) and [stability evidence](SENSITIVITY_RESULTS.md) remain unchanged.

## What a fair mechanism comparison must hold fixed

- The same eligible provider cases and exact fitting/validation/outer assignments, including every related field and magnification. Clinical identifier sensitivity remains a final-benchmark requirement.
- The same pinned encoder, image transforms, frozen feature files, field budget, and fitting-case-only normalization. Backbone adaptation must be a separate experiment or shared across all methods.
- The same eight labels, all 15 nonempty subsets of 40×/100×/200×/400×, case/class weighting, selection objective, and prediction unit.
- The same prespecified validation opportunity and training-seed policy. Neural comparisons share optimizer grid, fixed epochs, case batches, and subset objective unless the changed element is explicitly the experiment.
- Report primary mean-subset macro F1, full/worst-subset scores, subtype recall, all split/seed outcomes, parameter count, and compute. Do not select only a favorable trial.
- Match parameter capacity when testing the proposed component. Removing a component must not silently remove the entire residual/classifier path or change available inputs.
- Keep outer cases untouched until method/protocol choices freeze. Final intervals must account for repeated cases across provider holdouts.

Linear versus neural methods necessarily have different fitting algorithms. Four configurations each is a matched selection-count budget, not proof of equal compute or equal capacity. Within-neural causal ablations require tighter controls.

## Next registered comparison: normalization, capacity, and regularization

Before submission, fix this matrix: Phikon-v2, both existing inner split seeds 812/813, training seed 812, five provider trials, all fields, all 15 subsets. Do not repeat completed original neural runs.

1. Ten new CPU mean-linear fits use `--normalization field-moments-v2`, exactly the training-case-balanced field moments used by hierarchical models. Historical `case-mean-v1` remains the default so old results reproduce.
2. Forty new GPU configurations compare mean/mean and attention/attention controls under two separate changes. Small-capacity changes only hidden width 128→32. Stronger dropout changes only 0.2→0.5. All other settings, including 100 final epochs and four optimizer settings, stay fixed.
3. Reference the 20 already completed original neural configurations with hidden 128/dropout0.2 at those same splits/methods. Each seed/policy is a separate complete matrix.
4. The collector recomputes metrics from probabilities, checks identical cases/cache provenance, and compares actual saved linear/neural normalization arrays for exact equality. It rejects mismatched inputs and unintended configuration changes.

This is a targeted test of the measured near-zero-fitting-loss/weak-validation pattern. It is not another arbitrary architecture sweep or a new mechanism claim. Training is capped at four GPU and four CPU workers. All outcomes are retained.

## Which published works we need to compare against

- [MagViT](https://arxiv.org/html/2608.16959v1) is the closest missing-magnification task comparator. Its source task is binary and its gate/encoder/grouping differ. Our contextual gated baseline is not its full reproduction. Reproduce an exact compatible track where possible and label an eight-class adaptation explicitly; do not compare its binary headline accuracy directly with our eight-class macro F1.
- [M2OTCA](https://doi.org/10.1016/j.media.2026.104082) is a close cross-magnification matching comparator. Full equations/implementation remain to verify. A clearly documented adaptation and an ordinary cross-attention/transport control are required before claiming a new matching mechanism outperforms prior work.
- Standard attention MIL and set attention are implemented as controls, not claimed as inventions. Our one-block attention model is not the full published Set Transformer architecture.

Exact MagViT/M2OTCA reproductions, the candidate matching model, and capacity-matched candidate ablations are not completed. We have established internal baselines and a reproducible protocol; we have not established state-of-the-art superiority.

## Second-dataset comparison

All 65,312 TCGA-MS candidate images now passed full RGB decoding at 224×224 with zero errors. It has 16 labels and seven resolutions; those differ from BreakHis. Define native eligibility, class mapping, missing-class handling, resolution subsets, and pretraining-overlap checks before fitting. Compare every method within that same second task. Do not merge the two datasets' label spaces or average their raw scores into one interchangeable leaderboard.
