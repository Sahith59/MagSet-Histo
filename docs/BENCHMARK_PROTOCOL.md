# Frozen-feature validation pilot v1

This document describes the implemented initial pilot. It does not assert a final patient-level benchmark or a new mechanism result.

## Data and prediction unit

Use the official 7,909-image archive and the recorded image manifest. A prediction case is the provider-defined subtype-qualified identifier, reconstructed from the filename prefix and raw identifier. There are 82 such cases. Preserve the distinction from 81 unqualified identifiers and 70 stripped numeric stems.

The original five trial lists are repeated holdouts. Each trial assigns 54 cases to training and 28 to testing, with no provider-case overlap. The original test assignments are retained and unused by the pilot. Shared numeric stems can cross these assignments; a stricter stem protocol is a separately reported sensitivity analysis, not silently substituted for the provider's protocol.

For each trial, choose one inner-validation case per subtype from outer training using seed 812 and sorted case lists. This gives 46 fit cases, eight validation cases, and 28 untouched outer cases. No test case enters fitting, normalization, or hyperparameter selection. The validation set is very small, especially for rare classes; pilot scores must not be interpreted as precise generalization estimates.

## Frozen encoders and transforms

- Phikon-v2: repository/revision in `configs/encoders.json`; official processor; CLS token from the last hidden state, 1,024 dimensions.
- ResNet-50: torchvision 0.21, ImageNet-1K V2 weights; official transforms; pre-classifier global average pool, 2,048 dimensions.

Images are decoded as RGB and processed with the encoder's documented transform. This initial center-crop/resizing policy can omit edge tissue; tiling/padding are future ablations. Encoding is in float32 with TF32 disabled. The encoder is externally pretrained, frozen, and not fitted to this cohort. Consequently all split features can be cached without using evaluation labels for adaptation.

Image feature caches are sharded by encoder and magnification, with image-path ordering, checkpoint/manifest/source revisions, a feature-file checksum, transform description, GPU, precision, throughput, and memory metadata. Every shard must exactly cover its manifest images before case pooling.

## Case representations and baselines

Mean-pool all field features within each case and magnification. This gives one tensor with shape `(cases, 4, feature_dimension)`. The same full-field policy is used by all three methods. Fit per-scale/per-channel normalization on the 46 fitting cases only; replace near-zero scale factors by one.

Evaluate all 15 nonempty observed-scale subsets:

- **Mean fusion:** average observed standardized scale features and append four availability indicators.
- **Masked concatenation:** zero absent standardized scale vectors, concatenate four vectors, and append availability indicators. Missing placeholder values cannot contribute to this representation.
- **Late fusion:** fit one multiclass classifier per scale and average the observed scales' subtype probability vectors.

The classifiers are class-balanced logistic regressions. Mean and concatenation are fitted on all 15 subsets, with each derived row weighted 1/15 so each original case retains total weight one. Subsets are augmentations, not independent patients. Late fusion fits one classifier per scale on the original fitting cases.

Select C from the fixed grid `[0.01, 0.1, 1, 10]` using equal-subset mean validation macro F1. Ties select the first/smaller C. There is no outer-test-driven parameter, epoch, or method selection. Each run stores validation score vectors and linear coefficients/normalization without pickle.

## Metrics and outputs

For every subset report eight-class macro F1, accuracy, and balanced accuracy. The selection criterion is the equal-weight average of the 15 macro F1 values; also retain the worst subset and the full-view result. Use the fixed label order from `magset.data.CLASS_NAMES`.

The first matrix has two encoders × three methods × five trials = 30 validation configurations, each with seed 812. It is a controlled baseline matrix, not thirty independent clinical samples. Repeated seeds, calibration, intervals, strict grouping sensitivity, learned fusion, matched-capacity ablations, and a second dataset are not completed by this pilot.

The CLI requires an explicit `--evaluate-test` to produce outer scores; scheduler pilot scripts do not pass it. Keep outer evaluation disabled until the final protocol/method choices are frozen. Future reported test intervals must account for repeated cases across provider trials rather than treating trial estimates as independent folds.

## Parallel execution

A one-GPU CUDA/real-feature smoke test gates an eight-task feature array, capped at four GPU workers. A separate four-shard CPU job verifies every PNG. The feature array depends on both pilot and decode success. The 30 baseline jobs use cached features on CPUs, capped at six workers. Dependency failure blocks downstream work rather than producing partial benchmark summaries.
