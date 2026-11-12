# Methods draft: implemented baseline study

Working text for revision with Professor Dong. It describes the current implementation, not a finalized novel method or completed generalization study. Final test results are intentionally absent.

## Task and data unit

We study eight-subtype classification from sets of breast histopathology fields at 40×, 100×, 200×, and 400×. For case i and magnification m, X_im denotes its unordered set of image fields. The prediction model receives any nonempty subset S of the four magnifications and returns eight class probabilities. We evaluate all 15 possible S while retaining the same eligible cases. This is controlled, artificial withholding of available magnifications; the present cohort does not establish the clinical frequency of naturally missing views.

The official BreakHis archive contains 7,909 images and 82 subtype-qualified provider cases. We retain these provider identities without claiming independently reverified clinical patient identity. The provider's five trial files define repeated 54/28 train/test holdouts. Within each training partition, a prespecified seed selects one case per subtype for inner validation, leaving 46 fitting and eight validation cases. Numeric identifier stems can overlap across provider assignments and require a separate grouping sensitivity analysis. The five trials are overlapping holdouts, not independent cross-validation folds.

## Image representation

Frozen externally pretrained Phikon-v2 and ResNet-50 encoders transform each image using their fixed model-specific preprocessing. Phikon-v2 provides the final CLS embedding (1,024 dimensions); ResNet-50 provides the pre-classifier pooled representation (2,048 dimensions). Cache provenance includes model revision, image manifest checksum, preprocessing, and precision. No cohort labels adapt the encoders in the current study. Potential overlap between encoder pretraining and later evaluation datasets must be audited before interpreting external-dataset results.

## Aggregation controls

The initial controls average embeddings within each case/magnification, then use masked feature averaging, masked concatenation, or probability-level averaging with class-balanced logistic regressions. The second round compares field classification followed by probability aggregation with a normalization-matched early-feature-mean control. Field fitting weights give equal total influence to cases within subtype regardless of the number of fields.

Learned controls project field embeddings into 128 dimensions and aggregate within each scale using a mean or standard gated-attention MIL operator, following the established [attention MIL formulation](https://proceedings.mlr.press/v80/ilse18a.html). A scale identity embedding is added before view fusion. Available-view fusion uses a mean, a contextual masked gate, or one self-attention block with four heads and mean aggregation. The attention control follows the general [set-learning principle](https://proceedings.mlr.press/v97/lee19d.html); it is not an exact implementation of the full published Set Transformer architecture. Missing-view gating already appears in [MagViT](https://arxiv.org/html/2608.16959v1), so these components are not claimed as novel.

For learned controls, per-scale normalization uses equal-case first and second field moments estimated only from fitting cases. Padded field contents are removed before learned projection. Unavailable scales are excluded from keys, values, gates, and aggregation. Reordering fields or scale tokens with their identity labels preserves predictions. No registered cross-scale field pairing is inferred from filename sequence numbers.

## Fitting and model selection

Neural fitting minimizes class-balanced cross entropy averaged equally over all 15 availability subsets. The fitting objective treats each case as one unit; subset augmentation does not create independent samples. We fix 100 epochs and evaluate only the final epoch for each of four prespecified AdamW configurations, crossing learning rates {0.0003, 0.001} with weight decays {0.01, 0.1}. The case batch size is eight, dropout is 0.2, and gradient norm is capped at one. Select mean validation macro F1 across subsets; break ties by grid order. A single training seed is used for this exploratory screen; split and training-seed sensitivity remain required.

For linear controls, select regularization C from {0.01, 0.1, 1, 10} using the same inner-validation objective. The neural comparison shares its complete selection budget across architectures. Parameters, optimization, and normalization differ between rounds; no cross-round difference is attributed to a single architectural component without matched controls.

## Evaluation and reproducibility

The primary endpoint is equal-subset mean eight-class macro F1. Retain per-subset macro F1, full-view and worst-subset scores, accuracy, and balanced accuracy. Planned final analyses also include subtype recall, calibration, grouping sensitivity, and uncertainty accounting for repeated cases. Inner-validation scores guide development and are not final generalization estimates.

All runs save score vectors, class order, case assignments, dataset/model/source/configuration hashes, selected parameters, and timing. Learned checkpoints retain model/optimizer and RNG state. Reloaded models must reproduce saved probabilities, and an independent collector recomputes metrics from every saved validation record. Outer-test scoring is unavailable in the new pilot CLIs. Final outer evaluation will follow a separately frozen method/protocol.
