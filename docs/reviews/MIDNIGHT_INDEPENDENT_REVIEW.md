# Midnight extraction: independent source review

November 15, 2026. Read-only review of `magset/midnight_features.py`, `scripts/prepare_midnight.py`, `configs/midnight-12k-v1.json` and `tests/test_midnight_features.py`, cross-checked with the existing [primary-source audit](IMPROVEMENT_PRIMARY_RESEARCH.md) and [registered extraction specification](../MIDNIGHT_EXTRACTION_SPEC.md). No technical blocker was found for the planned immutable-source preparation and fitting-free technical smoke. This review did not download artifacts, construct the actual Midnight model, run its encoder, submit jobs or fit a classifier.

Reviewed bytes:

| File | SHA256 |
| --- | --- |
| Extractor | `cffc7891b617fa2bbac15f4972d925dfab0d76ad6a838dbbb4235cf5699ee80c` |
| Preparation script | `f7ae5703d708fa277969fdac05afbc284c8dccab2ad1c6e00e7b7d715c7d65f5` |
| Configuration | `71da2f7ee8bbfb27876a212f7a2c6dd765ba23f4ca86d7d80e4b2982e7b017ce` |
| Fixture tests | `27375a343aaf95d24c73ed7620f07a2c68e8b33d115f56a1d4283cf8d614094a` |

## Representation and loading

The implementation pins the public Midnight-12k revision, checkpoint byte count/SHA256 and official model configuration/card hashes. It constructs the standard Transformers `Dinov2Model` with the pinned 40-layer, hidden-width-1536, patch-14, SwiGLU architecture and eager attention. Meta-device construction avoids allocating and initializing a second full model. Strict `assign=True` loading requires FP32 state, matching keys/shapes and materialized FP32 parameters. Evaluation mode and disabled parameter gradients freeze the encoder. No remote model code is executed.

`Fields` supplies RGB images. The transform matches the recorded official example: bilinear antialiased shorter-side resize to 224, central 224 crop, tensor conversion and channel mean/std 0.5. It does not reuse Phikon preprocessing or infer a 518-pixel inference size from the positional-embedding configuration. For a 224/14 input, the output check requires 257 tokens of width 1536. Concatenating the final CLS vector and the mean of the 256 final patch vectors gives a finite FP32 vector of width 3072.

Execution disables AMP through ordinary FP32 inference without an autocast context, disables TF32 and freezes all weights. The recorded runtime is checked against the configured existing cluster versions, including Transformers 4.51.3, Torch 2.6.0+cu124 and torchvision 0.21.0+cu124. Actual compatibility with the billion-parameter checkpoint remains the technical smoke's responsibility; the tiny state-loading fixture is not a substitute.

## Preparation, parity and cache acceptance

Preparation requests immutable public URLs without credentials. Published artifacts are reused only after exact length/hash verification. Resumption validates HTTP 206 range start, total length and body length; a server returning 200 restarts the partial instead of appending a complete response. Oversized bodies, wrong ranges, incomplete final files and SHA mismatches do not publish a checkpoint. Retries are bounded. The preparation receipt records artifact, configuration, source/script and scheduler provenance. No model is built during preparation.

Extraction checks a repeated first batch and first-image single-versus-batch agreement. The separate four-image smoke also destroys and reconstructs the model before checking agreement. Both parity tolerances are fixed in source at absolute/relative `1e-4` before fitting or classification outcomes. Every comparison requires matching shape and finite values. No extra unregistered numerical admission threshold is proposed by this review.

The full manifest hash and 7,909-image release count are checked before magnification selection. Relative paths are confined beneath the image root and loader order must exactly match selected manifest order. Each output contains only original relative paths and finite FP32 3072-dimensional descriptors. Publication refuses existing targets, uses same-filesystem no-overwrite links, then reloads the NPZ and checks exact values/order. Metadata binds feature bytes, encoder/revision/weights, transform, configuration, source file, implementation, environment, magnification, smoke status and parity receipts. Existing archives are reused only after the implemented provenance and coverage checks.

The extractor never consumes subtype/binary labels for model fitting or selection. Its manifest reader can see label columns, but forward inference uses only image paths and pixels. It does not produce classification outcomes. Full feature coverage therefore does not change the fitting/validation/excluded role contract of later classifiers. Full-cache audit and a cache-bound classifier registration remain necessary before any fitting.

## Immutable binding and remaining validation

The Python CLI checks that the supplied source identifier is 40 hexadecimal characters and records the actual source-file hash; it does not independently prove that identifier names the executing checkout. The planned wrapper assertion against `.source-sha`, with the immutable snapshot/archive receipts, is therefore an essential part of the launch provenance. The parent has explicitly retained that responsibility. This is not a request for another approval or a new runtime policy.

All ten existing tiny fixtures passed locally, including validated download resumption, wrong-hash/range rejection, transform output, strict tiny meta assignment, token concatenation, parity rejection and cache roundtrip/coverage/overwrite guards. This local run used Python 3.13, Torch 2.6.0, torchvision 0.21.0, NumPy 2.3.0 and Pillow 11.3.0. An offline attempt to use the exact pinned test environment could not resolve a locally uncached torchvision wheel and transferred nothing. No full-model loading claim follows from these local checks; cluster runtime/version validation and the actual fitting-free smoke remain outstanding.

This source review supports proceeding to those already planned checks. It establishes no feature-extraction completion, classifier improvement, 94–95% result or clinical generalization claim.
