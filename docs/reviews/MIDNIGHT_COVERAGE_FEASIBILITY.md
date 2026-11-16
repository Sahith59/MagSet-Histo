# Frozen Midnight three-crop coverage assessment

November 16, 2026. **Recommend one fixed three-horizontal-crop comparison if improvement work continues after the planned zero-forward ensemble.** The current transform retains about 65–66% of the resized field width. A left/center/right feature mean would expose the omitted periphery while preserving the existing resize scale and frozen encoder. This is a coverage hypothesis with a small bounded extraction cost. Evidence currently supports testing it, with no numerical prediction of its accuracy.

This review inspected the pinned Torchvision 0.21.0 implementation and ran synthetic PIL geometry and pixel-transform checks. It constructed no encoder and performed zero neural forwards, fits, model downloads or jobs. The LoRA context below uses the complete root-reviewed public aggregate.

## Exact transform

Read the original checked image as RGB. Apply the same PIL-backed `v2.Resize(224, interpolation=BILINEAR, antialias=True)` used by the accepted Midnight cache. Torchvision sets the short edge to 224 and computes the other edge by integer truncation. The audited release contains 7,835 images at 700×460 and 74 at 700×456. The exact resized shapes and exclusive-right crop boxes are below. [Resize documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.v2.Resize.html) · [Pinned resize source](https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/transforms/functional.py) · [Image audit](../DATA_AUDIT.md)

| Original width×height | Resized width×height | Left crop | Center crop | Right crop |
| --- | --- | --- | --- | --- |
| 700×460 | 340×224 | `(0,0,224,224)` | `(58,0,282,224)` | `(116,0,340,224)` |
| 700×456 | 343×224 | `(0,0,224,224)` | `(60,0,284,224)` | `(119,0,343,224)` |

Use offsets `[0,int(round((resized_width-224)/2)),resized_width-224]` and top zero. The 343-pixel width gives center offset 60 because Python rounds 59.5 to the even integer. The common 460-pixel height gives width 340. Synthetic checks verified bitwise equality between the explicit middle crop's normalized tensor and the accepted official center-crop transform for both source heights. [Pinned crop-anchor source](https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/transforms/v2/functional/_geometry.py) · [CenterCrop documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.v2.CenterCrop.html)

Each resulting image is exactly 224×224. Apply the existing `ToTensor()` and channel mean/std `.5`, with the same PIL/Torchvision versions and operator order. Use exactly three distinct horizontal boxes. The resized height already equals the crop height, so top/bottom corner pairs would coincide. The three windows cover every resized column. Their overlap gives central columns more representation than peripheral columns, so the average is not a uniform spatial integration.

Encode each crop with the original frozen Midnight checkpoint and the existing CLS-plus-mean-patch 3,072-dimensional feature definition. For each original field, compute `(z_left+z_center+z_right)/3` in float64, then save one float32 vector. Use the accepted cached center vector exactly once. Store the new derivative cache under its own provenance identity. The unchanged original field path remains the grouping and weighting unit.

## Scientific rationale and counterevidence

The current center crop omits 34.12% / 34.69% of resized width for the two audited shapes. Broader tissue context could add subtype evidence when informative morphology lies outside that rectangle. Current mucinous-to-ductal and papillary-to-mucinous confusions make those labels relevant diagnostics, but the saved predictions establish no class-specific peripheral-location effect. Extra background or nondiagnostic tissue may instead dilute the existing subtype signal. Raw embedding averaging can also cancel useful crop variation before the nonlinear kernel calculation.

The earlier Phikon regional experiment is substantive counterevidence to assuming that extra crops improve performance. It changed eight-class F1 by −.005833 / +.005889 and failed its screen. That candidate combined the center feature with four raw-image quadrants at 50/50 weight, changing local physical scale and coverage under a fixed-C linear head. The proposed candidate preserves the current full-field resize scale and uses the current five-category Midnight KDE. Its outcome still needs measurement. [Regional actual-output review](BREAKHIS_MULTICROP_RESULTS_REVIEW.md)

The completed LoRA dual study scores 68.33% / 59.83% accuracy and .494406 / .375418 F1. Its fixed-C1 initialization scores 68.17% / 59.50%, so the measured accuracy changes from that starting point are +0.17 / +0.33 points. Eight of ten dual runs select the head-only epoch two, including all five arrangement-813 runs. Fitting accuracy reaches 99.13%. This combined initialization/augmentation/head/adapter recipe supplies limited evidence for additional feature fitting in this small cohort. Its absolute comparison with the validation-selected frozen heads also changes the initialization choice. LoRA training and evaluation still use the same center rectangle, leaving the peripheral-coverage hypothesis open. [Completed LoRA aggregate](../evidence/BREAKHIS_LORA_RESULTS.json)

## One canonical downstream comparison

Use the existing centered-cosine KDE recipe with the new one-vector-per-original-field cache. Recompute each role's support-case-balanced per-view center from its fitting cases. For fitting diagnostics, remove the full query case before centering and support construction. Keep float64 centered L2 normalization with epsilon `1e-12`, the same head-class/case/field density weights, uniform head priors, shared temperatures `[.01,.03,.1,.3]`, joint field probability composition, case routing at malignant probability ≥.5 and all fifteen view subsets.

Retain the original ten role arrangements, direct/dual arms, four candidate opportunities per arm and earliest maximum end-to-end macro-F1 selection. This gives eighty candidate pipelines, zero optimized classifier coefficients and the same similarity-matrix sizes as the original KDE. Compare with the fixed accepted single-center KDE and report all scores, class recalls and routing/typing diagnostics under the existing descriptive screen. Preserve the eight-case validation supports and identify the 31 distinct reused validation provider keys.

## Extraction, storage and validation

Only left/right descriptors need full extraction, giving **15,818 new crop forwards** plus separately counted center-parity probes. Batch eight must mean eight crop images. Retain the same frozen FP32 eager-attention execution and disabled TF32. Extrapolating from the accepted four-shard Midnight extraction gives roughly 8–10 minutes with two A40 workers, before queue and verification time. This is an estimate rather than a measured new runtime.

Uncompressed left/right float32 vectors require 194,371,584 bytes, about 185.37 MiB. The fused cache requires 97,185,792 bytes, about 92.68 MiB. Together they add approximately 278.05 MiB plus paths and metadata. Preserve the raw left/right descriptors so an independent cache audit can recompute every mean using the pinned original center archives.

Before reuse, bind all four original center NPZ/JSON hashes, raw manifest/image hashes, encoder checkpoint/configuration hashes, transforms, versions, crop order and exact boxes. A declared fitting-image probe should reproduce the existing center embedding with the established `atol=rtol=1e-4` allclose rule. The same probe should verify crop-batch/repeat consistency. Record actual errors, GPU context, memory and timing. Each final shard must cover its exact original paths once, with both expected input heights and finite 3,072-dimensional vectors. Reopen archives and independently verify the float64-to-float32 averaging arithmetic before classifier use.

Private `.local/midnight-coverage-feasibility/geometry.json` records the synthetic offsets, full-column coverage and bitwise middle-tensor parity for both heights. No raw cohort image or model execution was used for those checks.
