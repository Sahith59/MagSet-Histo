# Frozen Midnight native-field feasibility

November 16, 2026. **The pinned DINOv2 API supports the proposed rectangular input.** One frozen 448×700 comparison is technically reasonable if the parent chooses further improvement work after the completed coverage study. It tests fine spatial detail and joint whole-field context, with materially more compute than the 224-pixel route. No implementation, scheduler submission or model construction was performed for this assessment.

The actual cached Midnight configuration was read through a small read-only metadata query and its SHA256 rechecked as `0362e014a3f3b1ad2f2faf115d21cbc05ed48b3a6114bdb4ddec897bc83cc0fd`. It specifies hidden size 1536, 40 layers, 24 attention heads, patch size 14, FP32 and configured image size 518. Keep that original configuration for strict checkpoint loading. [Pinned official configuration](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/config.json)

## Input and API

Read RGB and apply `CenterCrop((448,700))` directly to the original field. The 700×460 image uses box `(0,6,700,454)` and retains 97.39% of its pixels. The 700×456 image uses `(0,4,700,452)` and retains 98.25%. Width stays 700 and no image resampling occurs. Then apply the established tensor conversion and channel normalization with mean/std `.5`.

Call `Dinov2Model(pixel_values=..., output_attentions=False, output_hidden_states=False, return_dict=True)` with tensor shape `[B,3,448,700]`. The 4.51.3 patch embedding uses a stride-14 convolution and checks channel count. Its rectangular output is 32×50 patches. Embeddings automatically interpolate positional encodings for the input height and width. There is no additional interpolation flag in this forward signature. [Versioned API](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/dinov2) · [Official 4.51.3 implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/dinov2/modeling_dinov2.py)

The learned positional grid is 37×37 from configured size 518. The implementation preserves the CLS positional vector and interpolates patch positions to 32×50 using bicubic FP32 arithmetic with `align_corners=False`. The resulting last hidden state should be `[B,1601,1536]`. A new feature helper must verify that shape, dtype and finiteness, then concatenate CLS with the mean of all 1,600 final patch tokens to produce `[B,3072]`. The existing helper's 257-token assertion belongs to the accepted 224-pixel extractor and should be preserved.

## A40 cost estimate

The actual four-image 224 smoke reported 4,690,073,600 allocated GPU bytes, approximately 4.37 GiB. The fixed weights occupy about 4.23 GiB. At native batch four, one eager attention matrix contains `4*24*1601²` FP32 entries, approximately 0.917 GiB. Two live score/probability arrays require about 1.83 GiB before other activations and workspace. A planning estimate is roughly **7–10 GiB allocated**, to be measured on A40. Inference with attention/hidden-state outputs disabled avoids retaining every layer's tensors.

Token count grows by 6.23× and attention matrix size by 38.81×. A dense-plus-attention arithmetic estimate for this architecture gives approximately 7.11× forward work per image. Scaling the accepted two-A40 four-shard extraction wall time of about 4.5 minutes gives roughly **32 minutes** before changes in efficiency and overhead. Plan around **30–60 minutes on two A40 workers** and replace the estimate with a batch-four smoke measurement. The older 5.89-second smoke includes repeat/reconstruction work and cannot supply production throughput by dividing its four images by that duration.

## Hypothesis and bounded comparison

The three-crop result reaches 75.50% / 76.83% mean-subset dual accuracy, with F1 .592510 / .579947 and 82.50% all-four-view accuracy on both arrangements. Its resized crops still summarize independently encoded low-resolution windows. Native input preserves finer source pixels and allows attention across the retained field in one pass. The mean-patch descriptor will nevertheless average 1,600 tokens, and broader detail can include nuisance tissue.

Rectangular positional interpolation is supported software behavior. Its transfer quality for this checkpoint and dataset remains a hypothesis. The original official benchmark transform uses 224 pixels. Earlier regional-feature averaging failed its screen, and LoRA offered little improvement over its fixed-C initialization. These outcomes support a single measured test rather than an accuracy forecast.

If pursued, use the same ten original roles, one field descriptor, centered-cosine KDE, four shared temperatures, leave-case-out fitting diagnostics and all fifteen view subsets. Validation selects complete pipelines under the current endpoint. A technical probe should check native shape, repeat/single-batch/fresh-load parity and measured memory/runtime. The feature cache needs its own transform identity and full path/hash audit. Reused validation, sparse rare-type cases and unresolved clinical grouping continue to limit generalization claims.
