# Conditional final GPU follow-up: Midnight LoRA

November 16, 2026. Design audit only. No package changes, weights downloads, model construction, jobs, fitting or current KDE-result inspection. Root decides after the complete KDE study. Current frozen Midnight accuracy is about 71–74%, with strong binary routing and substantially weaker malignant typing; neither this proposal nor the LoRA literature predicts 94–95% here.

**Assessment: technically feasible on the established A40 runtime, but scientifically high risk with two lobular and three papillary fitting cases.** If KDE fails, one fixed, bounded adaptation comparison is defensible within the user's requested improvement work. Do not turn it into a rank/layer/augmentation sweep. Keep the proposed last-four-layer Q/V scope; no larger method is needed. Correct the smoke so it actually exercises LoRA, preserve the original fixed-C head input contract, and add a real adapter reload probe. Those are necessary implementation details, not requests for new approval.

## Exact official module targets and LoRA behavior

The [official Transformers 4.51.3 DINOv2 source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/dinov2/modeling_dinov2.py) defines separate query/key/value linear projections. For the existing standard `Dinov2Model`, target exactly these eight modules:

```text
encoder.layer.{36,37,38,39}.attention.attention.query
encoder.layer.{36,37,38,39}.attention.attention.value
```

Assert these exact eight 1536→1536 modules exist before injection; do not match every module containing a short suffix such as `value`. Strict-load the complete official checkpoint first, then attach adapters. Leave original projection weights/biases, key/output projections, MLPs, normalizers, embeddings and all other backbone parameters frozen. Keep the established FP32 eager-attention path and disabled TF32.

[LoRA's original paper](https://arxiv.org/abs/2106.09685v2) supplies the standard frozen-weight, low-rank-update construction; it does not validate this pathology recipe. The [official Microsoft implementation](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py) uses a random A matrix, initially zero B matrix, dropout on the adapter input, and alpha/r scaling. With rank 4 and alpha 8, the residual scale is 2. For each projection, A has shape 4×1536 and B has shape 1536×4; the update is `2 * B(A(dropout(x)))`. Zero B makes the initial model exactly the pretrained function, subject to numerical replay tolerance.

Eight projections × two matrices × 1536 × 4 gives **98,304 adapter parameters**. Existing field heads have 3076 inputs: 3072 features plus four one-hot magnification flags. A five-output affine head has **15,385 parameters**. A scalar binary affine head plus four-output expert has the same count. Total trainable parameters per arm are therefore **113,689**, not billions. Assert the complete trainable-name/shape list rather than relying only on this count.

The official library defaults to merging updates into base weights when switching to evaluation. **Disable merging** (`merge_weights=False` or an equivalent unmerged wrapper). Otherwise evaluation mutates the nominally frozen base tensors, complicates state proofs and risks double application on reload. No new PEFT/loralib installation is necessary for this small standard operation; an audited unmerged wrapper can express it using existing Torch.

## Preserve the head initialization contract

Use each role/arm's **fixed C=1 field checkpoint, candidate index 2** from the accepted frozen comparison, with its exact source/cache/config hashes. Never initialize from the validation-selected candidate. Copy both coefficients and intercepts in canonical class order.

Those saved heads expect the original **fit-only, per-view mean/std normalization**, followed by the one-hot magnification flags. Keep those original normalizers fixed while adapting the raw 3072-dimensional features. Substituting the KDE center/L2 transform, omitting flags, recomputing normalization on validation, or silently updating moments during training breaks the stated initialization. Before any optimizer step and without augmentation, compare a fixed fitting-image probe against the accepted C=1 frozen model. Record feature/probability error, tolerances and the exact copied head checkpoint hash.

During training, activate adapter dropout .1; disable it during all evaluation and saved-feature extraction. Ensure head-only warmup genuinely disables A/B gradients, then explicitly enables only those A/B parameters and heads. In the first LoRA step, zero-initialized B can receive gradients while A initially has zero gradient; later steps must show A can change. This is expected behavior, not evidence that a missing A gradient on the first step is a bug.

## Objective, augmentation and selection

The proposed sample budget is coherent: two fields per magnification, eight fields per case, every one of 46 fitting cases once per epoch, case batches of four. Match field selection, case order and augmentation seeds between direct/dual arms. Use the same four magnifications and inherited case grouping. Inference uses all original fields, and the 15 view subsets reuse their field outputs rather than requiring 15 backbone executions.

For case-balanced losses, average the eight sampled field losses within each case. Direct-five class weight is `46/(5*n_class)`; binary weight is `46/(2*n_binary_class)`. For true-malignant expert fields only, use `46/(4*n_malignant_type)` with the **all-case batch denominator**; a benign-only batch has zero expert term. This prevents accidentally diluting expert loss by a factor 35/46 or averaging by a random number of malignant cases. The proposed dual objective is half the sum of the binary and expert terms. Specify how the final two-case minibatch is scaled before fitting, and record realized case/field/class masses.

Keep the proposed two head-only epochs plus 18 adapter/head epochs, AdamW LR and decay 1e-4, clip 1, cosine schedule, and checkpoint epochs 2/6/12/20 if executing this single recipe. Pin whether cosine decay covers all 20 epochs or begins at adapter unfreezing. Do not add rank, learning-rate, threshold or separate-head checkpoint searches. Report when epoch 2 is selected: it is a head-only augmented checkpoint, so an improvement there cannot be attributed to learned LoRA weights.

After the official resize/center crop, use a fixed orientation policy (uniform multiples of 90 degrees with flips, or its equivalent uniform eight-element orientation group). Make color jitter explicit: **brightness/contrast/saturation .1, hue 0** is a conservative interpretation. Hue .1 is a substantial color rotation and should not be inferred from the phrase “jitter .1.” This augmentation is a stain-robustness hypothesis, not stain normalization or proven biological invariance. Validation and replay always use the original deterministic transform.

## Cost and verification limits

Each production run needs **7,360 sampled training image forwards** (`20*46*8`) and about **240 optimizer updates** (`20*ceil(46/4)`). Across 20 runs that is **147,200 training forwards**, plus backward computation through the final four blocks. Low adapter parameter count removes large optimizer states; it does not remove 40-layer forward computation or the last four blocks' activation/gradient cost.

Four full validation passes and a selected full-fitting pass add considerable work. Using only the dataset-wide average image count per case gives roughly **300,000 total forwards** across the study; actual role-specific field counts determine the true total. This is an arithmetic estimate, not a benchmark. A 90-minute task cap requires roughly three total image forwards per second after allowing for the evaluation workload, plus backward cost. Validate wall time and peak memory in the smoke before promising this ceiling. Batch 32 may fit, but the successful batch-8 inference run does not establish training memory. If needed, predeclare four eight-field microbatches with gradient accumulation to preserve a four-case optimizer batch.

Twenty 90-minute allocations imply a **30 GPU-hour ceiling**, even with up to eight concurrent A40 tasks. Concurrency reduces elapsed queue/runtime but does not reduce work. Avoid a full-encoder finetune or distributed training; neither is needed for these small adapters. Training weights, gradients and Adam moments for 113,689 FP32 parameters occupy only about 1.7 MiB combined; the frozen 4.23-GiB backbone and activations dominate GPU memory.

**The proposed two-epoch smoke is insufficient:** with two head-only warmup epochs, it never activates LoRA. Use **three epochs × two fitting-only batches**, preserving the two-epoch warmup and exercising the adapter phase in epoch 3, or explicitly register an equivalent two-phase technical smoke. Verify zero-update initialization parity, nonzero adapter updates, finite losses/gradients, unchanged frozen tensors, peak memory and actual base+adapter+head reconstruction.

Retain all four candidate adapter/head states: they are tiny. Save resumable optimizer/scheduler/RNG state for selected/last checkpoints, with the external base model hash and exact module whitelist. A selected fixed fitting-image probe must reproduce archived adapted features/probabilities after destroying and reconstructing the full base+adapter model. Four candidate validation feature/probability archives allow a CPU collector to replay pooling, routing, metric calculation and selection. **CPU replay of cached features does not independently re-execute the adapted image encoder**; state hashes and a producer-recorded reload guard must be described with that limit. Do not claim an independent raw-image adaptation reproduction from arithmetic replay alone.

## Generalization and final decision

There are still only two independent lobular and three papillary fitting provider cases. Around 114,000 trainable parameters plus repeated fields and augmentations can learn provider/stain signatures; rank four is no assurance against that. Field labels are inherited case labels, while many fields can contain nondiagnostic tissue. Strong binary routing can coexist with persistent subtype errors. Earlier supervised/adaptation failures are substantive counterevidence to automatic optimism.

Keep all 20 matched results and the registered four-checkpoint selection rule, including negative outcomes. An improvement would apply to this reused provider-case development protocol and the combined initialization/augmentation/adapter recipe; without extra matched controls, it cannot identify LoRA alone as the cause or establish clinical generalization. If this is executed, treat it as the final bounded GPU study and preserve enough time to verify and finish the five-category project deliverable.
