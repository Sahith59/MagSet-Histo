# One bounded Midnight adapter comparison

Registered November16,2026, before fitting. Goal: improve the five-category BreakHis capstone model, especially malignant subtype recognition. Current frozen Midnight case-dual accuracy is72.0%/71.3%; the subsequently completed similarity dual-step improves to75.67%/75.00%. Both are pinned references before adapter fitting. This standard LoRA experiment is a candidate improvement, without a94–95% guarantee or novelty claim. [Feasibility review](reviews/MIDNIGHT_LORA_FEASIBILITY.md) · [Fixed configuration](../configs/breakhis-midnight-lora-v1.json).

## Same task and cases

Only BreakHis images/labels are fitted/evaluated. Final labels are benign plus ductal/lobular/mucinous/papillary carcinoma. The direct-five control and true binary→malignant-four pipeline each use all ten existing seed/trial roles:46 fitting,8 validation,28 excluded. Original eight-label membership is preserved before pooling; all related fields and four magnifications remain with each provider key. The35 true-malignant fitting cases train the expert. There are only two lobular and three papillary fitting keys. Clinical identity remains unresolved and validation is repeatedly reused; no globally fresh confirmation cohort is created.

## Initialization and architecture

Load the exact official Midnight-12k FP32 checkpoint already pinned and cached. Its40-layer DINOv2 ViT-g/14 emits1536CLS plus1536mean-patch features. The input remains deterministic resize-shorter224/center-crop224/RGB/mean-std.5 at evaluation.

Start heads from the **fixed C=1 field candidate, index2**, for the same role and arm in the completed frozen Midnight comparison. All20 initial model hashes are pinned in the new configuration. These are fixed candidates, not validation-selected initializations. Preserve their fitting-only four-view mean/std,3072 standardized features and four one-hot magnification indicators.

Attach unmerged rank4,alpha8,dropout.1 LoRA branches to query/value projections of layers36–39 only. Freeze every original parameter and buffer; no eval-time weight merging. Eight adapter targets have98,304 parameters. Either direct-five or scalar-binary/four-way heads have15,385 parameters, for113,689 trainable parameters per arm. No full-model optimizer or new dependency installation is needed.

## One training recipe

Every fitting case is visited once per epoch. Uniformly sample two fields from each of its four views: eight images per case. Process one case at a time and accumulate four cases before an optimizer update. Divide each weighted case loss by the fixed denominator4, including the final two-case batch. This gives each visited case the same nominal loss mass.

Direct weights are46/(5*class_count); binary weights46/(2*broad_count). The expert uses true-malignant cases only, weight46/(4*type_count), with zero expert contribution for benign cases. Average sampled fields within each case. The dual loss is half the sum of binary and expert terms. Losses count the all-case batch denominator consistently.

Use20 epochs: two head-only warmup, then18 adapter-and-head epochs. AdamW learning rate1e-4, weight decay1e-4, betas.9/.999,epsilon1e-8; clip gradient norm1. Cosine decay covers all240 updates, with zero minimum rate. FP32 eager attention, TF32 disabled. Seed812+10000*split_seed+trial is common across arms. Case order, sampled fields and per-field CPU augmentation seeds are isolated from model/dropout randomness.

After resize/center crop, training uses uniform90-degree rotations plus independent horizontal flips, followed by brightness/contrast/saturation jitter.1 and hue0, then tensor conversion/normalization. Evaluation has no augmentation. This is a combined initialization/augmentation/adapter recipe; a gain alone cannot attribute causality to LoRA.

## Selection, artifacts and checks

Evaluate all original validation fields at epochs2/6/12/20. These are the four complete candidate opportunities per arm. Compose joint five-way probabilities at field level; average fields within each view and available views equally. Use malignant probability>=.5 for the binary routing decision, then expert argmax. Evaluate all15 nonempty view combinations. Select the earliest maximum end-to-end mean15-subset macro F1; do not tune a threshold or heads separately. Epoch2 is head-only, so a selected epoch2 cannot demonstrate learned-adapter benefit.

Save all four small adapter/head states, head NPZs, validation raw feature NPZs, predictions and metrics. Save selected full-fitting features/predictions and resumable last optimizer/scheduler/RNG state. The4.5GB pretrained base stays external. Record source/config/dataset/cache/initial checkpoint hashes, runtime, losses, gradient/parameter manifests, frozen-base digest and peak memory.

Before optimization, deterministic fitting probes must reproduce the fixed-C1 cached features and head outputs. At completion, destroy/rebuild the base and small selected state and reproduce fitting-probe features/probabilities within fixed absolute/relative1e-4 tolerances. Verify original frozen tensors are unchanged. The CPU collector checks the complete20-run/80-candidate matrix, saved states/head agreement, exact validation/fitting coverage, probabilities, hard labels, all metrics and candidate selection. CPU replay of adapted cached features does **not** independently re-execute the raw-image adapted encoder; disclose that limit.

## Technical check and resource budget

Run one fitting-only technical check per arm, at seed812/trial1: three epochs, two four-case batches per epoch. Two warmup epochs are followed by one adapter epoch, exercising B updates and subsequent A gradients. No validation or excluded predictions are made. Measure actual time/peak memory, finite losses/gradients, initial parity, frozen-state integrity, adapter change and reconstruction. Synthetic checks are implementation evidence, not accuracy results.

Production is20 independent A40 tasks, at most eight concurrent, four CPU cores/64GB host RAM/90minutes each, maximum30 allocated GPU-hours. Only launch on available permitted nodes; leave unrelated jobs alone. Use immutable source snapshots. Actual technical-check timing informs the ETA. This is one bounded GPU comparison, followed by verification and the capstone report, rather than a new sweep of architectures or splits.

Report raw accuracy, macro F1, balanced accuracy, all-view/worst-subset scores, per-type recall, router and conditional-typing metrics. Compare both arrangements against accepted frozen Midnight case and field references and retain negative outcomes. The registered+.02F1/nondecreasing-accuracy screen is descriptive evidence of consistency; it does not replace the measured results or establish independent generalization.
