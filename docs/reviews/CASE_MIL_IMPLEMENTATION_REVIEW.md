# Case-MIL trainer and collector review

November 16, 2026. **No invalidating implementation defect was found in the reviewed trainer and collector.** They are ready for the separately required fitting-only A40 technical checks after source freeze. This is implementation acceptance, not evidence that attention improves accuracy or reaches 95%.

The reviewer authored `case_mil.py`; this review's independent target is the other author's [trainer](../../magset/case_mil_five.py) and [collector](../../magset/case_mil_collect.py). The [registered recipe](../BREAKHIS_NEXT_ROUND_SPEC.md) and [configuration](../../configs/breakhis-case-mil-v1.json) determine the intended behavior.

Reviewed SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| Trainer |`07e0df5cc984112aab8f7b4a15ca5609a7d20ca5ffd3673e32ce2907e8352ff0`|
| Collector |`70f0c790ec4c6c2fc2acea50008ec64dcff4c065b81feb3682ca0a2bf2f192f2`|
| Configuration |`464c2890dfe4537a2dfd4458163f05071465660242e6056c43c09684771f21ce`|

## Training, roles and randomness

The loader delegates to the accepted F input loader, retaining all fields, provider memberships, original-eight-label splitting and final-five mapping. It binds the F configuration, audit/cache lineage and new source files. Fitting-only case-balanced field moments are reproduced before training, saved in each checkpoint and required exactly by the collector. Normalization uses FP32 arithmetic on the accepted FP32 features for both training and inference; canonical scoring then casts those normalized inputs and saved parameters to FP64.

Each production epoch visits every one of the 46 fitting cases once in a seeded permutation. Four-case batches include the last two cases, so there are 12 updates per epoch and 1,200 per trajectory. The trainer passes the actual fitting class counts into the core loss. The direct factor is 46/(5*n_class); the two-step loss is half the sum of binary 46/(2*n_broad) BCE and true-malignant 46/(4*n_type) CE. Both average all 15 subset losses equally and retain denominator four on the last batch. The malignant factor is intentionally 46, matching the registration, rather than the expert population 35. Benign-only batches do not supply expert gradients; optimizer-step verification accounts for those skipped expert updates.

Learning rates are assigned by the registered epoch cosine, beginning at .0003 and ending at .00003 at epoch 100. Validation does not alter the schedule. The seed contains the arrangement and trial, but neither pooling mode nor arm. The trainer resets Torch/CUDA randomness after model initialization. Shared projection/head initialization is compared across pooling modes. Candidate inference reconstructs models inside a CPU RNG fork and disables dropout, preserving the subsequent training stream. The synthetic repeated-trajectory fixture reproduces final parameters and CPU RNG state exactly.

## Scoring and complete artifact verification

Whole-case outputs are case logits, rather than means of the previous field-classifier probabilities. Two-step probabilities compose the binary sigmoid and conditional four-type softmax into five leaves. Hard routing uses R >= .5, while the direct arm uses the five-way argmax. Both retain original-eight truth and the canonical final-five mapping. There are no excluded classifier predictions.

All four registered checkpoints are saved and scored on fitting and validation cases. Selection is the earliest maximum of mean 15-subset validation macro F1. The collector independently implements affine/ReLU projection, gated attention, per-view pooling, equal available-view means, flags and both heads in NumPy FP64. It checks every checkpoint's exact parameter whitelist, FP32 saved types, original normalizers, context and hashes, then replays every probability, hard decision and metric against trusted roles.

The collector rejects missing or extra trajectory/checkpoint/score artifacts and verifies the declared 40 trajectories, 160 checkpoints, 320 prediction archives and 129,600 dependent case/subset records. It checks epoch orders, rates, optimizer moments/update counts, final-candidate identity, paired initial states and earliest checkpoint choice. Reference files F/C/P are hash-bound; F feature lineage and unchanged task/data methods are checked. Fitting results are explicitly labeled in-sample neural diagnostics, separate from KDE leave-case-out results. Public comparison output excludes provider identifiers.

## Independent checks and remaining limits

My independent scalar/NumPy oracle passed **505 assertions**, covering both pools/arms, all subsets, normalization, head logits, joint composition, class mapping, deterministic orders and LR endpoints. Its maximum discrepancy against trainer inference and collector replay was **8.88e-16**. Another **21 assertions** accepted valid synthetic states and rejected altered state schemas, nonfinite/different-dtype parameters, normalizers, context/epochs, duplicate/missing/changed score rows, AdamW step counts and training histories.

The combined pinned suite passed **23 tests in 9.214 seconds**, including the other author's synthetic trajectory and rehashed-corruption fixtures. Its forty-trajectory counting fixture mocks per-run verification; it does not represent forty real fits. Actual CUDA timing, memory use, deterministic-kernel operation and cache-scale execution remain for the fitting-only technical runs. CPU fixture runtime checks are explicitly mocked, not presented as A40 execution. Deterministic CUDA also requires the wrapper's `CUBLAS_WORKSPACE_CONFIG` setting.

`last.pt` retains optimizer and RNG state, but the current executable refuses an existing output directory and provides no resume path. **Save/reload is tested; interrupted-run continuation is not implemented or claimed.** This is consistent with the registered requirements. No real cohort fitting, GPU jobs, source edits to the reviewed modules or commits occurred during this review. Synthetic numerical checks do not resolve sparse rare-class support, reused validation, clinical identity uncertainty or the substantial overfitting risk of approximately 0.4 million parameters for 46 fitting cases.
