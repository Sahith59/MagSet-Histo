# Independent actual A40 technical-smoke review

November 15, 2026. **Accept both actual fitting-only technical smokes and the conservative capacity admission for the registered 20-fit comparison. No technical admission blocker remains.** Production must retain the identical immutable neural source, configuration, pretrained weights, roles, A40 family, 50-epoch/600-update budget, four-candidate selection schedule and four-GPU concurrency cap. Production results and their complete-matrix review are still pending.

This is execution evidence for the [registered case-bag recipe](../BREAKHIS_CASEBAG_SPEC.md), following the [checkpoint source review](CASEBAG_CHECKPOINT_REVIEW.md). It provides no validation accuracy, 94–95% result, clinical-performance or novelty claim.

## Input audit and actual smoke scope

The independent metadata-only input reviewer passed **15,898 assertions**. It checked the embedded source/archive and implementation hashes, registered configuration, 7,909-image manifest, 82 qualified cases, all ten original-eight-label role assignments and supports, complete membership fingerprint, the selected 46/8/28 role assignment, dependency/runtime receipt and zero-work counters. No images were opened by that audit. The preparation receipt binds the pinned public weight bytes and strict six-group/1,280-feature model-state loading with no fitting or encoder forward.

Exactly two actual smokes were reviewed: direct and dual, each trial 1 / arrangement 812 on NVIDIA A40. Each runs a head-only epoch followed by one adaptation epoch, two batches/eight fitting cases per epoch. It restores the authoritative epoch-one model/optimizer/RNG once before continuing. Neither smoke selects a checkpoint or scores validation/excluded cases.

| Actual smoke | Optimizer updates | Training field forwards | Fitting reload-probe forwards | Validation / excluded forwards | Reported computation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 4 | 128 | 8 | 0 / 0 | 11.754 s |
| Dual | 4 | 128 | 8 | 0 / 0 | 30.839 s |

These computation times include setup and artifact work inside the runner; they are not total scheduler allocation costs. Maximum observed allocated GPU memory was 1,265,717,248 bytes and reserved memory was 1,516,240,896 bytes. No production memory bound is inferred solely from those observations.

## Independent artifact checks

A standalone reviewer using pinned Python 3.11/Torch 2.6.0 passed **2,512 assertions**, without importing production `magset` helpers, constructing a CNN, reading images or executing a model forward.

- The actual pretrained safetensors file has the registered SHA256 and 28,840,352-byte size. Its tensors were read directly. The reviewer independently reconstructed the tiny seeded linear initialization and checked all initial/final state fingerprints against the actual checkpoints.
- Each model has **5,865,129 parameters**, including a 5,858,704-parameter backbone and 6,425 head parameters. The selected adaptive group contains **5,433,748 parameters**; the same architecture/head parameter count is used in both arms.
- For each arm, **246 frozen pretrained state tensors are bitwise unchanged**, including all **177 BatchNorm running-state/counter tensors**. All **172 selected adaptive tensors changed** after adaptation; head state changed. The warmup history retains the complete initial backbone fingerprint.
- Actual state keys, shapes, dtypes and finite values match the bound pretrained/head schema. AdamW groups, order, rates, decay, betas, epsilon, moments and exact per-parameter update counts were checked: **174 optimizer parameter states for direct and 176 for dual**. Selected backbone parameters have two updates; supervised head parameters retain their four updates rather than reset at phase transition. Specialist updates follow true-malignant minibatch membership.
- The reviewer independently regenerated both epochs' case permutations, sampled fields and augmentation-key receipts from trusted fitting metadata. Receipts match the actual checkpoint/history and are identical across arms. Cases, batches, phase schedule and training-field counts are exact.
- Recorded permitted-group gradients are finite and nonzero when active; the adaptive group has zero gradient during warmup and positive gradient during adaptation. Logged combined/component losses are finite and arithmetically consistent. Checkpoint model, phase, history, initialization, RNG schemas and epoch-one reload receipt agree with the reports.
- Actual directories contain only the started context, authoritative last checkpoint and final smoke report. No candidate, fitting-diagnostic, validation or excluded prediction/cache archives were accepted. All saved forward counters show 128 training plus eight fitting-probe fields and zero other roles.
- Both actual runtime receipts identify A40, the pinned CUDA/dependency environment, Python 3.11.18, CUDA 12.4, cuDNN 90100 and CUBLAS workspace `:4096:8`. Source/configuration/data/roles/weights bindings match the accepted input audit.

Both runner reports record **0.0 maximum feature difference and 0.0 probability difference** for same-runtime selected-model reload. The immutable runner checks all 15 probe subsets, finite arrays and matching explicit decisions against the fixed 1e-6 tolerance.

**Replay scope:** the smoke reports retain maximum-error scalars rather than both raw probe tensors. This independent reviewer checked those finite actual scalars, checkpoint identities and the immutable all-subset guard; it did not independently recompute GPU forward differences. No additional real-image forwards were performed for this review. Production candidate/selected caches retain sufficient statistics for the later independent numerical replay.

## Runtime admission arithmetic

The independent runtime reviewer reconstructed all ten role-specific field counts from the manifest and original role assignments. Complete production work is **531,912 field forwards**: 368,000 training, 60,232 candidate-validation, 88,462 selected-fitting, 15,058 selected-validation replay and 160 fixed reload probes. The largest complete fit has **27,874 fields**.

The slowest observed 64-field warmup and adaptation epochs took 3.504028 and 1.853860 seconds. Phase-scaled 50-epoch training is 580.427 seconds before conservatism. Every inference field is costed at the slower observed training rate; this is a conservative proxy, not an independently measured full-field inference throughput. Both phase/inference costs are doubled. The entire worst smoke non-phase overhead, 26.131606 seconds, is additionally charged **57 times** for 50 last checkpoints, four candidates, two selected-role outputs and startup.

That arithmetic gives **3,687.767 seconds, or 61.46 minutes, for the largest fit**, below its 5,400-second/90-minute allocation. Average-fit admission cost is 59.13 minutes. The independent calculations exactly match the admission ledger and all field budgets. The requested production allocation remains 30 A40 GPU-hours, at most four concurrent GPUs.

This is a capacity admission estimate, not a measured production ETA, a queue forecast or permission to extend a failed job's budget. Real production epoch/inference timings and checkpoint costs must update the estimate. Recovery work consumes the same declared resource budget and must be disclosed.

## Artifact bindings and remaining acceptance

| Artifact | Binding |
| --- | --- |
| Immutable neural source | `b5f0df18a47c1607be492bdaab7ab6cd2171226b` |
| Source archive SHA256 | `d4655c0290997adc5eaf19aef69c2c5f3e6ab460df66e443e98ff0532613e9ab` |
| Registered configuration SHA256 | `3f633d0e43f66f51c4e27b3324a17180cda2bf7f2a91c9a757d7b1187da7d7b9` |
| Input audit SHA256 | `8cdb842d4ad72b2070d7ca352c9175804c88cbf57b06ab6c8c07d8c6eaca48b5` |
| Direct smoke report SHA256 | `e493723113871ad33c5b27a578eca16d30c4a9f5ada4dc7fb0a35a1638764ad8` |
| Dual smoke report SHA256 | `b1dfedbf7a5d043364fbc052f14bc43d4281fc77ddfbe911040016e32fb7dee3` |
| Private independent smoke-review SHA256 | `dfde4ec7f7123be7506da9840b0c2f478941b638ffc8311b3e1c60a1b70174e2` |
| Private independent runtime-review SHA256 | `7684906801bfcbd6d24ff99a1b6230257a457f30789b9037ae738773e3754706` |

No source/configuration change can silently inherit these technical receipts. The complete 20-fit production matrix must still pass independent checkpoint, role, selection and saved-metric review before accepting results. All arrangements/trials/subsets and negative outcomes must remain visible. Clinical identity uncertainty, reused validation and the absence of a globally fresh cohort are unchanged.

The reviewer performed zero classifier fits, CNN constructions/forwards, raw-image reads, downloads or job mutations. Only existing metadata, pretrained/checkpoint tensors and saved receipts were inspected privately; public output contains aggregate counts and hashes.
