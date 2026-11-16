# Independent LoRA smoke acceptance — pending actual artifacts

**Status: NOT RUN on real smoke outputs.** This page is a review template and implementation description, not acceptance of a job, checkpoint or model result. Full production launch must wait for an accepted private JSON receipt from `scripts/review_lora_smoke.py` after both immutable technical jobs finish.

The CPU-only reviewer accepts exactly the registered direct/dual fitting-only smokes at arrangement 812, trial 1. Each must contain three epochs, two optimizer updates and eight fitting cases per epoch: 192 sampled image forwards per arm. The two head-only epochs must leave adapters unchanged. The third epoch must show actual B gradients on its first update and actual A and B gradients on its second update. Parameter changes alone cannot prove A received gradients, because AdamW decay can change A even with a zero gradient.

The reviewer binds trusted dataset/roles/cache/config/source/audit and fixed-C1 initial head hashes; verifies exact small-state tensor names/shapes/dtypes and original fixed normalizers; checks all 113,689 allowed parameter counts; replays case/field sampling and the six learning rates; verifies warmup/final group digests, finite actual gradient receipts, positive timing/loss/memory and unchanged frozen-base digests. Both arms must share case order, sampled fields, learning rates, initial A matrices and frozen base tensors.

The only permitted smoke artifacts inside each run directory are `run.json` and `last.pt`. Candidate/validation/excluded feature or prediction archives are rejected. Initial and reconstructed-model receipts must reference the exact eight deterministic fitting-only probes with the registered 1e-4 absolute/relative tolerance and reported producer acceptance.

**Verification limit:** this review constructs no backbone and executes no GPU/model inference. It verifies saved state, recorded gradient evidence and the producer's raw-image parity receipts. It does not independently re-execute the adapted encoder. No accuracy threshold, validation performance, 95% claim or clinical claim is part of smoke acceptance.

After actual review, retain its accepted private JSON and hashes. A public status update may quote its `public_summary`, which contains no case identifiers or private paths. Do not replace this pending status with a completed claim solely because the source or synthetic tests pass.
