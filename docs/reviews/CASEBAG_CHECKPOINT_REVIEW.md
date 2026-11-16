# Case-bag checkpoint and inference review

November 15, 2026. Scope: checkpoint recovery and deterministic inference in the prospective [case-bag draft](../BREAKHIS_CASEBAG_DRAFT.md), its [configuration](../../configs/breakhis-casebag-draft.json), [runner](../../magset/five_casebag.py) and [tiny-model fixtures](../../tests/test_five_casebag.py). The accepted first five-class experiment and its immutable source are unchanged.

**Current decision: accept the revised checkpoint/inference implementation for the bounded technical gate, conditional on final registration and immutable source/configuration binding. The concrete source blockers found in this review are closed.** This is not acceptance of the 20 production GPU fits: actual pinned model-load, parameter-partition, two-smoke and runtime receipts remain required. CPU preparation that verifies the pinned public weights and strictly loads model state without fitting or image forwards is a separate zero-fit activity.

| Reviewed artifact | SHA256 |
| --- | --- |
| Revised runner | `e34aba8a2112e67b90960cafbeb5aaca1e4c7c8b0568c6f78f4d7af75178663d` |
| Focused fixtures | `c81dd345124a52ad461bed166d8b223ba3147e02f3ebbfcdb65c06a58de57632` |
| Draft configuration at closure | `8491f80369f2007783f99e8b86c6e06719c5acd7a29949922c1b8330ba8f5404` |

These hashes identify the reviewed working source. Registration may change draft status/metadata; review those differences and bind the resulting immutable archive before real fitting.

## Findings and implemented closures

1. **One numerical routing rule — closed.** A tiny independent calculation found that a float32 intercept of −1e-20 can produce a saved malignant probability of exactly 0.5 while a logit-sign test returns benign. The revised scorer uses the exact saved-probability ≥0.5 rule. Fixtures now test tiny negative, exact-zero and positive logits, plus hard-route/global-five-way-argmax disagreement.
2. **Authoritative checkpoint validation — closed.** Before loading optimizer state, the revised validator checks complete epoch/phase and deterministic sampling receipts, initialization/final model fingerprints, active/frozen group evidence, exact AdamW groups/hyperparameters, moment shapes/dtypes/finite values, and per-parameter update counts. Corrupt moment dtype/NaN, update counts, case order and learning rate are rejected before optimizer restoration. Model and RNG restoration remain bound to the identical context and original epoch budget.
3. **Committed candidate bindings — closed.** Committed checkpoint/state/feature/prediction hashes, exact validation identities/truths, field counts, explicit subset axes, float32 descriptors and head arrays/classes are checked. All 15 saved scores and metrics are replayed from the bound sufficient statistics. Earliest-best snapshot identity is verified; the selected convenience checkpoint is reconstructed from that authoritative best state.
4. **Finite selected probability replay — closed.** The revised path directly compares all archived selected-validation probability vectors within 1e-6, matches hard decisions and still checks the feature replay and selection score. A shared maximum-difference helper rejects NaN, infinity, empty arrays and shape changes before comparing tolerance.
5. **Runtime/context binding — closed at source level.** Python, CUDA/cuDNN, dependency versions, CUBLAS workspace, source/archive/configuration/data/role/weight hashes and GPU family enter the context. The revised configuration pins A40; changed GPU family or context is rejected. Execution checks the declared workspace, rejects real CPU fitting and requires a frozen production-ready configuration plus both accepted technical smoke receipts for production.
6. **Interrupted smoke receipt — closed.** Independent tiny reproduction found that stopping immediately after the initial epoch-one atomic commit could leave an empty reload receipt; a subsequent valid resume completed epoch two but remained ineligible for smoke acceptance. The revised resume records that actual epoch-one restoration atomically before continuing. A targeted fixture matches recovered final fingerprints to the uninterrupted smoke, without duplicating real-data fitting.

The implementation owner resolved these findings before proposed CNN scientific fitting. Actual model/load/smoke checks remain separate from this source acceptance.

## Recovery and budget contract

The authoritative `last.pt` must atomically contain the completed epoch's model, optimizer, phase, RNG, exact case/draw history, candidate receipts and best state. An interrupted write must leave the preceding committed checkpoint usable. `selected.pt` is a derived convenience artifact; resume must never rewind training to that validation-selected epoch.

Production visits 46 fitting cases once per epoch, with batch size four and the final two-case batch retained. Therefore each complete epoch has 12 optimizer updates and the 50-epoch model has exactly 600 logical updates. After committed epoch e, the remaining logical budget is 50−e epochs and 600−12e updates. Selected backbone parameters have 12×max(e−5,0) updates. Direct/router parameters have 12e. Specialist parameters skip batches with no true malignant cases, so their correct AdamW step count must be derived from the actual deterministic minibatch labels; it is not automatically 600.

Resume is at epoch boundaries. Stateless case order, field selection and augmentation keys reconstruct the next epoch without depending on a loader's mutable iterator state. Restoring Python, NumPy, CPU Torch and applicable CUDA RNG states remains part of the receipt. Any physical work repeated after an uncommitted interruption consumes resources and must be disclosed; 600 logical updates do not imply zero recovery cost. Resubmission cannot extend the registered scientific or resource budget.

Validation is read-only and occurs only at epochs 10/20/35/50, using all fields and all 15 subsets. The earliest checkpoint maximizing mean-15 **end-to-end hard-label** five-class macro F1 wins. Run all 50 epochs. No validation gradients, optimizer updates, rate changes, extra checkpoints or excluded-role forwards are allowed. The two fitting-only technical smokes have their own compressed two-epoch/four-update budget and no validation selection.

## Inference and decision contract checked

The intended evaluation path encodes every allowed field in float32, accumulates within-view and available-view means in float64, casts pooled descriptors to float32, appends four availability flags and evaluates saved float32 head parameters in NumPy float64. Archive exact per-view means/counts, descriptors, head arrays, labels and all subset scores. This differs deliberately from float32 sampled-bag training and must remain explicit in reports.

Direct classification uses the five-way winner. Dual classification uses the fixed router then malignant-four winner, with no benign expert. A composed probability vector can have a different global winner; classification metrics and checkpoint selection consume the explicit routed decision, while CE/Brier consume probabilities. Stable log-sigmoid and log-softmax retain finite cross-entropy for extreme finite logits. The existing tiny numerical tests exercise this distinction and extreme-score path.

Source inspection found fitting/validation allowlists checked before image access, all-view field accumulation, fixed BatchNorm buffers and selected late affine adaptation, one encoder execution reused across the 15 masks, fixed short-batch loss scaling and specialist `grad=None` on all-benign batches. These contracts still need actual strict model-load and GPU smoke evidence. The corrected configuration explicitly declares the shared training seed/epoch/count permutation across roles and arms; case/view/path keys determine actual field draws and augmentations. This common randomness is not another independent training seed.

## Verification performed and remaining actual gates

The reviewer independently reran all **15 final focused tiny-model/image tests**, which passed in **7.940 seconds**, under Python 3.11 with NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.6.1, Pillow 11.1.0 and Torch 2.6.0. They cover uninterrupted/resumed trajectories, stale selected-file recovery, committed-artifact corruption, optimizer/sampling corruption before loading, the rounded routing tie, NaN replay, worker-count input invariance, smoke receipt binding and the epoch-one interruption edge. Tiny fixtures and toy arithmetic are software evidence, not CNN training performance. The implementation owner additionally reports a passing full 162-test suite; this reviewer independently executed the focused 15-test suite.

Before production acceptance, require final source/config registration and immutable binding, the completed strict fixture suite, pinned actual weight bytes and model-state schema, actual parameter/BatchNorm partition receipts, and exactly two bounded fitting-only technical smokes. Smokes must check finite permitted gradients, fixed early/BN state, expected head/late updates, optimizer steps/groups, same-family reload probabilities, one continuation from the existing epoch-one checkpoint and measured runtime supporting the declared budget. Do not add duplicate real-data fits to demonstrate uninterrupted/resumed parity; tiny fixtures supply that trajectory test.

After all 20 production fits, independently verify every complete 50-epoch/600-update history, four candidate states/caches, earliest-maximum choice, selected fitting/validation outputs, all 15 subset metrics and the complete two-arrangement comparison. No partial-matrix ranking or favorable-view headline is accepted. Actual-output review, generalization, clinical validity and novelty are separate claims.

This reviewer used no real CNN weights, real-image model forwards, production fitting, downloads or scheduler jobs. Only tiny synthetic fixtures and array arithmetic were executed. No source/configuration files were edited by this reviewer. The recipe remains prospective and makes no 94–95%, fresh-cohort or clinical-performance claim.
