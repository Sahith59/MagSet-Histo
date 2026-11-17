# Matched study: forensic proof audit

**Later checkpoint:** the raw-image replays proposed below have now completed and reproduced the original three scores. See the [actual proof review](DEEP_PROOF_RESULTS_REVIEW.md) and [plain diagnosis](../DEEP_DIAGNOSIS_AND_PROOF.md). Earlier pending rows below are the preserved pre-execution audit, not the current status.

November 17, 2026. **The three published scores are correct for the archived predictions and the frozen evaluation rules.** A fresh independent arithmetic pass, source-archive comparison, exact split reconstruction and scheduler audit found no discrepancy. This establishes the saved-result calculation and recorded workflow; the remaining raw-image execution checks are listed explicitly below.

The audited [complete result](../evidence/MATCHED_THREE_WAY_RESULTS.json) has SHA-256 `3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62`. Its private error record is empty. The scientific producer remains `a6e6a54e4fac090051837df17b0dea14efca3bb3`, not a newly trained revision. The private machine audit completes with no errors; its receipt SHA-256 is `4a09816e44ccf5175acba0b7d7bd14107b9a56c1fe78c711261b3eee029a2133`.

## Exact numerical proof

The independent calculation starts from saved field NPZ arrays, not copied headline metrics. It verifies field membership/order and finite normalized five-label probabilities, averages fields within each zoom, equally averages active zooms, applies `p(malignant) >= 0.5`, and selects the largest malignant component. It reproduces every final prediction, all 15 confusion matrices per model, explicit-five-label macro F1 and the published mean accuracies. Neural validation head arrays also reproduce their saved joint-five probabilities.

| Selected method | Correct final-five case/mask events | Mean15 final-five accuracy | Mean15 macro F1 | Correct binary events | Mean15 binary accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midnight P, temperature 0.1 | 162/240 | **67.50%** | **0.3756751549** | 227/240 | **94.5833%** |
| CNN–Transformer, epoch 40 | 136/240 | 56.6667% | 0.3481798942 | 207/240 | 86.25% |
| Swin, epoch 20 | 142/240 | 59.1667% | 0.3049467787 | 198/240 | 82.50% |

Each denominator is 16 provider cases × 15 masks. These are **16 cases in 13 conservative groups**, not 240 independent specimens. Macro F1 is the mean of 15 separately computed five-class macro F1 scores, not the F1 of a confusion matrix pooled across masks. All-four accuracy separately reproduces 11/16, 9/16 and 9/16.

The error accounting also closes exactly:

| Method | Benign routed malignant | Malignant routed benign | Wrong malignant type after correct routing | Total final errors |
| --- | ---: | ---: | ---: | ---: |
| Midnight P | 8 | 5 | 65 | **78 = 240−162** |
| CNN–Transformer | 28 | 5 | 71 | **104 = 240−136** |
| Swin | 41 | 1 | 56 | **98 = 240−142** |

Thus P's binary and final-five scores are different measured quantities. Its 65 typing errors identify where the saved decisions fail; they do not prove a biological cause, label error or particular training defect. Keeping its router unchanged would keep binary accuracy at 227/240, even with a perfect new subtype expert. A typing-only change therefore cannot establish the requested >97% binary objective. None of the present models meets >90% mean15 final-five or >97% mean15 binary accuracy.

## Source, data and selection proof

The local source archive independently hashes to `d51f0f95b0bb62743dc4c10072be83a88bb5600443d44d0ce182613bd4f6fb07`. All **11 scientific implementation files** match their registered hashes, the bytes inside that archive, the corresponding Git commit objects and the current files. The archived configuration matches `1cdc7f143941a25be7b595c3b058e545c57c19a3a367df68123d977f757d5d5c`.

The input-audit bytes match `cec9c0e13f2673046b41e6502154ceb9bed047f44086afbe04d57df562ef0812`. The audit describes all 7,909 original fields, 82 provider keys and 70 conservative groups. Indexed labels and field/view memberships reproduce the saved split arrays. Every pair of roles has zero shared provider keys and zero shared conservative groups. The role counts are 49/17/16 cases, 42/15/13 groups and 4,621/1,584/1,704 fields.

To avoid silently relying on a different local library version, this audit executed the two exact split-algorithm methods captured from the actual **scikit-learn 1.6.1** runtime. Every fold assignment, the seed-812 group-code permutation, and the first eligible pair—test fold 0, validation fold 1—reproduce exactly. Eligibility includes all five labels in every role and at least two fitting groups per target.

The training weight NPZ was independently reconstructed from case classes, four views and field counts. Maximum difference is below 1e-12; each head's global fitting-field weight mass is 4,621. Benign subtype weights and all validation/test loss weights are exactly zero. The frozen code uses summed sample-weighted cross entropy divided by the configured batch size 16, including the final partial batch of 13, with equal head coefficients. Both full histories contain 40 epochs × 289 updates = 11,560 updates.

All **12 validation candidates** reproduce from their saved probabilities. First-maximum validation macro F1 selects the reported settings. The selection hashes match the common gate and the selected-test receipts; available candidate prediction/record/metric files match their receipts. Eight large neural candidate H5 files are absent from this local metadata mirror, so this audit does not claim an independent local rehash of them. The accepted cluster checker performs those file checks.

A precise distinction matters: the programs open a global crop memmap/label lookup, and P loads frozen feature bags globally. Therefore “no test data or labels were loaded into memory before selection” is **false**. The verifiable code claim is that optimization and validation selection index the permitted fitting/validation roles, and selected test model queries occur only after the common gate.

## Execution and gate chronology

The authoritative scheduler snapshot was inspected in full: **33 records covering 11 main jobs**, including batch/extern steps. Both A40 fits completed before the gate. In the scheduler's recorded clock, the gate ran 18:36:32–18:36:56; selected P test began 18:37:03. The final A40 CNN test ran 18:56:32–18:57:56 and Swin 18:58:08–18:59:36. The final checker began 18:59:38 and completed with exit 0. The gate's three selection hashes, all selected-test gate references and the scheduler ordering agree.

The A30 attempts remain **failed**, with both main/batch exit codes 1:0, even though their extern steps show exit 0. Both error logs stop at the unchanged validation replay guard before test-index construction or test forwarding. The failed checker JSON contains no accepted model metrics. No tolerance was relaxed. The later A40 receipts report exact zero binary/subtype probability difference on the saved first 16 validation fields; this audit verifies those receipts and their hash bindings, rather than claiming to have executed that reload itself.

These observations verify the recorded frozen workflow. They do not establish that unrecorded access outside that workflow was impossible. The older four-epoch interrupted CNN fit is also disclosed; it produced no eligible validation candidate and was not used for the final selection. “There were only two fitted neural processes of any duration” would be false; “both selected full trajectories completed the registered 40 epochs” is true.

## Claim ledger

| Concrete claim | Audit status | Reason |
| --- | --- | --- |
| The published final-five and binary arithmetic reproduces from saved NPZs | **TRUE** | Independent field-to-case replay, exact counts and fixed-five-label metrics agree. |
| The registered role assignments are conservative-group disjoint | **TRUE** | Zero overlaps; exact pinned algorithm and indices reproduce. |
| Four-candidate first-maximum selection and the recorded gate order agree | **TRUE** | Raw candidate scores, hashes, frozen control flow and scheduler timing agree. |
| 94.58% is P's final-five accuracy | **FALSE** | It is 227/240 binary; final-five is 162/240 = 67.50%. |
| P leads every endpoint or has established superiority | **FALSE** | CNN has higher all-four macro F1; Swin has stronger worst-subset metrics; all pairwise intervals include zero. |
| The test cohort is globally fresh, or 240 masks are independent specimens | **FALSE** | All cases entered earlier development; masks repeat 16 cases. |
| The current study meets >90% final-five and >97% binary | **FALSE** | The exact table above does not meet either target. |
| Every original field contains correctly labelled diagnostic tumor tissue | **UNTESTED** | No independent expert morphology/label adjudication has been completed. |
| Stain, resolution, tissue content or encoder quality caused the rare-type errors | **UNTESTED** | Error counts and probabilities do not isolate those causes. |
| Foundation pretraining has no BreakHis overlap | **UNTESTED** | Checkpoint/file provenance does not establish corpus exclusion. |
| An independent raw-image forward pass has reproduced all three score paths | **UNTESTED at this receipt** | New raw-file replay is being prepared; no such result is accepted by this audit yet. |

The independently implemented paired bootstrap also reproduces every accepted bound within 2e-12. It resamples 13 groups together across methods and preserves all case/mask dependencies. In 1,533/2,000 draws at least one true class is missing; fixed five labels remain in the calculation. These broad, zero-crossing intervals neither establish superiority nor prove equivalence.

## Narrow replay that closes the execution gap

The next proof step can use **fixed selected weights only**, with no optimizer, fitting, new model choice or threshold adjustment:

1. Seal the independent verifier source and existing model/checkpoint/manifest hashes. Reuse the frozen architecture definitions but independently implement raw PNG reads, labels, crop extraction, field ordering, pooling and metrics. Use A40 and the recorded runtime; the observed A30 discrepancy does not establish its exact kernel cause.
2. Verify original image bytes, independently construct the three RGB crops for both registered image geometries, and compare uint8 pixels exactly with the crop cache. Check only deterministically selected validation fields first, covering all five labels and four zooms. Retain the existing neural probability tolerance of 1e-5.
3. Once the validation check passes, run the selected CNN and Swin over all **1,704 fixed test fields**, `training=False`, batch 16. Compare every field probability, all 240 routed decisions per method and the exact aggregate scores with the sealed artifacts. Any discrepancy stops acceptance and is preserved; it must not trigger test-driven tolerance or setting changes.
4. For P, fresh raw-image features for the test fields close the **query encoder** path only. Fully independent raw-feature verification of its scoring path also requires its **4,621 fitting support fields**—6,325 fitting/test fields total—or all 7,909 fields to redo validation selection as well. Preserve the registered Midnight FP32/eager/TF32-disabled path, batch 8 and existing feature parity tolerances (`atol=rtol=1e-4`), then recompute fitting-only centers, fixed-temperature probabilities and case scores.

That replay would strengthen execution proof; it would not make the already viewed test cohort a new confirmation set. An improvement study must be registered separately and evaluated honestly. A future >90%/>97% result remains an objective to measure, not a conclusion supplied by this audit.
