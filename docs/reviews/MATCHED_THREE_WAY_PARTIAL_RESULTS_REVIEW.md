# Matched three-way comparison: partial outcome review

November 17, 2026, 18:54 UTC. **The validation selections are verified; no final three-model test comparison is accepted.** Both neural trajectories completed 40 epochs and 11,560 optimizer updates. Selected-checkpoint replay on A30 exceeded the registered probability tolerance before neural test inference. The fixed tolerance remains unchanged.

This review covers the [registered comparison](../MATCHED_THREE_WAY_SPEC.md), [configuration](../../configs/breakhis-matched-three-way-v1.json), [public validation evidence](../evidence/MATCHED_THREE_WAY_VALIDATION.json), saved candidate predictions and validation-only diagnostics. The scientific producer is `a6e6a54e4fac090051837df17b0dea14efca3bb3`; configuration SHA-256 is `1cdc7f143941a25be7b595c3b058e545c57c19a3a367df68123d977f757d5d5c`. The bound input-audit SHA-256 is `cec9c0e13f2673046b41e6502154ceb9bed047f44086afbe04d57df562ef0812`.

## Independently verified validation outcomes

A separate NumPy/scikit-learn calculation replayed **all 12 validation candidates** from saved field-probability NPZ files. It independently averaged joint five-label probabilities over fields within each magnification, equally averaged active magnifications, applied the fixed malignant probability threshold of 0.5, and recomputed every case/mask prediction, confusion matrix, macro F1 and accuracy. The saved neural binary/malignant heads also reproduce their composed joint-five distributions. Candidate prediction, record and metric hashes match the sealed selection receipts.

| Method | Selected setting | Validation mean15 macro F1 | Validation mean15 accuracy |
| --- | --- | ---: | ---: |
| Midnight P | Temperature 0.1 | 0.577663 | 79.6078% |
| CNN–Transformer | Epoch 40 | 0.484162 | 59.2157% |
| Swin | Epoch 20 | 0.393336 | 59.6078% |

These are **validation selection results**, not final test estimates or an accepted model winner. Each candidate uses the same 17 validation cases and all 15 nonempty magnification subsets. The resulting 255 case/mask records are dependent views of 17 cases, rather than 255 independent observations. Each setting is the first maximum of its four validation macro-F1 choices; all three unchanged selection hashes match the common sealed gate. Both neural saved histories independently agree with 40 epochs, the fixed learning-rate schedule and 289 updates per epoch.

The audited roles contain 49/17/16 provider cases and 42/15/13 conservative groups for training/validation/test. The reviewed memberships and saved split arrays agree, and no conservative group crosses roles. Clinical patient identity remains unresolved, and these cases have been used in earlier project development. The rare validation labels have only two lobular, two mucinous and one papillary case. The comparison matches cases, crop information, prediction unit, pooling and validation-choice count; pretrained readers, learning algorithms and computational budgets remain different full-pipeline properties.

## Checkpoint replay boundary

The selected neural checkpoints were trained on A40. Before selected test inference on A30, the frozen [test procedure](../../scripts/run_matched_three_way.py#L218) loads the selected weights and compares predictions on the saved first 16 **validation** fields. It requires both head probability arrays to agree within an absolute tolerance of `1e-5`. The guard precedes construction of test indices, the test forward call and all test artifact writes.

A separate saved-array review of the later validation-only diagnostic reproduces these exact maximum errors against the original saved candidate heads:

| Selected checkpoint on A30 | Binary maximum absolute error | Malignant-head maximum absolute error | Registered guard |
| --- | ---: | ---: | --- |
| CNN, epoch 40 | 1.0468182154e-5 | 0.0020158589 | Fail |
| Swin, epoch 20 | 3.5762786865e-7 | 0.0001609921 | Fail |

The diagnostic's expected arrays exactly match the selected candidates' saved first-16 validation heads. Its native and compiled prediction arrays are identical to one another, and both have zero head-argmax disagreements with the saved arrays on those 16 fields. This does **not** satisfy the probability tolerance or establish equivalence over the full validation or test role. Repeated-native agreement of zero is recorded by the diagnostic receipt; this reviewer did not rerun that model execution. The discrepancy's exact cause is not established by these measurements alone.

At review, both failed neural test directories contained zero files. The failed checker explicitly reported an incomplete comparison and emitted no accepted model metrics. Its failure result SHA-256 is `d1d8d1e3083946469f417c75f284220c070b098d9cca3dd34c71dcc430a964d8`. Both archived error logs end at the unchanged line-231 validation replay guard with `Selected real checkpoint validation replay failed`. Source inspection places test-index construction and the test forward call after that exception, confirming the pre-test failure boundary. No neural test result, paired test interval or final three-way ranking is accepted here. A selected P test artifact alone cannot complete that comparison and is not used to declare a winner.

## Scope and next acceptance condition

This reviewer performed no fitting, raw-image backbone inference or independent model reload. It reviewed saved probabilities, records, histories, source and receipts; the large checkpoint and shared pixel files were not independently replayed locally. The validation-only diagnostics recorded zero new fits and zero test-image forwards. No selected setting, checkpoint, seed, split, training recipe or tolerance was changed in response to the failure.

The next acceptance condition is successful selected-checkpoint validation replay under the existing guard, followed by all three selected test artifacts and the complete independent checker. Until then, the validation table above is the available measured comparison. Neither 94–95% final accuracy nor a final test winner has been established.
