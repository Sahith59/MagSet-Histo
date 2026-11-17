# Matched comparison: complete and independently checked

November 17, 2026. **The matched three-way test comparison is complete and accepted.** Selected A40 test array 4987130 completed in 1m24s / 1m28s with zero validation reload error for both models; independent checker 4987131 completed in 1m20s with an accepted JSON and no private review errors.

| Method | Final-five test accuracy, mean of 15 zoom subsets | Test macro F1 | Benign/malignant test accuracy only |
| --- | ---: | ---: | ---: |
| Midnight P | 67.50% | 0.375675 | 94.58% |
| CNN–Transformer adaptation | 56.67% | 0.348180 | 86.25% |
| Swin adaptation | 59.17% | 0.304947 | 82.50% |

All rows use the same 16 test provider cases, 13 conservative groups and 1,704 original images. **The 94.58% binary score is not the final-five score.** P leads the registered mean-15 point estimates; CNN leads all-four-view macro F1 and Swin leads worst-subset performance. Every paired interval includes zero. This small reused-development benchmark does not establish population superiority, clinical validation or 95% final accuracy. [Complete results and limitations](MATCHED_THREE_WAY_RESULTS.md) · [accepted evidence](evidence/MATCHED_THREE_WAY_RESULTS.json) · [independent final review](reviews/MATCHED_THREE_WAY_RESULTS_REVIEW.md).

The dated execution history below preserves the completed training, validation selection and failed A30 attempt. Its pending states were true at that checkpoint and are superseded by the completed A40 result above.

## Earlier checkpoint: November 17, 18:53 UTC

These are the independently replayed **validation results used to select settings**, averaged over all 15 zoom combinations on the same 17 provider cases:

| Method | Selected setting | Five-category validation accuracy | Validation macro F1 |
| --- | --- | ---: | ---: |
| Midnight P | Shared temperature 0.1 | 79.61% | 0.577663 |
| CNN–Transformer adaptation | Epoch 40 | 59.22% | 0.484162 |
| Swin adaptation | Epoch 20 | 59.61% | 0.393336 |

These are not final test accuracies. Midnight P leads this validation comparison; a final test ranking remains pending. The [hash-bound validation receipt](evidence/MATCHED_THREE_WAY_VALIDATION.json) includes the separate all-four-view validation results and the numerical replay diagnostics. All 12 validation candidates, first-maximum selections and the common selection seal passed [independent probability-to-metric arithmetic replay](reviews/MATCHED_THREE_WAY_PARTIAL_RESULTS_REVIEW.md). No model or setting was changed from test outcomes.

All three answer the same question: benign, ductal, lobular, mucinous or papillary? They share the original BreakHis fields, three image crops, grouped roles and all 15 nonempty subsets of four magnifications. Each answer describes a provider case after pooling its images. Earlier notebook scores describe different tasks and splits.

| Shared role | Provider cases | Conservative identifier groups | Original images |
| --- | ---: | ---: | ---: |
| Training | 49 | 42 | 4,621 |
| Validation: choose settings | 17 | 15 | 1,584 |
| Test: compare selected models | 16 | 13 | 1,704 |

Related identifiers and all their images stay together. These groups are not verified clinical patients. All cases have appeared in earlier project development, so this is a reused-development benchmark. The test cohort has only one lobular and one papillary case; uncertainty will be substantial. The 255 validation case/subset events are dependent observations of 17 cases.

The [fixed recipe](MATCHED_THREE_WAY_SPEC.md) was registered before model outcomes. The [input audit](evidence/MATCHED_THREE_WAY_INPUT_AUDIT.json) and [A30 technical smokes](evidence/MATCHED_THREE_WAY_TECHNICAL.json) passed. Production fitting used A40, with 11,560 optimizer updates per model. The immutable producer is `a6e6a54e4fac090051837df17b0dea14efca3bb3`; its GitHub checks passed. The [protocol review](reviews/MATCHED_THREE_WAY_PROTOCOL_REVIEW.md) records the scope of the actual preflight checks.

| Stage | Scheduler job | Observed outcome |
| --- | --- | --- |
| Input audit / shared crops | 4986135 / 4986136 | Complete, exit 0 |
| CNN / Swin technical smokes | 4986138 | Both complete, exit 0; synthetic checks |
| Midnight P validation selection | 4986139 | Complete, exit 0 |
| CNN / Swin fitting | 4986295 | Both complete, exit 0; 54m32s / 2h07m59s |
| Common three-selection seal | 4986296 | Complete, exit 0 |
| Selected P test inference | 4986298 | Complete, exit 0; comparison not yet accepted |
| Original pending A40 neural tests | 4986297 | Held, verified unstarted, then canceled for A30 queue move |
| Selected A30 neural test attempts | 4987098 | Both failed validation reload; no neural test forward |
| First independent checker | 4986299 | Fail-closed JSON: incomplete, no accepted model metrics; scheduler exit 0 is not acceptance |
| A30 validation-only diagnostics | 4987127 | Both complete, exit 0; confirmed replay differences |
| Selected A40 neural test retry | 4987130 | Subsequently complete, exit 0; both reload errors zero |
| New independent checker | 4987131 | Subsequently complete, accepted result JSON; no errors |

At the earlier scheduler check, the A40 array was estimated to start at **22:12 UTC / 5:12 p.m. Eastern on November 17**. That was a queue estimate, not a promised completion time. A40 resources became available earlier; both selected-only evaluations and the checker subsequently completed. No training run, refit, selection change or extra candidate was added.

The A30 move preserved the frozen recipe: the pending test array and checker were held and verified before rebinding the checker and canceling only the unstarted original array. Both A30 attempts stopped at the registered first-16-validation-field replay guard, before constructing test indices or running a test forward pass. Their empty output directories, logs and failed checker are archived privately.

The validation-only diagnostic measured maximum binary/subtype probability differences of about **0.00001047 / 0.00201586 for CNN** and **0.00000036 / 0.00016099 for Swin**. Repeated A30 inference was identical, compiling the fresh model did not change the differences, and neither head changed its argmax on those 16 validation fields. This does not establish agreement on the full evaluation set. Cross-GPU arithmetic is the leading hypothesis, pending an A40 replay. The original **1e-5 tolerance stays unchanged**. If A40 also fails, test remains blocked while serialization and inference state are investigated.

All three selections were sealed before test evaluation. Final reporting requires successful selected-checkpoint replay, complete selected-only test inference and an accepted independent checker JSON. It will include final-five accuracy, macro F1, all-four-view and worst-subset results and paired group uncertainty. No 94–95% final result is established.

One interrupted fitting attempt is preserved. During a move from the public GPU queue to the lab A40 queue, CNN job 4986140_0 was canceled after four epochs and 1,156 optimizer updates. It had produced no validation candidate or test predictions. Its files are archived privately; Swin had not started. The new runs start from the original pretrained weights with the same source, split, seed and recipe. They do not resume the interrupted weights. No model setting was changed from its outcomes.
