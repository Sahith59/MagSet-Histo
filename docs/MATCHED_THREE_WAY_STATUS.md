# Matched comparison: execution status

November 17, 2026, 16:40 UTC. **CNN–Transformer and Swin are training in parallel on two NVIDIA A40 GPUs. No matched test score or winner exists yet.** Midnight P has completed its four validation choices; its test evaluation remains locked with the other two.

All three answer the same question: benign, ductal, lobular, mucinous or papillary? They use the same BreakHis cases, original images, three image crops and 15 combinations of the four magnifications. Each prediction describes a provider case, after pooling its images. This is a new comparison; the earlier notebook scores describe different experiments.

| Shared role | Provider cases | Conservative identifier groups | Original images |
| --- | ---: | ---: | ---: |
| Training | 49 | 42 | 4,621 |
| Validation: choose settings | 17 | 15 | 1,584 |
| Test: compare selected models | 16 | 13 | 1,704 |

Related identifiers and all their images stay together. These groups are not verified clinical patients. All cases have appeared in earlier project development, so this is a reused-development benchmark, with substantial uncertainty from the small test cohort.

The full [fixed recipe](MATCHED_THREE_WAY_SPEC.md) was registered before model outcomes. The [input audit](evidence/MATCHED_THREE_WAY_INPUT_AUDIT.json) passed. Both raw image geometries have exactly matching crop pixels across the pipelines. Both GPU technical checks on A30 passed full and partial batches, finite-loss training updates and fresh-model weight reloads; reload differences were zero. Production training uses A40 GPUs. These are [technical checks](evidence/MATCHED_THREE_WAY_TECHNICAL.json), not classification results. The immutable producer is `a6e6a54e4fac090051837df17b0dea14efca3bb3`; its GitHub checks passed. The [independent protocol review](reviews/MATCHED_THREE_WAY_PROTOCOL_REVIEW.md) found no remaining static blocker.

| Stage | Scheduler job | Observed state |
| --- | --- | --- |
| Input audit | 4986135 | Complete, exit 0 |
| Shared image crops | 4986136 | Complete, exit 0 |
| CNN and Swin technical checks | 4986138 | Both complete, exit 0 |
| Midnight P validation selection | 4986139 | Complete, exit 0; no test query |
| CNN and Swin training | 4986295 | Both running on A40; 8 and 4 epochs complete at this check |
| Seal all three validation selections | 4986296 | Waiting for both training runs |
| Selected neural test evaluation | 4986297 | Waiting for selection seal |
| Selected P test evaluation | 4986298 | Waiting for selection seal |
| Independent metrics and artifact check | 4986299 | Waiting for test evaluations |

Both neural models receive 40 epochs and exactly four validation choices: epochs 10, 20, 30 and 40. All three selections must be sealed before any selected-model test query. The scheduled test jobs and independent checker then run automatically. We will report final-five accuracy alongside macro F1, which gives the rare categories equal importance, plus all-four-view results and paired group uncertainty intervals. We cannot infer the final accuracy from fitting loss or promise 95%.

One interrupted fitting attempt is preserved. During a move from the public GPU queue to the lab A40 queue, CNN job 4986140_0 was canceled after four epochs and 1,156 optimizer updates. It had produced no validation candidate or test predictions. Its files are archived privately; Swin had not started. The new runs start from the original pretrained weights with the same source, split, seed and recipe. They do not resume the interrupted weights. No model setting was changed from its outcomes.
