# Learned binary and malignant similarity: completed results

November 17, 2026. **Learning feature weights produces small improvements, but the registered primary screen fails.** The primary both-head arm scores **76.17% / 75.83%** final five-category accuracy and **.668624 / .587323** macro F1 in arrangements 812 / 813. Its F1 gains over the matched UNI2-h identity control are **+.004444 / +.012148**, below the required +.02 in each arrangement. Retain the established reference decision; **94–95% accuracy and novelty remain unachieved**.

This page reports the complete [exact result ledger](evidence/BREAKHIS_SPECIALIST_METRIC_RESULTS.json), [registered recipe](BREAKHIS_SPECIALIST_METRIC_SPEC.md) and [configuration](../configs/breakhis-specialist-metric-v1.json). The comparison used original roles and inherited temperatures. No new validation setting, checkpoint, threshold or split was selected.

## What was learned and evaluated

The frozen UNI2-h reader supplies the same original 1,536-dimensional field descriptors. Two independent bounded diagonal metrics learn which dimensions to emphasize: one for benign/malignant, one for ductal/lobular/mucinous/papillary. Each uses 200 fixed full-batch Adam updates, learning rate .01, seed 812, FP32 with deterministic execution and TF32 disabled. The trainable scales are `exp(.5*tanh(theta))`, with identity penalty `10*mean(log_scale**2)`.

Training uses fitting-case mean descriptors, the original full-46-case per-view center and different-case density support. The binary head uses 46 fitting cases; the subtype head uses 35 malignant cases, including only two lobular and three papillary. Inference applies the learned scales to original fields, preserves class/case/field-balanced KDE, composes the two-step field-level joint probabilities, averages within fields/views and keeps the .5 malignant route. This case-mean training versus field-level inference distinction is material.

All ten original roles completed: five overlapping trials per arrangement, each with 46 fitting/eight validation/28 excluded provider cases, all four magnifications and fifteen nonempty subsets. Each arrangement contributes 600 dependent validation case/subset events; the all-four-view endpoint contributes 40. Macro F1 is averaged within trial/subset, not computed from one pooled confusion matrix. The four arms share the same two trained metrics per role.

## All four validation arms

| Two-step arm | Accuracy 812 | Macro F1 812 | Accuracy 813 | Macro F1 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity: exact accepted UNI2-h replay | 75.83% | .664180 | 75.17% | .575175 | 77.50% / 75.00% |
| Binary metric only | 75.67% | .660624 | 75.33% | .578434 | 77.50% / 75.00% |
| Malignant metric only | 76.33% | .672180 | 75.50% | .581397 | 80.00% / 75.00% |
| **Both metrics: primary arm** | **76.17%** | **.668624** | **75.83%** | **.587323** | **80.00% / 77.50%** |

Both metrics give 457/600 and 455/600 correct events, versus identity's 455/600 and 451/600: net gains of two and four dependent events, or +.33/+ .67 percentage points. Both all-four-view results improve by one event out of 40. These small descriptive gains do not pass the prospectively specified paired F1 screen. All three learned arms fail the +.02 screen in each arrangement; all retain positive recall for every final label. No favorable arm from one arrangement is combined with another arrangement's winner.

The malignant-only arm has the highest F1 in this family on 812; the both-head arm leads on 813. These are ablation observations, not permission to replace the fixed primary comparison or to choose a different arm per arrangement. The identity replay matched the accepted UNI2-h reference under the collector's registered numerical checks.

For the primary arm, mean-subset cross-entropy improves from .957088/.836066 to **.949363/.829686**, and Brier loss from .472184/.409558 to **.468098/.406887**. Mean trial worst-subset F1 rises from .480000/.377556 to **.484286/.384222**. Proper-loss gains describe probability quality; no empirical calibration claim follows from them alone.

## Binary and malignant diagnosis

| Arm | Router accuracy 812 / 813 | True-malignant subtype accuracy 812 / 813 |
| --- | ---: | ---: |
| Identity | 83.33% / 90.33% | 72.00% / 60.67% |
| Binary only | 83.33% / 90.67% | 72.00% / 60.67% |
| Malignant only | 83.33% / 90.33% | 73.33% / 61.33% |
| Both | 83.33% / 90.67% | 73.33% / 61.67% |

The malignant metric modestly improves subtype discrimination. The binary metric leaves 812 router accuracy unchanged and gains .33 percentage points on 813. Binary-only can also change pooled subtype decisions because it changes field malignancy weights before joint-probability pooling; it is not solely a final routing intervention. Malignant-only preserves the original binary outputs.

With both metrics, 812 has 143 final errors: **100 routing errors and 43 wrong types after a correct route**, versus identity's 145 = 100 + 45. Arrangement 813 has **145 = 56 + 89**, versus identity's 149 = 58 + 91. The remaining specialist errors remain substantial, particularly on 813. These counts describe reused case/subset events rather than independent patients.

Final class recalls, in benign/ductal/lobular/mucinous/papillary order:

| Arm / arrangement | Benign | Ductal | Lobular | Mucinous | Papillary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity / 812 | 84.67% | 64.00% | 100.00% | 77.33% | 26.67% |
| Both / 812 | 84.67% | 65.33% | 100.00% | 80.00% | 25.33% |
| Identity / 813 | 95.33% | 56.00% | 93.33% | 42.67% | 28.00% |
| Both / 813 | 95.33% | 54.67% | 93.33% | 44.00% | 33.33% |

Papillary recall therefore worsens slightly on 812 while improving on 813; ductal recall falls slightly on 813. The learner does not deliver universal rare-subtype recovery. The old perfect-router ceilings held the old conditional predictions fixed; changed metric predictions must not be evaluated against those numbers as though they were universal bounds.

## Optimization succeeded; transfer gains were small

All twenty production head objectives decreased. Across ten roles, the mean binary regularized objective falls from **.250724 to .231938**, and malignant objective from **1.600855 to 1.233660**. The corresponding final mean data cross-entropies are .218403 and 1.059352; regularization contributes .013535 and .174309. Actual factors stay inside the registered bounds: observed binary extrema .784476–1.536757 and malignant extrema .666390–1.580210.

Both-head fitting diagnostic F1 increases from identity's .476545/.498092 to .515871/.549355, with fitting accuracy 66.38%/69.39%. **These learned-arm scores are in-sample learned-metric diagnostics with the query omitted from density support and support centering.** The query's label and inputs still trained the metric. They are not leave-case-out model generalization, and cannot be compared to identity's unoptimized support-exclusion scores as equivalent independent fitting tests.

Lower objective and finite active gradients establish successful optimization of the registered question. They do not establish that it learned transferable morphology. A plausible remaining mismatch is optimizing similarities between case means while deploying field-level density and joint pooling; the current result does not isolate that explanation. Sparse independent subtype diversity and nuisance variation remain other possibilities. These are research considerations, not measured causes or authorization for another automatic experiment.

## Reference decision and scope

[Retained P](BREAKHIS_PRIOR_CENTERED_RESULTS.md) remains **76.33% / 75.67% accuracy**, F1 **.602715 / .602512**, with **85% / 85%** in the separate all-four-view condition. Primary both-head learning has higher 812 F1 but lower 812 accuracy, and higher 813 accuracy but lower 813 F1. It therefore supplies no consistent paired replacement of P. [F](BREAKHIS_COVERAGE_RESULTS.md) retains its previously accepted 813 two-step raw accuracy of **76.83%**. Do not relabel P as the universal F1 leader: UNI2-h and the current family have higher 812 F1. Do not mix P/F/UNI predictions or use their all-four-view scores as the fifteen-subset headline.

The family has four prespecified arms and no new validation-based hyperparameter, seed, checkpoint or temperature selection. A retrospective arm choice would still add four arm choices. UNI2-h was chosen adaptively after earlier reader results and its inherited temperatures were previously selected on validation. The same 31 distinct validation keys and overlapping trials are reused, all 82 qualified provider keys have entered development, and clinical identities remain unresolved. These results are exploratory development, with no globally fresh final cohort, no clinical validation and no established novelty. **94–95% final five-category accuracy remains unachieved.**

## Completed budget and smoke recovery

Production completed **20 optimized head fits, 40 arm/role pipelines, 80 prediction archives and 32,400 dependent fitting/validation records**. The collector performed zero fits. No new encoder forwards or excluded-role predictions occurred. Each production role used two fixed 200-step head fits; model weights remained frozen. The registered allocation was one A40 GPU, four CPUs and sixteen GiB per role, fifteen minutes per task with at most four concurrent GPU tasks; the CPU collector had thirty minutes. Saved per-role program elapsed times range from **37.42 to 49.66 seconds**, summing to 416.43 seconds; this is neither cluster queue time nor GPU utilization.

The first technical smoke, job **4983923** under source `e7afde39ac075cb73e6dd010c1de6ea94e50d937`, completed two full head optimizations, then failed while summarizing only two fitting queries because the summary requires all five final labels. Initial audit 4983922 completed; dependent review/production/collector jobs 4983924–4983926 were canceled before starting. The initial artifacts were preserved. This was a real reporting failure after compute, not a failed optimization or an omitted attempt.

The retry scored the first fitting case from each final label, without validation scoring or a changed training objective. Its two new optimizations give **four cumulative technical head fits**, in addition to twenty production fits. Under source `da7f169013590366f4284de2de690c1c6e980e23`, audit **4983941**, smoke **4983942**, independent NumPy smoke review **4983943**, ten-task production array **4983944**, and complete collector **4983945** all completed successfully.

The accepted independent fitting-only review covers five fitting queries, four arms and **300 fitting records**, with zero validation records or reviewer fits. It checks all 200 updates per head, finite active gradients, saved factors/state, the common fitting-only center, NumPy objective parity and prediction replay. Maximum objective differences were **9.324e-8 binary / 1.795e-7 malignant**, within the fixed 1e-5 tolerance; prediction tolerance was 1e-12. These checks certify saved-state execution, not clinical performance or an independent raw-image re-encoding. The complete production collector then replayed the actual role/arm matrix and all 80 archives.

## Exact provenance

The public result ledger is the exact completed collector output, with models, raw cache arrays, private job records and fitting histories retained privately. The smoke-review digest below identifies the accepted independent technical receipt; its substantive findings are reported above.

| Artifact | Revision / SHA256 |
| --- | --- |
| Numerical producer source | `da7f169013590366f4284de2de690c1c6e980e23` |
| Producer source archive | `308ee5cef62b64e4ac77c6e546f394def2a303813589f3c8cc5625e7a83156d7` |
| [Complete result ledger](evidence/BREAKHIS_SPECIALIST_METRIC_RESULTS.json) | `a298435b5f373003b72c302f8ee9f093a92510d2345340b6439f16b706e4a9cb` |
| [Registered configuration](../configs/breakhis-specialist-metric-v1.json) | `a188b13940c48dba2b48bddba27421747a7b14416273a4d17b9d1db28bf8f1fd` |
| Accepted independent smoke review | `fa4ff0fdc28c28f28cb7ed93086f828b44c04dfa7319d15cb52c332891b65677` |
| Input audit | `a1f5a53c513e84e55772e6b067010945519884b32714a403acc1e700371149df` |
| Dataset manifest | `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31` |
| Original role protocol | `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134` |
| UNI2-h feature producer source | `375bcbc499770635dde6c9586c5ce49e2c44300d` |
| Accepted full feature audit | `9f5895589bad5fc29a9d2c77378ca72e98ea26da9a98ee1512dc25ae166051ff` |
| [Matched UNI2-h reference](evidence/BREAKHIS_UNI2_H_RESULTS.json) | `f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230` |
| [Contextual P reference](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) | `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c` |

The training receipts record PyTorch 2.6.0+cu124; the numerical replay environment records Python 3.11.18, NumPy 1.26.4 and SciPy 1.15.2. Per-view cache hashes, encoder/configuration revisions, per-role inherited temperatures, implementation hashes, model hashes, head artifact hashes and all aggregate/candidate outcomes are preserved in the complete ledger. The result is a completed modest-gain/failed-screen study, not a synthetic performance estimate or a successful 95% model.
