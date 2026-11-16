# Approved pathology readers: five-category comparison

November 16, 2026. All three frozen readers and complete classifier collectors have finished and passed independent result review. **UNI2-h produces the highest observed arrangement-812 two-step macro F1, .664180, at 75.83% accuracy**, but its arrangement-813 result is .575175 at 75.17%. P retains the arrangement-813 two-step F1 lead, and F retains the arrangement-813 two-step raw-accuracy lead. Virchow2 and H-optimus-0 do not improve the requested two-step endpoint. All three readers fail the complete two-arrangement improvement screen. **94–95% final five-category accuracy remains unachieved.**

| Reader and decision rule | Accuracy 812 / 813 | Macro F1 812 / 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: |
| C, single-center Midnight / two-step | 75.67% / 75.00% | .589691 / .564292 | 82.50% / 80.00% |
| F, three-crop Midnight / two-step | 75.50% / 76.83% | .592510 / .579947 | 82.50% / 82.50% |
| P, head-specific centers / two-step | 76.33% / 75.67% | .602715 / .602512 | 85.00% / 85.00% |
| UNI2-h / two-step | 75.83% / 75.17% | .664180 / .575175 | 77.50% / 75.00% |
| UNI2-h / direct-five | 75.17% / 77.67% | .648631 / .634493 | 75.00% / 80.00% |
| Virchow2 / two-step | 72.50% / 73.00% | .575767 / .538743 | 72.50% / 77.50% |
| Virchow2 / direct-five | 72.83% / 75.00% | .590080 / .587676 | 75.00% / 77.50% |
| H-optimus-0 / two-step | 73.50% / 71.17% | .509710 / .506235 | 77.50% / 75.00% |
| H-optimus-0 / direct-five | 77.33% / 75.17% | .603560 / .606677 | 87.50% / 82.50% |

Headline scores average all 15 nonempty magnification subsets and five overlapping trials per arrangement. The all-four-view column is a separate endpoint. H-optimus-0's 87.50% on 812 is a one-stage full-view result, not a two-step or mean-15-subset result. The earlier GenBio direct-five control still has the higher 813 one-stage F1, .640603 at 77.83% accuracy. Do not combine different readers' best arrangements into a new headline model.

## The measured tradeoffs

UNI2-h changes two-step macro F1 versus C by **+.074489 / +.010883**, with raw accuracy changes **+0.167 / +0.167 percentage points**. The 812 screen passes, but the 813 F1 increase is below the registered .02 requirement, so the complete screen fails. Relative to P, UNI2-h has higher 812 F1 by .061465, lower 813 F1 by .027337 and 0.50 percentage points less raw accuracy on both arrangements. Its improvement is uneven, rather than absent or a uniform new winner.

UNI2-h's 812 balance shifts toward malignant typing: true-malignant four-type accuracy rises from C's 56.67% to 72.00%, while router accuracy falls from 92.00% to 83.33%. Mucinous recall rises from 41.33% to 77.33%, but benign recall falls from 97.00% to 84.67%. On 813, its typing accuracy is 60.67% and router accuracy 90.33%; papillary recall is 28.00%, below C's 37.33%. This explains why the large first-arrangement F1 gain does not reproduce uniformly in the second.

Virchow2 lowers two-step accuracy and F1 versus C in both arrangements, despite better joint-probability losses. H-optimus-0 is also below C in two-step accuracy/F1, while its direct-five control has higher F1 than its own hierarchy on both arrangements. H-optimus-0's two-step papillary recall is only 1.33% on 812. These negative results remain part of the comparison.

| Two-step class recall | Arrangement | Benign | Ductal | Lobular | Mucinous | Papillary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UNI2-h | 812 | 84.67% | 64.00% | 100.00% | 77.33% | 26.67% |
| UNI2-h | 813 | 95.33% | 56.00% | 93.33% | 42.67% | 28.00% |
| Virchow2 | 812 | 88.00% | 66.67% | 98.67% | 44.00% | 18.67% |
| Virchow2 | 813 | 94.67% | 42.67% | 97.33% | 36.00% | 29.33% |
| H-optimus-0 | 812 | 97.67% | 89.33% | 60.00% | 46.67% | 1.33% |
| H-optimus-0 | 813 | 96.67% | 25.33% | 93.33% | 30.67% | 33.33% |

| Two-step decision diagnostic | Router accuracy 812 / 813 | True-malignant typing 812 / 813 | Ideal-router ceiling with current fixed type predictions |
| --- | ---: | ---: | ---: |
| UNI2-h | 83.33% / 90.33% | 72.00% / 60.67% | 86.00% / 80.33% |
| Virchow2 | 89.00% / 89.50% | 59.33% / 56.00% | 79.67% / 78.00% |
| H-optimus-0 | 92.83% / 84.33% | 58.33% / 60.33% | 79.17% / 80.17% |

The ideal-router figures are arithmetic diagnostics with the current case-conditional type predictions held fixed, not measured new models. Replacing field routing or changing pooling could also change those type predictions. No threshold or cascade search follows from these numbers. The direct/dual comparison changes density grouping, composition, final decisions and separately selected temperatures; it does not isolate the routing threshold alone.

## Probability quality

Lower cross-entropy and Brier error are preferred. These losses assess the composed five-category probabilities; they do not replace explicit routed-label accuracy/F1 or establish calibrated confidence by themselves.

| Reader and decision rule | Cross-entropy 812 / 813 | Brier error 812 / 813 |
| --- | ---: | ---: |
| C / two-step reference | .9422 / .9253 | .4391 / .4504 |
| UNI2-h / two-step | .9571 / .8361 | .4722 / .4096 |
| UNI2-h / direct-five | 1.1909 / 1.0433 | .5930 / .5189 |
| Virchow2 / two-step | .8904 / .8219 | .4248 / .4066 |
| Virchow2 / direct-five | .9784 / 1.0061 | .4757 / .5124 |
| H-optimus-0 / two-step | .9210 / .9652 | .4141 / .4861 |
| H-optimus-0 / direct-five | 1.1252 / 1.1544 | .5451 / .5845 |

UNI2-h's two-step probability losses worsen against C on 812 and improve on 813; H-optimus-0 shows the opposite pattern. Virchow2 improves both losses against C in both arrangements while losing accuracy/F1. Every direct-five reader has worse probability losses than its own two-step arm, even where its hard-label metrics improve. No reader is uniformly strongest across arrangements, decisions and probability quality.

## What was compared

The experiment replaces the picture reader while preserving the registered C classifier: common case-balanced per-view centers, L2-normalized fields, cosine kernel densities with class/case/field weighting, and temperatures 0.01/0.03/0.1/0.3. Validation chooses the earliest maximum mean-15-subset macro F1 separately for each reader and direct/dual arm. The two-step system routes at malignant probability at least 0.5; benign stops, and malignant predictions receive one of four carcinoma labels. Its explicit routed labels determine accuracy and F1. Direct-five controls predict the same five final categories in one decision.

Each role retains 46 fitting, eight validation and 28 excluded provider keys. Centers and density support use fitting cases only. Fitting diagnostics leave the entire query case out of both support and centering. Field joint probabilities are averaged within each magnification and equally across available magnifications. No adapter, learned pooler, new centering mode, threshold sweep, model fusion or split search was added. C is the registered primary reference, F the secondary reference and P the retained context. [Registered comparison](BREAKHIS_APPROVED_ENCODERS_SPEC.md).

| Frozen reader | Descriptor | Declared field preparation |
| --- | --- | --- |
| UNI2-h | 1,536-dimensional CLS | Shorter-side 224 bilinear resize, center crop 224, ImageNet normalization |
| Virchow2 | CLS plus mean patch tokens, excluding four registers; 2,560 dimensions | Shorter-side 224 bicubic resize, center crop 224, ImageNet normalization |
| H-optimus-0 | 1,536-dimensional CLS | Registered shorter-side 224 bicubic resize and center crop, followed by official model-specific normalization |

The comparison measures transfer of the reader together with its declared input and feature definition. H-optimus-0's native example expects tiles at 0.5 microns/pixel; resizing BreakHis does not establish that physical scale. All current fitting/evaluation images and labels come from BreakHis. The readers' upstream pathology pretraining is external transfer learning, and exact upstream BreakHis overlap remains unresolved. UNI2-h/Virchow2 academic-use restrictions and H-optimus-0's Apache 2.0 license are recorded; checkpoints and features remain private. [Official-source and license review](reviews/APPROVED_ENCODER_RESEARCH.md).

## Technical execution and provenance

All three pinned checkpoints passed complete size/SHA256 verification, strict state loading and exact FP32 parameter-copy checks. Technical checks used the same four predeclared fitting-only fields, two from each audited source geometry. Repeat, outer host-batch partition and fresh strict-reload feature differences were all zero, within the unchanged 1e-4 absolute/relative tolerances. These are consistency checks on four fields, not performance measurements or an independent re-encoding of every original image.

| Technical receipt | UNI2-h | Virchow2 | H-optimus-0 |
| --- | ---: | ---: | ---: |
| Exact loaded FP32 parameters | 681,394,176 | 631,239,424 | 1,134,774,272 |
| State tensors | 343 | 455 | 567 |
| Peak allocated GPU memory, bytes | 2,820,907,008 | 2,603,693,568 | 4,650,800,128 |
| Steady four-image host batch, seconds | 0.133 | 0.125 | 0.221 |
| Successful technical image calls | 16 | 16 | 16 |
| Complete original-field calls | 7,909 | 7,909 | 7,909 |

Execution used NVIDIA A40 GPUs, frozen FP32 parameters/input/output, eager attention, no autocast and TF32 disabled. A host batch contains four images for I/O; the model receives one original image per call. Host-partition parity therefore checks the canonical one-image policy, not native multi-image model-batch equivalence. Every reader's four accepted shards contain the original 1,995/2,081/2,013/1,820 fields at 40/100/200/400×. Across readers this is 23,727 original-field calls, plus 48 successful technical calls. No new images or independent cases were created.

Feature producer source is `375bcbc499770635dde6c9586c5ce49e2c44300d`. Its [CI run](https://github.com/Sahith59/MagSet-Histo/actions/runs/43927220032) passed all 390 neural tests without skips, plus lightweight checks. CPU feature audits verified source/configuration/artifact hashes, ordered full-manifest coverage, finite FP32 arrays, dimensions and geometry declarations; the audits themselves performed no image decoding or encoder inference. Their zero-forward counters do not erase the preceding GPU extraction.

H-optimus-0 classifier source is `20b3463467901efd370025d93b1465c7a101dc97`; UNI2-h/Virchow2 classifier source is `a981baf38df116c3d5cfd2381a76630b078b11bf`. Later source revisions bind accepted feature hashes without changing the registered classifier mathematics. Each complete collector verifies ten paired roles, 80 candidate pipelines, 160 prediction archives and 64,800 dependent case/subset records. The three-reader family totals 240 candidates, 480 archives and 194,400 records. Four temperatures per reader give **12 reader/temperature choices per arm and role** if selecting across readers. These are validation selection opportunities, not independent experiments or patients.

The independent actual-result review passed 968,317 arithmetic/provenance assertions, including all 60 selected temperatures, plus 192 input-receipt assertions. It independently checks saved-ledger arithmetic and source/configuration bindings. The collectors' complete cached-feature prediction replay is separate evidence; the reviewer did not independently replay private raw images, model weights, feature arrays or prediction archives. [Actual-result review](reviews/APPROVED_ENCODER_ACTUAL_RESULTS_REVIEW.md) · [Model-outcome diagnosis](reviews/APPROVED_ENCODER_ACTUAL_MODEL_DIAGNOSIS.md).

| Accepted public ledger | SHA256 |
| --- | --- |
| [UNI2-h](evidence/BREAKHIS_UNI2_H_RESULTS.json) | `f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230` |
| [Virchow2](evidence/BREAKHIS_VIRCHOW2_RESULTS.json) | `dbd0085b03fde377d8e9ec2b3359a232a3740847abaf72e58f05eb54907c5cc6` |
| [H-optimus-0](evidence/BREAKHIS_H_OPTIMUS_0_RESULTS.json) | `9e316a67c805dffeb71ce4a73de29f5861d719945398cb7f38772ff1132e9659` |

## Interpretation limits

Each arrangement's headline averages five overlapping trials and all 15 nonempty magnification subsets: 600 selected validation decisions, with class supports 300 benign and 75 for each carcinoma. All-four-view results use 40 repeated case decisions per arrangement. Both denominators are dependent; the two arrangements/trials reuse 31 unique validation keys. All 82 qualified provider keys entered earlier development, and their clinical identity relationships remain unresolved. Excluded roles are not a globally fresh final cohort. These results supply exploratory provider-case validation, without a clinical-validation or novel-method claim.

The descriptive screen requires at least a 0.02 macro-F1 gain, nondecreasing raw accuracy and positive recall for all five final labels in both arrangements. A pass would be a descriptive result on reused validation, not a new independent confirmation. Binary routing, malignant conditional typing and all-four-view accuracy remain separate from end-to-end five-category accuracy. The 94–95% final accuracy goal requires a measured result on that endpoint.

The capstone should retain the complete scorecard and explain the uneven UNI2-h benefit alongside P/F and the negative comparisons. Finish the five-category presentation/demo. This study does not authorize automatic reader fusion or another setting search; further experiments require a separately fixed question and matched controls.
