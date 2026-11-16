# Native audit and the pending benchmark questions

Reviewed November 16, 2026. Read-only examination of the completed native feature audit and bound classifier registration; no new model execution, fits, cohort predictions, jobs or source/config changes. Native and supervised-centering classifier scores were not available to this review.

**Concrete assessment:** native extraction is now an accepted engineering artifact, not an accuracy improvement. The strongest next step is to complete the two already fixed comparisons and interpret their distinct mechanisms. More validation search is not justified by the audit succeeding.

## What the actual audit establishes

Audit SHA256 `51e4680bbffa2262997f23e54ea43191d8f605297e692aba00330d629903b9b9` matches bound classifier configuration `24c78a449a2b8da09ef0bb90e1492e94c2b956a302eeb0bdc609b0b1b508fac0`. Producer revision `9eeb4edf…`, extraction config `71528e9b…` and current producer-file bytes agree. All four shard receipts match, with 1,995/2,081/2,013/1,820 fields totaling 7,909. The input/token declarations are `[3,448,700]` and `[1601,1536]`; crop geometry and the frozen checkpoint identity agree with the registration.

The actual four-image A40 smoke reports zero repeat/reconstruction error and single-versus-batch maximum error `5.96e-8`. Peak allocated/reserved memory is approximately **6.31/6.51 GiB**. Its measured steady four-image batch took **1.583 seconds**. These replace the earlier feasibility estimates for that smoke; they are not a separately measured full-array throughput or clinical validation result.

The audit's `new_encoder_forwards: 0` describes the **CPU audit itself**. Extraction performed 7,909 production native forwards, plus 16 smoke forwards. The audit checks complete finite FP32 archives, ordered paths, geometry declarations and byte lineage; it neither independently repeats all raw-image encoding nor establishes subtype separability.

## What the benchmark should test

Native input retains 97.39%/98.25% of source area without resampling, while changing resolution, global context and rectangular positional interpolation together. The [official DINOv2 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/dinov2/modeling_dinov2.py) supports that interpolation. Supported software behavior is not evidence that this pathology checkpoint, used by the prior 224-pixel reader, transfers better at native resolution. Mean pooling still compresses 1,600 patch tokens into one descriptor; more pixels may add nuisance tissue as well as diagnostic morphology.

[SimpleShot's primary paper](https://arxiv.org/abs/1911.04623v2) supports treating centering/normalization as consequential baseline choices. It does not establish that class-balanced or head-specific centering is superior here. The paired centering study is therefore useful because shared mu5 and conditional mu5/mu2/mu4 isolate a supervised-geometry hypothesis on **unchanged F features**. It should not be presented as repairing invalid Bayesian priors. [Tip-Adapter's primary implementation](https://github.com/gaopengcuhk/Tip-Adapter/blob/d0e2d6f8c5feb8b6ce937b757810761f7155d4d5/utils.py) supports simple cache retrieval as an established alternative, not a guarantee that additional architecture will recover rare-type generalization.

The published Midnight BreakHis number remains unsuitable as a 95% target: its [pinned evaluation loader](https://github.com/kaiko-ai/eva/blob/3e902dd5878d927126ca230e681e5beae8dffae4/src/eva/vision/data/datasets/classification/breakhis.py) defaults to 40× and four selected types, omitting lobular and papillary carcinoma. Our pooled-benign/four-malignant case endpoint and missing-view evaluation differ materially.

## Stronger next-step rationale

Finish and independently collect the registered native and centering matrices. Compare native primarily with F `39d15d94…`, with C `e6e98fc…` secondary. Report each centering mode separately; choosing a mode afterward inherits eight temperature opportunities, and repeated direct controls are not independent evidence.

Use a fixed interpretation before inspecting outcomes: a native gain supports the combined input-resolution/context intervention; an additional conditional-over-shared centering gain supports head-specific geometry beyond global class balancing. A shared-only gain supports the simpler supervised center. Neither result identifies a unique biological mechanism. Gains in both studies would **not** validate combining their changes; that combination would be another recipe and selection layer.

Report five-category accuracy, macro F1, balanced accuracy, binary routing, true-malignant typing and class recall together. The existing two LC and three PC fitting cases and reused provider-key validation remain unchanged. If neither fixed intervention passes its screen, retain the specified reference and complete the capstone with the full negative evidence. The current artifacts provide no reliable prediction of 94–95%, and execution success should not be used as one.

## Actual native outcome: the fixed input intervention failed

Added after collection on November 16, preserving the preceding pre-result assessment. The accepted native comparison has SHA256 `417ec2a4a3806d63154f2c76300db378568e19a29b0d5d1531d745ea891fd6e4`, classifier revision `6c216c5c…`, and the same extraction/audit bindings reviewed above. It contains ten paired roles, 80 candidates, 160 verified prediction archives and 64,800 case/subset records. Collection performed no additional encoder forwards or optimized classifier fits. The supervised-centering collector was still pending during this review; its outcomes were not inspected.

The native input is substantially worse than the specified three-crop reference F. The following validation summaries average all 15 subsets and five trials within each arrangement:

| Arrangement | Native arm | Five-category accuracy | Macro F1 | Balanced accuracy | F dual accuracy / macro F1 |
|---|---|---:|---:|---:|---:|
| 812 | direct | 59.50% | .445619 | 52.00% | 75.50% / .592510 |
| 812 | dual | 60.00% | .439405 | 51.00% | 75.50% / .592510 |
| 813 | direct | 56.83% | .503846 | 58.13% | 76.83% / .579947 |
| 813 | dual | 61.00% | .501395 | 57.20% | 76.83% / .579947 |

Dual accuracy falls by 15.50/15.83 percentage points and macro F1 by .153105/.078552 relative to F. It also fails against the secondary single-center reference C. With all four views present, dual accuracy is still only **60%/65%**, versus F's **82.5%/82.5%**. This excludes missing-view averaging as the sole explanation. Dual cross-entropy and Brier scores also worsen against F in both arrangements; the adverse result is not confined to one accuracy threshold.

Temperature selection follows the registered earliest maximum validation macro-F1 rule. Across five trials, direct/dual selections for arrangement812 are respectively `{.01:1,.03:2,.1:2}` and `{.01:1,.03:3,.1:1}`; for813 they are `{.03:2,.1:3}` and `{.03:1,.1:3,.3:1}`. This is not universal saturation of a grid endpoint. No wider temperature search is warranted by this diagnostic. Independent arithmetic checks verified all 20 selected indices/temperatures plus the eight selected-summary confusion/support/accuracy/recall identities: **44 checks passed**. The accepted `kernel_model.py`, `field_five.py` and `kernel_collect.py` hashes equal both C and F; the native coordinator uses the same kernel-runner bytes as F.

The leave-whole-query-case-out fitting diagnostics are weak too. Native direct macro F1 is .374502/.362009 and dual is .389717/.369308; corresponding dual accuracies are 64.99%/56.12%. F's dual LOO macro F1 was .418854/.396298. These are support-excluded predictions, not self-retrieval training accuracy. Native dual LOO PC recall is only 2.22%/3.56%. The observations support poor generalization of this fixed representation/readout on the available provider cases, without identifying its cause. LOO fitting supports differ from validation supports, so their levels should not be interpreted as a controlled estimate of validation optimism.

The failure is also more specific than uniformly worse malignant representation. In812, dual routing accuracy falls from F's 92.17% to 84.83%, while true-malignant typing falls from 57.00% to 46.67%. In813, **typing instead rises from 59.33% to 65.00%**, but routing accuracy falls from 93.17% to 71.67%. Native813 router sensitivity/specificity are 76.00%/67.33%; native812 values are 94.67%/75.00%. Thus the final endpoint loses through different mixtures of routing and subtype errors.

In label order benign/DC/LC/MC/PC, native dual validation recalls are `[75.00,80.00,58.67,28.00,13.33]%` for812 and `[67.33,74.67,98.67,40.00,5.33]%` for813. In812, 45 of 75 repeated PC records and 31 of 75 LC records become DC; in813, PC mainly becomes benign (37) or MC (29), whereas 74 of 75 LC records are correct. These are repetitions across roles/subsets, not independent patients. The asymmetric pattern does not establish a class-map error, biological explanation, or a universally destroyed feature space. Direct five-way argmax and its derived binary sum-threshold diagnostic can also disagree by definition; that difference is not itself a routing implementation defect.

**Validity conclusion:** no execution mismatch is indicated by the accepted lineage, smoke parity, unchanged classifier mathematics, collector replay or the independent numeric checks. The CPU replay validates classification from stored embeddings; it does not independently rerun all raw-image encodings. Retaining almost all pixels with supported positional interpolation did not improve this benchmark. Resolution, context, aspect ratio and token pooling changed together, so these scores cannot isolate which caused the degradation. Keep F as the specified reference, report N as a failed fixed intervention, and finish the separately registered centering study without adapting its recipe to N's confusions. Neither this negative result nor any binary submetric establishes the requested 94–95% five-category endpoint.

## Actual centering outcome: a limited macro-F1 improvement

Added after the separate centering collection. The accepted public artifact is [BREAKHIS_PRIOR_CENTERED_RESULTS.json](../evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json), SHA256 `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c`. It uses the existing F three-crop embeddings, not native embeddings. Both registered modes completed with zero new encoder forwards and zero optimized classifier fits. These are 160 logical/120 structurally unique candidate definitions, 320 logical/240 structurally nonduplicated archives and 129,600 logical/97,200 structurally nonduplicated records. Duplicate direct controls do not create additional evidence.

| Arrangement | Dual centering | Five-category accuracy | Macro F1 | Balanced accuracy | All-four accuracy |
|---|---|---:|---:|---:|---:|
| 812 | old F case-balanced center | 75.50% | .592510 | 62.80% | 82.50% |
| 812 | class shared | 76.17% | .599196 | 63.87% | 85.00% |
| 812 | head specific | 76.33% | .602715 | 64.73% | 85.00% |
| 813 | old F case-balanced center | 76.83% | .579947 | 64.93% | 82.50% |
| 813 | class shared | 75.33% | .577788 | 64.53% | 82.50% |
| 813 | head specific | 75.67% | .602512 | 66.47% | 85.00% |

Head-specific centering has the highest verified mean-subset macro F1 in both arrangements. Relative to F, its gains are **.010205/.022565**, with accuracy changes **+0.83/−1.17 percentage points**. The 812 gain misses the registered .02 macro-F1 margin; both registered mode screens remain false. The 85% all-four result is useful descriptively, but cannot replace the primary all-subset endpoint after seeing the result.

The shared control matters. Against class-shared centering, head-specific centering gains only **.003520/.024724** macro F1 and **0.17/0.33 percentage points** accuracy. These correspond to one/two additional correct repeated records out of 600 within each arrangement, not one/two independent patients. Direct controls have identical predictions, metrics and temperature selections across modes, as required. Their validation accuracy/F1 is 75.50%/.588675 and 74.33%/.589168. The actual comparison therefore supports a modest change in the dual-head readout, especially 813, without demonstrating that any old center was invalid or that head-specific geometry universally improves classification.

The 813 recall tradeoff explains why macro F1 can improve while accuracy falls. Against F, head-specific centering increases PC recall by 10.67 points, DC by 5.33 and LC by 2.67, but decreases benign and MC recall by 5.67 and 5.33 points. Its final recalls are `[91.00,61.33,98.67,36.00,45.33]%`. Benign contributes four of eight validation cases per role, so its loss has greater raw-accuracy mass than one malignant type. In 812, recalls are `[95.67,65.33,93.33,49.33,20.00]%`: small improvements across the four malignant recalls accompany a one-point benign loss. Persistent PC/MC errors remain material; 813 PC still becomes MC in 30 of 75 repeated records, and MC becomes DC in 35 of 75.

Head-specific routing accuracy is 94.83%/93.00%, whereas true-malignant typing is only 58.33%/62.33%. Thus a strong binary score does not establish the final five-category target. Moreover, probability-quality metrics worsen: compared with F, head-specific mean-subset cross-entropy increases by .057019/.186983 and Brier score by .027078/.097756. They also worsen against the class-shared control. Higher macro F1 consequently does not imply more reliable probabilities; these proper-loss changes alone do not isolate calibration from discrimination.

LOO fitting macro F1 is .429886/.424453 for head-specific dual, versus F's .418854/.396298. The improvement is visible outside self-retrieval, but LOO PC recalls remain 11.11%/4.44%. The support composition and tiny rare-class counts differ from the validation setting, and repeated subsets are not independent cases. These diagnostics cannot support a clinical or biological conclusion.

Independent checks verified the 40 logical selections, all 16 selected-summary confusion/support/accuracy/recall identities, ten direct selected-result duplicates, 40 candidate direct duplicates and both aggregate direct duplicates: **140 checks passed**. Mode-tagged model/metrics file hashes and elapsed times legitimately differ; direct prediction hashes and numerical outputs do not. The kernel-density, pooling and collector source hashes still match F. No new consistency issue was found.

Interpret the selected-mode comparison with its actual selection budget: four temperatures per named mode/arm and eight if choosing between modes afterward. Head-specific dual selects `.03×2,.1×2,.3×1` in 812 and `.03×1,.3×4` in 813. Frequent upper-endpoint selection does not authorize extending the grid. Independently selected temperatures also mean the final selected summaries compare complete registered recipes, rather than isolating a center change at one common temperature.

**Combined conclusion:** N is a failed native-input intervention. P supplies the strongest macro-F1 option so far, with modest gains, worse proper losses and no passed improvement screen. Preserve F as the predeclared reference and present the P tradeoff explicitly alongside it. There is no evidence here that combining native input with supervised centering would recover the lost performance, and neither result justifies promising 94–95% five-category accuracy or claiming a novel method.
