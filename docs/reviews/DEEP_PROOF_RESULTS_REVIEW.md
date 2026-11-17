# Deep proof and specialist follow-up: actual-result review

November 17, 2026. **Selected-checkpoint inference and reported metrics for all three original methods now reproduce through raw-image execution, with unchanged final decisions.** The new fixed-router specialist reaches 82.92% on the already viewed follow-up set, but fails its validation progression screen. Neither >90% final-five nor >97% mean15 binary accuracy has been achieved.

This review checked the completed proof/diagnosis receipts, their source and artifact hashes, and independently replayed all newly saved probability arrays and statistics. Its private machine receipt has no errors and SHA-256 `22d334445997c9a9ae938254f737084727da561b3fe19aaa9b5db5b350460103`. No fitting, job submission or scientific-code change was performed for this review.

## The original three scores are reproduced

The previous [forensic audit](MATCHED_PROOF_AUDIT.md) proved the saved-array arithmetic, frozen source, split and recorded execution chronology. These new passes close its raw-forward gap for the original three pipelines:

| Pipeline | Raw fields verified | What was recomputed | Maximum probability difference | Routed case/mask disagreements |
| --- | ---: | --- | ---: | ---: |
| CNN–Transformer | All 1,704 test fields | Original PNG → independent crops → fresh selected checkpoint → joint probabilities | **0.0** | **0/240** |
| Swin | All 1,704 test fields | Same complete selected-model path | **0.0** | **0/240** |
| Midnight P | All **7,909 fields**, including 49 fitting supports and every validation/test field | Three fresh raw-image descriptors → independent centers/densities → four validation candidates and fixed selected test prediction | 8.5522141e-7 across validation/test comparisons; selected test 7.7715612e-16 | **0 in every comparison** |

For both neural readers, the newly saved field probabilities are bit-for-bit identical to the original arrays. Separate arithmetic reproduces their 16-case/15-subset scores: CNN 136/240 = **56.6667%**, macro F1 **0.3481798942**; Swin 142/240 = **59.1667%**, macro F1 **0.3049467787**.

P re-extraction covers every original field across four complete magnification shards, including its support statistics rather than only its test queries. It verifies PNG hashes, filename/directory targets and exact independent RGB crop equality. The hash-bound CPU collector reports maximum feature difference **1.4901161e-6**, within the previously registered feature `atol=rtol=1e-4`, and unchanged fitting head centers. Its independently implemented float64 KDE uses the fixed supports and registered temperatures. All four validation decision sets and the selected test decisions/metrics reproduce exactly. P remains 162/240 = **67.50%**, macro F1 **0.3756751549**, with binary accuracy 227/240 = **94.5833%**.

The local review independently rehashed the available proof outputs and recomputed all five P comparison probability differences, case decisions and scores. The large raw-feature shard NPZ files were not copied to this local metadata mirror; their full byte checks and feature-parity measurements were performed by the reviewed cluster collector, bound to the four extraction receipts. This distinction does not affect the independently checked output-probability/decision comparisons.

These passes comprise **11,317 raw-image visits**: 7,909 for P plus 1,704 each for CNN and Swin, or 33,951 crop evaluations. They involve the same **7,909 unique original fields and 82 provider keys**, not 11,317 new independent specimens. The test comparison still uses 16 provider cases in 13 conservative groups.

The neural verifier source is `9f80dde93efdb91a3d7bea569eccb42c4de1cf8d`; the all-field P proof source is `fc15fb97c5ac29bbdf6362ffd86a8443b89f2cce`. Both reuse the frozen scientific model definitions from `a6e6a54e4fac090051837df17b0dea14efca3bb3` while independently reading pixels and calculating scores. The accepted P proof receipt SHA-256 is `db1aa20aca8f09ebf5c48e87e221a05bf092519388556b07438e1a1b8c34edbc`. This is execution verification of those definitions, not a separate neural-architecture implementation or expert adjudication of medical labels.

## The new specialist helps retrospectively but fails progression

The prespecified family keeps P's binary router fixed and compares four malignant temperatures for each of two heads: its P control and a frozen UNI2-h specialist. All eight validation NPZs and both selected follow-up NPZs were independently replayed through field/zoom pooling, all 15 decisions, macro F1 and accuracy. The first-maximum selections, identity anchor, support/context hashes and both-arm gate agree. Every field binary marginal agrees with P within 1e-12, and every case/mask binary decision is identical.

| Selected head, both T=0.1 | Validation correct events | Validation mean15 accuracy / macro F1 | Retrospective follow-up correct events | Follow-up mean15 accuracy / macro F1 |
| --- | ---: | ---: | ---: | ---: |
| P control | 203/255 | **79.6078% / 0.577663** | 162/240 | 67.50% / 0.375675 |
| UNI2-h specialist | 197/255 | **77.2549% / 0.590347** | 199/240 | **82.9167% / 0.663685** |

The validation macro-F1 delta is **+0.0126838**, below the registered +0.02 screen, and accuracy falls **2.35294 percentage points**. The progression screen therefore **fails on both conditions**. The larger retrospective gain cannot override that outcome or justify promotion from test performance alone. All-four follow-up accuracy is 13/16 = 81.25%, macro F1 0.564706, also below the requested final-five target.

The specialist has the same 227/240 binary correctness as P by construction. Its final errors fall from P's 78 to 41 on these dependent follow-up events: the same 13 routing errors plus 28 malignant typing errors. Those are genuine saved-array differences, not 37 new independent cases or fresh confirmation. This family was designed after the parent test cohort had been viewed and is explicitly retrospective. The new UNI2-h score is checked against its pinned feature cache and independent density arithmetic; a new raw-image UNI2-h encoder replay is not part of the original three-pipeline proof above.

The accepted family source is `842b3b83024a2ab78012f356296d1682a0e590e3`; its checker receipt SHA-256 is `b17a23e5804b8d721c7e251e7ca76ddf33371a4c04128fa9f68b67494659be9e`.

## Router diagnosis is measured, not a 97% improvement

The separate router diagnostic makes no threshold choice, calibration fit, optimized head or test query. For each fitting query it excludes **every case in that query's conservative group** before computing support weights and centers. There are 49 fitting queries across 42 groups, leaving 46–48 support cases per query. Validation statistics use only the previously saved selected P validation probabilities.

Independent calculations from the private case-probability NPZ reproduce every reported scalar and calibration-bin count:

| Diagnostic role | Fixed-threshold binary correct events | Mean accuracy | Balanced accuracy | Mean ROC AUC | Brier / cross entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fitting-group-out | 644/735 | **87.6190%** | 85.1905% | 0.952109 | 0.103022 / 0.357585 |
| Existing validation | 238/255 | **93.3333%** | 95.2778% | **1.000000** | 0.071961 / 0.274471 |

Fitting-group-out is not fitted accuracy, and AUC 1.0 is not 100% thresholded accuracy. The differing probability calibration is a concrete diagnostic observation; it does not prove that changing a threshold or calibration will achieve >97% on future cases. No such intervention was fitted or accepted here.

## What is proved and what remains open

**Verified:** the original three numerical results, original raw-image execution paths, unchanged selected settings and decisions, P's complete raw support/query feature coverage, the specialist's measured outcomes and failed progression screen, fixed binary invariance, and the router diagnostic statistics.

**Still untested:** independent expert confirmation that each original field contains correctly labelled diagnostic tumor, verified clinical patient identity, foundation pretraining exclusion of BreakHis, a causal explanation of rare-subtype errors, a new raw-UNI2-h forward replay for the specialist, and a >90%/>97% result on an appropriate newly registered evaluation. Source/byte proof does not resolve those different questions.

The [original matched comparison](../MATCHED_THREE_WAY_RESULTS.md) remains unchanged. Its small, previously used cohort and broad paired intervals still limit generalization. The new evidence proves execution and reports a promising retrospective typing observation; it does not turn the failed validation screen into a successful replacement or make the requested accuracy targets measured facts.
