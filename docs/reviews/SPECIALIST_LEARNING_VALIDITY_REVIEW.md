# Independent validity review: specialist learning

November 17, 2026. This review evaluates the next proposed malignant-type learning study against the accepted five-category provider-case protocol. It independently reads the existing approved-reader ledgers and source, without fitting a model, decoding an image, running an encoder or submitting a scheduler job. A prospective review is not acceptance of an implementation or a future result.

**Current status: accept the complete metric family's recorded technical receipts, source bindings and result-ledger arithmetic. The registered primary improvement screen fails.** Both learned heads score **76.17% / 75.83%**, with macro F1 **.668624 / .587323**. These are small gains over the exact UNI2-h identity control; neither reaches the registered +.02 F1 improvement. The malignant-only ablation has the highest observed 812 two-step F1, **.672180**; P retains the 813 lead, **.602512**. These observations must not be combined into one pipeline. The [approved-reader actual-results review](APPROVED_ENCODER_ACTUAL_RESULTS_REVIEW.md) remains the accepted account of the preceding round.

## What 75.83% and 75.17% mean

An independent read of the accepted UNI2-h ledger (`f85ccff63d84cc3117f30a93b86374f9aad6ad7ac69c22197e8ae561f2ada230`) confirms the requested two-step final-five counts:

| Arrangement | Correct final decisions | Final-five accuracy | Mean subset/trial macro F1 | Separate all-four-view result |
| --- | ---: | ---: | ---: | ---: |
| 812 | 455/600 | 75.833333% | .6641798942 | 31/40 = 77.50% |
| 813 | 451/600 | 75.166667% | .5751746032 | 30/40 = 75.00% |

Each denominator 600 is eight validation provider keys × 15 nonempty magnification subsets × five overlapping trials. Correct counts come from the trace of each saved pooled five-class confusion matrix. Because all subsets/trials have equal eight-case support, their mean accuracy equals pooled correct/600. This equivalence does **not** allow replacement of the reported mean subset/trial macro F1 with macro F1 computed from the pooled confusion matrix.

The 600 decisions are dependent, and the separate all-four-view denominator is 40. The two arrangements have 31 distinct validation keys in their union, rather than 1,200 independent patients. The results do not mean that two categories scored 75.83% and 75.17%, and neither percentage is binary routing accuracy. Each is final-five accuracy for an entire original split arrangement.

Twenty independent checks also confirmed the preceding ledger's selected candidate for all ten roles and both arms: take the earliest index attaining maximum validation mean-15-subset final-five macro F1 among the four registered temperatures. This audit checks saved arithmetic and selection consistency; it does not independently replay the private field features or raw prediction archives.

## Fixed endpoint and selection requirements

- Keep pooled benign plus ductal, lobular, mucinous and papillary carcinoma. The requested pathway is binary benign/malignant first; benign stops and malignant enters one four-type expert. This study holds the two-step architecture fixed and tests head-specific feature geometry. Its matched identity comparator and the two single-head ablations isolate those changes; no new direct-five model is required to make this narrower comparison. The preceding reader's direct-five outcomes remain contextual evidence, rather than a newly matched learned-geometry direct-five control.
- Preserve the original 46 fitting/eight validation/28 excluded provider-key roles, arrangements 812/813 and five overlapping trials. Fit no transform or learner on validation/excluded cases and produce no excluded predictions.
- Keep all four magnifications and all 15 nonempty subsets. Do not infer physical field correspondence from equal image sequence numbers. Report complete arrangements separately; no per-arrangement model substitution or favorable-subset headline.
- Before outcomes, declare every candidate value, their order and exact tie handling. Select separately within each named arm by earliest maximum **end-to-end final-five validation mean-15-subset macro F1**, not malignant accuracy, binary accuracy, pooled macro F1, all-four-view accuracy or a post-hoc accuracy preference.
- Disclose the full candidate family, including distinct learner modes and identity/control variants. Prior reader and classifier selection already used these validation roles. A smaller new grid does not erase the prior search history or turn this into fresh confirmation.
- Report accuracy, macro F1, balanced accuracy, each class recall/confusion, binary routing, conditional malignant typing, proper probability losses, all-four-view and worst-subset results. Keep every negative candidate and both arrangements visible.

## Fitting and diagnostic safeguards

The concrete intervention has two independently trained diagonal metrics per role, each with 1,536 trainable parameters initialized to zero and bounded positive factors `d = exp(.5*tanh(theta))`. Each metric shares its factors over four views. The binary objective uses 46 fitting keys with labels benign/malignant; the malignant objective uses the 35 malignant fitting keys with four subtype labels. Each training observation is one case-mean vector per view, centered using the same equal-case center over all 46 fitting keys as the original UNI2-h classifier. Centered vectors are L2-normalized, transformed and normalized again. Runtime inference applies the selected head's factors to each original centered/normalized field. The identity arm uses neither learned factor; binary-only, malignant-only and both use exactly the corresponding fixed final factors.

Training is full-batch, 200 Adam steps at learning rate .01, fixed seed 812, FP32 with TF32 disabled. The original role-specific selected UNI2-h temperature is inherited for every head/arm and remains fixed. Each head's loss is mean cross-entropy across four independent same-view leave-anchor-case-out class-balanced KDE predictions, plus `10 * mean((log(d))**2)`. Within each view, each anchor case has weight `1/(K*n_class)`, with `K=2` or `K=4` for the respective head. Each class density divides by its remaining support count after removing the anchor. Consequently, the positive density uses `n_class-1` supports and the other class densities use `n_class`. Binary fitting counts 11/35 and malignant counts 25/2/5/3 leave at least one positive support after exclusion. The two head losses are independent, rather than a joint end-to-end final-five training loss. No across-view tissue correspondence is required or assumed.

There are 20 production optimized metrics shared by 40 named two-step pipelines, zero validation-based checkpoint/temperature selections, 80 fitting/validation prediction archives and 32,400 dependent records: ten roles × four arms × (46 fitting + eight validation keys) × 15 subsets. Four cumulative fitting-only technical fits are separate from production: two in the failed original technical attempt and two in the corrected retry. Validation denominators remain 600 records per arrangement per arm. The 1,536 bounded trainable parameters per metric must be reported; a bounded diagonal parameterization does not itself establish low effective capacity or sufficient independent data.

This objective uses case-mean training vectors while runtime inference uses individual fields. That is a deliberate testable choice, not a mathematical identity or a guarantee that the training objective will improve final-five validation. An identity control must reproduce the accepted prior selected UNI2-h predictions, with exact decisions/metrics and a declared tolerance for probability arithmetic. Re-normalizing already normalized vectors can cause floating-point changes even at an identity factor; the implementation must account for this without loosening classification acceptance.

Malignant-only must preserve the original binary field scores and routing decisions. Binary-only preserves the original malignant **field** probabilities, but can still change the pooled conditional malignant distribution because the joint field probabilities weight each field's subtype probabilities by that field's malignant probability before averaging. A change in that pooled diagnostic cannot alone be attributed to a newly learned malignant representation.

Every learned center, scaling statistic, projection, distance, class weight and classifier must use the current fitting support only. Weight classes and provider cases explicitly so cases with more fields do not silently dominate. The precise weighting of magnifications and subset-expanded examples must be registered with the objective, not inferred from convenient row counts.

An in-sample supervised fitting score measures memorization capacity. It cannot be compared directly with the earlier KDE's whole-query-case-out fitting score as if the difference measured generalization. A true whole-case-out diagnostic must recompute **all** train-dependent transforms and the learner without the entire queried case. Merely withholding that case from the final neighbors after fitting a representation on it is insufficient. Current final-five fitting supports are 11/25/2/5/3, so a one-case exclusion retains every class; any future support failure must be reported rather than repaired using validation examples.

The proposed cheaper diagnostic is explicitly **in-sample learned metric, with the whole query case omitted from density support**. That title is accurate: the query's labels contributed to learning the metric. It is not the earlier fully case-excluded KDE fitting diagnostic or an independent generalization estimate. Its centers/support exclusions must still follow the registered rule for that diagnostic.

For a linear or metric model, register regularization, rank, objective, iterations/stopping, deterministic seed and numerical tolerances before validation. High-dimensional frozen features and only 46 fitting cases make covariance/projection rank and numerical stability material. Selecting these values after checking subtype outcomes adds new validation opportunities.

For an SVM, explicitly register how scores become probabilities and how the 0.5 router is applied. Hidden field-random internal calibration folds can place same-case fields on both sides of calibration fitting and introduce extra machinery. Any grouped calibration, or fixed margin transformation instead of calibration, must be disclosed. A fixed softmax of margins supplies normalized scores but does not by itself establish calibration.

## Interpretation carried forward

All 82 qualified provider keys have entered development. Clinical identity remains unresolved, the trials overlap and the magnification/subset observations are dependent. The new study can test a narrowly fixed learning intervention under reused provider-case development validation; it cannot establish clinical validation, a fresh final patient test, novelty or publication readiness. A technically correct experiment may still fail its improvement screen. The 94–95% final-five target remains unachieved until supported by correctly scoped measured results.

Choosing UNI2-h and this objective after inspecting the previous reader round is adaptive development. The old temperature/reader opportunities remain inherited even though this new study has no validation selection among trained checkpoints or temperatures. Four named geometry arms are examined; selecting a winner across them would add four current arm opportunities, rather than make the whole workflow selection-free. These are staged opportunities, not evidence that every old-reader/old-temperature/new-arm Cartesian combination was fitted. Lock the primary learned arm and screen before outcomes and report all ablations, both arrangements and all class recalls without choosing a different arm or training recipe for each arrangement.

## Source and arithmetic review

The final [registration](../BREAKHIS_SPECIALIST_METRIC_SPEC.md) and configuration explicitly fix **both versus identity** as primary, with binary-only and malignant-only as ablations. Independent checks confirm all ten inherited temperatures against the hash-bound accepted UNI2-h ledger, and confirm the base-configuration binding and 20-fit/40-pipeline/80-archive/32,400-record arithmetic.

Independent scalar-loop reconstruction passed **24 mathematical assertions** for binary and malignant objectives at three inherited temperatures: class density denominators, class-balanced anchor/four-view loss, identity penalty and case-order invariance all match `numpy_objective` within 1e-12. These checks use synthetic arrays without fitting any metric. The separate gradient/fitting tests authored with the metric module remain that author's evidence; this review does not call them independent implementation reproduction.

Independent execution of the eight runner fixtures passed. They cover exact identity-path reuse, separate literal field replay, malignant-only binary preservation, a .5 routing tie, omission of unavailable views, field/case order, whole-query support exclusion, float64-before-mean precision and stored-factor mutation rejection. This verifies source mechanics, not cohort performance.

One concrete precision finding was resolved before production: the ordinary approved-reader loader's `z` uses float32 field means, whereas the inherited KDE center casts each field bag to float64 before averaging. The runner now constructs fitting centroids directly from the raw cached field bags in float64 and asserts exact center identity. The optimizer receives only those fitting centroids and fitting labels.

Source inspection confirms final-step-only fitting, independent binary/malignant optimizations, original support/class/case/field weighting, same-view comparisons, original joint-before-pooling semantics and no new validation selector. The collector reconstructs field probabilities without the runner's preparation/transform helpers; verifies complete role/arm/artifact coverage and source/configuration bindings; and checks stored states, losses, histories, normalized training-vector digests, head counts and exact malignant-only routing. The collector performs no fitting. Loss arithmetic reuses the separately scalar-checked NumPy objective, rather than constituting a second independent objective implementation.

Actual Torch FP32 log factors and factors cast to float64 are authoritative saved inference state. A float64 theta-to-factor reconstruction is checked with the declared 1e-6 rounding tolerance; it is not falsely required to reproduce different arithmetic bitwise. The separately registered fitting-only technical objective comparison uses 1e-5, and literal stored-prediction replay uses 1e-12.

Reviewed source snapshots (SHA256):

| File | SHA256 |
| --- | --- |
| `configs/breakhis-specialist-metric-v1.json` | `a188b13940c48dba2b48bddba27421747a7b14416273a4d17b9d1db28bf8f1fd` |
| `magset/specialist_metric.py` | `8db18be1022021da0e35caf28dae7d8847af47febc92c65a53698dcf96740bfc` |
| `magset/specialist_five.py` | `39cecb852940a7a784c9b8156c6e480a3ba3974e40e4bc4e55e45ef3a490e975` |
| `magset/specialist_collect.py` | `409f22a383d504129da14899bb31e662b0991179c1239311731511532339a6e5` |

The reviewer authored `scripts/review_specialist_smoke.py` as a NumPy-only actual technical acceptance entry point. It requires the original 812/trial-1 fitting role, two complete 200-update head histories with active gradients/nonidentity states, exact artifact reload, source/configuration/audit/model bindings, the registered objective tolerance and literal replay of exactly 300 fitting-only records from four arms. Queries are independently selected as the first fitting key in each of the five final classes, with 75 records per arm. It refuses validation prediction files and writes technical diagnostics without classification scores. Writing or compiling this entry point does not establish that a real GPU smoke has passed.

The original technical attempt (job 4983923) completed its two head optimizations but failed when its first-two-query sample could not supply all five labels required by the existing summary function. This is a technical query-selection defect, not a measured model failure. The failed attempt remains counted and preserved; no validation outcome or optimizer/hyperparameter change motivated the retry. The corrected selection covers all five fitting classes. Two independently executed no-fit regression tests reproduce the old incomplete-class failure, confirm all four corrected arm summaries total 300 records, and reject missing fitting classes before replay. The technical objective tolerance remains unchanged at 1e-5.

The prospective source verdict above supported the bounded technical check. The actual review below supersedes its pending-execution status.

## Actual technical and complete-result acceptance

The corrected technical check, numerical technical reviewer, all ten production roles and complete collector finished at immutable source **`da7f169013590366f4284de2de690c1c6e980e23`**. Independent reading of the accepted technical/result ledgers, the exact local source archive and committed source bytes passed **229,233 arithmetic/provenance assertions**, including **3,840 confusion matrices** across all 40 pipelines and fitting/validation roles. These comprise 3,600 subset matrices and 240 pooled matrices, covering final-five, binary and true-malignant conditional tasks. All 32,400 dependent candidate records, 80 prediction archives and 20 production metrics remain represented.

This actual review independently recalculates saved confusion-matrix metrics, subset/trial aggregates, counts, reference identity, deltas/screens, conditional accounting and probability-loss decomposition. It matches all 24 recorded implementation hashes, the configuration, reviewer script, original dataset/protocol/feature bindings and source archive. It does **not** independently load the private NPZ factor/feature arrays, original images or raw prediction JSONL here. The cluster collector's literal cached-field prediction replay and technical review are separate execution evidence. Probability losses and AUROC/AP are accepted as saved aggregates/decomposition checks, rather than reconstructed here from raw probability records.

| Accepted artifact | SHA256 |
| --- | --- |
| Complete metric result ledger | `a298435b5f373003b72c302f8ee9f093a92510d2345340b6439f16b706e4a9cb` |
| Actual fitting-only technical review | `fa4ff0fdc28c28f28cb7ed93086f828b44c04dfa7319d15cb52c332891b65677` |
| Immutable source archive | `308ee5cef62b64e4ac77c6e546f394def2a303813589f3c8cc5625e7a83156d7` |

The real accepted smoke records two completed 200-update fits, five fitting queries, 300 fitting-only records, zero validation records and zero reviewer optimizations. Binary/malignant maximum gradient norms are .007397/.048743; maximum FP32/float64 objective errors are **9.32e-8 / 1.80e-7**, below the unchanged 1e-5 tolerance. Both center hashes agree, factors stay within their registered bounds, and the smoke theta hashes/factor statistics reproduce the later 812/trial-1 production fits exactly. Four cumulative technical fits include the two preserved fits in the failed original attempt; there are still twenty production fits.

All values below are ordered **812 / 813**. Every accuracy denominator is 600 dependent validation records; each all-four-view denominator is separately 40.

| Fixed two-step arm | Final accuracy | Correct /600 | Mean subset/trial macro F1 | Separate all-four-view accuracy |
| --- | ---: | ---: | ---: | ---: |
| Identity UNI2-h | 75.83% / 75.17% | 455 / 451 | .664180 / .575175 | 77.50% / 75.00% |
| Binary-only metric | 75.67% / 75.33% | 454 / 452 | .660624 / .578434 | 77.50% / 75.00% |
| Malignant-only metric | 76.33% / 75.50% | 458 / 453 | .672180 / .581397 | 80.00% / 75.00% |
| **Both metrics: registered primary** | **76.17% / 75.83%** | **457 / 455** | **.668624 / .587323** | **80.00% / 77.50%** |

The primary gains over identity are **+2/+4 correct dependent records**, **+.33/+.67 percentage points accuracy**, and **+.004444/+.012148 F1**. Both F1 gains are below +.02. All three learned arms fail their paired identity improvement screens. The malignant-only ablation improves both arrangements modestly and has the best 812 F1, but it was not the prespecified primary arm. Binary-only is mixed and does not establish that learning a router helps both arrangements.

Against contextual P, the primary arm's accuracy changes by **−.17/+.17 points**, and F1 by **+.065909/−.015189**. P therefore retains the stronger second-arrangement F1 and its separate 85%/85% all-four-view endpoint. The malignant-only ablation ties P's 812 accuracy and exceeds its 812 F1, while remaining below P's 813 accuracy/F1. No consistent replacement, cross-arrangement winner mixture or 95% result is established.

The primary router accuracy is **83.33%/90.67%**, versus identity **83.33%/90.33%**. True-malignant conditional type accuracy is **73.33%/61.67%**, versus **72.00%/60.67%**. Malignant-only binary summaries equal identity exactly. Primary final recalls, ordered benign/ductal/lobular/mucinous/papillary, are **84.67/65.33/100.00/80.00/25.33%** and **95.33/54.67/93.33/44.00/33.33%**. The modest gains leave substantial ductal, mucinous and papillary transfer weaknesses; numerous dependent field/subset events do not add rare independent cases.

Both primary proper losses improve over identity on both arrangements: cross-entropy deltas **−.007724/−.006380**, Brier deltas **−.004086/−.002671**. Versus P, cross-entropy improves on both (**−.028713/−.239467**), while Brier worsens on 812 (**+.010286**) and improves on 813 (**−.124240**). Classification and probability-loss comparisons should remain separate.

**Actual verdict:** accept the complete fixed comparison's recorded execution and independently checked arithmetic/source bindings. The primary descriptive improvement screen fails despite small measured gains. The 94–95% objective, clinical validation and novelty remain unachieved. This closes this metric recipe without a post-hoc threshold, split, regularization or checkpoint search.
