# The BreakHis project in plain language

Updated November 17, 2026. We predict **five categories: pooled benign, ductal, lobular, mucinous and papillary carcinoma**. The new two-head feature-learning study gives **76.17% / 75.83% final accuracy** and macro F1 **.668624 / .587323**. This is a small improvement over its same-reader control, and it fails the registered larger-improvement screen. **94–95% final accuracy has not been achieved.** [New verified result](BREAKHIS_SPECIALIST_METRIC_RESULTS.md), [full scorecard](../paper/breakhis-five-class-report.md).

**What does 75.83% / 75.17% mean?** Imagine giving the earlier UNI2-h model two sets of quizzes. The pictures come from the same BreakHis dataset, but the fitting/validation arrangement differs. It answered **455 of 600 checks** correctly in arrangement 812 and **451 of 600** in arrangement 813. The slash separates those two validation results. Both measure the final five-category answer. They are not a training/test pair or the two decisions' scores. Each case appears in several zoom combinations and overlapping trials, so 600 checks do not mean 600 different people. Training both feature-adjustment heads now gives **457/600 and 455/600** on those same quizzes.

The separate malignant-only ablation reaches .672180 F1 on 812, the highest observed two-step 812 F1 so far; the retained Midnight reference P still leads 813 F1 at .602512. P accuracy is 76.33% / 75.67%, and F retains higher 813 raw accuracy at 76.83%. All-four-view scores are a separate endpoint. The next registered standard supervised UNI2-h classifier is undergoing fitting-only checks before full evaluation.

## What goes in and comes out?

Input is one provider case's tissue images, grouped by the available 40×, 100×, 200× and 400× magnifications. Each magnification can contain several fields. Image numbers at different magnifications do not establish physical correspondence.

Step 1 predicts benign or malignant. Benign stops with the pooled benign label. Malignant proceeds to a specialist choosing one of four carcinoma types. There is no benign-subtype specialist. **A routing error counts as a final error**, as does the wrong malignant type. A direct-five control predicts the same five labels in one step.

An encoder turns each image into a numerical description. The retained pipeline uses frozen Midnight descriptions from three crops per original image. It compares these with fitting examples, combines evidence within and across available magnifications, and uses separate fitting-derived reference centers for the two decisions. Label-informed centers are supervised preprocessing. This is an evaluated similarity classifier; novelty has not been established. [Task contract](DUAL_STEP_SCOPE.md), [centering explanation](BREAKHIS_PRIOR_CENTERED_RESULTS.md).

## What data and split do we use?

Only **BreakHis's 7,909 images** supply current task fitting/evaluation data. TCGA work is deferred. External pretrained weights provide transfer learning; they do not add TCGA images to this task.

There are **82 subtype-qualified provider keys**, whose clinical identity relationships remain unresolved. Every trial keeps a key's related fields and magnifications together:

| Role in each trial | Provider cases | Percentage of 82 | Purpose |
| --- | ---: | ---: | --- |
| Fitting | 46 | 56.1% | Learn parameters or construct labeled similarity support |
| Validation | 8 | 9.8% | Select the registered setting and report development results |
| Excluded outer role | 28 | 34.1% | Excluded from that trial's fitting, selection and scoring |

These are **five overlapping repeated holdouts**, each with two existing validation arrangements, rather than disjoint five-fold cross-validation. The 54 original training-role cases become 46 fitting and eight validation cases before benign labels are pooled. Fitting then has 11 benign / 25 ductal / 2 lobular / 5 mucinous / 3 papillary cases; validation has 4 benign / 1 ductal / 1 lobular / 1 mucinous / 1 papillary.

Arrangements 812 and 813 contain 22 and 20 distinct validation keys across trials, with 31 in their union. **All 82 keys have entered development somewhere. The 28 excluded cases in one trial are not a globally untouched final test cohort.** These are provider-case development results, without verified clinical-patient validation.

Every validation case is scored under **all 15 nonempty magnification subsets**. One arrangement has **8 × 15 × 5 = 600 dependent case/subset records**. Accuracy equals correct records divided by 600 because trial sizes are equal. Macro F1 is computed within each subset/trial and then averaged; it is not F1 from one pooled matrix. All-four-view accuracy uses **8 × 5 = 40 dependent records**. Fields, masks and repeated trials do not create new independent cases.

## How do the historical numbers compare?

The notebooks are two different models; their two scores are not arrangements 812 and 813.

| Historical experiment | Target and prediction unit | Split and evaluation count | Accuracy |
| --- | --- | --- | ---: |
| EfficientNetV2B0 + non-local attention notebook | Eight types; one 40× image | Image-random 80/20: 1,596 fitting / 399 reused validation images; provider separation not enforced | 93.73% = 374/399 |
| EfficientNetV2B0 + spatial gate notebook | Eight types; one 40× image | Image-random 80/20: 1,596 fitting / 399 reused validation images; provider separation not enforced | 94.99% = 379/399 |
| Supplied ResNet-50 + XGBoost paper | **Binary** benign/malignant headline; individual images from all magnifications | Author-reported patient-separated approximately 70/30; exact evaluation denominator not independently reconstructed | **86.22% binary**, not final-five accuracy |
| Our earlier frozen Phikon-v2 reference | Eight types; one case under all 15 subsets | 46 fit / 8 validation / 28 excluded per trial; 600 records per arrangement | 45.67% / 38.33% = 274/600 / 230/600 |

The notebooks reuse validation for checkpoint/learning-rate choices and reported evaluation. Their saved confusion matrices were checked; training was not independently reproduced, and exact historical case overlap is unknown. Their roughly 95% scores do not measure today's five-label, grouped-case, missing-view task. The older paper's malignant-type recall is another separate measure. This comparison does not demonstrate a measured fall from 95% to 40% on the same task. [Original-artifact audit](reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md), [historical case ledger](evidence/BREAKHIS_ADAPTATION_RESULTS.json).

## What have our five-category experiments achieved?

Every row below is **final-five, two-step case validation across all 15 subsets**, with **46 fit / 8 validation / 28 excluded cases** per trial and **600 scored records per arrangement**. Scores and correct-record counts are ordered **812 / 813**.

| Completed two-step pipeline | Final accuracy | Correct records, each /600 | Macro F1 |
| --- | ---: | ---: | ---: |
| First frozen Phikon-v2 case means | 65.83% / 68.83% | 395 / 413 | .4975 / .5325 |
| Case-supervised EfficientNet CNN | 46.50% / 53.33% | 279 / 320 | .2346 / .3108 |
| C: frozen Midnight single-center similarity | 75.67% / 75.00% | 454 / 450 | .5897 / .5643 |
| D: Midnight LoRA | 68.33% / 59.83% | 410 / 359 | .4944 / .3754 |
| E: fixed equal-probability ensemble | 74.17% / 72.17% | 445 / 433 | .5781 / .5326 |
| F: frozen Midnight three-crop similarity | 75.50% / 76.83% | 453 / 461 | .5925 / .5799 |
| N: native global input | 60.00% / 61.00% | 360 / 366 | .4394 / .5014 |
| P: class-shared centering | 76.17% / 75.33% | 457 / 452 | .5992 / .5778 |
| P: decision-specific centering | **76.33% / 75.67%** | **458 / 454** | **.6027 / .6025** |
| GenBio frozen similarity | 70.50% / 75.17% | 423 / 451 | .5837 / .5813 |
| Whole-case learned mean | 70.17% / 71.83% | 421 / 431 | .5199 / .5094 |
| Whole-case gated attention | 71.67% / 70.50% | 430 / 423 | .5453 / .5234 |
| UNI2-h frozen similarity | 75.83% / 75.17% | 455 / 451 | .6642 / .5752 |
| Virchow2 frozen similarity | 72.50% / 73.00% | 435 / 438 | .5758 / .5387 |
| H-optimus-0 frozen similarity | 73.50% / 71.17% | 441 / 427 | .5097 / .5062 |

The [full scorecard](../paper/breakhis-five-class-report.md) also retains intermediate field/case controls and all direct-five comparisons. Important qualifications:

- P remains the current reference under the registered paired criteria. UNI2-h leads observed two-step F1 on 812 and its simple two-arrangement F1 average is .619677 versus P's .602614; P has higher 813 F1 and higher accuracy on both arrangements. There is no universal F1 leader. F retains higher 813 accuracy. P's **85% / 85% = 34/40 / 34/40 all-four-view accuracy** is a separate endpoint. P also worsens probability errors versus F.
- GenBio's **direct-five** control scores **71.33% / 77.83%**, F1 **.5960 / .6406**, using 428/600 / 467/600 records under the same roles. Its stronger 813 result is one-stage development evidence, without a consistent two-step gain.
- The approved readers' direct-five controls are also separate one-stage evidence: UNI2-h **75.17% / 77.67%**, F1 **.6486 / .6345**, counts **451 / 466**; Virchow2 **72.83% / 75.00%**, F1 **.5901 / .5877**, counts **437 / 450**; H-optimus-0 **77.33% / 75.17%**, F1 **.6036 / .6067**, counts **464 / 451**, each out of 600. H-optimus-0's direct all-four-view score **87.50% / 82.50% = 35/40 / 33/40** is another separate endpoint.
- LoRA's fixed-C=1 initialization was already **68.17% / 59.50%**; the entire gap from C cannot be attributed to adapter damage. Eight of ten dual runs selected the head-only epoch-2 checkpoint.
- Whole-case neural models fit their fitting cases at about **99.13%** accuracy but transfer much less well. This does not establish one sole cause. Learned mean also has slightly better Brier error than F despite weaker accuracy/F1, so every metric did not worsen.

Source ledgers: [first five-class](evidence/BREAKHIS_FIVE_CLASS_RESULTS.json), [CNN](evidence/BREAKHIS_CASEBAG_RESULTS.json), [C](evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), [LoRA](evidence/BREAKHIS_LORA_RESULTS.json), [ensemble](evidence/BREAKHIS_FIXED_ENSEMBLE_RESULTS.json), [F](evidence/BREAKHIS_COVERAGE_RESULTS.json), [native](evidence/BREAKHIS_NATIVE_RESULTS.json), [P](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json), [GenBio](evidence/BREAKHIS_GENBIO_RESULTS.json), [whole-case mean/attention](evidence/BREAKHIS_CASE_MIL_RESULTS.json), [UNI2-h](evidence/BREAKHIS_UNI2_H_RESULTS.json), [Virchow2](evidence/BREAKHIS_VIRCHOW2_RESULTS.json), [H-optimus-0](evidence/BREAKHIS_H_OPTIMUS_0_RESULTS.json). Table accuracy, F1, denominators and checkpoint selections were independently recalculated from these saved ledgers; no new fitting or inference was performed for this explanation.

## What counts as success, and what remains?

The registered descriptive improvement screen requires **at least +0.02 macro F1 on both arrangements**, no accuracy decrease on either, and positive recall for all five labels, against each study's declared fixed reference. C passed its original screen; F, both P modes, GenBio, both whole-case models and all three approved readers failed theirs. Passing is development evidence, without independent confirmation or proof of novelty. A technical collector passing means saved outputs satisfy its checks; it does not mean predictive improvement succeeded.

P's binary router scores **94.83% / 93.00%**, while its specialist's accuracy on truly malignant cases is **58.33% / 62.33%**. These answer different questions from final-five accuracy. Mucinous and papillary recognition remains weak. More images from a fitting case cannot supply additional independent cases, and binary accuracy cannot replace subtype performance.

The completed [approved-reader comparison](BREAKHIS_APPROVED_ENCODERS_SPEC.md) preserves the same endpoint and direct-five controls. UNI2-h improves malignant typing to **72.00% / 60.67%**, while its router scores **83.33% / 90.33%**; its 812 F1 gain comes with a routing tradeoff. Even a perfect router with those current conditional predictions fixed gives **86.00% / 80.33%**, not 95%. The bound does not apply to a different representation or pooling rule. Virchow2 and H-optimus-0 also have substantial subtype/routing errors. [Actual model diagnosis](reviews/APPROVED_ENCODER_ACTUAL_MODEL_DIAGNOSIS.md).

All three readers and both arms remain visible. Selecting among them adds **12 reader/temperature settings per arm/role**, rather than four, and no reader is chosen separately for each arrangement to manufacture a combined headline. Their mixed accuracy, F1 and probability-quality outcomes do not establish one universal winner or authorize an automatic hybrid.

The five-category capstone report, presentation and demo remain the immediate deliverable. Completed negative results stay visible. New fitting must not change splits or tune routing thresholds to manufacture 95%. TCGA expansion and publication claims remain separate, deferred work.
