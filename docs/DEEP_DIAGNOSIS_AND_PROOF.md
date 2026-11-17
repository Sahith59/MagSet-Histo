# Deep diagnosis and proof of the reported runs

November 17, 2026. **The original three results are reproducible under the stated BreakHis protocol.** The extra raw-image verification found no numerical scoring discrepancy. The scores describe this small reused-development experiment; reproducibility does not establish clinical accuracy or independent patient identity.

| Original method | Correct final answers / 240 case/subset events | Final-five accuracy | Correct binary answers / 240 | Binary accuracy |
| --- | ---: | ---: | ---: | ---: |
| Midnight P | 162 | 67.50% | 227 | 94.58% |
| CNN–Transformer | 136 | 56.67% | 207 | 86.25% |
| Swin | 142 | 59.17% | 198 | 82.50% |

Every row uses the same 16 provider cases, 13 conservative identifier groups and 15 nonempty magnification subsets. These are 240 dependent predictions of 16 cases, not 240 independent specimens. Macro F1 is computed separately for each subset and then averaged; it is not accuracy or the F1 of a pooled confusion matrix.

## What the new proof actually executed

- **CNN and Swin:** separately reopened all 1,704 original test PNGs per model; checked every file hash, filename/directory label and crop; rebuilt the selected checkpoint; and independently composed probabilities, pooled case/views and calculated scores. Every field probability matched exactly, with zero changed case decisions. The frozen architecture builder is shared; image I/O, crop construction and score arithmetic are separate executions. [Neural raw-replay evidence](evidence/MATCHED_NEURAL_RAW_PROOF.json).
- **Midnight P:** reopened all **7,909** PNGs, including every fitting support, validation and test field; re-encoded all **23,727** crops through the frozen original reader; and rebuilt the centers, cosine densities, case/class/field weighting and pooling with an independent NumPy implementation. Maximum feature difference was **1.49e-6**, within the original feature-replay tolerance. Maximum field-probability difference across the four validation settings was **8.56e-7**; selected-test field difference was approximately **7.77e-16**. Every original case decision, score and validation selection remained unchanged. [All-field raw P proof](evidence/MATCHED_MIDNIGHT_RAW_PROOF.json).
- **Source and execution:** independent audits matched original archive/code/configuration/cache/checkpoint hashes, exact fitting weights, pinned split implementation, saved candidate selection and actual scheduler chronology. Both selections and test execution were checked rather than inferring success from an exit code. [Original forensic audit](reviews/MATCHED_PROOF_AUDIT.md) · [completed actual proof review](reviews/DEEP_PROOF_RESULTS_REVIEW.md).

The raw proof involved 11,317 image visits—7,909 unique images, with the neural test images revisited—and 33,951 crop evaluations. It did not create new cases or performance estimates. No model parameter, split, score definition or tolerance was changed to obtain a higher reported accuracy.

The independent reviewer verified every new saved-array calculation and receipt. The large raw P feature archives remain on the cluster; their complete byte and feature comparisons were executed by the hash-bound CPU checker, rather than mirrored locally. The newer UNI2 specialist uses its previously verified frozen feature cache; its encoder was not part of this original-three-model raw replay.

## What appears to be going wrong

**No demonstrated label, scaling, loss-mask, save/reload or scoring bug was found in these matched runs.** This is a result of the checks performed, not proof that every possible implementation issue is impossible.

The strongest measured signal is poor transfer of subtype decisions to different case groups. CNN’s last online training binary/malignant losses are about .0217/.0928, while its balanced validation-field cross-entropies are about 1.54/4.06. Swin has a similar gap. These use different training/evaluation modes and are not a comparison of fitting accuracy. Many wrong neural subtype predictions are highly confident. [Detailed training diagnosis](reviews/MATCHED_TRAINING_DIAGNOSIS.md).

There are only **two lobular fitting cases** and four papillary fitting cases. More fields or masked examples from those cases do not create more independent patients. The original test has just one lobular and one papillary case; all three methods miss those cases in the evaluated conditions. These observations support investigating subtype transfer and case-level learning. They do not establish mislabeled tissue, a stain defect or a resolution defect without appropriate original-field review.

For P, the 78 wrong final predictions split into **13 binary-routing errors and 65 malignant-type errors after correct routing**. Improving only the first decision cannot remove those 65 typing errors. Conversely, a frozen first decision cannot raise its binary accuracy above 94.58%.

## The targeted specialist experiment

The first bounded intervention keeps P’s binary decision fixed and compares its malignant expert with a frozen UNI2-h expert. Both receive four predeclared malignant-temperature choices, fitted statistics use only the same 49 training cases, and both validation selections are sealed before the two selected retrospective follow-ups.

| Method | Validation accuracy | Validation macro F1 | Retrospective follow-up accuracy | Follow-up macro F1 |
| --- | ---: | ---: | ---: | ---: |
| P specialist control | 79.61% | .577663 | 67.50% | .375675 |
| UNI2-h malignant specialist | 77.25% | .590347 | 82.92% | .663685 |

**This is a mixed result.** The specialist improves the already-viewed follow-up by 37 of 240 final answers, but validation loses six of 255 answers. Its validation macro-F1 gain, .012684, is below the registered .02 screen, and validation accuracy falls. Therefore the registered improvement screen **fails**. We report the follow-up gain, but do not promote the candidate based on that gain or describe it as fresh confirmation. Binary accuracy stays **94.58%** by construction. [Registered plan](BREAKHIS_SPECIALIST_READER_PLAN.md) · [complete checked evidence](evidence/BREAKHIS_SPECIALIST_READER_RESULTS.json).

## The binary diagnosis

A separate diagnostic excludes the **entire conservative group** for each fitting query before recalculating any support statistic. It uses the fixed P setting and makes no new test query, threshold search or calibration fit.

| Role | Cases | Binary accuracy | Mean subset ROC AUC |
| --- | ---: | ---: | ---: |
| Fitting queries with whole-group exclusion | 49 | 87.62% | .9521 |
| Existing validation role | 17 | 93.33% | 1.0000 |

The validation ranking is stronger than its fixed-threshold accuracy, suggesting that calibration is worth investigating. Training-group-out ranking is imperfect, and the role is small; calibration alone is not established as a transferable rescue and does not repair the malignant expert. [Complete binary diagnostic](evidence/BREAKHIS_ROUTER_GROUP_DIAGNOSIS.json).

## How the next bounded test addresses both decisions

The next registered development probe learns from whole-case summaries rather than assigning a subtype lesson to each individual field. It compares compact linear and RBF SVCs on the same frozen UNI2 features, with the same four C choices per arm, training-only normalization and true benign/malignant → malignant-four routing. This can change both decisions. It remains a hypothesis test, with a strong linear control and the previous case-learning failures retained as counterevidence. The fixed budget must not expand after outcomes are seen. [Case-SVC plan](BREAKHIS_UNI_CASE_SVM_PLAN.md).

**Over 90% final-five and over 97% binary accuracy remain unachieved objectives.** Neither a larger model nor more GPU time guarantees either score. All subsequent follow-ups on the already-viewed cases are development observations. Independent new cases would be needed for fresh confirmation before a strong research-generalization claim.
