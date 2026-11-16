# Completed complementary-reader and case-supervision round

November 16, 2026. Three parallel agents completed data diagnosis, training diagnosis, accessible-model research and independent reviews. Both new experiments are now complete and accepted. **94–95% final five-category accuracy remains unachieved.**

| Two-step pipeline | Accuracy 812 / 813 | Macro F1 812 / 813 | Decision |
| --- | ---: | ---: | --- |
| Retained head-specific center, P |76.33% /75.67%|.6027 /.6025|Highest two-step macro F1; retain |
| Earlier three-crop KDE, F |75.50% /76.83%|.5925 /.5799|Higher 813 raw accuracy; retain comparison |
| New GenBio KDE |70.50% /75.17%|.5837 /.5813|No consistent gain |
| New whole-case mean |70.17% /71.83%|.5199 /.5094|Below retained references |
| New whole-case attention |71.67% /70.50%|.5453 /.5234|F1 rises over learned mean; below references |

These scores average all 15 zoom combinations and five overlapping trials per arrangement. P’s 85%/85% with all four zooms is a separate endpoint. GenBio's direct-five control scores 71.33%/77.83%, F1 .5960/.6406; its stronger 813 one-stage result is neither a two-step score nor a uniform accuracy improvement. Keep every declared row; do not mix different models' best arrangements.

The learned models fit the fitting cases at approximately 99.1% accuracy and .982 macro F1, but transfer poorly to validation cases. The fixed GenBio specialist improves true-malignant typing to 60%/67.67% while weakening binary routing to 82%/89.83%. Even perfect routing with those fixed conditional type predictions reaches only 80%/83.83%. These findings identify the next question as malignant case generalization, without proving one biological or implementation cause.

## Completed execution

- Frozen GenBio preparation 4983043 verified all six official artifacts. Initial GPU check 4983050 exposed batch-shape sensitivity before publishing features. The canonical one-image-call correction retained the official representation and unchanged tolerance. GPU check 4983063 passed repeat/host-partition/fresh-reload parity with zero differences, 4.65GB peak memory and .404seconds per steady four-image host batch.
- All four GenBio magnification tasks 4983078 completed in 3m54–4m28. Independent audit 4983079 accepted all 7909 original images/23,727 channel examples in 7 seconds. Classifier audit 4983121 accepted all ten original roles in 9 seconds. All ten paired CPU runs 4983141 completed in 35–38seconds; collector 4983142 completed in 4m46 with empty error output, checking 80 candidates/160 prediction archives/64,800 dependent records.
- Four fitting-only MIL GPU checks 4983061 passed, followed by CPU acceptance 4983062. All 40 production trajectories 4983086 completed in 32–47seconds each; collector 4983087 completed in 3m06 with empty error output, independently replaying 160 checkpoints/320 prediction archives/129,600 dependent records in NumPy FP64. No excluded predictions or new encoder execution occurred during MIL.
- Immutable learning/feature source `6861fbfe8d847f16fa09b4f839193716c1e8d501` passed all 370 neural tests in [CI](https://github.com/Sahith59/MagSet-Histo/actions/runs/43904610827), plus the lightweight job. The redundant cluster suite 4983059 was stopped after exact-source CI passed. An old synthetic scalar-oracle test received a magnitude-scaled FP64 rounding tolerance; no production math, collector threshold or prior result changed.
- GenBio classifier source `dfaa6c190f2ae8a32028aea6ba70ca82154ee6f6` only binds completed feature hashes and status documents; its three binding checks passed. Actual cache and classifier artifacts match their accepted hashes.

Root independently checked 344 MIL and 68 GenBio selection/aggregate/recall calculations. Independent reviewers checked 268 MIL identities and 1,237 GenBio arithmetic/provenance assertions, including all 2,400 GenBio candidate subset confusion matrices. Separate history analysis verified 184,000 fitting-case visits and 48,000 optimizer steps in MIL. The model-diagnosis author identifies their authorship of the trainer; the independent validity reviews are separate.

No project computation remains active. No predictor fusion, threshold search, resplit or post-hoc checkpoint was added. Source/runtime/candidate budgets and all negative outcomes are retained.

## Next action

Freeze the accepted evidence and finish the five-category capstone presentation/demo. Additional predictive work needs a separately registered specialist-generalization hypothesis, not automatic architecture escalation. Access to the gated UNI2-h, Virchow2 and H-optimus0 models has not been confirmed for this project. No new access application, terms acceptance or personal-information submission occurred.

[GenBio complete results](BREAKHIS_GENBIO_RESULTS.md) · [MIL complete results](BREAKHIS_CASE_MIL_RESULTS.md) · [Registered recipe](BREAKHIS_NEXT_ROUND_SPEC.md) · [Data diagnosis](reviews/NEXT_ROUND_DATA_DIAGNOSIS.md) · [Model search](reviews/NEXT_ROUND_MODEL_SEARCH.md) · [MIL validity review](reviews/CASE_MIL_ACTUAL_VALIDITY_REVIEW.md) · [GenBio validity review](reviews/GENBIO_ACTUAL_VALIDITY_REVIEW.md).
