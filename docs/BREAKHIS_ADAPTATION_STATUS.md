# BreakHis adaptation: completed, no meaningful gain

Updated November 12, 2026. All ten production runs 4962271 completed successfully. Collector 4962297 passed the complete artifact comparison in30 seconds. First training start to checked completion:25minutes11seconds. No MagSet adaptation job remains active.

| Split | Frozen mean case accuracy | Adapted mean case accuracy | Frozen macroF1 | Adapted macroF1 |
| --- | ---: | ---: | ---: | ---: |
|812|45.7%|45.7%|.3843|.3840|
|813|38.3%|38.8%|.3169|.3205|

Scores average15 view combinations and five overlapping trials per split. The registered requirement to improve primary F1 on both splits **failed**. Phyllodes recall remains 0. Keep the frozen baseline as the default; no novel-method success is claimed.

[Complete results and all ten trials](BREAKHIS_ADAPTATION_RESULTS.md) · [Public benchmark context](BREAKHIS_PUBLIC_BENCHMARKS.md) · [Registered recipe](BREAKHIS_ADAPTATION_SPEC.md) · [Independent review](reviews/BREAKHIS_ADAPTATION_REVIEW.md).

## What passed

The separate fitting-only technical smoke4962114 passed in59seconds. All 70 software tests and CI passed. Production verification checked the fixed six-epoch recipe, only allowed training cases/layers, canonical tensor and AdamW states/update counts, checkpoint/cache/head/source bindings, exact54-case caches, fitting-only normalization, saved-model probability replay, labels and every subset metric. Independent local auditing agrees with all 10 runs and both aggregate tables. No outer predictions or replacement configurations were produced.

Immutable training source: `3a8dde70d809b115928bf773511a0cd6e0f8e30c`. Immutable collector source: `d5567ee402dd11d75b295b33f16b1eebb5799f61`. [Execution ledger](evidence/BREAKHIS_ADAPTATION_EXECUTION.json) · [Complete result/hash ledger](evidence/BREAKHIS_ADAPTATION_RESULTS.json).

## What happens next

Stay on BreakHis. The [saved-artifact diagnosis](BREAKHIS_DIAGNOSIS_RESULTS.md) is complete and finds a large generalization gap in both heads; neither auxiliary readout rescues validation. Next audit split/class coverage and specify a compatible supervised image-model baseline before registering another training recipe. No automatic longer training, model sweep, outer scoring or return to TCGA follows this result.

Current accuracy is concerning for a strong classifier. Only 46 cases fit each model, including one adenosis and one phyllodes case; repeated images cannot supply additional independent variety. The training image loss falls strongly while case validation barely changes, but this does not prove a single failure cause. All-view accuracy is also weak. Clinical grouping and clean confirmation remain unresolved, so no honest ETA to 80% or 90% accuracy can be given.
