# Completed Midnight adapter comparison

November16,2026. All20 A40 runs and the complete output checker finished successfully. The adapter recipe did **not** beat our retained similarity model. Current best remains **75.67%/75.00%** final five-category accuracy.94–95% remains unachieved.

| Two-step method | Accuracy812 /813 | Macro F1812 /813 |
| --- | ---: | ---: |
| Retained frozen Midnight similarity model |75.67% /75.00%|.5897 /.5643|
| Frozen Midnight case logistic heads |72.00% /71.33%|.5558 /.5350|
| Frozen Midnight validation-selected field heads |73.67% /70.00%|.5732 /.5017|
| Fixed C=1 field initialization for this adapter recipe |68.17% /59.50%|.4870 /.3682|
| Completed adapter-recipe two-step |68.33% /59.83%|.4944 /.3754|
| Adapter-recipe direct-five control |71.33% /67.50%|.5101 /.4147|

These average the same five trials and15 nonempty zoom settings per arrangement. They are reused provider-case validation scores. The fixed C=1 row is reconstructed from the already accepted frozen candidate ledger, not a newly fitted control. Its role is to explain the actual starting point.

## What we learned

The selected two-step recipe improves only0.17/0.33 raw percentage points over its fixed C=1 initialization. The initialization was already substantially below the validation-selected frozen heads. It would be incorrect to attribute the whole gap from75% to60% to adapter damage.

Eight of ten two-step runs selected epoch2, before adapter training starts; all five arrangement813 runs did so. Across both arms,12of20 selected epoch2, three selected6, four selected12 and one selected20. The results provide little evidence that the learned adapters help this recipe.

Selected fitting accuracy is99.13% on both arrangements, while two-step validation is68.33%/59.83%. The original C=1 heads already had the same fitting accuracy. This is a large generalization gap, rather than failure to fit the examples. There are only two lobular and three papillary fitting provider cases per role.

Actual router accuracy is92.67%/90.17%, but true-malignant typing is51.00%/39.00%. Final lobular recall is21.33%/20.00% and papillary recall20.00%/12.00%. Most remaining errors occur after a correct malignant routing decision. Better binary recognition alone cannot deliver95% final accuracy with these subtype predictions.

The next bounded tests examine complementary frozen-model evidence and image coverage. A fixed equal-probability mixture uses the accepted similarity and case-logistic models without selecting a weight. A separate three-crop frozen-reader comparison examines left/center/right coverage. These are unmeasured hypotheses, not novelty claims or95% forecasts.

## Execution and verification

Producer `7daff480a44b35f57ca859390a08d447cdd9df46`; registered config SHA256 `d4559a4b9e559f422f06b5a6afcbdfd347eb72a1206456826df40671957488c5`. All20 array4980370 tasks completed successfully in14m51s–17m14s. Collector4980371 completed successfully in84seconds, with no error output. The source had277 passing pinned tests and successful CI.

The checker verified20 runs,80 candidate checkpoints,9,600 candidate validation case/subset records and13,800 selected fitting records. It replayed saved adapted features/heads, original role coverage, fixed normalizers, hard routing, metrics, selection, checkpoint and optimizer bindings. Producer receipts recorded actual initial and fresh-reconstruction image probes and unchanged original weights. CPU replay does not independently re-execute the adapted image encoder.

Root independently checked20 selections and12 aggregate means. Additional independent reviews checked all80 candidates and source/normalization/objective consistency. No invalidating label, split, stage or scoring mismatch was found. The evidence supports a weak generalization result for this fixed combined initialization/augmentation/head/adapter recipe.

[Complete accepted JSON](evidence/BREAKHIS_LORA_RESULTS.json), SHA256 `570892f048f3b8bdbf0095ed6c5dcdd4c5e6736ac2af86fe48d0db089e28072e` · [Fixed recipe](BREAKHIS_LORA_SPEC.md) · [Actual diagnosis](reviews/LORA_ACTUAL_RESULTS_DIAGNOSIS.md) · [Validity review](reviews/LORA_ACTUAL_VALIDITY_REVIEW.md) · [Retained best](BREAKHIS_KERNEL_RESULTS.md).

Only BreakHis images/labels were fitted/evaluated.46 fitting/eight validation/28 excluded provider cases per run remain unchanged, with31 distinct validation keys across overlapping trials/arrangements. Clinical identifier relationships remain unresolved, and all82 provider keys entered development somewhere. Repeated fields/masks are dependent. Neither fresh clinical generalization nor a novel mechanism is established.
