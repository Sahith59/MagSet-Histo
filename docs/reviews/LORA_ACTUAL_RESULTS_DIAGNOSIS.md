# Midnight LoRA: actual-results diagnosis

November 16, 2026. The complete LoRA comparison is verified, but it does not improve the retained KDE result. Two-step accuracy is **68.33% / 59.83%**, macro F1 **0.4944 / 0.3754**, versus KDE **75.67% / 75.00%**, F1 **0.5897 / 0.5643**. Keep KDE as the best verified two-step option. Neither result reaches 94–95%.

This read-only diagnosis checked accepted ledger SHA256 `570892f048f3b8bdbf0095ed6c5dcdd4c5e6736ac2af86fe48d0db089e28072e`, source `7daff480a44b35f57ca859390a08d447cdd9df46`, LoRA configuration `d4559a4b9e559f422f06b5a6afcbdfd347eb72a1206456826df40671957488c5`. Ninety-two checks independently reconstructed selected-epoch choices, selected fitting/validation confusion totals and mean accuracy/F1/balanced accuracy. Private calculations remain under `.local/lora-result-diagnosis/`. No new fitting, GPU execution, threshold search, excluded predictions or source changes occurred. Accepted collector replay covers cached features and heads; independent raw-image adapted-encoder reproduction remains unperformed.

## What improved, and what failed

All entries are end-to-end five-category development accuracy / macro F1, averaged over five overlapping trials and all 15 magnification subsets.

| Pipeline | Arrangement 812 | Arrangement 813 |
| --- | ---: | ---: |
| Frozen Midnight case two-step | 72.00% / 0.5558 | 71.33% / 0.5350 |
| Frozen Midnight KDE two-step | 75.67% / 0.5897 | 75.00% / 0.5643 |
| Fixed-C1 field two-step initialization | 68.17% / 0.4870 | 59.50% / 0.3682 |
| Selected LoRA two-step | 68.33% / 0.4944 | 59.83% / 0.3754 |
| Selected LoRA direct-five | 71.33% / 0.5101 | 67.50% / 0.4147 |

The initialization matters. LoRA started from the prospectively fixed C=1 field heads, **not** the validation-selected heads used in the stronger published Midnight comparison. Relative to its actual fixed initialization, two-step accuracy improves by only **0.17 / 0.33 percentage points**, and F1 by **0.0074 / 0.0072**. Validation cross-entropy improves slightly, from 0.9302/0.8864 to 0.9256/0.8836. Calling the entire difference from KDE a degradation caused by LoRA would overstate the evidence: much of the weakness already exists in C1 before adaptation.

Selected fitting accuracy is **99.13%** and balanced accuracy **99.68%** for both arms and arrangements. The fixed C1 heads already had that same fitting accuracy. The large generalization gap persists; increasing fitting capacity was not the missing achievement.

## Checkpoint evidence

Eight of ten two-step runs select epoch 2: all five on arrangement 813 and three on 812. Under the registered schedule, epoch 2 completes head-only warmup **before adapters learn**. The other two choose epochs 12 and 20. The five-trial two-step accuracy trajectory at epochs 2/6/12/20 is:

- 812: **68.17 / 68.17 / 67.83 / 68.17%**.
- 813: **59.83 / 58.83 / 59.17 / 59.17%**.

Later adaptation supplies no consistent validation recovery. Every selected two-step trial scores below its matched KDE trial in accuracy and F1. This is broader than one unlucky trial or selection of the final epoch. It supports retaining the frozen support classifier, while separating the initialization, head update, augmentation and adapter contributions; the registered study changed that package together.

## The remaining errors are mostly malignant typing

Two-step binary accuracy is **92.67% / 90.17%**, whereas conditional typing accuracy on all truly malignant validation cases is **51.00% / 39.00%**. Final five-class accuracy counts both decisions.

| Two-step recall | LoRA 812 / 813 | KDE 812 / 813 |
| --- | ---: | ---: |
| Benign | 85.67 / 81.00% | 97.00 / 96.00% |
| Ductal | 98.67 / 81.33% | 66.67 / 44.00% |
| Lobular | 21.33 / 20.00% | 90.67 / 93.33% |
| Mucinous | 64.00 / 41.33% | 41.33 / 41.33% |
| Papillary | 20.00 / 12.00% | 18.67 / 37.33% |

Across each arrangement's 600 dependent validation events, routing contributes **44/59** errors and malignant wrong-subtype decisions **146/182**. Lobular, mucinous and papillary account for **76.84%/70.54%** of final errors; 139/126 rare-type events are called ductal. Their fitting recalls are 100%, but each run has only two lobular and three papillary fitting cases.

Even a hypothetical perfect router with the current LoRA subtype predictions reaches only **75.50%/69.50%**. A perfect specialist with its current router is capped at **92.67%/90.17%**. These are arithmetic diagnostics, not fitted models. A routing-threshold adjustment alone cannot establish the desired 95% five-class result.

## Best next bounded hypothesis

**Register one fixed, equal-weight joint-probability blend of the existing frozen KDE and frozen Midnight case pipelines**, with matching roles and subsets, then derive its binary route and conditional malignant choice from the blended five-vector. Include their corresponding direct-five blend. Use no per-class weights, threshold search, favorable-case selection or new encoder execution.

The hypothesis is specific: KDE strongly recognizes lobular cases, while frozen Midnight case pooling has higher ductal recall, 85.33/74.67% versus KDE 66.67/44.00%. A fixed blend may preserve some complementary evidence. Class-level recall differences motivate this test; they do not prove that a mixture improves individual cases. Both constituents already used validation for selection, so this remains another explicitly disclosed development comparison, not independent confirmation. Retain every outcome and compare against KDE on both arrangements. The evidence supports this cheaper test more directly than another capacity increase or longer LoRA schedule, and supplies no 94–95% forecast.

The unchanged protocol has 82 qualified provider keys, 46 fitting/eight validation/28 excluded per run, four magnifications and 31 distinct validation keys across the reused arrangements. Clinical identity remains unresolved; excluded roles are not a globally fresh cohort. Execution success, exploratory improvement and clinical generalization remain separate claims.

References: [Midnight comparison](../evidence/BREAKHIS_MIDNIGHT_RESULTS.json), SHA256 `c9fdd48763e17c2458aa375797827842d1695e6bdefa3fa5354ca1317bc235fe`; [KDE comparison](../evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), SHA256 `e6e98fc26b3579261baf39b677e75551f7bbfbc5f3a7fe399db59c7169511274`.
