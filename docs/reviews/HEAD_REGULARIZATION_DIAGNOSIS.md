# Saved-head regularization diagnosis

November 15–16, 2026. Allowing different router and malignant-head regularization settings gives a modest posthoc gain on arrangement 812 and a small gain on 813. The saved settings still leave the rare malignant classes weak.

| Arrangement | Registered four shared-C choices, selected accuracy / F1 | Posthoc sixteen router/expert pairs, selected accuracy / F1 | Accuracy change | F1 change |
| --- | ---: | ---: | ---: | ---: |
| 812 | 65.83% / 0.497517 | 67.50% / 0.516756 | +1.67 percentage points | +0.019238 |
| 813 | 68.83% / 0.532497 | 69.00% / 0.532942 | +0.17 percentage points | +0.000444 |

Each column selects by mean-15 end-to-end five-class macro F1 within each run, then averages the five trials. The larger search improves F1 in two of five 812 runs and one of five 813 runs. The main change comes from 812 trial 1, where router C=1 and expert C=0.01 improve F1 by 0.094286 and accuracy by 7.50 points. The other gains are 0.001905 F1 / 0.83 accuracy points on 812 trial 5 and 0.002222 F1 / 0.83 points on 813 trial 3. Seven runs retain the registered F1 value. When a tie admits different head settings, the earliest pair in ascending router-C then expert-C order is retained.

Choosing each run's highest accuracy among all sixteen pairs instead gives mean accuracy 67.67% / 69.00%. This supplies an optimistic upper bound for selecting one complete pair per run from these saved settings on this validation endpoint.

The following recall ceilings allow a different pair for each class and each run. They average each run's maximum recall across the five trials. Such choices use validation labels separately for each class. Deployment requires one fixed complete pipeline.

| Malignant label | Arrangement 812 maximum recall | Arrangement 813 maximum recall |
| --- | ---: | ---: |
| Lobular | 56.00% | 54.67% |
| Mucinous | 65.33% | 60.00% |
| Papillary | 13.33% | 18.67% |

These classwise ceilings are identical for the original four shared-C settings and the expanded sixteen pairs. Even allowing a separate choice for every saved case/subset event yields the same malignant-class ceilings here. The result scopes the effect of coupling within C values `[0.01,0.1,1,10]`. Broader regularization ranges or different model families remain unmeasured by this calculation.

## Calculation and verification

The analysis uses all forty accepted dual candidate validation files from the original frozen-Phikon comparison. It aligns the 120 case/subset records per candidate by exact saved provider key and magnification tuple, checks identical truths and roles, and verifies each prediction-file hash against its accepted candidate metadata. All ten original diagonal selections and aggregate accuracy/F1/balanced accuracy reproduce within 1e-12. The accepted comparison is bound by SHA256 `bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795`. [Accepted comparison](../evidence/BREAKHIS_FIVE_CLASS_RESULTS.json)

For each router/expert combination, binary log probabilities come from the router candidate and conditional malignant log probabilities come from the expert candidate. The composed five-class log distribution is `[log P(B), log P(M)+log P(type|M)]`. Explicit final predictions route malignant when the saved router probability is at least 0.5, then choose the first malignant argmax. The existing reviewed five-class metric implementation calculates complete end-to-end summaries with finite log probabilities. The study preserves all original roles, all fifteen subsets and all ten runs.

This is a posthoc hyperparameter-coupling diagnostic with sixteen selection opportunities per run. The larger search increases validation-selection optimism. It performs zero fits, encoder forwards, downloads or excluded-case predictions. The measured changes support a limited contribution from the shared-C constraint under these saved settings. Scores above measure reused-validation selection behavior, with gains small relative to the gap to 94–95% accuracy.

The private `.local/head-regularization` directory retains the calculation script, all sixteen pairs per run, classwise oracle calculations and input hashes. Script SHA256 is `d792ead287ab19466bdebf2c89cb410bcbc091ee91bdbe19d0ff8b5d2a6cb31c`. Complete diagnostic JSON SHA256 is `1b3b25e3d2cfd471e6711b61d459b8da9375fd70f89c409998fb8a05f19302e0`.
