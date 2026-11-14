# Frozen regional-view comparison: negative result

November14,2026. **All four feature shards and20 fixed-C heads completed; independent cache and numerical reviews passed. The improvement gate failed. Keep the historical frozen reference.**

One fixed conventional preprocessing experiment compared the original Phikon central-view cache with a50/50 blend of that vector and the mean of four regional vectors. The same frozen reader and official processor were used. The task, roles, feature dimension, normalization policy and fixedC0.01 were matched.

## Matched result

| Arrangement | Fixed-C control F1 | Regional F1 | Change | Control accuracy | Regional accuracy | Regional phyllodes recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3463 | 0.3405 | -0.0058 | 43.00% | 42.00% | 0.0% |
| 813 | 0.2808 | 0.2867 | +0.0059 | 34.33% | 34.67% | 0.0% |

These average all15 view combinations and five overlapping trials. The mixed tiny difference fails the≥+0.02-F1-on-both-arrangements and positive-phyllodes-recall screen. The original retained reference scoresF1.3843/.3169 and accuracy45.7%/38.3%, with historical C-grid selection; it is reported separately because its tuning budget differs.

## Every fixed head

| Arm | Arrangement | Trial | Validation macro F1 | Validation accuracy | Fitting balanced accuracy | All-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| thumbnail | 812 | 1 | 0.6042 | 66.67% | 95.90% | 75.0% |
| thumbnail | 812 | 2 | 0.2144 | 25.83% | 96.26% | 25.0% |
| thumbnail | 812 | 3 | 0.3089 | 40.83% | 98.53% | 37.5% |
| thumbnail | 812 | 4 | 0.2597 | 35.00% | 95.89% | 37.5% |
| thumbnail | 812 | 5 | 0.3444 | 46.67% | 97.96% | 50.0% |
| thumbnail | 813 | 1 | 0.3444 | 40.83% | 98.22% | 37.5% |
| thumbnail | 813 | 2 | 0.2458 | 29.17% | 95.90% | 25.0% |
| thumbnail | 813 | 3 | 0.3917 | 46.67% | 97.03% | 50.0% |
| thumbnail | 813 | 4 | 0.2958 | 40.00% | 94.07% | 37.5% |
| thumbnail | 813 | 5 | 0.1264 | 15.00% | 98.54% | 12.5% |
| multicrop | 812 | 1 | 0.6153 | 68.33% | 96.47% | 75.0% |
| multicrop | 812 | 2 | 0.2194 | 25.00% | 96.54% | 25.0% |
| multicrop | 812 | 3 | 0.2914 | 39.17% | 98.63% | 37.5% |
| multicrop | 812 | 4 | 0.2167 | 30.00% | 96.26% | 37.5% |
| multicrop | 812 | 5 | 0.3597 | 47.50% | 98.68% | 50.0% |
| multicrop | 813 | 1 | 0.3561 | 41.67% | 97.96% | 37.5% |
| multicrop | 813 | 2 | 0.2533 | 29.17% | 96.45% | 25.0% |
| multicrop | 813 | 3 | 0.3625 | 43.33% | 97.25% | 50.0% |
| multicrop | 813 | 4 | 0.2972 | 40.00% | 95.27% | 37.5% |
| multicrop | 813 | 5 | 0.1644 | 19.17% | 98.82% | 12.5% |

## Execution and checks

The fitting-only smoke4971319 completed in21s, cache parity1.385807991e-6 and exact blend replay; its error log was empty. Feature tasks4971322 all completed in134/137/125/115s with empty logs, covering7,909 original fields and31,636 region forwards. No backbone weights were fitted. CPU job4971323 completed all20 heads in466s with an empty error log. All actual scientific outputs use source `47ad41657979c3e154cc9db7b89f1fec4b1f40ac`, config SHA256 `42e8e98e1310c7d9daf339c2f0376d2abb6fcc270dbb350def389318402498af`.

Independent four-cache auditing verified all paths/raw hashes/dimensions/bounds, exact original vectors and standalone float64-to-float32 blend arithmetic. All20 saved normalizers reconstructed bitwise from fitting-only field moments. Saved coefficients, exact roles, all16,200 fitting/validation case-subset records and30,850 metric checks passed independent contraction/metric replay; probability error≤1.22e-15 and metric error≤6.66e-16. [Cache review](reviews/BREAKHIS_MULTICROP_CACHE_REVIEW.md) · [Head/result review](reviews/BREAKHIS_MULTICROP_RESULTS_REVIEW.md).

The original validation code repeatedly inflated compressed arrays and was slow. The job nevertheless finished its full20-head budget before any stop attempt; it remainedCOMPLETED0. A later bounded-read performance fix and optional-Torch test fix are not the source of these outputs. No feature extraction or fitting was repeated, no scientific settings changed, and no tolerance was relaxed.

Comparison SHA256 `13e547269d9c5dc9d0ab3b867b6cba36e068501d14f80109f9b8e4fa17c7164b`. [Complete identifier-free aggregate ledger](evidence/BREAKHIS_MULTICROP_RESULTS.json) retains every run/subset/class/confusion/loss and provenance hash. [Registered recipe](BREAKHIS_MULTICROP_SPEC.md).

This negative result does not prove that image context/resolution never matters. It rejects this one fixed representation under the reused provider-case development regime. Sparse rare-class cases, uncertain clinical grouping and no fresh confirmation cohort remain. Freeze the capstone model and finish the report/demo; no crop/encoder/weight/C sweep follows.
