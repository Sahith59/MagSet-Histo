# Validation error analysis

Computed from all 30 verified validation runs. Outer-test predictions were not used. Each trial has eight validation cases, and trials can reuse a case. Repeated records below are descriptive; they do not increase the independent sample size.

## Available views

Each cell is mean macro F1 across subsets of that size and the five trials.

| Encoder / fusion | One view | Two views | Three views | Four views |
| --- | ---: | ---: | ---: | ---: |
| phikon-v2 / concat | 0.2942 | 0.3522 | 0.3896 | 0.4208 |
| phikon-v2 / late | 0.3183 | 0.3485 | 0.3854 | 0.3642 |
| phikon-v2 / mean | 0.3540 | 0.3915 | 0.4062 | 0.4125 |
| resnet50 / concat | 0.2265 | 0.3061 | 0.3367 | 0.3042 |
| resnet50 / late | 0.2360 | 0.2364 | 0.2504 | 0.2642 |
| resnet50 / mean | 0.2529 | 0.2676 | 0.2921 | 0.2808 |

## Subtype failures

Recall is correct records / all records for that subtype across trials and subsets. The distinct-case count shows how little independent evidence supports each row.

| Encoder / fusion | Subtype | Recall | Distinct validation cases |
| --- | --- | ---: | ---: |
| phikon-v2 / concat | adenosis | 0.560 | 2 |
| phikon-v2 / concat | ductal_carcinoma | 0.840 | 4 |
| phikon-v2 / concat | fibroadenoma | 0.320 | 3 |
| phikon-v2 / concat | lobular_carcinoma | 0.520 | 3 |
| phikon-v2 / concat | mucinous_carcinoma | 0.707 | 2 |
| phikon-v2 / concat | papillary_carcinoma | 0.107 | 3 |
| phikon-v2 / concat | phyllodes_tumor | 0.013 | 2 |
| phikon-v2 / concat | tubular_adenoma | 0.373 | 3 |
| phikon-v2 / late | adenosis | 0.547 | 2 |
| phikon-v2 / late | ductal_carcinoma | 0.893 | 4 |
| phikon-v2 / late | fibroadenoma | 0.360 | 3 |
| phikon-v2 / late | lobular_carcinoma | 0.560 | 3 |
| phikon-v2 / late | mucinous_carcinoma | 0.680 | 2 |
| phikon-v2 / late | papillary_carcinoma | 0.080 | 3 |
| phikon-v2 / late | phyllodes_tumor | 0.000 | 2 |
| phikon-v2 / late | tubular_adenoma | 0.373 | 3 |
| phikon-v2 / mean | adenosis | 0.600 | 2 |
| phikon-v2 / mean | ductal_carcinoma | 0.813 | 4 |
| phikon-v2 / mean | fibroadenoma | 0.360 | 3 |
| phikon-v2 / mean | lobular_carcinoma | 0.547 | 3 |
| phikon-v2 / mean | mucinous_carcinoma | 0.787 | 2 |
| phikon-v2 / mean | papillary_carcinoma | 0.133 | 3 |
| phikon-v2 / mean | phyllodes_tumor | 0.000 | 2 |
| phikon-v2 / mean | tubular_adenoma | 0.427 | 3 |
| resnet50 / concat | adenosis | 0.387 | 2 |
| resnet50 / concat | ductal_carcinoma | 0.867 | 4 |
| resnet50 / concat | fibroadenoma | 0.453 | 3 |
| resnet50 / concat | lobular_carcinoma | 0.427 | 3 |
| resnet50 / concat | mucinous_carcinoma | 0.267 | 2 |
| resnet50 / concat | papillary_carcinoma | 0.080 | 3 |
| resnet50 / concat | phyllodes_tumor | 0.253 | 2 |
| resnet50 / concat | tubular_adenoma | 0.173 | 3 |
| resnet50 / late | adenosis | 0.253 | 2 |
| resnet50 / late | ductal_carcinoma | 0.973 | 4 |
| resnet50 / late | fibroadenoma | 0.547 | 3 |
| resnet50 / late | lobular_carcinoma | 0.307 | 3 |
| resnet50 / late | mucinous_carcinoma | 0.347 | 2 |
| resnet50 / late | papillary_carcinoma | 0.053 | 3 |
| resnet50 / late | phyllodes_tumor | 0.027 | 2 |
| resnet50 / late | tubular_adenoma | 0.147 | 3 |
| resnet50 / mean | adenosis | 0.560 | 2 |
| resnet50 / mean | ductal_carcinoma | 0.960 | 4 |
| resnet50 / mean | fibroadenoma | 0.307 | 3 |
| resnet50 / mean | lobular_carcinoma | 0.413 | 3 |
| resnet50 / mean | mucinous_carcinoma | 0.293 | 2 |
| resnet50 / mean | papillary_carcinoma | 0.067 | 3 |
| resnet50 / mean | phyllodes_tumor | 0.000 | 2 |
| resnet50 / mean | tubular_adenoma | 0.120 | 3 |

## Adding a view

Compare each subset with the same case plus exactly one additional magnification. These comparisons overlap. More available views need not improve an individual prediction.

| Encoder / fusion | Correct → wrong | Wrong → correct | Comparisons |
| --- | ---: | ---: | ---: |
| phikon-v2 / concat | 26 | 62 | 1120 |
| phikon-v2 / late | 21 | 53 | 1120 |
| phikon-v2 / mean | 16 | 40 | 1120 |
| resnet50 / concat | 53 | 94 | 1120 |
| resnet50 / late | 43 | 58 | 1120 |
| resnet50 / mean | 39 | 55 | 1120 |

## What to test next

1. Compare classification of individual fields followed by case-balanced probability aggregation with classification after averaging field features. This tests whether early averaging loses useful distinctions.
2. Compare learned mean and gated-attention field pooling, then mean/gated/set-attention view fusion using identical inputs and validation budgets. These are established baselines, not novelty claims.
3. Repeat split selection and training seeds before drawing conclusions. Keep the outer test held out and resolve grouping sensitivity before final evaluation.
