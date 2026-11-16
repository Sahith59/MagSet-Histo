# Completed Midnight result critique

November 16, 2026. **Midnight case pooling is the most balanced current two-step option across the two arrangements**, with 72.00% / 71.33% accuracy and macro F1 0.555818 / 0.534982. Its arrangement-813 raw-accuracy gain comes from better routing despite more subtype mistakes after correct routing. Papillary recognition improves on both arrangements, while lobular and mucinous gains remain inconsistent.

Prioritize malignant feature generalization after the registered kernel result is complete. If another adaptation study is pursued, the proposed rank-4 LoRA candidate on the query/value projections in Midnight's last four transformer blocks addresses that remaining representation question. Compare its complete two-step output with the fixed frozen Midnight case pipeline, preserving the original roles, all fifteen subsets, class/case/view weighting and a declared selection budget. Both heads remain relevant because the current router also limits the route to 95%. The LoRA recipe and feasibility remain for the parent decision and primary-source assessment.

| Completed dual pipeline | 812 accuracy / F1 | 813 accuracy / F1 |
| --- | ---: | ---: |
| Phikon case | 65.83% / .497517 | 68.83% / .532497 |
| Phikon field | 66.67% / .497530 | 69.50% / .520724 |
| Midnight case | 72.00% / .555818 | 71.33% / .534982 |
| Midnight field | 73.67% / .573162 | 70.00% / .501659 |

Midnight field pooling raises arrangement 812 and lowers 813 against Midnight case pooling. Midnight case has the higher minimum accuracy and F1 across arrangements. Its F1 improvement over Phikon case is +.058301 / +.002485. Both Midnight readouts failed the registered requirement of at least +.02 F1 on both arrangements. The all-four-view Midnight case accuracies are 75.00% / 72.50%, separate from the mean-fifteen endpoint above. [Accepted Midnight comparison](../evidence/BREAKHIS_MIDNIGHT_RESULTS.json)

## What changed with the encoder

These comparisons preserve the cases, labels and readout family. They measure complete encoder/head pipelines, including their representation dimensions and selected C values.

| Case-dual class recall | Phikon → Midnight, 812 | Phikon → Midnight, 813 |
| --- | ---: | ---: |
| Benign | 81.33% → 85.67% | 81.33% → 89.33% |
| Ductal | 78.67% → 85.33% | 97.33% → 74.67% |
| Lobular | 56.00% → 57.33% | 54.67% → 53.33% |
| Mucinous | 57.33% → 61.33% | 56.00% → 40.00% |
| Papillary | 9.33% → 29.33% | 17.33% → 45.33% |

Papillary recall increases by 20 and 28 percentage points. Lobular remains around 53–57%. Mucinous and ductal decline on 813. True-malignant four-type accuracy changes from 55.67% to 60.00% on 812 and from 56.33% to 54.33% on 813. Binary accuracy improves from 87.83% to 91.17% and from 88.33% to 94.17%.

Each arrangement contains 600 repeated case/subset events. On 812, Midnight gains 37 correct events over Phikon case. This equals 20 fewer routing errors plus 17 fewer subtype errors after correct routing. On 813, it gains 15 correct events through 35 fewer routing errors, offset by 20 more correctly routed subtype errors. The field comparison has the same pattern on 813, with 34 fewer routing errors and 31 more subtype errors against Phikon field. [Phikon case comparison](../evidence/BREAKHIS_FIVE_CLASS_RESULTS.json) · [Phikon field comparison](../evidence/BREAKHIS_PHIKON_FIELD_RESULTS.json)

## Remaining error concentration

| Midnight case dual | Arrangement 812 | Arrangement 813 |
| --- | ---: | ---: |
| Benign routed malignant | 43 | 32 |
| Malignant routed benign | 10 | 3 |
| Correct malignant route, wrong subtype | 115 | 137 |
| Subtype share of all final errors | 68.45% | 79.65% |

Lobular, mucinous and papillary account for 107/115 and 118/137 of those subtype errors. Lobular is commonly called ductal, at 31/75 and 35/75 events. Mucinous becomes ductal at 29/75 and 42/75. Papillary commonly becomes mucinous, at 27/75 and 28/75. Arrangement 813 also has 19/75 ductal events called mucinous. These aggregate patterns locate errors. Establishing a tissue, stain or annotation cause requires evidence beyond these class counts.

Two algebraic diagnostics scope the remaining task. With the current subtype decisions and an oracle-perfect router, accuracy would be 80.00% / 77.17%. With the current router and perfect subtype decisions, the upper bounds are its 91.17% / 94.17% binary accuracies. Approaching 95% therefore requires substantial typing improvement alongside routing improvement. These are saved-decision bounds using the same validation mixture.

## Limits on an adaptation decision

Midnight case fitting accuracy is 97.54% / 98.06%, and true-malignant fitting accuracy is 97.26% / 97.79%. Validation malignant accuracy is 60.00% / 54.33%. An adapter's value must appear in held-case subtype performance and the complete endpoint. Additional fitting accuracy would offer little evidence of progress.

Each run has 46 fitting cases, including only two lobular and three papillary cases, and eight validation cases with one example of each malignant type. Across both arrangements there are 31 distinct validation provider keys. Trial mean-subset case accuracy ranges from 54.17% to 89.17%. Cases recur across trials and subsets, and all 82 provider keys have entered development. The comparisons describe reused provider-case development validation. Provider-key grouping also retains the recorded unresolved clinical-identity relationships.

The review recomputed the error decompositions from accepted public confusion matrices and diagnostic summaries. It verified their arithmetic against the published accuracies and binary accuracies. Private `.local/midnight-result-review` files retain the aggregate-only script, source hashes and calculations. The accepted Midnight comparison SHA256 is `c9fdd48763e17c2458aa375797827842d1695e6bdefa3fa5354ca1317bc235fe`. This review performed zero new fits, encoder forwards, downloads or jobs.
