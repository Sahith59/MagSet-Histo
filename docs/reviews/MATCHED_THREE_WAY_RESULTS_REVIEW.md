# Matched three-way comparison: independent outcome review

November 17, 2026. **The complete saved-artifact comparison is accepted. Midnight P has the highest primary point estimate, but the paired intervals do not establish superiority.** The [complete evidence](../evidence/MATCHED_THREE_WAY_RESULTS.json) records checker status `complete accepted matched three-way comparison`. Its SHA-256 is `3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62`; the bound private review has no errors and SHA-256 `5b521f3f57a1029a5bf0b3ae28ee9aff4a78244270249b38912e6700689143e5`.

This review preserves the dated [partial review](MATCHED_THREE_WAY_PARTIAL_RESULTS_REVIEW.md), including the failed A30 replay. It covers the [fixed comparison](../MATCHED_THREE_WAY_SPEC.md), source `a6e6a54e4fac090051837df17b0dea14efca3bb3`, configuration SHA-256 `1cdc7f143941a25be7b595c3b058e545c57c19a3a367df68123d977f757d5d5c`, and input-audit SHA-256 `cec9c0e13f2673046b41e6502154ceb9bed047f44086afbe04d57df562ef0812`.

## Independently replayed measurements

Separate NumPy/scikit-learn calculations replayed all 12 validation candidates and all three selected test NPZ archives, without calling the collector's scoring helpers. Test archives exactly cover the same 1,704 fields from 16 provider cases. The independent calculation averages joint five-class field probabilities within each magnification, equally averages active magnifications, applies the fixed malignant threshold of 0.5, and reproduces every saved case/mask decision, confusion matrix, macro F1 and accuracy. Saved joint-five record probabilities/logs and artifact hashes agree. The neural validation head arrays independently reproduce their composed joint-five probabilities.

The primary score averages five-label macro F1 over the 15 nonempty magnification subsets. Accuracy averages the corresponding 15 case accuracies. The all-four columns describe that single subset.

| Selected method | Mean15 macro F1 | Mean15 final-five accuracy | All-four macro F1 | All-four accuracy | Worst-subset macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midnight P, T=0.1 | **0.375675** | **67.50%** | 0.341176 | 68.75% — 11/16 | 0.228571 |
| CNN–Transformer, epoch 40 | 0.348180 | 56.67% | 0.372857 | 56.25% — 9/16 | 0.245000 |
| Swin, epoch 20 | 0.304947 | 59.17% | 0.278992 | 56.25% — 9/16 | 0.278992 |

The mean15 accuracy numerators are 162/240, 136/240 and 142/240 correct case/mask events, respectively. Those are dependent repeated views of **16 cases**, not 240 independent test examples. P's highest primary point estimate does not extend to every endpoint: CNN has higher all-four macro F1, and Swin has higher worst-subset macro F1.

The binary and typing outcomes remain distinct. The following typing columns average per-subset recalls/conditional accuracies; conditional accuracy uses only true malignant cases routed malignant in each subset.

| Method | Mean15 binary accuracy | Mean15 end-to-end malignant-four recall | Mean15 subtype accuracy conditional on correct malignant routing |
| --- | ---: | ---: | ---: |
| Midnight P | 94.5833% | 57.5758% | 59.4007% |
| CNN–Transformer | 86.2500% | 53.9394% | 55.7643% |
| Swin | 82.5000% | 65.4545% | 65.9394% |

These binary percentages must not be presented as final-five accuracy. Stronger benign/malignant decisions do not ensure correct malignant typing.

## Rare-class behavior and uncertainty

The test support is benign/DC/LC/MC/PC = **5/7/1/2/1 cases**. All-four correct counts independently reconstructed from the confusion matrices are:

| True label | Support | Midnight P correct | CNN correct | Swin correct |
| --- | ---: | ---: | ---: | ---: |
| Benign | 5 | 5 | 3 | 2 |
| Ductal | 7 | 6 | 5 | 7 |
| Lobular | 1 | 0 | 0 | 0 |
| Mucinous | 2 | 0 | 1 | 0 |
| Papillary | 1 | 0 | 0 | 0 |

All three miss the single lobular and papillary cases in the all-four condition. This is an observed weakness in this small cohort, not an estimate of population-wide zero recall or a biological explanation.

An independently implemented, vectorized cluster-confusion calculation reproduced **every paired bootstrap interval bound** in the accepted checker within 2e-12. It uses the same 2,000 seeded draws of 13 conservative groups, with every provider case and all 15 masks within each sampled group kept together and paired across methods.

| Difference | Primary mean15 macro-F1 point difference | Paired 95% percentile interval |
| --- | ---: | ---: |
| P minus CNN | +0.027495 | [−0.068735, +0.098652] |
| P minus Swin | +0.070728 | [−0.044783, +0.150354] |
| CNN minus Swin | +0.043233 | [−0.066393, +0.107967] |

All primary intervals include zero. Accuracy-difference intervals also include zero: P minus CNN [−10.4167, +30.0000] percentage points, P minus Swin [−15.2381, +31.9316], and CNN minus Swin [−26.6667, +12.5926]. These exploratory intervals have no multiplicity correction and do not establish an ordered population ranking.

**1,533 of 2,000 draws (76.65%) omit at least one true class.** The predeclared fixed-five-label macro-F1 definition is retained in those draws; they are not discarded or conditioned away. This high omission frequency emphasizes the limited rare-class support. Thirteen conservative identifier groups are not verified clinical patients, and the intervals do not account for prior development reuse or dataset shift.

## Selection, execution and review limits

The shared training/validation/test roles contain 49/17/16 cases and 42/15/13 conservative groups, with no group crossing roles. Both neural histories complete 40 epochs and 11,560 updates under the registered schedule. Independent validation replay agrees with the first-maximum selections: P temperature 0.1, CNN epoch 40 and Swin epoch 20. All three selection hashes match the common gate sealed before selected test queries. No setting changed after test outcomes.

Both original selected-test attempts on A30 stopped at the validation replay guard before neural test inference. Their failures and subsequent validation-only diagnostics remain preserved. The later A40 selected-test runs used the same scientific source, selected checkpoints and unchanged `1e-5` guard. Their saved first-16-validation-field replay receipts report **zero maximum error for both heads**, and the complete checker accepts both. This review verified the receipts and their hash bindings; it did not independently reload or execute those neural models.

The accepted collector also replays P's cached-descriptor geometry from exactly the 49 fitting supports, with maximum field-probability error zero. That geometry replay explicitly depends on the reviewed `prior_centered_model` implementation. The additional arithmetic in this review independently verifies saved field-to-case pooling, decisions, metrics and bootstrap calculations. It does not independently reproduce the raw-image encoders, shared crop cache or large checkpoint bytes locally. No fitting, raw-image inference, job submission or scientific-source change was performed for this review.

This is a matched **reused-development** comparison with common cases, labels, roles, crop information, pooling, prediction unit and four validation choices. Different pretrained readers, learning algorithms and computational budgets remain deliberate full-pipeline differences. The results are separate from earlier image-level notebook scores and previous provider-case role arrangements. They establish neither a globally fresh clinical validation nor 94–95% final-five accuracy. P is the highest primary point estimate in this run; consistent superiority and a robust rare-subtype solution remain unestablished.
