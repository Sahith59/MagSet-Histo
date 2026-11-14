# Capstone saved-prediction endpoint bridge

November14,2026. All ten accepted models and all five fixed image slices were checked. **Returning to individual40× image scoring does not restore the original95% figure in this supervised ResNet pipeline.** No model was fitted, executed or reselected in this analysis.

Historical notebooks saved93.73%/94.99% accuracy for single40× images under an image-random, reused validation partition. Their EfficientNet architecture, fitting and selection differ from this ResNet. Exact historical case overlap and the separate effects of these differences are unknown. These numbers are not an apples-to-apples model ranking. [Before/now explanation](CAPSTONE_BEFORE_AND_NOW.md).

## Individual-image endpoint

Each row is the mean of five trials. Raw image accuracy gives every field one vote; the weighted version gives equal mass to class, case within class and available zoom, then fields within each case/zoom. These are different units from one vote per case. No view is selected as the best endpoint.

| Arrangement | Image slice | Raw image accuracy | Weighted image accuracy | Raw image macro F1 | Weighted image macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 812 | all_fields | 24.97% | 31.23% | 0.2469 | 0.2711 |
| 812 | 40 | 27.13% | 32.89% | 0.2536 | 0.2827 |
| 812 | 100 | 28.41% | 36.48% | 0.2693 | 0.2992 |
| 812 | 200 | 24.60% | 30.67% | 0.2360 | 0.2567 |
| 812 | 400 | 19.15% | 24.86% | 0.1993 | 0.2209 |
| 813 | all_fields | 31.01% | 32.63% | 0.2516 | 0.2722 |
| 813 | 40 | 29.94% | 31.82% | 0.2398 | 0.2593 |
| 813 | 100 | 34.52% | 34.68% | 0.2812 | 0.2951 |
| 813 | 200 | 30.26% | 31.96% | 0.2401 | 0.2649 |
| 813 | 400 | 29.14% | 32.07% | 0.2361 | 0.2620 |

## Same40× fields, same weights, case averaging

| Arrangement | Raw image accuracy | Weighted image accuracy | Case accuracy | Case macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 812 | 27.13% | 32.89% | 37.5% | 0.3017 |
| 813 | 29.94% | 31.82% | 35.0% | 0.2783 |

Case probabilities average all40× field probabilities equally within a case. These reproduce the archived40× case scores; this is the same trained head, not a second classifier. The larger case accuracy does not prove a causal learning benefit: the prediction/evaluation unit changes. Both endpoints remain weak. Missing-view averaging alone does not explain the current poor transfer.

## Coverage, verification and limits

The accepted validation union contains31 distinct provider cases and2,914 distinct fields across all four views. There are7,529 repeated field evaluations across the ten runs; these are not7,529 independent patients. Arrangement812 has22 distinct cases and2,267 fields; arrangement813 has20 cases and2,004 fields.

Trusted labels/roles, exact original source/artifact hashes, all ten report/field/prediction sets, full field coverage and stable probability arithmetic passed. An independent NumPy audit recomputed all raw/weighted slice metrics,40× case means, both summaries and distinct counts without invoking the analysis metric helper. Maximum numerical difference2.27e-14; case probability mean difference4.44e-16. Full model/optimizer verification remains evidence from the previously completed scheduled checker, not a new local tensor audit. [Review](reviews/CAPSTONE_BRIDGE_REVIEW.md).

Input comparison SHA256 `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`; aggregate bridge SHA256 `51c71fd8122f6e1684eb969f5c465d7279b53b61eb5c5718c6855ede382d1811`. [Complete aggregate ledger](evidence/CAPSTONE_ENDPOINT_BRIDGE.json) retains every run/slice/class/confusion/loss/source hash without provider identifiers. [Registered analysis](CAPSTONE_BRIDGE_SPEC.md).

All data remain exploratory development data with overlapping cases, adaptive history and unresolved clinical identity. There is no new test cohort, new training, outer-role scoring or novel-method evidence. Keep the frozen case reference. The capstone report separates historical outputs, current implementation results and future research claims.
