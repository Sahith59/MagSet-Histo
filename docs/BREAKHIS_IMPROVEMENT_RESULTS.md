# BreakHis: completed improvement results

Updated November 16, 2026. The strongest consistent two-step option so far is **frozen Midnight with case-mean pooling: 72.00% / 71.33% five-category validation accuracy**, compared with the original frozen Phikon model's 65.83% / 68.83%. The measured gains are **6.167 / 2.500 percentage points** on arrangements 812 / 813. Its end-to-end macro F1 is **0.5558 / 0.5350**, up from 0.4975 / 0.5325. The 94–95% final-classification target remains unachieved.

The earlier Phikon field-supervision change produced smaller accuracy gains, reaching **66.67% / 69.50%**. Midnight with field supervision reached **73.67% / 70.00%**: higher on arrangement 812, lower on 813 than Midnight case pooling. Its macro F1 also falls below the original reference on 813. Midnight case pooling therefore provides the steadier two-step result across the two arrangements. These are completed, checked development-validation outcomes, not fresh test or clinical results.

The centered-cosine KDE follow-up is running. Its results are pending and are not included below.

## Complete five-category comparison

Final labels are pooled benign, ductal, lobular, mucinous and papillary carcinoma. A two-step prediction first routes benign/malignant; benign stops, while malignant receives one of four carcinoma labels. Final accuracy counts both routing and subtype mistakes. Direct-five controls predict the same five final labels in one head.

Every value below averages five overlapping trials and all **15 nonempty magnification subsets**. Accuracy is a percentage; macro F1 is on a 0–1 scale. The two arrangements are reported separately.

| Model and readout | Arm | 812 accuracy / macro F1 | 813 accuracy / macro F1 |
| --- | --- | ---: | ---: |
| Original frozen Phikon, case means | Direct five | 66.67% / 0.5161 | 64.67% / 0.5035 |
| Original frozen Phikon, case means | Two-step | 65.83% / 0.4975 | 68.83% / 0.5325 |
| Adapted EfficientNet case bags | Direct five | 48.83% / 0.3018 | 49.00% / 0.3016 |
| Adapted EfficientNet case bags | Two-step | 46.50% / 0.2346 | 53.33% / 0.3108 |
| **A: frozen Phikon, field supervision** | Direct five | 66.83% / 0.5125 | 66.17% / 0.4872 |
| **A: frozen Phikon, field supervision** | Two-step | 66.67% / 0.4975 | 69.50% / 0.5207 |
| **B: frozen Midnight, case means** | Direct five | 71.17% / 0.5476 | 73.17% / 0.5281 |
| **B: frozen Midnight, case means** | **Two-step** | **72.00% / 0.5558** | **71.33% / 0.5350** |
| **B: frozen Midnight, field supervision** | Direct five | 73.50% / 0.5649 | 72.17% / 0.5122 |
| **B: frozen Midnight, field supervision** | Two-step | 73.67% / 0.5732 | 70.00% / 0.5017 |

A classifies frozen Phikon fields before pooling their probabilities. B changes the frozen encoder to Midnight and measures both case-mean and field-supervised readouts. Both use the same provider-case assignments and four complete regularization candidates per arm. The field readout composes each field's joint five-class probabilities before averaging within each view and then equally across available views. No physical field correspondence between magnifications is assumed.

The direct-five controls remain competitive. With Midnight case pooling, two-step accuracy is higher on 812 and lower on 813 than direct-five accuracy. The results support an improvement from the measured encoder/readout pipelines; they do not establish that the hierarchy itself always improves prediction.

For Midnight case two-step, balanced accuracy is **63.80% / 60.53%**, versus 56.53% / 61.33% for the original two-step reference. Thus raw accuracy and macro F1 improve on both arrangements, while balanced accuracy decreases by 0.80 percentage points on 813. With all four magnifications available, its accuracy is **75.00% / 72.50%**; these all-view scores are separate from the mean-15-subset headline.

The predeclared larger-improvement screen asked for at least +0.02 macro F1, accuracy no lower and positive recall in every final class on both arrangements. Neither A nor either B readout meets that complete screen. Midnight case two-step improves F1 by +0.0583 on 812 and +0.0025 on 813. The screen describes whether a larger, consistent F1 improvement was observed; the measured accuracy gains above remain the result.

## Routing improves; malignant typing still limits the final answer

The following scores belong to two-step models. Binary accuracy evaluates benign/malignant routing. Typing accuracy evaluates the four-way subtype decision on **all truly malignant validation cases**, independently of whether the router sent them onward. Neither is final five-category accuracy.

| Two-step model | Binary accuracy, 812 / 813 | True-malignant typing accuracy, 812 / 813 |
| --- | ---: | ---: |
| Original Phikon case means | 87.83% / 88.33% | 55.67% / 56.33% |
| EfficientNet case bags | 75.17% / 82.17% | 38.33% / 37.33% |
| A: Phikon field supervision | 89.17% / 89.33% | 54.67% / 55.33% |
| B: Midnight case means | 91.17% / 94.17% | 60.00% / 54.33% |
| B: Midnight field supervision | 93.00% / 95.00% | 61.00% / 50.00% |

The 95.00% entry is binary routing; that pipeline's final five-category accuracy on the same arrangement is 70.00%. Improving malignant typing remains necessary to approach the user's final-classification target.

The rare-type end-to-end recalls show the remaining tradeoffs:

| Two-step model | Lobular recall, 812 / 813 | Mucinous recall, 812 / 813 | Papillary recall, 812 / 813 |
| --- | ---: | ---: | ---: |
| Original Phikon case means | 56.00% / 54.67% | 57.33% / 56.00% | 9.33% / 17.33% |
| A: Phikon field supervision | 52.00% / 53.33% | 57.33% / 61.33% | 6.67% / 10.67% |
| B: Midnight case means | 57.33% / 53.33% | 61.33% / 40.00% | 29.33% / 45.33% |
| B: Midnight field supervision | 48.00% / 48.00% | 66.67% / 41.33% | 32.00% / 28.00% |

Midnight case pooling substantially improves papillary recall, but lobular recall stays near 53–57%. Its arrangement-813 mucinous recall falls from 56.00% to 40.00%, and ductal recall falls from 97.33% to 74.67%. Better benign routing and some subtype gains therefore coexist with important subtype regressions. Raw accuracy alone does not capture those changes.

## Evaluation scope and verification

The cohort contains **82 subtype-qualified provider keys and 7,909 images**, covering 40×, 100×, 200× and 400×. Each run keeps 46 fitting, eight validation and 28 excluded cases, with all related fields/magnifications in their assigned provider-key role. The fitting class counts are 11 benign, 25 ductal, two lobular, five mucinous and three papillary. Validation contains four benign cases and one per malignant type. That gives benign half the weight in raw accuracy; macro F1, balanced accuracy and class recalls remain necessary.

Validation reuses the same provider cases across trials and subsets: arrangements 812/813 contain 22/20 distinct validation keys, with **31 in their union**. Clinical patient identity has not been independently resolved. All 82 provider keys have entered development somewhere, so the 28 excluded cases per run do not form a globally fresh confirmation cohort. No novelty, independent clinical generalization or 94–95% five-category performance is established.

A's complete collector accepted ten paired runs, 80 candidates, 120 heads and 64,800 dependent case/subset records. B's collector accepted both readouts together: 20 paired runs, 160 candidates, 240 heads and 129,600 dependent records. The checks replay saved probabilities, explicit hard decisions, metrics, fitting-only normalization, candidate selection, class order, role membership and artifact hashes; field verification additionally reconstructs weights and objective diagnostics. These execution checks support the reported outputs, while the reused-validation limits remain.

## Source and evidence bindings

All results above come from the complete accepted ledgers; earlier negative outcomes are retained. Both A and B compare against the original frozen five-category ledger SHA256 `bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795`.

| Result | Immutable source | Configuration | Accepted evidence |
| --- | --- | --- | --- |
| Original Phikon | [0471cd4](https://github.com/Sahith59/MagSet-Histo/tree/0471cd4ab8fb1da71eba8870f2ea0fa59ce0e8f3) | [Original five-class recipe](../configs/breakhis-five-class-v1.json) | [Original ledger](evidence/BREAKHIS_FIVE_CLASS_RESULTS.json) |
| EfficientNet | [b5f0df1](https://github.com/Sahith59/MagSet-Histo/tree/b5f0df18a47c1607be492bdaab7ab6cd2171226b) | [Case-bag recipe](../configs/breakhis-casebag-v1.json) | [CNN ledger](evidence/BREAKHIS_CASEBAG_RESULTS.json) |
| A: Phikon fields | [a9a2fda](https://github.com/Sahith59/MagSet-Histo/tree/a9a2fda4461aca74e6596a747ab191aefa7a96da) | [Field recipe](../configs/breakhis-phikon-field-v1.json) | [A ledger](evidence/BREAKHIS_PHIKON_FIELD_RESULTS.json) |
| B: Midnight | [1cbb909](https://github.com/Sahith59/MagSet-Histo/tree/1cbb90986261a9541b71d77e9f2f4eec260e8350) | [Midnight classifier recipe](../configs/breakhis-midnight-five-v1.json) | [B ledger](evidence/BREAKHIS_MIDNIGHT_RESULTS.json) |

Configuration SHA256 values: original `d3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b`; EfficientNet `3f633d0e43f66f51c4e27b3324a17180cda2bf7f2a91c9a757d7b1187da7d7b9`; A's separate field recipe `b70bf70726dd733c2ac357198e5047ed27aa8e35454e172b5797af3300349e15`; B `07b14d634f4bbebd1256e7a5c73f2fa4fce7273849e1663d14c688ca57aad6d3`. A also retains the original base-input configuration binding.

Accepted evidence SHA256 values: CNN `2a904fac355a21c8229cfef24f4a1579a27edf3986c51f828c538d11a1db493a`; A `9ae68ff58813c247c6bddbf824caf084e25af86eb6a79f791825e1943205ace3`; B `c9fdd48763e17c2458aa375797827842d1695e6bdefa3fa5354ca1317bc235fe`. The ledgers retain complete encoder, cache, manifest, protocol, source-archive and per-run artifact bindings.
