# Initial BreakHis archive audit

Audit of the official release obtained on November 11, 2026 (America/New_York). This is an inventory and identifier audit; full image decoding and grouping verification are separate steps.

| Magnification | Images |
| --- | ---: |
| 40× | 1,995 |
| 100× | 2,081 |
| 200× | 2,013 |
| 400× | 1,820 |
| Total | 7,909 |

All file SHA-256 values were indexed. No byte-identical PNG files were found. There are 7,835 images at 700 × 460 pixels and 74 at 700 × 456. Preserve aspect ratio and handle both dimensions rather than silently assuming one exact height.

The archive contains 82 subtype-qualified case identifiers, 81 bare raw identifiers, and 70 numeric stems after stripping trailing section letters. All of these groups have all four magnifications. These are filename-derived counts, not an independent verification of the published 82-patient count.

One bare raw identifier occurs under two subtypes. Numeric-stem grouping also combines several multi-subtype sets. A single patient-level subtype label cannot therefore be assigned by blindly collapsing those fields. Preserve the subtype-qualified specimen as the prediction unit, inspect the original fold metadata, and evaluate conservative grouping as a sensitivity analysis. Do not call exploratory group-disjoint results a verified patient-level benchmark.

| Filename subtype | Images |
| --- | ---: |
| Adenosis | 444 |
| Fibroadenoma | 1,014 |
| Phyllodes tumor | 453 |
| Tubular adenoma | 569 |
| Ductal carcinoma | 3,451 |
| Lobular carcinoma | 626 |
| Mucinous carcinoma | 792 |
| Papillary carcinoma | 560 |

The filename-derived phyllodes/tubular counts differ from the labels on a table in the supplied manuscript. Final tables will come from the audited manifest, with discrepancies explained rather than copied.

Source: [official BreakHis provider](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-BreakHis/). The download script records the archive URL/hash and every image hash under the selected data root. Image files and detailed working manifests are not bundled here.

## Provider protocol reconciliation

The provider's `mkfold.tar.gz` was inspected as data; its Python 2 script was not executed. Each of its five trial files covers all 7,909 images exactly once and assigns each of the 82 subtype-qualified cases entirely to train or test. Each trial has 54 training cases and 28 test cases, with all eight subtypes represented on both sides.

The five test image counts are 2,904, 2,403, 2,577, 2,698, and 3,083. These are five repeated holdouts, not five disjoint cross-validation folds. Numeric stems cross train/test in 8, 6, 7, 7, and 6 groups respectively. The stem relationship is not clinically verified; we will report the provider-defined case protocol and a stricter stem-group sensitivity analysis separately.

Filename-derived case counts are A=4, F=10, PT=3, TA=7, DC=38, LC=5, MC=9, and PC=6. Stripping suffix letters leaves only two adenosis stem groups, so a conventional three-fold class-covered stem protocol is infeasible. The earlier provisional three-fold plan must not be applied blindly.

The initial validation baselines retain the provider's outer test cases, then select one validation case per subtype from outer training only. This leaves 46 fitting cases and eight validation cases per trial. Outer scoring is disabled by default. A large fraction of the dataset is correlated within cases, and eight validation cases give noisy model selection; these pilots establish pipeline behavior, not final publication evidence.

Archive SHA-256: `d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640`. This is our recorded hash of the downloaded archive; no independent provider checksum was verified.
