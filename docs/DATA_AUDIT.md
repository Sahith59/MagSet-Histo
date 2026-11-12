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
