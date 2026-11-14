# Independent BreakHis coverage and reference-split review

Reviewed November 13, 2026. This was a read-only metadata and saved-membership audit. No training, feature extraction, encoder forward passes, new predictions, outer-role scoring, or split changes were performed. The released reference implementation was read as text; none of its code was executed.

**Finding:** the ten existing BreakHis runs implement their declared provider-case roles consistently. No filename, class-index, case-membership, view-coverage, path-overlap, or recorded-content-hash defect was found. Their rare-class limitation is real: each run fits only one adenosis case and one phyllodes case. The closest inspected public reference uses more training cases and has released-artifact identity and cross-view inconsistencies that prevent treating its headline scores as an exact four-view comparison.

## Evidence and scope

Authoritative inputs were the local manifest and protocol, all ten frozen-baseline `metrics.json` files, and all ten adaptation `metrics.json` files. The audit independently reconstructed inner membership with Python's standard-library random generator, canonical class order, sorted eligible provider-case keys, and one `choice` per class. It compared that membership with both saved experiment families. The project data/protocol modules were inspected; author modules were never imported.

| Input | SHA256 / revision |
| --- | --- |
| BreakHis manifest | `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31` |
| BreakHis protocol | `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134` |
| Public graph reference source | `09371ccff9860cb4ef9cb1a6625f6fd269e8da64` |

All twenty saved runs match the manifest/protocol hashes, canonical eight-class order, and independently reconstructed fit/validation/excluded-outer memberships. Hash-disjointness below means **hashes recorded in the authoritative manifest**; this audit did not reread and rehash the remote image pixels. It also does not establish clinical identity from filenames.

## Cohort and class coverage

The manifest has 7,909 unique image paths, filenames and recorded SHA256 values. Every filename independently agrees with its recorded subtype code, benign/malignant category, bare identifier, magnification and field sequence. Every provider case has nonempty image sets at all four magnifications. Equal field sequence numbers were not treated as registered views of the same tissue.

| Subtype | Provider cases | Bare IDs within class | Conservative stems within class | 40× images | 100× | 200× | 400× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adenosis | 4 | 4 | 2 | 114 | 113 | 111 | 106 |
| Ductal carcinoma | 38 | 38 | 38 | 864 | 903 | 896 | 788 |
| Fibroadenoma | 10 | 10 | 7 | 253 | 260 | 264 | 237 |
| Lobular carcinoma | 5 | 5 | 4 | 156 | 170 | 163 | 137 |
| Mucinous carcinoma | 9 | 9 | 7 | 205 | 222 | 196 | 169 |
| Papillary carcinoma | 6 | 6 | 6 | 145 | 142 | 135 | 138 |
| Phyllodes tumor | 3 | 3 | 3 | 109 | 121 | 108 | 115 |
| Tubular adenoma | 7 | 7 | 6 | 149 | 150 | 140 | 130 |
| Globally distinct total | **82** | **81** | **70** | **1,995** | **2,081** | **2,013** | **1,820** |

Within-class stem counts do not sum to 70 because three stems occur under multiple subtype labels. One bare ID occurs under two subtype-qualified provider cases, with ductal and lobular labels. These are naming collisions/relationships in the supplied material; they do not establish that the images are, or are not, from the same clinical patient. The provider-qualified key retains the full subtype prefix and original bare identifier. The conservative stem removes the trailing letter suffix and is a separate audit grouping.

Every one of the ten existing run arrangements has exactly these case counts:

| Subtype | Fit | Validation | Excluded outer role |
| --- | ---: | ---: | ---: |
| Adenosis | 1 | 1 | 2 |
| Ductal carcinoma | 25 | 1 | 12 |
| Fibroadenoma | 6 | 1 | 3 |
| Lobular carcinoma | 2 | 1 | 2 |
| Mucinous carcinoma | 5 | 1 | 3 |
| Papillary carcinoma | 3 | 1 | 2 |
| Phyllodes tumor | 1 | 1 | 1 |
| Tubular adenoma | 3 | 1 | 3 |
| Total | **46** | **8** | **28** |

Hundreds of fields from a rare fitting case do not provide hundreds of independent fitting cases. Each class's validation recall in one run is based on one case; its 15 missing-view outcomes remain correlated observations from that same case.

## Disjointness of the existing roles

For all ten arrangements, the three roles cover all 82 provider cases and all 7,909 images exactly once. Every pair of roles has zero provider-qualified-case, relative-path and recorded-hash intersection. There are no empty magnification bags or role-specific missing-view exclusions.

The table distinguishes bare-ID and conservative-stem overlap from provider-key overlap. Each overlap triple is ordered **fit–validation / fit–outer / validation–outer**.

| Inner seed | Trial | Images: fit / validation / outer | Bare-ID overlap | Stem overlap |
| --- | ---: | --- | --- | --- |
| 812 | 1 | 4,364 / 641 / 2,904 | 0 / 1 / 0 | 2 / 7 / 2 |
| 812 | 2 | 4,516 / 990 / 2,403 | 0 / 0 / 0 | 1 / 4 / 3 |
| 812 | 3 | 4,542 / 790 / 2,577 | 0 / 1 / 0 | 1 / 6 / 2 |
| 812 | 4 | 4,519 / 692 / 2,698 | 0 / 0 / 0 | 0 / 6 / 1 |
| 812 | 5 | 4,132 / 694 / 3,083 | 0 / 1 / 0 | 1 / 5 / 1 |
| 813 | 1 | 4,416 / 589 / 2,904 | 0 / 1 / 0 | 1 / 6 / 3 |
| 813 | 2 | 4,724 / 782 / 2,403 | 0 / 0 / 0 | 2 / 5 / 2 |
| 813 | 3 | 4,501 / 831 / 2,577 | 0 / 1 / 0 | 2 / 5 / 3 |
| 813 | 4 | 4,446 / 765 / 2,698 | 0 / 0 / 0 | 1 / 5 / 2 |
| 813 | 5 | 4,071 / 755 / 3,083 | 0 / 1 / 0 | 2 / 3 / 3 |

Consequently these are **provider-case exploratory validation roles**, not clinically reverified patient-disjoint partitions. The bare-ID cross-role collision occurs only across fit/outer here; some conservative stems also cross fit/validation. Neither observation is evidence of exact duplicate image leakage, and neither can be dismissed as verified clinical independence.

The validation unions contain 22 distinct provider cases for seed812 and 20 for seed813, with 31 across both. The union of fitting roles contains79; the excluded-outer union contains69; the union of fitting and validation contains all82. The five trials overlap. There is no globally untouched BreakHis confirmation cohort in this development history. Two conservative adenosis stems also prevent a strict stem-disjoint, class-covered three-way split. This review preserved the existing roles and did not search for a split with better scores or counts.

## Pinned public reference: what is compatible and what blocks direct reuse

All80 released split files were read using a UTF-8-BOM-aware CSV reader: binary/multiclass × four magnifications × five folds × train/validation. Every filename maps uniquely to our manifest, the portable path suffix agrees, all label/magnification/bare-ID fields agree, and each individual train/validation pair covers its magnification exactly. Within each pair, raw IDs, provider-qualified cases, paths and recorded hashes are disjoint. This is a useful compatible image inventory. See the pinned [locked split directory](https://github.com/zaiedhussein/breakhis-graph-transformer/tree/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/resources/locked_splits).

However, the following distinctions must be resolved before calling an experiment an exact reproduction:

1. **Different training and endpoint budgets.** Each released multiclass pair has65 provider-qualified training cases and17 validation cases, versus our46 fit and8 validation cases. Its training cases per class are adenosis3, ductal30–31, fibroadenoma8, lobular4, mucinous7–8, papillary4–5, phyllodes2, tubular5–6. Its per-magnification, single-view endpoint differs from our average over15 four-view subsets. Author headline scores are context, not paired evidence of an improvement under our roles.
2. **Validation folds overlap in the released multiclass artifact.** At every magnification, five validation files contain85 provider-case appearances covering82 distinct cases; three cases appear twice. In the CSV's bare-ID convention, these counts are84 appearances/81 distinct, again three repeated twice. Binary validation is nonoverlapping:82 provider-case appearances/82 distinct, or81/81 bare IDs. Repeated validation cases are not intrafold train/validation leakage, but they preclude treating multiclass results as ordinary disjoint out-of-fold predictions.
3. **Multiclass fold3 is inconsistent across magnifications.** The40× and100× roles agree, as do200× and400×, but two adenosis provider cases exchange roles between these view groups. Unioning all four released train sets and all four validation sets would produce66 training cases,18 validation cases and **two cases in both roles**. Folds1/2/4/5 have no such cross-view conflicts. The released multiclass CSVs cannot be concatenated into a joint four-view protocol. Binary roles agree across all four magnifications.
4. **The released identity convention differs from current source.** Released CSV `patient` values are bare IDs; the pinned [filename parser](https://github.com/zaiedhussein/breakhis-graph-transformer/blob/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/src/breakhis_graph/data/filename_parser.py) constructs subtype-qualified IDs. The pinned [split generator](https://github.com/zaiedhussein/breakhis-graph-transformer/blob/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/src/breakhis_graph/data/splits.py) rejects multi-label patient groups and promises nonoverlapping validation folds under its default strict protocol; it separately permits rare-class patching as sensitivity analysis. The released multiclass CSVs exhibit a multi-subtype bare ID and repeated validation membership. Current default code and bundled artifact therefore do not express one identical identity/protocol contract. The exact historical generation path has not been established.
5. **A concrete aggregation ambiguity exists if released IDs are passed through current aggregation.** The pinned [aggregation function](https://github.com/zaiedhussein/breakhis-graph-transformer/blob/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/src/breakhis_graph/evaluation/aggregation.py) groups probabilities by `patient` and sets truth to the first tile's label, without checking that group labels agree. Released multiclass fold2 validation contains the bare-ID group with both ductal and lobular images at every magnification. Passing that artifact's IDs directly to this function would merge the two provider-qualified cases and choose one truth label. All eight **image subtypes are present** in every released multiclass validation file; deriving coverage by taking one label per bare-ID group can incorrectly report lobular as absent. This is an operational incompatibility, not proof of which path produced the reported paper numbers.

No conclusion about the paper's validity, clinical patient relationships, or reproduced model performance follows from this static audit. No third party was contacted and no public issue was filed.

## Decision for the next BreakHis experiment

Keep all existing role assignments and the frozen baseline. The audit found no implementation defect that would justify silently correcting our split or relabeling measured results. It supports one preregistered, controlled supervised image-model reference under the **same existing ten roles**, with fit-only augmentation/optimization/normalization, validation-only selection, saved revisions and equal case/view field budgeting. Freeze its architecture, initialization, transforms, epoch/selection budget, image-to-case aggregation and15-subset reporting before fitting.

Any author-code adaptation must explicitly replace the ambiguous identity boundary with our declared provider-case key and preserve all views in one role; that becomes a protocol-matched implementation comparison, not an exact reproduction of the released fold numbers. Single-view reference reproduction, if later pursued, requires a separate labeled track and an explicit single-label identity policy; its scores cannot be mixed with our15-subset endpoint. Do not use the excluded outer roles to choose a rescue model or introduce a new split to improve the reported count or score.

The current evidence establishes scarce independent rare-class examples, a large observed fitting–validation gap from the completed diagnosis, and an imperfectly matched public reference. It does not prove that more epochs, a larger encoder, a smaller encoder, or a new fusion mechanism will solve the problem. Those remain hypotheses for a bounded, documented comparison. Clinical grouping and genuinely fresh confirmation remain open limitations of the final project report.
