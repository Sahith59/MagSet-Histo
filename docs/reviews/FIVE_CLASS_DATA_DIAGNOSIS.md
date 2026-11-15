# Five-category BreakHis data and split diagnosis

Reviewed November 15, 2026 against source revision `9c16837b131565b04ddec1ae9339046786d9947d`. This is an independent read-only audit of the existing data, roles, feature-cache inventory and saved prediction labels. The corrected target is pooled benign plus ductal, lobular, mucinous and papillary carcinoma, with a binary router and a malignant-only four-class specialist. There is no benign subtype specialist. [Authoritative scope](../DUAL_STEP_SCOPE.md).

**Finding:** no filename parsing, label-index, manifest-coverage, saved-membership or original frozen-cache defect was found. The existing 46 fitting / 8 validation / 28 excluded provider-case roles reproduce exactly. Scarce independent malignant examples and unresolved clinical identity remain real limitations. This audit does not establish one cause of poor generalization, train the five-category target or demonstrate 94–95% performance.

## Independent checks and evidence

The audit independently parsed every filename and checked its benign/malignant category, subtype code/name, original identifier, conservative stem, magnification, sequence and corresponding portable path fields. All 7,909 records agree. Image paths, filenames and recorded image-content hashes are each unique. All 82 subtype-qualified provider cases have a nonempty bag at all four magnifications. Equal sequence numbers across magnifications were never treated as registered tissue correspondence.

All five original provider fold files were reread as data from the recorded fold archive. Each covers the exact manifest once, keeps every subtype-qualified case in one role and exactly reproduces the saved protocol. An independent standard-library reconstruction used the original eight-class order, sorted eligible keys, the declared inner seed and one random choice per original subtype. All ten arrangements reproduce the saved roles.

The ten frozen, ten adapted, ten supervised, ten regional-control and ten regional-candidate metadata files all match these reconstructed memberships, manifest/protocol hashes and canonical eight-class order. Their 6,000 archived validation case/subset records have the expected labels, allowed role, unique case/subset keys and complete eight-case × 15-subset coverage. These are repeated appearances, not 6,000 independent patients. No prediction probabilities or historical scores were relabeled as five-category results.

The four original frozen Phikon-v2 feature NPZ files were read locally with pickle disabled. Their byte hashes match recorded metadata; each exactly covers its magnification's manifest paths once. The arrays are finite float32 with 1,024 channels and contain 1,995 / 2,081 / 2,013 / 1,820 images at 40× / 100× / 200× / 400×. Their encoder revision and recorded transforms agree. This audit did not reread or rehash original image pixels, rerun decoding, execute the encoder or regenerate features.

| Evidence | SHA-256 |
| --- | --- |
| Manifest | `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31` |
| Provider protocol | `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134` |
| Original provider fold archive | `db7e2f16c398fbc42286b4f7d65e5f55c8f94d95123a46dfd052d8117d68ae27` |
| Regenerated ten-arrangement membership fingerprint | `23b24dd7e307114c0b80f76953839b3f9e33a32c2aaccfc69fb23795e2827d96` |
| Frozen 40× feature array | `b5daddf7c8539851dadce9be88c093aa3bc866efffc9dccffd3d6f1e02ad614d` |
| Frozen 100× feature array | `047a9ff53e07d857aafc671b972d721e4b2190492b88c134d7a027f90b87c0a0` |
| Frozen 200× feature array | `26e8635b1773fe78aa41943fb09fba71d8ee77197b35a38c9fb41237090e782d` |
| Frozen 400× feature array | `2d61a4325ae82dd295445678134885af7a3a37622cec1be3cc26dbc076da3385` |

The membership fingerprint hashes UTF-8 JSON of a list ordered by seed 812, 813 and then trial 1–5. Each item has `seed`, `trial` and `roles`; each role contains sorted original qualified keys. JSON uses sorted object keys and compact separators. Only its fingerprint is public here.

## Exact class supports

Every existing arrangement has the same class counts, although its selected cases and image counts differ.

| Original subtype | Fitting cases | Validation cases | Excluded cases |
| --- | ---: | ---: | ---: |
| Adenosis | 1 | 1 | 2 |
| Fibroadenoma | 6 | 1 | 3 |
| Phyllodes tumor | 1 | 1 | 1 |
| Tubular adenoma | 3 | 1 | 3 |
| Ductal carcinoma | 25 | 1 | 12 |
| Lobular carcinoma | 2 | 1 | 2 |
| Mucinous carcinoma | 5 | 1 | 3 |
| Papillary carcinoma | 3 | 1 | 2 |
| **Total** | **46** | **8** | **28** |

Pooling the four benign labels changes the task and support distribution:

| Final category | Complete cohort cases | Complete cohort images | Fitting cases | Validation cases | Excluded cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled benign | 24 | 2,480 | 11 | 4 | 9 |
| Ductal carcinoma | 38 | 3,451 | 25 | 1 | 12 |
| Lobular carcinoma | 5 | 626 | 2 | 1 | 2 |
| Mucinous carcinoma | 9 | 792 | 5 | 1 | 3 |
| Papillary carcinoma | 6 | 560 | 3 | 1 | 2 |
| **Total** | **82** | **7,909** | **46** | **8** | **28** |

The router therefore learns from 11 benign and 35 malignant fitting cases. The specialist learns only from the 35 truly malignant fitting cases, with subtype support 25 / 2 / 5 / 3. Pooling benign removes the need to distinguish benign subtypes, but creates no new independent cases for the scarce malignant types. Large field counts and 15 masking augmentations do not increase these independent case counts.

Validation support is 4 / 1 / 1 / 1 / 1. Raw five-category accuracy is `0.5 × benign recall + 0.125 × sum(four malignant recalls)`; balanced accuracy averages the five recalls equally. An always-benign prediction yields 50% raw accuracy and 20% balanced accuracy. Preserve macro F1, balanced accuracy and all five recalls alongside accuracy. A gain caused by benign pooling is a target change, not evidence that hierarchy improves the model.

## Disjointness and identity uncertainty

For every arrangement, the three roles cover all 82 qualified cases and all 7,909 images exactly once. Every role pair has zero overlap in qualified keys, image paths and **recorded** image hashes. There are no missing-view exclusions. Bare-ID and conservative-stem overlap are separate identity concerns; they do not establish byte-identical image leakage or clinical independence.

Overlap triples below are ordered fitting–validation / fitting–excluded / validation–excluded.

| Inner seed | Trial | Images: fitting / validation / excluded | Bare-ID overlap | Conservative-stem overlap |
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

The 82 qualified cases correspond to 81 bare identifiers and 70 stripped stems. One bare identifier has both ductal and lobular labels; qualification prevents those specimens from silently merging. Three stems combine original subtypes; pooling benign resolves two of those *label* conflicts, while the ductal/lobular conflict remains. It does not verify their clinical relationship. Automatically replacing qualified keys with bare IDs or stems is therefore unsupported. Preserve raw identity fields and retain the description **exploratory provider-case validation**, rather than claiming verified patient separation.

Validation unions contain 22 distinct cases for arrangement 812, 20 for 813 and 31 together. The fitting union contains 79; fitting plus validation covers all 82. Excluded roles are per run: their union contains 69 cases, many used for development in other trials. No globally fresh BreakHis confirmation cohort remains. These repeated holdouts and their view subsets cannot support independent-fold or independent-patient uncertainty calculations without accounting for reuse.

## Preprocessing assessment and next action

The inspected path decodes RGB, uses the pinned official Phikon processor, pools fields within each view, then pools available views. Recorded processing resizes the shortest edge to 224 and center-crops 224 × 224. That documented crop can omit lateral tissue; it is a representation limitation, not a discovered metadata mismatch. Original image sizes are 7,835 at 700 × 460 and 74 at 700 × 456. The audited parser does not assume one height. Cache path matching uses the full exact relative path, and the retained field-moments normalizer is fitting-case-only and case-balanced by source inspection. This audit did not independently reconstruct saved normalizer values or isolate the crop's causal effect on accuracy.

Proceed with one registered direct-five versus binary-plus-malignant-four comparison on this unchanged original frozen representation. **Construct the existing roles from the original eight-class identities first, then pool target labels.** Reapplying the inner splitter to five pooled classes would change the validation membership/count and violate the 46 / 8 / 28 contract. Require exact membership checks before every fit.

Lock head-specific class weighting, case/subset weighting, normalization, common tuning opportunities, route threshold, hard decision rule and five-class metric definitions before outcomes. Give each original case total training weight one across its 15 subset rows before the declared class weighting. Use truly malignant fitting cases only for the specialist. Preserve all four views, all 15 subsets, both arrangements and all trials. Count incorrect routing and incorrect malignant typing in end-to-end five-class errors; final hard-route accuracy cannot exceed router accuracy on the same examples.

Do not change roles, choose favorable cases, score excluded cases to rescue a result or promise 94–95%. Pooling and a new classifier may change measured performance, but the present audit supplies no performance forecast, clinical claim or independent confirmation. TCGA remains deferred. No fitting, new predictions, encoder forwards, scheduler jobs, resplitting or changes to supplied originals occurred in this review.

[Existing protocol](../BENCHMARK_PROTOCOL.md) · [Earlier coverage review](BREAKHIS_COVERAGE_REVIEW.md) · [Saved-model diagnosis](../BREAKHIS_DIAGNOSIS_RESULTS.md) · [Corrected architecture and score definitions](../DUAL_STEP_SCOPE.md).
