# Fixed-router specialist-reader comparison

Registered November 17, 2026, before this family's validation scoring. The user requests legitimate improvement toward >90% final-five and >97% binary accuracy. This single bounded experiment tests malignant typing; its binary router is fixed and cannot improve either arm's binary accuracy. No new result or performance promise is established by registration. [Configuration](../configs/breakhis-specialist-reader-v1.json) · [methodological critique](reviews/MATCHED_METHOD_CRITIQUE.md) · [completed parent comparison](MATCHED_THREE_WAY_RESULTS.md).

## Question and two prespecified arms

Compare **UNI2-h malignant specialization** against **P malignant specialization**, each attached to the identical frozen P binary decision. Earlier reader comparisons, completed before the matched test, suggested stronger UNI2-h typing but weaker routing. The inconsistent gains, weak papillary recall and negative learned-head studies remain counterevidence. This is a frozen readout-package experiment, not a novel algorithm or a pure backbone ablation.

| Arm | Binary head | Malignant head | Validation choices |
| --- | --- | --- | --- |
| `p_control` | P head-specific binary KDE, temperature 0.1 | P head-specific malignant KDE | 0.01 / 0.03 / 0.1 / 0.3 |
| `uni2_specialist`, fixed primary | Same P binary probabilities | Accepted UNI2-h C-geometry malignant KDE | 0.01 / 0.03 / 0.1 / 0.3 |

Each malignant temperature is shared across the four views. There are eight validation candidate pipelines, not four. The parent's earlier selection among four shared binary/malignant temperatures remains part of the selection history. The new P control receives the same four malignant-only choices as UNI2-h; binary temperature 0.1 never changes. Report the fixed primary UNI2-h comparison and every negative outcome, rather than pick an arm from test performance.

## Immutable parent, fields and roles

Bind the accepted parent result SHA-256 `3e8fc1f1ff8b28bc845b39a217bc34597c07c575842f1729fcfbce077c7bcb62`, its scientific source `a6e6a54e4fac090051837df17b0dea14efca3bb3`, configuration, canonical manifest and exact private input audit. The new source is frozen separately before execution. The configuration records all reference hashes; do not copy a different role assignment from an older reader configuration.

| Role | Provider specimens | Conservative stems | Original fields | Benign / DC / LC / MC / PC specimens |
| --- | ---: | ---: | ---: | --- |
| Fitting | 49 | 42 | 4,621 | 14 / 24 / 2 / 5 / 4 |
| Validation | 17 | 15 | 1,584 | 5 / 7 / 2 / 2 / 1 |
| Retrospective follow-up | 16 | 13 | 1,704 | 5 / 7 / 1 / 2 / 1 |

Preserve original eight-type truth beside the final-five target. Keep every related specimen, field, magnification and derived image within its parent role. Use all four magnifications and all 15 nonempty subsets. Clinical identity is unresolved, all cases entered earlier development, and the parent test outcomes have now been viewed. The 16-case follow-up is explicitly reused development evidence, never a new independent confirmation cohort. No new seed, fold, zoom selection, label correction or train-plus-validation refit is permitted.

Reuse only the accepted immutable P three-crop and UNI2-h center-crop raw-field feature caches. The hashed reference configurations bind their complete shard/metadata/encoder/extraction provenance. Match each descriptor through canonical relative path, image SHA, provider specimen, conservative group and magnification; require a unique complete bijection over 7,909 fields. Never assume cache row positions agree or infer correspondence between equally numbered fields at different magnifications.

P uses a mean of three 3,072-dimensional crop descriptors; UNI2-h uses a single 1,536-dimensional center descriptor. This coverage difference, the reader pretraining and the two-reader compute are disclosed package differences. There are no new image decodes or encoder forwards.

## Fitting-only geometry and densities

Recompute every center, class count and support from the current 49 fitting specimens. Never load a center/readout fitted under the earlier 46-case roles. No validation or follow-up feature contributes to centering, support, normalization, metric learning or calibration.

For the shared P binary head, keep the accepted head-specific center: within each view, take each fitting specimen's raw-field mean, average case means within benign and malignant groups, then average those two group centers equally. Binary support contains all 49 fitting specimens, with uniform binary-class priors and equal case/field mass inside each class.

For the P malignant control, use its existing head-specific geometry: mean raw fields per fitting malignant case, mean case vectors within each subtype, then equal mean over the four subtype centers. Its support contains only the 35 fitting malignant specimens.

For the UNI2-h malignant head, retain the accepted C geometry: compute each fitting case's raw-field mean in float64 within each view, then average all 49 case means equally, including benign cases in this centering statistic. Subtract that current fitting-only center from each raw UNI2-h field and L2-normalize in float64 using `max(norm, 1e-12)`. Do not substitute float32 case means from an unrelated loader, subtype-balanced centering, a standardizer, PCA or a learned metric.

UNI2-h malignant support also contains only the 35 fitting malignant cases, with subtype counts 24/2/5/4. In every same-view subtype density, each support case has equal mass and each of its fields has mass `1/(n_cases_subtype * n_fields_case_view)`. Use clipped cosine similarities, the registered temperature and stabilized float64 exponentiation, then normalize over the four types. The P control preserves the corresponding accepted density arithmetic. Query labels and other query fields do not determine a center or density weight.

## Composition, identity and selection

For every query field, compose `[P(B), P(M)*q_DC, P(M)*q_LC, P(M)*q_MC, P(M)*q_PC]` using the same fixed P binary probabilities and the arm's conditional malignant distribution. Average joint probabilities over fields within each view, then equally across the available views. Route malignant at pooled probability at least 0.5 and choose its malignant argmax; otherwise stop at benign. Do not average the two heads independently or replace the cascade by a five-class argmax.

Before validation selection, the `p_control` candidate at malignant temperature 0.1 must reproduce the parent's selected P validation probabilities within absolute tolerance 1e-10, zero relative tolerance, and identical decisions/confusion counts. This is the identity anchor, not an extra candidate. Require binary field/case marginals to match P across all candidates within 1e-12 and binary routes exactly. Unexplained route changes invalidate the typing-only claim; do not loosen the tolerance or adjust the threshold. Require finite values, probability row sums within 1e-12 and exact saved-array reload.

Select each arm's first maximum **validation mean-15 final-five macro F1**, with all five labels fixed, from temperature order 0.01/0.03/0.1/0.3. The primary comparison is the selected UNI2-h specialist against the separately selected P control. The prespecified development progression screen requires at least +0.02 macro F1 and nondecreasing final-five accuracy; report every class recall. It is a practical development screen, not a significance claim or permission to choose a different primary from the follow-up results.

The screen governs whether further development is justified; it does not suppress a negative arm or change which selected outcomes are reported. Complete the prespecified selected follow-up for both technically valid arms and retain both outcomes regardless of which validation score is higher.

Seal both selections, candidate artifacts and source/configuration/cache/input bindings in a new family-specific gate before either follow-up query. Do not overwrite the parent study's selection seal. Then score only the selected candidate from each arm on the unchanged 16-case role. This adds two selected retrospective pipelines, not another temperature sweep. Preserve the original parent metrics alongside the new comparison; no follow-up result changes the selected reader, temperature, router, source or progression screen.

## Checks, artifacts and budget

Run synthetic checks of field-path alignment and duplicate/missing rejection, exact P identity, fitting-only statistics, expected support counts/weights, finite normalized densities, constant router marginals, half-threshold routing, field permutation and inactive-view invariance, save/reload and first-tie selection. An independent CPU checker must replay the densities, eight validation candidates, both selections/seal and both selected follow-up archives. It must fail closed without accepted metrics on incomplete or inconsistent evidence.

Save dataset/split/cache/source/configuration revisions; path-alignment digests; fitting support and center hashes; all candidate field probabilities, case/subset records and metrics; parent-identity/router-invariance receipts; selected temperatures and sealed hashes; independent result/error reports; and every negative or failed outcome. Keep per-case identifiers, paths and feature arrays private. Validation contributes 2,040 dependent specimen/subset records and selected follow-up contributes 480; these are not new independent patients.

The bounded production allocation is **one CPU scheduler task, four CPUs, 24 GB, 30 minutes**. The independent CPU collector has the same ceiling. There are zero GPUs, image decodes, encoder forwards and optimized-head fits; fitting consists of frozen-cache statistics and labeled density support. No neural model is retrained. A separate training-only leave-whole-group-out router diagnostic may motivate later binary work, but it is outside this candidate family and cannot alter these fixed choices.

Both arms retain the parent's binary performance by construction; this study cannot reach >97% binary through specialist replacement. No >90% final-five guarantee, fresh clinical validation or novelty claim follows from registration or a favorable development result.
