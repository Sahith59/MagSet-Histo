# Native four-fit pilot: independent review

Prepared November 12, 2026. This review targets `docs/NATIVE_DEVELOPMENT_PILOT_SPEC.md` and the forthcoming native development implementation. It is a design/code review, not a native classifier result or a novelty certification.

## Design gates reviewed before implementation

The proposed four-arm, one-setting-per-arm experiment is a substantially more disciplined next step than transferring the earlier broad search to 16 new development patients. Its useful safeguards are the frozen patient quarantine, ImageNet ResNet provenance control, fixed projection/temperature/classifier setting, same-dimensional extra-feature controls, and retention of the complete baseline block. None guarantees a positive result or establishes novelty.

The following are implementation acceptance requirements, not measured native outcomes:

| Requirement | Exact expected behavior |
| --- | --- |
| Supervised roles | Use only 80 fit patients for fitting; predict only 16 development patients; produce zero sealed-patient classifier predictions. |
| Per-class support | Five fit and one development patient per canonical class, with labels 0–15. The 32 sealed patients remain outside every fitting/selection operation. |
| Fit-only moments | Per-view first/second field moments give equal weight to each fitting patient with that view. Development and sealed fields cannot fit normalization or auxiliary statistics. Missing views cannot contribute zero placeholders to these moments. |
| Natural incompleteness | Six fitting patients lack the 0.25-mpp view; preserve them without inventing observations or changing their labels/roles. |
| Subset rows and weights | 74 complete fit patients ×15 subsets +6 incomplete fit patients ×7 subsets =1,152 fitting rows. Every patient's derived weights sum to one; total weight80 and mass5 per class. |
| Expanded-row class weights | Use case subset weights with `class_weight=None`. Applying sklearn's row-count `balanced` rule would reweight classes according to incomplete-view row counts and defeat the intended equal-case/class mass. |
| Model budget | Exactly four fits, one per arm, all using fixed C0.1/L2/max_iter1000. No score-selected seed, projection width, temperature, C, view budget, or extra retry setting. |
| Added features | Arms2–4 add exactly32 projected features, share the same label-blind orthogonal projection, and preserve the identical baseline feature/mask block. |
| Expected classifier parameters | Baseline:16×2,052 coefficients +16 intercepts =32,848. Added-feature arms:16×2,084 +16 =33,360. Fixed projection values are stored constants, not learned classifier parameters. |
| Prediction coverage | Every arm saves16 development patients ×15 subsets =240 prediction records;960 across the four arms. No sealed prediction records are permitted. |
| Metric denominator | Explicitly fix all16 labels for macro F1/recall and probability-column order. Do not silently infer a varying class set. |
| Calibration metric | Specify multiclass Brier convention explicitly, preferably the mean per-patient sum of16 squared probability errors. A division by16 is a different reported scale. |
| Reload | Reconstruct predictions from the saved normalization/projection/model artifacts and compare with saved scores, not only an in-memory second evaluation. |

## Numerical and invariance requirements

The affinity field score for field i in observed view m should average peer-view contributions:

`a_mi = mean over j in S except m of log(mean over fields k in j of exp(cos(z_mi,z_jk)/0.2)))`.

Softmax is then applied **within view m**. Its weighted projected mean minus the unweighted projected mean forms the contrast; available-view contrasts are averaged. Cosine comparisons use L2-normalized copies. All arm value blocks should use the same unnormalized projected field representation, otherwise a hidden normalization difference can explain the comparison.

Required boundaries include zero/identical projected vectors, one field in a bag, absent bags, all-missing rejection, one-view exactly-zero cross-view contrast, finite log-mean-exp/softmax outputs, field/view permutation, and missing-content invariance. Peer contributions must be averaged rather than summed so the effective affinity temperature does not sharpen merely because more views are present. Path-hash random partitions must not depend on input order.

## Interpretation guard

A softmax-weighted mean minus a uniform mean is a **signed contrast**. It is not a partial transport plan, a calibrated matching probability, or a nonnegative unmatched-mass estimate. An identity such as `weighted + (uniform − weighted) = uniform` checks algebraic reconstruction; it does not establish morphological correspondence or mass conservation. The valid information-preservation statement is that the baseline feature block remains present. Adding features can still change the fitted coefficients and worsen predictions.

Equal added dimensions make the duplicate/random/affinity arms comparable in classifier coefficient count. They do not make the baseline and augmented classifiers identical in capacity or remove regularization/feature-scale effects. The duplicate-mean arm is useful precisely because redundant projected features can change effective L2 regularization. Report fitting feature norms and compute descriptively without using development scores to retrospectively rescale the experiment.

The decision rule should remain a feasibility gate: losing to the mean or same-dimensional controls stops the agreement-specific claim; a favorable development result permits stronger prespecified association-disruption/heterogeneity controls. It does not establish statistical significance, semantic explanation, novelty, or permission to inspect sealed outcomes. There is only one development patient per class.

## Implementation findings and repairs

The reviewed native path now satisfies the central design requirements:

- `load_cohort` verifies the pinned protocol/cache audit and retains only fit/development patients before any moment fitting or classifier construction. It verifies the 80/16 role counts and five/one per-class support.
- `fit_moments` rejects non-fitting roles and gives each available fitting patient equal moment mass. Missing views are omitted rather than zero-filled.
- `design` enumerates only each patient's actually available nonempty subsets and uses inverse-subset-count sample weights. The classifier explicitly uses `class_weight=None`.
- The projection is fixed, label-blind, and orthogonal; path sorting and path hashing make field-order changes irrelevant. The affinity implementation averages peer log-mean-exp scores and uses unit copies for cosine scores while keeping raw projected values in the contrast.
- All four fixed models must converge before any development scores are emitted. Convergence warnings become failures, and a surviving output marker prevents automatic overwrite/retry of the fixed run.
- Saved preprocessing is reloaded and compared exactly before development feature construction. Saved classifier coefficients then reconstruct probabilities and are checked against sklearn predictions. The independent prediction audit enforces exactly the trusted development patients, canonical truths, all15 subsets, and all four arms; sealed-role records are rejected.
- Metrics explicitly use16 labels. Brier uses the prespecified sum-of-squared-errors convention. Probability normalization now uses `rtol=0` with the specified absolute tolerance.

### P1 repaired: repeated full-shard decompression

The initial `load_cohort` accessed `saved['features']` inside the per-patient loop. NumPy's NPZ reader reloads/decompresses a member on each such access; a roughly9,000×2,048 embedding shard could be decompressed up to96 times per view. This was an actual avoidable performance issue, not a scientific-model failure.

The implementation now loads the shard array once into `values`, then indexes permitted patients from that array. Frozen label-blind reads remain allowed; no sealed bag is retained for statistics or prediction. This repair removes the redundant decompression without changing the intended experiment.

### P1 repaired: preprocessing was saved but not reloaded for inference

The first version verified saved coefficients against predictions on development features built from in-memory preprocessing. It now reloads `preprocessing.npz`, asserts exact equality for mean/std/projection, and actually uses those loaded arrays to build development features. This strengthens the save/reload guarantee without another fit or a development-dependent change.

### P2 repaired: implicit relative probability tolerance

The first probability-sum check specified `atol=1e-10` but left NumPy's default relative tolerance in effect. The check now explicitly sets `rtol=0`, and the metrics test includes a normalization-error regression. This is a strict audit repair; valid sklearn softmax probabilities were not observed failing the original check.

## Independent execution evidence

The reviewer ran the nine focused native-pilot tests in pinned Python3.11/NumPy1.26.4/SciPy1.15.2/sklearn1.6.1/Pillow11.1/Torch2.6.0. The first focused run passed all9 in approximately3.1s. A second focused run on the repaired snapshot passed all9 in3.533s; the source hashes below remained unchanged through that run. The suite covers fixed projection, fit-only moments, incomplete-view case weighting, field/view permutations and missing contents, signed contrast/one-view boundaries, explicit16-class metrics/Brier, sealed-role rejection, numeric coefficient reload, trusted prediction tampering, and a complete synthetic128-patient pipeline.

The synthetic pipeline checks exactly four fits,1,152 fitting rows, total weight80, class mass5, expected coefficient counts, and no reserved predictions. This is software execution evidence only. Its printed uniform-classifier F1 values are **not native dataset performance**.

## Remaining nonblocking limitations

1. The source specification still uses the phrase “conservation of matched/unmatched branches” in one checklist. Replace it with “baseline-block equality and signed-contrast reconstruction” to match the implemented mathematics and the clearer surrounding specification.
2. Reported feature-block RMS values are currently unweighted averages over expanded subset rows. They are diagnostic, not fitting weights or model-selection criteria. Label that convention or use case-weighted RMS if interpreting the statistic as patient-balanced feature amplitude. Do not rescale blocks after seeing development results in this pilot.
3. Equal coefficient counts among augmented arms do not prove that any favorable affinity result is agreement-specific. A positive outcome still requires the prespecified association-disruption/heterogeneity controls and a prior-art distinction. Only16 development patients support the first screen.

## Acceptance and scope

**Accepted for the single fixed four-fit development pilot after the repairs above and focused tests.** No sealed-label scoring, new hyperparameter grid, or repeated favorable-setting search is implied. The acceptance concerns reproducibility, role separation, numerical implementation, and conformance to the registered pilot—not novelty, significance, clinical validity, or permission to reveal the32-patient confirmation cohort.

Reviewed repaired source hashes:

- `magset/native_pilot.py`: `0cc180c4fc9016cea233edff2cf9165e0f9b7a618895f93db90bb7b6eac2d938`.
- `tests/test_native_pilot.py`: `b6047218f804e0cd67bce4dcefc789c0f280672d42c750869c1b302f6e040601`.
- Pilot specification: `e0f27e23e47031b5ed2936d951aac59cd7600caa58b1186df46fbe936dfdcb30`.
- Scheduler entry point: `b3afccc2c6372f3aa8cd838ee2dddf3f2265295f824bda2ab09d7331ca312920`.

These hashes identify the reviewed working snapshot before immutable experimental packaging. If another patch changes source semantics, record the new revision and focused verification in the closure ledger.

## Actual fixed-pilot result audit

The registered real pilot completed as scheduler job `4958291`, with experimental source `c1364c6202c2a5de7f9111b927ef0f41c94e0664`. This section audits its saved artifacts; it does not add a classifier fit, change a setting, or inspect the32-patient sealed cohort. The numerical `magset/native_pilot.py` hash remains the reviewed `0cc180c4fc9016cea233edff2cf9165e0f9b7a618895f93db90bb7b6eac2d938`.

### Independent checks performed

- The actual protocol file matches the registered SHA `f1baf1ac21f5f7e117768951619dc6f94a96ba63c72ad810f31fc103fe100c69`. The run's manifest hash matches that trusted protocol. The metadata retains80 fit,16 development, and32 sealed patients.
- All960 saved prediction records are unique by arm/patient/subset. Each arm contains exactly the16 trusted development patients at every one of15 nonempty subsets. All truths match the protocol; no sealed patient occurs in the prediction file.
- An independent confusion-matrix calculation, separate from the production metric helper, recomputed all60 arm/subset macro F1, accuracy, balanced accuracy, class recall, confusion matrices, and summed multiclass Brier values. Stored values agree within1e-12. Mean/worst-subset aggregation also agrees.
- All seven required artifact hashes match the actual files: start receipt, preprocessing, four classifier files, and predictions. The start receipt's fixed configuration/source agrees with the final report.
- Saved preprocessing has mean/std shape4×2,048, positive finite stds, and a2,048×32 orthogonal projection. The saved coefficient/intercept/class arrays are finite, have canonical16-class order and the expected dimensions/counts. The projection orthogonality check also passed in the pinned NumPy1.26.4 environment.
- The report records four completed fits,1,152 fitting rows, total fitting weight80, and approximately5 weight per class, consistent with the reviewed weighting construction and incomplete-view contract. Its preprocessing reload flag and source-level development-only data path support the recorded reload/role guarantees.

### Recomputed primary outcome

| Arm | Mean development subset macro F1 |
| --- | ---: |
| Mean + mask | 0.3340277778 |
| Mean + duplicate mean | 0.3395833333 |
| Mean + random contrast | 0.3340277778 |
| Mean + affinity contrast | 0.3381944444 |

**The preregistered agreement-specific improvement gate is not met.** Affinity trails the duplicate-mean control by0.0013888889. Full-view F1 and worst-subset F1 are identical across the four arms. Relative to mean-only predictions, duplicate/random/affinity change3/0/1 of240 decisions, affecting3/0/1 distinct development patients. The apparent primary-score differences therefore rest on very few decisions, not independent evidence across960 patients.

The proper outcome is a **no-go for an agreement-specific improvement claim from this fixed pilot**. It does not prove every affinity-based method is ineffective. It does mean the current result cannot justify claiming that agreement modeling beats a same-dimensional redundant-feature control, expanding a favorable-setting search, or revealing sealed outcomes. Preserve all arms and the negative decision in the professor evidence package.

### Post-run reporting repairs and regression distinction

`scripts/report_native_pilot.py` was added after the experimental run to audit and render the already saved outputs. The review identified three reporting-integrity gaps: an omitted artifact hash entry could evade verification, the report's manifest hash was not explicitly checked against the trusted protocol, and classifier dimensions were accepted from metadata alone. These have now been repaired without refitting: the reporter requires exactly the seven registered artifacts, checks the trusted manifest hash and start receipt, and independently inspects saved preprocessing/classifier arrays.

The reviewer executed the repaired reporter against temporary copies of the real artifacts in pinned NumPy1.26.4/SciPy1.15.2/sklearn1.6.1. Valid artifacts passed; deleting a model's hash entry and changing the reported manifest hash were each correctly rejected. No real classifier fitting or sealed prediction was performed. The generated [development results](../NATIVE_DEVELOPMENT_RESULTS.md) correctly identify the secondary16-class multi-organ task, report all arms/subsets, and apply the no-go decision without a significance or novelty claim.

The additional reporter assertions in `tests/test_native_pilot.py` were added **after launch**. They verify artifact/metadata rejection and public evidence generation; they are not extra real classifier candidates and do not alter the experiment's fixed four-fit budget. The original pre-launch test-file hash above is therefore expected to differ from the current post-run regression file.

The lingering “matched/unmatched conservation” wording in the specification was also replaced with “baseline-block equality and signed-contrast reconstruction.” The specification explicitly identifies that as an editorial clarification after the pilot. The immutable experiment retains its original specification hash `e0f27e23e47031b5ed2936d951aac59cd7600caa58b1186df46fbe936dfdcb30`; the clarification does not change model settings or outcomes.

Audited artifact hashes:

- Private pilot report: `7972dae0ab004fb98c52dc56747a778c627ba81a074c497ed8356aab29377612`.
- Saved predictions: `043d993caf37475b5413c23476cf63f64f3921ac2f9e77cd888c2764c0e21ad6`.
- Repaired reporter source at review: `0d8354914aa48e0283de03ff418efec60815754c1ca61b79d2c04e72845da5ca`.
- Post-run native test file at review: `cf738214a9b467b22be74a60ec0e9f320b6b598c6ea8f2d5d544dad20d7fed57`.
