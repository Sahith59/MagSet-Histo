# BreakHis milestone: independent review

Prepared November 12, 2026. Immediate scope is the BreakHis project deliverable for Professor Dong: a reproducible existing baseline, a raw-image prediction demonstration, and an honest measured report. Novel-method work, publication claims, and the secondary-dataset work are deferred from this milestone.

## Existing baseline audit

The reviewer checked all ten existing normalization-matched Phikon-v2 mean baselines: inner seeds 812/813, five provider trials each. Every report passed trusted manifest/protocol verification, including regenerated split memberships and case-to-class truth labels. A separate scalar confusion-matrix implementation recomputed macro F1 and accuracy from all 1,200 saved validation score records and agreed with stored values within 1e-12. No outer-role predictions were generated or scored.

The model is the frozen pinned Phikon-v2 encoder, mean field embeddings within each available magnification, fitting-case-only field-moment normalization, averaging available view features, four availability indicators, and an eight-class logistic classifier. Its regularization C was selected from the recorded four-value grid on the corresponding inner validation cases. It is an established project baseline, not a demonstrated new mechanism.

### Five-trial averages

| Inner split seed | Mean across 15 subsets: macro F1 | Mean across 15 subsets: accuracy | Full-view macro F1 | Full-view accuracy | Distinct validation provider cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 0.3842777778 | 0.4566666667 | 0.4083333333 | 0.4750000000 | 22 |
| 813 | 0.3168888889 | 0.3833333333 | 0.3083333333 | 0.3750000000 | 20 |

For plain-language reporting, full-view validation accuracy averages 47.5% for seed 812 and 37.5% for seed 813. Those percentages are not macro F1. The same cases recur across trials, and the union across both seeds contains 31 distinct validation provider cases; it is not 42 independent cases.

### Per-trial values

Every row has 46 fitting cases, eight validation cases, 28 assigned outer-role cases, and 120 validation prediction records (eight cases × 15 observed-view subsets). Each validation set has one case per subtype.

| Seed | Trial | Selected C | Mean subset macro F1 | Mean subset accuracy | Full-view macro F1 | Full-view accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 0.01 | 0.6041666667 | 0.6666666667 | 0.7083333333 | 0.7500000000 |
| 812 | 2 | 1.0 | 0.2566666667 | 0.2666666667 | 0.2500000000 | 0.2500000000 |
| 812 | 3 | 10.0 | 0.3091666667 | 0.4166666667 | 0.2500000000 | 0.3750000000 |
| 812 | 4 | 0.01 | 0.2597222222 | 0.3500000000 | 0.2916666667 | 0.3750000000 |
| 812 | 5 | 10.0 | 0.4916666667 | 0.5833333333 | 0.5416666667 | 0.6250000000 |
| 813 | 1 | 0.01 | 0.3444444444 | 0.4083333333 | 0.3333333333 | 0.3750000000 |
| 813 | 2 | 0.1 | 0.3255555556 | 0.3666666667 | 0.3333333333 | 0.3750000000 |
| 813 | 3 | 0.1 | 0.4075000000 | 0.4833333333 | 0.3750000000 | 0.5000000000 |
| 813 | 4 | 0.01 | 0.2958333333 | 0.4000000000 | 0.2916666667 | 0.3750000000 |
| 813 | 5 | 10.0 | 0.2111111111 | 0.2583333333 | 0.2083333333 | 0.2500000000 |

All summaries average the corresponding trial/subset metric values. Pooling all repeated predictions and then computing macro F1 is a different statistic and must not replace these values without a separate label.

## Conditions for accepting the raw-image demo

The demo should preserve the pinned encoder/checkpoint transform and canonical class order, use the saved per-scale normalization, and implement the same available-view mean plus mask. Input fields belong to one provider case and are not falsely paired by filename sequence across magnifications.

For a claimed exact validation replay, use all saved-case fields at each requested observed magnification and compare the re-encoded probabilities against the recorded case/subset prediction within a declared numerical tolerance. A single selected image per view or arbitrary user input is a different inference context; it can demonstrate functionality but cannot reproduce the all-field case-level score.

A default trial 1/seed 812 checkpoint is acceptable as a conventional fixed example, but trial 1 is the strongest initial split in this particular table. Show all ten runs in the report and do not imply that the example's 75% full-view accuracy is the overall project result. Choose the replay case from metadata deterministically before looking at correctness/confidence; a wrong example is still a valid demonstration if reported honestly. The validation replay path must reject fitting/outer-role cases when it claims validation-only evidence.

## Exact limitations to retain in the quick deliverable

1. This is eight-subtype, provider-case **development validation**. The subtype-qualified prediction unit is not an independently reverified clinical patient identity. Related numeric stems remain a grouping uncertainty.
2. Only eight cases support each trial's validation score; some subtype results across five trials use just two distinct cases. Repeating 15 masks does not add patients. Accuracy and balanced accuracy coincide for these balanced one-per-class validation sets.
3. C was selected on these same validation cases, and the wider project adaptively examined 175 development configurations. These are not unbiased final performance estimates or significance tests.
4. Provider trials are overlapping repeated holdouts. A case can be fitted or validated in another trial even when assigned an outer role in the chosen trial. The correct restriction is that no outer-role prediction has been scored and each run excludes its own outer cases—not that all BreakHis cases are globally untouched.
5. Frozen encoder preprocessing may crop/rescale tissue. The demo and baseline need identical preprocessing; visual attractiveness or one correct case does not establish robustness to unseen patients or acquisition settings.
6. The supplied historical 40× notebook accuracies and published binary-task accuracies are not interchangeable with these eight-class case-set scores. Do not claim 95% project accuracy or state-of-the-art improvement.
7. The current scores and split sensitivity indicate a useful reproducible baseline but weak/uncertain generalization. The immediate deliverable can be complete as a working baseline/demo/report while model improvement and paper novelty remain future research.

## Inference and report source review

The first source review inspected `magset/breakhis_predict.py` (SHA-256 `fb4eeea8cb9f6bcaddf3e7e5e3a67c6b154851377015942f9e8a1b96876638e1`), `scripts/report_breakhis_milestone.py` (SHA-256 `c309217ba0ff7c0ad08af9c84219a1ff36146fafc60094f1ad34a496b36238cd`), `tests/test_breakhis_predict.py`, and `slurm/breakhis_demo.sbatch`.

The inference implementation reconstructs the saved available-view mean plus mask, accepts only the fixed trial 1/seed 812 matched mean checkpoint, verifies trusted memberships/truths, checks the checkpoint checksum and class order, verifies complete cached fields, and replays all 120 saved validation case/subset predictions before raw-image encoding. The raw-image path verifies the manifest checksum and dimensions of every observed field and refuses paths outside the image root. The default case choice is the first sorted validation identifier. Its default all-four-view path replays all 15 subsets and stops if a probability differs from the archive by more than 1e-4. No fitting is present in this path. Four CPU inference/guard tests passed independently; these are software checks, not measured classification performance.

The report's baseline aggregation agrees with the independent table above. It labels the evidence as development validation, presents every trial and both seeds, and does not turn the example's 75% trial accuracy into an overall accuracy claim.

**Blocking integrity finding in the first reporter revision:** the optional demo guard checked only that error scalars were not greater than their tolerance, without requiring finiteness or nonnegativity. It also did not validate per-subset probability values, predicted labels, truth, or counts against the trusted selected case. A temporary synthetic corruption fixture was accepted as a verified raw-image demonstration despite having NaN in both error scalars and all probability vectors, an invalid predicted subtype, image count−1, and malformed field counts. This fixture was an artifact-integrity test; it did not encode images, fit a model, or score any additional cases. No generated fixture/report was retained in the repository.

Required repair before report acceptance: independently validate the demo's trusted case/truth, exact observed field and total image counts, pinned encoder/class order, finite nonnegative error scalars, and each eight-class probability vector. Recompute all 15 differences from the trusted archived selected-case probabilities and require the reported labels and maximum error to agree. Include a corruption regression, retain the demo artifact hash and model/source provenance, and distinguish source review from actual cluster replay success.

### Repair verification

The revised reporter at SHA-256 `4a0ff452a403848cada73421a7f7ca29c832cee16933f665a3563ff24bc8c154` closes the demonstrated defect. It binds the copied saved head by checksum/shape, requires the expected immutable source revision and inference-file hash, checks trusted truth/counts/encoder metadata, requires finite nonnegative error values, validates every eight-class probability distribution, and independently recomputes all 15 archived differences and the maximum. An independent temporary test accepted a coherent synthetic artifact and rejected 13 separate corruptions: nonfinite raw/cached errors, wrong truth, negative image count, wrong field counts, wrong model/source hashes or revision, outer-scoring flag, nonfinite or mismatched probabilities, wrong predicted label, and nonfinite per-subset error. These checks fit no models and encode no images. They verify report-artifact handling, not the actual cluster demonstration.

## Actual execution artifact acceptance

The saved raw-image execution artifact for SLURM job `4959781` is now available and independently checked. It uses the first sorted validation provider case of the fixed trial 1/seed 812 run, whose trusted subtype is adenosis. Its private case identifier remains outside this public review. Direct manifest counting confirms 121 original fields: 29 at 40×, 30 at 100×, 32 at 200×, and30 at 400×. The selected case belongs to that run's validation group and to neither its fitting nor its excluded outer group.

The reviewer independently checked all 15 saved raw-image probability vectors for finite normalized eight-class values, canonical class order, predicted-label consistency, and equality to their corresponding archived validation vectors within the registered tolerance. Recomputing the 15 maximum absolute probability differences gives an overall maximum of `5.517742757144006e-08`, exactly matching the saved execution record and comfortably below `1e-4`. The artifact records a prior replay of all 120 cached validation records with maximum difference `4.440892098500626e-16`; this cached reconstruction was enforced by the reviewed execution code, rather than newly repeated from full feature caches in this local review. The reviewer performed no new image encoding, fitting, or outer-role scoring.

The actual copied head has the expected normalization/classifier shapes, finite values, positive normalization scales, and canonical class order. The source Git object, immutable source archive, current inference file, and artifact's inference-file hash agree. Manifest, protocol, metrics, predictions, and public ledger references also agree with their trusted artifacts.

| Evidence | Checked value |
| --- | --- |
| Immutable execution source | `9448a7e319167b7417670a882552edb372603fd3` |
| Inference file SHA-256 | `fb4eeea8cb9f6bcaddf3e7e5e3a67c6b154851377015942f9e8a1b96876638e1` |
| Saved head SHA-256 | `318a38488b84845b0731be38fe7d8d55e46739526f0ef196a4ba480152db8f21` |
| Private demo artifact SHA-256 | `e5d3f445282b421910f246adcc1683951f0115cbab3954700d9dd5de4f9b136b` |
| Final reviewed reporter SHA-256 | `53bc4f5eed7fcc90790d4cbd47719905dee47d89db4403af4d544742ddf83979` |

The final reporter's two artifact-guard regression tests and four inference tests pass independently. Its later prose/professor-brief additions are separate from the earlier guarded-source review. The generated first-results page and Professor Dong brief retain both seeds' modest scores, repeated-case limitations, absence of a globally fresh confirmation cohort, and the distinction between execution parity and predictive improvement.

## Review status

**The BreakHis baseline milestone is accepted:** ten trusted saved baseline runs, all 1,200 development score records, the saved-checkpoint raw-image execution artifact, and its report checks support a reproducible working baseline/demo deliverable for Professor Dong. The first reporter's demonstrated integrity defect has been repaired. This acceptance does not establish strong final predictive performance, verified clinical identity grouping, independent confirmation, a novel contribution, or publication readiness. Those research questions remain unfinished.

Trusted manifest SHA-256: `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31`. Trusted provider-protocol SHA-256: `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134`.
