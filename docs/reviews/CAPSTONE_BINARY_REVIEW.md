# Retrospective binary secondary readout: independent actual-output review

November 14, 2026. **Accepted as a correctly calculated retrospective secondary diagnostic.** This is a different label endpoint from the eight-subtype primary task; it does not establish an eight-class improvement, justify choosing a new winner, or supply a clinical test. No new fit, encoder/classifier forward, raw-image decoding, scheduler job or outer-role scoring was performed for this review.

Reviewed binary JSON SHA256: `7502fe88b95869e964861d36004d7a46ebd7b342ad042ca25d47626612450ae1`. The specification [CAPSTONE_BINARY_SPEC.md](../CAPSTONE_BINARY_SPEC.md) was verified byte-for-byte against commit `cdfb58f`. It explicitly describes specification before the binary-summary computation **after the eight-class outcomes had already been examined**. This is transparent retrospective analysis, not an independent preregistered confirmation test.

## Independent actual-output audit

Used a standalone script, without calling the production binary reader, metric, summary or collector functions. Repeated the same checks with the pinned lightweight environment (Python3.11, NumPy1.26.4, SciPy1.15.2, scikit-learn1.6.1). No model was refitted.

- Independently checked all30 actual metrics reports and saved eight-class case-prediction files: ten frozen, ten adapted and ten supervised. Rehashed report/prediction artifacts against the accepted parent ledgers and the binary artifact ledger. Frozen-reference report and prediction hashes agree in both parents. Checked the expected encoder/readout identities and the adapted/supervised training source/configuration identities. Parent hashes are adaptation `c26398dcfbeac6ac76f737ab686a987f82c2ea6874a45132fe9a0a7fc8b5637b` and supervised `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`.
- Checked the actual manifest/protocol hashes and independently reconstructed subtype-qualified identities, labels, each trial's outer roles and the two seeded class-covered inner choices. Verified46 fitting/eight validation/28 excluded cases, identical validation memberships across all three pipelines, and exactly one validation case per subtype. Every saved prediction's subtype truth matches the trusted manifest.
- Independently checked every manifest row's provider `broad_class` against its subtype. The fixed malignant indices are1/3/4/5 and benign indices0/2/6/7. This is the dataset's grouping. Each binary score was recomputed from its original saved eight-probability vector, with finite/bounds/sum checks and exact120-record coverage per run over eight cases and15 subsets.
- Applied the fixed malignant decision `p_malignant >= 0.5`, including ties. There were no exact0.5 scores among the3,600 actual events. No threshold search, score inversion, calibration, model selection or retraining occurred in the reviewed analysis.
- Independently used scikit-learn `confusion_matrix`, `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `roc_auc_score`, `average_precision_score` and `brier_score_loss`; computed sensitivity, specificity and support directly from counts. This separately checks the production tied-rank ROC-AUC and grouped-threshold AP implementations.
- Recomputed every trial's15 subset metrics, equal-subset means, full-view metrics, separately defined worst subset for each metric and pooled case/subset metrics. Recomputed both five-trial arrangement summaries for all three pipelines, including every mean confusion/support array. Five-trial/equal-subset averages are not substituted with nonlinear metrics computed on pooled observations.
- Rechecked the analytic always-malignant reference: each run contains four benign/four malignant validation cases, so its accuracy/balanced accuracy is50%, sensitivity1, specificity0 and binary macro F1 is1/3. This is an analytic reference, not a newly fitted model.

All7,508 checked metric scalars/arrays agreed; maximum absolute discrepancy was `2.220446049250313e-16`. All3,600 saved case/subset events and the complete30-run matrix were covered. No numerical, provenance, role, truth or coverage blocker remained.

## Checked summaries, with all pipelines retained

Each row averages15 view subsets per trial and then five overlapping trials. Macro F1 is the mean of the benign and malignant F1 values; it is not accuracy.

| Pipeline | Arrangement | Binary accuracy | Binary macro F1 | Malignant sensitivity | Benign specificity | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen Phikon |812|87.00%|0.866624|98.67%|75.33%|0.961667|
| Frozen Phikon |813|80.67%|0.787744|91.33%|70.00%|0.940833|
| Adapted Phikon |812|86.50%|0.861714|98.67%|74.33%|0.965833|
| Adapted Phikon |813|82.00%|0.804660|93.33%|70.67%|0.955833|
| Supervised ResNet |812|84.67%|0.845026|94.33%|75.00%|0.887500|
| Supervised ResNet |813|82.17%|0.813104|87.67%|76.67%|0.921667|

**ROC AUC is a ranking measure, not accuracy or a percentage of correctly classified cases.** These ROC AUC values must not be advertised as approximately94–97% accuracy. The corresponding thresholded accuracies are explicitly listed above. Average precision likewise summarizes ranking across score thresholds rather than accuracy at0.5. Positive-class Brier here is `mean((p_malignant-y)^2)`; it differs from the eight-class sum-of-squared-probability error.

Frozen full-view binary accuracy is87.5%/77.5%, with F1 of0.873016/0.750418. Its mean of trial-specific worst-subset accuracies is80.0%/72.5%. These are descriptive summaries of different fixed input availability conditions. The worst subset can differ by metric; the analysis does not select a new readout from those subsets.

## Distinct support, privacy and interpretation

Independently verified22 distinct validation cases in812 (10 benign/12 malignant) and20 in813 (10/10). Their union is31 distinct cases (14 benign/17 malignant). Each pipeline contributes1,200 repeated case/subset events,600 per arrangement; all three together contribute3,600 events. Those repeated predictions and the three model variants do not create3,600 independent patients. Each individual trial/subset has four cases per broad class.

Inspected the complete reviewed aggregate JSON against every manifest provider-case identifier and original image path. None occur. It contains no local/cluster machine paths or individual case-probability rows. Aggregate numbers, support arrays and cryptographic artifact hashes at this exact reviewed hash are suitable for public evidence; private per-case inputs remain private.

The frozen model's87.0%/80.67% broad-label accuracy and its45.7%/38.3% eight-subtype accuracy answer different questions. Collapsing several subtypes into benign/malignant can count within-broad-class subtype errors as correct. It does not improve the unchanged eight-class model, resolve zero phyllodes subtype recall, or pass any failed primary improvement gate. Selected models were originally chosen using eight-class validation; none is adopted as a new binary winner here.

The original reference paper's86.22% binary image result uses a different image-level protocol and is not an exact paired benchmark for this grouped case analysis. These secondary scores do not reproduce that paper or the historical approximately95% eight-class EfficientNet notebook. All82 cases have entered development somewhere, validation trials overlap, and clinical identifier relationships remain unresolved. No globally fresh confirmation cohort or clinical validity claim follows from this analysis. Keep all three pipelines in the report, retain the existing eight-class default and present this as a bounded retrospective secondary result.

## Review limits

This review accepts source-bound saved-score arithmetic, identities, aggregate metrics and the inspected privacy boundary. It does not replay the original encoders, retrain the original classifiers, independently verify their full optimization trajectories, or establish causal reasons for the binary/subtype gap. The specified retrospective chronology is documented in the registered specification; it must not be relabeled as analysis conducted before any outcome inspection. Any later changed threshold, selected pipeline or changed cohort would require a newly labeled analysis rather than inherit this acceptance.
