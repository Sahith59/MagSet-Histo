# Fixed retrospective benign/malignant secondary readout

November14,2026. This analysis is specified before computing its binary summaries, **after the eight-class validation outcomes were already inspected**. It is a retrospective secondary diagnostic, not a preregistered independent test or a replacement for the eight-subtype primary endpoint.

The original supplied paper reports binary benign/malignant accuracy. The project plan also already defines the sum of malignant subtype probabilities as a binary secondary score. We now report that different question explicitly so the capstone can separate broad-label detection from fine subtype recognition. No high score is promised or used to choose the task, threshold or model.

## Inputs and fixed rule

Use only the three previously accepted complete pipelines: frozen Phikon reference, last-block Phikon adaptation and supervised ResNet-50. Bind all30 saved report/prediction sets to trusted manifest/protocol and both accepted parent comparisons: adaptation SHA256 `c26398dcfbeac6ac76f737ab686a987f82c2ea6874a45132fe9a0a7fc8b5637b`, supervised SHA256 `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`. Frozen reference hashes must agree in both parents. The new regional-view experiment is reported separately.

Verify each manifest subtype agrees with its provider `broad_class`. In the canonical eight-class order, benign indices are0/2/6/7 (adenosis, fibroadenoma, phyllodes tumor, tubular adenoma) and malignant indices1/3/4/5 (ductal, lobular, mucinous, papillary carcinoma). This is the dataset's grouping, not a universal clinical statement about every tumor with those names.

For each saved eight-probability case vector, sum the four malignant entries. Predict malignant at a fixed score≥0.5, including an exact tie; otherwise benign. No threshold search, calibration, inversion, regrouping, new fit, classifier/encoder forward, role change or outer prediction. Selected models were chosen for eight-class F1, not binary performance.

## Reporting

Keep all three pipelines, both inner arrangements, five trials and all15 view combinations. Each validation partition contains four benign and four malignant cases. Report accuracy, balanced accuracy, binary macro F1, malignant sensitivity, benign specificity, two-class confusion/support, tied-rank ROC AUC, grouped-threshold average precision and positive-class Brier `mean((p_malignant-y)^2)`. Cross-entropy is omitted because old probability archives do not retain stable log probabilities for underflowed entries. The constant-majority malignant reference has50% validation accuracy here; it is an analytic reference, not a newly trained model.

Report every trial/subset, equal-subset mean, all-view and worst-subset results, distinct case support and repeated-observation counts. Verify the full120-record coverage per run, source/truth/probability integrity and independent metric arithmetic before publication. Publish aggregate data and hashes only.

Do not choose a binary winner, change the retained eight-class model or declare its failed primary gate passed based on this analysis. Even a higher binary score remains reused exploratory provider-case validation:31 distinct cases across both arrangements, overlapping trials, uncertain clinical identity and no globally fresh cohort. The earlier paper's86.22% is still a different image-level protocol and is not an exact paired benchmark. Novelty and clinical confirmation are not established.
