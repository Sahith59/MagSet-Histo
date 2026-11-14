# BreakHis project report: current evidence draft

Working report for review with Professor Dong, November14,2026. The evidence package is due November17–18 for a presentation around November20; exact day and rubric are unspecified. All planned comparisons and independent artifact reviews are complete. It is not a submission-ready manuscript or a novel-method claim. The older [methods draft](methods-draft.md) records earlier experimental rounds.

## Question

Can we classify a breast-tissue case into one of eight tumor subtypes when we have different combinations of microscope magnifications? We use images at40×,100×,200× and400×. We do not assume that similarly numbered images show precisely the same tissue location.

## Data and task

The obtained BreakHis release contains7,909 RGB images and82 subtype-qualified provider cases. The eight targets, in canonical model order, are adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor and tubular adenoma. Output is eight probabilities and the highest-probability subtype for a case.

Every provider case has all four views. Our15 input settings deliberately withhold different combinations; this does not measure how often views are naturally missing in clinical practice.

## Evaluation contract

Five existing overlapping provider holdouts are retained. For each, inner seeds812 and813 select one validation case per subtype from the54 training-side cases. Each model therefore uses46 fitting cases, eight validation cases and excludes28 outer-role cases. Both inner arrangements and all five trials are reported. No outer-role images are scored by these models.

Primary score is macro F1 averaged equally over the15 nonempty view combinations. Macro F1 gives each subtype equal influence; it is not accuracy. We also report case accuracy, balanced accuracy, full-view and worst-combination F1, subtype recall/confusion, Brier error and cross-entropy. Validation contains one case per class, so its case accuracy equals balanced accuracy. Field metrics, when included, are labeled separately.

There are22/20 distinct validation cases in the two arrangements and31 in their union. Repeated trials, fields and input combinations are dependent observations. All82 cases have entered development somewhere; there is no globally untouched BreakHis confirmation cohort. Provider identity is not independently reverified clinical-patient identity.

## Completed pipelines and findings

The retained reference uses frozen Phikon-v2 image features, within-view means, fitting-only normalization, available-view mean plus availability flags and a class-balanced logistic classifier. Four fixed regularization choices are selected using validation mean-subset macro F1.

| Pipeline | Split812 F1 / accuracy | Split813 F1 / accuracy | Status |
| --- | --- | --- | --- |
| Frozen Phikon-v2 mean reference |0.3843 /45.7%|0.3169 /38.3%|Ten runs and raw-image replay verified |
| Last-block Phikon adaptation, then same case head |0.3840 /45.7%|0.3205 /38.8%|Ten runs verified; prespecified improvement gate failed |
| Fully supervised ResNet-50 reference |0.3084 /37.0%|0.3006 /40.0%|Ten fits and full checker passed; improvement gate failed |

Scores are five-trial means of the15-subset endpoint. The [baseline report](../docs/BREAKHIS_FIRST_RESULTS.md) and [adaptation report](../docs/BREAKHIS_ADAPTATION_RESULTS.md) retain all trials and class weaknesses. The earlier original-image execution replay reproduced the selected baseline's archived probabilities within5.52e-08.

The [saved-model diagnosis](../docs/BREAKHIS_DIAGNOSIS_RESULTS.md) found98.5–98.7% fitting balanced case accuracy versus38.8–45.7% validation in adapted case heads. The auxiliary image head also showed a large gap. Neither fixed auxiliary aggregation probe rescued validation F1. This supports weak generalization under these roles, without identifying one sole cause.

## Completed registered supervised comparison

The supervised reference initializes torchvision ResNet-50 from fixed ImageNet1K_V2 weights. A2,048-dimensional pooled image representation feeds Dropout(.3) and an eight-class linear head. Epochs1–2 train only the head; epochs3–20 update the full backbone and head. Pretrained BatchNorm running buffers remain fixed; affine parameters train after warmup.

AdamW uses backbone/head learning rates1e-4/1e-3, weight decay1e-4 and gradient clipping1. After warmup, a fixed cosine multiplier decreases from1 to0.1. Label-smoothed cross-entropy(.1) trains on batch32. Each epoch makes exactly as many weighted replacement draws as fitting fields: equal expected mass per class, case within class, zoom within case, and field within zoom. This balances practice; it does not create new independent cases.

Fitting-only flips, quarter-turns and mild color changes are followed by whole-field bicubic resizing to224×224 and fixed ImageNet normalization. Resizing changes the original rectangular aspect ratio. Canonical scoring has no augmentation. Average field probabilities within each view, then give available views equal mass. There is no logistic refit or calibration step.

Select the earliest highest mean-subset validation F1 at epochs5/10/15/20; complete all20 epochs. Save all four candidates and selected fitting/validation field scores. The [registered specification](../docs/BREAKHIS_SUPERVISED_SPEC.md) defines the complete recipe and descriptive improvement screen. Four checkpoint opportunities and four logistic regularization choices are different search spaces, not identical optimization budgets. This is a complete-pipeline comparison, not an isolated fusion ablation.

## Published context and interpretation

Public eight-class results exceed our present exploratory scores, but their grouping, training budgets and endpoints differ. Static inspection of one author's released code/splits found material incompatibilities for joint four-view evaluation. [Benchmark context](../docs/BREAKHIS_PUBLIC_BENCHMARKS.md) and [pinned reference audit](../docs/BREAKHIS_REFERENCE_AUDIT.md) keep author-reported numbers distinct from our direct comparisons. No exact published-method reproduction has been completed.

Each fitting set contains one adenosis and one phyllodes provider case. Augmentation and many fields cannot substitute for independent rare-class diversity. Existing results do not establish reliable clinical classification, biological correspondence, method novelty or external confirmation. The immediate deliverable is a transparent, reproducible BreakHis project study with complete positive and negative evidence. Final claims and a credible confirmation design require review with Professor Dong.

## Completion checklist

- [x] Verified baseline, original-image replay, fixed adaptation and saved-model diagnosis.
- [x] Coverage/reference audit; supervised recipe registered before outcomes; technical gate passed.
- [x] Complete and verify all ten supervised fits; report every trial, both arrangements, rare-class support, score gaps and runtime/model size. [Full negative result](../docs/BREAKHIS_SUPERVISED_RESULTS.md).
- [x] Independently review all actual numerical outputs; update results and professor handoff without selecting a favorable split. Full model/optimizer verification remains remote-checker evidence.
- [ ] Agree on acceptable grouping/claim scope and a credible final-evaluation route before making stronger research claims.

No conference submission, new dataset access request or message to Professor Dong has been sent.

## Capstone endpoint alignment and completed regional comparison

The original supplied notebooks saved93.73%/94.99% on40× image-random reused validation; they did not implement the present case/missing-view task. [Original/current contract](../docs/CAPSTONE_BEFORE_AND_NOW.md) prevents describing this as a measured model drop. A fixed, non-fitting image-endpoint analysis of accepted ResNet validation arrays reports every magnification and a40× case-aggregation bridge; all ten runs and source hashes were independently rechecked. [Results](../docs/CAPSTONE_BRIDGE_RESULTS.md).

The conventional regional-view comparison is complete: original Phikon central-view features versus a fixed50/50 blend with four regional features, unchanged official processor and frozen weights. All four GPU shards and20 fixedC0.01 heads completed, followed by independent cache and numerical checks. Against its matched fixed-C control, F1 changes−.0058/+.0059 with zero phyllodes recall; the gate failed. The historical retained reference used a different C-selection budget and is kept separate. [All results](../docs/BREAKHIS_MULTICROP_RESULTS.md). No follow-on sweep.

## Benign/malignant secondary analysis

A fixed retrospective analysis sums the probabilities for ductal, lobular, mucinous and papillary classes. Score≥0.5 predicts malignant, otherwise benign. It uses existing outputs with no new fitting, encoder execution, threshold selection or calibration. The eight-class primary task is retained. The outcome was computed after eight-class results were known, on reused exploratory validation, not a new confirmation cohort.

| Earlier pipeline | Binary accuracy812 /813 | Binary macro F1 | Malignant sensitivity | Benign specificity |
| --- | --- | --- | --- | --- |
| Retained frozen Phikon |87.00% /80.67%|.8666 /.7877|98.67% /91.33%|75.33% /70.00%|
| Last-block adaptation |86.50% /82.00%|.8617 /.8047|98.67% /93.33%|74.33% /70.67%|
| Supervised ResNet |84.67% /82.17%|.8450 /.8131|94.33% /87.67%|75.00% /76.67%|

These average all15 subsets and five trials. The broad group is often correct while exact subtypes within it are confused. No binary winner is selected retrospectively. The retained model's mean AUROC.9617/.9408 measures ranking, not accuracy. Neither binary accuracy nor AUROC is an improved eight-class score. The original paper's86.22% binary image result has a different protocol and is not directly beaten here. Independent replay accepted30reports/3,600dependent events and7,508metric checks with maximum difference2.22e-16. [Complete results and limits](../docs/CAPSTONE_BINARY_RESULTS.md).

## Working demonstration and delivery

A fresh saved-model replay4971403 processed121 original fields in39seconds. All15 view outputs reproduced archived validation probabilities within6.05e-08, with no fitting or outer scoring. The fixed lexicographic example was not selected for a correct prediction. [Interactive saved-output viewer](../docs/evidence/CAPSTONE_PREDICTION_REPLAY.html) · [Actual inference instructions](../docs/CAPSTONE_DEMO.md). The viewer is not a live image-upload application and one example does not estimate accuracy.

Freeze the reference for the capstone package. Complete presentation/report review and rehearsal by November17–18; no novel mechanism, new-patient confirmation or conference-ready claim follows from successful execution. Research-paper development requires a separate evidence and confirmation plan.
