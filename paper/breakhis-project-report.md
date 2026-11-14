# BreakHis project report: current evidence draft

Working report for review with Professor Dong, November13,2026. This is a project-progress draft; the supervised comparison is still running. It is not a submission-ready manuscript or a novel-method claim. The older [methods draft](methods-draft.md) records earlier experimental rounds.

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
| Fully supervised ResNet-50 reference |Pending|Pending|Ten fits submitted; complete artifact check required |

Scores are five-trial means of the15-subset endpoint. The [baseline report](../docs/BREAKHIS_FIRST_RESULTS.md) and [adaptation report](../docs/BREAKHIS_ADAPTATION_RESULTS.md) retain all trials and class weaknesses. The original-image execution replay reproduced the selected baseline's archived probabilities within5.52e-08.

The [saved-model diagnosis](../docs/BREAKHIS_DIAGNOSIS_RESULTS.md) found98.5–98.7% fitting balanced case accuracy versus38.8–45.7% validation in adapted case heads. The auxiliary image head also showed a large gap. Neither fixed auxiliary aggregation probe rescued validation F1. This supports weak generalization under these roles, without identifying one sole cause.

## Registered supervised comparison

The next reference initializes torchvision ResNet-50 from fixed ImageNet1K_V2 weights. A2,048-dimensional pooled image representation feeds Dropout(.3) and an eight-class linear head. Epochs1–2 train only the head; epochs3–20 update the full backbone and head. Pretrained BatchNorm running buffers remain fixed; affine parameters train after warmup.

AdamW uses backbone/head learning rates1e-4/1e-3, weight decay1e-4 and gradient clipping1. After warmup, a fixed cosine multiplier decreases from1 to0.1. Label-smoothed cross-entropy(.1) trains on batch32. Each epoch makes exactly as many weighted replacement draws as fitting fields: equal expected mass per class, case within class, zoom within case, and field within zoom. This balances practice; it does not create new independent cases.

Fitting-only flips, quarter-turns and mild color changes are followed by whole-field bicubic resizing to224×224 and fixed ImageNet normalization. Resizing changes the original rectangular aspect ratio. Canonical scoring has no augmentation. Average field probabilities within each view, then give available views equal mass. There is no logistic refit or calibration step.

Select the earliest highest mean-subset validation F1 at epochs5/10/15/20; complete all20 epochs. Save all four candidates and selected fitting/validation field scores. The [registered specification](../docs/BREAKHIS_SUPERVISED_SPEC.md) defines the complete recipe and descriptive improvement screen. Four checkpoint opportunities and four logistic regularization choices are different search spaces, not identical optimization budgets. This is a complete-pipeline comparison, not an isolated fusion ablation.

## Published context and interpretation

Public eight-class results exceed our present exploratory scores, but their grouping, training budgets and endpoints differ. Static inspection of one author's released code/splits found material incompatibilities for joint four-view evaluation. [Benchmark context](../docs/BREAKHIS_PUBLIC_BENCHMARKS.md) and [pinned reference audit](../docs/BREAKHIS_REFERENCE_AUDIT.md) keep author-reported numbers distinct from our direct comparisons. No exact published-method reproduction has been completed.

Each fitting set contains one adenosis and one phyllodes provider case. Augmentation and many fields cannot substitute for independent rare-class diversity. Existing results do not establish reliable clinical classification, biological correspondence, method novelty or external confirmation. The immediate deliverable is a transparent, reproducible BreakHis project study with complete positive and negative evidence. Final claims and a credible confirmation design require review with Professor Dong.

## Completion checklist

- [x] Verified baseline, original-image replay, fixed adaptation and saved-model diagnosis.
- [x] Coverage/reference audit; supervised recipe registered before outcomes; technical gate passed.
- [ ] Complete and verify all ten supervised fits; report every trial, both arrangements, rare-class support, score gaps and runtime/model size.
- [ ] Independently review actual outputs; update results and professor handoff without selecting a favorable split.
- [ ] Agree on acceptable grouping/claim scope and a credible final-evaluation route before making stronger research claims.

No conference submission, new dataset access request or message to Professor Dong has been sent.
