# MagSet-Histo

BreakHis is our main dataset. We are building a breast histology classification project for Professor Dong's master's capstone, with a research paper as a later goal. TCGA follow-up is deferred.

The system first decides **benign or malignant**. Benign is the final answer; malignant images proceed to a specialist that chooses ductal, lobular, mucinous or papillary carcinoma. We use the available 40×/100×/200×/400× images of each provider case and evaluate all 15 nonempty zoom combinations. [Task and architecture](docs/DUAL_STEP_SCOPE.md) · [Plain-language training guide](docs/TRAINING_STRATEGY.md).

**Latest verified result:** decision-specific centering has our highest observed macro F1 among two-step pipelines, **.6027 / .6025**, with **76.33% / 75.67% final five-category accuracy**. With all four zooms available, accuracy is 85% in both arrangements; that is a separate endpoint. Changes are mixed: the earlier three-crop model still has higher second-arrangement accuracy at 76.83%. **94–95% final accuracy has not been achieved.** [Latest results](docs/BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Professor brief](docs/evidence/PROFESSOR_BRIEF.md).

The complementary GenBio reader and whole-case mean/attention round is complete and verified. Neither consistently improves the requested two-step endpoint. All 40 GPU training trajectories and ten paired GenBio classifier jobs completed; all 370 neural tests passed. [Completed execution and decision](docs/BREAKHIS_NEXT_ROUND_STATUS.md).

## Current five-category results

Accuracy/F1 below averages all 15 zoom combinations and five overlapping trials in each arrangement. These are reused provider-case development scores, with 31 distinct validation keys overall and unresolved clinical identity. They are not an untouched final patient test. Every row uses the same 46 fitting/eight validation/28 excluded roles and explicit two-step final labels.

| Two-step pipeline | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| Phikon-v2, case means |65.83% /68.83%|.4975 /.5325|
| EfficientNetV2B0, case supervision |46.50% /53.33%|.2346 /.3108|
| Phikon-v2, field supervision |66.67% /69.50%|.4975 /.5207|
| Midnight-12k, case means |72.00% /71.33%|.5558 /.5350|
| Midnight-12k, field supervision |73.67% /70.00%|.5732 /.5017|
| C: single-center similarity |75.67% /75.00%|.5897 /.5643|
| D: adapter recipe |68.33% /59.83%|.4944 /.3754|
| E: fixed equal-probability vote |74.17% /72.17%|.5781 /.5326|
| F: three-crop similarity |75.50% /76.83%|.5925 /.5799|
| N: native-pixel similarity |60.00% /61.00%|.4394 /.5014|
| P: shared class-balanced center |76.17% /75.33%|.5992 /.5778|
| P: decision-specific centers |76.33% / 75.67%|**.6027 /.6025**|
| GenBio-PathFM similarity |70.50% /75.17%|.5837 /.5813|
| Whole-case learned mean |70.17% /71.83%|.5199 /.5094|
| Whole-case gated attention |71.67% /70.50%|.5453 /.5234|

GenBio's one-stage control scores 71.33%/77.83%, F1 .5960/.6406. Its stronger 813 result is a different readout, with weaker 812 accuracy; it does not establish a two-step gain. [GenBio results](docs/BREAKHIS_GENBIO_RESULTS.md) · [Whole-case results](docs/BREAKHIS_CASE_MIL_RESULTS.md).

All comparisons above are complete and verified. More pixels and extra adapter training did not automatically help. Decision-specific centers produce modest F1 gains, with accuracy and recall tradeoffs; both centering modes fail the registered larger-improvement screen. Their direct-five controls are exact duplicates, leaving four structurally unique direct settings and eight dual settings across the mode family. No novel mechanism or 95% result is established. [Native negative result](docs/BREAKHIS_NATIVE_RESULTS.md) · [Centering evidence](docs/BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Full scorecard](paper/breakhis-five-class-report.md).

The main weakness remains malignant typing. The latest head-specific model scores 94.83%/93.00% on benign/malignant routing, but 58.33%/62.33% on true-malignant subtyping. Binary accuracy is not final five-category accuracy. Only BreakHis images/labels are fitted/evaluated; public pathology pretraining is external transfer learning.

The supplied notebooks saved 93.73%/94.99% on eight-type 40× individual images with image-random, reused validation. Their task, prediction unit and split differ from the current case experiment. [Before/now explanation](docs/CAPSTONE_BEFORE_AND_NOW.md). Original notebooks and earlier results remain historical evidence.

## Earlier eight-class BreakHis baseline numbers

| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |
| --- | ---: | ---: | ---: |
|812|0.3843|45.7%|47.5%|
|813|0.3169|38.3%|37.5%|

These are exploratory provider-case validation results, averaged over five overlapping trials with46 fitting/8 validation/28 excluded outer-role cases each. F1 is not accuracy. Both splits together involve 31 distinct validation cases; 1,200 saved case/subset records do not represent 1,200 independent patients. All 82 provider cases entered development somewhere, and some clinical identifier relationships remain unresolved. No globally fresh BreakHis confirmation cohort remains.

The [complete first-results report](docs/BREAKHIS_FIRST_RESULTS.md) shows every trial, class weaknesses, provenance and execution checks. The baseline uses a frozen Phikon-v2 image reader, within-view mean pooling, saved fitting-only normalization, available-view mean plus mask, and a logistic classifier. The working repository name does not imply that this baseline is a novel transformer.

## Immediate deliverable

Finish the five-category capstone report, presentation and demonstration using the complete accepted scorecard and representative errors. The working materials target is November 17–18, with presentation around November 20; the exact date remains unconfirmed. Preserve every negative comparison and state what the model can and cannot answer. [Next tasks](docs/NEXT_STEPS.md) · [Project report](paper/breakhis-five-class-report.md).

The original six supplied references remain unchanged and excluded from public source. Raw data, weights, provider identifiers, private cluster operations and conversation memory are also excluded. Source/configuration revisions, audits and complete aggregate evidence are recorded for reproducibility.

## Run checks

```bash
python3 -m pip install numpy==1.26.4 scipy==1.15.2 scikit-learn==1.6.1 Pillow==11.1.0
python3 -m unittest discover -s tests -v
```

The full neural checks additionally need torch==2.6.0 and torchvision==0.21.0. The latest full pinned suite passed277 tests. The real raw-image replay uses the existing pinned GPU environment, original dataset, pretrained encoder and saved baseline head; these artifacts are not bundled in Git. Submit heavy work through SLURM using [the cluster guide](docs/CLUSTER_SETUP.md). The replay entry point is `python -m magset.breakhis_predict --help`; its validation-only guard excludes fitting and outer-role inputs.

## Evidence and research record

- [Data audit](docs/DATA_AUDIT.md) and [BreakHis evaluation protocol](docs/BENCHMARK_PROTOCOL.md).
- [Initial baselines](docs/VALIDATION_PILOT.md), [neural controls](docs/FUSION_RESULTS.md), [sensitivity checks](docs/SENSITIVITY_RESULTS.md), [matched comparison](docs/CAPACITY_RESULTS.md).
- [Project explanation](docs/PROJECT_EXPLAINED.md), [execution plan](plan.md), [implemented methods draft](paper/methods-draft.md).
- [BreakHis review](docs/reviews/BREAKHIS_MILESTONE_REVIEW.md) and [prior critique ledger](docs/reviews/CLOSURE_LEDGER.md).
- Deferred research records: [related work](docs/RELATED_WORK.md), [TCGA protocol](docs/TCGA_QUARANTINE_PROTOCOL.md), [secondary pilot](docs/NATIVE_DEVELOPMENT_RESULTS.md), [source audit](docs/METHOD_COMPARISON_AUDIT.md).

Historical notebook accuracies use a different image-based split/selection procedure and are not comparable with our case-level results. Baseline completion, final generalization evidence and method novelty are separate claims.
