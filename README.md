# MagSet-Histo

**BreakHis is our main dataset. The immediate goal is the quick BreakHis project deliverable for Professor Dong.** TCGA follow-up is deferred. A novel method and conference paper are later research milestones.

We classify a provider case into one of eight breast-tumor subtypes using its available microscope views at 40×,100×,200× and 400×. We support all 15 nonempty view combinations without assuming that fields are physically paired across zooms.

## First checked BreakHis numbers

| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |
| --- | ---: | ---: | ---: |
|812|0.3843|45.7%|47.5%|
|813|0.3169|38.3%|37.5%|

These are exploratory provider-case validation results, averaged over five overlapping trials with46 fitting/8 validation/28 excluded outer-role cases each. F1 is not accuracy. Both splits together involve 31 distinct validation cases; 1,200 saved case/subset records do not represent 1,200 independent patients. All 82 provider cases entered development somewhere, and some clinical identifier relationships remain unresolved. No globally fresh BreakHis confirmation cohort remains.

The [complete first-results report](docs/BREAKHIS_FIRST_RESULTS.md) shows every trial, class weaknesses, provenance and execution checks. The baseline uses a frozen Phikon-v2 image reader, within-view mean pooling, saved fitting-only normalization, available-view mean plus mask, and a logistic classifier. The working repository name does not imply that this baseline is a novel transformer.

## Immediate deliverable

- Checked eight-class BreakHis baseline and complete development-results table.
- Raw-image validation replay verified: 121 original images, all 15 view settings; archived probabilities reproduced within 0.000000056.
- Reproducible source, saved model/prediction checks, clear limits and a short professor handoff.
- One bounded BreakHis improvement comparison after this baseline is packaged.

Start with the [focused milestone plan](docs/BREAKHIS_MILESTONE_PLAN.md), [next tasks](docs/NEXT_STEPS.md), and [plain training explanation](docs/TRAINING_STRATEGY.md).

The [next improvement](docs/BREAKHIS_ADAPTATION_STATUS.md) is now in execution: a fixed last-block Phikon-v2 adaptation recipe, checked fitting-only GPU smoke, and ten submitted production runs with four-way GPU concurrency. Improved validation scores are still pending. The baseline above remains the current measured result.

## Run checks

```bash
python3 -m pip install numpy==1.26.4 scipy==1.15.2 scikit-learn==1.6.1 Pillow==11.1.0
python3 -m unittest discover -s tests -v
```

The full neural checks additionally need torch==2.6.0. The real raw-image replay uses the existing pinned GPU environment, original dataset, pretrained encoder and saved baseline head; these artifacts are not bundled in Git. Submit heavy work through SLURM using [the cluster guide](docs/CLUSTER_SETUP.md). The replay entry point is `python -m magset.breakhis_predict --help`; its validation-only guard excludes fitting and outer-role inputs.

## Evidence and research record

- [Data audit](docs/DATA_AUDIT.md) and [BreakHis evaluation protocol](docs/BENCHMARK_PROTOCOL.md).
- [Initial baselines](docs/VALIDATION_PILOT.md), [neural controls](docs/FUSION_RESULTS.md), [sensitivity checks](docs/SENSITIVITY_RESULTS.md), [matched comparison](docs/CAPACITY_RESULTS.md).
- [Project explanation](docs/PROJECT_EXPLAINED.md), [execution plan](plan.md), [implemented methods draft](paper/methods-draft.md).
- [BreakHis review](docs/reviews/BREAKHIS_MILESTONE_REVIEW.md) and [prior critique ledger](docs/reviews/CLOSURE_LEDGER.md).
- Deferred research records: [related work](docs/RELATED_WORK.md), [TCGA protocol](docs/TCGA_QUARANTINE_PROTOCOL.md), [secondary pilot](docs/NATIVE_DEVELOPMENT_RESULTS.md), [source audit](docs/METHOD_COMPARISON_AUDIT.md).

Historical notebook accuracies use a different image-based split/selection procedure and are not comparable with our case-level results. Baseline completion, final generalization evidence and method novelty are separate claims.
