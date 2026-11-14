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
- Complete the registered supervised ResNet-50 comparison under the same case roles, then finish the BreakHis project report.

Start with the [focused milestone plan](docs/BREAKHIS_MILESTONE_PLAN.md), [next tasks](docs/NEXT_STEPS.md), and [plain training explanation](docs/TRAINING_STRATEGY.md).

The [fixed adaptation comparison](docs/BREAKHIS_ADAPTATION_RESULTS.md) is complete: all ten runs and the artifact checker passed, but accuracy is45.7%/38.8% and the two-split F1 improvement gate failed. Keep the frozen baseline above as the default. [Published benchmark context](docs/BREAKHIS_PUBLIC_BENCHMARKS.md) explains related eight-class results and protocol differences.

The [saved-model diagnosis](docs/BREAKHIS_DIAGNOSIS_RESULTS.md) is also complete. The adapted case classifier scores98.5–98.7% fitting balanced accuracy and38.8–45.7% on validation. The auxiliary image classifier scores about98% fitting versus32–35% validation under class/case/view weighting. Neither fixed auxiliary readout rescues the result. This supports a large generalization gap; it does not identify one sole cause. No new fitting or encoder execution was used.

The [coverage/reference audit](docs/BREAKHIS_REFERENCE_AUDIT.md) and [supervised ResNet-50 comparison](docs/BREAKHIS_SUPERVISED_RESULTS.md) are complete. All ten models and the checker passed after disclosed technical recovery. Mean-subset F1 is0.3084/0.3006 and case accuracy37.0%/40.0%; F1 is lower than the frozen reference on both arrangements. The registered gate failed; keep the frozen reference. [Execution record](docs/BREAKHIS_SUPERVISED_STATUS.md) · [Fixed recipe](docs/BREAKHIS_SUPERVISED_SPEC.md) · [BreakHis report draft](paper/breakhis-project-report.md).

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
