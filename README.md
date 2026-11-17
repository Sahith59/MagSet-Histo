# MagSet-Histo

BreakHis is our main dataset. We are building a breast histology classification project for Professor Dong's master's capstone, with a research paper as a later goal. TCGA follow-up is deferred.

The system first decides **benign or malignant**. Benign is the final answer; malignant images proceed to a specialist that chooses ductal, lobular, mucinous or papillary carcinoma. We use the available 40×/100×/200×/400× images of each provider case and evaluate all 15 nonempty zoom combinations. [Task and architecture](docs/DUAL_STEP_SCOPE.md) · [Plain-language training guide](docs/TRAINING_STRATEGY.md).

The **matched three-way comparison is complete and independently checked**. On the same 16 test cases and all 15 zoom subsets, final-five accuracy / macro F1 is **67.50% / .375675 for Midnight P**, **56.67% / .348180 for CNN–Transformer**, and **59.17% / .304947 for Swin**. P’s separate benign/malignant accuracy is **94.58%**; that is not final-five accuracy. P leads the primary point estimates, while other endpoints differ and all paired intervals include zero. This is a small reused-development test, with no established clinical superiority or 95% final result. [Complete matched results](docs/MATCHED_THREE_WAY_RESULTS.md) · [Execution and preserved failures](docs/MATCHED_THREE_WAY_STATUS.md) · [Independent review](docs/reviews/MATCHED_THREE_WAY_RESULTS_REVIEW.md). Earlier scores below describe different experiments.

Both supplied notebook experiments are **complete and independently checked**. CNN–Transformer gives **97.09% binary / 86.41% eight-type image accuracy** with its declared input correction, versus **68.58% / 43.62%** with original scaling. Swin dual gives **93.38% binary / 64.37% final-five image accuracy**, macro F1 **.441602**; it trained 17 epochs and selected epoch 7. The main capstone was parked during these runs. Its case-based scorecard remains separate: the notebooks use different targets, prediction units and splits. [Side-by-side comparison](docs/SUPPLIED_NOTEBOOKS_COMPARISON.md) · [Swin complete results](docs/SUPPLIED_SWIN_STATUS.md) · [CNN complete results](docs/SUPPLIED_CNN_TRANSFORMER_STATUS.md).

**Earlier studies under the original trial roles:** learning both UNI2-h feature metrics gives **76.17% / 75.83% final accuracy**, with macro F1 **.668624 / .587323**. This is a small gain over its matched frozen control, below the registered larger-improvement screen. Standard supervised UNI2-h field heads score **73.83% / 73.67%**, a completed negative result. The retained Midnight reference P remains **76.33% / 75.67%**, with **85% / 85%** in the separate all-four-view condition. The malignant-only metric ablation leads observed 812 two-step F1 at **.672180**; P leads 813 at **.602512**. **94–95% final accuracy remains unachieved.** [Metric results](docs/BREAKHIS_SPECIALIST_METRIC_RESULTS.md) · [Linear results](docs/BREAKHIS_UNI_LINEAR_RESULTS.md) · [Plain-language explanation](docs/PROJECT_EXPLAINED.md).

The complementary GenBio reader and whole-case mean/attention round is complete and verified. Neither consistently improves the requested two-step endpoint. All 40 GPU training trajectories and ten paired GenBio classifier jobs completed; all 370 neural tests passed. [Completed execution and decision](docs/BREAKHIS_NEXT_ROUND_STATUS.md).

## Earlier five-category trial results

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
| UNI2-h similarity |75.83% /75.17%|**.6642** /.5752|
| Virchow2 similarity |72.50% /73.00%|.5758 /.5387|
| H-optimus-0 similarity |73.50% /71.17%|.5097 /.5062|
| UNI2-h binary metric only |75.67% /75.33%|.6606 /.5784|
| UNI2-h malignant metric only |76.33% /75.50%|.6722 /.5814|
| UNI2-h both metrics, primary |76.17% /75.83%|.6686 /.5873|
| UNI2-h supervised field heads |73.83% /73.67%|.5753 /.5354|

GenBio's one-stage control scores 71.33%/77.83%, F1 .5960/.6406. Its stronger 813 result is a different readout, with weaker 812 accuracy; it does not establish a two-step gain. [GenBio results](docs/BREAKHIS_GENBIO_RESULTS.md) · [Whole-case results](docs/BREAKHIS_CASE_MIL_RESULTS.md).

All comparisons above are complete and verified. The three newly approved readers add frozen-feature comparisons under the same registered roles and decision structure. UNI2-h improves first-arrangement F1, with lower accuracy than the retained reference in both arrangements. Decision-specific centers produce modest F1 gains, with accuracy and recall tradeoffs; both centering modes fail the registered larger-improvement screen. Their direct-five controls are exact duplicates, leaving four structurally unique direct settings and eight dual settings across the mode family. No novel mechanism or 95% result is established. [Native negative result](docs/BREAKHIS_NATIVE_RESULTS.md) · [Centering evidence](docs/BREAKHIS_PRIOR_CENTERED_RESULTS.md) · [Full scorecard](paper/breakhis-five-class-report.md).

The retained P model's main weakness is malignant typing. It scores 94.83%/93.00% on benign/malignant routing, but 58.33%/62.33% on true-malignant subtyping. Binary accuracy is not final five-category accuracy. Only BreakHis images/labels are fitted/evaluated; public pathology pretraining is external transfer learning.

The supplied notebooks saved 93.73%/94.99% on eight-type 40× individual images with image-random, reused validation. Their task, prediction unit and split differ from the current case experiment. [Before/now explanation](docs/CAPSTONE_BEFORE_AND_NOW.md). Original notebooks and earlier results remain historical evidence.

## Earlier eight-class BreakHis baseline numbers

| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |
| --- | ---: | ---: | ---: |
|812|0.3843|45.7%|47.5%|
|813|0.3169|38.3%|37.5%|

These are exploratory provider-case validation results, averaged over five overlapping trials with46 fitting/8 validation/28 excluded outer-role cases each. F1 is not accuracy. Both splits together involve 31 distinct validation cases; 1,200 saved case/subset records do not represent 1,200 independent patients. All 82 provider cases entered development somewhere, and some clinical identifier relationships remain unresolved. No globally fresh BreakHis confirmation cohort remains.

The [complete first-results report](docs/BREAKHIS_FIRST_RESULTS.md) shows every trial, class weaknesses, provenance and execution checks. The baseline uses a frozen Phikon-v2 image reader, within-view mean pooling, saved fitting-only normalization, available-view mean plus mask, and a logistic classifier. The working repository name does not imply that this baseline is a novel transformer.

The completed [case-error audit](docs/BREAKHIS_SPECIALIST_ERROR_AUDIT.md) traces the five selected pipelines to 31 reused provider keys. The metric primary corrects eight repeated events and regresses on two, for a net six; persistent papillary errors remain. Its private original-field index is ready for appropriate review and makes no morphological conclusion.

## Immediate deliverable

Finish the five-category capstone report, presentation and demonstration using the complete accepted scorecard and representative errors. The working materials target is November 17–18, with presentation around November 20; the exact date remains unconfirmed. Preserve every negative comparison and state what the model can and cannot answer. [Next tasks](docs/NEXT_STEPS.md) · [Project report](paper/breakhis-five-class-report.md).

The original six supplied references remain unchanged and excluded from public source. Raw data, weights, provider identifiers, private cluster operations and conversation memory are also excluded. Source/configuration revisions, audits and complete aggregate evidence are recorded for reproducibility.

## Run checks

```bash
python3 -m pip install numpy==1.26.4 scipy==1.15.2 scikit-learn==1.6.1 Pillow==11.1.0
python3 -m unittest discover -s tests -v
```

The full neural checks additionally need torch==2.6.0 and torchvision==0.21.0. The latest linear source passed all 415 full pinned neural tests, including the metric and readout checks. The real raw-image replay uses the existing pinned GPU environment, original dataset, pretrained encoder and saved baseline head; these artifacts are not bundled in Git. Submit heavy work through SLURM using [the cluster guide](docs/CLUSTER_SETUP.md). The replay entry point is `python -m magset.breakhis_predict --help`; its validation-only guard excludes fitting and outer-role inputs.

## Evidence and research record

- [Data audit](docs/DATA_AUDIT.md) and [BreakHis evaluation protocol](docs/BENCHMARK_PROTOCOL.md).
- [Initial baselines](docs/VALIDATION_PILOT.md), [neural controls](docs/FUSION_RESULTS.md), [sensitivity checks](docs/SENSITIVITY_RESULTS.md), [matched comparison](docs/CAPACITY_RESULTS.md).
- [Project explanation](docs/PROJECT_EXPLAINED.md), [execution plan](plan.md), [implemented methods draft](paper/methods-draft.md).
- [BreakHis review](docs/reviews/BREAKHIS_MILESTONE_REVIEW.md) and [prior critique ledger](docs/reviews/CLOSURE_LEDGER.md).
- Deferred research records: [related work](docs/RELATED_WORK.md), [TCGA protocol](docs/TCGA_QUARANTINE_PROTOCOL.md), [secondary pilot](docs/NATIVE_DEVELOPMENT_RESULTS.md), [source audit](docs/METHOD_COMPARISON_AUDIT.md).

Historical notebook accuracies use a different image-based split/selection procedure and are not comparable with our case-level results. Baseline completion, final generalization evidence and method novelty are separate claims.
