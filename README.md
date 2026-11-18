# MagSet-Histo

For Professor Dong’s master’s capstone, we asked whether one model could look at microscope photos from a breast tissue case and answer two questions. Is the tissue benign or malignant? If it is malignant, which of four cancer types does it show? We trained and evaluated our classifier on BreakHis.

**[Download the three-page Word report](https://raw.githubusercontent.com/Sahith59/MagSet-Histo/main/outputs/capstone/BreakHis%20UNI2-h%20Capstone%20Report.docx)** · [Compare the models on the same cases](docs/BREAKHIS_MODEL_COMPARISON.md) · [Read the one-page brief](output/pdf/BreakHis-five-category-brief.pdf)

## How the selected model works

We kept the original UNI2-h model that combines the photos from a case before making its decision. UNI2-h had already learned to recognize patterns in microscope images. We kept that part fixed and trained one small classifier. It can use any combination of 40×, 100×, 200× and 400× photos. Its five possible answers are benign, ductal carcinoma, lobular carcinoma, mucinous carcinoma and papillary carcinoma.

![Architecture diagram. A frozen UNI2-h reader describes each photo. We average the descriptions for each zoom, combine the available zooms and pass them to one classifier. A benign decision ends the process. A malignant decision selects one of the four cancer types.](outputs/capstone/uni2h-architecture.png)

## What we measured

BreakHis supplied 7,909 photos in 82 case folders. We used 49 folders for training, 17 for choosing the saved model, and 16 for an internal benchmark. All images from the same folder stayed together. Each case was scored under 15 combinations of zooms, so every row below contains repeated answers from the same cases.

| Selected UNI2-h model | Exact five-category diagnosis | Benign or malignant |
| --- | ---: | ---: |
| Validation on 17 cases | 193/255 (75.69%) | 229/255 (89.80%) |
| Internal benchmark on 16 cases | **199/240 (82.92%)** | **231/240 (96.25%)** |

UNI2-h tied a two-reader system for the highest five-category result on the internal benchmark. The adapted Swin and CNN–Transformer models scored 59.17% and 56.67% on the same task. Earlier experiments had already used the 16 benchmark cases, and we revisited the architecture choice after seeing the comparison. Accuracy on new cases is still unknown. [Matched results and original notebook scores](docs/BREAKHIS_MODEL_COMPARISON.md)

Naming the cancer type was harder than recognizing malignancy. Of the 41 wrong five-category answers, 32 correctly identified the tissue as malignant and then named the wrong type. A wider crop added three correct ductal answers on validation. Lobular and papillary results stayed the same. The next step is a blinded review of the tissue fields and an evaluation with more independent rare cases. [Wider-image experiment](docs/BREAKHIS_WIDER_FIELD.md)

## Other UNI2-h experiments

The follow-up tested whether recognizing patterns in each original field before averaging would help. On the same validation cases it also scored 75.69% five-category accuracy, with 90.59% binary accuracy and lower macro F1 (0.4738). In a separate check holding out each of the 42 training groups in turn, the original / field-first five-category accuracies were 65.85% / 68.84% and the binary accuracies were 91.56% / 93.88%. The original shared classifier provides the historical reference. The joint targets are **above 90% five-category accuracy and above 95% benign/malignant accuracy from one checkpoint**. [Comparison, rare-type results and next step](docs/FIELD_HIERARCHY.md).

## Earlier comparison studies

The **matched three-way comparison is complete and independently checked**. On the same 16 test cases and all 15 zoom subsets, final-five accuracy / macro F1 is **67.50% / .375675 for Midnight P**, **56.67% / .348180 for CNN–Transformer**, and **59.17% / .304947 for Swin**. P’s separate benign/malignant accuracy is **94.58%**; that is not final-five accuracy. P leads the primary point estimates, while other endpoints differ and all paired intervals include zero. This is a small reused-development test, with no established clinical superiority or 95% final result. [Complete matched results](docs/MATCHED_THREE_WAY_RESULTS.md) · [Execution and preserved failures](docs/MATCHED_THREE_WAY_STATUS.md) · [Independent review](docs/reviews/MATCHED_THREE_WAY_RESULTS_REVIEW.md). Earlier scores below describe different experiments.

The latest deep review also **reproduced selected-checkpoint inference and reported metrics from raw images**. A targeted UNI2-h malignant specialist scores 82.92% on the already-viewed follow-up, but fails its predefined validation progression screen; it is a mixed development result, not a confirmed upgrade. Binary routing remains 94.58%. The completed case-SVC retrospective follow-up gives 74.58% linear / 69.17% RBF final accuracy and 92.08% binary for both; its registered validation screen also fails. Neither requested >90% final-five nor >97% binary follow-up objective is met. [Proof and diagnosis](docs/DEEP_DIAGNOSIS_AND_PROOF.md) · [complete SVC result](docs/BREAKHIS_UNI_CASE_SVM_RESULTS.md).

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

The three-page report is ready for the capstone presentation. The detailed project record includes earlier experiments and representative errors. [Next tasks](docs/NEXT_STEPS.md) · [Detailed project report](paper/breakhis-five-class-report.md)

The original six supplied references remain unchanged and excluded from public source. Raw data, weights, provider identifiers, private cluster operations and conversation memory are also excluded. Source/configuration revisions, audits and complete aggregate evidence are recorded for reproducibility.

## Run checks

Use a Python 3.11 virtual environment to match the versions used by GitHub Actions.

```bash
python3 -m pip install numpy==1.26.4 scipy==1.15.2 scikit-learn==1.6.1 Pillow==11.1.0 pandas==2.2.3 nbformat==5.10.4
python3 -m unittest discover -s tests -v
```

The full neural checks also use torch==2.6.0, torchvision==0.21.0 and safetensors==0.5.3. Replaying an experiment from raw images requires the original dataset, the appropriate pretrained encoder and the saved classifier weights. The historical baseline replay command is `python -m magset.breakhis_predict --help`. [Cluster setup](docs/CLUSTER_SETUP.md) · [Selected UNI2-h training and verification](docs/SINGLE_MODEL_DUAL_STEP.md)

## Evidence and research record

- [Data audit](docs/DATA_AUDIT.md) and [BreakHis evaluation protocol](docs/BENCHMARK_PROTOCOL.md).
- [Initial baselines](docs/VALIDATION_PILOT.md), [neural controls](docs/FUSION_RESULTS.md), [sensitivity checks](docs/SENSITIVITY_RESULTS.md), [matched comparison](docs/CAPACITY_RESULTS.md).
- [Project explanation](docs/PROJECT_EXPLAINED.md), [execution plan](plan.md), [implemented methods draft](paper/methods-draft.md).
- [BreakHis review](docs/reviews/BREAKHIS_MILESTONE_REVIEW.md) and [prior critique ledger](docs/reviews/CLOSURE_LEDGER.md).
- Deferred research records: [related work](docs/RELATED_WORK.md), [TCGA protocol](docs/TCGA_QUARANTINE_PROTOCOL.md), [secondary pilot](docs/NATIVE_DEVELOPMENT_RESULTS.md), [source audit](docs/METHOD_COMPARISON_AUDIT.md).

Historical notebook accuracies use a different image-based split/selection procedure and are not comparable with our case-level results. Baseline completion, final generalization evidence and method novelty are separate claims.
