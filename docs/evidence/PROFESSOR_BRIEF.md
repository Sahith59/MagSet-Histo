# BreakHis master’s capstone: progress for Professor Dong

**Current priority: finish the BreakHis project deliverable first. TCGA follow-up and novel-paper work are deferred.**

The model reads the available microscope images of one provider case and predicts one of eight breast-tumor subtypes. It uses a frozen Phikon-v2 picture reader and a trained mean-pooling/logistic decision maker. This is an established baseline, not a novel mechanism.

| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |
| --- | ---: | ---: | ---: |
| 812 | 0.3843 | 45.7% | 47.5% |
| 813 | 0.3169 | 38.3% | 37.5% |

Ten saved baseline runs and 1,200 validation records passed trusted label/split/metric checks. The same cases recur across five provider holdouts and 15 view settings; there are 31 distinct validation cases across both splits. Each trial has 46 fit/8 validation/28 excluded cases. These are modest exploratory scores, not final test performance.

The raw-image replay passed on 121 original images, covering all 15 view settings of the first sorted validation case. Fresh39-second replay4971403: maximum archived-probability difference6.05e-08. The demonstration verifies execution/parity rather than improved accuracy. No new model fitting or outer-role scoring occurred.

**Fixed improvement tested:** ten six-epoch last-block Phikon-v2 adaptation models, plus a separate fitting-only smoke. All jobs and the complete checker passed in25minutes11seconds from first start to checked completion. Adapted mean accuracy45.7%/38.8%; primary F1.3840/.3205 versus frozen.3843/.3169. The two-split improvement gate failed; phyllodes recall remains 0. Keep the frozen baseline. No outer predictions or automatic replacement sweep.

**Diagnosis completed November13:** no new fitting or encoder execution. Saved adapted case heads achieve98.5–98.7% fitting balanced accuracy versus38.8–45.7% validation. The auxiliary image head achieves about98% fitting versus32–35% validation with class/case/view weighting. Neither fixed auxiliary case readout rescues the result. This supports a large fit–validation/generalization gap, without proving one sole cause. Independent auditing recomputed all saved diagnostic score arithmetic. [Diagnosis and figure](../BREAKHIS_DIAGNOSIS_RESULTS.md).

**Supervised reference completed November14:** all ten fixed ResNet-50 fits and the full checker passed after disclosed wall-time/replay recovery. Macro F1.3084/.3006 versus frozen.3843/.3169; case accuracy37.0%/40.0%. The registered gate failed; phyllodes recall remains0. Fitting balanced case accuracy99.6%/99.7% again exposes a large generalization gap. Keep the frozen reference. [Complete comparison](../BREAKHIS_SUPERVISED_RESULTS.md) · [Technical execution](../BREAKHIS_SUPERVISED_STATUS.md).

**Coverage/reference audit completed:** each fitting set has only one adenosis and one phyllodes provider case. All7,909 fields are accounted for. The inspected author's released folds and identities are incompatible with a direct joint four-view reproduction, so published scores stay separate. [Audit](../BREAKHIS_REFERENCE_AUDIT.md) · [Project-report draft](../../paper/breakhis-project-report.md).

**Open limits:** clinical relationships between some provider IDs remain unverified; all 82 cases entered development somewhere, so there is no globally fresh BreakHis confirmation cohort. Reliable final evaluation, method novelty and a submission-ready paper remain unfinished.

[Complete first results](../BREAKHIS_FIRST_RESULTS.md) · [Focused plan](../BREAKHIS_MILESTONE_PLAN.md) · [Training explanation](../TRAINING_STRATEGY.md).

**Original versus current:** the supplied notebooks saved93.73%/94.99% for single40× image-random, reused validation. Current case/subset numbers are a different endpoint, not a measured95%→40% degradation. [Independent original review](../reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md). The saved-prediction image bridge is complete: ResNet40× raw image accuracy27.13%/29.94%, equal-case weighted32.89%/31.82%, case averaging37.5%/35.0%; no new fits/forwards. This does not reproduce EfficientNet and does not make the protocols interchangeable.

**Regional comparison complete:** four extraction shards and20 matched fixed-C heads passed independent reviews. F1 changes−.0058/+.0059 and zero phyllodes recall fail the gate. Keep the frozen reference; model search is frozen. [Results](../BREAKHIS_MULTICROP_RESULTS.md).

**Separate retrospective binary result:** retained benign/malignant accuracy87.0%/80.7%, malignant sensitivity98.7%/91.3%, benign specificity75.3%/70.0%. This sums existing subtype probabilities at a fixed0.5 threshold; no new fitting or fresh test. Eight-class accuracy remains45.7%/38.3%. All three earlier pipelines, both arrangements and all trials are retained. AUROC is ranking, not accuracy; no direct superiority over the earlier paper follows. [Checked binary report](../CAPSTONE_BINARY_RESULTS.md).

**Delivery:** user needs the package by November17–18 for a presentation around November20. Exact day and formal task/rubric are unspecified. Finish report/presentation/demo and rehearse; TCGA/novelty remain deferred. [Plan](../CAPSTONE_DELIVERY_PLAN.md) · [Recorded prediction viewer](CAPSTONE_PREDICTION_REPLAY.html).
