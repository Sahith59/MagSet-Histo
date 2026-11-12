# BreakHis project: first baseline for Professor Dong

**Current priority: finish the BreakHis project deliverable first. TCGA follow-up and novel-paper work are deferred.**

The model reads the available microscope images of one provider case and predicts one of eight breast-tumor subtypes. It uses a frozen Phikon-v2 picture reader and a trained mean-pooling/logistic decision maker. This is an established baseline, not a novel mechanism.

| Validation split | Mean-subset macro F1 | Mean-subset case accuracy | All-view case accuracy |
| --- | ---: | ---: | ---: |
| 812 | 0.3843 | 45.7% | 47.5% |
| 813 | 0.3169 | 38.3% | 37.5% |

Ten saved baseline runs and 1,200 validation records passed trusted label/split/metric checks. The same cases recur across five provider holdouts and 15 view settings; there are 31 distinct validation cases across both splits. Each trial has 46 fit/8 validation/28 excluded cases. These are modest exploratory scores, not final test performance.

The raw-image replay passed on 121 original images, covering all 15 view settings of the first sorted validation case. Maximum archived-probability difference: 5.52e-08. The demonstration verifies execution/parity rather than improved accuracy. No new model fitting or outer-role scoring occurred.

**Next:** package the baseline, then register one bounded standard BreakHis improvement comparison. Working estimate: the next one or two work sessions for the new experiment and initial results, subject to implementation/queue. No positive gain or completion date is guaranteed.

**Open limits:** clinical relationships between some provider IDs remain unverified; all 82 cases entered development somewhere, so there is no globally fresh BreakHis confirmation cohort. Reliable final evaluation, method novelty and a submission-ready paper remain unfinished.

[Complete first results](../BREAKHIS_FIRST_RESULTS.md) · [Focused plan](../BREAKHIS_MILESTONE_PLAN.md) · [Training explanation](../TRAINING_STRATEGY.md).
