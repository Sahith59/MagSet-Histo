# Next tasks: finish the BreakHis capstone

November14,2026. The user needs the package in3–4days, by November17–18, for a presentation around November20. Exact day and rubric are unspecified. **BreakHis remains main; keep eight subtypes and four magnifications.** TCGA and novel-paper development are deferred.

## What is complete

- Original notebook outputs checked:93.73%/94.99% saved40× image-random, reused validation. Historical, not newly reproduced.
- Retained frozen Phikon reference: eight-class accuracy45.7%/38.3%, macro F1.3843/.3169, averaged across five trials and all15 view combinations.
- Adaptation, supervised ResNet and regional-view comparisons complete; none passed its improvement gate. Every outcome is retained.
- No-fit image endpoint bridge complete and independently checked on all ten runs.
- Retrospective benign/malignant secondary analysis complete: retained accuracy87.0%/80.7%; sensitivity/specificity and all three earlier pipelines reported. This does not replace eight-class evaluation.
- Fresh raw-image replay4971403 completed in39s:121 fields,15 combinations, maximum probability difference6.05e-08; no fitting or outer scoring.
- Three GPT-6.1 Sol review streams completed. Full pinned local suite:119 tests passed. Latest queue check: no active MagSet job; unrelated jobs untouched.

## Current work queue

| Order | Work | Completion condition |
| --- | --- | --- |
| 1 | Freeze model and publish evidence | Frozen reference retained; every result and failed gate preserved; labels clear |
| 2 | Finish report and presentation | Data, roles, architecture, training, metrics, comparisons, limits and references included |
| 3 | Prepare demonstration | [Verified saved-output viewer](evidence/CAPSTONE_PREDICTION_REPLAY.html), actual inference instructions and replay/live distinction |
| 4 | Review/rehearse by Nov17–18 | Check every number, practise a short explanation/questions, incorporate actual rubric requirements |
| 5 | Scope subsequent research | Reconcile clinical identities, obtain fresh confirmation and define one specific novel hypothesis |

Each trial uses46 fitting/eight validation/28 excluded provider cases. Arrangements812/813 cover22/20 distinct validation cases,31 in their union. Trials/views/images repeat; all82 cases entered development somewhere. Clinical identity remains unverified and no globally fresh confirmation cohort remains.

Low subtype accuracy remains a material limitation. ResNet's99.6–99.7% fitting balanced accuracy versus37–40% validation shows a large generalization gap; each fitting set contains one adenosis and one phyllodes case. No single cause is proven. More GPUs do not create independent cases. No automatic architecture/crop/rate/resplit sweep or outer rescue follows.

[Before/now](CAPSTONE_BEFORE_AND_NOW.md) · [Binary results](CAPSTONE_BINARY_RESULTS.md) · [Regional result](BREAKHIS_MULTICROP_RESULTS.md) · [Dates](CAPSTONE_DELIVERY_PLAN.md) · [Report](../paper/breakhis-project-report.md).

No message to Professor Dong, access request or conference submission has been sent.
