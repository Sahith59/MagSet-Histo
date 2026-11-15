# Next tasks: finish the BreakHis capstone

Updated November15,2026. The November14 timing request gives a working package deadline of November17–18 and presentation around November20. Exact day and rubric are unspecified. **BreakHis remains main; the corrected target is five final categories with four magnifications.** TCGA and novel-paper development are deferred.

## November15 correction: binary first, malignant subtype second

The user superseded the earlier eight-type hierarchy choice. Final labels are benign plus ductal/lobular/mucinous/papillary carcinoma. Step1 predicts benign/malignant; only predicted malignant cases reach Step2. [Corrected scope](DUAL_STEP_SCOPE.md). Retain the completed eight-class baseline evidence. First explain the task/score changes, then register one bounded direct-five control versus true dual-step comparison. No new training or five-class/hierarchical score exists yet;94–95% remains a goal. Preserve cases/roles/all magnifications and count routing errors in final five-class metrics. This supersedes earlier blanket freeze and eight-target language only for the explicitly requested task change.

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
| 1 | Fix the five-category experiment contract | Direct-five versus binary-plus-malignant-four; same roles/representation/subsets and declared selection budget |
| 2 | Implement/check both pipelines | Canonical label mapping, fitting-only normalization, routing, finite scores, save/reload and no case-role leakage |
| 3 | Run a bounded comparison | Report both arrangements/all trials, final five-class accuracy/F1/balanced accuracy, binary routing and malignant typing separately |
| 4 | Revise report/demo/presentation | New task/results clearly separated from completed eight-class and retrospective binary evidence; no invented94–95% result |
| 5 | Review/rehearse and scope later research | Work toward Nov17–18 materials/~Nov20 presentation; novel paper/fresh confirmation remain later work |

Each trial uses46 fitting/eight validation/28 excluded provider cases. Arrangements812/813 cover22/20 distinct validation cases,31 in their union. Trials/views/images repeat; all82 cases entered development somewhere. Clinical identity remains unverified and no globally fresh confirmation cohort remains.

Low subtype accuracy remains a material limitation. ResNet's99.6–99.7% fitting balanced accuracy versus37–40% validation shows a large generalization gap; each fitting set contains one adenosis and one phyllodes case. No single cause is proven. More GPUs do not create independent cases. No automatic architecture/crop/rate/resplit sweep or outer rescue follows.

[Before/now](CAPSTONE_BEFORE_AND_NOW.md) · [Binary results](CAPSTONE_BINARY_RESULTS.md) · [Regional result](BREAKHIS_MULTICROP_RESULTS.md) · [Dates](CAPSTONE_DELIVERY_PLAN.md) · [Report](../paper/breakhis-project-report.md).

No message to Professor Dong, access request or conference submission has been sent.
