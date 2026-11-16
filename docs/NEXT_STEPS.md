# Next tasks: finish the BreakHis capstone

Updated November15,2026. The November14 timing request gives a working package deadline of November17–18 and presentation around November20. Exact day and rubric are unspecified. **BreakHis remains main; the corrected target is five final categories with four magnifications.** TCGA and novel-paper development are deferred.

## November15 correction: binary first, malignant subtype second

The user superseded the earlier eight-type hierarchy choice. Final labels are benign plus ductal/lobular/mucinous/papillary carcinoma. Step1 predicts benign/malignant; only predicted malignant cases reach Step2. [Corrected scope](DUAL_STEP_SCOPE.md). Retain the completed eight-class baseline evidence. First explain the task/score changes, then register one bounded direct-five control versus true dual-step comparison. The first trained five-class comparison is now complete: dual65.8%/68.8% vs direct66.7%/64.7%, gate failed.94–95% remains a goal. [Verified results](BREAKHIS_FIVE_CLASS_RESULTS.md). Preserve cases/roles/all magnifications and count routing errors in final five-class metrics. This supersedes earlier blanket freeze and eight-target language only for the explicitly requested task change.

## What is complete

- Original notebook outputs checked:93.73%/94.99% saved40× image-random, reused validation. Historical, not newly reproduced.
- Retained frozen Phikon reference: eight-class accuracy45.7%/38.3%, macro F1.3843/.3169, averaged across five trials and all15 view combinations.
- Adaptation, supervised ResNet and regional-view comparisons complete; none passed its improvement gate. Every outcome is retained.
- No-fit image endpoint bridge complete and independently checked on all ten runs.
- Retrospective benign/malignant secondary analysis complete: retained accuracy87.0%/80.7%; sensitivity/specificity and all three earlier pipelines reported. This does not replace eight-class evaluation.
- Fresh raw-image replay4971403 completed in39s:121 fields,15 combinations, maximum probability difference6.05e-08; no fitting or outer scoring.
- Three GPT-6.1 Sol review streams completed. The earlier pinned suite had119 tests; the registered CNN source passed162 tests; the later complete collector/reviewer suite passed187 tests. Technical A40 checks4979613 passed independent review;20 full fits are submitted as array4979622. Unrelated jobs remain untouched.

[Current CNN execution record](BREAKHIS_CASEBAG_STATUS.md).

## Current priority after completed CNN runs

All20 CNN fits and both verification jobs completed successfully. CNNtwo-step accuracy46.50%/53.33% is below the frozen-Phikon65.83%/68.83%. Review saved rare-type errors and the fitting/validation gap, then revise the capstone materials around the best verified model and the complete comparison. [Accepted results](BREAKHIS_CASEBAG_RESULTS.md).

## Earlier execution queue

| Order | Work | Completion condition |
| --- | --- | --- |
| 1 | Preserve first five-class comparison | Complete:147tests,120headfits/fullcollector/independent numeric review; mixed negative result retained |
| 2 | Fix next case-supervised CNN recipe | Complete: source `b5f0df1` and recipe frozen before fitting;162 tests passed |
| 3 | Verify setup and technical execution | Preparation and zero-fit audit passed; both fitting-only A40 technical checks completed. Independent actual-output acceptance passed2,512 assertions |
| 4 | Run bounded CNN comparison | Array4979622 active:20 fixed50epoch fits, maxfour A40 GPUs; CPUcollector4979632 then independentreview4979664 automatically verify all results before publication |
| 5 | Update report/demo/presentation and rehearse | Only accepted measured results; Nov17–18 working materials/~Nov20 presentation; novelty/fresh confirmation later |


Each trial uses46 fitting/eight validation/28 excluded provider cases. Arrangements812/813 cover22/20 distinct validation cases,31 in their union. Trials/views/images repeat; all82 cases entered development somewhere. Clinical identity remains unverified and no globally fresh confirmation cohort remains.

Low subtype accuracy remains a material limitation. ResNet's99.6–99.7% fitting balanced accuracy versus37–40% validation shows a large generalization gap; each fitting set contains one adenosis and one phyllodes case. No single cause is proven. More GPUs do not create independent cases. No automatic architecture/crop/rate/resplit sweep or outer rescue follows.

[Before/now](CAPSTONE_BEFORE_AND_NOW.md) · [Binary results](CAPSTONE_BINARY_RESULTS.md) · [Regional result](BREAKHIS_MULTICROP_RESULTS.md) · [Dates](CAPSTONE_DELIVERY_PLAN.md) · [Report](../paper/breakhis-project-report.md).

No message to Professor Dong, access request or conference submission has been sent.
