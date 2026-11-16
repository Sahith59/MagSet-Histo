# BreakHis five-category dual-step: current capstone evidence

November15,2026. Final outputs: pooled benign plus ductal/lobular/mucinous/papillary carcinoma. Only BreakHis images/labels are fitted/evaluated. Step1is benign/malignant; predicted benign stops and predicted malignant proceeds to a four-type classifier.

| First trained model | Accuracy812 /813 | Macro F1812 /813 |
| --- | ---: | ---: |
| Direct five-class |66.7% /64.7%|.5161 /.5035|
| Dual-step |65.8% /68.8%|.4975 /.5325|

These average all15view settings and five overlapping trials per arrangement. The mixed result fails the prospectively declared both-arrangement improvement gate.94–95% is not achieved. Dual binary accuracy87.8%/88.3% and true-malignant subtype accuracy55.7%/56.3% locate a major remaining typing limitation; they are not whole-system accuracy.

Each run uses46fitting/eightvalidation/28excludedprovider cases, original eight-label roles before pooling. Router sees46cases; specialist35true-malignant cases. Validation has4Ben/1eachMaltype; rawaccuracy andbalancedaccuracydiffer. Allrelatedfields/zoomsremainwiththecase.31distinctvalidationcasesacrossarrangements,unresolvedclinicalidentity,andall82casesalreadyentereddevelopmentmeanthisisexploratoryvalidation,nofreshfinalclinicaltest.

Actualsource0471cd4ab8fb1da71eba8870f2ea0fa59ce0e8f3;147pinnedtestsandCIpassed. Audit4979472,10pairedtasks4979473andcollector4979474allcompleted0. Independent4,924,986checksaccepted80candidates/120heads/64,800dependentrecords,allnormalizersbitwiseandmaxmetricerror4.44e-16. [All results](../BREAKHIS_FIVE_CLASS_RESULTS.md) · [Current report](../../paper/breakhis-five-class-report.md) · [Independent review](../reviews/FIVE_CLASS_ACTUAL_RESULTS_REVIEW.md).

Next: one separately registered EfficientNetV2B0 case-supervised direct5/dual comparison, with fixed source/data/recipe, public pinned ImageNet weights, tests and two fitting-only A40 smokes before20productionfits. It has no production result yet. Capstone remains priority; novel-paper/TCGA/fresh confirmation later. No advisor message or submission sent.

Historical supplied notebooks93.73%/94.99% were eight-type40×image-random reused validation, not independently retrained. Earlier case8accuracy45.7%/38.3% and derivedbinary87.0%/80.7% stay distinct. Existing eight-class deck/demo are dated baseline artifacts and need task-specific revision; do not relabel their scores as five-category performance. [Before/now explanation](../CAPSTONE_BEFORE_AND_NOW.md).
