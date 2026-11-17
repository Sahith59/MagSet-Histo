# Where the remaining two-step errors occur

November 17, 2026. **Persistent provider cases and malignant-type mistakes explain where performance is lost, but this audit does not establish why.** The both-head metric corrects eight previously wrong events and introduces two regressions, a net gain of six across 1,200 dependent events. Papillary carcinoma remains difficult: **106/150 errors with both metrics, versus 123/150 with the linear classifier**, with very different routing/type balances.

This is a read-only, post-hoc join of already saved validation predictions. The [exact public audit ledger](evidence/BREAKHIS_SPECIALIST_ERROR_AUDIT.json) binds the [metric results](BREAKHIS_SPECIALIST_METRIC_RESULTS.md) and [linear results](BREAKHIS_UNI_LINEAR_RESULTS.md). It adds **zero fits, encoder forwards, predictions, excluded-role predictions or selected settings**. Both original improvement studies remain failed-screen outcomes; **94–95% final accuracy and novelty remain unachieved**.

## What was joined

Five pipelines are joined on the same original role, provider key and magnification subset: UNI2-h identity KDE, binary-only metric, malignant-only metric, both metrics, and the selected UNI2-h linear two-step classifier. The audit verifies **50 selected validation prediction archives across ten original roles**, each with eight cases and fifteen subsets: **1,200 records per pipeline, 6,000 dependent records overall**. It does not include every training candidate or the linear direct-five control.

The headline endpoint is unchanged: **600 dependent events per arrangement per pipeline**, with five trials and fifteen subsets. Combining arrangements for an error inventory does not create a new independent confirmation endpoint. The separate all-four-view inventory has 80 events per pipeline, forty per arrangement, and remains a different condition.

The repeated validation records cover **31 distinct provider keys: 14 benign and 17 malignant**. The malignant keys comprise **six ductal, three lobular, four mucinous and four papillary**. These are distinct observed validation keys, not a newly assembled cohort or independently verified clinical patients. Each original fitting role still has only two lobular and three papillary cases. All 82 qualified provider keys have entered development somewhere.

## Stage errors across both arrangements

Every incorrect broad route is a final error. A wrong malignant type contributes a separate final error only after a correct route. The following counts partition final errors without double counting:

| Pipeline | Final errors / 1,200 | Routing errors | Wrong subtype after correct route |
| --- | ---: | ---: | ---: |
| Identity | 294 | 158 | 136 |
| Binary metric only | 294 | 156 | 138 |
| Malignant metric only | 289 | 158 | 131 |
| Both metrics | 288 | 156 | 132 |
| Linear two-step | 315 | 72 | 243 |

Both metrics reduce identity's routing errors by two and post-route type errors by four. Linear has much better broad separation but substantially more wrong malignant types. A conditional subtype diagnostic evaluated on every true-malignant event can also be wrong on an event routed to benign; that diagnostic must not be added to routing errors as though it were a disjoint stage count.

Within each arrangement, both-head final errors are **143 = 100 routing + 43 type** on 812 and **145 = 56 + 89** on 813. Linear has **157 = 49 + 108** and **158 = 23 + 135**. These reconcile exactly with the original result pages; the audit changes no headline metrics.

## Corrections and regressions are both retained

| Pipeline versus identity | Corrections | Regressions | Net correct change | Changed final predictions |
| --- | ---: | ---: | ---: | ---: |
| Binary only | 1 | 1 | 0 | 4 |
| Malignant only | 6 | 1 | +5 | 11 |
| Both | 8 | 2 | +6 | 16 |
| Linear two-step | 77 | 98 | -21 | 284 |

For both metrics, arrangement 812 contributes three corrections and one regression, net +2; 813 contributes five and one, net +4. Six of the sixteen changed final predictions merely exchange one wrong label for another. Four routes change. In the separate all-four-view condition, both corrects two events without regressions, net +2 out of 80. These are repeated case/subset events, not eight newly recovered patients or two independently harmed patients.

The unchanged raw total for binary-only conceals a correction and regression. Conversely, linear's 77 corrections do not justify selecting it when 98 previously correct events regress. Reporting only rescued examples would misrepresent the paired comparison.

## Papillary: two different failure patterns

Each malignant class contributes 150 pooled validation events across arrangements, although the number and exposure of underlying keys differ.

| Papillary pipeline | Final errors / 150 | Routed to benign | Wrong type after correct route |
| --- | ---: | ---: | ---: |
| Identity | 109 | 77 | 32 |
| Both metrics | 106 | 75 | 31 |
| Linear two-step | 123 | 3 | 120 |

Both metrics leave much of papillary's burden in broad routing. Linear routes nearly every papillary event to malignant but usually assigns another carcinoma type. Therefore a single explanation such as “only fix the router” does not apply across these pipelines. The raw papillary error burden remains highest among malignant classes for both and linear, but this does not identify mislabeled images, stain artifacts or missing morphology.

For both metrics, class error counts are **60 benign, 60 ductal, five lobular, 57 mucinous and 106 papillary**, totaling 288. For linear they are **69, 11, 68, 44 and 123**, totaling 315. Linear's ductal recovery occurs alongside larger lobular/papillary losses, consistent with the original class-confusion reports.

## Unequal exposure and descriptive equal-key rates

A key appears in between **15 and 120 observed validation events** across overlapping roles/subsets. Ranking keys by raw numbers of errors therefore ranks **observed error burden**, not intrinsic failure probability or clinical severity. For both metrics, the top three raw-error keys contribute 105/288 errors (36.46%); the top five contribute 158/288 (54.86%). Those concentrations must be read alongside their different exposure counts.

For a separate descriptive view, first divide each key's observed errors by its own exposure, then average those rates equally over keys in the stated scope:

    equal-key descriptive error rate = mean_over_keys(errors_for_key / exposures_for_key).

This is **not majority voting, a new patient-level accuracy, a fresh test, an independent-case estimate or a replacement headline endpoint**. It averages performance under the roles/subsets in which each key happened to be observed; case mixes, fitted supports and temperatures can differ between those exposures.

| Pipeline | Event-weighted error rate, 1,200 events | Equal-key descriptive mean error rate, 31 keys |
| --- | ---: | ---: |
| Identity | 24.50% | 30.29% |
| Binary only | 24.50% | 30.29% |
| Malignant only | 24.08% | 29.82% |
| Both | 24.00% | 29.75% |
| Linear two-step | 26.25% | 30.91% |

Within each malignant class, the same distinction matters:

| Class | Distinct validation keys | Both event error rate | Both equal-key descriptive error rate | Linear event error rate | Linear equal-key descriptive error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ductal | 6 | 40.00% | 37.04% | 7.33% | 6.11% |
| Lobular | 3 | 3.33% | 2.78% | 45.33% | 54.44% |
| Mucinous | 4 | 38.00% | 51.53% | 29.33% | 49.17% |
| Papillary | 4 | 70.67% | 73.89% | 82.00% | 85.00% |

For example, both's mucinous error is 38% when weighting repeated events, versus 51.53% when giving its four observed keys equal weight. This reveals dependence on unequal validation exposure, not a new measured causal explanation. Equal weighting does not make the reused keys independent or resolve clinical identity.

## Persistent keys and the image-inspection limit

The private index was used here only to derive anonymized class-count facts. **Three keys are wrong in every observed event under identity, and three under both**: one benign, one ductal and one papillary in each pipeline. Linear has **six always-wrong keys: two benign, two papillary, one mucinous and one lobular**. “Always wrong” means all their observed selected-validation events, not an assertion about every possible view, future role or new clinical sample. The counts locate persistent cases without exposing identifiers.

Identity and both also have fourteen keys with no observed errors; linear has fifteen, despite its worse overall total. Such counts alone cannot rank overall quality when exposures and where errors occur differ.

Original field paths and identifiers remain private. The audit records fixed first/middle/last lexicographic fields per view for possible later inspection, but **decoded, copied and inspected no images**. The presence of a path in that index is not evidence about label quality, staining, tumor content or morphology. Even later visual inspection of those fixed examples would not by itself causally explain a pooled prediction that depends on all available fields, view weights and support cases. No representative-image explanation or label correction is claimed here.

## Exact provenance and execution

Read-only audit job **4983980** completed in approximately **five seconds** under source `1bedac7835ddb5fc506f03cdfae9a51ea8970632`. The public ledger preserves all fifty selected archive receipts and the underlying numerical producer/configuration bindings. The private audit and joined prediction bundle remain private; their hashes bind the evidence without publishing field paths or case identifiers.

| Artifact | Revision / SHA256 |
| --- | --- |
| Audit source | `1bedac7835ddb5fc506f03cdfae9a51ea8970632` |
| Audit source archive | `5e59fc6b9c4bb4e3bb55eb1e288408bd6435e8d6b4cc547b07b34299883f9489` |
| Audit script | `cb0a71a824c6ec89732555a11b92acc59e718824e1545a158b12868a663e7814` |
| [Public audit ledger](evidence/BREAKHIS_SPECIALIST_ERROR_AUDIT.json) | `fe8763582d918dcf5b1e916707f8a5df0b6ae0ddbfce0e28efdeb0a3b5697c11` |
| Private audit | `a2c6314171374ae30346ffa0ac4ca8b58cf662c11866fe05f44e16144e550df8` |
| Private selected-prediction bundle | `c9aef24985441aa997079fbc20658c669d054aaae32611f8d1b427a899176974` |
| [Metric result ledger](evidence/BREAKHIS_SPECIALIST_METRIC_RESULTS.json) | `a298435b5f373003b72c302f8ee9f093a92510d2345340b6439f16b706e4a9cb` |
| [Linear result ledger](evidence/BREAKHIS_UNI_LINEAR_RESULTS.json) | `465f89c6ea8431ee0c9e37a835f85913602673a32a6288ae209e60f5becab9ed` |
| Metric numerical producer | `da7f169013590366f4284de2de690c1c6e980e23` |
| Metric configuration | `a188b13940c48dba2b48bddba27421747a7b14416273a4d17b9d1db28bf8f1fd` |
| Linear numerical producer | `3369de0280ffbdfa6080cc10699d748a2a75cb3e` |
| Linear configuration | `39ac36c3ade6693c0a3fb49937e64b1c105a42a348be3124dc4d996089ca77b0` |
| Dataset manifest | `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31` |
| Original role protocol | `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134` |

This audit establishes matched-event changes, error stages and exposure-aware descriptive patterns. It does not establish a stain, morphology, identity or label-defect cause, add fresh validation, or justify an automatic model/threshold/split change.
