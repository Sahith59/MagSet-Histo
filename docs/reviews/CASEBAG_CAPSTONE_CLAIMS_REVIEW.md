# Independent case-bag capstone claims review

November 15, 2026. This review checks the current five-class report, professor brief, accepted first-five results/review, registered case-bag recipe/configuration and historical delivery materials. The case-bag source supplied by the coordinator is `b5f0df18a47c1607be492bdaab7ab6cd2171226b`. This reviewer performed document/source inspection only: no fitting, image decoding, encoder forward, model download, scheduler operation or independent repeat of the numerical acceptance audit. Only this review file was written.

**Recommendation:** finish the one registered CNN comparison if its two technical smokes and timing review pass, while preparing the capstone around the already accepted five-class result. The professor can trust a reproducible, clearly qualified comparison now. The evidence does not support a 95% promise, clinical effectiveness, a novel architecture or a submission-ready research paper. No new scientific blocker was found in the fixed recipe; actual smoke, production and complete-output acceptance remain separate gates.

## What is established

The [current report](../../paper/breakhis-five-class-report.md) and [professor brief](../evidence/PROFESSOR_BRIEF.md) correctly describe the requested task: pooled benign plus ductal, lobular, mucinous and papillary carcinoma. The dual system has a binary router and a malignant-only four-class specialist; predicted benign stops. It has no benign subtype expert.

The [first trained comparison](../BREAKHIS_FIVE_CLASS_RESULTS.md) is complete and independently accepted under comparison SHA256 `bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795`:

| Arrangement | Pipeline | Mean-15 five-class accuracy | Macro F1 | Balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| 812 | Direct five | 66.67% | 0.5161 | 58.27% |
| 812 | Dual step | 65.83% | 0.4975 | 56.53% |
| 813 | Direct five | 64.67% | 0.5035 | 58.07% |
| 813 | Dual step | 68.83% | 0.5325 | 61.33% |

These average metrics across all 15 view subsets and five overlapping trials per arrangement. The dual model worsens one arrangement and improves the other; the registered both-arrangement improvement gate fails. The correct conclusion is **mixed evidence for this hierarchy on this frozen representation**, not that dual-step classification reliably improves accuracy. Preserve the requested dual prototype and its direct control together.

The [independent actual-output review](FIVE_CLASS_ACTUAL_RESULTS_REVIEW.md) supports confidence in roles, fitting-only normalization, saved models, coefficient/probability replay, metrics, candidate selection and gate arithmetic. Its 4,924,986 checks are software/provenance evidence, not that many disease observations. The accepted 64,800 fitting/validation case-subset records repeat a small cohort; they are not independent patients. Multiple AI reviews likewise do not constitute external clinical validation.

The actual dual router reaches 87.83%/88.33% accuracy, while true-malignant subtype accuracy is 55.67%/56.33%. The latter supplies the correct malignant group and omits routing failures. Neither is final five-class accuracy. Low papillary final recall, 9.33%/17.33%, is especially important to show alongside the aggregate result. These observations locate a typing weakness and a fitting–validation gap; they do not establish a single biological, annotation, stain or encoder cause. [Accepted error and recall tables](FIVE_CLASS_ACTUAL_RESULTS_REVIEW.md).

## What the CNN experiment can establish

The [registered case-bag recipe](../BREAKHIS_CASEBAG_SPEC.md) is a defensible bounded follow-up: one ImageNet EfficientNetV2B0 reader, sampled case bags, mean pooling, all 15 masks, staged selective adaptation and matched direct-five/dual arms under the unchanged roles. It tests complete supervised pipelines. Losses, encoder gradients, preprocessing and representation differ from the frozen-Phikon comparison; a gain would not isolate hierarchy, pooling or a single training change as its cause.

The [preparation receipt](../evidence/CASEBAG_PREPARATION.json) records verified public weight/wheel bytes, unchanged core dependencies, strict CPU state loading and the expected topology. It explicitly records zero fits, zero encoder forwards and zero predictions. The coordinator reports the zero-fit input audit4979604 completed successfully. These establish preparation, not predictive performance. At this review checkpoint, no accepted case-bag production result is available.

Two fixed trial1/812 A40 fitting-only smokes must establish actual permitted gradients, selective parameter changes, fixed BN buffers, optimizer/phase/receipt integrity, reload parity and feasible timing/memory. A passing smoke says the implementation runs under its contract. It cannot predict 95% validation accuracy. `production_ready=true` means the recipe is frozen; it does not replace both accepted smoke receipts. All 20 production runs, all four checkpoints and independent complete-matrix acceptance are needed before a result enters the capstone comparison.

The runtime decision must count the full 531,752 production field forwards plus 160 reload probes, image I/O, serialization and actual memory. The 368,000 training forwards alone undercount the job. If the fixed work cannot complete with enough collection/review/packaging time, report the accepted first-five comparison and the CNN's exact pending status. Do not shorten the declared scientific matrix or choose a favorable partial result to meet the deadline.

## Claims that need careful wording

- **Validation, not a fresh test.** Each run uses 46 fitting, eight validation and 28 excluded provider cases. Excluded is a per-run role. All 82 provider keys entered development somewhere, clinical identity remains unresolved and only 31 distinct cases appear in the validation union. Selection and reported scores reuse validation. A new split, more masks or more reviewers would not restore an untouched cohort.
- **Five-class accuracy, not binary accuracy.** Benign must route correctly; malignant must route correctly and receive the correct subtype. Hard-route accuracy cannot exceed router accuracy on identical records. The composed five-probability argmax can differ from the actual hard decision. The current reports correctly keep these outputs distinct.
- **Class-balanced visibility.** Validation support is benign4/DC1/LC1/MC1/PC1. Always-benign gives 50% raw accuracy and 20% balanced accuracy. Macro F1, balanced accuracy and all five recalls must accompany accuracy; this validation support is not a measured clinical disease prevalence.
- **Missing-view benchmark, not an upload product.** The raw case-bag scorer requires all four nonempty bags and evaluates masks after encoding. Its feature core supports nonempty subsets. This tests deliberately withheld views, not natural clinical missingness or a newly implemented raw missing-view upload API. No physical registration of fields across zooms is assumed.
- **Historical scores retain their tasks.** Notebook93.73%/94.99% concerns eight-class individual40× images with image-random reused validation. Earlier45.7%/38.3% concerns eight-class case validation;87.0%/80.7% is its retrospective binary readout. Lang's86.22% is binary image accuracy. None is a same-task comparator proving a 95-to-40 drop or a new five-class improvement. [Original/current explanation](../CAPSTONE_BEFORE_AND_NOW.md), [primary-source critique](FIVE_CLASS_RESEARCH_CRITIQUE.md).
- **Model score, not clinical risk.** AUROC is ranking, not accuracy. CE/Brier and saved probabilities do not alone establish clinical calibration. A successful demonstration establishes execution, not diagnostic accuracy, safety or benefit.

A measured improvement screen of +0.02 macro F1 on both arrangements would be useful exploratory evidence; passing it would not mean the 94–95% accuracy goal was achieved. Any eventual 95% statement must name the actual endpoint, view setting, role/selection scheme and complete denominator. One trial, one view subset, fitting accuracy or binary routing is insufficient for the requested whole-system claim.

## Final-package consistency issues

The current five-class report/brief and top README table are aligned with the accepted result. However, the older [project report](../../paper/breakhis-project-report.md), [delivery plan](../CAPSTONE_DELIVERY_PLAN.md), [rehearsal guide](../../paper/capstone-rehearsal.md), eight-class deck and [recorded demo](../CAPSTONE_DEMO.md) retain historical task/status language. The older report says the five-class system has not been trained, asks an eight-class question and says accuracy equals balanced accuracy. Those statements describe an earlier checkpoint; they are unsuitable as the current five-class report. Preserve that evidence with a clear historical label and route the final handoff to the current report.

The registered CNN specification also retains status-text leftovers: its closing paragraph says the draft remains unregistered, and its software paragraph says timm is absent despite the successful preparation receipt. The authoritative registration header/configuration correctly state the recipe status and mandatory smoke gate. Clarify these inconsistencies in the current status/claims page or a documented source revision before fitting as appropriate; do not silently mutate hash-bound experiment bytes between smoke and production. Status prose is not evidence that a result exists.

The current five-class report's phrase “router is reasonably useful” should be read only as an exploratory numerical observation. Prefer “router accuracy is higher than final subtype accuracy in these validation roles” to avoid suggesting demonstrated clinical usefulness. The current report should link the registered CNN recipe and identify its exact accepted stage without anticipating an outcome.

## Minimum final deliverable

1. **One current report and short professor brief.** Lead with the five labels and decision diagram, explain case/view pooling, show the four-row table above, the failed gate, key subtype weaknesses and the 46/8/28/31-case limitations. Add the CNN comparison only after full acceptance; otherwise give its exact stage. Put the old eight-class studies in a dated historical section or appendix.
2. **One consistent presentation and rehearsal guide.** Use the same task names, denominators and scores as the report. Show raw accuracy, macro F1 and balanced accuracy together. Explain why the historical notebook95% cannot be transferred to the current task. Do not let an older PDF become the current presentation merely because it is already rendered.
3. **A reproducibility bundle.** Include registered recipes/configuration, source/dependency/weight/data/split receipts, complete aggregate ledger, selected-model/reload checks and independent acceptance records. Provide commands and expected artifacts. Keep raw data, weights, private IDs, credentials and supplied reference files out of public history.
4. **A clearly labeled worked example.** At minimum, display one deterministically chosen case's accepted five-class saved scores across all 15 subsets, labeled as a saved-output viewer. Disclose the selection rule and its timing; do not search for a correct prediction or imply retrospective viewer selection was preregistered. Retain the checked eight-class raw-image replay as historical execution evidence. A new live uploader or extra GPU demonstration is not required to make the current result reviewable.
5. **An honest conclusion and next step.** State what executed successfully, what did not generalize, and what remains unproven. The minimum defensible capstone is a transparent comparison with reusable code and a clear negative/mixed finding. Finish packaging and rehearsal after this fixed comparison; do not convert the deadline into a larger encoder, threshold, seed or split search.

## Paper and novelty judgment

No novel-method claim is justified by the present evidence. Frozen feature extraction, logistic heads, hierarchical decisions, EfficientNet transfer learning and mean set pooling are established components; the reference paper already uses the same five-label dual-step hierarchy. The current project has no isolated new mechanism, external confirmation or exact published-method reproduction demonstrating a contribution beyond the bounded engineering study. [Primary-source research critique](FIVE_CLASS_RESEARCH_CRITIQUE.md), [related-work record](../RELATED_WORK.md).

A careful capstone or technical report about reproducible grouped evaluation and the observed generalization limits is defensible. Whether it later supports a particular publication requires Professor Dong's assessment and a separate contribution/confirmation plan; publication acceptance cannot be inferred from clean software or a high reused-validation score. Keep TCGA and novelty work deferred while completing this BreakHis deliverable.
