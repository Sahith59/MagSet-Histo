# Decision-specific centering: completed comparison

November 16, 2026. Giving the two decision stages their own reference centers produces our **highest observed macro F1 in both arrangements**, at **.6027 / .6025**. Final five-category accuracy is **76.33% / 75.67%**. With all four zooms present it is **85.00% / 85.00%**, a separate endpoint. **94–95% final accuracy remains unachieved.**

The gains are modest and mixed. Relative to the fixed three-crop reference, macro F1 changes by +.0102 / +.0226, while accuracy changes by +0.83 / −1.17 percentage points. Both registered mode screens fail: the head-specific result misses the +.02 F1 requirement on 812 and nondecreasing accuracy on 813. Keep the reference and complete scorecard visible.

| Two-step method | Accuracy 812 / 813 | Macro F1 812 / 813 | Balanced accuracy 812 / 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: | ---: |
| Single-center input, C |75.67% /75.00%|.5897 /.5643|62.87% /62.40%|82.50% /80.00%|
| Three-crop input, F |75.50% /76.83%|.5925 /.5799|62.80% /64.93%|82.50% /82.50%|
| P: shared class-balanced center |76.17% /75.33%|.5992 /.5778|63.87% /64.53%|85.00% /82.50%|
| P: decision-specific centers |76.33% /75.67%|.6027 /.6025|64.73% /66.47%|85.00% /85.00%|

Headline scores average all 15 nonempty zoom combinations and five overlapping trials per arrangement. Do not combine a model's best 812 score with another model's best 813 score as a single result. The all-four-view column does not replace the registered 15-subset endpoint.

## What changed, simply

The frozen reader still describes each original image using the accepted three-crop F features. The classifier still compares those descriptions with fitting examples. A center is the reference point subtracted before measuring directions and similarities.

The first mode gives each of the five class means equal weight and uses that one center for every decision. The second keeps this center for direct-five prediction, but gives the benign/malignant decision a two-group center and the malignant specialist a four-malignant-type center. No benign examples contribute to that specialist's center. Centers use fitting labels; this is supervised preprocessing, even though no classifier coefficients or encoder weights are optimized.

The original center was mathematically valid. These alternatives test cosine geometry rather than repairing invalid probability priors. Whole query cases are removed before support and center construction for fitting diagnostics. The validation query's true label never chooses a center. Temperatures, density weights, probability pooling, hard routing and provider roles remain fixed.

Direct-five controls are exactly duplicated across modes: accuracy 75.50% /74.33%, F1 .5887 /.5892. Each named mode/arm has four temperature settings. Choosing between modes inherits eight logical settings per arm, with only four structurally unique direct settings and eight dual settings. Repeated controls are not independent evidence.

## What still limits final accuracy

The decision-specific binary router scores 94.83% /93.00%, while true-malignant typing scores 58.33% /62.33%. The binary score is not final five-category accuracy. Benign/ductal/lobular/mucinous/papillary recall is 95.67/65.33/93.33/49.33/20.00% on 812 and 91.00/61.33/98.67/36.00/45.33% on 813. Papillary recognition improves on 813, but several types remain unreliable and benign recall also changes.

The head-specific mode improves macro F1 slightly over the shared mode on 812 and more on 813; raw gains over that mode are only 0.17/0.33 points. This is evidence about these fixed development comparisons, not a biological mechanism, an established novel algorithm or fresh clinical generalization.

Probability quality also worsens relative to F: cross-entropy increases by .0570/.1870 and Brier error by .0271/.0978 on 812/813; lower is better for both. The macro-F1 gain therefore does not imply better confidence estimates. Do not tune a new calibration rule on these same outcomes and present it as independent confirmation.

## Complete evidence

Input audit 4982973 completed successfully in 8 seconds. All ten paired CPU tasks 4982974 completed in 53–57seconds, then collector 4982975 completed in 8m18 with empty error output. The collector verified all 160 logical candidate pipelines, 320 prediction archives and 129,600 dependent records. After removing the structurally duplicated direct controls, these correspond to 120 candidate definitions, 240 archive definitions and 97,200 records. Structural counts are not independent patients or guaranteed different numerical outcomes.

Root independently checked 136 selection, aggregate and class-recall calculations and verified identical direct predictions across modes. There were no new encoder forwards or optimized classifier coefficients. Full 330 tests and 16 final focused tests passed; immutable source CI passed. CPU collection replays cached-feature classifier arithmetic, not raw-image encoding.

Producer `f1b63139f54a790c80ec671522c38a77205e2ff0`; config SHA256 `4f13430d458561964186b0a154ffcaa6a51d8fea49aecc9e8251d7c40eda6f7e`. Accepted comparison SHA256 `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c`.

[Complete accepted JSON](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) · [Registered recipe](BREAKHIS_PRIOR_CENTERED_SPEC.md) · [Independent implementation review](reviews/PRIOR_CENTERED_IMPLEMENTATION_REVIEW.md) · [Actual-outcome critique](reviews/NATIVE_AND_CENTERING_BENCHMARK_REVIEW.md).

BreakHis-only fitting/evaluation, 46 fitting/eight validation/28 excluded provider cases, 31 distinct reused validation keys and unresolved clinical identity remain unchanged. All 82 provider keys entered development; there is no globally fresh confirmation cohort. No automatic combination with the failed native-input intervention is justified.
