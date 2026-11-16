# GenBio actual-results validity review

November 16, 2026. **The complete GenBio result is internally consistent and valid as a reused development comparison. It does not establish a consistent improvement in the requested two-step task.** GenBio improves malignant subtype information, while its binary routing loses accuracy. The strong arrangement-813 direct-five result must remain a one-stage result.

This review uses the accepted [public ledger](../evidence/BREAKHIS_GENBIO_RESULTS.json), SHA256 `92e2ff976b1be20a69a0ccee2147537b2905293926675d2ff3be85283cc589d1`. Classifier source is `dfaa6c190f2ae8a32028aea6ba70ca82154ee6f6`; classifier configuration SHA256 is `196608c55ab8e3418f03b052f712fc77dcfe8155d3b38293d69c51610e435737`. No new models, composite predictions or GPU jobs were produced during this review.

## What the measurements say

Accuracy/F1 averages five overlapping trials and all 15 nonempty magnification subsets in each arrangement.

| Pipeline | Final accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| C: single-center Midnight, two-step |75.67% / 75.00%|.589691 / .564292|
| F: three-crop Midnight, two-step |75.50% / 76.83%|.592510 / .579947|
| P: head-specific centers, two-step |76.33% / 75.67%|.602715 / .602512|
| GenBio, two-step |70.50% / 75.17%|.583661 / .581348|
| GenBio, direct-five |71.33% / 77.83%|.595998 / .640603|

Against the primary C reference, GenBio two-step accuracy changes **−5.17 / +0.17 percentage points** and F1 **−.006030 / +.017056**. Both the primary and secondary two-arrangement improvement screens fail. P retains higher two-step F1 in both arrangements; F retains higher 813 two-step accuracy. GenBio's 77.83%/.640603 is the strongest arrangement-813 one-stage row among the completed comparisons inspected, with a weaker first arrangement. It is neither a two-step result nor a uniform winner.

The two-step all-four-view accuracies are **72.5% / 75.0%**. They are separate from the mean-15-subset endpoint and do not approach 95%.

## Representation gains and routing losses

| Two-step diagnostic | GenBio 812 / 813 | C 812 / 813 |
| --- | ---: | ---: |
| Binary accuracy |82.00% / 89.83%|92.00% / 92.67%|
| True-malignant typing accuracy |60.00% / 67.67%|56.67% / 56.67%|
| Conditional subtype cross-entropy |1.0091 / .9938|1.2331 / 1.0897|
| Binary ROC AUC |.8742 / .9850|.9850 / .9858|

Conditional subtype accuracy improves by **3.33 / 11.00 points** versus C, and conditional cross-entropy improves in both arrangements. This is evidence for complementary representation, rather than a claim based only on one favorable hard decision. Final recall remains uneven: benign **85/95%**, ductal **49.33/50.67%**, lobular **100/100%**, mucinous **48/41.33%**, papillary **26.67/29.33%**.

Among 600 dependent validation events per arrangement, GenBio makes:

- 812: **45 benign-to-malignant**, **63 malignant-to-benign**, and **69 malignant wrong-type** errors.
- 813: **15 benign-to-malignant**, **46 malignant-to-benign**, and **88 malignant wrong-type** errors.

Papillary-to-benign errors account for **50/75** papillary events in 812 and **40/75** in 813. Routing loss is therefore concentrated, not an abstract overall percentage. The 812 AUC decline also gives no support for describing this solely as a threshold-calibration problem. Threshold search remains outside the registered experiment.

Overall probability quality is mixed. Versus C, five-category cross-entropy/Brier worsen by **.00453/.02261** in 812, but improve by **.08845/.02499** in 813. A subtype gain is not equivalent to better routing or universally better confidence. For the direct arm, a binary score derived by summing malignant probabilities and thresholding at .5 is a different decision from its five-way argmax; its derived binary diagnostic must not be substituted for direct final accuracy.

## Fitting diagnostics and temperature selection

The fitting diagnostic removes the whole query case from both support and centering. It is not self-included training accuracy. GenBio two-step leave-case-out accuracy is **64.87% / 65.39%**, F1 **.481153 / .494170**, versus C F1 **.417942 / .389295**. The improved fitting F1 is consistent with some cross-case representation gain. Rare-class reliability remains limited: GenBio fitting recalls are lobular **35.33/42.00%**, mucinous **50.93/59.73%**, papillary **15.11/16.89%**. Validation lobular 100% does not establish general lobular reliability with only two fitting cases per role.

All 20 arm/role selections match the earliest maximum validation F1 over the registered four temperatures. Two-step selections are T=.1 in seven roles, .03 in two, .3 in one; direct selections are .1 in six, .03 in one, .3 in three. T=.01 is never selected. This is an outcome of the fixed grid, not permission to widen it after seeing results. Direct and dual choose independently within their declared budgets.

## Provenance and independent verification

I verified the ledger/configuration hashes, all current recorded implementation hashes, accepted feature-audit/input-audit bytes and local manifest/protocol bytes. The six unchanged task/data/KDE method hashes match C/F/P. All ten roles retain **46 fitting/eight validation/28 excluded** cases and supports **11/25/2/5/3** fitting, **4/1/1/1/1** validation. The complete ledger contains 80 candidates, 160 prediction archives and 64,800 dependent records. No excluded classifier predictions are reported.

Independent arithmetic checked every selection, candidate subset mean, selected aggregate and class recall, then reconstructed accuracy/F1/balanced accuracy from **all 2,400 candidate subset confusion matrices** and summed the eight selected pooled matrices. The initial arithmetic/provenance pass contains 1,237 assertions; these are checks on artifacts, not independent patient observations. This review did not independently replay raw images or the private prediction archives; the accepted collector performed the full saved-prediction replay.

The frozen representation concatenates three channel-specific CLS vectors, dimension 4608, under the pinned official GenBio input/channel policy. The experiment changes the reader and its prescribed preprocessing together. Feature production used **7,909 image inputs / 23,727 channel passes**; zero new encoder forwards applies only to classifier collection. Public weights, source and configuration pins remain in the ledger. Support labels still define density classes despite zero optimized coefficients.

## Future combination: hypothesis only

P's stronger router and GenBio's improved conditional subtype scores make a fixed hybrid a reasonable **future, separately registered hypothesis**. Aggregate scores cannot determine whether their errors complement on the same cases. No hybrid was constructed here, and the current two-lane registration explicitly excludes automatic combinations.

Holding GenBio's current case-conditional predictions fixed, a hypothetical perfect router yields at most **80.00% / 83.83%** final accuracy because validation is half benign and half malignant. A perfect specialist with its current router is bounded by **82.00% / 89.83%**. These arithmetic bounds are not fitted models or predictions of a P/GenBio cascade.

There is a further definition issue: GenBio's reported case-conditional probabilities are obtained after joint field/view probability pooling, so they already depend on its own router's field weights. Replacing routing before pooling would also change those conditional probabilities. Any future recipe must declare the combination level, preserve exact trusted case/subset joins, bind both inherited selections and report all outcomes without another threshold or mode search.

All 82 qualified provider keys entered development, 31 distinct keys appear in the reused validation arrangements, clinical identity remains unresolved, and fields/masks/trials are dependent. No fresh clinical generalization, novelty or 94–95% result follows from this accepted comparison.
