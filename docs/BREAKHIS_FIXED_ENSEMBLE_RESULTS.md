# Fixed model vote: completed result

November16,2026. The registered equal-probability mixture did not improve the retained two-step model. Final five-category accuracy is **74.17%/72.17%**, compared with the similarity reference **75.67%/75.00%**. Keep the reference.94–95% final accuracy remains unachieved.

| Model | Arrangement | Final accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Retained similarity two-step |812|75.67%|.5897|62.87%|82.50%|
| Fixed equal vote, direct-five |812|75.17%|.5782|64.67%|77.50%|
| Fixed equal vote, two-step |812|74.17%|.5781|64.67%|80.00%|
| Retained similarity two-step |813|75.00%|.5643|62.40%|80.00%|
| Fixed equal vote, direct-five |813|73.33%|.5314|60.73%|75.00%|
| Fixed equal vote, two-step |813|72.17%|.5326|60.67%|75.00%|

The first three score columns average all15 available-view combinations and five trials. All-four-view accuracy is a separate setting. The mixture improves some first-arrangement class recalls/balanced accuracy, but its final accuracy and macro F1 are lower on both arrangements. The descriptive improvement screen fails.

The two-step binary decision scores92.83%/95.00%, while true-malignant typing scores59.00%/53.67%. The95% binary score is **not**95% final five-category accuracy. Final benign/ductal/lobular/mucinous/papillary recall is90.0/80.0/70.7/61.3/21.3% on812 and91.3/74.7/57.3/40.0/40.0% on813. Complementary constituent errors did not yield a consistent overall improvement under the fixed equal vote.

## What was tested

For the same role/arm/case/view subset, mix the previously accepted selected Midnight case-logistic and centered-cosine similarity joint-five probabilities with equal weights. The published probabilities and hard routing use canonical float64 arithmetic; stable joint logs support loss calculations. No weight, class-specific rule or threshold was selected. No new coefficient fitting or image-encoder execution occurred.

Each constituent had four previous selection settings, so this ensemble inherits eight component selection opportunities per role/arm. It is an extra reused-development comparison, not an equal-four-total-budget method claim. No ensemble fitting metric was produced because constituent fitting diagnostics used different self-exclusion conditions.

## Complete execution and evidence

All ten paired tasks4982891 completed successfully in10–12seconds. Collector4982892 completed successfully in13seconds. The input audit verified all40 accepted source archives and4,800 dependent records against source-run/candidate/prediction hashes, original roles/labels and canonical case/subset joins. The collector independently replayed all20 output archives and2,400 dependent records. Root checked12 aggregate means. The full source suite passed299 tests; a synthetic independent review had also checked1,000 exact-routing-boundary pairs and extreme-underflow behavior.

Producer `0ceb91c4036333d5043aa60f769a082e277f1eb6`; configuration SHA256 `4f47a78e43a5cd008ebea80cd0bd4e4dc05d03cb6f7bb28e17f9d107d70c3d96`. [Accepted complete JSON](evidence/BREAKHIS_FIXED_ENSEMBLE_RESULTS.json), SHA256 `7962b981684076a74526e35f052709e92944654db47c6c0feb2ffc606063cf69` · [Fixed recipe](BREAKHIS_FIXED_ENSEMBLE_SPEC.md) · [Independent implementation review](reviews/FIXED_ENSEMBLE_IMPLEMENTATION_REVIEW.md) · [Retained reference](BREAKHIS_KERNEL_RESULTS.md).

Only BreakHis is evaluated, with unchanged46/8/28 provider roles.31 distinct reused validation keys, unresolved clinical identity and dependent repeated masks remain limits. These results do not establish a fresh clinical test or novelty. The separate frozen three-crop coverage comparison is the next measured hypothesis.
