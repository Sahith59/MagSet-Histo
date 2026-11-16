# Independent actual-output review: first trained five-class comparison

November 15, 2026. **Accept the execution and complete saved results. The registered dual-step improvement gate failed.** The hierarchy improves arrangement 813 but worsens 812; reporting only 813 would misrepresent the comparison. Neither pipeline reaches 94–95% end-to-end five-class accuracy.

This review independently checks the first trained direct-five control and binary-plus-malignant-four pipeline under the [registered specification](../BREAKHIS_FIVE_CLASS_SPEC.md). These are actual trained five-category results, separate from the earlier eight-class results and retrospective score collapses. The prediction unit remains a provider-case bag, with all 15 nonempty magnification subsets.

## Accepted main results and gate

Each result averages per-subset metrics equally over all 15 subsets and then five overlapping trials. F1 is not accuracy.

| Arrangement | Trained pipeline | Five-class macro F1 | Case accuracy | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| 812 | Direct five | 0.516146 | 66.67% | 58.27% | 67.50% |
| 812 | Dual step | 0.497517 | 65.83% | 56.53% | 67.50% |
| 813 | Direct five | 0.503473 | 64.67% | 58.07% | 65.00% |
| 813 | Dual step | 0.532497 | 68.83% | 61.33% | 72.50% |

Dual minus direct mean-subset F1 is **−0.0186285714 on 812** and **+0.0290243386 on 813**. Accuracy changes are −0.83 and +4.17 percentage points. All five pooled recalls are positive on both arrangements, but the gate requires at least +0.02 F1 and nondecreasing accuracy on **both**. Arrangement 812 fails; the overall gate fails. No consistent hierarchy improvement or new default winner is established by this comparison.

All selected complete pipelines are retained below; selection uses each arm's earliest maximum validation mean-15 end-to-end macro F1 across the four registered C values.

| Arrangement | Trial | Direct F1 | Dual F1 | Direct accuracy | Dual accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 1 | 0.606190 | 0.567778 | 69.17% | 66.67% |
| 812 | 2 | 0.382095 | 0.367333 | 51.67% | 50.83% |
| 812 | 3 | 0.417206 | 0.384698 | 62.50% | 59.17% |
| 812 | 4 | 0.473651 | 0.480317 | 66.67% | 71.67% |
| 812 | 5 | 0.701587 | 0.687460 | 83.33% | 80.83% |
| 813 | 1 | 0.496222 | 0.569365 | 55.00% | 69.17% |
| 813 | 2 | 0.495111 | 0.504444 | 74.17% | 75.00% |
| 813 | 3 | 0.451111 | 0.476296 | 70.00% | 72.50% |
| 813 | 4 | 0.523492 | 0.550159 | 67.50% | 73.33% |
| 813 | 5 | 0.551429 | 0.562222 | 56.67% | 54.17% |

## Routing, typing and class weaknesses

The dual pipeline's actual trained router is substantially stronger than its final subtype endpoint. Malignant typing below supplies the true malignant group and evaluates the specialist; it is diagnostic, not end-to-end accuracy.

| Arrangement | Actual router accuracy | Malignant sensitivity | Benign specificity | True-malignant four-type accuracy | Final five-class accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 812 | 87.83% | 94.33% | 81.33% | 55.67% | 65.83% |
| 813 | 88.33% | 95.33% | 81.33% | 56.33% | 68.83% |

Across the 600 repeated validation case/subset events per arrangement, the dual router makes 56 benign false positives on each arrangement and 17/14 malignant false negatives on 812/813. A further 132/117 events are correctly routed malignant but assigned the wrong malignant subtype. Thus final accuracy is router accuracy minus 132/600 and 117/600 respectively. All per-subset hard-routing accuracy bounds pass.

Final-label recalls from pooled repeated validation events:

| Final label | Direct 812 | Dual 812 | Direct 813 | Dual 813 |
| --- | ---: | ---: | ---: | ---: |
| Benign | 80.67% | 81.33% | 75.67% | 81.33% |
| Ductal | 81.33% | 78.67% | 92.00% | 97.33% |
| Lobular | 56.00% | 56.00% | 56.00% | 54.67% |
| Mucinous | 62.67% | 57.33% | 49.33% | 56.00% |
| Papillary | 10.67% | 9.33% | 17.33% | 17.33% |

Pooled support is 300 benign and 75 for each malignant label per arrangement. These are repeated events, not that many independent cases. Papillary remains particularly weak; correctly routed papillary cases frequently become ductal or mucinous. Benign pooling has removed benign-subtype distinctions from the target, but malignant-subtype errors remain.

The dual pipeline's fitting balanced accuracy is 96.81%/96.64%, versus 56.53%/61.33% validation. Its specialist's fitting balanced accuracy is 98.84%/98.87%, versus 55.67%/56.33% validation. These observed gaps and error categories support limited generalization under this representation and protocol. They do not isolate a sole encoder, pooling, stain, biological or annotation cause, nor prove that another architecture would solve it.

## Every input subset was retained

Five-trial validation macro F1 for each of the 15 registered subsets:

| Available magnifications | Direct 812 | Dual 812 | Direct 813 | Dual 813 |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.473810 | 0.371905 | 0.351619 | 0.482794 |
| 100 | 0.545238 | 0.545238 | 0.491556 | 0.488508 |
| 200 | 0.538857 | 0.481238 | 0.489841 | 0.522222 |
| 400 | 0.467905 | 0.434571 | 0.509683 | 0.510000 |
| 40+100 | 0.480476 | 0.486190 | 0.441841 | 0.520127 |
| 40+200 | 0.513810 | 0.519524 | 0.559175 | 0.546794 |
| 40+400 | 0.443143 | 0.439524 | 0.505841 | 0.461841 |
| 100+200 | 0.561238 | 0.561238 | 0.491111 | 0.516508 |
| 100+400 | 0.514571 | 0.505238 | 0.524444 | 0.569841 |
| 200+400 | 0.592190 | 0.491905 | 0.536508 | 0.554286 |
| 40+100+200 | 0.539524 | 0.545238 | 0.496508 | 0.532508 |
| 40+100+400 | 0.506190 | 0.511905 | 0.535175 | 0.552508 |
| 40+200+400 | 0.487143 | 0.511905 | 0.559175 | 0.576508 |
| 100+200+400 | 0.545238 | 0.545238 | 0.524444 | 0.576508 |
| 40+100+200+400 | 0.532857 | 0.511905 | 0.535175 | 0.576508 |

No favorable trial, arrangement or magnification setting replaces the complete endpoint. Mean trial worst-subset F1 is 0.379524/0.331619 for direct and 0.277619/0.426286 for dual. The separate minimum across all trial/subset cells is 0.200000/0.200000 for direct and 0.150000/0.360000 for dual.

## Independent verification performed

The standalone private reviewer imports no production `magset` implementation or metric helper. It independently reconstructs qualified cases and original eight-label roles from the manifest/protocol, checks accepted feature bytes and coverage, derives case/view means and fitting-only normalizers, and uses separate NumPy/SciPy calculations plus scikit-learn metric functions.

- **Complete matrix:** ten runs, two arms and four C values each; 80 saved candidate pipelines, 120 classifier heads, and 64,800 saved fitting/validation case-subset records. No missing or unexpected model/prediction artifacts were accepted.
- **Provenance and roles:** frozen source archive and embedded commit, configuration, all implementation modules, all four cache NPZ/JSON pairs, manifest/protocol, dependency versions, exact fitting/validation/excluded memberships and original/final truths were verified. All ten normalizers match the independent fitting-only reconstruction bitwise. The actual zero-fit audit receipts match independently.
- **Numerical replay:** every saved candidate coefficient array reproduces its probabilities, stable log probabilities, binary and specialist scores and explicit final decisions. Maximum probability/log-score replay difference is **1.7763568394002505e-14**.
- **Metrics and selection:** all per-subset, pooled, mean, all-view and worst metrics; binary AUC/AP, sensitivity/specificity, CE/Brier; true-malignant diagnostics; all candidate C choices; and complete selected summaries were recomputed. Maximum metric difference is **4.440892098500626e-16**. Independent public aggregate summaries and paired gate values have **zero difference** from the collector.
- **Fitting objective:** without refitting, the reviewer recomputed each head's prescribed class-balanced × 1/15 weighted logistic objective and gradient from fitting rows. All 120 normalized maximum absolute gradients are at most the registered 1e-4 tolerance; the largest is **9.999008111280675e-05**. Effective head mass is 46 fitting cases for direct/router and 35 for the specialist, with the intended class balancing.
- **Privacy:** a separate recursive scan of 137,023 public-comparison fields found no forbidden provider identifiers or private path fields.

The complete independent output review passed **4,924,986 assertions**. The reviewer performed zero classifier fits, encoder forwards, raw-image scoring, outer-role predictions, resplits or scheduler submissions. Existing coefficient replay on allowed saved fitting/validation records is verification, not a newly fitted experiment.

| Accepted artifact | Binding |
| --- | --- |
| Training source commit | `0471cd4ab8fb1da71eba8870f2ea0fa59ce0e8f3` |
| Immutable source archive SHA256 | `8099025d5f010b40a00f6d05f0c82fa547df76673f8a4fe6d2a9e3f0f7191cd7` |
| Configuration SHA256 | `d3a5ac3ca581666646d2d1a90dc3bccad72699253dd522f7e964f44daa44345b` |
| Actual input audit SHA256 | `ec166e757816c028025289f288bf120b55dd764156739ef4566bc4f387d3a0aa` |
| Complete comparison SHA256 | `bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795` |
| Private independent output-review SHA256 | `fdd3ff7d6ca06191ae58b64a020e63a5578a204419634b31b91787872caa1915` |

The evidence remains exploratory provider-case validation: 22/20 distinct validation cases, 31 in their union, with overlapping trials and repeated subsets. All 82 provider cases entered development somewhere, clinical identity is not fully reconciled and no globally fresh confirmation cohort remains. The two pipelines also differ in head-specific balanced objectives and capacity; this experiment does not identify a hierarchy-only causal effect. Preserve the complete mixed result and failed gate. Further work requires its own bounded registration and cannot recast this outcome as 94–95%, a fresh test, clinical validation or novelty evidence.
