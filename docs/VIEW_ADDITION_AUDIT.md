# What happens when we add another view?

Posthoc descriptive analysis of the existing fixed native development pilot. No new classifier was fitted and no reserved patient was scored.

For each of the same 16 development patients, compare every nonempty availability subset with each subset obtained by adding exactly one view. There are 28 such edges and 448 patient/edge comparisons per arm. These remain 16 independent patients, not 448 observations from different patients.

| Arm | Wrong → right | Right → wrong | Both right | Both wrong | Patients harmed at least once | Newly wrong with higher maximum probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | 26 | 22 | 167 | 233 | 5 | 8 |
| duplicate | 25 | 21 | 170 | 232 | 5 | 9 |
| random | 26 | 22 | 167 | 233 | 5 | 8 |
| affinity | 25 | 21 | 170 | 232 | 5 | 8 |

“Harmed” means that this fixed classifier changes a correct label to a wrong label after another view is supplied. Higher maximum probability describes its numerical confidence; it is not evidence that the confidence is calibrated. A patient can be helped on one edge and harmed on another.

## Availability scores already measured

| Arm |0.25mpp only F1|0.5mpp only F1|1.0mpp only F1|2.0mpp only F1|All four F1|
| --- | ---: | ---: | ---: | ---: | ---: |
| mean | 0.239583 | 0.229167 | 0.364583 | 0.489583 | 0.343750 |
| duplicate | 0.239583 | 0.229167 | 0.364583 | 0.489583 | 0.343750 |
| random | 0.239583 | 0.229167 | 0.364583 | 0.489583 | 0.343750 |
| affinity | 0.239583 | 0.229167 | 0.364583 | 0.489583 | 0.343750 |

## Every one-view addition for the mean control

| From mpp | To mpp | Helped | Harmed | Mean true-label probability change | Mean Brier change ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.25 | 0.25+0.5 | 0 | 0 | 0.024524 | -0.046920 |
| 0.25 | 0.25+1.0 | 2 | 0 | 0.036700 | -0.084726 |
| 0.25 | 0.25+2.0 | 3 | 0 | 0.031085 | -0.095683 |
| 0.5 | 0.25+0.5 | 0 | 0 | -0.020875 | 0.010824 |
| 0.5 | 0.5+1.0 | 2 | 1 | 0.016788 | -0.045684 |
| 0.5 | 0.5+2.0 | 3 | 1 | 0.011065 | -0.055744 |
| 1.0 | 0.25+1.0 | 1 | 1 | -0.043500 | 0.041891 |
| 1.0 | 0.5+1.0 | 0 | 1 | -0.018012 | 0.023189 |
| 1.0 | 1.0+2.0 | 2 | 0 | -0.007867 | 0.001422 |
| 2.0 | 0.25+2.0 | 1 | 2 | -0.027099 | 0.007682 |
| 2.0 | 0.5+2.0 | 0 | 2 | -0.001720 | -0.010123 |
| 2.0 | 1.0+2.0 | 0 | 0 | 0.014148 | -0.021829 |
| 0.25+0.5 | 0.25+0.5+1.0 | 2 | 0 | 0.015799 | -0.034695 |
| 0.25+0.5 | 0.25+0.5+2.0 | 1 | 0 | 0.011123 | -0.044548 |
| 0.25+1.0 | 0.25+0.5+1.0 | 0 | 0 | 0.003623 | 0.003111 |
| 0.25+1.0 | 0.25+1.0+2.0 | 1 | 1 | 0.012167 | -0.031171 |
| 0.25+2.0 | 0.25+0.5+2.0 | 0 | 2 | 0.004561 | 0.004215 |
| 0.25+2.0 | 0.25+1.0+2.0 | 0 | 1 | 0.017782 | -0.020214 |
| 0.5+1.0 | 0.25+0.5+1.0 | 2 | 1 | -0.021865 | 0.021813 |
| 0.5+1.0 | 0.5+1.0+2.0 | 1 | 0 | 0.003173 | -0.018605 |
| 0.5+2.0 | 0.25+0.5+2.0 | 1 | 2 | -0.020818 | 0.022020 |
| 0.5+2.0 | 0.5+1.0+2.0 | 1 | 1 | 0.008896 | -0.008544 |
| 1.0+2.0 | 0.25+1.0+2.0 | 0 | 2 | -0.023465 | 0.009298 |
| 1.0+2.0 | 0.5+1.0+2.0 | 0 | 2 | -0.006972 | 0.003162 |
| 0.25+0.5+1.0 | 0.25+0.5+1.0+2.0 | 1 | 1 | 0.007321 | -0.025967 |
| 0.25+0.5+2.0 | 0.25+0.5+1.0+2.0 | 2 | 1 | 0.011997 | -0.016114 |
| 0.25+1.0+2.0 | 0.25+0.5+1.0+2.0 | 0 | 0 | -0.001223 | 0.008315 |
| 0.5+1.0+2.0 | 0.25+0.5+1.0+2.0 | 0 | 0 | -0.017717 | 0.014451 |

## What this permits us to say

Additional input can change a fixed model from correct to wrong. This supports investigating failure patterns, but does not prove that a new gate, uncertainty method, transport mechanism, or larger encoder will fix them. Selective fusion has substantial prior work. The agreement-specific pilot still loses to the duplicate control. No new primary endpoint, best mask, model setting, or significance claim is selected here.

Patient-level examples remain private; the public JSON retains aggregates for every edge and arm. All counts reuse the same small development cohort. The 32 reserved patients remain outside this analysis.

Input report SHA256: `7972dae0ab004fb98c52dc56747a778c627ba81a074c497ed8356aab29377612`. Analysis source SHA256: `4d1e485b40713204042d7d81a5d7d422e457b0fcf11cc74aaeadb2161b029c1b`.

[Original pilot](NATIVE_DEVELOPMENT_RESULTS.md) · [Next tasks](NEXT_STEPS.md)
