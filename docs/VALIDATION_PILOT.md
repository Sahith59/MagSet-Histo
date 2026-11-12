# Initial frozen-feature validation results

All 30 configurations completed, and their saved probabilities were independently checked against the reported metrics. There were no convergence warnings, numerical warnings, or tracebacks in their validation logs. Outer-test scoring was disabled.

Each trial uses only eight inner-validation cases, one per subtype. The five provider trials overlap in cases. These are noisy exploratory validation results used to plan the next experiments, not final test performance or independent clinical evidence.

| Encoder | Fusion | Mean validation macro F1 across subsets/trials | Mean validation full-view macro F1 |
| --- | --- | ---: | ---: |

| phikon-v2 | mean | 0.3868 | 0.4125 |

| phikon-v2 | concat | 0.3513 | 0.4208 |

| phikon-v2 | late | 0.3513 | 0.3642 |

| resnet50 | mean | 0.2711 | 0.2808 |

| resnet50 | concat | 0.2929 | 0.3042 |

| resnet50 | late | 0.2419 | 0.2642 |


The pathology-pretrained encoder has higher average pilot scores than the ImageNet encoder in these comparisons. This is a descriptive observation, not a statistically established advantage. The low scores and sensitivity to available views justify investigating field pooling, sample scarcity, and failure patterns before adding a complicated mechanism.

Next controlled comparisons: case-balanced image-level heads with aggregation, learned within-scale pooling, masked gating, and set attention under the same held-out cases and tuning constraints. Repeat validation selection/seeds to assess instability; do not repeatedly tune against outer-test outcomes.

Baseline source revision: `76d4d7c8b5afa66fdc6f982ebed8de912e4ec637`. Archive and protocol details are in [the data audit](DATA_AUDIT.md) and [pilot protocol](BENCHMARK_PROTOCOL.md).
