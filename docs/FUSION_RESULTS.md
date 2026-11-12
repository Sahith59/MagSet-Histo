# Field and learned-fusion validation results

All 20 field-linear and 30 hierarchical neural configurations completed. Each saved probability record independently reproduces the reported metrics. All selected models passed reload checks. Outer-test outcomes were not scored.

These are exploratory results from eight inner-validation cases per trial and five overlapping provider holdouts. Hyperparameters were selected on these validation cases. The tables do not establish independent generalization or statistical significance.

## Field aggregation controls

Both controls use identical case-balanced field normalization and the same four C values. `field-late` classifies individual fields, then averages probabilities. `field-early-late` classifies the mean field embedding. Both average available scale probabilities.

| Encoder | Aggregation | Mean subset macro F1 | Full-view macro F1 |
| --- | --- | ---: | ---: |
| phikon-v2 | field-late | 0.3574 | 0.3742 |
| phikon-v2 | field-early-late | 0.3579 | 0.3642 |
| resnet50 | field-late | 0.3229 | 0.3117 |
| resnet50 | field-early-late | 0.2880 | 0.3367 |

## Neural controls with Phikon-v2

All methods use the same frozen fields, 100 fixed epochs, four optimizer settings, split, and training seed. Parameter counts differ; see the registered protocol.

| Field pooling | View fusion | Mean subset macro F1 | Full-view macro F1 |
| --- | --- | ---: | ---: |
| mean | mean | 0.3268 | 0.3433 |
| mean | gated | 0.3349 | 0.3408 |
| mean | attention | 0.3387 | 0.3450 |
| attention | mean | 0.3474 | 0.3575 |
| attention | gated | 0.3347 | 0.3633 |
| attention | attention | 0.3562 | 0.3783 |

## Interpretation and next decision

The original Phikon mean-feature linear baseline remains higher on the primary descriptive validation average (0.3868). Neural capacity has not produced an improvement in this screen. Normalization and optimization differ from the original round, so this is not a pure capacity ablation.

For Phikon, the matched field-classification and early-mean controls have very similar averages; the present results do not support a strong claim that early averaging is the dominant problem. The ResNet controls behave differently, which motivates encoder sensitivity rather than a universal assertion.

The neural fitting losses approach zero while validation scores vary widely by trial. This is consistent with overfitting or limited case diversity; it does not by itself prove the cause. Check split/training-seed stability and lower-capacity controls before investing in a complex matching mechanism.

### Rare-subtype recall in the neural controls

Recall below counts repeated validation predictions across subsets/trials; distinct-case support is only two phyllodes cases and three papillary cases in the seed-812 pilot.

| Field pooling / view fusion | Phyllodes recall | Papillary recall |
| --- | ---: | ---: |
| attention / attention | 0.000 | 0.187 |
| attention / gated | 0.000 | 0.147 |
| attention / mean | 0.000 | 0.160 |
| mean / attention | 0.000 | 0.160 |
| mean / gated | 0.000 | 0.173 |
| mean / mean | 0.000 | 0.133 |

Source revision for this round: `1b2a0dd917495c4d2e70e2bd3fc805f43141a087`. See [the protocol](FUSION_PILOT.md), [initial error analysis](ERROR_ANALYSIS.md), and [contribution gates](RESEARCH_DIRECTION.md).

Reproduce with `python -m magset.round2_report --initial-runs /path/to/validation-v1 --field-runs /path/to/field-linear-v1 --fusion-runs /path/to/fusion-v1 --manifest /path/to/manifest.csv --protocol /path/to/protocol.json --output docs/FUSION_RESULTS.md`.
