# Split and training seed sensitivity

All 45 prespecified follow-up configurations completed and their validation probabilities independently reproduce the metrics. No outer-test scoring occurred. Each cell averages equal-subset macro F1 over five overlapping provider trials. Inner-validation case counts remain eight per trial.

| Phikon neural control | Split 812 / train 812 | Split 813 / train 812 | Split 812 / train 813 |
| --- | ---: | ---: | ---: |
| field-mean-view-mean | 0.3268 | 0.2555 | 0.3117 |
| field-mean-view-gated | 0.3349 | 0.2678 | — |
| field-mean-view-attention | 0.3387 | 0.2689 | — |
| field-attention-view-mean | 0.3474 | 0.2801 | — |
| field-attention-view-gated | 0.3347 | 0.2617 | — |
| field-attention-view-attention | 0.3562 | 0.2468 | 0.3500 |

The original Phikon mean-feature linear control changes from 0.3868 (split seed 812) to 0.3073 (split seed 813), with its fixed four-C selection grid. No training seed is relevant to this deterministic fitting control.

## Decision

Changing the inner-validation cases changes both the absolute scores and the strongest neural row. For the two repeated neural controls, the measured change under a new training seed is smaller than the change under a new inner split in this screen. Two seeds do not establish a general variance decomposition.

The linear control remains higher than the neural rows on the primary average for both checked inner splits. These observations support prioritizing case scarcity, regularization/capacity, grouping sensitivity, and a compatible second dataset. They do not support claiming that learned attention or the unimplemented matching mechanism improves robustness.

Do not choose the favorable split for the paper. Preserve all rows, freeze a final protocol, and account for repeated cases in later uncertainty analyses. The new split/training-seed runs used source `ae647d02426d2119e29865f199cbe2031198bd8a`; model/optimizer settings were unchanged.

Reproduce with `python -m magset.sensitivity_report --results-root /path/to/runs --manifest /path/to/manifest.csv --protocol /path/to/protocol.json --output docs/SENSITIVITY_RESULTS.md`.
