# Normalization-matched linear and capacity controls

All 50 new configurations completed and their saved probabilities reproduce reported metrics. The comparison also references 20 previously completed neural configurations. Input caches, case assignments, and the actual saved normalization arrays match for every linear/neural comparison. Each setting receives four validation-selected candidates; no outer-test outcome is used.

| Policy / method | Split seed | Parameters (neural) | Mean subset macro F1 | Full-view macro F1 |
| --- | ---: | ---: | ---: | ---: |
| matched-linear / mean | 812 | — | 0.3843 | 0.4083 |
| original / field-mean-view-mean | 812 | 133000 | 0.3268 | 0.3433 |
| original / field-attention-view-attention | 812 | 298632 | 0.3562 | 0.3783 |
| small / field-mean-view-mean | 812 | 33256 | 0.3308 | 0.3592 |
| small / field-attention-view-attention | 812 | 43944 | 0.3274 | 0.3842 |
| dropout / field-mean-view-mean | 812 | 133000 | 0.3263 | 0.3433 |
| dropout / field-attention-view-attention | 812 | 298632 | 0.3636 | 0.3825 |
| matched-linear / mean | 813 | — | 0.3169 | 0.3083 |
| original / field-mean-view-mean | 813 | 133000 | 0.2555 | 0.2583 |
| original / field-attention-view-attention | 813 | 298632 | 0.2468 | 0.2167 |
| small / field-mean-view-mean | 813 | 33256 | 0.2744 | 0.2667 |
| small / field-attention-view-attention | 813 | 43944 | 0.2679 | 0.2458 |
| dropout / field-mean-view-mean | 813 | 133000 | 0.2566 | 0.2708 |
| dropout / field-attention-view-attention | 813 | 298632 | 0.2608 | 0.2500 |

The small policy changes only hidden width 128→32. The dropout policy changes only dropout 0.2→0.5. Other optimization/input settings remain fixed. Neural parameter counts differ intentionally; a proposed-mechanism comparison will additionally require matched-capacity controls.

These remain noisy exploratory results: eight validation cases per trial, overlapping provider holdouts, and selected hyperparameters. A numerical lead does not establish generalization or novelty.

## What this round establishes

| Inner split seed | Matched linear mean | Highest tested neural average |
| --- | ---: | ---: |
| 812 | 0.3843 | 0.3636 |
| 813 | 0.3169 | 0.2744 |

The matched linear mean remains higher than the tested neural controls on both inner splits. Reducing hidden width or increasing dropout did not close that descriptive gap. The normalization difference between earlier rounds no longer explains this comparison.

Retain these results as an exploratory baseline reference. The next research work is to settle native second-dataset eligibility/metrics and pretraining overlap, complete compatible published-method comparisons, and specify a defensible candidate mechanism. More architecture trials alone do not establish a contribution. No outer-test evaluation or second-dataset model result is reported here.

Experimental source revision: `ed8db0f49294cb3a19b4ee4e45f8e1a4c6064b08`. Regenerate using `python -m magset.capacity_report --runs-root /path/to/runs --output /path/to/verified-comparison.json --markdown docs/CAPACITY_RESULTS.md`.
