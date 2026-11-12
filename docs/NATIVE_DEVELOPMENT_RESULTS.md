# Native four-fit development results

Measured November 12, 2026. Secondary 16-class multi-organ task; not eight-class breast validation, a published-method reproduction, or final test performance.

Exactly four fits used the registered 80 fit /16 development /32 reserved patient protocol. The 32 reserved patients were excluded from classifier construction and prediction. Each arm has 240 development records (16 patients ×15 view subsets), representing 16 independent patients.

| Arm | Mean subset macro F1 | Full-view F1 | Worst-subset F1 | Mean Brier ↓ | Coefficients | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean + mask | 0.334028 | 0.343750 | 0.208333 | 0.791930 | 32848 | 2.155 |
| Mean + duplicate mean | 0.339583 | 0.343750 | 0.208333 | 0.791727 | 33360 | 1.086 |
| Mean + random contrast | 0.334028 | 0.343750 | 0.208333 | 0.791867 | 33360 | 0.960 |
| Mean + affinity contrast | 0.338194 | 0.343750 | 0.208333 | 0.791925 | 33360 | 0.904 |

## Prespecified decision

The affinity arm does not exceed all three controls on the prespecified development endpoint. Stop the agreement-specific improvement claim for this pilot; do not expand a score-driven search or inspect sealed outcomes.

Compared with mean-only predictions, duplicate/random/affinity change 3/0/1 of240 decisions, covering 3/0/1 distinct development patients respectively. These are descriptive counts, not independent replications.

All models converged before development scoring. Fixed ImageNet ResNet features, fitting-only normalization, 32D projection seed812, temperature0.2, logistic C0.1/L2/LBFGS, and inverse-subset weights were shared as specified. No setting was selected from these scores. Patient training weight totals80 and5 per class.

## All availability subsets

The same16 development patients appear in every row. Rows and class outcomes are dependent; no significance test or naive row-bootstrap confidence interval is claimed.

| Available resolutions (microns/pixel) | Mean F1 | Duplicate F1 | Random F1 | Affinity F1 |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | 0.239583 | 0.239583 | 0.239583 | 0.239583 |
| 0.5 | 0.229167 | 0.229167 | 0.229167 | 0.229167 |
| 1.0 | 0.364583 | 0.364583 | 0.364583 | 0.364583 |
| 2.0 | 0.489583 | 0.489583 | 0.489583 | 0.489583 |
| 0.25+0.5 | 0.208333 | 0.208333 | 0.208333 | 0.208333 |
| 0.25+1.0 | 0.343750 | 0.343750 | 0.343750 | 0.343750 |
| 0.25+2.0 | 0.385417 | 0.385417 | 0.385417 | 0.385417 |
| 0.5+1.0 | 0.302083 | 0.364583 | 0.302083 | 0.364583 |
| 0.5+2.0 | 0.343750 | 0.343750 | 0.343750 | 0.343750 |
| 1.0+2.0 | 0.479167 | 0.479167 | 0.479167 | 0.479167 |
| 0.25+0.5+1.0 | 0.333333 | 0.354167 | 0.333333 | 0.333333 |
| 0.25+0.5+2.0 | 0.270833 | 0.270833 | 0.270833 | 0.270833 |
| 0.25+1.0+2.0 | 0.343750 | 0.343750 | 0.343750 | 0.343750 |
| 0.5+1.0+2.0 | 0.333333 | 0.333333 | 0.333333 | 0.333333 |
| 0.25+0.5+1.0+2.0 | 0.343750 | 0.343750 | 0.343750 | 0.343750 |

## Compute, artifacts, and limits

All field features were cached before this pilot. Shared feature construction took 1.616s for fitting patients and 0.319s for development patients. Added-feature arms have the same classifier coefficient count, but equal dimensions do not establish identical effective capacity. Saved block RMS diagnostics are unweighted over expanded rows; they were not used to rescale or select models.

The collector independently checked all960 prediction records against trusted protocol roles, labels, exact patient/subset coverage, canonical16-class probability columns, and recomputed metrics. Artifact hashes and reload checks passed. Full per-class recall/confusion matrices, Brier values, and per-patient predictions are retained privately. The public ledger contains aggregates and hashes, not patient identifiers.

Only one development patient per class supports this screen. Six naturally incomplete patients are fit-only. Frozen ImageNet ResNet is the initial provenance control; Phikon has known TCGA-source pretraining. The wider project still needs compatible prior-work comparisons, independent breast evidence, and a defensible contribution before submission.

Experimental source: `c1364c6202c2a5de7f9111b927ef0f41c94e0664`. SLURM job: `4958291`. Full private report SHA256: `7972dae0ab004fb98c52dc56747a778c627ba81a074c497ed8356aab29377612`.

[Registered specification](NATIVE_DEVELOPMENT_PILOT_SPEC.md) · [Plain-language project guide](PROJECT_EXPLAINED.md) · [Native protocol](TCGA_QUARANTINE_PROTOCOL.md)
