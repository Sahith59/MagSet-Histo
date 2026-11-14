# BreakHis public benchmark context

Checked November 12, 2026. These are author-reported public results, not our reproductions or an audited universal leaderboard. The project remains eight-class BreakHis; no new training is launched by this comparison.

| Work | Reported setting | Reported result | Compatibility with our score |
| --- | --- | --- | --- |
| Alhaj/Öztürk, graph transformer, 2026 | Eight classes, patient-disjoint folds, case aggregation, evaluated by magnification | Case accuracy71.5–76.3%; macroF1.508–.598. Their patient-disjoint tile score is58.6% accuracy | Closest newly found task context; splits, aggregation, class prevalence and missing-subset endpoint still differ |
| Boudissa et al., cross-architecture distillation, IEEE Access 2026 | Eight classes; authors report a patient-wise 60/20/20 train/validation/test protocol | 96.87% accuracy for ResNet152→ViT-Large; their patient-wise ResNet50 reference is 86.50% | Different training, split and scoring protocol; the reported patient-wise split must not be confused with our case/subset endpoint |
| MagViT, 2026 author manuscript | Binary benign/malignant task; patient-level cross-validation and multi-view fusion | 96.43% patient accuracy reported in summary table | Two classes, not eight; no direct eight-subtype target |

Sources: [graph-transformer author repository](https://github.com/zaiedhussein/breakhis-graph-transformer), pinned at `09371ccff9860cb4ef9cb1a6625f6fd269e8da64`; [graph-paper DOI](https://doi.org/10.1038/s41598-026-65826-5); [distillation author manuscript, experimental setup and Tables4/5](https://pmc.ncbi.nlm.nih.gov/articles/PMC13155459/), [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/42111280/); [MagViT full author manuscript](https://arxiv.org/html/2608.16959v1).

The graph publisher page/PDF could not be retrieved in this check; its author repository supplies the numbers and publication citation. It is primary author evidence, not an independent reproduction. The distillation manuscript was retrieved through search despite an intermittent PMC browser challenge. Code for that exact 2026 pathology paper has not been confirmed; similarly named cross-architecture-KD repositories can refer to different papers.

MagViT also needs an arithmetic audit before using it as a reproduction target: its Table V patient fold values average about 92.86%, whereas Table IV reports 96.43%. The manuscript does not resolve this discrepancy in the inspected text. Retain both source claims; do not silently choose the higher value.

## Why the numbers differ

Our validation sets contain exactly one case from each subtype. Accuracy therefore equals balanced accuracy within each run; every subtype has equal influence. A population-weighted score can put more weight on common subtypes. The graph work's much lower macroF1 than raw accuracy illustrates why both metrics matter.

We score a provider case after combining its fields, and average over15 available-view combinations. Other papers may score individual images, one magnification, full inputs, binary labels or a different patient split. Even our all-view accuracy remains weak: the fixed adaptation gives47.5%/40.0% on the two arrangements. Missing views alone do not explain the gap.

Patient-disjoint eight-class public reports are higher than ours. We should not explain away our performance by asserting that all stronger studies leak data. Conversely, their reported numbers cannot establish an exact performance gap until their grouping, splits, labels, scoring, selection and implementation are reproduced compatibly.

## Compatibility audit and current comparison

The [pinned code/CSV audit](BREAKHIS_REFERENCE_AUDIT.md) is complete without executing author code. Released multiclass files have overlapping validation membership, a bare-ID aggregation ambiguity and cross-view role conflicts in one fold. This does not establish the historical execution path or invalidate the publication; it prevents copying those splits into an exact joint four-view reproduction.

Preserve our existing roles. A [standard supervised ResNet-50 pipeline](BREAKHIS_SUPERVISED_SPEC.md) completed as a task/role-matched reference; [the checked result](BREAKHIS_SUPERVISED_RESULTS.md) failed its improvement gate. It is not an exact graph-paper reproduction or identical-budget fusion ablation. Keep author numbers and our implementation results separate. No exact matching eight-class15-subset public leaderboard was established; that is a search limitation, not proof that none exists.

The [completed adaptation result](BREAKHIS_ADAPTATION_RESULTS.md) fails its primary improvement gate; the [diagnosis](BREAKHIS_DIAGNOSIS_RESULTS.md) found a large fitting/validation gap. The supervised comparison also failed; keep the frozen reference and preserve every negative outcome.
