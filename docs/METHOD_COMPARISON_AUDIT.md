# Closest-method implementation audit

Checked November 12, 2026. This is a targeted audit of known primary sources, not an exhaustive systematic review or a novelty certificate. No published method has been newly trained or numerically reproduced in this step.

## What can be compared?

| Source | Confirmed mechanism and training | Reproduction status and consequence |
| --- | --- | --- |
| [MagViT author manuscript, §§3–5](https://arxiv.org/html/2608.16959v1) | Binary BreakHis; shared ViT-B/16 CLS features, concatenation-conditioned gate, masked softmax, weighted fusion, and patient voting. Stated defaults: 15 epochs, AdamW, learning rate 0.00005, batch 32, weight decay 0.01, warmup 0.1, weighted cross-entropy with smoothing. | No author-linked implementation was verified. Exact fold memberships, field-group construction, gate details and selection artifacts remain unresolved. Our unpaired field bags, class count, encoders and metric differ. A controlled gate inspired by this design must be labeled an **adaptation**, not a reproduction. |
| [M2OTCA author abstract](https://pubmed.ncbi.nlm.nih.gov/42001680/) | Published Medical Image Analysis article. Structural prototypes reduce cross-magnification OT cost; gradient-derived attention connects 20× features with 10×/5× features, followed by per-scale predictions and final fusion. | Full methods/supplement were not retrieved: publisher access returned 403. No official implementation was verified in the targeted search. OT type, solver, prototype budget, losses and missing-view behavior are unresolved. Do not invent these choices or call our ordinary affinity contrast M2OTCA. |
| [Mind the Gap author manuscript, Appendix D](https://arxiv.org/html/2601.02198v1#A4) | Frozen embeddings; patch-level classifiers trained across four standard resolutions and evaluated separately at seven resolutions. The stated feature rule combines CLS and mean patch tokens; model selection uses balanced accuracy, with logistic and cosine-neighbor grids. | This profiles encoders across scale. It does not evaluate our case-level 15-subset fusion endpoint. Different targets of evaluation prevent ranking its headline scores against our pilot. |

## Official code inspected

Pinned author repository commit: `83f05d8eed96e805b737b9ea935efa1c4bedb8a1`.

- [Dataset reader](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/dataloaders.py): source role columns select slide sets; `label_id` supplies targets; sidecar-named PNG entries are excluded.
- [Evaluation entry point](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/evaluate_benchmarks.py): patch rows from training resolutions are concatenated; logistic C has nine candidates and cosine kNN has seven. Selection uses development balanced accuracy. Source-fold test predictions are produced by the normal evaluation path. **Do not run that path across our reserved patients.**
- [Token wrapper](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/model_loading/load_models.py): embedding construction depends on model/token mode. Its source feature rule cannot be assumed equivalent to our cached ResNet GAP or Phikon CLS.

These are source inspections, not execution evidence. Third-party code was downloaded privately and checked against Git blob hashes; it was not executed or copied into public project source.

## Numerical/protocol reconciliation before any headline comparison

MagViT's §3.1 total is 9,109 while its benign/malignant counts sum to 7,909. Arithmetic means of Table V's five rows are image accuracy 0.92606, patient accuracy 0.92856, F1 0.91278; Table IV Branch 1 lists 0.9191/0.9643/0.9042. These differences need artifacts or author clarification. They are not evidence of misconduct. We do not treat the headline as an interchangeable target. [Tables IV–V](https://arxiv.org/html/2608.16959v1#S5.SS1).

Our obtained TCGA release has 65,312 valid images rather than the stated 67,200. Native per-case/per-view counts do not always reach 75. The inspected reader uses released files without fresh patch subsampling; archive preparation may already impose a cap. Our source-fold metadata also omits some classes from some roles. These are obtained-release facts retained in the data audits, not an assertion that the intended benchmark cannot work.

## Comparison contract for the next experiment

1. Preserve the declared case-level task and supervised quarantine. Reusing the official rotating folds would put reserved patients into fitting elsewhere.
2. Fix the same encoder, source fields, normalization, roles, available-view combinations and fitting budget for every arm.
3. Include mean and equally sized controls. A gate, shared/specific branch, missing mask or OT component is existing machinery until a specific distinction is established.
4. State every adaptation: extra field aggregation, output class changes, encoder freezing/adaptation, masked-input training and metric policy.
5. Do not implement an alleged exact M2OTCA reproduction from its abstract. Keep that comparison unresolved until the necessary method specification is obtained.

The current agreement-specific claim failed its declared gate. A next method needs an explicit hypothesis and control that excludes a simpler explanation. The [view-addition audit](VIEW_ADDITION_AUDIT.md) supplies a descriptive failure pattern; it does not establish a novel fix.

## Search record

Search window: November 12,2026. Starting sources were the three known papers, author-linked GitHub and Hugging Face releases, and the primary publisher/PubMed record. Queries included exact titles/IDs plus `M2OTCA github code Zhonghang`, `2608.16959 github code MagViT`, and GitHub repository searches `MagViT histopathology` and `M2OTCA` (both returned zero repositories in that search). No exhaustive search count is claimed.

Raw queries, retrieved source revisions and file checksums are retained in the private literature-audit directory. Main exclusions were unrelated tasks as numerical comparators, unverified third-party implementations, and search snippets as methodological evidence. Remaining limitation: no full M2OTCA method/supplement or official MagViT/M2OTCA code was verified. Broader prior-art screening remains open.
