# Papers, benchmark tracks, and the novelty audit

Verified or inspected November 11, 2026. This is a targeted starting list, not a completed systematic review. Numbers below are author-reported and are not directly comparable across tasks or protocols.

## What base paper and target paper mean here

- Historical base/reference: Lang et al., **Breast cancer magnification-independent multi-class histopathology classification using dual-step model**, SPIE Medical Imaging 2021. [DOI](https://doi.org/10.1117/12.2582299). Dong Hye Ye is a coauthor. The supplied manuscript appears to be a version of this published work. It is prior work from named authors, not a publication authored by the current project simply because it is in the folder. Author-primary corroboration: [Saxena CV](https://devsaxena.info/wp-content/uploads/2024/12/saxena_cv_oct_24.pdf), [Ye institutional CV](https://www.marquette.edu/electrical-computer-engineering/directory/documents/dong-h-ye-cv.pdf). The publisher URL resolves but automated full-text retrieval was blocked; the local manuscript was inspected directly.
- Closest task comparator: **MagViT** (2026), [paper](https://arxiv.org/abs/2608.16959). It uses four BreakHis magnifications with missing-scale masks and patient-level evaluation for a binary source task. It reports 91.91% mean image accuracy, 96.43% patient accuracy, and 0.9042 macro F1. Its model-selection and grouping details must be audited before reproducing; no repository link was confirmed in the inspected article body. These are not eight-class missing-subset benchmark targets.
- Close mechanism comparator: **M2OTCA: Multiple-magnification optimal transport-based cross-attention learning for whole slide image classification** (Medical Image Analysis, 2026), [publisher](https://www.sciencedirect.com/science/article/pii/S1361841526001519), DOI 10.1016/j.media.2026.104082. It matches cross-magnification prototypes using OT and adds gradient-based cross-attention for WSI classification. Cross-magnification OT alone cannot be claimed as our invention. Full methods/supplement/code audit remains pending.
- Target manuscript: our future paper, working title **Breast tumor subtype classification from incomplete magnification sets**. The working model name MAGSET-Histo does not make it novel. The final title/mechanism follows evidence.
- Target venue: provisional MICCAI 2027, subject to Professor Dong's preference and results. A target venue is different from a paper we compare against.

## Other necessary references

| Work | Why relevant | What can and cannot be compared |
| --- | --- | --- |
| [Original BreakHis dataset paper](https://www.inf.ufpr.br/lesoliveira/download/TBME-00608-2015-R2-preprint.pdf) and [provider](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-BreakHis/) | Acquisition, filename metadata, patient-separated protocol | Reproduce its exact protocol in a separate track; verify counts from the obtained archive |
| [Patient-aware CNN/transformer benchmark, 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1752938/full) | Binary BreakHis comparison with patient-aware folds | Reports roughly 92.3–92.4% mean accuracy for its strongest CNNs in 10-fold analysis; not an eight-class multi-view score |
| [CrossViT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.html) | Attention across multiscale branches | General architecture prior art, not a matching BreakHis evaluation |
| [HAG-MIL, IJCAI 2023](https://ijcai.org/proceedings/2023/176) | Multiresolution pathology transformer/MIL | WSI task and hierarchy differ from unpaired BreakHis field sets |
| [mmFormer, MICCAI 2022](https://arxiv.org/abs/2206.02425) | Incomplete-input medical transformer | MRI segmentation; conceptual precedent, not comparable classification metrics |
| [MOTCat, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf) and [official code](https://github.com/Innse/MOTCat) | Unbalanced OT co-attention in pathology/genomics | OT is existing machinery; survival outcomes differ from tumor subtyping |
| [GAUGE, 2026](https://arxiv.org/abs/2608.05608) | Counterfactual gating for incomplete modalities | Contribution-aware gating is not an unoccupied novelty direction |
| [ShapKO, MICCAI 2026](https://papers.miccai.org/miccai-2026/0956-Paper3767.html) | Shapley-adaptive modality knockout | Adaptive dropout/importance training is established; inspect actual paper as well as reviews |
| [Mind the Gap, 2026](https://arxiv.org/abs/2601.02198) | Magnification robustness and TCGA-MS/BRACS-MS benchmarks | Continuous-scale robustness differs from a fixed four-view missingness task |
| [BRACS dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9575967/) | Breast WSI dataset and seven-category validation opportunity | Different labels; validate the mechanism with a separate head, not the identical eight-class classifier |

## Backbone candidates

First: [Phikon-v2 official model card](https://huggingface.co/owkin/phikon-v2), a pathology ViT-L with 1024-dimensional features. Stronger comparison if permitted/access available: [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and [official UNI repository](https://github.com/mahmoodlab/UNI). A larger model is a candidate to measure, not automatically the strongest model for this cohort. Preserve each model's input transform and fix its revision.

## Our benchmark contract

No directly interchangeable, universal leaderboard for **eight-class, patient-set, four-magnification classification across every missing subset** was established by this search. This does not prove none exists. We must verify closest code and protocols, then release an exact reproducible comparison rather than compare unrelated headline accuracies.

Track I: image-level subtype prediction with patient-disjoint splits, per magnification and pooled magnifications.

Track II: patient/slide-set subtype prediction with 15 explicit nonempty availability subsets, fixed eligible patients, and field budgets.

Use the same backbone, folds, seeds, and tuning budget for masked mean, averaging, gating, attention, closest-method adaptations, and the candidate. Report original author numbers separately from our reproductions. Never compare a binary five/ten-fold score directly with eight-class image-split outputs as evidence of superiority.

## Candidate novelty matrix

| Idea | Existing overlap | Required distinction/evidence |
| --- | --- | --- |
| Transformer plus several scales | CrossViT, HAG-MIL, MagViT | Insufficient alone |
| Missing masks/tokens and view dropout | MagViT, mmFormer, ShapKO | Insufficient alone |
| Contribution-aware gates | GAUGE and broader valuation methods | Insufficient alone |
| Cross-magnification OT matching | M2OTCA | Insufficient alone |
| Partial matching plus explicit preservation of unmatched evidence conditioned on observed scale sets | Strongly adjacent to OT/MIL and incomplete-input learning | Full prior-art audit; matched-capacity tests against M2OTCA-style fusion; show benefits specifically under unpaired fields and missingness |

The last row is a **candidate hypothesis**, not a certified novel mechanism. Eight-class extension and careful evaluation can be useful without being sufficient for a top-tier method paper. A compatible second dataset and substantive ablations are important.

## Venue dates

As of November 11, 2026:

- [MICCAI Society future conferences](https://miccai.org/upcoming-conferences/) lists MICCAI 2027 in Auckland, September 26–October 1, with paper deadline to be announced.
- [ISBI 2027 author instructions](https://biomedicalimaging.org/2027/papers/) list November 2, 2026 as the regular submission deadline, now past.
- SPIE Medical Imaging's 2027 author-guideline listing gives August 5, 2026 for abstracts and January 27, 2027 for manuscripts in that pipeline. Do not treat the later manuscript date as an open new abstract submission window. Recheck the current [SPIE author page](https://spie.org/conferences-and-exhibitions/medical-imaging/presenters/abstract-submission-guidelines) before using it for a submission decision.

No submission or communication to authors/conference staff has been made.
