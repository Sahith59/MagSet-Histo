# MAGSET-Histo in one page

| Question | Answer |
| --- | --- |
| Problem | Identify breast tumor subtype using the microscope magnifications available for an unseen patient, even when some views are missing. |
| Dataset | BreakHis: the standard release describes 7,909 images, 82 patients, four magnifications, and eight subtypes. Actual downloaded contents will be audited. |
| Inputs | Sets of images at 40X, 100X, 200X, and 400X, plus an availability mask. No fabricated exact field pairing. |
| Output | Eight subtype scores and one subtype prediction for the defined patient/slide group. Image-level predictions form a separate benchmark track. |
| Encoder | First candidate: pretrained Phikon-v2 ViT-L; compare UNI2-h if access is available and an ImageNet encoder. Freeze features first, then test controlled adapter/fine-tuning. |
| Architecture | Encode images, aggregate within each magnification, fuse the observed scale representations, classify. A mask excludes absent views. |
| Candidate mechanism | Availability-conditioned partial evidence transport: match comparable evidence across observed scale sets while preserving unmatched scale-specific evidence. This is a hypothesis; OT, missing-view masking, and residuals already have prior art. |
| Split | Group by audited patient identifiers. Provisional primary protocol: group-disjoint outer evaluation, class-covered inner validation where feasible, and repeated training seeds. Exact protocol freezes after identifier reconciliation. |
| Primary result | Mean eight-class patient macro F1 across all 15 nonempty magnification subsets. Also full-view and worst-subset F1, class recall, calibration, and uncertainty. |
| Current results | Only historical 40X notebook outputs: 93.73% and 94.99% accuracy. No new model results. |
| Historical base paper | Lang et al., SPIE Medical Imaging 2021, DOI 10.1117/12.2582299, with Dong Hye Ye as coauthor. |
| Closest comparison | MagViT (2026), multi-magnification BreakHis with missing-scale masking; its reported source task is binary. |
| Mechanism comparison | M2OTCA (2026): cross-magnification optimal transport for WSI classification. Must compare and establish a substantive distinction. |
| Target paper | Our future manuscript, working title: Breast tumor subtype classification from incomplete magnification sets. |
| Working venue | MICCAI 2027 is an ambitious provisional target. Its official deadline is not yet announced as of November 11, 2026. Faster workshop calls will be checked separately. |

The mechanism in plain words: each microscope zoom sees a different collection of clues. We will test whether matching clues that can be compared, retaining clues unique to a zoom, and explicitly handling absent zooms improves prediction. Matching features is not proof that two images show the exact same physical cells.

A large cluster accelerates feature extraction, repeated experiments, and larger-data validation. It does not add patients to BreakHis. A second suitable WSI dataset is a priority for a strong general-method paper.
