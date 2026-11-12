# Research direction and evidence gates

Updated November 12, 2026. The implemented neural models are baselines. The candidate partial-matching method remains an untested hypothesis.

## What current evidence supports

The initial verified validation pilot shows subtype failures, especially for phyllodes and papillary tumors, and individual predictions can deteriorate when an extra magnification is provided. Only two or three distinct validation cases support several rare-subtype observations. See [the computed analysis](ERROR_ANALYSIS.md). These facts motivate testing field preservation and selective fusion. They do not demonstrate that optimal transport is necessary or that the candidate is novel.

The current round isolates field aggregation using normalization-matched linear controls and compares six learned field/view pooling combinations under a fixed validation budget. A final epoch is fixed in advance; test outcomes cannot select it. The code, selection protocol, and checks are in [the registered pilot](FUSION_PILOT.md).

## Closest mechanisms: inspected evidence and open questions

| Primary source | Confirmed overlap | Consequence for our claim |
| --- | --- | --- |
| [MagViT, full author manuscript](https://arxiv.org/html/2608.16959v1), §§4.3–4.8 | Binary BreakHis; shared ViT, scale-gated weighted sum with missing-scale masking. The gate uses concatenated scale representations; branches include normalization/residual fusion. | A masked gate and four magnifications are established components. Our contextual gate is an adapted control, not a reproduction of their exact architecture. Their field grouping and model-selection protocol still require code-level verification before comparing headline scores. |
| [M2OTCA, publisher abstract/introduction](https://www.sciencedirect.com/science/article/pii/S1361841526001519) | Condenses fields into structural prototypes, aligns distributions at 20× versus 10×/5× using OT, and uses gradient-driven cross-attention and magnification-specific predictions. | Prototype compression plus cross-scale OT is occupied territory. The full equations, supplement, and official implementation have not been obtained; do not assert that its transport is balanced or that it lacks residual information. |
| [Attention MIL, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html) | Learned permutation-invariant bag aggregation. | Gated field attention is a baseline. |
| [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html) | Attention between set elements followed by invariant aggregation. | Our one-block masked self-attention is a generic control, not a full reproduction of the published Set Transformer. |
| [Intent-controlled partial OT, 2026](https://arxiv.org/abs/2605.20030) | Pointwise rejection costs can depend on reliability or other side information. | Availability-conditioned rejection costs are not automatically a new mathematical contribution. Retaining unmatched evidence needs its own task-specific argument and ablation. |

This targeted audit rules out several weak novelty claims. It does not certify that the remaining idea is unique. A broader related-work search, including multimodal residual/shared-specific representation learning and incomplete-view MIL, remains necessary.

## Candidate question, with a falsifiable outcome

Question: for unpaired field sets under changing available magnifications, does routing unmatched evidence explicitly improve subtype classification beyond a capacity-matched attention model with the same residual path?

Required comparisons: no matching; ordinary cross-attention; a clearly specified existing OT adaptation; partial matching; partial matching plus unmatched evidence; and a residual-only model with equivalent parameter count. Keep encoder, field inputs, cases, seeds, and selection opportunities fixed. Check one-view behavior and report all 15 subsets. Attention/transport weights are model associations, not physical registration or biological explanations.

Continue only if the effect survives different inner splits/training seeds and helps more than one favorable trial. A small score increase on eight selected validation cases is insufficient. If generic attention or a residual-only model explains the gains, narrow or replace the mechanism. Do not repeatedly examine the outer test while making that decision.

## Second-dataset feasibility

[The official continuous-magnification repository](https://github.com/bifold-pathomics/continuous-magnification-sampling) now includes evaluation/preparation code and links to [TCGA-MS](https://huggingface.co/datasets/bifold-pathomics/TCGA-MS). The public API confirms ungated access to a 6,579,294,465-byte archive at a pinned revision; its LFS checksum is recorded in `configs/tcga-ms-source.json`. Initial work downloads and inventories the archive only. Labels, grouping, image correspondence, pretraining overlap, and evaluation eligibility require inspection before training.

[Mind the Gap](https://arxiv.org/html/2601.02198v1) describes TCGA-MS as 16 morphological classes and seven resolutions, and BRACS-MS as six lesion classes derived from BRACS ROIs. These differ from BreakHis's eight tumor subtypes and four acquisitions. A second head and native data contract are required. Whole-slide/ROI identifiers must keep related patches and scales together; do not assume every ROI in one slide has one label.

[BRACS details](https://www.bracs.icar.cnr.it/details/) describe seven original categories and provide patient/split metadata. Its [download page](https://www.bracs.icar.cnr.it/download/) currently requires an account, and its [rules](https://www.bracs.icar.cnr.it/rules/) require accurate registration and citation. No registration, author contact, or terms acceptance has been performed. TCGA-MS offers an immediately inspectable secondary route while BRACS access remains an optional later step.

The TCGA-MS download, checksum/metadata audit, and full image decoding are complete; see [the audit](TCGA_MS_AUDIT.md). After excluding 19 `._` sidecar-named entries, all 65,312 images passed RGB decoding at 224×224 with zero errors. They cover 128 labeled slides/patients; 122 have all seven resolutions. The count differs from the paper, some fold roles omit classes, and filename coordinates do not establish seven-scale pairing. The native evaluation protocol and pretraining-overlap audit remain prerequisites to training.

The [fair-comparison contract](COMPARISON_PLAN.md) distinguishes measured internal controls from unfinished published-method adaptations. The next small round aligns actual linear/neural normalization and tests capacity or dropout individually. Its scores must not be selected using outer-test outcomes.

## Remaining paper gates

1. Strong baseline and split/training-seed stability evidence.
2. A complete close-mechanism comparison and defensible contribution.
3. Identifier sensitivity and a frozen final evaluation protocol.
4. Independent outer evaluation, calibration/intervals, and a compatible second dataset.
5. Professor Dong's review of claims, authorship, and venue.
