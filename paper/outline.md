# Manuscript working outline

This is a writing scaffold, not a completed paper. The candidate mechanism and its novelty remain unverified; outer-test results are not yet available.

Working title: **Breast tumor subtype classification from incomplete magnification sets**.

## Research questions

1. How much do different observed magnification combinations change subtype classification on held-out cases?
2. Do learned interactions improve over strong frozen-encoder averaging/gating baselines under matched data and tuning budgets?
3. Does preserving unmatched evidence during cross-scale matching improve robustness when magnifications are withheld, beyond existing matching methods?

Question 3 is the candidate method hypothesis. Cross-scale transformers, missing-view masking, and optimal transport already have prior art. Full comparison to MagViT/M2OTCA and other close methods must establish any substantive distinction.

## Introduction

Describe the complementary tissue and cell detail visible across magnifications and define the incomplete-input task. State the prediction unit and intended research setting precisely. Motivate reliable performance across subsets rather than a single complete-view accuracy. Clinical prevalence of missing magnifications requires external evidence; simulated missingness alone does not establish it.

## Related work

Cover the historical SPIE study, multiscale pathology/MIL, missing-input fusion, matching/transport methods, and strong pathology encoders. Distinguish binary from eight-class tasks and provider-case protocols from independently verified patient identities. Use the primary references in `docs/RELATED_WORK.md`; do not copy unrelated headline scores into one leaderboard.

## Data and protocol

Describe the verified archive, all four magnifications, raw/provisional identifier relationships, provider trial assignments, and the separate grouping-sensitivity question. Explain that all cases currently have all four scales and the initial missingness experiment withholds observed data artificially. Keep preprocessing and feature/model versions reproducible.

## Methods

The implemented frozen-feature, field, and learned fusion controls are drafted in [methods-draft.md](methods-draft.md), with protocols in `docs/BENCHMARK_PROTOCOL.md` and `docs/FUSION_PILOT.md`. The candidate matching mechanism remains unimplemented. Separate common method components from the hypothesized contribution.

## Experiments and results

Planned tables: dataset inventory; baseline comparisons; all 15 subset outcomes; controlled component ablations; grouping sensitivity; second-dataset evaluation; compute/calibration/failure analysis. Validation pilots establish the pipeline and guide hypotheses. They are not substitutes for frozen final outer evaluation.

All result tables and figures must be generated from saved predictions and provenance. Do not insert guessed performance, significance, or paper-acceptance claims.

## Discussion and limitations

Address small independent case counts, rare subtypes, identifier uncertainty, simulated missingness, crop/field coverage, pretraining provenance, external validity, and any negative results. Image-level success is not evidence of improved treatment or survival.

## Reproducibility and submission

Record archive/model/code/split revisions, seeds, environment, GPU/CPU usage, retention policy, data access conditions, and result-generation commands. Confirm contribution, authorship, and venue with Professor Dong. Submit only after explicit instruction and review of a completed manuscript.
