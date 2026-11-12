# TCGA target-label lineage audit

Audited November 12, 2026. This checks how numerical targets enter the classifier. It does not independently validate clinical diagnoses.

## Confirmed

The pinned [author loader](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/dataloaders.py#L202) reads `label_id` for each slide. Our sorted canonical remapping preserves exactly those targets. The other numeric column, `label`, is one-to-one with `label_id` in this release; neither is inferred from the histology-description text.

The verified CSV contains 128 supplied case IDs, 128 slide IDs, and 16 target groups with eight cases each. Six groups have multiple description strings; two cases have blank description text. Our reader does not drop or relabel them. The [aggregate audit](evidence/TCGA_LABEL_LINEAGE.json) records every group's observed descriptions, ICD-O strings, tissue groups, and source-fold class coverage without publishing case identifiers.

## What the paper explains, and what remains uncertain

The authors describe pathologist-assisted curation of morphological categories. Appendix D.2 names the intended classes, while the release retains broader descriptions for some entries. This variation alone does not establish an incorrect target. The paper gives no numeric source-ID/name crosswalk in that table. [Author manuscript, §§3.2.1 and D.2](https://arxiv.org/html/2601.02198v1#S3.SS2.SSS1).

For example, source group 32 contains generic adenocarcinoma text; the paper includes tubular adenocarcinoma in its class list. Equating these solely by elimination would be an inference, so we have not renamed the group. We also have not independently verified patient pathology records or the curation process.

The inspected official tree contains the label-consuming loader and evaluation pipeline, but no TCGA label-creation script or numeric-to-clinical-name dictionary was identified. This is a limitation of the inspected artifacts, not proof that no such records exist elsewhere. No author contact has been made.

## Effect on the project

The completed pilot's target-column choice is confirmed. Its numerical results remain the registered source-label experiment. Clinical class names and label quality require careful wording. The findings do not justify changing labels, moving patients, expanding a search, or examining the reserved group after development scores.

The source paper's patch-level benchmark and our case-level missing-view task remain different. [The comparison audit](METHOD_COMPARISON_AUDIT.md) records which changes make an implementation an adaptation.

## Reproduce

`scripts/audit_tcga_label_lineage.py` checks the pinned CSV checksum, binds the read-only author loader to its recorded checksum, inspects its target mapping with Python's syntax parser, and summarizes the metadata. It never imports/runs author code or fits a classifier. Local source copies and their download record remain private; the public result contains aggregates and source hashes.

Author commit: `83f05d8eed96e805b737b9ea935efa1c4bedb8a1`. Label CSV SHA256: `2d97bbfec20d69edb7fcaf15c2a8068d6d4fe2e2f6161884a07a2ff3fd12ace1`.
