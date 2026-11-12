# Registered TCGA supervised quarantine

Registered November 12, 2026, before any TCGA classifier fitting. This is a new secondary multi-organ task, not an exact reproduction of the source paper or external validation of the eight-class breast classifier.

## Fixed data and roles

Use the pinned archive and audited label/inventory hashes in `configs/tcga-ms-source.json`. Four native views are 0.25, 0.5, 1.0, and 2.0 microns per pixel. They provide 37,024 images from 128 supplied patient IDs and 16 source classes. Canonical labels follow the sorted source IDs; labels are not inferred from free-text descriptions.

| Role | Patients | Patients per class | Permitted use |
| --- | ---: | ---: | --- |
| Fit | 80 | 5 | Fit classifier/normalization/adaptation |
| Development | 16 | 1 | Small prespecified model selection only |
| Sealed test | 32 | 2 | One final evaluation after method/protocol freeze |

For each class, order complete patients by SHA256 of `magset-tcga-quarantine-v1-20261112|patient_id`. Reserve the first two for sealed test, the next for development, and fit the remaining five. The salt is fixed, not searched against scores. All six patients lacking 0.25 mpp remain in fit; they must not later be presented as independent naturally incomplete test cases. Every development/test patient has all four views, supporting the same 15-subset population.

Native manifest SHA256: `a668cf85151f29a163628210851c007faaf8c9679b7c9f5eaadc8c031f06e27b`.

Registered protocol SHA256: `f1baf1ac21f5f7e117768951619dc6f94a96ba63c72ad810f31fc103fe100c69`.

The source five-fold role columns are retained as provenance but are not used to train additional folds across these reserved patients. Doing so would defeat this global quarantine. This is a procedural supervised-data restriction, not a claim that public labels are encrypted or inaccessible.

## Encoder and metric policy

Start with frozen ImageNet-pretrained ResNet-50 as a provenance control. Phikon-v2's pinned card confirms TCGA source pretraining; any later Phikon comparison must be labeled overlap-sensitive, with exact slide overlap unresolved. Neither result should be called contamination-free external validation without further evidence.

Frozen, label-blind feature caching is allowed for all roles; classifier fitting, feature-statistic fitting, prototype adaptation, and selection use only permitted fit/development roles. The encoder's official transform and pinned revision remain unchanged. Predictions on sealed patients are not produced in the current preparation phase.

A future preregistered primary endpoint is equal-subset mean 16-class macro F1, with all 15 subset scores and patient dependence retained. Both development and sealed cohorts cover all 16 classes, avoiding missing-class denominator changes. Report full/worst subsets, class recall, calibration, and patient-level uncertainty when estimated. Sixteen development and 32 test patients remain small; masks and image counts do not increase independent sample size.

The immediate scope is metadata registration and frozen features only. Before any classifier pilot, freeze a minimal arm/grid budget in a new experiment specification. Do not carry the 175-run BreakHis search onto the 16 development cases.

## Hardening and reproduction

`python -m magset.tcga_protocol --inventory /path/to/inventory-v1 --output /path/to/new-protocol-directory`

The generator rejects wrong source hashes, duplicate patients/slides/fields, class-count changes, changed missing-view patterns, or incorrect per-class/global roles. Existing protocol directories are never overwritten. The native feature reader validates path/slide/patient/label/view identity and is bound to the registered protocol hash supplied at submission.

Future jobs use read-only tracked-source snapshots, bound to the Git archive's embedded commit and archive checksum. `slurm/tcga_features.sbatch` requires that source and protocol hash. This protects queued work from subsequent edits to the mutable working copy.
