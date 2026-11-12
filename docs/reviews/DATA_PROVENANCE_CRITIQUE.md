# Independent critique: data and encoder provenance

Reviewed November 12, 2026. Scope: readiness of TCGA-MS as a second task, encoder training provenance, release metadata, and a defensible next experiment. This is a critique and protocol recommendation, not an experimental result.

## Overall assessment

**Image readiness is strong; scientific evaluation readiness is conditional.** The obtained archive, checksums, labels, and decoding have useful evidence behind them. TCGA-MS can support a secondary multi-organ morphology experiment, but it cannot presently support a claim of contamination-free external evaluation with Phikon-v2. It also does not validate the identical eight-class breast classifier: its labels and disease sites differ.

The fastest useful next step is a sealed, native 16-class protocol with an ImageNet-pretrained control, followed by a small prespecified validation experiment. Another large search on the existing tiny development sets would add little credibility.

| Review dimension | Score / 5 | Assessment |
| --- | ---: | --- |
| Data integrity and traceability | 4 | Archive/revision/checksum and full decoding are documented; release/paper count mismatch remains unexplained. |
| Group and split evidence | 4 | TCGA patient metadata and true disjoint source test folds are auditable; a globally sealed development protocol is still needed for this project. |
| Encoder provenance | 2 | Phikon has confirmed TCGA source exposure; exact release-slide intersection is unresolved. |
| Same-task comparison readiness | 2 | The current code assumes eight labels/four BreakHis magnifications; the native 16-class interface is not yet a completed benchmark. |
| External generalization evidence | 1 | No second-dataset model result exists, and no contamination-free cross-task claim has been demonstrated. |

These ratings describe readiness, not a score for the quality of an unwritten final paper.

## Critical and high-priority findings

### Critical: Phikon-v2 is not a clean external encoder for a TCGA claim

The [official model card at the exact encoder revision used here](https://huggingface.co/owkin/phikon-v2/blob/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd/README.md) describes PANCAN-XL pretraining with a substantial TCGA component. The [model paper, §3.1 and Extended Table 6](https://arxiv.org/html/2409.09173v1) identifies 29,502 TCGA slides and includes the relevant breast, kidney, stomach, and sarcoma sources. **Source-level overlap is confirmed. Exact overlap of these 128 slides or their patches is not established by this audit.** A slide-level pretraining membership list was not obtained.

This does not prove label leakage into our supervised head, and it does not make every TCGA experiment invalid. It changes the permitted interpretation: Phikon results would be a secondary task using a potentially exposed frozen representation, not evidence of generalization to unseen pretraining patients. State this prominently rather than hiding it in an access footnote.

The pinned ResNet-50 configuration uses torchvision `IMAGENET1K_V2`. Its [official torchvision 0.21 documentation](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet50.html) documents ImageNet-1K weights and the associated resizing/cropping transform. Use this as the first control without a known TCGA pathology-pretraining source. It is not a substitute for a strong pathology encoder with demonstrably appropriate provenance.

**Decision:** start any native TCGA supervised pilot with the frozen ImageNet control. Keep Phikon as an explicitly overlap-sensitive comparison. Do not advertise a contamination-free pathology-foundation-model validation unless that claim is independently supported.

### Critical: patch count is not independent sample size

The audited release has 65,312 valid images, but only **128 labeled slides from 128 distinct supplied patient IDs**, with eight slides per canonical class. Only 32 patients are labeled as breast in the supplied tissue field; the others are stomach (37), soft tissue (38), and kidney (21). This is a multi-organ morphology task, not another eight-subtype breast cohort.

Thousands of fields, views, or mask augmentations do not increase the number of independent patients. The same caution applies to the 175 completed BreakHis development configurations: repeated configurations are not new evidence cohorts.

### High: completeness and evaluation population must be fixed

I independently recomputed the following from the supplied `labels.csv` and archive `members.csv`, excluding `._` sidecars:

| Native four-resolution setting | Audited quantity |
| --- | ---: |
| Views | 0.25, 0.5, 1.0, 2.0 microns per pixel |
| Patients with at least one selected view | 128 |
| Selected-view images across those patients | 37,024 |
| Patients complete at all four views | 122 |
| Images in the complete four-view cohort | 35,723 |
| Incomplete patients | 6 |
| Missing view in every incomplete case | 0.25 mpp only |

The complete cohort has six to eight patients per class. Comparing different availability subsets on different eligible patient populations would confound missingness with case selection. Use a fixed complete cohort for a controlled 15-subset evaluation. Release incompleteness does not establish why a view is missing in clinical practice.

If the six incomplete cases are retained in fitting under the recommended new holdout below, they cannot also be reported as an independent naturally incomplete test cohort.

### High: source folds are genuine test folds, but using every dev fold can compromise a later global holdout story

The archive's supplied metadata differs from BreakHis's repeated holdouts: each of the 128 TCGA patients is assigned `test` **exactly once**, `dev` exactly once, and `train` three times across the five source folds. Patient IDs do not cross roles within a fold. Test fold sizes sum to 128.

However, globally choosing an architecture after examining development outcomes across all five rotating folds uses outcomes from every patient somewhere. That is not a globally untouched cohort for subsequent architecture discovery. A conventional nested evaluation can be specified, but it must be nested at the level of the actual choices being made. Do not relabel a heavily reused development process as a fresh final test.

All source training partitions cover all 16 classes. Development and test partitions do not:

| Source fold | Train / dev / test patients | Missing dev classes | Missing test classes |
| --- | --- | --- | --- |
| 0 | 75 / 25 / 28 | 22 | 27, 44 |
| 1 | 77 / 26 / 25 | 32, 33 | 22 |
| 2 | 73 / 29 / 26 | 12, 24, 28 | 32, 33 |
| 3 | 79 / 20 / 29 | 24, 25, 38 | 12, 24, 28 |
| 4 | 80 / 28 / 20 | 27, 44 | 24, 25, 38 |

Numbers in the last two columns are original label IDs, not counts. Filtering to complete four-view cases changes those role sizes and can remove further classes from a role. Preserve an eligibility ledger rather than silently modifying the source protocol.

### High: current interfaces are not native TCGA interfaces

`magset/data.py` recognizes eight BreakHis subtype strings, fixed magnifications, and a BreakHis filename-based case identifier. `magset/features.py` filters with that schema. The current classifier/metric paths also assume eight outputs. Passing TCGA filenames through those parsers or renaming mpp directories to BreakHis magnification numbers would misstate semantics.

Keep the reusable frozen encoder loader, but add an explicit native dataset adapter and a separate cache namespace. Carry `dataset`, `patient_id`, `slide_id`, original `label_id`, contiguous class index, and `resolution_mpp` as distinct fields. A resolution identity is not an objective magnification or a new image acquisition.

The 16 original IDs are `12, 13, 14, 15, 16, 18, 22, 24, 25, 27, 28, 32, 33, 38, 41, 44`. Freeze their sorted mapping to indices 0–15. Do not reconstruct labels from free-text histology or tissue site: descriptions can be multiple, blank, or shared across canonical IDs.

## Recommended native protocol for the next week

### Preferred new development/holdout track

The parent agent proposed the following hardening before any TCGA classifier fitting. I support it subject to the limitations below:

1. Choose four native resolutions in advance: 0.25, 0.5, 1.0, 2.0 mpp. Treat fields within each case/view as unpaired bags.
2. Within each canonical class, consider complete four-view patients sorted by SHA-256 of `magset-tcga-quarantine-v1-20261112|case_uuid`. Assign the first two to a sealed test set and the next one to development. Put all remaining patients, including the six incomplete cases, in fitting.
3. Freeze and hash the assignments before classifier fitting. This yields **80 fitting, 16 development, 32 sealed-test patients**, with five/one/two patients per class respectively. Preserve source fold columns as provenance; this is a **new protocol**, not an exact reproduction of the source paper.
4. Fit normalization and any adaptation using the 80 fitting patients only. Test labels may define stratified membership, but must not tune models, thresholds, field choices, or hyperparameters. Keep the test results sealed until method and selection rules freeze.
5. For incomplete fitting cases, average the objective over nonempty subsets of the views actually present, normalizing total contribution per case. Do not penalize these cases less or more merely because they have seven supported subsets instead of fifteen.

With every class covered in the 16-case development partition, explicit fixed-16-class macro F1 is a reasonable prespecified selection metric. Keep a very small fixed tuning budget. One development patient per class is still weak evidence and can be overfit through repeated architecture selection.

The final primary endpoint can be equal-subset mean fixed-16-class macro F1 over the same 32 complete test patients and all 15 subsets. Also report full-view and worst-subset scores and per-class support. All methods must use the same cases, fields, preprocessing, feature caches, masks, and selection opportunity.

For uncertainty and paired comparisons, resample patient identities jointly across all subsets and methods. The 480 test probability records from 32 × 15 are not 480 independent patients. With two test patients per class, intervals and hypothesis tests have little power; a bootstrap cannot create absent patient diversity. State that uncertainty is conditional on the trained models and selected cohort, not a complete estimate of retraining variance.

### Optional exact-source reference track

If a separate source-protocol reproduction is later run, keep its defaults distinct from the new task. The [official evaluator](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/evaluate_benchmarks.py) pools training patches from four mpp values, evaluates per-resolution patch classifiers, defaults to CLS-plus-mean token features, and tunes logistic regression or kNN on development data. The [data loader](https://github.com/bifold-pathomics/continuous-magnification-sampling/blob/83f05d8eed96e805b737b9ea935efa1c4bedb8a1/dataloaders.py) uses supplied fold columns. Our case-bag, CLS-only, missing-subset experiment is a changed task/representation, even with the same backbone.

For that reference track, prespecify development scoring over explicitly present true classes and report class coverage. Do not let an implicit union of true and predicted labels change a macro-F1 denominator by model. Future final pooled out-of-fold predictions can support fixed-16-class macro F1 because every patient is tested once, provided model selection is properly contained. That track must not reuse the new sealed patients for architecture discovery and then claim they were untouched.

## What to show Professor Dong next week

- A verified data/provenance table and the sealed patient assignment hashes, with an explicit distinction between the two label spaces.
- A small validation-only baseline table with fixed ImageNet features and no final test scores.
- The confirmed TCGA pretraining-source overlap note for Phikon and the remaining exact-membership uncertainty.
- A clear scientific question and a preregistered comparison plan, including what result would stop the proposed method.
- The existing negative BreakHis findings, rather than a claim that more GPU work already demonstrates a novel method.

This package can establish that the project is becoming reproducible and scientifically controlled. It will not by itself establish clinical validity, a new mechanism, or conference readiness.

## Scope and reproducibility of this review

Metadata calculations were read-only and used the pinned release already obtained by the project. No outer model outcomes were read, no classifier/encoder was fitted, no cluster job was modified, and no account registration, terms acceptance, or author contact occurred. The official evaluation repository head inspected was `83f05d8eed96e805b737b9ea935efa1c4bedb8a1`; downloaded dataset revision remains the separate revision recorded in `configs/tcga-ms-source.json`.

Confidence: high for metadata counts, role multiplicity, native view coverage, and documented TCGA source exposure; medium for implications for eventual generalization; unresolved for exact Phikon pretraining-slide/patch membership.
