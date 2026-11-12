# TCGA-MS archive and metadata audit

The pinned official archive downloaded successfully and matched the provider LFS SHA-256. The archive was inventoried without executing its contents. This is dataset preparation, not a second-dataset result.

| Item | Audited count |
| --- | ---: |
| PNG-suffixed archive entries | 65331 |
| Excluded `._` sidecar-named entries | 19 |
| Candidate image files | 65312 |
| Images with a supplied slide label | 65312 |
| Images without a supplied slide label | 0 |
| Slide identifiers in image paths | 128 |
| Labeled slides / distinct labeled patients | 128 / 128 |
| Labeled slides with all seven resolutions | 122 |
| Classes | 16 |

The candidate image count differs from the 67,200 patches described in [the paper](https://arxiv.org/html/2601.02198v1). In the [official release](https://huggingface.co/datasets/bifold-pathomics/TCGA-MS), 19 PNG-suffixed entries have `._` sidecar names and are only 233 bytes; they are excluded by filename policy before matching slides to labels. The remaining images all match the label table. Do not silently claim that the paper image count was reproduced; the reason for its difference remains unverified.

## Resolution inventory

| Resolution directory | Images |
| --- | ---: |
| mpp_0.25 | 9144 |
| mpp_0.375 | 9144 |
| mpp_0.5 | 9598 |
| mpp_0.75 | 9598 |
| mpp_1.0 | 9546 |
| mpp_1.5 | 9546 |
| mpp_2.0 | 8736 |

The supplied table contains 16 noncontiguous label IDs, with eight labeled slides per class. All five provider folds have zero patient-role overlap in the supplied metadata. Counts follow.

| Provider fold | Train slides | Dev slides | Test slides | Missing class/role combinations |
| --- | ---: | ---: | ---: | ---: |
| 0 | 75 | 25 | 28 | 3 |
| 1 | 77 | 26 | 25 | 3 |
| 2 | 73 | 29 | 26 | 5 |
| 3 | 79 | 20 | 29 | 6 |
| 4 | 80 | 28 | 20 | 5 |

## Consequences for the next experiment

- Restrict fitting eligibility to explicitly labeled slides and exclude resource sidecars; preserve an exclusion ledger. Full PNG decoding is verified below.
- Keep patient IDs, all slide fields, and all resolutions together using the supplied patient/split metadata. Remap the 16 label IDs explicitly.
- Treat each slide/resolution as an unpaired field bag. Equal coordinate strings do not establish physical alignment; no coordinate key spans all seven resolutions in this archive.
- Define four-native-resolution and/or seven-resolution availability protocols in advance. BreakHis objective magnifications and TCGA microns per pixel are different units.
- Audit encoder-pretraining overlap before treating this cohort as independent external evidence. No encoder adaptation or TCGA scoring has run.

Portable data-readiness jobs are `slurm/extract_tcga.sbatch` followed by `slurm/decode_tcga.sbatch`. They verify the pinned archive, safely extract, exclude sidecar-named entries from decoding, and record every error. Seven decode shards run with at most four workers.

Source revision/checksum: `configs/tcga-ms-source.json`. Reproduce with `python -m magset.tcga_audit --inventory /path/to/inventory-v1 --output /path/to/private-audit.json --markdown docs/TCGA_MS_AUDIT.md`.

## Full decoding verified November 12, 2026

All 7 resolution reports were verified against the inventory and pinned archive checksum. All 65312 images passed CRC/structure verification and full loading, with zero errors. Every decoded image is RGB and224×224. Extraction and all seven scheduler tasks completed successfully. This establishes image readiness, not a trained second-dataset benchmark.

Decode source revision: `328473a508c4a93f89a9ddafd9aabf9882f9a7d5`. Regenerate this section by adding `--decode-reports /path/to/decode-v1` to the command above.
