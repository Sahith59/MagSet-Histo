# MagSet-Histo

Breast tissue can look very different at different microscope magnifications. A wider view shows tissue structure; a closer view reveals cell detail. This project studies how to combine those views to classify breast tumor subtypes, including cases where some magnifications are unavailable.

We use **MAGSET** as a working name for a Magnification-Aware Set Transformer. The method is under development. The first experiments use BreakHis at **40×, 100×, 200×, and 400×**, with eight tumor-subtype labels.

## What we are testing

The starting point is a pretrained pathology image encoder followed by a model that combines the available sets of images. We will compare simple averaging, learned gating, and transformer fusion before testing a candidate that matches comparable evidence across magnifications while retaining evidence unique to each view.

Cross-magnification matching and missing-view models already have substantial prior work. Our proposed extension is a research hypothesis, and its value will depend on controlled comparisons and ablations.

## Data and evaluation

The official BreakHis archive contains **7,909 images**. Our initial audit confirms the four magnification totals. Patient/specimen grouping is still being reconciled with the original protocol: bare filename identifiers are not globally unique, and some specimens share numeric identifier stems.

We will evaluate every nonempty combination of the four magnifications, giving **15 availability settings**. Related images must remain in the same split. Eight-class macro F1, per-class recall, full-view and worst-subset performance, uncertainty, and computation will be reported separately from image-level accuracy.

The code includes pinned frozen-feature extraction for Phikon-v2 and ResNet-50, provider-protocol auditing, full PNG validation, linear field/case controls, and six masked hierarchical pooling/attention baselines. Pilot model selection uses inner validation only. The new training CLIs do not score outer cases.

The first 30 validation-only baseline runs are complete; see [the pilot results](docs/VALIDATION_PILOT.md). Final outer-test evaluation has not been run. The supplied historical notebooks reported 93.73% and 94.99% accuracy on 40× images, but their split and model-selection procedures do not establish generalization to unseen patients. Those scores are not MAGSET results.

The next 50 controlled comparisons are also complete and verified; see [field and neural results](docs/FUSION_RESULTS.md). Learned attention has not improved the initial Phikon mean baseline in this exploratory screen. Split/training-seed stability, grouping sensitivity, and external validation remain important next steps. The proposed matching mechanism is still unimplemented and unproven.

## Getting started

The download/indexing tools use Python 3.9 or newer and the standard library. The baseline checks additionally need NumPy, SciPy, scikit-learn, and Pillow:

```bash
python3 -m pip install numpy==1.26.4 scipy==1.15.2 scikit-learn==1.6.1 Pillow==11.1.0
python3 -m unittest discover -s tests -v
python3 scripts/prepare_breakhis.py --data-root /path/to/breakhis --download
```

On a shared cluster, submit download, validation, and training through its scheduler. See [the portable cluster guide](docs/CLUSTER_SETUP.md) for configuration. The original dataset and pretrained weights are obtained from their providers, not bundled in this repository.

Neural checks additionally require `torch==2.6.0`; without it those checks are explicitly skipped. CI runs them in a separate pinned CPU-PyTorch job. Cluster training uses the pinned CUDA runtime described in the cluster guide.

## Project documents

- [Research and execution plan](plan.md)
- [One-page project brief](docs/PROJECT_BRIEF.md)
- [Related work and comparison rules](docs/RELATED_WORK.md)
- [Data audit](docs/DATA_AUDIT.md)
- [Implemented validation pilot protocol](docs/BENCHMARK_PROTOCOL.md)
- [Initial validation results](docs/VALIDATION_PILOT.md)
- [Computed error analysis](docs/ERROR_ANALYSIS.md)
- [Controlled fusion protocol](docs/FUSION_PILOT.md) and [results](docs/FUSION_RESULTS.md)
- [Research direction and contribution gates](docs/RESEARCH_DIRECTION.md)
- [Second-dataset archive and metadata audit](docs/TCGA_MS_AUDIT.md)
- [Implemented methods draft](paper/methods-draft.md)

The historical reference is [Lang et al., SPIE Medical Imaging 2021](https://doi.org/10.1117/12.2582299). The closest task and mechanism comparisons currently include [MagViT](https://arxiv.org/abs/2608.16959) and [M2OTCA](https://doi.org/10.1016/j.media.2026.104082). See the related-work document for distinctions between their tasks and our intended evaluation.
