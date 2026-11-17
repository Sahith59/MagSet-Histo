# Supplied CNN–Transformer notebook results

**Both registered runs are complete and independently accepted.** The input-corrected run reaches **97.09% benign/malignant accuracy and 86.41% eight-type image accuracy**. These are different tasks; 97.09% is not a five-category result. The exact original notebook remains unchanged privately. It contained fifteen unexecuted cells and no saved outputs; its hardcoded comparison-table entries were earlier reference claims.

| Checked metric | Faithful input | Input corrected |
| --- | ---: | ---: |
| Benign/malignant accuracy |68.58% (1,085 / 1,582)|97.09% (1,536 / 1,582)|
| Benign recall |0.00% (0 / 497)|96.58% (480 / 497)|
| Malignant recall |100.00% (1,085 / 1,085)|97.33% (1,056 / 1,085)|
| Binary ROC AUC |.538069|.994656|
| Eight-type image accuracy |43.62% (690 / 1,582)|86.41% (1,367 / 1,582)|
| Eight-type macro F1 |.075924|.842481|
| Eight-type macro recall |.125000|.847969|
| Eight-type macro one-vs-rest ROC AUC |.570418|.984252|
| Ungated true-malignant subtype accuracy |63.59% (690 / 1,085)|85.99% (933 / 1,085)|
| Strict malignant cascade recall |63.59% (690 / 1,085)|85.53% (928 / 1,085)|
| Both heads correct, all eight-type images |43.62% (690 / 1,582)|86.09% (1,362 / 1,582)|
| Completed / selected epoch |13 / 1|50 / 41|

[Complete accepted aggregate and artifact hashes](evidence/SUPPLIED_CNN_TRANSFORMER_RESULTS.json) · [Earlier faithful-only snapshot, preserved](evidence/SUPPLIED_CNN_TRANSFORMER_FAITHFUL_RESULTS.json) · [CNN/Swin side-by-side explanation](SUPPLIED_NOTEBOOKS_COMPARISON.md).

The faithful model predicts **malignant for every image and ductal carcinoma for every subtype answer**. Its two accuracies equal the majority-label frequencies. The input-corrected companion removes the external division by 255 to match EfficientNetV2's built-in preprocessing. Both use the same split, architecture, augmentation, optimizer and callback rules, plus the declared common compatibility repairs. The correction improves binary accuracy by 28.51 percentage points and eight-type accuracy by 42.79 points in this fixed pair. Both outcomes are retained; no further variant or split was selected.

## What the model does

It takes one tissue picture at any of the four zoom levels. ImageNet-pretrained EfficientNetV2B0 describes its regions; four Transformer blocks compare those descriptions. Two connected heads predict benign/malignant and one of eight tissue types. The eight-type head sees all fitting images, including benign subtypes, and receives the binary probabilities as extra inputs. This differs from the capstone's benign-stop/four-malignant-type provider-case task.

All **7,909 original BreakHis images** are covered by the audit: **6,327 fitting images and 1,582 reused validation images**. The same accepted image order and indices were used by both variants. All **82 provider keys** and all **70 conservative groups** appear on both sides through different images. These are held-out pictures of familiar cases, rather than a new provider-case test. The checking images also selected the checkpoint and controlled learning-rate reduction and early stopping. [Public input audit](evidence/SUPPLIED_CNN_TRANSFORMER_INPUT_AUDIT.json).

Both runs preserve the authored maximum fifty epochs, batch size 32, seed 812 and best eight-subtype validation checkpoint rule. Faithful ended after thirteen epochs through early stopping; input corrected completed fifty and selected epoch 41. The faithful checkpoint selected epoch 1. Both selected whole-model checkpoints reproduced the scored probabilities on the fixed 32-image reload probe with **0.0 maximum absolute difference for both heads**. The independent saved-prediction checker accepted both results and found no review errors. It performed no new fitting or inference.

## Execution and interpretation

| Stage | Job | Final accepted state |
| --- | --- | --- |
| Isolated TensorFlow environment |4985455|Complete |
| Shared image/label/order/split audit |4985474|Complete |
| Two synthetic CPU serialization checks |4985475|Both complete |
| Faithful GPU training |4985476_0|Complete |
| Input-corrected GPU training |4985476_1|Complete |
| Independent saved-prediction checker |4985477|Complete; both variants accepted |
| Plain-language result renderer |4985494|Complete |

The synthetic CPU checks used eight artificial images and one optimization batch per variant, with zero real validation predictions. Their H5-weight, original.keras-weight and full-model reload differences were all 0.0. Those scores certify mechanics only. [Technical aggregate](evidence/SUPPLIED_CNN_TRANSFORMER_TECHNICAL.json), [pre-production independent review](reviews/SUPPLIED_CNN_TRANSFORMER_REVIEW.md).

Path, custom-layer serialization and official-weight-cache repairs preserved the registered scientific settings. Faithful keeps the actual one external division by 255; input corrected removes it. The generator's `rescale` metadata was never applied by `standardize()`, so this was not an applied double external division. [Fixed recipe](SUPPLIED_CNN_TRANSFORMER_RUN.md), [registered configuration](../configs/supplied-cnn-transformer-v1.json).

The notebook's original malignant subtype measure is **ungated**: a correct subtype counts even when the binary route is wrong. Strict malignant cascade recall additionally requires a correct malignant route. Both-head correctness additionally requires the correct binary answer for every eight-type image. None is automatically the capstone's five-category provider-case accuracy. No CNN five-category score is reported here.

Both adapted and executed notebooks, histories, progress, selected checkpoints and prediction arrays remain preserved privately. Public evidence contains aggregate metrics, confusion matrices, class supports and hashes. The visualization is a **token-norm proxy**, not attention rollout or causal evidence. In faithful, its raw-pixel visualization input scale differs from scored inputs; input corrected aligns those scales.

These results do not replace the separate five-category case scorecard or establish 94–95% on that endpoint, clinical validation, novelty or a matched improvement over a published paper. Pooling the four magnifications as independent images does not demonstrate multi-view case fusion or robustness across all fifteen view subsets.

Numerical producer source: `474bb0199eba585aec3e70bd252a4b840357429d`. Exact original notebook SHA256: `2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`. Accepted complete comparison SHA256: `d7972fc41c4d4594d606b06111bc3ecb219a7e037061debf61b731a74e3a7f25`.
