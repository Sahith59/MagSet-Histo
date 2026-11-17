# Supplied notebook execution status

Snapshot: November 17, 2026, 07:10 AM EST. **Both GPU variants are training. No final notebook accuracy is available yet.** The original file had fifteen unexecuted cells and no saved outputs; its comparison-table constants are earlier reference claims.

| Stage | Job | Observed state |
| --- | --- | --- |
| Isolated TensorFlow environment |4985455|Complete, 5m54s |
| Shared image/label/order/split audit |4985474|Complete, 1m35s |
| Two synthetic CPU serialization checks |4985475|Both complete, 2m19s / 2m10s |
| Faithful and input_fixed GPU training |4985476_0 /4985476_1|Both running, one visible GPU each |
| Independent saved-prediction checker |4985477|Pending completion of the training array |

The scheduler allocated GPUs sooner than its earlier capacity forecast. Both runs have passed their actual prefit image/label/index guards and pinned TensorFlow/Keras/official-weight checks. Each uses the notebook's maximum fifty epochs, best eight-subtype validation checkpoint, learning-rate reduction and early stopping. An epoch progress CSV is saved while fitting; interim epoch scores are not final checked results.

## What this notebook does

It takes one tissue picture at any of the four zoom levels. EfficientNetV2B0 describes its regions; four Transformer blocks compare those descriptions. Two connected heads predict benign/malignant and one of eight tissue types. This is a conditioned two-head image model, separate from our benign-stop/four-malignant-type provider-case task.

The shared audit verifies all 7,909 original images and the exact 6,327 fitting /1,582 reused validation split. All82 provider keys and all 70 conservative groups appear on both sides through different images. Consequently these scores describe held-out pictures of familiar cases, without a new provider-case test. [Exact public input audit](evidence/SUPPLIED_CNN_TRANSFORMER_INPUT_AUDIT.json).

Faithful preserves the notebook's actual external division by 255. Input_fixed removes that division to match EfficientNetV2's built-in preprocessing; all other scientific settings and image indices remain fixed. Common compatibility repairs permit paths, shared official-weight caching and complete custom-layer serialization. [Fixed recipe](SUPPLIED_CNN_TRANSFORMER_RUN.md).

Both actual synthetic CPU checks use eight artificial images and one optimization batch, with zero real validation predictions. H5-weight, original.keras-weight and full-model reload differences are all 0.0 in both runs. These certify mechanics; their synthetic classification scores are not cohort performance. [Technical aggregate](evidence/SUPPLIED_CNN_TRANSFORMER_TECHNICAL.json), [independent review](reviews/SUPPLIED_CNN_TRANSFORMER_REVIEW.md).

## Automatic outputs

Each variant preserves its adapted/executed notebook, best.keras checkpoint, history, progress CSV, split and prediction arrays privately under `runs/supplied-cnn-transformer-v1/`. After both training tasks terminate, the independent CPU checker reconstructs actual binary/eight-type scores and strict cascade diagnostics. Missing or failed runs produce an explicit incomplete status without invented accuracy.

The machine-readable result is `runs/supplied-cnn-transformer-v1/checked/comparison.json`. A plain-language summary is scheduled after that checker at `runs/supplied-cnn-transformer-v1/summary.md`. Both are research artifacts outside public Git until the completed results are reviewed. The notebook's ungated malignant subtype measure remains distinct from actual routed malignant recall. No value here replaces the current five-category case scorecard or establishes 95% on that endpoint.

Numerical producer source: `474bb0199eba585aec3e70bd252a4b840357429d`. Exact original notebook SHA256: `2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`.
