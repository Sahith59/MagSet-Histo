# Independent review: supplied CNN-transformer notebook

November 17, 2026. **Accept the inspected source, recorded common-data audit and both actual CPU synthetic technical checks. GPU cohort outcomes remain pending.** The supplied original notebook has fifteen unexecuted code cells and no saved outputs. Its exact private original SHA256 is `2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`. No accuracy in its comments or hardcoded comparison rows is a measured result from this file.

This reviewer performs read-only notebook/runner inspection and authors the separately requested read-only CPU collector. No fitting, scheduler action, original-image inspection or runner-source edit is performed here. The user requested execution of the supplied notebook; its original image-based endpoint must remain visible rather than being silently replaced by the project's five-category provider-case benchmark.

## Exact supplied method

The notebook combines individual 40×/100×/200×/400× images into one image table. It stretches each image to 256×256, center crops to 224×224, and uses subtype-stratified image-random 80/20 splitting with `random_state=42`. Model/augmentation seed is 812. If all 7,909 images load, the split is 6,327 fitting and 1,582 held images. The held set also supplies checkpoint selection, learning-rate decisions and early stopping, so it is reused validation despite its `X_test` name.

There is no provider-case grouping or missing-view protocol. Related fields and magnifications can occur in both image roles. The filesystem walks are unsorted, so the seed alone does not pin image identities across different filesystem enumerations. A shared frozen image order and explicit split indices are needed to make the two declared executions comparable; actual provider-key overlap must be measured rather than assumed.

| Component | Supplied implementation |
| --- | --- |
| Reader | ImageNet EfficientNetV2B0 without its classification top; earlier layers frozen and the final sixty Keras layers trainable |
| Tokens | One image's 7×7 CNN map becomes 49 spatial tokens; 512-dimensional projection, learned CLS token and positions |
| Transformer | Four encoder blocks, eight attention heads, FFN width 1,024, dropout .2 and layer normalization |
| Binary output | CLS → Dense128/GELU → dropout .4 → two-way softmax |
| Subtype output | CLS concatenated with the predicted binary probabilities → Dense256/GELU → dropout .4 → Dense128/GELU → dropout .3 → eight-way softmax |
| Fitting | Adam 1e-4, batch 32, at most fifty epochs; binary/subtype losses weighted .3/.7 plus declared kernel regularizers |
| Selection | Best validation eight-subtype accuracy; ReduceLROnPlateau patience four and EarlyStopping patience twelve |

Both heads run on every image and receive joint supervision. The subtype loss also backpropagates through its concatenated binary input. This is a conditioned multi-task classifier, rather than a hard benign-stop/malignant-four cascade. Its transformer attends within one image, not across four magnifications. The notebook's L2 penalties apply to selected projection/head dense kernels; its Adam optimizer is not decoupled AdamW weight decay.

## Scaling and the two authorized variants

The supplied custom generator passes `img/255.` to `ImageDataGenerator.random_transform`; evaluation also uses `X_test/255.`. The generator's separate `rescale=1./255` setting is not applied by that direct transform call: Keras performs normalization in `standardize`, which this generator does not call. Thus the supplied model receives approximately [0,1], rather than another external division to [0,1/255]. [Official Keras 2.15 preprocessing source](https://raw.githubusercontent.com/keras-team/keras/v2.15.0/keras/preprocessing/image.py).

EfficientNetV2's default includes internal preprocessing and expects input pixel values in [0,255]. The external division therefore mismatches its documented transfer-learning input convention. [Official TensorFlow EfficientNetV2B0 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B0).

The root explicitly fixes two named variants before outcomes: **faithful** retains the supplied external division, and **input_fixed** removes that division from training/validation generation and numerical evaluation only. This is a declared scientific comparison, not an unnoticed compatibility patch. Both must consume the same exact ordered image list, labels and split indices, with the same architecture, trainability, seeds, augmentations, losses, optimizer, callbacks and epoch budget. Neither result becomes the existing provider-case five-category result. Two variants and repeated checkpoint monitoring supply reused-validation choices, not independent replication.

## Serialization and faithful adaptation

The custom transformer block's `get_config` omits its required `dff` constructor argument and actual dropout rate. It also reads `self.attention.key_dim`, while official Keras 2.15 stores `_key_dim` without that public property. This can prevent the first whole-model checkpoint save in that version; in other versions an incomplete configuration still prevents faithful full-model reconstruction. [Official Keras 2.15 MultiHeadAttention source](https://raw.githubusercontent.com/keras-team/keras/v2.15.0/keras/layers/attention/multi_head_attention.py).

A declared serialization-only patch may store and return all constructor settings without changing forward computation or initializers. The notebook's existing-graph `load_weights` path and full-model reconstruction are different operations; a complete model cannot be assumed reloadable merely because existing-graph weight loading succeeds. Actual save/reload prediction parity is required before cohort fitting. Original and adapted notebook hashes, exact changed cells, framework versions and trainable-layer names/parameter counts must remain recorded privately.

An eight-image synthetic batch can test model build/save/reload/predict. It cannot pass the original eight-class stratified 80/20 split with one sample per class. A technical smoke must explicitly avoid that impossible split and avoid presenting synthetic outcomes as cohort metrics.

## Metrics and visualization

The original headline eight-class accuracy/F1/AUC uses the eight-subtype argmax for all held images, without binary gating. The notebook's variable named malignant subtype recall counts correct subtype predictions among all truly malignant images but never checks the binary prediction. It is a conditional eight-head subtype accuracy, not the advertised hard-cascade recall. Its unused `['DC','LC','MC','PC']` membership test also does not match the full subtype directory names in the project dataset.

Preserve those original code-defined numbers. Separately label any strict malignant cascade recall as the fraction of truly malignant held images with **both** a malignant binary decision and correct original subtype. A five-category cascade diagnostic must declare its own rule: binary benign terminates as pooled benign; binary malignant selects only among the four malignant subtype scores using the actual encoder class-name mapping. Post-hoc label pooling or malignant masking changes the endpoint and cannot be relabeled as original eight-class performance or the provider-case benchmark. Quadratic kappa is also dependent on the nominal classes' arbitrary alphabetical ordering.

The function called attention rollout extracts no attention matrices: its extraction loop contains `pass`, and it displays final patch-token L2 norms. That is an activation-norm proxy, not layer/head attention rollout or causal tissue evidence. The visualization cell additionally sends raw [0,255] images to a faithful model numerically evaluated on /255 inputs, so its displayed predictions/maps use a different input path. Its RGB image is blended with an OpenCV BGR heatmap before a whole-overlay channel swap. These display issues must be disclosed; a separately corrected visualization may not silently replace the original numerical evaluation.

## Runner acceptance checklist

Read-only source acceptance must verify the exact original hash, enumerated declared patches, consumed shared split artifact, per-variant input range, strict GPU requirement for full execution, preserved original metrics and explicit additional diagnostics. Both successful and failed nbclient executions must retain the executed/partial notebook, traceback, exact source/environment and checkpoint state without overwriting the original. Actual cohort accuracy remains unknown until execution and independent saved-output review complete.

## Adapted source and independent collector review

The inspected adapter preserves literal unsorted notebook enumeration and binds every file to the canonical 7,909-image manifest and image-byte hashes. Its zero-fit audit saves the exact ordered CSV, full encoded labels and 6,327/1,582 image indices. Before either real fit, the original notebook's actual dataframe order, split labels and pixel-array indexing must match that shared artifact. This does not remove the original image-level provider overlap; the actual overlap counts will be reported.

Both variants preserve the original model/training cells. The scientific difference is exactly the two external-division substitutions in cell indices 4 and 10. Common changes are task paths, complete custom-layer serialization, pinned runtime/GPU checks, data proof, progress/export instrumentation and checkpoint verification. A common Keras I/O wrapper sets an explicit project cache directory without changing the official download URL or hash validation. The original ImageNet backbone is pinned as `efficientnetv2-b0_notop.h5`, 24,274,472 bytes, SHA256 **`fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285`**. A pre-fitting receipt and post-run cache receipts must match it for both variants.

Real execution requires a visible allocated GPU and TensorFlow 2.15.1/Keras 2.15.0. The CPU synthetic check uses eight generated images, one technical batch, inline plotting and three serialization paths: H5 weight loading, the notebook's actual `load_weights` call on a `.keras` checkpoint, and full-model reconstruction. These are technical checks, not cohort outcomes. Production separately reconstructs the best full checkpoint and checks both outputs on the first 32 original held images, using the existing variant-correct `X_test_norm`, at fixed tolerance 1e-5.

The adapter retains an immutable preexecution notebook, hashes every actual instrumented cell, and writes an atomic partial executed notebook after each completed cell and on error. Failed execution retains its traceback, available checkpoint/artifacts and an explicit failed result rather than overwriting the original or claiming complete performance.

The reviewer-authored `scripts/collect_supplied_cnn_transformer.py` performs no TensorFlow import, fitting or inference. It independently reconstructs original metrics from the saved 1,582-row probability NPZ, using confusion-matrix arithmetic for accuracy/precision/recall/F1/quadratic kappa and the standard ROC-AUC routine. It checks exact shared indices/order/truths, original argmax predictions, finite normalized scores, first-maximum checkpoint epoch, completed-epoch budget and reported metrics. Ungated true-malignant subtype accuracy, strict binary-AND-subtype malignant recall and joint eight-head correctness remain separately labeled. No unregistered five-category masking score is introduced automatically.

The collector also checks source/configuration/notebook/artifact hashes, original-to-adapted and adapted-to-executed cell sources, the pre-fitting CSV/index proof, actual GPU runtime and exact official pretrained weight bytes. It reuses only the frozen adapter's pure string construction to verify the executed-source recipe, not its splitting, training or metric calculations. Missing, failed or invalid variants produce an incomplete/failure comparison with sanitized public status and private error details. Actual numeric results are never invented for an unfinished variant, and a completed counterpart remains separately identifiable.

Fifteen independent no-fitting checks passed against standard metric oracles, strict-cascade arithmetic, changed split/truth/argmax/probability rejection and public-safe shared-input failure reporting. A further synthetic saved-artifact fixture exercised full 1,582-row acceptance, incorrect-best-epoch rejection and missing-variant status, without TensorFlow, a real checkpoint or model inference. These certify collector mechanics only; they do not constitute native TensorFlow execution or research performance.

Reviewed SHA256 snapshots:

| File | SHA256 |
| --- | --- |
| Registered configuration | `df81d26d562b931d80964d962251772f4932f32eef353123f55a59c2406e9c7b` |
| Notebook run adapter | `6368de912091db95ee7505a2b53078f2c52365da9b58e90913883d338fdfde8c` |
| Independent CPU collector | `a94b010245fb334656ccb8f679cd1821a81600d1d0d5d5aa3d383e1f91e59a5b` |

**Source verdict:** no remaining inspected-source blocker for the common zero-fit audit and bounded CPU technical checks. The recorded preflight acceptance below supersedes its pending-execution status. GPU cohort outcomes and saved-output verification remain required; queue placement, synthetic checks and a compatible environment establish no new accuracy.

## Actual common-data and CPU technical acceptance

The common zero-fit data audit and both native TensorFlow CPU technical checks completed at source **`474bb0199eba585aec3e70bd252a4b840357429d`**. Independent review passed **238 source/configuration/notebook/artifact/receipt assertions**, including reconstruction of both synthetic notebooks' five executed-cell source digests and all fifteen original/compatibility patch receipts. The immutable source archive SHA256 is `2ad8f54d02e3ffc32d15069585a5f4a480505c2c844df6f92228aff8c32d752b`; its actual bytes, committed adapter and configuration match the recorded receipts and the reviewed snapshots above.

| Actual accepted receipt | SHA256 |
| --- | --- |
| Common input audit, public | `4bd1a3cc5e6f5dbdd7b2c6e11a32c1fe0e5e806b060d345a6e660add3e9b6752` |
| Faithful CPU technical result | `f81296104c0b4dfc064a9fd498970538d7b11cef31e4492a00d2a510bad0eacf` |
| Input-fixed CPU technical result | `296ca9741caa3b5816f91c27a43d39924f7579897966b8650d8e02c01510c9e2` |

The common audit records exact full 7,909-image membership, pinned canonical manifest, current unsorted image order, label encodings and shared **6,327 fitting/1,582 reused-validation indices**. All **82 provider keys** and all **70 conservative identifier groups** occur in **both** image roles. This is measured overlap in this supplied-notebook split, not a provider-case-disjoint or independently verified clinical-patient test. The notebook's original image endpoint remains separate from the project's existing case/subset endpoint.

Both CPU technical receipts use Python 3.11.18, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4 and scikit-learn 1.6.1. Their visible GPU count is zero: these are CPU compatibility checks, not GPU execution evidence. Each processes eight generated images with one synthetic training batch, verifies inline plotting, and records zero real validation predictions. All three maximum absolute reload errors are **0.0** in both variants: H5 weights, existing-graph loading of the `.keras` file, and full-model reconstruction. Their saved environment/technical/weight JSON hashes match the execution artifact receipts.

Before and after the synthetic batch, both variants bind the same official `efficientnetv2-b0_notop.h5`, exactly **24,274,472 bytes** with SHA256 **`fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285`**. The common explicit project-cache I/O override is disclosed; no alternate weights, model, loss or dataset protocol was substituted.

This review independently checks the downloaded recorded receipts, source archive and committed source bytes. It does **not** rerun TensorFlow locally or independently load the private synthetic checkpoint arrays/notebooks or all 7,909 original images here. In particular, the native zero reload differences are accepted recorded technical evidence, not a second local numerical reproduction. The shared-order/indices hashes are verified as recorded bindings; actual source enforces them again before either GPU fit.

**Actual technical verdict:** accept both CPU checks and the recorded common-data contract as support for the two declared GPU executions. Synthetic batch losses/accuracies are not research results and are deliberately omitted here. No final cohort accuracy, paper superiority, 95% result, clinical validation or novelty is established by these checks. The completed GPU notebooks, saved 1,582-row probabilities and independent CPU collector still determine the actual reported numbers.
