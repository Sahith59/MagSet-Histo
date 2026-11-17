# Supplied CNN–Transformer notebook: fixed execution recipe

November 17, 2026. We will run the supplied notebook in two versions: **faithful** preserves its actual input scaling; **input_fixed** removes the external division by 255 because EfficientNetV2 already includes preprocessing. Both keep the same architecture, image split and training settings. **The notebook contains no saved outputs or completed results, and neither variant has a performance claim here.** This eight-subtype image experiment is separate from the current five-category provider-case project endpoint.

The private original notebook is preserved unchanged, SHA256 **`2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`**. The [registered configuration](../configs/supplied-cnn-transformer-v1.json) fixes both variants before their outcomes. Original notebook contents, image arrays, weights, exact path order and split indices stay private; reports retain hashes and safe aggregates.

## Two prespecified input variants

| Variant | Input passed to the model during training, validation and scoring | Other changes |
| --- | --- | --- |
| `faithful` | Authored RGB image divided once by 255, approximately 0–1 | Dataset/output path compatibility and complete custom-layer serialization only |
| `input_fixed` | RGB float pixels approximately 0–255; remove external `/255` in generator and scoring | Same compatibility repairs; architecture, split, augmentation, losses, callbacks and budgets unchanged |

EfficientNetV2's default `include_preprocessing=True` retains its own preprocessing in both variants. Official Keras documentation specifies 0–255 float inputs in this mode. This motivates the input correction without establishing its eventual accuracy. [Keras EfficientNetV2 documentation](https://keras.io/2/api/applications/efficientnet_v2/).

The notebook's generators call `datagen.random_transform(img / 255.)` but never `datagen.standardize()`. Its `ImageDataGenerator(rescale=1./255)` setting therefore does **not** introduce a second external division. Faithful preserves the actual one-division path; input_fixed removes that division for both train and validation generators and removes `X_test / 255.` during scoring. Do not describe this as removing an applied double-rescale operation or modify unrelated augmentation.

## Image split and interpretation

Use all **7,909 BreakHis images** across 40×/100×/200×/400× as individual samples. Subtype labels are the eight LabelEncoder classes: adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor and tubular adenoma. Binary labels are benign/malignant. This is neither pooled-benign five-category fitting nor a four-type malignant-only expert.

Preserve the notebook's authored unsorted `os.listdir`/`os.walk` enumeration on the current dataset. A zero-fit audit binds every enumerated image to the canonical manifest, stores the exact ordered membership and labels, and records the split indices. Apply the authored subtype-stratified `train_test_split(test_size=.2, random_state=42)`: **6,327 fitting images and 1,582 reused validation images**. Both variants must load the same accepted order/indices and assert them again before fitting. No historical saved file ordering is available, so this does not claim recovery of an unknown past notebook split.

This image-random split permits fields and magnifications from the same provider case on both sides. Quantify provider-key overlap in the audit rather than claiming independent patient separation. The validation set also guides checkpoint selection, learning-rate reduction and early stopping, then supplies the final reported scores. The notebook calls it “test”; reports must explain that it is reused validation. There is no additional fresh test cohort.

The existing 46-fitting/eight-validation/28-excluded roles, fifteen magnification subsets and five-category scorecards remain a different experiment. Pooling all magnifications as separate images does not implement multi-view case fusion or demonstrate missing-view robustness. Results here cannot replace that endpoint or establish a fair paper superiority claim.

## Exact preprocessing and model

Decode with OpenCV, convert BGR to RGB, resize to **256×256 with INTER_CUBIC**, then center-crop **224×224** and cast to float32. Training augmentation retains random rotations up to ±90°, horizontal and vertical flips, and reflection fill. Validation has no geometric augmentation. Preserve the authored generator's training shuffling, batch size **32**, 198 fitting steps and 50 validation steps per epoch, including final short batches. Seed Python, NumPy, TensorFlow and Keras as authored with **812**; record actual runtime/device versions rather than promising identical hardware trajectories.

Use ImageNet-pretrained **EfficientNetV2B0**, `include_top=False`, with built-in preprocessing retained. Freeze all but the final **60 backbone layers** and retain the notebook's normal layer training behavior. A 7×7×1,280 feature map becomes **49 tokens**, projected to 512 dimensions, layer-normalized and dropped out at .2. Add a trainable zero-initialized CLS token and random-normal positional embeddings for a **50-token sequence**.

Apply **four Transformer blocks**, each with eight attention heads, key dimension 64, model width 512, GELU feed-forward width 1,024, dropout .2 and layer-normalization epsilon 1e-6. Preserve the source's attention/dropout/residual/norm and feed-forward/dropout/residual/norm order. Layer-normalize the final CLS output.

The binary branch has a 128-unit GELU layer, dropout .4 and a two-way softmax. The subtype branch concatenates CLS with the **binary softmax probabilities**, preserving gradients through that connection, then applies 256/GELU/dropout .4 and 128/GELU/dropout .3, followed by an **eight-way softmax**. The subtype head trains on **all fitting images, including benign subtypes**, and is not a hard-routed four-class malignant expert.

Kernel L2 regularization **1e-4** applies exactly to `patch_projection`, `s1_dense`, `s2_dense1` and `s2_dense2`. The notebook does not apply it to Transformer feed-forward Dense layers or output Dense layers. Preserve this distinction; the regularizer is not AdamW weight decay.

## Training and validation reuse

Compile with Adam learning rate **1e-4**, sparse categorical cross-entropy for each output, binary loss weight **.3** and subtype loss weight **.7**, plus the specified regularization. There are no manual class/case weights or new malignant-only losses. Batch size is 32 and maximum duration is **50 epochs per variant**.

Preserve callback order and monitor `val_subtype_output_accuracy`, mode `max`:

- ModelCheckpoint saves the best whole-model `.keras` checkpoint; equal scores retain the earlier best.
- ReduceLROnPlateau uses patience four, factor .5, minimum learning rate 1e-7, default min_delta 1e-4 and cooldown zero.
- EarlyStopping uses patience twelve, min_delta zero and `restore_best_weights=True`.

The scoring path reloads the selected checkpoint as authored. Report every completed epoch, stop reason, selected epoch, learning-rate history, training/validation losses, actual runtime and model/checkpoint hashes. Keep both variants regardless of which is better. Their up-to-fifty validation checkpoint opportunities and callback adaptation are part of selection history; no additional epoch, optimizer, split, seed or threshold search follows this registration.

## Compatibility and metric corrections

Replace Kaggle-specific dataset, CSV and checkpoint paths with audited task paths without changing ordered membership. Repair `TransformerEncoderBlock.get_config()` to preserve **d_model, num_heads, dff and dropout_rate**, using stored constructor arguments rather than assuming MultiHeadAttention exposes a public `key_dim` attribute. These repairs must preserve the instantiated architecture and be covered by synthetic serialization/reload checks. They do not authorize a model redesign.

Preserve and name the notebook's binary accuracy, benign/malignant recall and AUC; eight-subtype accuracy, macro precision/recall/F1, authored quadratic-weighted kappa, macro one-versus-rest AUC and class confusion/recall. The nominal eight-class order must be saved with every artifact.

The notebook's variable **`mal_sub_recall` is ungated**: it counts correct eight-subtype predictions among true-malignant validation images without checking the binary prediction. Its comments incorrectly describe this as a strict cascade. Keep the authored number but call it **ungated true-malignant subtype accuracy**. The computed list of DC/LC/MC/PC abbreviations is not actually used to filter its prediction or denominator.

Add separately labeled strict diagnostics without replacing the authored measure:

- **Strict malignant cascade recall:** correctly predicted malignant route **and** correct subtype, divided by all true-malignant validation images.
- **Strict joint eight-class correctness:** binary label **and** subtype both correct, divided by all validation images.

These are additional named counts on the same predictions, not a retrained five-category cascade or a new score selected to improve the result. Published-paper numbers and historical baseline strings printed by the notebook are contextual author claims with different protocols. Do not endorse its automatic “Beat?” flags, novelty statement or attention-based reasoning language as results of a controlled comparison.

## Visualization is not attention rollout

The authored visualization takes L2 norms of the last block's spatial token outputs and min-max normalizes a 7×7 grid. It neither extracts attention probabilities nor multiplies attention matrices across layers, so call it a **token-norm proxy**, not attention rollout or causal attribution.

Its visualization cells pass raw image pixels to the model. In faithful, scoring uses divided pixels, so visualization predictions/maps use a different input scale from the reported scored predictions. Preserve this authored discrepancy and disclose it. In input_fixed those scales align, but the map still is not attention rollout and cannot establish which tissue caused a prediction.

## Execution boundary and current status

Use a separate environment with **TensorFlow 2.15.1 / Keras 2.15.0**, preserving the existing project environment. Before GPU work, accept the zero-fit membership/order/split audit, isolated runtime/pretrained-weight receipts and synthetic CPU model/serialization check. Synthetic checks certify mechanics only; no notebook performance is already saved.

The bounded production budget is **one GPU per variant, at most two concurrently, two CPUs and 32 GiB host memory per task, three hours per variant**: at most six requested production GPU-hours. Use scheduler jobs and retain actual queue/allocation receipts. At registration, TReNDS reports maintenance; a public CPU queue is available for cache preparation and the approved account check permits qGPU48. The scheduler's observed estimate around 11:25 a.m. Eastern is changeable and is not a promised start or completion time. Recheck actual policy/allocation when submitting; leave unrelated jobs alone.

There are **no variant accuracy results, completed production claims or guaranteed ETA on this page**. Completion requires saved checkpoint replay, exact data/label/split and source/configuration receipts, complete histories and both variants' honestly labeled scores. The run can assess this supplied eight-subtype image recipe; it cannot establish the unachieved five-category 94–95% provider-case target, clinical validation, novelty or paired patient-level superiority.
