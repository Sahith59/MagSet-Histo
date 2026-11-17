# Supplied Swin notebook: independent protocol review

November 17, 2026. This review covers the privately preserved, previously unexecuted notebook with SHA-256 `fd88d851fbd8cf049a0ebbd718ed8757ffe28f629e2472262efca30bbca029af`, the [registered recipe](../SUPPLIED_SWIN_RUN.md), its [configuration](../../configs/supplied-swin-v1.json), and the execution adapter. It accepts the bounded default dual-model experiment as a separately reported notebook execution with declared adaptations. It does not establish training success, model accuracy, clinical independence, paper superiority or 95% performance. Runtime and final-result acceptance require the technical smoke and independent saved-artifact checker.

## Literal model and selection

Each image is an individual example. All four magnifications enter the same image classifier; this is not multiview case aggregation or evaluation of 15 available-view subsets. ImageNet Swin-Tiny supplies a pooled 768-dimensional description. Patch embeddings and encoder stages 0–1 are frozen; stages 2–3, the added normalization and both heads train.

The binary head predicts benign/malignant. The four-type malignant head receives image features and stopped-gradient binary probabilities. Its cross-entropy ignores benign targets. Both heads are evaluated for every image; the final prediction uses a hard cascade: malignant probability at least 0.5 selects the malignant-head argmax, otherwise benign. Stopping the binary-probability gradient protects that direct path into the binary head. The subtype loss still changes shared features and can affect later binary predictions.

The unchanged training recipe uses seed 812, batch 32, Adam at 1e-5, at most 40 epochs, training-image class weights and equal overall head-loss weights. Selection uses the first maximum of the validation mean of binary accuracy and end-to-end malignant-subtype correctness. Learning-rate and stopping callbacks observe that validation score; test outputs do not select the checkpoint.

## Independently reconstructed split

The literal split-seed-0 rules were independently replayed from the existing hash-audited canonical manifest without loading images or fitting a model.

| Role | Images | Bare filename IDs | Five-label support: benign / DC / LC / MC / PC |
| --- | ---: | ---: | --- |
| Training | 5,104 | 51 | 1,702 / 2,166 / 429 / 528 / 279 |
| Validation | 689 | 7 | 189 / 268 / 0 / 90 / 142 |
| Test | 2,116 | 23 | 589 / 1,017 / 197 / 174 / 139 |

No bare filename ID crosses roles. Four conservative identifier stems do cross: two training/test, one validation/test and one training/validation. Their clinical relationships remain unresolved; preserving the supplied roles is a declared choice, not evidence that they are independent patients. Validation contains no lobular images, so checkpoint selection cannot measure lobular performance. Earlier project development already involved all available provider keys, so this is not a globally untouched cohort.

One bare ID has images in two malignant subtypes and is kept together. The notebook's secondary voting groups by both ID and true five-category target, producing 24 test ID/target groups from 23 bare IDs. Report this as ID/target-group voting, with its denominator, rather than a single independent prediction per patient. Preserve the literal `value_counts().idxmax()` tie rule when replaying it.

## Metric and paper interpretation

Binary accuracy counts benign/malignant decisions. Final-five accuracy requires the complete routed label to be right. The notebook's malignant-subtype recall is image-weighted final correctness among all true malignant images, including routing errors. Its conditional subtype accuracy instead excludes malignant images routed to benign; those denominators must remain separate. Macro F1 and unweighted Cohen kappa are exported multiplied by 100. A synthetic six-example replay of the literal routing/metric functions agreed with hand-derived outcomes; it supplies no research-performance evidence.

The supplied Lang manuscript reports image-level results under a shuffled approximately 70/30 case protocol. Its published aggregate scores do not identify the exact examples, seeds and validation procedure used here. Matching malignant test quotas alone does not reproduce the paper folds. The notebook also adds vertical flips and applies rotation after center cropping, whereas the manuscript describes rotation/horizontal flipping before cropping. Thus its authored “directly comparable” and “Beat?” text must remain contextual author text, not an endorsed matched comparison. The paper's malignant-subtype figure is not a five-category overall accuracy. [Published work](https://doi.org/10.1117/12.2582299).

The last-cell token-norm heat maps measure feature magnitude. They are neither attention rollout nor validated causal tissue explanations.

## Execution compatibility and acceptance conditions

Static inspection confirms that TensorFlow 2.15.1 honors the legacy-Keras environment setting and that Transformers 4.46.3 imports `tf_keras` when available. The model exposes the embedding/stage attributes and pooled output used by the notebook. Set the environment before the first TensorFlow import and execute from cell 2 in a fresh kernel after setup. [TensorFlow source](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/python/util/lazy_loader.py), [Transformers source](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_tf_utils.py), [Swin implementation](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/swin/modeling_tf_swin.py).

The pinned repository contains official TensorFlow weights. The remote Transformers default can prefer PyTorch-format safetensors and convert them. Selecting the pinned local native TensorFlow file is an explicitly declared loading adaptation; cross-format numerical equivalence has not been established by this review. Record the revision, actual file hashes and loading diagnostics. [Official checkpoint files](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/tree/d00d478bfaf1f417d34a1186673ad4b4be264c51).

The initial adapter inspection found only declared path/loading replacements in original cells 2, 3 and 8. Patched scientific cells compile. Export/progress/reload instrumentation is appended; original losses, split quotas, preprocessing, callbacks and routing are preserved. Before production acceptance, the isolated runtime must exercise a masked-loss training batch and the exact `.weights.h5` save/reload path; that extension selects a distinct Keras weight-saving implementation. [Pinned Keras saving source](https://github.com/keras-team/tf-keras/blob/v2.15.0/tf_keras/saving/saving_api.py).

The accepted input audit must bind canonical membership, image bytes, original sorted order, labels and exact split indices. Revalidate those bindings before production execution. Retain complete and failed execution notebooks, history, checkpoint, probability outputs and source/configuration/environment receipts. Independently recompute all reported metrics, selected epoch, routing, per-magnification results and secondary voting before accepting measured outcomes. No score-driven resplit, threshold adjustment or additional arm follows from this execution.
