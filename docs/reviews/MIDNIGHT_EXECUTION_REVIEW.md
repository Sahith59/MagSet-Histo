# Midnight preparation and extraction implementation review

November 15, 2026. Scope: `scripts/prepare_midnight.py`, `magset/midnight_features.py`, and their tiny fixture tests. This review does not claim real checkpoint loading, completed extraction or model performance. The scheduler technical smoke must establish those runtime facts.

## Pin and dependency contract

The implementation accepts only `kaiko-ai/midnight` revision `adc6b15679c981cce6f9b018bbad09d16eeeda9f`. It verifies all three official artifacts before model construction:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 879 | `0362e014a3f3b1ad2f2faf115d21cbc05ed48b3a6114bdb4ddec897bc83cc0fd` |
| `README.md` | 8,080 | `7595d9ac11154d804a9e920abae185bd98798ebd4ba375490f1a845dc32d2001` |
| `model.safetensors` | 4,546,005,432 | `52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1` |

Sources: [pinned official model card](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/README.md), [official configuration](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/config.json), and [official checkpoint metadata](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/model.safetensors). Preparation uses only pinned public anonymous HTTPS paths and standard-library HTTP. It neither logs in nor accepts model access terms.

Inference uses the existing installed **Transformers 4.51.3** implementation of standard `Dinov2Config`/`Dinov2Model`, **safetensors 0.5.3**, and the registered Torch/Torchvision/Pillow versions. See [official versioned DINOv2 source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/dinov2/modeling_dinov2.py). There is no `trust_remote_code`, unpinned Torch Hub import, timm dependency, downloaded executable model code, or environment mutation. Runtime versions must match the canonical config exactly. Metadata also hashes the actual installed model/configuration source files.

## Construction and feature contract

The official model config is parsed locally. A standard model is constructed on Torch's meta device, avoiding an initialized duplicate of the billion-parameter model in host memory. The official safetensors state is loaded on CPU, every tensor must be FP32, and `load_state_dict(strict=True, assign=True)` rejects missing keys, extra keys and shape differences. The temporary state mapping is deleted before transfer to CUDA. Every model parameter must be materialized FP32; the model runs frozen in evaluation/inference mode.

The transform follows the pinned official example: RGB image, shorter-side resize to 224 using bilinear interpolation/antialiasing, center crop 224, tensor, normalization by channel mean/std .5. The official architecture's `image_size: 518` is preserved as model configuration; it is not substituted for the documented 224 inference input. FP32 eager attention is explicitly selected, AMP is unused and TF32 is disabled. No model adaptation or label fitting occurs.

At 224 pixels with patch size 14, the forward must produce finite FP32 tokens of shape `[batch, 257, 1536]`. The saved feature concatenates CLS and the mean of all 256 image-patch tokens, yielding exactly `[batch, 3072]`.

## Artifact and runtime safeguards

Both command entry points require a scheduler allocation. Preparation constructs no model and performs no forward. Its `.part` downloads resume only when HTTP range start/total and response lengths agree. A server ignoring Range triggers a full-body restart rather than corrupt concatenation. Publication requires the exact registered length and SHA256; existing complete artifacts must already match. Retries are bounded to three attempts.

Extraction verifies the full 7,909-row manifest hash before choosing a magnification shard. It preserves manifest path ordering and rejects escaping/absolute paths. Production uses all shard images; the only limited mode is a four-image technical smoke, which carries `smoke_limit: 4` and must use a separate output root. Existing complete outputs are accepted only after provenance, byte hash, exact ordered paths, dimensions, FP32 dtype and finiteness are rechecked; incomplete/mismatched outputs are refused.

NPZ files retain the project's `paths`/`features` schema and compatible encoder metadata. Publication uses exclusive temporary files and atomic no-overwrite links. Every newly written NPZ is actually reopened and compared exactly. Metadata records source/config/manifest/encoder hashes, precision, dependencies, GPU memory, timing and scheduler job ID. Raw image paths remain in private feature archives.

On the first batch, the runtime repeats the batch and compares its first image against a separate single-image forward. The four-image smoke additionally destroys and reconstructs the full model from the verified checkpoint and compares saved features. Fixed tolerances are `atol=rtol=1e-4`; actual maximum errors are recorded. This bounds numerical variation; it is not a classification-performance check.

## Validation and remaining acceptance

Ten tiny fixture tests passed locally. They cover schema mutations, resize/crop/normalization, CLS-plus-patch pooling, invalid tokens, strict state loading into a tiny meta linear module, parity rejection, actual NPZ reload/overwrite/coverage guards, relative paths, valid HTTP resume, ignored ranges and corrupt downloads. No real checkpoint or billion-parameter model was constructed locally. The official `v2.ToTensor()` call emits a deprecation warning with the local torchvision version; it is retained to match the pinned model card.

Before full extraction, the scheduled four-image CUDA smoke must verify actual whole-checkpoint strict compatibility, finite outputs, all three recorded parity checks, pinned runtime versions and memory/time. Only then should the four bounded magnification shards run. A separate full-cache audit must establish 7,909 exact unique paths, four complete shards and `load_bags` compatibility before any classifier fitting. No score or 94–95% claim is supported by these implementation checks.
