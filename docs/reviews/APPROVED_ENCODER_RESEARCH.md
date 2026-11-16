# Approved pathology encoders: fixed comparison

Reviewed November 16, 2026. The user reports approval for UNI2-h, Virchow2 and H-optimus; the present authenticated access audit confirmed all three exact repositories below. H-optimus **version 0** is the selected model. Access to version 1 was not confirmed. These are new, separately measured representations; none has a BreakHis result from this study yet. Earlier [model-search access findings](NEXT_ROUND_MODEL_SEARCH.md) describe the account state before this approval.

For a ten-year-old: the encoder is a trained picture reader that turns each microscope picture into a list of clues. We are trying three different readers while keeping the same sorting rules and the same practice/checking cases. The first sorter decides benign or malignant. Benign stops; a second sorter assigns malignant pictures to ductal, lobular, mucinous or papillary carcinoma. A better reader might produce better clues, but it can still miss an important clue or mistake one rare cancer for another. Testing establishes whether it helps.

## Why these three

UNI2-h offers a relatively compact 1536-number descriptor and broad H&E/IHC pretraining. The authors released UNI2-h in January 2025; it is distinct from the original UNI covered by the 2024 Nature Medicine paper. Its upstream benchmarks motivate transfer testing, but do not measure our five-category, provider-case, all-subset endpoint. [Author release and benchmarks](https://github.com/mahmoodlab/UNI)

Virchow2 is especially relevant to the hypothesis that features learned across resolutions may transfer better across our four views. Its paper describes 3.1 million slides, 225,401 patients, H&E/IHC and four training resolutions; 15% of slides were international consultation material. This does not establish correspondence between its scanner magnifications and BreakHis's optical magnification labels. Its pretraining KDE regularizer is also a different operation from our downstream similarity classifier. [Virchow2 author paper, sections 3 and 6](https://arxiv.org/html/2408.00738v3)

H-optimus-0 adds a 1.1-billion-parameter reader trained on more than 500,000 proprietary H&E slides. Its larger size is a reason to test another representation, not evidence it will beat the other readers here. The author's published evaluations concern other tissue/biomarker tasks. [H-optimus-0 author release](https://raw.githubusercontent.com/bioptimus/releases/main/models/h-optimus/v0/README.md)

All three comparisons are fixed before outcomes and all must be reported. A practical implementation order is UNI2-h, Virchow2, H-optimus-0, because UNI2-h is shallower and H-optimus-0 is largest. This is an engineering judgment, not a performance ranking. The pathology content is already learned upstream; this study freezes those weights and fits only the existing similarity readout on BreakHis.

## Artifact and loader contracts

The authenticated audit obtained these pinned metadata. The configs also bind the small configuration/model-card files and runtime versions. Downloads must verify the complete checkpoint hash before strict loading; access success alone is not a successful model construction.

| Reader and project config | Revision | Weight file | Bytes | SHA256 |
|---|---|---|---:|---|
| [UNI2-h](../../configs/uni2-h-v1.json) | `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` | `pytorch_model.bin` | 2,725,669,217 | `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e` |
| [Virchow2](../../configs/virchow2-v1.json) | `3158645804b69e3f3bc4439d4116edddf0840a72` | `model.safetensors` | 2,525,001,112 | `8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b` |
| [H-optimus-0](../../configs/h-optimus-0-v1.json) | `b145cc1e6c6b30d3251aa8b1f844e6974188a743` | `pytorch_model.bin` | 4,539,305,018 | `bf47b8467403079798661601b6ab06533be789da48d3f8b510aaefe2fbc55cd3` |

Official file inventories identify these formats; UNI2-h and H-optimus-0 do not list a safetensors alternative in the audited release. [UNI2-h files](https://huggingface.co/MahmoodLab/UNI2-h/tree/d517a8dd47902dd7c308b3c36f63bce47e7b9a43), [Virchow2 files](https://huggingface.co/paige-ai/Virchow2/tree/3158645804b69e3f3bc4439d4116edddf0840a72), [H-optimus-0 files](https://huggingface.co/bioptimus/H-optimus-0/tree/b145cc1e6c6b30d3251aa8b1f844e6974188a743)

| Model | Exact local timm construction | Frozen descriptor |
|---|---|---|
| UNI2-h | Base `vit_giant_patch14_224`; override image224, patch14, depth24, heads24, width1536, `init_values=1e-5`, `mlp_ratio=2.66667*2`, `num_classes=0`, `no_embed_class=True`, `SwiGLUPacked`, `SiLU`, eight registers, `dynamic_img_size=True`. | `model(image)` gives `[B,1536]`, the CLS descriptor. |
| Virchow2 | Base `vit_huge_patch14_224`; image224, `init_values=1e-5`, `num_classes=0`, four registers, `mlp_ratio=5.3375`, `global_pool=""`, `dynamic_img_size=True`, `SwiGLUPacked`, `SiLU`. | Output `[B,261,1280]`; concatenate token0 and the mean of tokens5 onward, yielding `[B,2560]`. Tokens1–4 are excluded registers. |
| H-optimus-0 | Base `vit_giant_patch14_reg4_dinov2`; explicitly `img_size=224`, `num_classes=0`, `init_values=1e-5`, `dynamic_img_size=False`. | `model(image)` gives `[B,1536]`, the CLS descriptor. |

The upstream examples support these readouts and model-specific arguments. The pinned timm base provides Virchow2's depth32/width1280/heads16 and H-optimus-0's depth40/width1536/heads24/SwiGLU; UNI2-h's overrides are essential because its named timm base has different defaults. [UNI2-h instructions](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2 instructions](https://huggingface.co/paige-ai/Virchow2), [H-optimus-0 instructions](https://huggingface.co/bioptimus/H-optimus-0), [timm 1.0.15 source](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.15/timm/models/vision_transformer.py)

The explicit H-optimus-0 image size matters for local loading: timm's DINOv2 base configuration has518×518 input. Its builder fills an unspecified model image size from that configuration even when pretrained loading is disabled. The Hub example receives224 from the H-optimus configuration; a local architecture name alone does not. Preserve224 positions and strict loading rather than resizing positional weights to mask a construction mismatch. [timm default configuration](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.15/timm/models/vision_transformer.py), [timm configuration builder](https://raw.githubusercontent.com/huggingface/pytorch-image-models/v1.0.15/timm/models/_builder.py)

Project extraction fixes FP32 parameters/input/output, eager attention, evaluation mode, no gradients, no autocast and TF32 disabled. The authors recommend FP16 autocast for Virchow2/H-optimus-0 throughput; that recommendation does not require FP16 inference. Their examples do not establish bitwise equivalence across precisions. Our four-field smoke must check finite descriptors, shape, repeat/single-image parity and a fresh strict reload before cohort extraction. No smoke can establish research accuracy.

## Preprocessing and scale

Every field becomes one RGB `[3,224,224]` tensor. UNI2-h and Virchow2 use the pinned model configuration's resolved timm evaluation recipe: shorter side224, center crop224, `crop_pct=1.0`, antialiasing enabled. UNI2-h uses bilinear interpolation; Virchow2 uses bicubic. Both use ImageNet mean `(.485,.456,.406)` and standard deviation `(.229,.224,.225)`. The UNI2-h Hub card's manual example omits the center crop, whereas its author repository includes it; the registered resolved transform removes that ambiguity. [Pinned UNI2-h configuration](https://huggingface.co/MahmoodLab/UNI2-h/blob/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/config.json), [pinned Virchow2 configuration](https://huggingface.co/paige-ai/Virchow2/blob/3158645804b69e3f3bc4439d4116edddf0840a72/config.json), [UNI author instructions](https://github.com/mahmoodlab/UNI)

H-optimus-0's official example starts with already prepared224 tiles at0.5µm/pixel and applies tensor conversion plus normalization. Our registered BreakHis adaptation resizes the shorter side to224 with bicubic interpolation, center-crops224 and applies mean `(.707223,.578729,.703617)`, standard deviation `(.211883,.230117,.177517)`. BreakHis's physical pixel scale is not established by those operators; do not describe this as matching native0.5µm/pixel. [Official H-optimus-0 input contract](https://huggingface.co/bioptimus/H-optimus-0)

These recipes differ from Midnight's bilinear center crop with mean/std`.5` and GenBio's full-image224 stretch with channel-concatenated descriptors. The comparison therefore tests each frozen reader together with its declared preprocessing/readout, rather than isolating architecture alone.

## Licenses and upstream overlap

UNI2-h and Virchow2 specify CC BY-NC-ND4.0 plus academic/noncommercial access terms, attribution and individual registration; the terms prohibit redistribution of the model weights. Keep checkpoints private and cite the authors in the capstone. H-optimus-0 uses Apache2.0; its gate still requires account agreement. Do not transfer version0's license to version1. [UNI2-h terms](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2 terms](https://huggingface.co/paige-ai/Virchow2), [H-optimus-0 license](https://raw.githubusercontent.com/bioptimus/releases/main/models/h-optimus/v0/LICENSE.md)

The stated upstream sources are proprietary MGB slides for UNI2-h, MSK/consultation material for Virchow2 and a proprietary collection for H-optimus-0. None of these reviewed sources establishes that our exact BreakHis images are absent from every upstream collection. The private corpora prevent an independent image-level deduplication here. Record overlap as **unresolved**, while keeping current fitting/evaluation BreakHis-only. Independent provider-case splits cannot repair unknown pretraining exposure. [UNI2-h data description](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2 data description](https://arxiv.org/html/2408.00738v3), [H-optimus-0 data description](https://raw.githubusercontent.com/bioptimus/releases/main/models/h-optimus/v0/README.md)

## Fixed scientific comparison

Use the retained C classifier: case-balanced common-center cosine KDE, temperatures`.01/.03/.1/.3`, the original46/8/28 fitting/validation/excluded roles, whole-query-case-out support/centering, direct-five and dual2+4 controls, all four magnifications and all15 nonempty subsets. Validation selects the temperature; excluded cases remain excluded from scored outcomes. Do not add crops, supervised centering variants, fusion or threshold searches in this study. Report all three readers and both arrangements, with raw accuracy, macro F1 and class recalls; report all-four-view results separately. The [accepted scorecard](../../paper/breakhis-five-class-report.md) remains the source of previous measured results.

Potential gains are a hypothesis. Rare malignant fitting cases, stain/scale differences, center-crop information loss and error propagation through the binary router can still dominate. A different pretrained reader does not increase the number of independent BreakHis cases. No reviewed official source demonstrates94–95% for this project's endpoint, and no new accuracy is claimed here.
