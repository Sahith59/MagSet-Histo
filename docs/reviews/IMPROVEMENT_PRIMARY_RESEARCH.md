# Fast bounded BreakHis improvement: primary-source audit

Inspected November 15, 2026. This is a recommendation, not an experiment registration or a measured improvement. No weights were downloaded, accounts changed, access terms accepted, or cluster jobs submitted for this audit.

## Recommendation

**Best next path: one frozen Midnight-12k encoder comparison, using the existing five-category mean-pooling/direct-versus-two-step protocol.** It is a genuinely accessible alternative pathology representation, with a modest one-time extraction cost. Keep its backbone frozen. The recent EfficientNet fitting/validation gap and only two lobular and three papillary fitting cases make another large supervised fine-tune a weak first use of the remaining three days.

**Backup: a case-balanced field-level logistic head on the already saved Phikon-v2 image features, averaging probabilities after classification.** This tests whether classifying individual fields before pooling helps, without downloading another model or executing an encoder. Register it separately and use it if the new encoder cannot clear its technical checks promptly, or as the single remaining follow-up after the encoder comparison. Neither path promises 94–95%.

The retained verified two-step scores remain 65.83%/68.83% accuracy and .497517/.532497 macro F1. EfficientNet gave 46.50%/53.33% and .234635/.310818. These are dependent provider-case development-validation results, not fresh clinical-patient test results. See [completed results](../BREAKHIS_CASEBAG_RESULTS.md) and [five-category baseline](../BREAKHIS_FIVE_CLASS_RESULTS.md).

## Public availability: permissive license does not imply ungated files

Anonymous official Hub metadata requests and model pages were inspected. The candidate access findings are:

| Official model | Observed access | License/access consequence for this deadline |
| --- | --- | --- |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | `gated: "auto"`; information-sharing/conditions screen | Apache-2.0, but accepting conditions is still required. It is not a no-agreement download path. |
| [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | `gated: "manual"`; institutional-email approval | CC BY-NC-ND 4.0; noncommercial research, attribution and restrictions on sharing derivatives. Do not make an access application the critical path. |
| [H0-mini](https://huggingface.co/bioptimus/H0-mini) | Official page requires institutional-email approval and acceptance | CC BY-NC-ND 4.0; also not an immediate ungated alternative. |
| [Hibou-L](https://huggingface.co/histai/hibou-L) | `gated: "auto"`; information-sharing screen | Apache-2.0, but still requires an access action. |
| [PathOrchestra](https://huggingface.co/AI4Pathology/PathOrchestra) | Official policy requires an academic access request | CC BY-NC-ND 4.0; not the present no-application route. |
| [Midnight-12k](https://huggingface.co/kaiko-ai/midnight) | `gated: false`, `private: false`; anonymous checkpoint HEAD returned HTTP 200 | MIT. No separate model-access approval/acceptance screen was observed. Preserve the applicable copyright/license notice. |

The official [Midnight repository license](https://github.com/kaiko-ai/Midnight/blob/e48dce1567a1db3c9cfedecace6f20262c07d5aa/LICENSE) supplies the MIT terms. The [Hub API](https://huggingface.co/api/models/kaiko-ai/midnight?blobs=true) supplied the access flag, current revision, exact checkpoint length and LFS digest. The checkpoint was checked with an anonymous **HEAD request only**, without transferring its body. This is evidence of public file access, not evidence that extraction has run on the university cluster.

## Exact Midnight candidate and implementation contract

| Item | Pinned value / required behavior |
| --- | --- |
| Repository | `kaiko-ai/midnight`, the public **Midnight-12k** release; not restricted Midnight-92k or Midnight-92k/392 |
| Hub revision | `adc6b15679c981cce6f9b018bbad09d16eeeda9f` |
| Author source revision | `kaiko-ai/Midnight` at `e48dce1567a1db3c9cfedecace6f20262c07d5aa` |
| Architecture | ViT-g/14; official config has 40 layers and hidden width 1536 |
| Checkpoint | `model.safetensors`, 4,546,005,432 bytes (about 4.23 GiB) |
| Expected SHA256 | `52c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1` |
| Input | RGB; official example: resize shorter side to 224, center crop 224×224, tensor, channel means/stds each `(0.5, 0.5, 0.5)` |
| Classification representation | Concatenate final CLS token and mean of final image-patch tokens: 1536 + 1536 = **3072**, not CLS alone |
| Loading | Standard Transformers `AutoModel`, pinned `revision`; do not copy Phikon's processor or infer a 518-pixel input from the architecture config |
| Pretraining | Author-reported TCGA FFPE slides; pretrained transfer learning, not permission to add TCGA or BRACS fitting/evaluation data now |

Sources: [pinned model card](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/README.md), [configuration](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/config.json), [checkpoint metadata](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/model.safetensors), and [author embedding extractor](https://github.com/kaiko-ai/Midnight/blob/e48dce1567a1db3c9cfedecace6f20262c07d5aa/eval/extract_cls_token.py). The stated 3072 dimension follows directly from the official concatenation and configured width. Pin the working Transformers/Torch/Torchvision versions after the technical smoke; the card is not a cluster compatibility guarantee.

The [author paper](https://arxiv.org/html/2504.05186v1) describes multiscale pretraining. This makes the representation a plausible all-magnification candidate; it does not establish robustness to our 15 missing-view combinations. Use all original 40×/100×/200×/400× fields and retain their recorded magnifications. Do not fabricate physical field alignment or calibrated microns per pixel.

## The published BreakHis score is a substantially different task

The author's table reports **.840 balanced accuracy for Midnight-12k versus .725 for Phikon-v2**. This is contextual support for testing a different representation, not an estimate for our five-label endpoint. The pinned implementation reveals a particularly consequential mismatch:

- The [Midnight evaluation configuration](https://github.com/kaiko-ai/Midnight/blob/e48dce1567a1db3c9cfedecace6f20262c07d5aa/eval/configs/vision/pathology/offline/classification/breakhis.yaml) has a four-output linear head.
- Its [pinned EVA loader](https://github.com/kaiko-ai/eva/blob/3e902dd5878d927126ca230e681e5beae8dffae4/src/eva/vision/data/datasets/classification/breakhis.py) selects **TA, MC, F, DC**, defaults to **40×**, and expects **1,132 training / 339 validation images**. It uses fixed patient-ID membership to form those roles.
- That loader excludes **lobular and papillary carcinoma**, precisely two scarce types that need improvement here. It also excludes other original subtypes. Its image-level score is not an all-four-view, five-category case score. The paper's full-dataset summary should not override what the pinned loader actually selects.

No 94–95% prediction follows from this benchmark. No claim of superiority over H-optimus, UNI or other models in our protocol follows either.

## Proposed bounded run, to register before fitting

Preserve the 46 fitting / 8 validation / 28 excluded provider-key roles for all five original trials and both 812/813 arrangements. Fit only on the assigned BreakHis fitting cases. Keep all related fields and magnifications together. Use the same final five labels, 15 subset endpoints, fit-only normalization, class balancing, C values `.01/.1/1/10`, hard routing at .5 (tie malignant), direct-five control and macro-F1 selection rule as the existing comparison. The intervention is the frozen representation. Dimension changes to 3072 require new config/cache hashes and dimension-aware code; never overwrite the accepted Phikon cache or its locked experiment.

1. Technical smoke: at most 64 fitting images covering every magnification; verify exact transform, feature dimension/finiteness, deterministic ordering, single/batch agreement within recorded tolerance, save/reload and checksum receipts. It must not inspect classification outcomes to choose preprocessing or precision.
2. Cache exactly 7,909 images once, in four magnification shards. Suggested ceiling: **one scheduler GPU for two hours**, with a bounded small-batch smoke to select a safe batch size. Any available authorized GPU is suitable for testing feasibility; there is no reason to wait specifically for an H100. Start conservatively and record measured peak memory/throughput. Two hours is a proposed allocation cap, not a runtime claim. Use FP32 for the first reproducible reference unless a mixed-precision recipe is fixed and validated before extraction.
3. Run the same ten paired direct/dual comparisons: **120 total logistic head fits**, four complete candidates per arm/run. Proposed CPU ceiling: **two hours total**; no unbounded retries or extra hyperparameter search. Scoring and artifact review follow. Encoder inference has no optimizer state, and the 3072-dimensional full feature matrix is only about 93 MiB before compression/metadata.
4. Report all paired outcomes, accuracy, macro F1, balanced accuracy, class recall, all-four and subset results, binary routing and true-malignant conditional typing. Evaluate improvement against the retained Phikon two-step result under the registered both-arrangement gate; preserve negative outcomes. Do not select a favorable trial/arrangement or use excluded-case outcomes for recipe changes.

This leaves day two for the registered backup only if justified, verification and packaging; day three remains for the report, five-category deck/demo and rehearsal. A larger GPU cannot create additional independent lobular or papillary fitting cases.

## Backup using the existing Phikon cache

The current [five-class code](../../magset/five_class.py) takes each view's **mean field embedding before** fitting a logistic head. The [bag loader](../../magset/bags.py) retains one feature row per original image, so the raw field-level feature information is still available. Its encoder is `owkin/phikon-v2` at `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`, using the 1024-dimensional CLS representation and official pinned processor; [the model card](https://huggingface.co/owkin/phikon-v2) explicitly supports frozen features and tile-level probes.

For one separately registered backup, fit direct-five and binary/malignant-four logistic heads on those field embeddings, inheriting case labels and using inverse field-count weights so every fitting case contributes equal mass at each magnification, with class balance defined over fitting cases. At inference, average **probabilities** within each magnification and then equally across available magnifications; apply the binary routing decision only after that aggregation. The malignant head fits only true malignant fitting cases. Keep the .5 routing threshold and the same four shared complete-pipeline C candidates; no additional learned attention module or threshold sweep.

This changes the order of nonlinear classification and pooling. In general, averaging softmax/sigmoid outputs differs from applying softmax/sigmoid to a mean embedding. Merely averaging linear logits would largely restore the old mean operation and would not test the intended hypothesis. Fit-only normalization and missing-view/permutation invariance remain mandatory. Ceiling: **zero encoder forwards, zero new weights, 120 CPU head fits, two CPU hours**. The earlier eight-class auxiliary field readouts failed to rescue their experiment; this backup is therefore a modest five-category hypothesis, not a proven fix.

No access applications, external contacts, large supervised fine-tuning sweep, new dataset or altered split is needed for either recommendation. Even a measured improvement would remain development validation; the current provider grouping and reused cohort cannot establish final clinical generalization.
