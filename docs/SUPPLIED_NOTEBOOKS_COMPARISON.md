# Supplied CNN–Transformer and Swin notebooks

November 17, 2026. The CNN–Transformer experiment is **complete in both registered versions**. The newly supplied Swin notebook has **no accepted measured outcome yet**. The existing multi-view capstone is parked while this separate supplied-notebook work is handled; its completed results remain preserved.

In plain language, the CNN notebook learns from individual pictures and gives two answers: benign/malignant and one of eight tissue types. Its corrected input scale gives **97.09% for the broad question and 86.41% for the eight-type question**. The Swin notebook is designed to answer benign first and, only if malignant, choose one of four cancer types. Its five final labels match the capstone's label hierarchy, while its image-based input and split still differ from the capstone. We must finish and check Swin before giving it a result or naming a winner.

## Side by side

| Item | Supplied CNN–Transformer | Supplied Swin, default `dual` run |
| --- | --- | --- |
| Input unit | One original tissue image | One original tissue image |
| Zooms | All 40×, 100×, 200× and 400× images pooled as separate samples | Same four zooms intended, pooled as separate samples |
| Data coverage | Audited 7,909 images | Literal reconstruction covers 7,909; production prefit acceptance pending |
| Preparation | RGB; resize 256×256; center crop 224×224 | Same resize and crop |
| Input scale | Faithful: divide by 255 before EfficientNetV2's built-in preprocessing. Corrected: remove that external division. | Divide by 255, then ImageNet channel mean/std normalization |
| Picture reader | ImageNet EfficientNetV2B0; last 60 backbone layers trainable | ImageNet Swin-Tiny, `microsoft/swin-tiny-patch4-window7-224`; embeddings and stages 0–1 frozen, stages 2–3 trainable |
| Feature processing | 49 spatial tokens projected to width 512, CLS/position tokens, four added Transformer blocks with eight heads each | Swin's shifted-window Transformer and pooled 768-dimensional feature; layer normalization |
| Output heads | Binary two-class head plus an eight-type head for all images | Binary two-class head plus four-class malignant head; benign examples have masked subtype loss |
| Connection between heads | Binary probabilities enter the eight-type head; subtype gradients pass through that connection | Binary probabilities enter the malignant head with `stop_gradient` at that connection; the shared reader still receives both losses |
| Final decision | Independently scored binary and eight-type outputs; extra strict diagnostic counts reported separately | Binary malignant probability ≥ .5 selects a malignant subtype; otherwise final answer is benign |
| Class labels | Adenosis, fibroadenoma, phyllodes tumor, tubular adenoma, ductal, lobular, mucinous and papillary carcinoma; binary label also learned | Pooled benign plus ductal, lobular, mucinous and papillary carcinoma |
| Fitting/checking split | Eight-type-stratified image-random 80/20, random state 42: 6,327 fitting / 1,582 reused validation images | Literal split seed 0: 5,104 fitting / 689 validation / 2,116 test images, grouped into 51 / 7 / 23 bare filename identifiers |
| Group separation | All 82 provider keys and all 70 conservative groups overlap fitting/checking | No bare filename ID crosses roles; four conservative identifier stems do. Clinical identity remains unresolved. |
| Selection images | The same 1,582 images choose the checkpoint, adapt learning rate, stop training and supply reported metrics | Separate validation group drives selection; declared test group is scored after checkpoint selection |
| Training | Adam 1e-4; batch 32; seed 812; maximum 50 epochs; binary/type losses .3/.7; no class weights | Adam 1e-5; batch 32; seed 812; maximum 40 epochs; binary/malignant losses .5/.5; fitting-derived class weights |
| Selection rule | Best validation eight-type accuracy; LR patience 4; early-stop patience 12 | Best average of validation binary accuracy and strict malignant subtype recall; LR patience 4; early-stop patience 10 |
| Actual completed fitting | Faithful 13 epochs, selected 1; corrected 50 epochs, selected 41 | Pending |
| Current status | Both saved-prediction checks accepted; selected checkpoints reload exactly on fixed probe | Supplied original unexecuted; production result pending |

Both use training rotations up to ±90°, horizontal/vertical flips and reflection fill. Neither notebook combines the different magnifications of a provider case into one multi-view prediction or evaluates all fifteen nonempty zoom subsets. Swin's optional `single5` mode and repeated seeds are additional notebook capabilities, not completed runs or results in this comparison. [Fixed Swin execution and split limitations](SUPPLIED_SWIN_RUN.md).

## Measured results

| Metric | CNN faithful | CNN input corrected | Swin dual |
| --- | ---: | ---: | ---: |
| Binary accuracy |68.58% (1,085 / 1,582)|97.09% (1,536 / 1,582)|Pending |
| Benign recall |0.00% (0 / 497)|96.58% (480 / 497)|Pending |
| Malignant recall |100.00% (1,085 / 1,085)|97.33% (1,056 / 1,085)|Pending |
| Eight-type accuracy |43.62% (690 / 1,582)|86.41% (1,367 / 1,582)|Different target; no eight-type head |
| Eight-type macro F1 |.075924|.842481|Different target |
| Ungated subtype accuracy among true malignant images |63.59% (690 / 1,085)|85.99% (933 / 1,085)|Pending |
| Strict malignant cascade recall |63.59% (690 / 1,085)|85.53% (928 / 1,085)|Pending |
| Five-category end-to-end accuracy / macro F1 |Not reported |Not reported |Pending |

The faithful CNN predicts malignant and ductal for every checking image. Its apparently moderate accuracies are majority-label frequencies. The declared input correction improves the measured performance substantially in the fixed pair; both versions remain in the record. The corrected 97.09% measures only benign/malignant. A malignant image with the wrong cancer subtype can still count as correct in that binary score.

The CNN notebook's original malignant-subtype score does not check the first decision. Strict cascade recall does: its numerator requires the correct malignant route and subtype, and its denominator is all 1,085 true-malignant checking images. This remains different from five-category accuracy, which also counts every benign image in its denominator. [Complete CNN status](SUPPLIED_CNN_TRANSFORMER_STATUS.md), [accepted numerical evidence](evidence/SUPPLIED_CNN_TRANSFORMER_RESULTS.json).

## How the parked capstone differs

The main capstone first pools the available pictures/views of a **provider case**, predicts benign/malignant and then classifies malignant cases into four types. It evaluates all fifteen available-zoom subsets in each of five overlapping trials per arrangement. The retained Midnight reference P has **76.33% / 75.67% five-category accuracy** in arrangements 812/813, with **85% / 85% in the separate all-four-view condition**. The UNI2-h learned-metric primary has **76.17% / 75.83%**, a separate completed comparison. These are different from the two CNN variants and are not seeds of either supplied notebook. [Full capstone scorecard](../paper/breakhis-five-class-report.md), [task contract](DUAL_STEP_SCOPE.md).

The capstone uses 46 fitting / eight validation / 28 excluded provider-role cases per trial and reports reused development performance. All 82 provider keys entered earlier development somewhere, and clinical identity remains unresolved. A new Swin test split can be held out from that particular fit and checkpoint selection, but it cannot become a globally fresh project cohort retrospectively. Its eventual image-weighted accuracy also cannot be ranked directly against the main case/subset average.

## Limits that govern the eventual comparison

The Swin notebook describes its split as patient-level and directly comparable with the Lang et al. paper. Those are authored claims, not accepted findings. Literal reconstruction finds four conservative identifier stems crossing roles: two fitting/test, one validation/test and one fitting/validation. Keep the declared roles and disclose that grouping limit. Its authored validation quotas omit lobular, adenosis and phyllodes identifier groups; there are no lobular validation images. Its secondary vote groups by identifier **and true label**, giving 24 test voting groups from 23 bare test IDs. This is not 24 established independent patients. Production membership, pretrained-weight revision and scoring/reload behavior still require acceptance before performance is reported.

Both notebooks' maps use feature/token norms, which are qualitative proxies rather than attention rollout or proof of a causal tissue region. CNN retains its declared path, serialization and official-cache repairs and the one declared scaling companion. Preserve the exact private originals and every outcome; do not promote printed paper “Beat?” flags or historical hardcoded numbers into verified results.

No result here establishes clinical validation, a new method, a matched paper superiority claim or 94–95% on the capstone's five-category provider-case endpoint. When Swin is accepted, add its exact image/group denominators, final five-category and binary scores, strict malignant recall, checkpoint/epoch and evidence link to the pending cells above while retaining these protocol distinctions.

CNN original SHA256: `2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`. Swin original SHA256: `fd88d851fbd8cf049a0ebbd718ed8757ffe28f629e2472262efca30bbca029af`. Original supplied notebooks, identifiers, split rows, predictions, weights and machine paths remain private.
