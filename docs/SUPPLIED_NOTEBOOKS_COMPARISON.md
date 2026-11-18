# Supplied CNN–Transformer and Swin notebooks

The later case-matched comparison and post-hoc five-label scoring of the originally saved image predictions are available in the [consolidated model comparison](BREAKHIS_MODEL_COMPARISON.md). The original notebook results and training protocols below are preserved.

November 17, 2026. **Both supplied-notebook experiments are complete and independently accepted**, including both registered CNN input variants. Swin reaches **93.38% binary accuracy and 64.37% final five-category image accuracy**. The existing multi-view capstone remains parked; its completed results remain preserved and are compared separately below.

## Three approaches at a glance

The main-project column describes its retained reference P. Other completed main-project models, including the UNI2-h learned-metric comparison, remain separate entries in the preserved scorecard.

| Item | Main capstone, retained P | Supplied CNN–Transformer | Supplied Swin, default `dual` |
| --- | --- | --- | --- |
| Input and prediction unit | Available images of one provider case, summarized into one case answer | One image, one answer per head | One image, one routed final answer |
| Final output | Five categories: benign stops; malignant proceeds to four cancer types | Benign/malignant plus eight tissue types, including benign subtypes | Five categories: benign stops; malignant proceeds to four cancer types |
| Role counts | 46 fitting / 8 validation / 28 excluded provider cases per trial; five trials in each of two arrangements | 6,327 fitting / 1,582 reused validation images; all 82 provider keys on both sides | Literal split: 5,104 fitting / 689 validation / 2,116 test images; 51 / 7 / 23 bare IDs; conservative-stem overlap remains |
| Zoom handling | 40×/100×/200×/400× case views; all 15 nonempty subsets evaluated | Four zooms pooled as separate images | Four zooms pooled as separate images |
| Model | Frozen Midnight-12k reader, three crops per image; decision-specific centers and class-balanced cosine-density router/specialist | ImageNet EfficientNetV2B0 plus four added Transformer blocks; binary and eight-type heads | ImageNet Swin-Tiny; binary and malignant-four heads |
| What learns from BreakHis | Fitting labels define support distributions and centers; validation selects temperature. No encoder or classifier-coefficient optimization in retained P. | Last 60 backbone layers, added Transformer blocks and both heads train jointly | Last two Swin stages and both heads train; malignant loss is masked for benign images |
| Current status | Completed and parked; five-category case accuracy 76.33% / 75.67% across all 15 subsets | Both runs accepted; corrected binary 97.09%, eight-type 86.41% | Accepted; binary image accuracy 93.38%, final five-category image accuracy 64.37% |

These columns describe different prediction units and splits. Their headline accuracies cannot establish a matched winner. In particular, CNN's 97.09% binary result is not five-category accuracy.

In plain language, the CNN notebook learns from individual pictures and gives two answers: benign/malignant and one of eight tissue types. Its corrected input scale gives **97.09% for the broad question and 86.41% for the eight-type question**. Swin answers benign first and, only if malignant, chooses one of four cancer types. It gets the broad question right **93.38%** of the time, but the complete five-category answer right **64.37%** of the time. Its main weakness is naming the cancer type. Its five final labels match the capstone's hierarchy, while its image-based input and split still differ from the capstone.

## Supplied notebooks in detail

| Item | Supplied CNN–Transformer | Supplied Swin, default `dual` run |
| --- | --- | --- |
| Input unit | One original tissue image | One original tissue image |
| Zooms | All 40×, 100×, 200× and 400× images pooled as separate samples | Same four zooms, pooled as separate samples |
| Data coverage | Audited 7,909 images | Audited 7,909 images; membership and split accepted |
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
| Actual completed fitting | Faithful 13 epochs, selected 1; corrected 50 epochs, selected 41 | 17 epochs; selected epoch 7 |
| Current status | Both saved-prediction checks accepted; selected checkpoints reload exactly on fixed probe | Saved-prediction check accepted; selected weights reload within the fixed tolerance |

Both use training rotations up to ±90°, horizontal/vertical flips and reflection fill. Neither notebook combines the different magnifications of a provider case into one multi-view prediction or evaluates all fifteen nonempty zoom subsets. Swin's optional `single5` mode and repeated seeds are additional notebook capabilities, not completed runs or results in this comparison. [Fixed Swin execution and split limitations](SUPPLIED_SWIN_RUN.md).

## Measured results

| Metric | CNN faithful | CNN input corrected | Swin dual |
| --- | ---: | ---: | ---: |
| Binary accuracy |68.58% (1,085 / 1,582)|97.09% (1,536 / 1,582)|93.38% (1,976 / 2,116)|
| Benign recall |0.00% (0 / 497)|96.58% (480 / 497)|86.25% (508 / 589)|
| Malignant recall |100.00% (1,085 / 1,085)|97.33% (1,056 / 1,085)|96.14% (1,468 / 1,527)|
| Eight-type accuracy |43.62% (690 / 1,582)|86.41% (1,367 / 1,582)|Different target; no eight-type head |
| Eight-type macro F1 |.075924|.842481|Different target |
| Ungated subtype accuracy among true malignant images |63.59% (690 / 1,085)|85.99% (933 / 1,085)|Not reported |
| Strict malignant cascade recall |63.59% (690 / 1,085)|85.53% (928 / 1,085)|55.93% (854 / 1,527)|
| Subtype accuracy among correctly routed true-malignant images |Not reported |Not reported |58.17% (854 / 1,468)|
| Five-category end-to-end accuracy |Not reported |Not reported |64.37% (1,362 / 2,116)|
| Five-category macro F1 |Different target |Different target |.441602|

The faithful CNN predicts malignant and ductal for every checking image. Its apparently moderate accuracies are majority-label frequencies. The declared input correction improves the measured performance substantially in the fixed pair; both versions remain in the record. The corrected 97.09% measures only benign/malignant. A malignant image with the wrong cancer subtype can still count as correct in that binary score.

The CNN notebook's original malignant-subtype score does not check the first decision. Strict cascade recall does: its numerator requires the correct malignant route and subtype, and its denominator is all 1,085 true-malignant checking images. This remains different from five-category accuracy, which also counts every benign image in its denominator. [Complete CNN status](SUPPLIED_CNN_TRANSFORMER_STATUS.md), [accepted numerical evidence](evidence/SUPPLIED_CNN_TRANSFORMER_RESULTS.json).

Swin correctly routes 1,468 of 1,527 malignant test images, but gives only 854 their correct cancer subtype. Together with 508 correctly recognized benign images, that produces 1,362 correct five-category answers out of 2,116. Its final-label recall is **86.25% benign, 72.47% ductal, 41.62% lobular, 19.54% mucinous and 0.72% papillary**. Only **one of 139 papillary images** receives the correct final answer. The strong binary result therefore coexists with weak malignant typing. [Accepted Swin metrics, class confusion and per-zoom results](evidence/SUPPLIED_SWIN_RESULTS.json).

Swin's secondary majority vote reaches **75.00% five-category accuracy (18 / 24)** and **100.00% binary accuracy (24 / 24)**. Those are votes over **bare filename ID plus true-label groups**, giving 24 groups from 23 bare IDs. They are secondary summaries of the same test-image predictions, rather than 24 established independent clinical cases or the main capstone endpoint. They do not replace the image-level results above.

## How the parked capstone differs

The main capstone first pools the available pictures/views of a **provider case**, predicts benign/malignant and then classifies malignant cases into four types. It evaluates all fifteen available-zoom subsets in each of five overlapping trials per arrangement. The retained Midnight reference P has **76.33% / 75.67% five-category accuracy** in arrangements 812/813, with **85% / 85% in the separate all-four-view condition**. The UNI2-h learned-metric primary has **76.17% / 75.83%**, a separate completed comparison. These are different from the two CNN variants and are not seeds of either supplied notebook. [Full capstone scorecard](../paper/breakhis-five-class-report.md), [task contract](DUAL_STEP_SCOPE.md).

The capstone uses 46 fitting / eight validation / 28 excluded provider-role cases per trial and reports reused development performance. All 82 provider keys entered earlier development somewhere, and clinical identity remains unresolved. Swin's test split was held out from its particular fit and checkpoint selection, but it cannot become a globally fresh project cohort retrospectively. Its image-weighted accuracy also cannot be ranked directly against the main case/subset average.

## Execution checks and interpretation limits

The Swin notebook describes its split as patient-level and directly comparable with the Lang et al. paper. Those remain authored claims. The accepted audit finds four conservative identifier stems crossing roles: two fitting/test, one validation/test and one fitting/validation. The declared roles were preserved with that grouping limit disclosed. Its authored validation quotas omit lobular, adenosis and phyllodes identifier groups; there are no lobular validation images. Its secondary vote groups by identifier **and true label**, giving 24 test voting groups from 23 bare test IDs. This is not 24 established independent patients. Production membership, pretrained-weight revision, saved-prediction metrics and checkpoint reload all passed the registered acceptance checks.

Swin's declared compatibility adaptations use the pinned official **native TensorFlow weights**; equivalence to an implicit PyTorch-to-TensorFlow conversion is not established. Loading omits twelve deterministic `relative_position_index` buffers. Every actual buffer passed exact expected 49×49 matrix, int32 dtype and nontrainability checks; no unresolved missing learned parameter, mismatch or loading error remained. A separate graph-mode window-reversal failure required exact integer window counting, `(height // window_size) * (width // window_size)`, in place of floating division and truncation. The repair preserves the original reshapes, transposes and learned operations. Reconstruction, eager/graph and gradient checks across sixteen spatial/batch cases all had maximum difference 0.0; the synthetic training/reload check passed before production. The isolated parity check did not reproduce the original graph failure, so the proposed floating-point cause remains a diagnosis rather than a confirmed trace. Earlier failed technical attempts remain preserved and supply no classification metrics. [Compatibility review](reviews/SUPPLIED_SWIN_REVIEW.md), [fixed execution](SUPPLIED_SWIN_RUN.md).

Swin completed seventeen epochs and selected epoch seven. Production job 4985804 completed successfully in 16m35s; independent CPU checker 4985805 completed successfully in 12s, without new fitting or inference. The selected weights' fixed 32-image reload probe differed by at most 6.86e-7 for binary probabilities and 3.88e-7 for subtype probabilities, below the registered 1e-5 tolerance. Numerical source: `a50aeee39a78b3210a5a8c33905f6566a0563875`.

Both notebooks' maps use feature/token norms, which are qualitative proxies rather than attention rollout or proof of a causal tissue region. CNN retains its declared path, serialization and official-cache repairs and the one declared scaling companion. Preserve the exact private originals and every outcome; do not promote printed paper “Beat?” flags or historical hardcoded numbers into verified results.

The completed comparison shows strong benign/malignant recognition alongside weaker full subtype performance in the input-corrected CNN run and Swin run, each under its own protocol. CNN's eight-type image-random result, Swin's five-category grouped-image result and the main project's case/subset result answer different questions. They do not establish a head-to-head model ranking. No result here establishes clinical validation, a new method, matched paper superiority or 94–95% on the capstone's five-category provider-case endpoint. Both CNN outcomes and the accepted Swin outcome remain in the record; no score-driven new split, extra seed or additional fit follows from this comparison.

CNN original SHA256: `2bb8eac92b33ad81c8a203743d202f8a1866e9cdcf4f71c0259be69b6c9aa709`. Swin original SHA256: `fd88d851fbd8cf049a0ebbd718ed8757ffe28f629e2472262efca30bbca029af`. Original supplied notebooks, identifiers, split rows, predictions, weights and machine paths remain private.
