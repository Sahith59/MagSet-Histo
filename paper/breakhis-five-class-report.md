# BreakHis five-category dual-step project report

Working capstone report, updated November 16, 2026. **UNI2-h now has the highest observed arrangement-812 two-step macro F1, 0.664180, at 75.83% final five-category accuracy**, but scores 0.575175 / 75.17% on 813. Head-specific centering, P, retains the 813 two-step F1 lead at **0.602512 / 75.67%**; its 812 result is 0.602715 / 76.33%. Three-crop Midnight, F, retains higher 813 two-step raw accuracy at **76.83%**. GenBio's one-stage control reaches **77.83% / F1 0.640603** on 813. All remain on the scorecard; there is no uniform winner. **94–95% final accuracy is not achieved.**

The approved UNI2-h, Virchow2 and H-optimus-0 reader comparisons are complete and independently reviewed, alongside GenBio and whole-case mean/attention. UNI2-h passes the descriptive improvement screen on 812 but not 813; none of the new readers passes the complete two-arrangement screen. All classification runs and collectors in these studies have finished. [Approved-reader execution and outcomes](../docs/BREAKHIS_APPROVED_ENCODERS_RESULTS.md).

These are exploratory provider-case development results. Execution completion, predictive improvement and clinical/research claims are reported separately. All completed negative outcomes are retained; no novelty or fresh clinical generalization is established.

## Task, data and evaluation

Input is a provider case's images at its available 40×, 100×, 200× and 400× views. Output is pooled benign, ductal carcinoma, lobular carcinoma, mucinous carcinoma or papillary carcinoma. A two-step model first routes benign/malignant. Benign stops; malignant receives one of four subtype labels. Direct-five controls predict the same five labels in one step. Final accuracy counts both routing and subtype errors.

BreakHis supplies 7,909 images and 82 subtype-qualified provider keys. Only BreakHis images and labels are fitted/evaluated; external pretrained weights provide transfer learning. Original eight-class roles are reconstructed before benign-label pooling. Each run has 46 fitting, eight validation and 28 excluded cases, with all related fields/views kept under their provider key. Equally numbered fields across magnifications are not assumed physically paired.

Fitting supports are 11 benign, 25 ductal, two lobular, five mucinous and three papillary. Validation contains four benign cases and one per malignant type. Raw accuracy therefore weights benign at 50%; macro F1, balanced accuracy and individual recalls remain necessary. Every case is scored under all 15 nonempty view combinations. Headline metrics average those subsets and five overlapping trials equally, separately for arrangements 812 and 813. All-four-view results are a separate endpoint.

The arrangements contain 22 and 20 distinct validation keys, with 31 in their union. Clinical identity relationships remain unresolved. All 82 provider keys have entered development, so the 28 excluded cases per run are not a globally fresh confirmation cohort. Repeated fields, masks and trials do not create independent patients. TCGA follow-up remains deferred.

## Complete development scorecard

| Completed two-step pipeline | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| Original frozen Phikon case means | 65.83% / 68.83% | 0.4975 / 0.5325 |
| Case-supervised EfficientNet | 46.50% / 53.33% | 0.2346 / 0.3108 |
| A: frozen Phikon field supervision | 66.67% / 69.50% | 0.4975 / 0.5207 |
| B: frozen Midnight case means | 72.00% / 71.33% | 0.5558 / 0.5350 |
| B: frozen Midnight field supervision | 73.67% / 70.00% | 0.5732 / 0.5017 |
| C: frozen Midnight single-center similarity | 75.67% / 75.00% | 0.5897 / 0.5643 |
| D: Midnight LoRA recipe | 68.33% / 59.83% | 0.4944 / 0.3754 |
| E: fixed equal-probability ensemble | 74.17% / 72.17% | 0.5781 / 0.5326 |
| F: frozen Midnight three-crop similarity | 75.50% / 76.83% | 0.5925 / 0.5799 |
| N: frozen Midnight native global input | 60.00% / 61.00% | 0.4394 / 0.5014 |
| P: class-shared balanced centering | 76.17% / 75.33% | 0.5992 / 0.5778 |
| P: head-specific centering | 76.33% / 75.67% | 0.6027 / 0.6025 |
| GenBio frozen similarity | 70.50% / 75.17% | 0.5837 / 0.5813 |
| Whole-case learned mean | 70.17% / 71.83% | 0.5199 / 0.5094 |
| Whole-case gated attention | 71.67% / 70.50% | 0.5453 / 0.5234 |
| UNI2-h frozen similarity | 75.83% / 75.17% | 0.6642 / 0.5752 |
| Virchow2 frozen similarity | 72.50% / 73.00% | 0.5758 / 0.5387 |
| H-optimus-0 frozen similarity | 73.50% / 71.17% | 0.5097 / 0.5062 |

Direct-five controls provide the following same-target comparisons:

| Direct-five pipeline | Accuracy 812 / 813 | Macro F1 812 / 813 |
| --- | ---: | ---: |
| Original Phikon case means | 66.67% / 64.67% | 0.5161 / 0.5035 |
| C: Midnight single-center similarity | 75.00% / 73.33% | 0.5847 / 0.5713 |
| D: Midnight LoRA recipe | 71.33% / 67.50% | 0.5101 / 0.4147 |
| E: fixed ensemble | 75.17% / 73.33% | 0.5782 / 0.5314 |
| F: Midnight three-crop similarity | 75.00% / 73.33% | 0.5852 / 0.5758 |
| N: Midnight native global input | 59.50% / 56.83% | 0.4456 / 0.5038 |
| P: balanced-centering direct control, identical in both modes | 75.50% / 74.33% | 0.5887 / 0.5892 |
| GenBio frozen similarity | 71.33% / 77.83% | 0.5960 / 0.6406 |
| Whole-case learned mean | 71.17% / 71.50% | 0.5225 / 0.5028 |
| Whole-case gated attention | 73.00% / 70.83% | 0.5514 / 0.4953 |
| UNI2-h frozen similarity | 75.17% / 77.67% | 0.6486 / 0.6345 |
| Virchow2 frozen similarity | 72.83% / 75.00% | 0.5901 / 0.5877 |
| H-optimus-0 frozen similarity | 77.33% / 75.17% | 0.6036 / 0.6067 |

Two-step prediction is not uniformly better than direct-five prediction across all studies. The first Phikon comparison had mixed arrangement-level gains; the LoRA, ensemble, GenBio, gated-attention, Virchow2 and H-optimus-0 direct controls outperform their two-step counterparts in raw accuracy. UNI2-h's hierarchy improves its 812 accuracy/F1 but loses both on 813. GenBio's 813 one-stage F1 exceeds P's two-step score; P's F1 lead is now limited to arrangement-813 two-step pipelines, with UNI2-h leading 812. Each comparison describes its complete declared objective and readout, rather than isolating hierarchy or a threshold change alone.

## Methods and what the completed studies show

The original Phikon baseline averages frozen 1,024-dimensional field features within each view, applies fitting-only field-moment normalization, averages available views and appends four availability flags. Balanced logistic heads use four complete C settings `[0.01, 0.1, 1, 10]`; the hierarchy shares C across router and expert. The fixed route is malignant at probability at least 0.5. Explicit routed labels control accuracy/F1, because the argmax of composed five-way probabilities can differ.

A moves supervision to individual frozen Phikon fields with class/case/view-balanced weights and joint-probability pooling. B uses frozen Midnight-12k descriptions: final CLS plus mean patch tokens, totaling 3,072 features. Both case-mean and field-supervised readouts were completed with matched roles and four selection settings per arm. Their gains and tradeoffs remain distinct from changing the target from eight to five labels.

C uses centered-cosine similarity densities over fitting support examples. Per-view centers give each fitting case equal mass; normalized fields use unchanged class/case/field-balanced density weights and uniform head-class priors. Four shared temperatures are selected by end-to-end validation macro F1. Fitting diagnostics exclude the entire query case from support and centering; validation uses the 46 fitting cases only. Joint field probabilities are averaged within each magnification, then equally across available magnifications before routing. Support construction uses fitting labels, while no classifier coefficients are optimized.

D is the completed 20-run LoRA study, with two head-only warmup epochs followed by 18 adapter epochs and four checkpoint opportunities. It starts from the fixed C=1 field heads, whose existing validation accuracy was already **68.17% / 59.50%**, F1 **0.4870 / 0.3682**. Selected two-step accuracy rises only **0.17 / 0.33 percentage points** from that starting point. Eight of ten two-step runs select epoch 2 before adapter learning begins, including all five arrangement-813 runs. Selected fitting accuracy is 99.13%, already matched by the initial C=1 heads. The large generalization gap persists. The entire gap from stronger validation-selected frozen models cannot be attributed to adapter damage; initialization, augmentation, head training and adaptation form one registered recipe. [Completed adapter evidence](../docs/BREAKHIS_LORA_RESULTS.md).

E averages the accepted selected C and Midnight case-logistic joint-five probabilities with fixed 50/50 weights. No new weight, threshold or coefficient is fitted. Its final accuracy and macro F1 decrease on both arrangements relative to C. The ensemble inherits **eight component-selection opportunities per role/arm**, four C values plus four temperatures; it is not a four-total-opportunity comparison. No ensemble fitting metric is reported because the constituents' fitting diagnostics use different self-exclusion conditions. Its 95.00% binary score on 813 remains separate from its **72.17% final five-category accuracy**. [Completed ensemble evidence](../docs/BREAKHIS_FIXED_ENSEMBLE_RESULTS.md).

F changes image coverage while preserving frozen weights, provider roles, density mathematics and four temperatures. Each original field receives left, cached official-center and right crop descriptions; their float64 arithmetic mean is cast once to FP32. This adds 15,818 crop forwards. Relative to C, final accuracy changes by **−0.167 / +1.833 percentage points** and macro F1 by **+0.002819 / +0.015655**. The descriptive larger-improvement screen is not met, but these measured small gains and regressions remain reportable. [Completed coverage evidence](../docs/BREAKHIS_COVERAGE_RESULTS.md).

N tests one frozen native global field, 448 pixels high and 700 pixels wide, instead of the resized three-crop representation. Its two-step accuracy is **60.00% / 61.00%**, F1 **0.439405 / 0.501395**, below F on both arrangements. The larger input changes resolution, coverage and global attention together; more retained pixels did not deliver a classification gain here. Its 7,909 native forwards are not directly comparable in computation with 15,818 additional crop forwards: each native input has 1,601 rather than 257 tokens. [Completed native-input evidence](../docs/BREAKHIS_NATIVE_RESULTS.md).

P reuses F's frozen three-crop features and the same density weights, priors, four temperatures and provider roles. Class-shared centering averages five class case-centroids and uses that center for all heads. Head-specific centering retains the same direct-five center, uses an equal mean of benign/malignant group centers for the router, and averages four malignant-type centers for the expert, excluding benign cases. Every center uses fitting support only; fitting diagnostics remove the whole query case first. The original center remains mathematically valid: this is a supervised cosine-geometry ablation, not a correction of invalid prior mathematics.

P's class-shared two-step model scores **76.17% / 75.33%**, F1 **0.599196 / 0.577788**. The head-specific model scores **76.33% / 75.67%**, F1 **0.602715 / 0.602512**. Compared with F, the head-specific model changes accuracy by **+0.833 / −1.167 percentage points** and F1 by **+0.010205 / +0.022565**. It led both arrangements before the approved-reader round and retains the 813 two-step F1 lead; UNI2-h now leads 812. Neither centering mode meets the complete descriptive screen. The direct controls repeat exactly across modes. Selecting between modes therefore exposes eight structural two-step settings and four distinct direct settings, rather than four total family choices. [Completed centering evidence](../docs/BREAKHIS_PRIOR_CENTERED_RESULTS.md).

Head-specific centering also worsens probability losses relative to F: cross-entropy increases by 0.0570/0.1870 and Brier error by 0.0271/0.0978 on 812/813, with lower values preferred. Its arrangement-813 two-step macro-F1 lead does not establish better confidence estimates or a uniform improvement.

GenBio replaces the reader and its prescribed preprocessing together. Its normalized R/G/B channels pass separately through one shared frozen backbone; three 1,536-dimensional CLS vectors form a 4,608-dimensional description. The original case-centered KDE, class/case/field weights, probability pooling, four temperatures and provider roles remain fixed. C is the pinned primary comparison, F secondary and P context; no automatic predictor fusion or P-centering combination was performed. Two-step scores are **70.50% / 75.17%**, F1 **0.583661 / 0.581348**. Relative to C, accuracy changes **−5.167 / +0.167 percentage points** and F1 **−0.006030 / +0.017056**, so neither two-arrangement screen passes. Direct-five scores **71.33% / 77.83%**, F1 **0.595998 / 0.640603**: a strong second-arrangement one-stage result, without a consistent gain across arrangements. [Completed GenBio evidence](../docs/BREAKHIS_GENBIO_RESULTS.md).

GenBio improves true-malignant typing from C's 56.67% / 56.67% to **60.00% / 67.67%**, while routing falls from 92.00% / 92.67% to **82.00% / 89.83%**. Fifty of 75 papillary events on 812 and 40 of 75 on 813 are routed to benign. Holding its current case-conditional type predictions fixed, even perfect routing would yield only 80.00% / 83.83% final accuracy. These arithmetic bounds are not results for a new cascade; replacing field-level routing would also change pooled conditional probabilities. Overall cross-entropy/Brier worsen versus C on 812 (.9467/.4617 versus .9422/.4391), but improve on 813 (.8369/.4254 versus .9253/.4504). Representation, routing and confidence gains therefore remain distinct.

Whole-case MIL uses the accepted frozen F descriptions, fitting-only field-moment normalization and every original field. A shared 3,072→128 projection, ReLU and dropout precede arithmetic field mean or standard gated attention; both average available views, append four flags and train direct-five or binary/malignant-four heads. The matched poolers use common initial projection/head states and paired random streams. Training runs 100 epochs with class/case-balanced whole-case losses over all 15 masks, a fixed four-case loss denominator and four checkpoints at 10/25/50/100. Benign examples do not train the expert. Each pool/arm has four checkpoint opportunities; selecting between pools inherits eight, rather than four total choices. No reader updates or new encoder forwards occur. Attention is established prior art, and this comparison is not an attention novelty claim.

Learned mean two-step scores **70.17% / 71.83%**, F1 **0.519915 / 0.509386**; attention scores **71.67% / 70.50%**, F1 **0.545323 / 0.523416**. Attention improves F1 against its matched mean control by **0.025407 / 0.014030**, with accuracy changes **+1.50 / −1.33 percentage points**. Both remain below F/P in accuracy and F1, and both registered screens fail. Selected fitting accuracy is approximately 99.13%, with F1 about .982: fitting the case bags did not establish transfer to validation cases. These in-sample neural fitting scores are not comparable with KDE's leave-case-out fitting diagnostics. Mean's Brier errors .4235/.4321 are slightly below F's .4307/.4334, despite weaker accuracy/F1; its cross-entropy is not uniformly better. Attention worsens both cross-entropy and Brier versus learned mean. No uniformly stronger or better-calibrated pooler is established. [Completed whole-case evidence](../docs/BREAKHIS_CASE_MIL_RESULTS.md).

## Approved pathology readers and their tradeoffs

The approved-reader study holds C's support-only per-view centers, cosine KDE, class/case/field weights, original roles and four temperatures fixed. UNI2-h supplies a 1,536-dimensional CLS descriptor; Virchow2 concatenates CLS with mean patch tokens excluding four registers, yielding 2,560 dimensions; H-optimus-0 supplies a 1,536-dimensional CLS descriptor. Each uses its registered 224-pixel field transform and normalization. H-optimus-0's resize/center crop is a declared adaptation, without established equivalence to its native 0.5-microns/pixel input. This compares complete reader packages, not architecture alone. No new adapter, attention, centering mode, fusion, threshold or split search was added. Only BreakHis supplies current fitting/evaluation images; exact upstream pretraining overlap remains unresolved.

UNI2-h two-step accuracy is **75.83% / 75.17%**, F1 **0.664180 / 0.575175**. Compared with C, accuracy rises 0.167 percentage points on both arrangements and F1 rises **0.074489 / 0.010883**. The 812 screen passes; the 813 F1 gain is below the registered .02 requirement, so the complete screen fails. Compared with P, UNI2-h gains .061465 F1 on 812, loses .027337 on 813 and loses 0.50 percentage points of accuracy on both. UNI2-h's direct control is **75.17% / 77.67%**, F1 **0.648631 / 0.634493**. Its uneven benefit is reported without promoting a different reader for each arrangement into a single model.

UNI2-h's two-step 812 malignant typing reaches 72.00% versus C's 56.67%, and mucinous recall rises to 77.33% versus 41.33%. Router accuracy drops to 83.33% versus 92.00%, and benign recall to 84.67% versus 97.00%. Its 813 typing is 60.67%, router accuracy 90.33% and papillary recall 28.00%. Virchow2 two-step scores **72.50% / 73.00%**, F1 **0.575767 / 0.538743**; its direct scores **72.83% / 75.00%**, F1 **0.590080 / 0.587676**. H-optimus-0 two-step scores **73.50% / 71.17%**, F1 **0.509710 / 0.506235**; direct scores **77.33% / 75.17%**, F1 **0.603560 / 0.606677**. Its 812 two-step papillary recall is only 1.33%. All complete reader screens fail against both C and F.

Probability quality is a separate tradeoff. UNI2-h's two-step cross-entropy/Brier are .9571/.4722 on 812 and .8361/.4096 on 813: worse than C on 812, better on 813. Virchow2 improves both losses versus C in both arrangements (.8904/.4248 and .8219/.4066), despite lower accuracy/F1. H-optimus-0 improves both on 812 (.9210/.4141) but worsens both on 813 (.9652/.4861). Every approved-reader direct arm has worse probability losses than its own two-step arm, even where hard-label scores improve. These loss comparisons do not establish calibrated confidence. [Complete reader results, class recalls and provenance](../docs/BREAKHIS_APPROVED_ENCODERS_RESULTS.md).

## Retained head-specific centering strengths and remaining errors

| Arrangement | Final accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy | Binary accuracy | True-malignant typing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 76.33% | 0.602715 | 64.73% | 85.00% | 94.83% | 58.33% |
| 813 | 75.67% | 0.602512 | 66.47% | 85.00% | 93.00% | 62.33% |

The final score, binary routing and conditional four-type score evaluate different decisions. Conditional typing uses all truly malignant cases independently of the router. High routing accuracy cannot substitute for accurate subtype answers. Head-specific all-four-view accuracy of 85.00% on both arrangements is the highest observed two-step full-view result, but does not replace the mean-15-subset headline; F's all-view score remains 82.50% / 82.50%.

The learned-mean router reaches 96.33% on 813, while true-malignant typing is 50.33% and final five-category accuracy 71.83%. Attention routing is 92.17% / 94.67%, typing 58.33% / 51.00%. Those high binary percentages do not constitute 95% final-five results.

| P head-specific end-to-end class recall | 812 | 813 |
| --- | ---: | ---: |
| Benign | 95.67% | 91.00% |
| Ductal carcinoma | 65.33% | 61.33% |
| Lobular carcinoma | 93.33% | 98.67% |
| Mucinous carcinoma | 49.33% | 36.00% |
| Papillary carcinoma | 20.00% | 45.33% |

Mucinous and papillary recognition remains weak despite high benign and lobular recall. Raw accuracy alone would obscure this imbalance. The small fitting population, including only two lobular and three papillary cases, limits what repeated image/augmentation observations can establish about generalization.

## Execution evidence

All completed studies passed their output collectors with source/configuration/cache/role bindings and complete candidate coverage. C, F, N and GenBio each contain ten paired runs, 80 candidates, 160 archives and 64,800 dependent case/subset records. LoRA contains 20 runs, 80 checkpoints, 9,600 candidate validation records and 13,800 selected fitting records. E audits all 40 selected source archives and 4,800 input records, then replays 20 ensemble archives and 2,400 output records. Whole-case MIL contains 40 trained trajectories, 160 checkpoints, 320 prediction archives and 129,600 dependent records. The checkpoints are not 160 independent training trajectories or new patients. Saved probabilities, explicit decisions, metrics, role membership and selection are checked. Adapter producer probes verify initial/fresh-reconstruction behavior; CPU cached-feature replay is not independent raw-image adapted-encoder reproduction.

Native extraction, audit and classifier evaluation are complete. The centering collector 4982975 completed successfully in 8 minutes 18 seconds with empty error output. It verified **160 logical / 120 structurally unique candidate definitions**, **320 logical / 240 structurally nonduplicated archives**, and **129,600 logical / 97,200 structurally nonduplicated records**. The repeated direct controls were checked for exact equality; 136 additional arithmetic/duplicate checks passed. These structural counts are not independent cases or guarantees of distinct file hashes.

The GenBio/MIL source CI passed **370 neural tests** before production. Four fitting-only whole-case A40 checks and their independent reviewer passed; all 40 production GPU trajectories completed in 32–47 seconds each, with at most four concurrent GPUs. Collector 4983087 completed in 3 minutes 6 seconds with empty error output and independently replayed every checkpoint in NumPy FP64. GenBio's four feature workers processed all 7,909 images, equivalent to 23,727 channel passes; its accepted CPU audit performed no encoder execution. All ten paired classifier runs completed, and collector 4983142 completed in 4 minutes 46 seconds with empty error output. Independent reviews checked selection, arithmetic, class recall and provenance.

The approved-reader feature source `375bcbc499770635dde6c9586c5ce49e2c44300d` passed **390 neural tests**. Each reader completed four fitting-only consistency/reload checks with zero differences under canonical one-image model calls, then all 7,909 fields and all four feature audits. Across readers there were 23,727 original-field calls plus 48 successful technical calls. All 30 paired classifier roles and three collectors are complete, covering **240 candidates, 480 archives and 194,400 dependent records**. Selecting across readers inherits **12 reader/temperature choices per arm/role**. Independent review passed 968,317 saved arithmetic/provenance assertions plus 192 input-receipt assertions; the collectors' complete cached-feature replay is separate evidence, without independent reviewer re-execution of private raw images or feature arrays. The selected validation denominator is 600 dependent case/subset decisions per arrangement, and 40 in the separate all-four-view condition, reusing 31 distinct validation keys. All classification jobs and collectors in these studies are complete.

Accepted ledgers retain all source/configuration revisions, hashes, candidates, fitting diagnostics and class confusions: [first five-category baseline](../docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json), [EfficientNet](../docs/evidence/BREAKHIS_CASEBAG_RESULTS.json), [Midnight case/field](../docs/evidence/BREAKHIS_MIDNIGHT_RESULTS.json), [C single-center similarity](../docs/evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), [D LoRA](../docs/evidence/BREAKHIS_LORA_RESULTS.json), [E ensemble](../docs/evidence/BREAKHIS_FIXED_ENSEMBLE_RESULTS.json), [F three-crop](../docs/evidence/BREAKHIS_COVERAGE_RESULTS.json), [N native input](../docs/evidence/BREAKHIS_NATIVE_RESULTS.json), [P centering](../docs/evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json), [GenBio](../docs/evidence/BREAKHIS_GENBIO_RESULTS.json), [whole-case mean/attention](../docs/evidence/BREAKHIS_CASE_MIL_RESULTS.json), [UNI2-h](../docs/evidence/BREAKHIS_UNI2_H_RESULTS.json), [Virchow2](../docs/evidence/BREAKHIS_VIRCHOW2_RESULTS.json), and [H-optimus-0](../docs/evidence/BREAKHIS_H_OPTIMUS_0_RESULTS.json). N's accepted ledger SHA256 is `417ec2a4a3806d63154f2c76300db378568e19a29b0d5d1531d745ea891fd6e4`; P's is `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c`. GenBio's is `92e2ff976b1be20a69a0ccee2147537b2905293926675d2ff3be85283cc589d1`; whole-case MIL's is `a7c00e08a9279e79311d24547cf5df06d5c7cbfe913b1a880e8c6c1e345ae2b5`.

## Historical scope and claims

The supplied notebooks saved 93.73% / 94.99% for eight-type, single-40× images under image-random, reused validation. Earlier case-based eight-class accuracy was 45.7% / 38.3%; the retrospective binary analysis gave 87.0% / 80.7%. Those numbers retain their original targets and units. Pooling benign labels changes the task and supports; it does not retrospectively improve the eight-class model or turn a binary score into five-category accuracy. [Historical evidence](../docs/CAPSTONE_BEFORE_AND_NOW.md).

The project has produced reproducible baseline execution, measured improvements and informative negative comparisons, including UNI2-h's substantial 812 F1 gain that does not repeat uniformly on 813. Reliable 94–95% final five-category accuracy, fresh clinical generalization, a novel method and conference acceptance remain unestablished. Access to UNI2-h, Virchow2 and H-optimus-0 is confirmed, and their comparisons are complete. The immediate deliverable is the five-category capstone presentation/demo, including the specialist's generalization limits and the full outcome tradeoffs. No automatic reader fusion or further setting search follows from this round.
