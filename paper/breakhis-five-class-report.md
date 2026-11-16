# BreakHis five-category dual-step project report

Working capstone report, updated November 16, 2026. The completed **head-specific centering model has the highest macro F1 on both arrangements, 0.602715 / 0.602512**, with final five-category accuracy **76.33% / 75.67%**. Three-crop case-shared centering gives **75.50% / 76.83%**, F1 **0.592510 / 0.579947**, and retains higher arrangement-813 accuracy. Both remain on the scorecard; there is no unqualified accuracy winner. **94–95% final accuracy is not achieved.**

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

Two-step prediction is not uniformly better than direct-five prediction across all studies. The first Phikon comparison had mixed arrangement-level gains; the LoRA and ensemble direct controls outperform their two-step counterparts in raw accuracy. Each result describes its complete declared objective and readout, rather than isolating the benefit of hierarchy alone.

## Methods and what the completed studies show

The original Phikon baseline averages frozen 1,024-dimensional field features within each view, applies fitting-only field-moment normalization, averages available views and appends four availability flags. Balanced logistic heads use four complete C settings `[0.01, 0.1, 1, 10]`; the hierarchy shares C across router and expert. The fixed route is malignant at probability at least 0.5. Explicit routed labels control accuracy/F1, because the argmax of composed five-way probabilities can differ.

A moves supervision to individual frozen Phikon fields with class/case/view-balanced weights and joint-probability pooling. B uses frozen Midnight-12k descriptions: final CLS plus mean patch tokens, totaling 3,072 features. Both case-mean and field-supervised readouts were completed with matched roles and four selection settings per arm. Their gains and tradeoffs remain distinct from changing the target from eight to five labels.

C uses centered-cosine similarity densities over fitting support examples. Per-view centers give each fitting case equal mass; normalized fields use unchanged class/case/field-balanced density weights and uniform head-class priors. Four shared temperatures are selected by end-to-end validation macro F1. Fitting diagnostics exclude the entire query case from support and centering; validation uses the 46 fitting cases only. Joint field probabilities are averaged within each magnification, then equally across available magnifications before routing. Support construction uses fitting labels, while no classifier coefficients are optimized.

D is the completed 20-run LoRA study, with two head-only warmup epochs followed by 18 adapter epochs and four checkpoint opportunities. It starts from the fixed C=1 field heads, whose existing validation accuracy was already **68.17% / 59.50%**, F1 **0.4870 / 0.3682**. Selected two-step accuracy rises only **0.17 / 0.33 percentage points** from that starting point. Eight of ten two-step runs select epoch 2 before adapter learning begins, including all five arrangement-813 runs. Selected fitting accuracy is 99.13%, already matched by the initial C=1 heads. The large generalization gap persists. The entire gap from stronger validation-selected frozen models cannot be attributed to adapter damage; initialization, augmentation, head training and adaptation form one registered recipe. [Completed adapter evidence](../docs/BREAKHIS_LORA_RESULTS.md).

E averages the accepted selected C and Midnight case-logistic joint-five probabilities with fixed 50/50 weights. No new weight, threshold or coefficient is fitted. Its final accuracy and macro F1 decrease on both arrangements relative to C. The ensemble inherits **eight component-selection opportunities per role/arm**, four C values plus four temperatures; it is not a four-total-opportunity comparison. No ensemble fitting metric is reported because the constituents' fitting diagnostics use different self-exclusion conditions. Its 95.00% binary score on 813 remains separate from its **72.17% final five-category accuracy**. [Completed ensemble evidence](../docs/BREAKHIS_FIXED_ENSEMBLE_RESULTS.md).

F changes image coverage while preserving frozen weights, provider roles, density mathematics and four temperatures. Each original field receives left, cached official-center and right crop descriptions; their float64 arithmetic mean is cast once to FP32. This adds 15,818 crop forwards. Relative to C, final accuracy changes by **−0.167 / +1.833 percentage points** and macro F1 by **+0.002819 / +0.015655**. The descriptive larger-improvement screen is not met, but these measured small gains and regressions remain reportable. [Completed coverage evidence](../docs/BREAKHIS_COVERAGE_RESULTS.md).

N tests one frozen native global field, 448 pixels high and 700 pixels wide, instead of the resized three-crop representation. Its two-step accuracy is **60.00% / 61.00%**, F1 **0.439405 / 0.501395**, below F on both arrangements. The larger input changes resolution, coverage and global attention together; more retained pixels did not deliver a classification gain here. Its 7,909 native forwards are not directly comparable in computation with 15,818 additional crop forwards: each native input has 1,601 rather than 257 tokens. [Completed native-input evidence](../docs/BREAKHIS_NATIVE_RESULTS.md).

P reuses F's frozen three-crop features and the same density weights, priors, four temperatures and provider roles. Class-shared centering averages five class case-centroids and uses that center for all heads. Head-specific centering retains the same direct-five center, uses an equal mean of benign/malignant group centers for the router, and averages four malignant-type centers for the expert, excluding benign cases. Every center uses fitting support only; fitting diagnostics remove the whole query case first. The original center remains mathematically valid: this is a supervised cosine-geometry ablation, not a correction of invalid prior mathematics.

P's class-shared two-step model scores **76.17% / 75.33%**, F1 **0.599196 / 0.577788**. The head-specific model scores **76.33% / 75.67%**, F1 **0.602715 / 0.602512**. Compared with F, the head-specific model changes accuracy by **+0.833 / −1.167 percentage points** and F1 by **+0.010205 / +0.022565**. It gives the highest measured F1 on both arrangements, while neither centering mode meets the complete descriptive screen. The direct controls repeat exactly across modes. Selecting between modes therefore exposes eight structural two-step settings and four distinct direct settings, rather than four total family choices. [Completed centering evidence](../docs/BREAKHIS_PRIOR_CENTERED_RESULTS.md).

Head-specific centering also worsens probability losses relative to F: cross-entropy increases by 0.0570/0.1870 and Brier error by 0.0271/0.0978 on 812/813, with lower values preferred. Its macro-F1 lead does not establish better confidence estimates or a uniform improvement.

## Current head-specific centering strengths and remaining errors

| Arrangement | Final accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy | Binary accuracy | True-malignant typing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 812 | 76.33% | 0.602715 | 64.73% | 85.00% | 94.83% | 58.33% |
| 813 | 75.67% | 0.602512 | 66.47% | 85.00% | 93.00% | 62.33% |

The final score, binary routing and conditional four-type score evaluate different decisions. Conditional typing uses all truly malignant cases independently of the router. High routing accuracy cannot substitute for accurate subtype answers. Head-specific all-four-view accuracy of 85.00% on both arrangements does not replace the mean-15-subset headline; F's all-view score remains 82.50% / 82.50%.

| P head-specific end-to-end class recall | 812 | 813 |
| --- | ---: | ---: |
| Benign | 95.67% | 91.00% |
| Ductal carcinoma | 65.33% | 61.33% |
| Lobular carcinoma | 93.33% | 98.67% |
| Mucinous carcinoma | 49.33% | 36.00% |
| Papillary carcinoma | 20.00% | 45.33% |

Mucinous and papillary recognition remains weak despite high benign and lobular recall. Raw accuracy alone would obscure this imbalance. The small fitting population, including only two lobular and three papillary cases, limits what repeated image/augmentation observations can establish about generalization.

## Execution evidence

All completed studies passed their output collectors with source/configuration/cache/role bindings and complete candidate coverage. C, F and N each contain ten paired runs, 80 candidates, 160 archives and 64,800 dependent case/subset records. LoRA contains 20 runs, 80 checkpoints, 9,600 candidate validation records and 13,800 selected fitting records. E audits all 40 selected source archives and 4,800 input records, then replays 20 ensemble archives and 2,400 output records. Saved probabilities, explicit decisions, metrics, role membership and selection are checked. Adapter producer probes verify initial/fresh-reconstruction behavior; CPU cached-feature replay is not independent raw-image adapted-encoder reproduction.

Native extraction, audit and classifier evaluation are complete. The centering collector 4982975 completed successfully in 8 minutes 18 seconds with empty error output. It verified **160 logical / 120 structurally unique candidate definitions**, **320 logical / 240 structurally nonduplicated archives**, and **129,600 logical / 97,200 structurally nonduplicated records**. The repeated direct controls were checked for exact equality; 136 additional arithmetic/duplicate checks passed. These structural counts are not independent cases or guarantees of distinct file hashes. No jobs remain pending, and all reported N/P outcomes are accepted results.

Accepted ledgers retain all source/configuration revisions, hashes, candidates, fitting diagnostics and class confusions: [first five-category baseline](../docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json), [EfficientNet](../docs/evidence/BREAKHIS_CASEBAG_RESULTS.json), [Midnight case/field](../docs/evidence/BREAKHIS_MIDNIGHT_RESULTS.json), [C single-center similarity](../docs/evidence/BREAKHIS_MIDNIGHT_KERNEL_RESULTS.json), [D LoRA](../docs/evidence/BREAKHIS_LORA_RESULTS.json), [E ensemble](../docs/evidence/BREAKHIS_FIXED_ENSEMBLE_RESULTS.json), [F three-crop](../docs/evidence/BREAKHIS_COVERAGE_RESULTS.json), [N native input](../docs/evidence/BREAKHIS_NATIVE_RESULTS.json), and [P centering](../docs/evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json). N's accepted ledger SHA256 is `417ec2a4a3806d63154f2c76300db378568e19a29b0d5d1531d745ea891fd6e4`; P's is `b9e0616382099a405295a211bae15dab4625091b68620a1e647196a51ebe1e3c`.

## Historical scope and claims

The supplied notebooks saved 93.73% / 94.99% for eight-type, single-40× images under image-random, reused validation. Earlier case-based eight-class accuracy was 45.7% / 38.3%; the retrospective binary analysis gave 87.0% / 80.7%. Those numbers retain their original targets and units. Pooling benign labels changes the task and supports; it does not retrospectively improve the eight-class model or turn a binary score into five-category accuracy. [Historical evidence](../docs/CAPSTONE_BEFORE_AND_NOW.md).

The project has produced reproducible baseline execution, measured improvements and informative negative comparisons. Reliable 94–95% final five-category accuracy, fresh clinical generalization, a novel method and conference acceptance remain unestablished. The immediate deliverable is the complete BreakHis capstone report and presentation with these distinctions intact.
