# MAGSET-Histo: current BreakHis project brief

Updated November 16, 2026. The immediate deliverable is the BreakHis capstone for Professor Dong. **94–95% final accuracy is a goal, not an achieved result.** TCGA follow-up and novel-paper work remain deferred.

| Question | Current answer |
| --- | --- |
| Problem | Classify a provider case from the microscope magnifications available, including missing-view settings. Clinical identity of the provider keys remains unresolved. |
| Dataset | Audited BreakHis release: 7,909 images, 82 provider case keys, four magnifications and eight original labels. Only BreakHis supplies current fitting/evaluation images and labels. |
| Inputs | Sets of fields at 40×, 100×, 200× and 400×. No fabricated exact correspondence between fields at different zooms. All 15 nonempty zoom subsets are evaluated. |
| Final outputs | Benign, ductal, lobular, mucinous or papillary carcinoma. Step 1 is benign/malignant; benign stops, malignant goes to a four-type specialist. There is no benign subtype expert. |
| Current reader | Frozen pretrained Midnight-12k; 3,072 features per original field. The completed three-crop version averages left/center/right embeddings before classification. External pretraining is transfer learning. |
| Current classifier | Same-zoom class/case/field-balanced cosine density estimates; compose field joint-five probabilities, then average fields and available zooms. Hard malignant routing uses probability ≥0.5. Direct-five is a matched control. |
| Development roles | 46 fitting/eight validation/28 excluded cases per role, preserving original eight-label assignments before pooling benign. Five overlapping trials in arrangements 812/813; excluded cases are not classifier-scored. |
| Minority support | Fitting: 11 benign, 25 ductal, two lobular, five mucinous, three papillary. Validation: four benign and one of each malignant type. |
| Selection and metrics | Four shared temperatures per named method/arm, selected by final macro F1 averaged over all 15 subsets. Report final accuracy, macro F1, balanced accuracy, class recalls, routing and malignant typing separately. |
| Leading completed references | C: accuracy 75.67%/75.00%, F1 .589691/.564292. F: 75.50%/76.83%, F1 .592510/.579947. P head-specific centers: 76.33%/75.67%, F1 .602715/.602512. These are arrangement 812/813, not separate datasets. |
| Main limitation | Head-specific P's binary accuracy is 94.83%/93.00%, but true-malignant typing is 58.33%/62.33%; mucinous and papillary recognition remains weak. Its all-four-view accuracy is 85%/85%, a separate endpoint. |
| Completed negative comparisons | LoRA dual 68.33%/59.83%; fixed equal-probability vote 74.17%/72.17%. Both are verified and retained. The LoRA fixed initialization was already 68.17%/59.50%, so its whole gap to C cannot be attributed to adaptation. |
| Completed input/geometry comparisons | Native 700×448 input, N: accuracy 60.00%/61.00%, F1 .439405/.501395; no gain. P shared center: 76.17%/75.33%, F1 .599196/.577788. P head-specific centers have the highest measured F1 on both arrangements, with raw-accuracy tradeoffs; both modes fail the registered screen against F. |
| Next deliverable | All registered jobs and collectors are complete and accepted. Close/freeze the verified evidence, then prioritize the capstone report, presentation and demo. Presentation day remains unconfirmed. |
| Evidence limits | 31 distinct reused validation keys; all 82 provider keys entered development somewhere. Dependent fields/masks/trials and unresolved clinical identity preclude a fresh clinical-test claim. |
| Contribution status | Implemented and audited standard baselines, controlled adaptations and failure analysis. No novel mechanism, 95% five-category result, clinical readiness or publication acceptance is established. |

Keep C, F and both P modes in the scorecard. Head-specific P's highest descriptive F1 is not a uniform improvement: 813 accuracy is below F's 76.83%, and both P modes fail the registered +.02 F1/nondecreasing-accuracy screen. The original 93.73%/94.99% notebook outputs used a different eight-type, 40× image-random reused-validation protocol and do not measure this five-category case task.

N tested fine pixel detail and global context with unchanged weights and did not help this recipe. P tested two supervised preprocessing geometries on accepted F features with unchanged density math; the original center remains mathematically valid. Direct controls repeat across P modes, leaving four structurally unique direct choices and eight dual choices across the family. Standard transfer learning, LoRA, cosine classification, masking, probability pooling and supervised centering are existing methods. Availability-conditioned evidence matching remains deferred. No automatic N+P combination or sweep follows; any further subtype-representation intervention needs separate registration and matched controls, without a promised 95% result.

[Training explanation](TRAINING_STRATEGY.md) · [Next steps](NEXT_STEPS.md) · [C results](BREAKHIS_KERNEL_RESULTS.md) · [F results](BREAKHIS_COVERAGE_RESULTS.md) · [P accepted JSON](evidence/BREAKHIS_PRIOR_CENTERED_RESULTS.json) · [N results](BREAKHIS_NATIVE_RESULTS.md) · [LoRA](BREAKHIS_LORA_RESULTS.md) · [Fixed vote](BREAKHIS_FIXED_ENSEMBLE_RESULTS.md) · [Current five-category report](../paper/breakhis-five-class-report.md).
