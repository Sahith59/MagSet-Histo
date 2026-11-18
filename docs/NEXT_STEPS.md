# Next BreakHis tasks

## Current checkpoint and next action

The frozen single-output UNI2-h model scored **60.00% five-category and 79.17% binary** on the separate 16-case retrospective benchmark. The 17-case checkpoint-selection figures were 80.39% and 99.22%. All 240 follow-up decisions reproduced from 1,704 cached field descriptors. The combined >90% / >95% target remains open. [Benchmark, comparison and next step](BREAKHIS_CAPSTONE_BENCHMARK.md).

One frozen UNI2-h reader now gives five class scores through a single fitted output layer. The same scores answer both questions. On 17 validation cases it reached **80.39% five-category and 99.22% binary** (205 and 253 of 255 answers). On the separate 42-group training check it reached **63.81% and 89.39%** (469 and 657 of 735). Each case supplies 15 zoom conditions. These are development results. The combined target remains open. All 84 center crops and GPU descriptors reproduced. Obtain blinded expert observations on the full-field/crop pairs, then decide whether to test wider coverage or prioritize additional independent rare cases. [Verified experiment and next decision](SINGLE_OUTPUT_CENTER.md).

## Earlier field-pooling checkpoint

The field-pooling comparison and 42-group follow-up are complete. The original shared classifier remains the reference. Field-first tied at 75.69% five-category validation accuracy and improved binary accuracy to 90.59%, while recognizing fewer lobular answers. The targets are above 90% final and above 95% binary together.

The next task is a blinded expert review of the 84 original fields and their 224-pixel center crops. Record whether a distinguishing feature is visible in the full image and at the model's input size. If cues are repeatedly cropped out, evaluate wider coverage with the same single reader under a protocol fixed before fitting. Seek additional independent rare cases for a separate confirmation cohort. [Numbers, evidence and review rationale](FIELD_HIERARCHY.md).

## Current plan for the single-model capstone

1. **Completed.** Fit one shared UNI2-h classifier for both decisions, compare all three eligible models on the same 17 validation cases and independently replay the selected result. Final-five accuracy is 75.69%, with 26 routing errors and 36 subtype errors over 255 case and zoom combinations.
2. **Review.** Inspect fixed training and validation fields with a qualified reviewer. Record tissue coverage, readable zooms and which subtype features can be seen. Keep the first review blind to the model prediction. Learn why the two validation lobular cases resemble ductal to the model.
3. **Confirm.** Seek additional independently grouped lobular and papillary cases. Before evaluating a new method on fresh cases, fix the reader, model weights, decision rule and metric. Report five-category accuracy, class recalls and binary accuracy separately.

[Single-model evidence and scorecard](SINGLE_MODEL_DUAL_STEP.md). The greater-than-90% final-five target remains open.

## Earlier plans and checkpoints

November 17 complete deep diagnosis: the original three scores reproduce from raw images. The fixed-router UNI2-h specialist gives 82.92% on the already-viewed follow-up but fails its validation improvement screen. The final bounded case-SVC comparison is also complete and independently accepted: linear 74.58% / .513007 final-five accuracy/F1; RBF 69.17% / .380729; both binary 92.08%. RBF’s validation macro-F1 gain is .009833, below the fixed .02 screen. Its 99.22% validation binary accuracy is not follow-up performance. [Proof and diagnosis](DEEP_DIAGNOSIS_AND_PROOF.md) · [complete SVC result](BREAKHIS_UNI_CASE_SVM_RESULTS.md).

Both requested follow-up targets, over 90% final-five and over 97% binary, remain unmet. All proof, specialist, router and case-SVC jobs completed; no project jobs remain active. Both bounded intervention families are closed without a confirmed replacement. Preserve every outcome and finish the capstone evidence/presentation, using the fixed TRAIN/validation field index for appropriate expert review. Do not add C/gamma/threshold/model/seed/split searches on the already-viewed follow-up. Suitable additional independent cases are needed for fresh confirmation. TCGA and novelty expansion remain deferred. The dated checkpoints below are historical.

The requested [matched comparison](MATCHED_THREE_WAY_RESULTS.md) is complete and independently accepted. Mean-15 final-five accuracy / macro F1 is P 67.50% / .375675, CNN 56.67% / .348180, and Swin 59.17% / .304947 on 16 test cases in 13 conservative groups. P’s 94.58% binary score is separate. Every paired interval includes zero; P also has weaker worst-subset performance. Both A40 test replay errors are zero, and the failed A30 attempt remains preserved. No project jobs remain active. Finish the capstone explanation and presentation using these matched results alongside clearly separated historical trial and notebook evidence. Do not claim 95% final accuracy or clinical superiority.

Both requested notebooks are complete and independently checked: CNN corrected **97.09% binary /86.41% eight-type**; Swin **93.38% binary /64.37% final-five**, macro F1 .441602. Preserve the original CNN failure and every declared repair. Their different units/targets/splits do not establish a matched winner. [Side-by-side comparison](SUPPLIED_NOTEBOOKS_COMPARISON.md), [Swin complete results](SUPPLIED_SWIN_STATUS.md). The main capstone was parked during this request; return now to the persistent malignant-error review and final five-category materials below. No new fit, extra fold, test-driven selection or TCGA work follows automatically.

Both new specialist studies are complete. Learning the two UNI2-h feature metrics gives 76.17% / 75.83% final accuracy, with small gains over the exact identity control. Standard supervised UNI2-h field heads give 73.83% / 73.67%, below their reference. Both declared improvement screens fail. The malignant-only metric ablation leads observed 812 two-step F1 at .672180; P leads 813 at .602512. P's separate all-four-view accuracy remains 85% / 85%. **94–95% final accuracy is still unachieved.** [Metric results](BREAKHIS_SPECIALIST_METRIC_RESULTS.md), [linear results](BREAKHIS_UNI_LINEAR_RESULTS.md).

## Review the persistent malignant errors

The read-only case audit is complete. It joins the same original case/trial/zoom records across all four metric arms and linear dual: 6,000 saved events across five pipelines, each reusing 31 validation provider keys. The metric primary corrects eight events and regresses on two versus identity, for a net six. It still misses 106/150 papillary events, including 75 routed to benign. Three keys remain wrong in every observed event, covering benign, ductal and papillary. These counts locate persistent cases; they do not establish a morphological or label cause. [Complete audit](BREAKHIS_SPECIALIST_ERROR_AUDIT.md).

The private inspection index retains original manifest paths and fixed first/middle/last field examples per view. These are uninspected case-bag examples, not fields proven to cause the pooled prediction. Review them with appropriate expertise before choosing a label, coverage, stain or representation intervention. The audit adds no fitting, predictions, model selection or independent cohort.

Use a specific finding to justify the next fixed comparison. Preserve all-four-view scores as a separate endpoint and distinguish unequal-exposure event counts from clearly labeled equal-key descriptive averages. Avoid another generic model sweep, post-hoc combinations of favorable heads, split changes or substitution of binary accuracy for final-five accuracy. The latest linear router is 96.17% on 813 while final accuracy is 73.67%; subtype errors still count.

## Finish the five-category capstone package

Update the presentation and demonstration to benign plus ductal, lobular, mucinous and papillary carcinoma. Existing eight-category artifacts remain historical. Explain the input images, frozen reader, reference or learned readout, probability pooling, binary decision and malignant-four decision. Include representative errors and the complete accepted positive, mixed and negative results. [Plain explanation](PROJECT_EXPLAINED.md), [professor brief](evidence/PROFESSOR_BRIEF.md), [report](../paper/breakhis-five-class-report.md).

Explain 75.83% / 75.17% as two validation arrangements of the earlier UNI2-h model: 455/600 and 451/600 correct repeated checks. Each denominator combines five overlapping trials, fifteen zoom subsets and eight validation cases. It does not count 600 independent patients. The new metric primary gives 457/600 and 455/600 on those same checks.

Keep the original 46 fitting / 8 validation / 28 excluded roles, five final labels and 15-subset scoring. Only BreakHis supplies current fitting/evaluation data; pathology pretraining is transfer learning. All 82 provider keys entered development, validation spans 31 distinct keys, and clinical identity remains unresolved. There is no globally fresh final cohort. [Independent metric review](reviews/SPECIALIST_LEARNING_VALIDITY_REVIEW.md), [independent linear review](reviews/UNI_LINEAR_ACTUAL_RESULTS_REVIEW.md).

Working materials target November 17–18, presentation around November 20; the exact day and rubric remain unconfirmed. TCGA expansion, novelty and conference submission remain deferred. The report should distinguish implementation completion, measured development improvement and independently confirmed research claims.
