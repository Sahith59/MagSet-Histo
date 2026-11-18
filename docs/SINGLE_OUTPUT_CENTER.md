# Can one five-category output make both decisions?

## Result

One frozen UNI2-h image reader and one five-category linear classifier made every prediction. Both decisions use the same five output scores. On the same 17 validation cases the one-output rule got 12 more final diagnoses and 24 more binary answers right than the earlier model. On the separate group check its accuracies fell. The binary target was reached on the validation cases only. The five-category target remains open.

| Evaluation | Final five | Benign/malignant | Macro F1 |
| --- | ---: | ---: | ---: |
| Earlier reference, 17 validation cases | 193/255 (75.69%) | 229/255 (89.80%) | 0.4982 |
| One output, same 17 validation cases | 205/255 (80.39%) | 253/255 (99.22%) | 0.6337 |
| Earlier reference, 49 cases with their groups held out | 484/735 (65.85%) | 673/735 (91.56%) | 0.4492 |
| One output, same 49 cases with their groups held out | 469/735 (63.81%) | 657/735 (89.39%) | 0.5107 |

The selected checkpoint was update 10. With all four zooms available it got 14/17 final diagnoses right. Each of the 17 cases contributes 15 answers to the main score. The two lobular cases belong to one conservative group and the papillary answers come from one case. In the group check, 42 independently fitted classifiers each predicted only the group omitted from their fit. Their predictions were never combined for a case. All training and validation images came from BreakHis. The 16 previously used follow-up cases produced zero new predictions.

| True label | Original correct on validation | One output correct on validation | One output correct with training groups held out |
| --- | ---: | ---: | ---: |
| Pooled benign | 75/75 | 73/75 | 140/210 |
| Ductal | 98/105 | 84/105 | 240/360 |
| Lobular | 5/30 | 17/30 | 24/30 |
| Mucinous | 15/30 | 30/30 | 55/75 |
| Papillary | 0/15 | 1/15 | 10/60 |

The validation classifier correctly typed 132/180 malignant answers. Of its 50 final errors, two were benign cases called malignant and 48 were cancers given the wrong type. It called the papillary case mucinous in 14 of 15 conditions. Ductal recognition also fell. Counting paired changes at case level, five validation cases improved, six worsened and six tied. The net gain in repeated answers therefore depends on which cases are present. To exceed 90% on this validation exercise still takes 25 additional correct final diagnoses.

### What the reader could see

All 84 displayed center crops matched the model pixels. Cropping excluded 34.1–34.7% of each image area. A simple color measurement placed a median 34.0% of colored pixels outside the crop. Color may include artifacts. One A40 then processed 252 windows across 21 cases. All 84 center descriptors matched the archived originals exactly. Typical similarity between the center and left or right windows was 0.893 or 0.902 on a cosine scale where 1 means identical direction. Those windows overlap by about 73–74%. The same-zoom median over 840 different-box center pairs was 0.214. These measurements establish how shifting the window affects the reader. Whether the missing edges contain subtype-defining tissue remains open. [Coverage measurements](evidence/SINGLE_OUTPUT_CROP_COVERAGE.json) · [GPU probe](evidence/SINGLE_READER_EDGE_PROBE.json).

### Evidence and next decision

The model, error-analysis and image-coverage reviews were allocated to three Astra agents. The independent NumPy reconstruction reproduced all 255 selected validation decisions and all 735 group-held-out decisions. The first cluster job fitted the model and then stopped because its checker compared 64-bit statistics directly with the model's stored 32-bit copies. The corrected checker verified the original weights on the cluster. The correction changed no training or prediction. No clinical review has been completed. [Full fit history](evidence/SINGLE_OUTPUT_CENTER_FIT.json) · [Selected validation replay](evidence/SINGLE_OUTPUT_CENTER_REPLAY.json) · [Training-group replay](evidence/SINGLE_OUTPUT_CENTER_GROUP.json).

Have a qualified reviewer document a pattern in each full field before revealing its center. Separate patterns outside the crop from patterns that become unreadable when reduced to 224 pixels. If recurring omissions explain subtype differences, test wider coverage with the same one-output model and fixed decision rule. If the relevant cues survive the crop, seek additional independent rare-type cases and verify the diagnosis labels. The stronger validation result and weaker group accuracies make an independent set important for estimating future performance.

## Registered comparison

Use only BreakHis. The 49 fitting cases and 17 validation cases, the 15 nonempty combinations of the four microscope magnifications, and the frozen UNI2-h center features match the previous reference. The previously used 16-case follow-up stays outside the run. Each validation case contributes 15 answers. The 17 cases belong to 15 conservative groups.

Fit one linear layer that gives five scores for benign, ductal, lobular, mucinous and papillary. Train it with one five-category loss. Each class has equal total fitting weight to give the two lobular and four papillary training cases a voice. At prediction, convert the five scores to fractions summing to one. If the four malignant fractions sum to at least one half, return the largest malignant score. Otherwise return benign. This fixed arithmetic makes both answers from a single output. The old reference learned a binary score and four subtype scores with different loss terms.

Use seed 812, AdamW, learning rate 0.001, weight decay 0.01 and 40 full-batch updates. At updates 10, 20, 30 and 40 calculate macro F1 separately for each zoom condition, then average. Keep the first update with the highest value. Score five-category and binary accuracy at that same update. The strict targets require at least 230 and 243 correct answers out of 255 respectively. Report class counts, both stages' errors and all-four-view performance. Recompute scores independently from the weights and pinned inputs. Comparison with the original changes classifier capacity, training loss and routing together. It evaluates the complete design. [Registered settings](../configs/breakhis-single-output-center-v1.json).

## Field review

Use the fixed anonymous packet of 84 fields from 21 training and validation cases. Compare the original field with the exact crop that reached UNI2-h. Record geometry and a simple color proxy separately from the qualified reader's observations. A supplemental feature check will run the same frozen reader on left, center and right windows at the same pixel scale and measure how similar their outputs are. This check uses no tumor labels or classifier fitting. An expert's observations are needed to tell whether excluded tissue carries cues that distinguish malignant subtypes. The next decision about wider coverage depends on those observations.

## Supplemental group check

After the primary fit produced a validation result, register a separate check before computing its outputs. Use only the 49 original training cases. Leave out each of their 42 conservative groups in turn, train on all remaining groups, then predict the omitted cases at all 15 zoom conditions. Recompute class weights and normalization in every fold. Fix ten updates for every fold, matching the earlier group study, with one seed and no model selection. Save fold weights and predictions. Independently replay all 735 answers. Compare with the two previously completed group checks under the same case assignments. This is a robustness check within a reused development cohort.
