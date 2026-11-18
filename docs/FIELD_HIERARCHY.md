# Can one model recognize useful tissue patterns before averaging the pictures?

## What happened

**Keep the original shared classifier as the capstone reference.** Changing the order of pooling improved binary routing by two answers on validation. Both methods correctly identified 193 of 255 final categories. The new model lost three correct lobular answers and gained three correct ductal answers. Its macro F1, which gives all five categories equal weight, fell. The two models each use one reader and one jointly trained classifier for both decisions.

| Evaluation | Where the shared layer is applied | Five-category accuracy | Benign/malignant accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| 17 validation cases | After averaging (original) | 193/255 (75.69%) | 229/255 (89.80%) | 0.4982 |
| 17 validation cases | Before averaging (candidate) | 193/255 (75.69%) | 231/255 (90.59%) | 0.4738 |
| 49 cases, one group held out at a time | After averaging (original) | 484/735 (65.85%) | 673/735 (91.56%) | 0.4492 |
| 49 cases, one group held out at a time | Before averaging (candidate) | 506/735 (68.84%) | 690/735 (93.88%) | 0.4720 |

The original validation set contains 15 conservative groups. Each of its 17 cases contributes 15 answers. Both arms selected update 10 by the fixed macro F1 rule. The 49-case check used 42 groups drawn exclusively from the original training role. Each fold refit both models for ten updates with its held-out group excluded. That check favored the field-first model by 22 final answers and 17 binary answers, and it still missed every papillary combination. It estimates performance across these training groups with 42 different fitted checkpoints per method. The two rows of evidence should be kept separate. Neither evaluation reaches the two accuracy goals together.

### Which cases need attention?

| True category | Training cases | Validation cases | Original correct validation answers | Field-first correct validation answers |
| --- | ---: | ---: | ---: | ---: |
| Benign | 14 | 5 | 75/75 | 75/75 |
| Ductal | 24 | 7 | 98/105 | 101/105 |
| Lobular | 2 | 2 | 5/30 | 2/30 |
| Mucinous | 5 | 2 | 15/30 | 15/30 |
| Papillary | 4 | 1 | 0/15 | 0/15 |

Both lobular validation cases share one conservative identifier group. Papillary also has only one validation group. The repeated zoom answers carry much less independent information than the event counts suggest.

The candidate made 24 routing errors and 38 subtype errors after correct routing on validation. The control made 26 and 36, respectively. For the control, supplying the true benign/malignant answer while keeping its actual subtype choices would yield only 196/255 correct (76.86%). Fixing the first decision alone cannot deliver 90% on these cases. Both models scored 13/17 with all four zooms present. The worst single subset accuracy was 64.71% for the control and 70.59% for the field-first model.

The descriptive paired 95% interval for the field-first minus control difference in five-category accuracy spans −3.16 to +3.53 percentage points. This resamples the 15 validation groups with all their cases and masks together and conditions on the selected checkpoints. The sample and repeated use of these cases limit generalization. The candidate failed its improvement screen.

### What to do with these results

Keep the 75.69% five-category and 89.80% binary pair from the original model together in the capstone. Report the 90.59% binary result with the field-first model's own 75.69% five-category accuracy. The greater-than-90% and greater-than-95% goals refer to one selected checkpoint. The original 16-case role received zero predictions in this study.

A private review packet now pairs 84 original fields from 21 fixed training and validation cases with their center crops at the model's 224-pixel input size. The center crop covers 65–66% of the resized image width for these fields. Whether diagnostic tissue lies outside this region is an open question. A qualified reviewer can record what is visible in the whole field first and then in the crop. If distinguishing features are repeatedly lost at the edges, a predeclared wider-coverage comparison with the same single reader would test that finding. If the cues are present, seek more independent subtype examples and review their labels before another model study.

### Run evidence

The primary cluster job completed in 42 seconds. An independent NumPy implementation matched all 255 decisions for each selected checkpoint, recalculated every checkpoint's metrics, and verified all 1,584 validation field vectors against the four pinned UNI2-h shards. The group collector replayed all 84 fold checkpoints, with identical logits and labels. Both experiments reuse frozen UNI2-h descriptions. Original fields, per-case predictions, review keys and weights stay private. [Comparison and thresholds](evidence/FIELD_HIERARCHY_COMPARISON.json) · [Control training history](evidence/FIELD_HIERARCHY_CONTROL.json) · [Field-first training history](evidence/FIELD_HIERARCHY_MODEL.json) · [Independent validation replay](evidence/FIELD_HIERARCHY_REPLAY.json) · [42-group results and replay](evidence/FIELD_HIERARCHY_GROUP_CHECK.json).

## Comparison fixed before fitting

Our last shared classifier got 193 of 255 five-category answers right on validation and 229 benign/malignant decisions right. It first averaged the UNI2-h descriptions of all pictures for each available zoom. That averaging can hide differences within a case. Whether those differences help identify malignant subtypes is an experimental question. Expert review of the fixed fields is pending. [Previous run and class counts](SINGLE_MODEL_DUAL_STEP.md).

**The single change.** In the new model the same 32-unit layer looks for patterns in each field first. A nonlinear activation can give two fields different responses even when averaging their original features would hide that difference. The network averages these responses within each zoom and then across the available zooms. The existing model applies that layer after the pictures are averaged. Both methods use equal weights for fields within a zoom, equal weights for active zooms, the same mask flags, and the same normalization based on the active case mean. Each method has a single UNI2-h reader and 49,477 trainable classifier parameters. Each makes both decisions in one forward pass. Each field receives supervision only through its case's final loss.

**What stays matched.** BreakHis, the same original center-cropped field descriptors, 49 training and 17 validation cases in 42 and 15 disjoint provider groups, respectively, all 15 nonempty subsets of 40×, 100×, 200× and 400×. Train-only normalization, first-layer widths, dropout, losses, class weights, hard routing at zero, optimizer, learning rate, seed 812 and 40 full-batch updates follow the earlier shared classifier. Save all four validation checkpoints' predictions. The first maximum of mean per-subset five-category macro F1 at updates 10, 20, 30 and 40 selects each model's checkpoint. Score both accuracy targets at that same checkpoint. The existing 16-case follow-up is excluded from fitting, selection and inference in this experiment.

**How to judge it.** Across 17 cases × 15 zoom combinations, strict accuracy above 90% on the five final labels requires at least 230/255 correct. Strict binary accuracy above 95% requires at least 243/255. Both must hold for the same model. For a smaller development improvement, require both correct counts to increase against the reproduced control and macro F1 to hold or improve. Report both outcomes even if this screen fails. Include the rare-type recalls, first-step and second-step errors, all-four and worst-mask performance, and paired descriptive intervals resampling whole conservative groups. Selection and comparison reuse the same validation cases. Fresh confirmation requires independent cases.

**Verification.** Match the prior control counts, F1 and selection for all four checkpoints, verify that freshly packed fields reproduce the audited case means, and replay each selected network from its saved weights with a separate implementation. Keep original predictions, weights and review materials private. Publish the aggregate results, evidence digests and replay report. [Fixed configuration](../configs/breakhis-field-hierarchy-v1.json).

## Additional check within the original training set

Before launching the new models we also fixed a check on 49 training cases. Temporarily leave one of the 42 conservative case groups out, fit each model on the other groups, and predict the held-out group. Repeat until every training case has been predicted once under all 15 zoom conditions. Each classifier learns from 46–48 cases per fit. Recompute normalization inside every fold and use exactly ten updates for every arm and fold, the checkpoint selected in the previous pilot. The 17 validation cases play no part in these fits or predictions. Summarize the resulting 735 answers as one secondary estimate, separately from the 255-answer validation comparison. All cases have informed earlier project development. This check measures sensitivity to which training cases are available and provides no independent external confirmation. [Supplemental protocol](../configs/breakhis-field-crossfit-v1.json).

The group replay accepts a maximum absolute logit difference of 0.00001 across CPU nodes and requires identical final decisions in every held-out group. It reports the actual difference for each arm. Applying a shared transform to set elements before aggregation has established precedent. Here the question is whether changing that order helps on this particular small histology cohort. [Zaheer et al., Deep Sets](https://papers.nips.cc/paper/6931-deep-sets).
