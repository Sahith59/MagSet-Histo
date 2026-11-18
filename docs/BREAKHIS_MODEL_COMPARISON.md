# BreakHis models compared on the same cases

## What to report

For the one-network, dual-step capstone, lead with the earlier case-first UNI2-h result of **82.92% five-category and 96.25% binary** (199 and 231 correct out of 240 answers). On the same 16-case retrospective benchmark, the later single five-probability output scored **60.00% and 79.17%**. It derives binary by summing four cancer probabilities. The earlier architecture learns the binary and subtype decisions together inside one network. The change in project priority came after these results were known. Both checkpoints and their original validation scores remain in the comparison. [Development of the earlier model](BREAKHIS_SUBTYPE_INITIALIZATION.md).

The wider-image candidate scored 76.86% five-category and 89.41% binary on the 17 validation cases. It gained three ductal answers and missed the same rare-type cases. It did not advance to a 16-case evaluation. [Wider-image comparison](BREAKHIS_WIDER_FIELD.md).

## What the two sets of cases mean

There are 82 BreakHis case folders in this split. The 49 fitting folders (4,621 images) teach the model. The 17 validation folders (1,584 images, 15 conservative identifier groups) help choose a saved version. The remaining 16 folders (1,704 images, 13 groups) were excluded from fitting and checkpoint selection for the matched methods. We later applied frozen versions to those 16 folders. Earlier project work had used those cases, which makes this a retrospective internal benchmark rather than independent external confirmation. Case folders are grouped using dataset identifiers; clinical patient identity has not been independently verified.

All rows below receive the original pictures for the same case under the same available zooms. We test each of the 15 nonempty combinations of 40×, 100×, 200× and 400×. This gives 255 validation and 240 follow-up answers. One folder therefore contributes 15 related decisions. Five-category correctness requires the exact label: benign, ductal carcinoma, lobular carcinoma, mucinous carcinoma or papillary carcinoma. Binary correctness only asks whether the case is benign or malignant. A true ductal case called lobular earns one binary point and zero five-category points.

| Case model | 17-case validation: five | 17-case validation: binary | 16-case follow-up: five | 16-case follow-up: binary |
| --- | ---: | ---: | ---: | ---: |
| Later UNI2-h, one five-category output | 205/255 (80.39%) | 253/255 (99.22%) | 144/240 (60.00%) | 190/240 (79.17%) |
| Earlier UNI2-h, case features then classifier | 193/255 (75.69%) | 229/255 (89.80%) | 199/240 (82.92%) | 231/240 (96.25%) |
| Earlier UNI2-h, classifier on each field first | 193/255 (75.69%) | 231/255 (90.59%) | 196/240 (81.67%) | 232/240 (96.67%) |
| CNN–Transformer, case adaptation | 151/255 (59.22%) | 198/255 (77.65%) | 136/240 (56.67%) | 207/240 (86.25%) |
| Swin-Tiny, case adaptation | 152/255 (59.61%) | 201/255 (78.82%) | 142/240 (59.17%) | 198/240 (82.50%) |
| Midnight P, separate decision rules | 203/255 (79.61%) | 238/255 (93.33%) | 162/240 (67.50%) | 227/240 (94.58%) |
| Midnight + UNI2-h, two readers | 197/255 (77.25%) | 238/255 (93.33%) | 199/240 (82.92%) | 227/240 (94.58%) |
| UNI2-h, two linear SVC classifiers | 195/255 (76.47%) | 247/255 (96.86%) | 179/240 (74.58%) | 221/240 (92.08%) |
| UNI2-h, two nonlinear SVC classifiers | 195/255 (76.47%) | 253/255 (99.22%) | 166/240 (69.17%) | 221/240 (92.08%) |

The two historically highest five-category point estimates are tied at 199/240. One belongs to the earlier single-network case-first classifier, the other to a two-reader system with a different binary score. No matched row clears both requested thresholds. At this denominator that would require at least 217/240 final and 229/240 binary answers correct from the same model. For the single-output comparison, all four zooms together gave one prediction per folder: 10/16 exact diagnoses and 13/16 correct binary decisions.

### Which parts of the models differ?

| Case model | Reader and input handling | How it gives the two answers |
| --- | --- | --- |
| Single-output UNI2-h comparison | Frozen pathology reader; center crop per image; average within each zoom, then combine available zooms | One linear layer yields five probabilities. Add the four malignant probabilities to decide binary, then choose the highest malignant type when appropriate. |
| Earlier UNI2-h pair | Same frozen center features. One model applies a learned nonlinear layer after pooling case features; the field-first model applies it before pooling field evidence. | One jointly fitted network has a learned binary decision and four conditional subtype scores. Both versions use the same number of parameters. |
| CNN–Transformer and Swin-Tiny adaptations | CNN uses EfficientNetV2-B0 plus four Transformer blocks. Swin uses shifted-window attention. Both use three crops per original field; training uses the same case split. | Both were fitted for a binary and a four-malignant-type decision. Field-level joint probabilities are averaged within zooms, then across the available zooms, for one case answer. |
| Midnight P, hybrid and SVC controls | P uses three-crop Midnight features and similarity rules. The hybrid adds a UNI2-h reader for subtype. The two SVC alternatives use a UNI2-h reader and separate linear or nonlinear fitted classifiers. | Each ultimately chooses benign or one of four malignancies. These are useful controls for the same prediction task with different decision designs. |

The common inputs are the same original fields, cases and available zooms. Each full pipeline keeps its own pixel preprocessing, pretraining and fitting method. The comparison measures whole pipelines. It cannot isolate which reader alone is better. [Matched CNN/Swin protocol](MATCHED_THREE_WAY_RESULTS.md) · [Frozen single-output benchmark](BREAKHIS_CAPSTONE_BENCHMARK.md) · [Older UNI2-h replay](evidence/MATCHED_SCORECARD_SUPPLEMENT.json) · [Hybrid results](evidence/BREAKHIS_SPECIALIST_READER_RESULTS.json) · [SVC results](evidence/BREAKHIS_UNI_CASE_SVM_RESULTS.json).

## What the notebooks achieved on their own images

| Supplied notebook and input | Evaluated images | Eight-category accuracy as supplied | Five-category image accuracy | Binary image accuracy |
| --- | ---: | ---: | ---: | ---: |
| Non-local attention (40×) | 399 | 374/399 (93.73%) | 377/399 (94.49%) | 399/399 (100.00%) |
| Spatial sequence gate (40×) | 399 | 379/399 (94.99%) | 381/399 (95.49%) | 398/399 (99.75%) |
| CNN–Transformer, original scaling | 1,582 | 690/1582 (43.62%) | 690/1582 (43.62%) | 1085/1582 (68.58%) |
| CNN–Transformer, corrected scaling | 1,582 | 1367/1582 (86.41%) | 1410/1582 (89.13%) | 1536/1582 (97.09%) |
| Swin-Tiny, original dual-head run | 2,116 | Five-label task | 1,362/2,116 (64.37%) | 1,976/2,116 (93.38%) |

For the two original 40× notebooks, we grouped their saved eight-class decisions into five labels and into two labels. Those two columns are newly calculated from the confusion matrices saved in the notebooks. Both notebooks classified individual 40× images on a stratified 1,596/399 image split and reused the 399 for checkpoint selection. Neither originally fitted a binary-then-subtype classifier. The high percentages describe a single-image task with image-level splitting.

The supplied CNN–Transformer used 6,327 training images and 1,582 reused validation images; all 82 case identifiers appear on both sides. Its authored outputs were a binary label and one of eight types. For the new five-label image score, we reused its saved binary decision and chose the highest of its four malignant-type scores whenever it called malignant. Correcting the input scale removed an extra division by 255 before the EfficientNet reader. The supplied Swin run already produced five labels. It used 5,104 training, 689 validation and 2,116 test images; some related identifier stems span roles. Both CNN and Swin included all four zooms as separate images. Their input unit and data split differ from the case benchmark above. [Original notebook audit](reviews/CAPSTONE_ORIGINAL_RESULTS_REVIEW.md) · [Original CNN and Swin runs](SUPPLIED_NOTEBOOKS_COMPARISON.md) · [Saved-image rescoring and hashes](evidence/NATIVE_NOTEBOOK_LABEL_COMPARISON.json).

## Improvement attempt with one five-category output

Giving benign half the training weight improved the 42-group development check from 469/735 to 478/735 five-category answers (63.81% to 65.03%) and from 657/735 to 660/735 binary answers (89.39% to 89.80%). On the separate 17 validation cases, five-category accuracy fell from 205/255 to 197/255 (80.39% to 77.25%) while binary accuracy stayed at 253/255 (99.22%). Ductal recognition lost ten answers and lobular recognition gained two. Keep the original one-output checkpoint as the benchmark. The candidate has no 16-case score. [Group check](evidence/MATCHED_SCORECARD_SUPPLEMENT.json) · [Fixed validation check](evidence/SINGLE_OUTPUT_WEIGHT_VALIDATION.json).

The next improvement decision depends on review of the original full fields and center crops: check false malignant calls on benign tissue and the patterns distinguishing ductal, lobular, mucinous and papillary cases. If the center omits distinguishing patterns, compare wider coverage using one output and development cases. Additional independently grouped rare cases are needed to confirm gains.

## Additional checks fixed before execution

For the two older UNI2-h comparators, replay both fixed update-10 checkpoints with their original center features and original decision rules on the same 16 cases as the established benchmark. Check their saved validation results, the four input shard commitments, the training normalizers and every predicted label with a separate NumPy implementation. Retain all 15 zoom conditions and the original five diagnoses. These historical comparators do not participate in model selection.

For improvement, change only the relative class weights in the single five-output classifier. The previous equal-class loss gave the four cancer categories four times the combined weight of benign. In the earlier leave-one-group-out development check, it called 70 of 210 benign answers malignant. Give benign half the total fitting weight, leaving each malignant type one eighth. Use the same single linear layer, ten updates, seed, frozen input, 42 training-group exclusions and 0.5 routing rule. Only the original 49 training cases enter these fits and predictions. Compare with the committed predictions from the earlier 42-group study. Progress requires both accuracy counts to increase and mean per-zoom-combination macro F1 to hold or improve. Training groups were used in prior development. The frozen 16-case benchmark stays fixed. [Locked settings](../configs/breakhis-matched-scorecard-v1.json).

The training-group candidate passed its progression rule with 478/735 five-category and 660/735 binary answers correct, versus 469 and 657 for the original one-output model. Before opening the candidate's validation predictions, fix ten updates, seed 812, the same training-only branch weights, all 49 fitting cases and the original routing threshold. Evaluate once on the existing 17 validation cases and keep the 16-case benchmark outside this candidate follow-up. Development results across both checks will determine the next research decision.
