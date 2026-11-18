# Does seeing more of each microscope photo improve the same dual-step model?

## Result and capstone decision

**Keep the original case-first UNI2-h.** On the same 17 validation cases, covering more of each image gave three additional correct five-category answers across two ductal cases. It lost one correct benign/malignant answer. Lobular and papillary recognition were unchanged. The three independent reviews agreed that this candidate falls short of the predeclared rule. [One-page capstone brief](../output/pdf/BreakHis-five-category-brief.pdf) · [Full verified results](evidence/WIDER_CASE_HIERARCHY.json)

| 17-case validation, all 15 zoom combinations | Original center | Wider image |
| --- | ---: | ---: |
| Five-category accuracy | 193/255 (75.69%) | 196/255 (76.86%) |
| Benign/malignant accuracy | 229/255 (89.80%) | 228/255 (89.41%) |
| Five-category macro F1 | .4982 | .5063 |
| Correct ductal diagnoses | 98/105 | 101/105 |
| Correct lobular diagnoses | 5/30 | 5/30 |
| Correct papillary diagnoses | 0/15 | 0/15 |
| Five-category answers with all four zooms together | 13/17 | 13/17 |

The separate leave-one-group-out check on the 49 training cases found no five-category gain at any of four seeds. The center and wider counts out of 735 were 484 and 484, 498 and 498, 512 and 512, and 502 and 501. The validation gain was 1.18 percentage points. A paired bootstrap over 15 validation groups gave a descriptive 95% interval of 0 to +2.92 points for exact diagnosis and −2.35 to +1.05 points for binary accuracy. These intervals retain all 15 zoom answers from each case together.

Four A40 tasks processed 6,205 fields from 49 training and 17 validation cases. All 6,205 center embeddings matched the original reader output exactly. All 264 case-by-zoom means for each treatment were reconstructed from the field embeddings. Slurm job 4991052 completed the classifier comparison in 58 seconds. All 338 saved fits replayed successfully, and all 3,195 center-crop control predictions matched earlier records. A separate local calculation reproduced the counts, macro F1 and progression decision from all five paired prediction files. [Feature verification](evidence/BREAKHIS_WIDE_UNI_FEATURES.json) · [Model and scoring verification](evidence/WIDER_CASE_HIERARCHY.json)

The previously frozen original network scored **82.92% five-category and 96.25% binary** on the 16-case, 13-group retrospective benchmark. Each case was tested under 15 zoom combinations, giving 199/240 exact answers and 231/240 binary answers. With all four zooms together it got 13/16 diagnoses and 15/16 binary decisions right. The wider version has no benchmark score on these 16 cases. [Original scorecard](BREAKHIS_MODEL_COMPARISON.md)

## Decision fixed before the comparison

Use the original case-first UNI2-h network. Compare the original center crop with three overlapping left, center and right crops at the same scale. Give the three crop descriptions equal weight to produce one description for each field. Use one frozen reader, the same averaging within each zoom and across available zooms, and the same trainable network for benign/malignant and malignant subtype decisions. The 84-field review packet documents the image geometry. Whether diagnostic tissue is missing from a center crop still requires qualified review. This comparison tests whether showing more pixels changes classification accuracy.

Use only the original 49 fitting and 17 validation cases. Retain all 15 zoom combinations and all five diagnoses. Check the center features against the original source. Use the same ten updates, initialization, loss, train-only normalization and case grouping as the original network. On the 49 fitting cases withhold each of the 42 identifier groups once for each of seeds 812, 813, 814 and 815. On seed 812 also fit all 49 cases and score the two architectures on the same 17 validation cases once. Both arms use the ordinary original initialization. The existing 16-case follow-up stays closed to the candidate.

A candidate advances only if both accuracy counts increase on validation, validation macro F1 holds or rises, and the five-category accuracy increases in at least three of the four training seeds. Mean paired changes in training-group binary accuracy and macro F1 must also be nonnegative. Archive both results regardless of direction. Show per-case and rare-type changes and verified replay of the saved models. These evaluations reuse development data and cannot establish accuracy on new cases. [Previous benchmark](BREAKHIS_MODEL_COMPARISON.md) · [Earlier initialization experiment](BREAKHIS_SUBTYPE_INITIALIZATION.md)

## What the three reviews established

The original network already recognized the two lobular validation cases as malignant in 30/30 answers. Its second decision called 25 of those ductal. For the one papillary validation case, forcing a malignant decision still selected mucinous 15/15 times. Those observations motivated examining the image input. The rare cases give too few independent examples to attribute either error to a particular tissue pattern. [Archived case traces and training-case comparisons](CASEFIRST_RARE_DECISIONS.md)

A review of the actual cropped pixels confirmed that the usual center contains about two thirds of the photo's width. At the same scale, the three model windows together span the resized field. They overlap substantially, so the middle contributes to more windows than the extreme edges. This tests a specific way of using wider images. It cannot establish whether a person could see diagnostically important features at the edges. The private, blinded review packet now shows the original, center, left and right views for 84 fields from 21 development cases. The qualified reviewer’s observation sheet is blank. [Center geometry measured on the source images](evidence/SINGLE_OUTPUT_CROP_COVERAGE.json)

A separate code review checked group exclusions, training-fold normalization, exact center matching, and equal weighting per field, active zoom and training setting in both arms. All 16 previously benchmarked cases were excluded from the new image processing and predictions. The [original decision rule and criteria](https://github.com/Sahith59/MagSet-Histo/blob/2d161e3/docs/BREAKHIS_WIDER_FIELD.md) were committed before this study ran.
