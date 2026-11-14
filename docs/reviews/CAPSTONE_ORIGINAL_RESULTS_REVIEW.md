# Independent review of original saved BreakHis results

Review date: November 14, 2026. Scope: read-only inspection of the two supplied notebooks, their saved outputs, the supplied paper and poster, and the existing project review/walkthrough. No training, checkpoint execution, new dataset evaluation, network or cluster action was performed. Originals were not changed. These are historical saved results, not independently reproduced performance.

## Provenance

SHA-256 of the exact inspected original bytes:

| Original local file (excluded source material) | SHA-256 |
| --- | --- |
| `non-localattention-breakhis.ipynb` | `8321e316e03364883b8138901df2630849d3888ee758c694bb3180b3139045bc` |
| `spatialsequenceattention-breakhis.ipynb` | `b3e2c0f1d1c076519e88df0ab84863171486b8129046cb12c73487ccf908aa58` |
| `Breast_Cancer_SPIE_final.pdf` | `f8d5aeb1088f8bfa68f1694f06074baf07221f4182559c2dacb27554af734cfd` |
| `Lang_2021SPIE_Poster.pdf` | `194de781ed12f38bf04141d1af98410f1116ffd358804fd8bb46f1c67e3529ab` |

These are supplied local reference artifacts, not evidence of the student's authorship, a rerun, or acceptance of a new paper. Notebook cells have **no `id` fields**; evidence below uses one-based physical cell position and saved execution count, which are distinct. Both notebooks have 15 cells, including an empty final cell. Existing `docs/PROJECT_REVIEW.md` and `docs/NOTEBOOK_WALKTHROUGH.md` agree with the independently parsed saved metric text and source inspected here. No original material needs to enter Git to preserve this aggregate audit.

## What the notebooks actually achieved in their saved outputs

| Saved endpoint | Non-local attention | Spatial sequence gate |
| --- | ---: | ---: |
| Final epoch (50) displayed training accuracy | 0.9972 = 99.72% | 0.9977 = 99.77% |
| Final epoch displayed validation accuracy | 0.9348 = 93.48% | 0.9449 = 94.49% |
| Highest checkpoint validation accuracy in log | 0.93734 at epoch 34 | 0.94987 at epoch 23 |
| Cell 12 printed accuracy | 93.73% | 94.99% |
| Saved confusion matrix correct / total | 374 / 399 | 379 / 399 |
| Accuracy recomputed from saved matrix | 93.7343358396% | 94.9874686717% |
| Macro F1 recomputed from saved matrix | 0.9344778089017155 | 0.9422209305771938 |
| Cell 12 printed macro precision | 94.64% | 95.45% |
| Cell 12 printed macro recall | 92.66% | 93.28% |
| Cell 12 printed macro F1 | 93.45% | 94.22% |
| Cell 12 printed quadratic kappa | 93.36% | 96.87% |
| Cell 12 printed macro OVR ROC AUC | 99.44% | 99.19% |

Training values are the **displayed Keras terminal lines**, rounded to four decimals. They are not a separately saved full-precision history array or a held-out score. For example, spatial epoch 50 has a transient progress line of 0.9976 followed by its completed epoch line of 0.9977; the table uses the completed line. Neither 99.72/99.77% training accuracy nor 99.44/99.19% AUC is the eight-class evaluation accuracy.

Evidence locations:

- Cell 2 selects only folders named `40X`; cell 3 resizes individual RGB images to 224×224 and assigns subtype targets.
- Cell 4: non-local execution 7; spatial execution 4. Saved shapes are 1,596 training and 399 evaluation images, totaling 1,995. Source uses `LabelEncoder` and `train_test_split(..., test_size=0.2, random_state=42, stratify=target)`.
- Cell 6 defines eight softmax outputs and `base_model.trainable = True`, despite a freezing comment. The backbone is ImageNet EfficientNetV2B0, not the paper's ResNet-50.
- Cell 8: non-local execution 11; spatial execution 8. The source trains 50 epochs, batch size 64, Adam, sparse categorical cross entropy. Its checkpoint and learning-rate callbacks both monitor `val_accuracy`; `validation_data=(X_test, y_test)`. Early stopping is constructed but omitted from `callbacks = [checkpoint, rlrop]`.
- Non-local cell 8 logs its last checkpoint improvement at epoch 34: `0.93484 to 0.93734`; spatial logs the last improvement at epoch 23: `0.94737 to 0.94987`. This separates selected checkpoint results from final-epoch curves.
- Cell 10 begins `model.load_weights(filepath)`. Non-local execution 13 has a saved prediction figure. Spatial execution count is `None`, yet outputs contain convolution algorithm numerical-mismatch warnings. Thus a clean, ordered execution cannot be certified for that cell. Its later matrix agrees with the best validation accuracy and differs from final-epoch accuracy; that is consistent with reloading the checkpoint, but does not prove checkpoint identity without weights/prediction records.
- Cell 11 predicts on `X_test` (non-local execution 14; spatial execution 11). Cell 12 computes the displayed metrics (non-local execution 15; spatial execution 12). Matrix row supports are `[23,173,50,31,41,29,22,30]` in both saved outputs; diagonal sums are 374 and 379. Recalculation here used the printed integers, not inference. F1 was calculated as the mean of `2*CM[i,i]/(row_sum[i]+column_sum[i])`.

The confusion matrices are already saved as machine-readable text, so no plot OCR is needed for these numbers. The original training-curve figures are visualization of history, not an additional independent evaluation.

## Protocol and interpretation

Both notebooks address **eight-subtype classification of one 40× image**. They do not combine four magnifications per case. Non-local attention relates locations within one feature map. The spatial gate expands one feature map into a sequence of length one before ConvLSTM; it is not a sequence of magnification views. Neither implements missing-view evaluation.

The split is stratified by subtype at the **image level**. It does not enforce provider-case/patient separation. Actual historical overlap cannot be established from notebook code or summary shapes alone: the original split membership, generated CSV and complete image inventory are not supplied with these artifacts. Random state 42 does not guarantee equal membership across the two runs when image discovery order is not fixed. Equal row supports do not establish paired predictions.

The same 399-image partition is used for epoch validation, checkpoint selection, learning-rate adaptation, and reported evaluation. It is therefore a reused validation partition, not an untouched outer test set. A high score is evidence about that saved experiment, not established new-patient generalization. The numerical matrix metrics are not invalidated merely by the qualitative display bug: cell 10's unordered `set` mapping and scalar-label `argmax` corrupt displayed label names, whereas cell 12 compares encoded integers directly.

The spatial matrix has five additional correct predictions (1.2531 percentage points). It does not establish a statistically reliable architectural advantage: no paired predictions, exact shared split, repeated seeds, independent test or uncertainty estimates are available. Numerical mismatch warnings are an additional provenance limitation, not proof that every saved metric is wrong.

## The supplied paper reports a different experiment

Independent PDF text extraction identifies the following evidence (page numbers are one-based PDF pages):

| Evidence | Exact reported result and meaning |
| --- | --- |
| Paper p.4, experimental setup and Table 4 | Reports patient-disjoint approximately 70/30 train/test allocation, with within-subtype patient reshuffling across folds. Table 4 malignant train/test counts: DC 27/10, MC 7/2, PC 4/2, LC 3/2. |
| Paper p.6, Table 5 | Binary overall accuracy: 40× 81.23%, 100× 83.08%, 200× 82.13%, 400× 81.88%, all-magnification classifier 86.22%. |
| Paper p.6, Table 6 | Single-pass five-class classifier collapsed to binary: 85.32% binary accuracy; proposed dual-step: 86.22%. Malignant subtype recall: 55.83% versus 57.06%. |
| Poster p.1, Table 2 | Repeats 85.32%/86.22% binary accuracy and 57.06% proposed subtype recall, but baseline subtype recall is **55.89%**, conflicting with paper's 55.83%. |

The paper's pipeline extracts ImageNet ResNet-50 features, then uses binary XGBoost; predicted malignant images pass to a four-malignant-subtype XGBoost classifier. Its five final labels are pooled benign, DC, LC, MC and PC. “All magnifications” means training a classifier that accepts individual images from different magnifications; it is not joint four-view fusion. Paper preprocessing uses resize to 256×256, training rotations/flips, center crop to 224×224 and ImageNet normalization, unlike the notebook resize-only loader.

**86.22% is binary accuracy, not five-class or eight-class accuracy.** The paper does not supply runnable implementation, raw predictions or a hash-bound split here; these are author-reported historical claims. The source's 37-versus-38 DC patient discrepancy and incomplete fold provenance prevent claiming exact reproduction. The poster baseline subtype-recall conflict must be preserved, not silently resolved by choosing one number. Clinical improvement or lives saved is not established by retrospective image classification.

## Relationship to today's BreakHis capstone results

The current provider-case eight-class evaluation, averaging across all 15 nonempty magnification subsets, has different sample units, grouping, input availability, encoder/training and endpoint aggregation. Current frozen results (45.6667% / 38.3333%, macro F1 0.3842778 / 0.3168889) and supervised reference results (37% / 40%, macro F1 0.3083889 / 0.3006111) are supplied contemporary context, **not independently re-audited in this original-artifact review**.

Do not describe the gap from the notebooks' 93.73/94.99% as measured model degradation. It is a comparison across different endpoints and splits, with an especially consequential change from image-split reused validation to grouped case evaluation. A master's capstone can truthfully report: historical single-image experiments saved high reused-validation accuracy; the current project implements a more demanding grouped, missing-view protocol; predictive quality under that protocol still requires improvement. Novelty and publication claims are separate later questions.

## Facts versus remaining uncertainty

Verified here: original byte hashes; notebook source protocol; saved final and best log values; saved matrix arithmetic; paper/poster endpoint meanings and discrepancy. Not verified: historical dataset integrity, exact patient overlap or paired membership, model checkpoint bytes, raw probability arrays/AUC recomputation, clean sequential execution, independent patient-level performance, or reproduction of the paper. No new research performance was generated by this review.
