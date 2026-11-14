# Capstone presentation outline

Draft story and speaking prompts, November14,2026. This is an outline, not a finished PowerPoint. Target a reviewable deck/report by November17, revisions on November18 and a presentation around November20. Exact presentation date and rubric are not supplied.

## 1. Breast histology classification across magnifications

Purpose: explain the task before discussing a model. Input is tissue images from a provider case at available40×/100×/200×/400× magnifications. Output is eight subtype scores; their source-defined grouping also gives benign/malignant scores. This is a research prototype.

Speaking prompt: “We ask the computer to read microscope pictures. Recognizing the broad benign/malignant category and naming the exact tumor subtype are different questions.”

## 2. Dataset and independent examples

Show7,909 images, four magnifications and eight subtypes. Distinguish82 subtype-qualified provider keys,81 bare IDs and70 numeric stems; clinical relationships remain unresolved. Many fields come from the same case.

Speaking prompt: “More photos of one case do not become more independent cases.”

## 3. Supplied experiments and our task

Show a protocol table rather than a misleading95%-versus40% improvement chart. Original notebooks: one40× image, image-random80/20 split, reused validation and EfficientNet variants; saved accuracy93.73%/94.99%. Present study: grouped provider cases, all four magnifications and15 nonempty availability subsets. Model/training/selection also differ.

Speaking prompt: “These are different exams. We have not measured a direct drop on the same test.”

## 4. Retained model and inference chain

Diagram: images→fixed official Phikon processor→frozen1,024D image reader→mean fields within zoom→fitting-only normalization→mean available zooms plus four flags→saved logistic classifier→eight probabilities→optional fixed benign/malignant sum.

State that no correspondence between equally numbered fields at different zooms is assumed. The Phikon processor resizes the shortest edge then center-crops; it does not encode every original pixel. Mean pooling is an established control, not a novel transformer.

## 5. Evaluation contract

Show46 fitting/eight validation/28 excluded roles per trial, five overlapping provider holdouts, inner arrangements812/813 and all15 subsets. Validation has one case per subtype; each arrangement contains22/20 distinct cases,31 in their union. No globally pristine confirmation cohort remains.

Explain macro F1 versus accuracy, class recall and why repeated view settings do not add people.

## 6. Eight-subtype results

Use checked full tables: frozen reference45.7%/38.3% accuracy andF1.3843/.3169; adaptation and supervised ResNet failed their improvement gates. Show the new regional test against its own fixed-C control separately, because the historical reference had a different C-selection budget.

Speaking prompt: “The pipeline works and its numbers reproduce, but exact subtype recognition remains weak.”

## 7. Benign/malignant secondary result

Use the independently checked binary secondary report with both arrangements, fixed0.5 threshold and all three accepted earlier pipelines. Show accuracy, sensitivity/specificity and AUROC with correct names. Explicitly label retrospective reused validation, derived from eight-class scores and not a newly trained binary model or an eight-class improvement.

Speaking prompt: “The model often gets the broad group right while confusing subtypes within that group.”

## 8. Generalization gap and rare classes

Show fitting versus validation balanced accuracy: supervised ResNet about99.6–99.7% fitting versus37–40% validation. Each fitting set has one adenosis and one phyllodes case. The image classifier is weak too; changed averaging alone did not rescue results. No single causal explanation is established.

## 9. Prediction demonstration

Use the lexicographically first validation case already chosen before its outcome, not a newly selected successful example. The saved model's121 raw fields reproduced all15 archived outputs within5.52e-08. Display eight scores and their fixed benign/malignant sum while toggling available views. Label any interactive saved-output viewer as a replay; keep the actual model command available.

## 10. What is complete and what comes next

Complete: data/source audit, reproducible training/inference, fixed-role baselines, negative comparisons, independent numerical checks and a prototype demonstration. Open: stronger subtype transfer, clinical identity reconciliation, genuinely fresh confirmation and any novel mechanism. Finish the master's capstone first; research-paper development follows with explicit evidence and data requirements.

Each final slide should cite the relevant result/review file in its notes. Keep historical results, current validation, final-test claims and future proposals distinct.
