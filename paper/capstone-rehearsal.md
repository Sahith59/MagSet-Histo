# Capstone: short explanation and questions

## A 90-second explanation

“We are teaching a computer to recognize eight types of breast tumors from microscope pictures. BreakHis is our main dataset:7,909 pictures at four zoom levels. There are many pictures of the same provider case, so we keep related pictures together when dividing the data.

“Our retained model has two parts. A pretrained picture reader stays fixed and turns each picture into1,024 numbers. We average pictures within a zoom, combine available zooms, and train a small classifier to choose among eight types. We practise and evaluate all15 nonempty combinations of the four zooms.

“The supplied notebooks saved about94–95% on individual40× pictures using an image-random split. Our current experiment asks a different question: classify a provider case across view combinations. Its eight-type accuracy is45.7% and38.3% in two validation arrangements. These are different experiments, not a measured95-to40 drop on one test.

“When we combine subtype probabilities into the separate benign/malignant groups, accuracy is87.0% and80.7%. That simpler answer does not improve exact-subtype accuracy. It is a retrospective reading of existing validation outputs, not a fresh test.

“We tested adaptation, supervised ResNet and regional crops. None met its improvement gate, so we kept the frozen reference. Our demo reproduces a saved case from121 original images. The capstone presents a working, reproducible prototype and a measured generalization problem; stronger subtype recognition and independent clinical confirmation remain future work.”

## Questions to practise

**What are the eight labels?** Adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor and tubular adenoma. BreakHis groups adenosis/fibroadenoma/phyllodes/tubular adenoma as benign and the other four as malignant. This is the dataset's grouping, not a universal clinical statement about every tumor with that name.

**What is one trial?** One complete fitting/validation arrangement and model. Each uses46 fitting cases, eight validation cases and28 excluded cases. Five trials overlap; the two inner arrangements812/813 vary validation-case choices. There are31 distinct validation cases across both arrangements, not thousands of independent patients.

**How does the small classifier learn?** It predicts subtype probabilities, compares them with fitting labels and adjusts its weights to reduce cross-entropy plus a weight-size penalty. Class balancing gives rare classes extra fitting weight; cases within a class contribute equally. Each case's mass is shared across its15 view combinations. Validation chooses among four fixed regularization settings; validation does not supply gradient updates. Normalization is learned only from fitting cases.

**What does frozen mean?** The pretrained image reader's weights do not change. We train the small case classifier. The adaptation and ResNet experiments used different, explicitly reported training recipes and did not replace this baseline.

**What is macro F1?** A score that rewards finding each subtype while avoiding false subtype predictions, then gives all eight types equal importance. It is not accuracy. Our primary result averages F1 over15 input combinations and five trials.

**Why are exact-subtype numbers low?** A large fitting/validation gap is measured. Each fitting set contains only one adenosis and one phyllodes provider case. Extra pictures or GPUs do not add independent cases. Transfer, preprocessing and objectives may also matter; we have not proven one sole cause.

**Is87% our new eight-type accuracy?** No.87.0%/80.7% answers benign versus malignant, derived with a fixed0.5 threshold from the same saved outputs. Eight-type accuracy remains45.7%/38.3%. The broad group is often right while subtypes inside it are confused.

**Is AUROC96% accuracy?** No. AUROC measures how well scores rank malignant above benign cases across possible thresholds. Accuracy counts correct decisions at our fixed threshold. Sensitivity and specificity show malignant detection and benign recognition separately.

**Is the demonstration live?** The original-image model replay ran on the cluster in39seconds of scheduled wall time. The viewer switches among its saved outputs; it accepts no uploads. One example is execution evidence, not an accuracy estimate.

**What is novel?** No new mechanism has been demonstrated yet. Mean pooling and logistic classification are conventional. The capstone evidence and reusable pipeline are complete engineering work; a novel-method paper requires a separate hypothesis, fair controls and fresh confirmation.

**Is this a clinical test?** No. Provider identifiers have unresolved clinical relationships, all82 keys entered development somewhere and there is no globally fresh BreakHis confirmation cohort. We report exploratory provider-case validation.

[Presentation](../docs/evidence/CAPSTONE_PRESENTATION.html) · [Report](breakhis-project-report.md) · [Verified numbers](../docs/CAPSTONE_BEFORE_AND_NOW.md).
