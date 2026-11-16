Latest November16 result: **75.67%/75.00%** final two-step accuracy and macro F1.5897/.5643 for the completed Midnight similarity classifier, on the same splits and15-subset scoring. This supersedes72.00%/71.33% as the best current paired accuracy result. [Verified similarity comparison](../docs/BREAKHIS_KERNEL_RESULTS.md). One bounded last-layer adapter comparison is being implemented;94–95% remains unachieved. Earlier execution notes below retain their original scope.

# BreakHis five-category dual-step project report

Working capstone report, updated November16,2026. The target is a pooled benign label or one of four malignant types. The first trained comparison is complete and independently verified; the separately registered case-supervised CNN comparison has also completed. This is exploratory provider-case validation, not a fresh final patient test or a novel-method claim.

## Current measured improvement

The stronger frozen Midnight-12k pathology encoder raises two-step accuracy to **72.00%/71.33%**, from the first frozen-Phikon65.83%/68.83%, on identical provider roles and scoring. Macro F1 improves from.4975/.5325 to.5558/.5350. This is a measured development gain of6.17/2.50 percentage points in raw accuracy, below the94–95% goal.

| Two-step pipeline | Accuracy812 /813 | Macro F1812 /813 |
| --- | ---: | ---: |
| First frozen Phikon case means |65.83% /68.83%|.4975 /.5325|
| Case-supervised EfficientNet |46.50% /53.33%|.2346 /.3108|
| Frozen Phikon field supervision |66.67% /69.50%|.4975 /.5207|
| **Frozen Midnight case means** |**72.00% /71.33%**|**.5558 /.5350**|
| Frozen Midnight field supervision |73.67% /70.00%|.5732 /.5017|

The Midnight case pipeline retains the same fitting-only normalization, available-view mean and four flags, and balanced logistic router/expert; each image now supplies3072 frozen features (CLS plus mean patch tokens), rather than Phikon's1024. Its router scores91.17%/94.17%, but true-malignant typing scores60.00%/54.33%. Most remaining whole-system errors are incorrect malignant types even when the broad routing decision is correct. A higher binary accuracy is not five-category accuracy.

All20 Midnight paired case/field runs,240 head fits and129,600 dependent candidate case/subset records were checked against exact source/config/cache/role bindings before aggregation. Four complete tuning candidates per arm were preserved. Further registered studies test a support-example similarity classifier and one bounded standard-LoRA adjustment of the encoder. [Complete improvement results and provenance](../docs/BREAKHIS_IMPROVEMENT_RESULTS.md) · [Adapter recipe](../docs/BREAKHIS_LORA_SPEC.md).

The following detailed sections retain the first Phikon comparison as a historical reference. Its results are superseded as the best current consistent pipeline by Midnight case means, without changing their meaning or hiding failed experiments.

## Problem and inputs/outputs

Input: all images belonging to a provider case at its available40×/100×/200×/400× microscope views. Fields at different zooms are not assumed to be physically paired. Output: benign, ductal carcinoma, lobular carcinoma, mucinous carcinoma or papillary carcinoma. A binary router chooses benign/malignant; only the malignant decision needs a subtype answer.

## Data and evaluation

BreakHis supplies7,909images and82qualifiedproviderkeys. Only BreakHis images/labels are fitted/evaluated. External pretrained weights are transfer learning. Retain original roles before pooling benign labels:46fitting/eightvalidation/28excludedperrun, five overlapping trials at812/813. There are31distinctvalidationcases overall. Clinical grouping remains uncertain and all82keys entered development somewhere; the excluded28are excludedperrun, not a globally untouched test set.

Fitting supports are11Ben/25DC/2LC/5MC/3PC; validation4Ben/oneeachmalignanttype. Each case is evaluated in all15nonempty zoom combinations. Accuracy counts complete final answers; macro F1/balanced accuracy preserve minority-type visibility. Raw and balanced accuracy differ because benign has half the validation support.

## Implemented methodology

The first pair reuses frozen Phikon-v2 field features. Average within each zoom, standardize with fitting-only field moments, combine available zooms by equal means and append four availability flags. Direct-five uses one multinomial logistic classifier. Dual-step uses binary and malignant-four logistic classifiers, fitted on all46and true-malignant35cases respectively. Each head is balanced for its own classes; each case's weight is divided across15subsets.

Both arms select among four complete C settings[.01,.1,1,10]by end-to-end mean15-subset five-class macro F1; hierarchy heads share C. Threshold0.5is fixed, including malignant ties. The explicit routed label defines dual accuracy/F1. Composed probability argmax can differ and is not substituted; stable composed probabilities provide CE/Brier. Differing factorized/head-balanced objectives make this a complete-pipeline comparison.

## Results and execution

| Model | Arrangement | Five-class accuracy | Macro F1 | Balanced accuracy | All-four-view accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
|direct|812|66.67%|0.5161|58.27%|67.50%|
|dual|812|65.83%|0.4975|56.53%|67.50%|
|direct|813|64.67%|0.5035|58.07%|65.00%|
|dual|813|68.83%|0.5325|61.33%|72.50%|

Accuracy/F1 average equally over all15 nonempty zoom settings and five overlapping trials. The all-four-view column is a different endpoint. The two arrangements choose different validation cases; they are not different datasets.

Dual minus direct F1 is−0.0186/+0.0290 and accuracy−0.83/+4.17percentage points. The mixed result fails the prospectively fixed both-arrangement screen. Preserve both pipelines and every candidate; do not promote the favorable arrangement alone or claim that hierarchy reliably improves the task.

## Where the remaining errors occur

| Model | Arrangement | Binary accuracy | Malignant sensitivity | Benign specificity | True-malignant four-type accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
|direct|812|88.33%|100.00%|76.67%|56.67%|
|dual|812|87.83%|94.33%|81.33%|55.67%|
|direct|813|82.17%|96.00%|68.33%|54.33%|
|dual|813|88.33%|95.33%|81.33%|56.33%|

For dual, binary values describe the actual router. For direct, they are a separate derived sum-of-malignant-probabilities diagnostic, not the broad group of its final argmax. True-malignant typing evaluates the specialist/conditional subtype scores on known malignant cases; it does not count routing failures and is not whole-system accuracy.

The trained dual router is reasonably useful, but malignant typing remains around56%. On these saved outputs, perfect routing with unchanged typing would give about77.8%/78.2% overall. This is an oracle diagnostic bound, not a new trained result. The evidence supports improving representation/case learning, without proving a single biological or preprocessing cause.

## Completed case-supervised EfficientNet comparison

The20-run EfficientNetV2B0 comparison completed on A40 GPUs and passed both independent output checks. Direct-five validation accuracy was48.83%/49.00%, and the two-step pipeline scored46.50%/53.33% with macro F1.2346/.3108. The two-step fitting accuracy was87.36%/85.13%, showing a substantial fitting/validation gap. At that stage, frozen Phikon remained best at65.83%/68.83%; the later Midnight improvement above supersedes it. Rare-type recall and model generalization are the next diagnosis priorities. [Full accepted comparison](../docs/BREAKHIS_CASEBAG_RESULTS.md).

## Complete evidence and limits

All ten paired runs and120headfits completed; the zero-fit audit, full collector and independent review passed. Original C candidates, all trials/subsets/class confusions/fitting gaps remain available. [Detailed results and full ledger](../docs/BREAKHIS_FIVE_CLASS_RESULTS.md) · [Independent numerical review](../docs/reviews/FIVE_CLASS_ACTUAL_RESULTS_REVIEW.md) · [Prospective first recipe](../docs/BREAKHIS_FIVE_CLASS_SPEC.md).

The result does not reach94–95%. Malignant subtype recognition remains weak, including scarce lobular/papillary fitting cases. The separately declared EfficientNetV2B0 comparison is now complete and scored below the earlier baseline. No threshold/seed/resplit sweep or excluded-case rescue is included.

The earlier notebooks'93.73%/94.99% were historical eight-class40×image-random reused validation, not independently retrained. Earlier case8accuracy45.7%/38.3% and binary87.0%/80.7% retain their meanings; changing to five labels is not an eight-class improvement. [Historical evidence](../docs/CAPSTONE_BEFORE_AND_NOW.md). No final clinical validity, new mechanism, conference-ready claim or institutional acceptance is established.
