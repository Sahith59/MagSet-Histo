# How we train, in plain language

Updated November 12, 2026. Current implementation: frozen image reader plus a trained decision maker. The repository's working name does not mean that the latest model is a transformer.

## The two pieces

The **picture reader** is an externally pretrained image encoder. It turns an image into numbers. ResNet-50 gives 2,048 numbers; Phikon-v2 gives 1,024. We keep its existing weights fixed in the current experiments. This is often called frozen-feature transfer learning.

The **decision maker** learns which numerical descriptions go with which class. The latest pilot uses a small regularized multinomial logistic classifier. It learns from the fitting cases, not from the reserved cases.

## One training example

Start with the related pictures of one case. Normalize image features using rules learned from fitting cases only. Average pictures within each available zoom; then average the zoom summaries and add four yes/no availability flags. The classifier receives this case description and the correct training label.

The classifier changes its decision weights to reduce its mistakes. Cross-entropy penalizes giving the correct class too little probability. The L2 penalty discourages excessively large decision weights. These penalties try to improve learning; they do not guarantee good performance on new patients.

## Practise with missing views

For a case with four views, training uses all 15 nonempty combinations. A case with three available views has seven combinations. Each original case contributes total training weight one across its combinations. A case with more pictures or combinations does not count as more people.

The native pilot has 80 fitting patients, five per class. Its 1,152 derived rows therefore have total weight 80 and class weight five. We do not apply an additional row-count balancing rule. Fitting-only normalization gives each case equal influence at each view it possesses.

## Practice questions and final exam

Sixteen native development patients provide the small practice check. Thirty-two reserved patients are excluded from classifier fitting, normalization, tuning, and current scoring. Frozen, label-blind image encoding was allowed for all roles. That preparation does not let reserved predictions guide model choices.

BreakHis uses a different exploratory repeated-trial protocol: 46 fit /8 validation /28 outer-role cases per trial. Across development every one of its 82 provider cases has been used for fitting or validation somewhere. It has no globally fresh confirmation cohort, and clinical identifier relationships remain unresolved.

## Exact latest pilot settings

All four arms used the same frozen ImageNet ResNet-50 features, fitting cases, normalization and subset weights. We fixed logistic C=0.1, L2, LBFGS, tolerance 0.0001 and at most 1,000 solver iterations. These are optimization iterations, not 1,000 passes through the original image dataset. There was no parameter grid.

The three augmented arms shared a fixed 32D projection (seed 812). Affinity temperature was 0.2. Four fits were completed before development scoring; each model and its preprocessing reloaded successfully. [The specification](NATIVE_DEVELOPMENT_PILOT_SPEC.md) defines every setting.

## What happens later?

First establish a useful combining mechanism against strong controls. If warranted, a later registered stage can update a limited part of the picture reader using fitting cases only. That is fine-tuning. Every compared method must share the same encoder/adaptation policy so that a stronger reader is not mistaken for a better fusion mechanism.

Fine-tuning and a larger reader are future possibilities. They were not run in the four-fit pilot. GPUs were used to encode images; the cached-feature classifier fits efficiently on CPUs. The latest agreement arm did not beat duplicate features, so the next work is the [audit and experiment plan](NEXT_STEPS.md), not an automatic architecture or hyperparameter sweep.
