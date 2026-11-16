# How our current BreakHis system learns

Updated November16,2026. Imagine a student looking at several microscope photographs of the same tissue. The student first asks, “Is it benign or malignant?” If malignant, the next question is, “Which malignant type?” Our five possible final answers are benign, ductal, lobular, mucinous and papillary carcinoma.

## What goes in and comes out

Input is one provider case's tissue photographs at the available40×/100×/200×/400× zooms. Different fields across zooms need not show the exact same physical spot. Output is one final category, with probabilities describing the model's confidence. We evaluate every one of the15 nonempty zoom combinations.

## Current best model: compare with fitting examples

1. **Read the photographs.** A pretrained pathology image reader, Midnight-12k, turns each image into3,072 numbers describing visual patterns. We keep this reader frozen: its original weights do not change.
2. **Make descriptions comparable.** For each zoom, calculate a center from fitting cases only, with equal case influence. Subtract it and normalize the description's length.
3. **Compare examples.** Compare a new image with fitting images from the same zoom using cosine similarity. A temperature controls how strongly close matches dominate the vote. Each case has equal influence inside its class, regardless of image count.
4. **Combine the evidence.** Calculate benign/malignant and malignant-four probabilities for fields. Compose their five-category probabilities, average fields within each zoom, then average the available zooms equally.
5. **Give the two-step answer.** If case malignant probability is below.5, answer benign. Otherwise choose the highest-probability malignant type. Exactly.5 routes malignant.

The fitting examples and their labels form this classifier's reference set. There are no newly optimized classifier coefficients in this similarity method. The reader itself was previously trained by its authors; our fitting/evaluation images and labels are BreakHis only. External pretrained weights are transfer learning.

## The split, in simple terms

Each run has46 fitting cases,8 validation cases and28 excluded cases. Related images and all zooms stay together. We repeat the comparison across five overlapping trials and two validation arrangements,812/813. These are arrangements of the same dataset, not two datasets.

The46 fitting cases contain11 benign,25 ductal,2 lobular,5 mucinous and3 papillary cases. The malignant expert uses35 true-malignant fitting cases. Validation contains4 benign cases and one of each malignant type. We preserve the original eight-label roles before merging benign labels.

We choose among four temperatures using final five-category macro F1 averaged over all15 zoom subsets. Selection looks at validation only; excluded cases supply no predictions or fitting statistics. Fitting diagnostics remove the entire query case from both reference examples and centering, so an example cannot match itself.

These validation arrangements have been reused during development. They contain31 distinct provider keys overall, with unresolved clinical identity relationships. They are not a fresh final clinical test.

## What the scores mean

**Current final accuracy is75.67%/75.00%.** That means roughly75–76 correct final answers per100 case/zoom decisions in these development arrangements. They are repeated, dependent decisions, not100 independent patients. Macro F1 is.5897/.5643 and gives minority classes more visibility than raw accuracy.

The benign/malignant decision is92.0%/92.7% correct. The malignant specialist gets56.7% of known malignant types correct. Neither number is the whole-system five-category accuracy. With all four zooms present, final accuracy is82.5%/80.0%; that is a separate setting. [Complete verified result](BREAKHIS_KERNEL_RESULTS.md).

## Next training experiment: adjust a small part of the reader

The next fixed recipe uses standard LoRA adapters in the final four reader blocks. It trains about114,000 adapter/head parameters while freezing the original encoder weights. Heads start from the fixed C=1 field model for the same fitting cases, rather than a validation-selected initialization.

For20 epochs, visit every fitting case once per epoch and sample two photographs from each zoom. The first two epochs update heads; the next18 update adapters and heads. Give rare case types more loss weight. Use the same image sampling across direct-five and two-step arms. Evaluate complete models at epochs2/6/12/20 and select by the same final macro F1 criterion. An epoch2 selection would identify a head-only result, not learned-adapter benefit.

There are20 matched runs: ten roles for each arm. Before launching them, fitting-only GPU checks exercise warmup and real adapter gradients, verify unchanged original weights and test saving/reloading. The completed runs then need output verification. [Fixed recipe and resource budget](BREAKHIS_LORA_SPEC.md).

The94–95% final-accuracy goal remains unmet. We retain improvements and failed models, use the same comparisons, and report the numbers the experiments produce.
