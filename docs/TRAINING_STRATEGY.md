# How we train the BreakHis baseline

Current priority: the BreakHis project with Professor Dong. The following describes its implemented baseline. TCGA has a different archived protocol and is deferred.

## Picture reader and decision maker

A pretrained **Phikon-v2 picture reader** turns each image into 1,024 numbers. Its weights stay fixed. We do not teach it again in the current baseline.

A small **decision maker**, a logistic classifier, learns from those numbers and the known eight-subtype labels. Its training penalty discourages wrong/confident answers and excessively large decision weights. This is frozen-feature transfer learning.

## What happens to a case

1. Gather all related images at its available 40×/100×/200×/400× views.
2. Read every field with the fixed image encoder and its pinned transform.
3. Average images within each zoom, using saved fitting-only normalization.
4. Average the available zoom summaries and add four yes/no availability flags.
5. Train the decision maker on fitting-case descriptions and labels.

The training code first pools fields and then applies the saved per-view normalization; this affine operation uses fitting-only field moments. The prediction replay reuses that exact ordering and vectorizer.

## Practise with missing zooms

All four views give 15 nonempty combinations. We train on all 15, distributing each case's base training mass across them. Pictures and masks do not become extra independent patients. BreakHis also uses class balancing: classes with fewer fitting cases receive extra weight, so common classes do not dominate. Cases within the same class have equal influence regardless of their field count.

## Split and model selection

Each of the five provider repeated trials has 46 fitting cases, 8 validation cases and 28 excluded outer-role cases. The validation group contains one case per subtype. Existing split seeds 812 and 813 are both reported. The original fitting grid uses C values 0.01,0.1,1,10; validation mean macro F1 over 15 view settings selects C. Outer-role outcomes never choose C in that trial.

Each model retains its normalization, coefficients and class order. An independent checker regenerates split memberships and compares saved labels/probabilities with the original manifest. The raw-image demo checks the actual saved trial 1/seed 812 head rather than fitting a new model.

The small classifier uses LBFGS optimization with at most 1,000 iterations. This is not 1,000 passes through every original image. Image encoding already used GPUs; cached-feature heads fit efficiently on CPUs.

## Current improvement experiment

The checked baseline and raw-image demonstration are packaged. The next experiment updates only Phikon-v2's last block and final output normalization. It first practises with an auxiliary image classifier for one epoch, then also updates those encoder layers for five epochs. Class-balanced sampling gives rare classes a fair share of practice; within each class it balances cases and zooms. Only fitting cases supply those updates.

Afterward, discard the auxiliary classifier, encode the fitting and validation images without augmentation, and train the same mean-plus-mask logistic case classifier described above. Keep the same four C choices, cases and metrics. All ten full runs and the artifact checker passed. Mean case accuracy is45.7%/38.8%; primary F1 does not improve on both splits, so this recipe fails its improvement gate and the frozen baseline remains the default. [Complete results](BREAKHIS_ADAPTATION_RESULTS.md). [Exact recipe](BREAKHIS_ADAPTATION_SPEC.md) · [Execution checkpoint](BREAKHIS_ADAPTATION_STATUS.md).

The current numbers are exploratory. All 82 provider cases have entered development somewhere, and clinical identity relationships remain uncertain. We must not describe them as a fresh independent clinical test. Novelty and a conference paper are later evidence gates, not requirements for completing the first working project milestone.

[Focused plan](BREAKHIS_MILESTONE_PLAN.md) · [First numbers](BREAKHIS_FIRST_RESULTS.md).

## What the diagnosis found

The saved adapted case classifier answers its fitting cases very well (98.5–98.7% balanced accuracy) but does poorly on validation (38.8–45.7%). The auxiliary image head also has a large gap: about98% on fitting pictures and32–35% on validation with class/case/view balancing. Keeping that head and using either fixed averaging readout does not fix validation. This is consistent with poor generalization; no single cause has been established. [Diagnosis, all readouts and figure](BREAKHIS_DIAGNOSIS_RESULTS.md).

For a linear head, classifying a mean feature equals averaging field logits with the same weights. That checked identity helps avoid blaming early averaging alone. The next training design should establish a compatible supervised reference and address the small independent case count; this diagnosis itself performs no fitting or new encoder execution.
