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

## What comes next

First package the checked baseline and raw-image demonstration. Then specify one limited supervised improvement on BreakHis, such as updating a small part of the image reader. That later step is fine-tuning; it has not been performed by this baseline or demonstration. Keep the encoder/adaptation policy identical when comparing fusion methods.

The current numbers are exploratory. All 82 provider cases have entered development somewhere, and clinical identity relationships remain uncertain. We must not describe them as a fresh independent clinical test. Novelty and a conference paper are later evidence gates, not requirements for completing the first working project milestone.

[Focused plan](BREAKHIS_MILESTONE_PLAN.md) · [First numbers](BREAKHIS_FIRST_RESULTS.md).
