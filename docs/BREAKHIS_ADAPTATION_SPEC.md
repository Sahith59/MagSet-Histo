# BreakHis improvement v1: fixed supervised adaptation

Registered November 12, 2026 before any adaptation outcome. BreakHis remains the main dataset. This is ordinary supervised transfer learning for the project, not a new fusion method or novelty claim.

## Question and budget

Does a small, fixed amount of fitting-only image-reader training improve the existing case-level mean baseline? Compare the frozen Phikon-v2 reference with one prespecified adaptation recipe. Keep all outcomes; do not search new training rates, widths, epochs or seeds after seeing validation scores.

Exactly **ten production encoder fits**: five provider repeated trials × inner split seeds812/813. Training seed812 for every run. One separate technical smoke is allowed: two fitting-only epochs, at most two batches per epoch, no validation encoding/classification/score. It gates execution and is never ranked as a model result.

Each production run uses the existing46 fit /8 validation /28 excluded outer-role provider cases. Regenerate membership from the trusted manifest/protocol and match it against the saved frozen reference. Adapted state and features are isolated per trial/split. No extra validation-derived epoch choice; no outer image may enter adaptation or adapted feature extraction.

## Fixed training recipe

The pinned Phikon-v2 checkpoint is a 24-block Dinov2Model with1,024D CLS output. Keep the first23 blocks and embeddings fixed. Train only the last block and final output LayerNorm, plus an auxiliary eight-class image head.

- Six total epochs, each with exactly N draws where N is the number of fitting image fields (4,071–4,724 for these runs). Batch16, no dropped final batch;255–296 steps per full epoch.
- Epoch1 trains only the auxiliary head. Epochs2–6 also train the last block/final LayerNorm. Optimizer groups contain those parameters from the start; warmup gradients remain absent.
- AdamW: encoder learning rate0.00001; auxiliary-head rate0.001; weight decay0.01; head dropout0.2; gradient norm capped at1. No schedule/grid/early stopping.
- Sample with replacement with P(class)=1/8; P(case|class)=1/n_class; P(view|case)=1/4; P(field|case,view)=1/n_fields. Use plain cross-entropy, without a second class-weight multiplier. This balances expected sampling mass; realized epoch counts vary and are recorded.
- Augment fitting images with deterministic horizontal/vertical flips, keyed by image path, epoch and training seed. Keep the pinned image processor. No validation augmentation, stain-template fitting or other adaptation.
- Float32, TF32 off, seeded deterministic PyTorch. Frozen prefix stays in evaluation mode. Only the last block/final normalization switch into the adaptation phase; gradients and fingerprints verify the boundary.

Every fit has only one adenosis and one phyllodes provider case. Their fitting class-case counts are [1,25,6,2,5,3,1,3] in canonical order. Balanced sampling does not create subtype diversity. Overfitting is a material limitation; do not infer reliable clinical accuracy from a selected development gain.

The auxiliary head uses each field's provider-case subtype as its target. Independent field-level diagnostic annotations are unavailable. This is weak field supervision; some fields may not show diagnostic tissue. A result assesses this whole training recipe, not verified localization of cellular patterns.

## Final encoder and identical downstream head

Save only the trainable encoder fragment, auxiliary head, optimizer, exact role/config/source bindings, epoch and RNG state. Replace one last checkpoint per epoch; do not store every epoch. Resume may continue the same bound run, never extend the registered six-epoch budget or change settings.

Before feature extraction, reload the final fragment into a fresh pinned base encoder and verify its output on fitting-only probe images. Prove the frozen-state fingerprint is unchanged and that allowed encoder parameters changed. The auxiliary image head is then discarded for the reported case-level classifier.

Before the first optimizer update, compare four canonical fitting-only probe fields (one per magnification) with the corresponding original frozen cache. Maximum absolute feature difference must be at most0.0001. This verifies the starting encoder/processor without opening validation images or changing the baseline. Record the observed difference and source hashes.

Encode all original fields of the46 fitting and8 validation cases, without augmentation. Construct that allowed54-case list before opening images. Save a trial/split-bound cache that exactly covers it. Do not encode or predict the28 outer-role cases with this fitted encoder.

Reuse `field-moments-v2` fitting-case-only normalization, within-view means, available-view mean+mask, class-balanced logistic regression and its original four-value C grid [0.01,0.1,1,10]. Validation mean macro F1 across all15 subsets selects C; ties keep the first C. Numerical normalization values may change with features, while the algorithm/role policy stays identical.

## Required gates and reporting

Test fit-only rows/weights, actual parameter whitelist, head-only warmup, prefix immutability, changed last-block weights, checkpoint binding/resume, missing-view/permutation invariance, finite scores, full cache memberships and reload. A real fitting-only GPU smoke must pass before the production array starts. Hardware/software failures may be repaired and resumed under unchanged scientific settings; no hidden replacement configurations.

Report every trial and both split means, paired deltas against the corresponding saved frozen baseline, all15 view subsets, full/worst F1, case accuracy and per-class recall. Recompute probabilities/metrics from saved artifacts and trusted labels. Partial progress must not produce a partial ranking. No significance claim from treating repeated trials/masks as independent.

An improvement supports this fixed adaptation package on exploratory provider-case validation. It does not isolate a new fusion mechanism, create a clean BreakHis confirmation cohort, establish patient identity, or prove novelty. The baseline remains the fallback if adaptation loses. TCGA stays deferred.
