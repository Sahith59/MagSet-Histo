# Whole-case mean versus attention: completed comparison

November 16, 2026. Learning from whole cases did **not beat the retained frozen similarity references on final accuracy or macro F1**. Two-step mean pooling scores **70.17% / 71.83%** final five-category accuracy, macro F1 **.5199 / .5094**. Two-step attention scores **71.67% / 70.50%**, F1 **.5453 / .5234**. Both registered improvement screens fail. 94–95% final accuracy remains unachieved.

| Pooling / decision | Accuracy 812 / 813 | Macro F1 812 / 813 | All-four-view accuracy 812 / 813 |
| --- | ---: | ---: | ---: |
| mean / direct | 71.17% / 71.50% | 0.5225 / 0.5028 | 67.50% / 72.50% |
| mean / dual | 70.17% / 71.83% | 0.5199 / 0.5094 | 70.00% / 72.50% |
| attention / direct | 73.00% / 70.83% | 0.5514 / 0.4953 | 70.00% / 70.00% |
| attention / dual | 71.67% / 70.50% | 0.5453 / 0.5234 | 72.50% / 70.00% |
| Three-crop KDE reference, F / dual |75.50% /76.83%|.5925 /.5799|82.50% /82.50%|
| Head-specific center, P / dual |76.33% /75.67%|.6027 /.6025|85.00% /85.00%|

Headline scores average all 15 zoom subsets and five overlapping trials in each arrangement. The full-four-view column is a separate endpoint. Attention improves dual macro F1 over learned mean by +.025407/+.014030, but accuracy changes by +1.50/−1.33 percentage points. This is a pooling-ablation result within the new method; it is not an improvement over F or P. Keep all four pool/arm rows visible.

## What was tested

The frozen F reader supplies 3072 numbers per original photograph. A small learned projection maps them to 128 features. Mean pooling averages fields within each zoom; standard gated attention learns which fields receive more weight. Both then average available zoom vectors, append four availability flags and use a direct-five head or separate benign/malignant and malignant-four heads.

Unlike the earlier field-label LoRA surrogate, the lesson is assigned to a whole case under all 15 subsets. The same AdamW recipe trains 100epochs with four checkpoint choices 10/25/50/100. Benign cases do not train the malignant expert. All fields are used, with fit-only normalization and fixed case/class weighting. Only the pooler changes between mean and attention; the shared initial projection/head states and random streams are paired. Mean has 394,009 trainable parameters and attention 402,297. No image encoder is updated or executed during this study.

Attention is an established method, and older eight-class attention and case-supervised CNN studies were negative. This tests the newer representation/task/objective bundle with a matched pooling control. It does not introduce a new attention algorithm or isolate hierarchy alone from its different training objective.

## What the outcome says

The models fit the fitting cases well: selected fitting accuracy is approximately 99.13%, macro F1 about .982. Validation remains much weaker. Fitting scores here are **in-sample neural diagnostics**, unlike KDE's leave-case-out fitting scores; do not compare those training columns as matching endpoints. The large macro-F1 gap is consistent with weak transfer beyond the fitting cases, not an inability to fit their bags. It does not prove one unique cause.

Attention raises final papillary recall by 13.33 points over mean in both arrangements, but shifts other errors. For the dual attention model, benign/ductal/lobular/mucinous/papillary recall is 86.33/90.67/54.67/46.67/36.00% on 812 and 91.00/64.00/44.00/36.00/56.00% on 813. The minority problem is not resolved.

Mean's arrangement 813 benign/malignant routing is 96.33%, while true-malignant typing is 50.33% and final five-category accuracy is 71.83%. Attention's binary scores are 92.17%/94.67%, with true-malignant typing 58.33%/51.00%. Binary accuracy is not final five-category accuracy. Attention also worsens dual cross-entropy and Brier probability error versus mean on both arrangements despite the F1 gains. No uniformly stronger or better-calibrated model is established.

## Complete execution evidence

All four fitting-only A40 checks 4983061 completed in 9–11seconds; CPU reviewer 4983062 accepted their hashes, original roles, optimizer updates, gradients and NumPy replay in 28seconds. They used five fitting cases, three epochs and six updates per trajectory, with no validation/excluded predictions. Maximum reload log difference was 1.045e-6 at the fixed 1e-5 tolerance. Full source CI passed all 370 neural tests, and independent pre-run numerical/guard checks passed.

All 40 full trajectories 4983086 completed successfully in 32–47seconds each, with at most four A40 GPUs concurrently. Each used 100epochs and all 46 fitting cases; the 40 trajectories are not 160 independently trained models or new patients. Collector 4983087 completed in 3m06 with empty error output. It independently replayed every saved candidate in NumPy FP64 from the accepted frozen features and saved FP32 parameters, verifying 160 checkpoints, 320 prediction archives and 129,600 dependent candidate records. There were zero collector fits, encoder forwards and excluded-case predictions. Root independently checked 344 aggregate/selection/class-recall calculations, all 40 selection decisions and the paired initial-state hashes.

Producer `6861fbfe8d847f16fa09b4f839193716c1e8d501`; config SHA256 `464c2890dfe4537a2dfd4458163f05071465660242e6056c43c09684771f21ce`. Accepted comparison SHA256 `a7c00e08a9279e79311d24547cf5df06d5c7cbfe913b1a880e8c6c1e345ae2b5`.

[Complete accepted JSON](evidence/BREAKHIS_CASE_MIL_RESULTS.json) · [Registered recipe](BREAKHIS_NEXT_ROUND_SPEC.md) · [Implementation review](reviews/CASE_MIL_IMPLEMENTATION_REVIEW.md) · [Actual validity review](reviews/CASE_MIL_ACTUAL_VALIDITY_REVIEW.md) · [Actual model diagnosis](reviews/CASE_MIL_ACTUAL_MODEL_DIAGNOSIS.md).

BreakHis-only fitting/evaluation, 46 fitting/eight validation/28 excluded provider cases, 31 reused validation keys and unresolved clinical identities are unchanged. All 82 provider keys entered development. No fresh clinical confirmation, novelty or 95% five-category result is established.
