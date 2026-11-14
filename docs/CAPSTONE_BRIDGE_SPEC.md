# Registered saved-prediction endpoint bridge

November14,2026, before this analysis was executed. No new fit, encoder/classifier forward, calibration, threshold, normalization or checkpoint selection.

Input is the complete accepted ten-run supervised comparison, SHA256 `60b2bdbcaa775bb6a03f7b5e9ec85b318e888c3844cc2a4653f17adb378e3078`, and its saved validation field/case artifacts. Revalidate trusted manifest/protocol roles and labels, exact report/field/prediction hashes, all ten runs, original class order, all eight validation cases and all four magnifications before analysis. Excluded-role fields are refused.

The five fixed image slices are all validation fields pooled,40×,100×,200× and400×. For each run/slice, report ordinary image-micro accuracy, macro F1, balanced accuracy, cross-entropy, Brier, class recall/support and confusion, plus the class/case/view-balanced weighted equivalents. For a one-view slice, weighting equalizes cases within class and fields within case. Do not select a favorable view or aggregate observations as independent patients.

Reconstruct40× case probabilities from the same saved fields by equal field probability averaging. Require parity with archived40× case scores, labels and metrics. This isolates the change of prediction unit for this fixed model/data; it does not isolate architecture, training or historical split effects.

Report both five-trial means, distinct case/image counts and all runs. Same images/cases recur; count unique fields separately from repeated evaluations. Frozen/adapted case headline scores and historical notebook numbers remain separately labeled. The supervised ResNet was trained on all zooms and selected with case-level mean-subset F1, so these image scores do not reproduce the40× EfficientNet notebook.

The complete fixed configuration is exported as `CONFIG` by `magset/capstone_bridge.py`; record its canonical hash, module hash and all accepted input hashes in output. Publish aggregates only. Every slice is retained regardless of outcome. These descriptive analyses cannot create a fresh test cohort or support a new clinical/novelty claim.
