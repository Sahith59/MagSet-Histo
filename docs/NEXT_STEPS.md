# What we do next

Updated November 12, 2026. The aim remains eight-subtype breast-tissue classification with missing microscope views. The secondary native experiment is a different 16-class task. We have working experiments and negative evidence; a novel-method advantage has not been established.

## Work completed

| Task | Evidence / outcome |
| --- | --- |
| Build the four-view pipeline | Starting-file review, manifests, image validation and frozen feature caches |
| Establish simple and neural controls | 175 exploratory BreakHis configurations; mean remains ahead of checked matched neural controls |
| Register a fresh secondary supervised split | 80 fit /16 development /32 reserved native patients |
| Test the small agreement idea fairly | Four fixed fits; affinity does not beat duplicate features, so that version's improvement claim stops |
| Audit target-label code | Author loader confirms `label_id`; numeric mapping retained; clinical-name crosswalk remains partly unresolved |
| Inspect closest methods | Targeted source/code audit complete; exact MagViT/M2OTCA reproduction remains unresolved |
| Investigate existing prediction failures | Five of 16 development patients have a correct-to-wrong transition when a view is added in at least one availability setting |
| Harden the analyses | 54 pinned tests pass; trusted hashes/roles and known-outcome fixtures checked |

## Next tasks, in order

1. **Choose one precise research question.** Explain the observed view-addition failures, then identify what existing methods fail to handle. Selective fusion and uncertainty have prior work. Deliver a one-page hypothesis with the strongest simpler explanation and a decisive control.
2. **Finish the comparison specification.** Define which published components can be faithfully implemented and which require adaptation. Resolve unavailable details where possible; do not invent an exact M2OTCA implementation. Keep original paper tasks and our task separate.
3. **Register one small next training experiment.** Fix arms, encoder, split, normalization, field budget, settings and maximum fit count before scoring. Include mean and appropriate matched controls. The failed native pilot gate does not authorize an automatic C/seed/width sweep.
4. **Run and verify that experiment.** Test invariances/reload first, then submit appropriate scheduler jobs. Preserve losing models and verify predictions. A favorable result must exceed its controls and survive declared robustness checks before supporting a method claim.
5. **Consider encoder fine-tuning when justified.** Updating part of the picture reader is a separate experiment. Keep adaptation identical across fusion methods. GPUs can support it; sample size, protocol and hypothesis determine whether it is informative.
6. **Strengthen breast evidence and clinical grouping.** Resolve BreakHis identity limits or obtain a suitable independent breast cohort through approved access. TCGA-MS cannot substitute for eight-class breast confirmation. BRACS access is not approved; no access request/contact has been sent.
7. **Finish the paper from verified evidence.** Write the question, methods, comparisons, results and limitations alongside experiments. Fix final claims/protocol before reserved scoring. Then review the manuscript and choose a suitable venue with Professor Dong.

The 32 reserved patients remain unscored. Do not run the original rotating TCGA source-fold evaluation across them: some reserved patients would enter fitting in another fold.

## Work that can proceed together

- Research design: closest-method equations, controls and claim-to-evidence mapping.
- Data quality: identifiers, label lineage, access and provenance.
- Writing: implemented methods, complete negative results and professor decision brief.
- Engineering: fixed configurations, tests, saved models/predictions and verification.

These tasks can proceed while a future registered job is queued. No new model-training job was launched in this audit step. November 18 is an internal preparation target; the professor meeting is unconfirmed. A positive result or conference acceptance is not promised by that date.

Start with [the training explanation](TRAINING_STRATEGY.md), [method comparison audit](METHOD_COMPARISON_AUDIT.md), [label audit](TCGA_LABEL_LINEAGE_AUDIT.md), and [view-addition findings](VIEW_ADDITION_AUDIT.md).
