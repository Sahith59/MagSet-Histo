# Supervised BreakHis reference: execution checkpoint

Updated November 13, 2026, 23:47 EST (November 14, 04:47 UTC). This is a dated launch checkpoint, not a live monitor or a completed result.

**Ten full fits are submitted: four running and six waiting for the four-task limit.** The automatic artifact checker waits for every fit to finish successfully. No improved accuracy or F1 is established yet; the frozen baseline remains the reference.

## In plain language

Our earlier model learned the practice cases very well but struggled with different cases. We are now teaching a standard ResNet-50 picture reader and its eight-answer classifier together. It sees only fitting pictures while learning. For a case, we average picture probabilities within each zoom, then average the available zooms. We test the same cases and the same 15 zoom combinations as before.

This asks whether an ordinary supervised pipeline transfers better. It does not test a new invention or establish which individual ingredient helps.

## Execution record

| Item | Evidence / status |
| --- | --- |
| Registered settings | [Fixed specification](BREAKHIS_SUPERVISED_SPEC.md); configuration SHA256 `d239cff3df4fcca16e64b9bf4f36309a5c8ea923c931198cd1a4b34f914b5839` |
| Training source | `c1b6b0ce24ffc99a00b3eb98cc09130934ffb0bb`, read-only snapshot |
| First technical smoke | Job4969999, completed in43seconds; fitting checks passed, but loader-shutdown warnings prevented production acceptance |
| Repaired technical smoke | Job4970065, completed in30seconds; artifact verification passed, error log empty, reload maximum logit difference0 |
| Technical budget | Two disclosed fitting-only smokes; neither scores validation or outer cases |
| Production array | Job4970079, ten fits, at most four concurrent GPUs,20epochs per fit |
| Automatic checker | Job4970104, dependency on successful completion of the full array; collector source `f4ba03ae1278ac7ecf1897328353abd43444b1f3` |
| Software checks |86 pinned local tests passed; three collector integrity tests repeated successfully |

The repair bounds the smoke's sampler before creating its loader, allowing normal worker shutdown. Production draws and scientific settings did not change. Both original and repeated smoke artifacts are retained. [Independent implementation review](reviews/BREAKHIS_SUPERVISED_REVIEW.md). [Public execution hashes](evidence/BREAKHIS_SUPERVISED_EXECUTION.json).

## What must pass before numbers are reported

The checker requires all ten models, all four checkpoint opportunities per fit, exact fitting/validation memberships, pretrained and model hashes, warmup/unfreeze proofs, unchanged BatchNorm statistics, optimizer step budgets, saved-model reload parity and complete field-score coverage. It independently rebuilds case probabilities and every reported metric. An incomplete matrix is not ranked.

The comparison reports both split means and every trial, all-view/worst-view scores, subtype recall, fitting–validation gaps, probability errors and model/runtime costs. The registered descriptive screen requires at least+0.02 macro F1 on both arrangements and nonzero phyllodes case recall on both. This is not a statistical or clinical acceptance threshold.

## Work alongside training

- Coverage and public-reference audits are complete: [findings](BREAKHIS_REFERENCE_AUDIT.md). Each fitting set has just one adenosis and one phyllodes provider case; more fields do not add independent cases.
- The [BreakHis project-report draft](../paper/breakhis-project-report.md) separates measured results, this pending comparison and unresolved limits.
- After the checker passes, review actual artifacts independently, publish the complete paired table and update the professor brief. A failure is evidence to retain, not permission for an automatic model sweep.

These remain overlapping provider-case exploratory validations, with31 distinct validation cases across both arrangements. Clinical identifier uncertainty, adaptive validation reuse and the absence of a globally fresh BreakHis confirmation cohort remain unresolved. TCGA and novelty work stay deferred.
