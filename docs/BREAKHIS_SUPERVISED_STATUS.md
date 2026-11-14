# Supervised BreakHis reference: execution checkpoint

Updated November14,2026. This is a dated execution checkpoint, not a live monitor or a completed result.

**All ten runs, the full checker and independent numerical review passed. The improvement gate failed.** Eight original runs finished in14m16s–17m29s with empty error logs. Two hit the scheduler limit; A40 continuations were rejected by the unchanged replay safeguard. Both subsequent A100 continuations completed successfully with empty error logs and passed their selected validation replay. Mean-subset F1 is0.3084/0.3006 and case accuracy37.0%/40.0%. Primary F1 is below the frozen reference on both arrangements; keep that reference. Independent actual-output numerical review passed. [Complete checked comparison](BREAKHIS_SUPERVISED_RESULTS.md).

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
| Production array | Job4970079: eight completed successfully, tasks0/1 timed out after2h00m18s; original logs and checkpoints retained |
| A40 recovery attempt | Job4971209: both tasks failed the unchanged selected-validation field replay check after1m09s/2m07s; no accepted results |
| A100-only continuation | Job4971227: both completed successfully in1m59s/3m24s, error logs empty; original pre-A40 committed states restored; wrapper source `752441c0fbe00dca90f184888ce8ee2ad15d6e38` |
| Automatic checker | Job4970104 COMPLETED0 in58seconds, error log empty; all ten outputs verified; collector source `f4ba03ae1278ac7ecf1897328353abd43444b1f3` |
| Software checks |86 pinned local tests passed; three collector integrity tests repeated successfully |

The repair bounds the smoke's sampler before creating its loader, allowing normal worker shutdown. Production draws and scientific settings did not change. Both original and repeated smoke artifacts are retained. [Independent implementation review](reviews/BREAKHIS_SUPERVISED_REVIEW.md). [Public execution hashes](evidence/BREAKHIS_SUPERVISED_EXECUTION.json).

## Scheduler recovery

The recovery wrapper preserves and hashes each interrupted last/selected state, orphan temporary file and existing selection archive before continuation. Only committed `last.pt` controls resume; an unfinished temporary checkpoint cannot replace it. Completed models are refused. Original committed boundaries are epoch20 for trial1 and epoch19 for trial2. Trial1 needs final replay/scoring; trial2 can repeat the uncommitted final epoch. The fixed20-epoch scientific budget, source, labels, roles and four checkpoint opportunities remain unchanged.

On the A40 attempt, selected validation macro F1 replay passed, but field probabilities failed the original1e-6 replay safeguard. The score is therefore not accepted; the tolerance is not relaxed. The second wrapper asserted A100 hardware and verifies original preserved file hashes before restoring the pre-migration committed checkpoint and selection archives. It preserves the rejected A40 states too. This is a technical replay recovery, not replacement of an unfavorable valid model. Both A100 continuations passed the selected replay and finished. The complete artifact checker and independent numerical review passed. Success on A100 does not establish a complete causal explanation of the A40 discrepancy.

The training report's elapsed time covers its final invocation for these two cases. Any runtime comparison must include original, rejected A40 and subsequent A100 SLURM segments and disclose repeated work, rather than present final-invocation time as their total cost. No completed model or unfavorable valid outcome is replaced.

## What must pass before numbers are reported

The checker requires all ten models, all four checkpoint opportunities per fit, exact fitting/validation memberships, pretrained and model hashes, warmup/unfreeze proofs, unchanged BatchNorm statistics, optimizer step budgets, saved-model reload parity and complete field-score coverage. It independently rebuilds case probabilities and every reported metric. An incomplete matrix is not ranked.

The comparison reports both split means and every trial, all-view/worst-view scores, subtype recall, fitting–validation gaps, probability errors and model/runtime costs. The registered descriptive screen requires at least+0.02 macro F1 on both arrangements and nonzero phyllodes case recall on both. This is not a statistical or clinical acceptance threshold.

## Work alongside training

- Coverage and public-reference audits are complete: [findings](BREAKHIS_REFERENCE_AUDIT.md). Each fitting set has just one adenosis and one phyllodes provider case; more fields do not add independent cases.
- The [BreakHis project-report draft](../paper/breakhis-project-report.md) separates measured results, the completed comparison and unresolved limits.
- The checker and independent numerical review passed; the complete paired table and professor brief are updated. Retain the negative result; it does not authorize an automatic model sweep.

These remain overlapping provider-case exploratory validations, with31 distinct validation cases across both arrangements. Clinical identifier uncertainty, adaptive validation reuse and the absence of a globally fresh BreakHis confirmation cohort remain unresolved. TCGA and novelty work stay deferred.
