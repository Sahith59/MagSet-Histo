# Critique findings, repairs, and open decisions

Updated November 12, 2026. Three agents completed independent validity, novelty, and data/provenance critiques, followed by three focused repair/claim reviews. Six reports are preserved alongside this ledger. Review opinions do not certify scientific validity; closure below depends on concrete checks.

| Finding | Action/evidence | Status |
| --- | --- | --- |
| Collector could accept consistent wrong truths or case assignments | Added trusted manifest/protocol and independently regenerated split checks; all 175 reports passed. Complete-path label permutation and same-class membership-swap regressions are rejected. | Repaired for declared metadata; not proof of clinical identity |
| Publication jobs could omit trusted checking | Collector/report CLIs require metadata; scheduler collection scripts pass it. Meeting builder revalidates every input and records a hash-linked result ledger. | Repaired in new workflow |
| Clinical meaning of shared numeric stems is unknown | Quantified overlap/embargo feasibility; two adenosis stems cannot support class-covered three-way strict grouping. Preserve provider-case terminology and ask Professor Dong to choose a defensible policy. | Open scientific decision |
| Old globally untouched-test wording was too strong | Corrected public protocol/methods; all 82 cases have entered development somewhere. No own-trial outer-role scores were computed. | Claim corrected; past adaptivity cannot be undone |
| TCGA cannot be called clean Phikon external evaluation | Pinned Phikon card confirms TCGA source exposure; exact slide overlap unknown. Use ImageNet ResNet as initial provenance control, label TCGA as secondary multi-organ task. | Interpretation corrected; independent breast cohort still unavailable |
| Future confirmation patients could be reused across folds | Registered deterministic 80fit / 16 dev / 32 sealed patient protocol before classifier fitting, with 5/1/2 per class and a fixed salt/hash. Do not train additional source folds across reserved patients. | Registered; future use must enforce it |
| Native schema could admit count/path/identity drift | Bind source metadata hashes; enforce 16 sourceIDs, 8 patients/class, 122 complete / 6 fit-only incomplete, exact roles/views, job-bound protocol hash and patient/slide/path/label identity. | Tests and independent review passed |
| Mutable working copies weaken queued-job provenance | New jobs use read-only Git-archive snapshots with embedded commit and archive checksum; fixed metadata hash supplied at submission. | Implemented for new native jobs; old jobs retain their original records |
| Matched/unmatched decomposition can be a reparameterization | Preserve mean-plus-contrast, duplicate/random/heterogeneity controls; partial OT is optional and not claimed as new theory. | Candidate design narrowed; no positive method evidence yet |
| Too much adaptive search for tiny validation sets |175 configurations involve 31 distinct validation cases, not 175 independent observations. Stop broad architecture sweeps; freeze a minimal next-task budget before scoring. | Policy/claim repaired; sample-size limit remains |

## What is ready for the meeting

The generated brief, eight-slide offline deck, result figure, trusted ledger, registered new split, six reviews, and explicit contribution/go-no-go proposal can be discussed now. Baseline results include negative evidence rather than selected successes. November 18 is an internal preparation target; the meeting is not confirmed. BRACS access is not approved.

## What remains unproven

Novelty, candidate benefit beyond equal-dimension controls, independent clinical generalization, compatible published-method reproduction/adaptation, final held-out performance, and a submission-ready paper. The next step is a small defensible experiment, not a claim that reviewers or more GPU time have resolved these questions.

## Native pilot follow-through

The fixed four-fit development pilot completed from immutable source c1364c6. Independent result audit verified960 development-only records, all16 labels, all15 subsets per arm, seven saved artifact hashes, preprocessing, and classifier dimensions. Zero reserved predictions were produced. All50 tests pass, including the synthetic full pipeline and reporter tampering checks. Source/reporter review is recorded in `NATIVE_PILOT_REVIEW.md`.

Affinity mean macro F1 of0.338194 does not exceed the duplicate control's0.339583. The prespecified agreement-specific improvement gate fails for this version. The public results retain every arm/subset and distinguish the secondary16-class task from breast confirmation. No extra search or sealed evaluation follows automatically. The old “matched/unmatched conservation” wording is clarified as signed-contrast reconstruction, and block RMS is explicitly labeled an unweighted expanded-row diagnostic.
