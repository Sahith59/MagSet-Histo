# Integrity repair follow-up

Prepared November 12, 2026. This second review targets the new trusted manifest/protocol guard in `magset/collect.py` and `tests/test_trusted_context.py`. It does not repeat the first statistical critique or assert that the scientific-design limitations have been repaired.

## Verdict

**The new guard closes the specific independent-label and split-membership gap identified in the first review. No P0 correctness regression was found in the tested guard.** A fresh read-only audit passed all 175 locally retained development reports against the actual recorded manifest and protocol. The remaining material integration issue is that the publication renderers and existing collector jobs still opt out of trusted verification unless callers explicitly supply the new metadata inputs.

This is an artifact-integrity repair. It does not certify clinical patient identity, independent validation, novelty, or generalization.

## What the repair now verifies

The reviewed implementation:

1. Checks that the trusted protocol names the supplied manifest's actual SHA-256 and assigns every known provider case to a supported role.
2. Checks each run's manifest/protocol hashes and declared prediction unit against the supplied metadata.
3. Independently regenerates the recorded trial/inner-seed split using canonical case ordering and one validation case per subtype.
4. Requires exact fitting, validation, and outer-role case membership, including absence of duplicate IDs at the `verify_run` entry point.
5. Compares saved prediction truth labels with the trusted manifest's case-to-class mapping.
6. Retains the existing probability, subset-coverage, and independent metric-arithmetic checks.
7. Labels the collection's verification level explicitly, separating trusted checks from the backward-compatible arithmetic-only path.

The independent split regeneration is usefully separate from importing the training split function; a change or mistake in one side can be detected rather than automatically shared.

## Checks performed in this review

No model training, cluster mutation, outer prediction scoring, or persistent result modification was performed. Tampering experiments used temporary copies of one existing validation report and its score records.

| Check | Observed outcome |
| --- | --- |
| All local full-run reports verified with the recorded trusted manifest/protocol | **175 passed, 0 failed** |
| Inner-seed coverage among those reports | 115 reports at seed 812; 60 at seed 813 |
| Valid one-run collection through `collect(..., manifest=..., protocol=...)` | Passed; reported `trusted labels and regenerated split plus score arithmetic` |
| Cyclically change every saved truth label and recompute all saved summary metrics | Arithmetic-only verification passed; trusted verification correctly rejected the manifest-label mismatch |
| Swap one validation case with a same-subtype fitting case, updating saved case IDs while preserving counts and metrics | Arithmetic-only verification passed; trusted verification correctly rejected the regenerated-split mismatch |
| Existing helper tests for altered truth, fitting/outer swap, seed, metadata hash, and a missing trusted path | Appropriate negative assertions are present |

The adversarial checks matter because they do not merely make the report internally inconsistent. They construct examples that would otherwise look arithmetically valid, then demonstrate that the trusted oracle detects the error.

## P1: trusted mode is not yet carried through every publication path

At the reviewed snapshot, the new mode is optional by design. Existing calls in `magset/capacity_report.py`, `magset/round2_report.py`, `magset/sensitivity_report.py`, and `magset/error_analysis.py` invoke `collect` or `verify_run` without trusted metadata. Existing `slurm/collect*.sbatch` commands likewise omit the new `--manifest` and `--protocol` flags.

**Consequence:** the new helper is correct, but regenerating a public table through those entry points still performs arithmetic-only checks. The returned JSON exposes that lower verification level, but the Markdown renderers do not currently surface it. A reader could mistake a generic “verified” table for one regenerated with the repaired oracle.

**Recommended repair:** choose one explicit publication route:

- Pass the trusted context through all audited report/collector entry points, with metadata paths mandatory for the professor-packet/final-report workflow; or
- Produce a frozen all-run trusted audit attestation containing the manifest/protocol hashes and per-run metrics/prediction hashes, then require public table generation to match that attestation. Label legacy arithmetic-only commands accordingly.

Backward compatibility is reasonable for synthetic tests and older utilities. It should not silently determine the verification standard of the meeting packet or final paper tables. The successful fresh 175-run audit means this integration issue is not evidence that the retained results contain mislabeled or wrongly assigned cases.

## P2: keep an end-to-end regression for the repaired guarantee

`tests/test_trusted_context.py` currently exercises the helper functions directly. Retain those fast tests and add at least one regression through `verify_run` or `collect` using a complete synthetic report and score file. The strongest case is the arithmetic-consistent truth permutation demonstrated above: it should pass legacy arithmetic verification and fail trusted collection. Also test a count-preserving same-class case swap and confirm the returned verification-level field.

This guards against a future refactor implementing correct helper logic but failing to call it in the actual collection path. A CLI negative test for providing only one metadata path would further verify argument wiring; it is lower priority than the complete collection regression.

## Boundaries of the repaired claim

The justified new statement is:

> All 175 retained development reports passed an independent check of their truth labels and exact declared trial/seed assignments against the recorded manifest and provider protocol, in addition to metric-arithmetic verification.

It should not be expanded into a claim that clinical patients are independently verified, the validation cases are untouched, or the selected model has confirmed generalization. The guard verifies conformity to the declared protocol; it cannot turn an unresolved identity rule or adaptively reused development sample into a clinically independent experiment.

The proposed TCGA 80-fit/16-development/32-quarantined patient protocol is an appropriate direction for separating subsequent label-outcome selection from confirmation, provided the split is frozen before any modeling outcomes and access paths enforce the quarantine. This follow-up did not execute or verify that not-yet-completed protocol. Explicitly acknowledging Phikon's source-level TCGA pretraining exposure also remains necessary: a new label-outcome quarantine does not undo backbone pretraining exposure. Avoid calling such a benchmark pretraining-independent external patient confirmation without additional evidence.

## Audit provenance

- Trusted manifest SHA-256: `958893f923a745220fac66f19656bcdbfb38996f70ef8044e1349619f4102e31`.
- Trusted provider-protocol SHA-256: `dca9a93bb570fd7cfec9fe72e5cad9ac76f838725b03c7a7a593824fecd51134`.
- Reviewed `magset/collect.py` SHA-256: `b3f3f7a8faaa51f3ed94c9c0d36842958c07cf6bba27faa74bdb8e7ad644a328`.
- Reviewed `tests/test_trusted_context.py` SHA-256: `607d4e0dc2efe5c0d7426f86eafe379a24c4e4a989f0ea9d973f46c93b6c62c3`.

Source was under active repair during this review. If a later patch closes the integration or regression recommendations, record the exact revision and test evidence in the closure ledger.
