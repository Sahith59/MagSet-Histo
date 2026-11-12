# Follow-up critique: native quarantine and frozen feature preparation

Reviewed November 12, 2026. This second review examined `magset/tcga_protocol.py`, `magset/tcga_features.py`, the native scheduler script, source-snapshot installer, and protocol tests. It supplements [the provenance critique](DATA_PROVENANCE_CRITIQUE.md). No source file was changed by this reviewer and no classifier was fitted or scored.

## Verdict

**No remaining blocking defect was found in the reviewed metadata partition and label-blind feature-preparation path after the identified guards were added.** This is a narrowly scoped code-readiness finding. It does not establish model quality, novelty, contamination-free Phikon evaluation, or final benchmark validity.

## What the review changed

The initial protocol draft checked global role totals, but not all of the promised release/class constraints. It now validates:

- The pinned label-table and inventory hashes.
- Exactly 128 unique labeled patients/slides and the expected 16 source label IDs, with eight patients per class.
- Exactly six incomplete patients, each missing only 0.25 mpp.
- Five fitting, one development, and two sealed-test patients in every class.
- Complete-view eligibility for development/test, with incomplete cases restricted to fitting.
- Unique image paths and coverage of every labeled patient.

The native feature reader initially trusted declared path/label fields too loosely. It now also checks path-derived resolution and slide identity, per-patient label/slide records, exact patient/view coverage, and the externally supplied registered protocol hash. The scheduler requires that hash, rather than accepting an arbitrary internally consistent manifest/protocol pair.

The source installer now checks the Git archive's embedded commit identifier against the requested full commit SHA, records the archive digest, installs to a commit-specific directory, and makes the installed source read-only. The scheduler checks the installed source marker before execution. This controls accidental drift between queued jobs and later source edits; it is not a cryptographic signature against a malicious archive producer.

## Independent checks performed

| Check | Observed result |
| --- | --- |
| Existing native protocol tests | Two tests passed, including changed hash/class-count/missing-view rejection |
| Reverse both source CSV row orders in a temporary fixture | Roles and canonical manifest checksum remained identical |
| Regenerate protocol from actual audited metadata in a temporary directory | 80 fitting / 16 development / 32 sealed-test patients; five/one/two per class |
| Actual native manifest | 37,024 images; unchanged SHA-256 `a668cf85151f29a163628210851c007faaf8c9679b7c9f5eaadc8c031f06e27b` |
| Feature-reader valid synthetic manifest | Accepted |
| Wrong externally expected protocol hash | Rejected |
| Alter resolution, slide, or patient metadata while updating the internal manifest hash | Rejected |
| Synthetic source archive with an inconsistent embedded commit identifier | Rejected |

The actual native view counts are 9,144 at 0.25 mpp, 9,598 at 0.5 mpp, 9,546 at 1.0 mpp, and 8,736 at 2.0 mpp. These checks validate metadata and guards; the fixture checks are not research-performance evidence.

## Leakage and mask assessment

The split salt is fixed in code and is not exposed as a tuning argument. Complete patients are ranked within class using SHA-256 of the fixed salt plus patient identifier. Unique patient IDs prevent the same supplied patient from entering multiple roles. The output-directory refusal helps preserve the registered version rather than silently replacing it.

The feature script uses ImageNet ResNet-50 only. The shared loader sets evaluation mode and disables parameter gradients. Encoding runs under `torch.inference_mode()`. Its image dataset consumes paths and pixels; labels and roles are validated as provenance but do not affect the frozen encoder. There is no native classifier, optimization loop, cohort-fitted normalization, or metric computation in this path. Consequently encoding sealed images is compatible with the declared label-blind cache policy.

Available-view metadata is preserved rather than fabricating absent 0.25-mpp fields. A future training adapter must use the actual fitting-case masks and normalize the subset objective per patient. The current feature-preparation code does not yet demonstrate that future training behavior; it only preserves the inputs needed to implement it correctly.

## Conditions that still apply

1. Keep the registered protocol hash and immutable source snapshot in every job/cache record. A successful model smoke check still needs real cluster evidence before large extraction is described as complete.
2. Formal regression tests for the independently exercised native feature-reader and source-installer guards would improve future protection; a one-off review script does not replace those tests.
3. Do not tune on the 32 sealed patients or interpret their cached embeddings as permission to inspect model outcomes early. Classifier fitting and any cohort normalization must be role-gated separately.
4. The six incomplete patients are all fitting cases. They cannot also form an independent natural-missingness test result under this split.
5. The 16-case development cohort remains small. Fix a compact candidate/selection budget before repeated model iteration; keep final uncertainty based on patient identities, carrying all subset predictions jointly.
6. This is a new supervised secondary task with 16 labels. It is not the source paper's exact protocol and does not validate an identical eight-label breast classifier. Phikon's documented TCGA pretraining-source overlap remains relevant if it is later added.

The review supports proceeding with frozen ImageNet feature preparation and a small development-only experiment after the smoke and provenance gates pass. It does not support unsealing final scores or claiming a novel mechanism yet.
