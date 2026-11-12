# Native TCGA label schema

The classifier uses the release `label_id` column, remapped in sorted numeric order to canonical classes 0–15. All 16 groups contain eight supplied patients. Other numeric columns, including `label`, are preserved as provenance and are not substituted for the target.

| Canonical class | Source label_id | Source label column | Source tissue group | Observed source histology descriptions |
| ---: | ---: | --- | --- | --- |
| 0 | 12 | 16 | Connective, subcutaneous and other soft tissues | Fibromyxosarcoma |
| 1 | 13 | 17 | Connective, subcutaneous and other soft tissues | Fibrous histiocytoma, malignant |
| 2 | 14 | 19 | Connective, subcutaneous and other soft tissues; Stomach | Leiomyosarcoma, NOS |
| 3 | 15 | 23 | Connective, subcutaneous and other soft tissues; Kidney | Synovial sarcoma, NOS; Synovial sarcoma, biphasic; Synovial sarcoma, spindle cell |
| 4 | 16 | 27 | Kidney | Clear cell adenocarcinoma, NOS |
| 5 | 18 | 29 | Breast; Stomach | Adenocarcinoma, NOS; Medullary carcinoma, NOS |
| 6 | 22 | 38 | Breast | Invasive carcinoma of no special type |
| 7 | 24 | 40 | Breast | Lobular carcinoma, NOS |
| 8 | 25 | 41 | Breast; Stomach | Adenocarcinoma, NOS; Mucinous adenocarcinoma |
| 9 | 27 | 43 | Kidney; Stomach | Adenocarcinoma, NOS; Papillary adenocarcinoma, NOS |
| 10 | 28 | 44 | Stomach | Adenocarcinoma, NOS; Signet ring cell carcinoma |
| 11 | 32 | 48 | Stomach | Adenocarcinoma, NOS |
| 12 | 33 | 49 | Breast | (blank); Metaplastic carcinoma, NOS |
| 13 | 38 | 57 | Kidney | Renal cell carcinoma, chromophobe type |
| 14 | 41 | 61 | Connective, subcutaneous and other soft tissues | Undifferentiated sarcoma |
| 15 | 44 | 66 | Stomach | Carcinoma, diffuse type |

These are descriptive observations from pinned metadata, not an independently validated clinical taxonomy. Some groups contain multiple histology descriptions or blank text. Labels were not merged, reassigned, or inferred from descriptions. The grouping rationale still needs author-method review before a clinical or pathology-taxonomy claim. NOS in the source descriptions means “not otherwise specified.”

The subsequent [label-lineage audit](TCGA_LABEL_LINEAGE_AUDIT.md) confirms our numerical target choice against the author's loader. It distinguishes the paper's intended names from variable raw metadata descriptions; this table intentionally retains the latter.

Pinned source CSV SHA256: `2d97bbfec20d69edb7fcaf15c2a8068d6d4fe2e2f6161884a07a2ff3fd12ace1`. See [the registered protocol](TCGA_QUARANTINE_PROTOCOL.md) and [the plain-language guide](PROJECT_EXPLAINED.md).
