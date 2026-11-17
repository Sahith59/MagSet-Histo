# Supplied Swin notebook execution status

Updated November 17, 2026, 09:54 AM EST. Main capstone experiments remain parked at the user's request. CNN results are complete; the default Swin dual-step experiment is **running on one allocated cluster GPU**. No Swin performance result has been accepted yet.

| Stage | Job | Observed outcome |
| --- | --- | --- |
| Separate pinned environment | 4985728 | Complete, exit 0 |
| Full image-byte/order/split audit | 4985800 | Complete, exit 0 |
| Synthetic training and weight reload | 4985801 | Complete, exit 0 |
| Full default dual fit | 4985804 | Running |
| Independent saved-result check | 4985805 | Queued after fitting |

Execution source is `a50aeee39a78b3210a5a8c33905f6566a0563875`. The original is preserved privately and unchanged. Maximum 40 epochs; no extra model, fold or tuning sweep. The accepted synthetic check used eight generated images and one optimization batch, **zero real test predictions**, exact checkpoint reload error **0**, masked benign subtype loss **0**, and stage trainability `[false, false, true, true]`. Total model parameters: **27,879,180**; trainable: **26,358,078**. Sixteen exact TensorFlow inverse-window/output/gradient checks also passed with zero errors. The isolated old-window graph did not reproduce the complete-model failure; the graph-optimizer explanation remains a supported diagnosis rather than a directly recorded intermediate trace.

Earlier technical attempts are preserved: one scheduler launch failed before script output; a strict weight-loading guard stopped on deterministic position indices; the next synthetic train detected the library graph reshape failure. None produced a production fit or a research accuracy. Declared compatibility fixes verify the omitted fixed indices and count windows with integers, preserving learned operations. [Fixed recipe](SUPPLIED_SWIN_RUN.md), [independent review](reviews/SUPPLIED_SWIN_REVIEW.md).

The actual audit confirms **5,104 training / 689 validation / 2,116 test images**, **51 / 7 / 23 bare filename IDs**, no bare-ID overlap, four conservative-stem crossings and zero lobular validation images. Secondary ID/target voting has 24 groups. These limits and all-four-zoom single-image inputs remain part of the result interpretation. [Accepted input audit](evidence/SUPPLIED_SWIN_INPUT_AUDIT.json), [technical evidence](evidence/SUPPLIED_SWIN_TECHNICAL.json).

The test result must be independently recomputed from all saved class probabilities, correct routed labels, source/configuration/split hashes, selected history and checkpoint replay. Until then, do not fill an accuracy from an interim epoch or the notebook's hardcoded paper figures. [Side-by-side comparison](SUPPLIED_NOTEBOOKS_COMPARISON.md).
