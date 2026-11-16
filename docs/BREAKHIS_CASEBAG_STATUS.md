# EfficientNet five-category experiment: execution record

Updated November 15, 2026 at 21:42 EST. All20 full training runs and both results checks completed successfully. EfficientNet two-step validation accuracy is46.50%/53.33%. The earlier frozen-Phikon two-step model is the best verified option at65.83%/68.83%. The94–95% target remains unachieved. [Completed comparison](BREAKHIS_CASEBAG_RESULTS.md).

The experiment asks whether teaching part of the image reader on the actual case-level task improves malignant subtype recognition. It compares direct five-category prediction with the requested benign/malignant decision followed by malignant subtyping. All four magnifications are used, and all 15 nonempty combinations are evaluated. This is a complete-pipeline comparison; the training losses differ between the two arms.

| Check | Actual outcome |
| --- | --- |
| Recipe and source review | Closed; 162 pinned local tests passed |
| Public ImageNet weight and package preparation | Job 4979545 completed successfully in 18 seconds; strict model load; zero image forwards or fits |
| Input and split audit | Job 4979604 completed successfully in 5 seconds; independently checked with 15,898 assertions; zero forwards or fits |
| Direct-model technical check | Array task 4979613_0 completed successfully in 20 seconds |
| Two-step technical check | Array task 4979613_1 completed successfully in 42 seconds |
| Independent acceptance of actual GPU checks | Accepted; 2,512 independent assertions passed |
| Twenty full training runs | All20 array4979622 tasks completed successfully, with empty error logs |
| Complete-matrix results verification | Job4979632 completed successfully in2m34s, accepted |
| Independent numerical verification | Job4979664 completed successfully in46s, accepted3,852,210 assertions |

The two technical checks each trained on eight fitting cases for two epochs, using two batches per epoch. They tested a heads-only phase, a partial-backbone adaptation phase, and checkpoint save/reload. Each performed 128 training field forwards plus eight reload-probe forwards, with no validation or excluded-case predictions. Both ran on A40 GPUs, reserved about 1.52 GB of GPU memory, and reported zero feature/probability differences on their fixed reload probes. Scheduler elapsed times include startup; internal model lifecycle times were 11.75 and 30.84 seconds. Actual technical-check accuracy is not a validation result and is not reported.

Source is frozen at `b5f0df18a47c1607be492bdaab7ab6cd2171226b`. The registered configuration SHA256 is `3f633d0e43f66f51c4e27b3324a17180cda2bf7f2a91c9a757d7b1187da7d7b9`. The same immutable source, configuration, public pretrained weights, dependency versions and A40 family must span the technical checks and production runs. A later documentation or collector commit does not become the training source.

Each full run uses 46 fitting cases, eight validation cases and 28 excluded cases. Original eight-type case assignments are reconstructed before benign labels are pooled. Fitting supports are 11 benign, 25 ductal, two lobular, five mucinous and three papillary cases. Validation has four benign cases and one case of each malignant type. The malignant specialist learns from the 35 malignant fitting cases only.

The plan is 20 fixed 50-epoch fits, at most four A40 GPUs concurrently. The first five epochs teach only the small decision heads; the remaining epochs adapt the last two backbone stages and image feature head. All BatchNorm running statistics stay fixed. Checkpoints at epochs 10, 20, 35 and 50 receive the same selection opportunity in both arms. Selection uses end-to-end five-category macro F1 averaged across all 15 input combinations; routing mistakes count as mistakes.

The known production work is 531,912 field forwards, including candidate scoring, selected fitting diagnostics, selected validation replay and reload probes. Runtime admission accounts for all of that work, raw image loading and serialization. A deliberately conservative admission calculation gives about 62 minutes for the largest run within its 90-minute allocation. This is a capacity check, not an ETA: actual first-run throughput will refine the estimate. Queue delay is additional. The production allocation ceiling is 30 GPU-hours, with technical checks disclosed separately.

The first submission attempt was rejected by SLURM because the already-completed audit dependency had expired from its active job records. That rejection created no training job or output. The identical two-check array was then submitted after actual audit completion and independent acceptance, without the expired dependency.

These are reused provider-case development results, not a globally untouched clinical test. The 28 excluded cases are excluded per run; all 82 provider keys have entered development somewhere. Repeated trials and magnification combinations are dependent. Clinical identity remains unresolved. No new method, clinical benefit, paper acceptance or 95% result is established by successful execution.

[Registered recipe](BREAKHIS_CASEBAG_SPEC.md) · [Configuration](../configs/breakhis-casebag-v1.json) · [Checkpoint review](reviews/CASEBAG_CHECKPOINT_REVIEW.md) · [Actual GPU-check review](reviews/CASEBAG_ACTUAL_SMOKE_REVIEW.md) · [Capstone claims review](reviews/CASEBAG_CAPSTONE_CLAIMS_REVIEW.md).

Actual smoke acceptance is bound by SHA256 `4568db1d88a7aae0f8f55cd4b7385de56026a0a00415a1e06abd2bc9e6730ab4`. Independent review checked pretrained and initial-head fingerprints, actual frozen/BatchNorm tensor equality, adaptive parameter changes, optimizer groups and step counts, paired sampling and fitting-only scope. The fixed GPU probes are checked through recorded errors and immutable source guards; their raw arrays were not archived for separate GPU forward re-execution. Production candidate and selected scores do retain replayable arrays.

The separate CPU collector checks all 20 runs and 80 candidate checkpoints before producing any public aggregate. It verifies the frozen training source independently of its own later checker revision, case assignments, pretrained bytes, optimizer lifecycle, saved feature/head probability replay, explicit two-step decisions and checkpoint selection. It decodes no images and performs no CNN forwards. The 12 focused corruption/selection tests passed independently; the full pinned suite passed 174 tests. [Collector source review](reviews/CASEBAG_COLLECTOR_REVIEW.md).

Historical observation during launch. the first two full direct models completed all50 epochs. Their selected validation feature/probability replay and fixed reload probes matched exactly. Internal model lifecycles took458.21 and486.15 seconds. These are runtime and integrity observations only; no partial classification ranking is published. The subsequent full completion and accepted results are recorded above.

A second, independently written NumPy/SciPy reviewer now reconstructs trusted roles and replays all120 saved score archives (80 candidates plus40 selected fitting/validation archives), 25,800 dependent records, checkpoint choices, both split summaries and the improvement screen. It imports neither the training package nor Torch and performs no image decoding. Its13 focused tests passed independently; the full pinned suite passed187 tests. Exact selected epochs and artifact hashes are required; independently calculated numeric metrics allow a fixed1e-10 roundoff tolerance. Both reviewers subsequently completed and accepted the full results.
