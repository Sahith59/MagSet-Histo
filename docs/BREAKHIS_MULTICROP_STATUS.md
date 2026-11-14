# Regional-view test: execution checkpoint

November14,2026. **The technical smoke passed; the four regional-feature jobs and dependent20-head comparison are submitted. Scientific results are pending.** This page is a dated checkpoint, not an autonomous monitor.

The test gives the same frozen pathology reader its existing central view plus four fixed regional views of each original image. These are crops, not extra patients or new optical magnifications. It changes image detail/context/coverage together; it is not a novel method or a guaranteed route to95%.

| Item | Evidence |
| --- | --- |
| Fixed recipe | [Registered specification](BREAKHIS_MULTICROP_SPEC.md); config SHA256 `42e8e98e1310c7d9daf339c2f0376d2abb6fcc270dbb350def389318402498af` |
| Immutable source | `47ad41657979c3e154cc9db7b89f1fec4b1f40ac` for extraction and head comparison |
| Tests |110 pinned local tests passed; shell syntax passed for every scheduler script |
| Source review |Independent GPT-6.1 Sol source reviews of extractor/evaluator found no blocker; actual outputs still require verification |
| Fitting-only technical smoke |Job4971319 completed0 in21seconds on one A40; error log empty; four fields,16 regional forwards and four cache-parity forwards |
| Observed cache parity |Maximum absolute feature difference0.00000138581, below the prospectively fixed0.0001 gate; saved blend replay exact |
| Feature array |Job4971322, four magnification shards, at most four A40 GPUs;31,636 additional production region forwards, no gradient fitting |
| Matched classifier comparison |Job4971323 after successful array completion;20 fixed-C CPU fits; full cache/source/config/role checks before fitting |

The tiny smoke's measured encoder work took3.53seconds plus loading/startup. It is a consistency and resource check, not a reliable full-run stopwatch. Each shard has a one-hour scheduler limit; this is a ceiling, not an ETA. [Safe execution ledger](evidence/BREAKHIS_MULTICROP_EXECUTION.json).

All20 heads use the exact existing case roles and all15 view combinations. Both representations use C0.01 and matching fitting-only field-moment normalization; there is no parameter grid. The original retained reference used historical C selection and is reported separately. Independent actual-output arithmetic review is required before any result is adopted or promoted.

The [capstone before/now report](CAPSTONE_BEFORE_AND_NOW.md) distinguishes the received93.73%/94.99% image-random saved validation from the current case-level task. The [completed image-endpoint bridge](CAPSTONE_BRIDGE_RESULTS.md) uses saved predictions only. Finish the capstone report regardless of the new experiment's outcome; keep the frozen reference until complete checked evidence supports a change.

No outer-role outcomes, extra region weights, alternative encoders or follow-on sweep are authorized by this experiment. Scarce independent cases, clinical-identity uncertainty and reused validation remain limitations.
