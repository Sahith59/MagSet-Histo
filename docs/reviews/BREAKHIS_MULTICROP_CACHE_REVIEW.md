# Independent actual multicrop cache review

November 14, 2026. **Accept the four full cache shards for provenance, coverage and saved-vector arithmetic.** This review is independent of the extractor author. It used standalone NumPy arithmetic and archive/hash checks; it did not invoke the producer's blend function, fit a model, run an encoder, score labels or submit/cancel jobs.

## Actual artifacts checked

The four local candidate NPZ/JSON pairs cover all7,909 manifest paths exactly once:40×1,995;100×2,081;200×2,013;400×1,820. Each shard has finite float32 thumbnail/blended vectors of shape `(N,1024)` and quadrant vectors `(N,4,1024)`. Every saved raw image hash, width, height and top-left/top-right/bottom-left/bottom-right bounding box agrees with the audited manifest; no field sequence correspondence across magnifications is assumed. Recorded raw hashes were compared with the manifest, rather than rehashing7,909 original PNGs in this review.

| Magnification | Candidate NPZ SHA256 | Metadata SHA256 |
| --- | --- | --- |
|40×|`7fbbe348678d21452781ee1abda39b95bbc346189ec95cf3534b56683338e302`|`9488d8d215043388900942f2e1e182607e7100e98cbd37fc665535914a959836`|
|100×|`4546d11f2f41678ac0a057415c6a2a13133e6e2cac38a835445faa4120172aec`|`e0b1b29887bdb01c716e6bce317b67b95f3f917fd490f9d27657b0b6833290c5`|
|200×|`055a80e0589b164a68a3ac5f5fbbaaf7668d540805744297f9f50de704973b15`|`14fe324f6592a7a52967349d6de885669ec240aa9ab97da15d6a424201cb3508`|
|400×|`3df8694121659eae1e8a7b9d3ad508795ff538c1b6504e695f4ac2e6be2101a1`|`a9b4da6594f95de421343ed827932b02aaca5b4a5fb2e294bcd312296882cbce`|

All four metadata records identify source `47ad41657979c3e154cc9db7b89f1fec4b1f40ac`, the exact registered config SHA256 `42e8e98e1310c7d9daf339c2f0376d2abb6fcc270dbb350def389318402498af`, and source archive SHA256 `73454a6ba398560d6ca54abedc550aff9130bce8b24cc4c9a38fe1962e129118`. The actual local archive hash and all six recorded dependency contents were independently verified. Config dictionaries, pinned encoder/processor identity, original thumbnail metadata and registered thumbnail file hashes match. Processor metadata describes shortest-edge resize224 followed by center crop224, applied separately to quadrant inputs; it is not a whole-image square warp.

All7,909 saved thumbnail vectors are exactly equal to the original accepted cache by path. The40×/100×/200× comparisons used the local original NPZs. The400× original NPZ was still transferring locally, so a bounded read-only SSH NumPy check compared its1,820 rows against the actual assembled candidate, hashed both files, and matched the local candidate thumbnail-vector byte hash `b9713b971ac0a721669561e591c612885b8ffd908517cc1044db255dca849d39`. The remote original400× NPZ hash matched its registered `2d61a4325ae82dd295445678134885af7a3a37622cec1be3cc26dbc076da3385` pin; the remote candidate hash matched the local candidate above.

Standalone recomputation of `float32(0.5*float64(g) + 0.5*mean(q, axis=quadrant, dtype=float64))` exactly matches every blended vector. All four full-cache metadata records exclude smoke mode, have no smoke roles, declare31,636 total quadrant forwards and zero technical-thumbnail forwards, and report zero blend replay error with no full-run thumbnail re-encoding claim.

## Scope and execution interpretation

These checks establish full manifest coverage, immutable-source/config binding, original-thumbnail preservation, geometry metadata and saved-vector consistency. They do not independently regenerate quadrant embeddings from pixels, prove their semantic diagnostic usefulness, establish model accuracy or clinical generalization, or accept the separate20-head outputs. Frozen/eval runtime behavior is supported by the inspected extractor source and execution provenance, not a second full encoder run here. Feature extraction over all fields is frozen external-pretraining inference; this review itself created no fitting or outer-label predictions.

The original head job completed under source47ad416 before any proposed stop took effect. The reported execution record is20 saved heads in466seconds with no cancellation or refit; the separate numerical head reviewer must verify those outputs. Later local improvements materialize compressed NPZ arrays once and permit explicit immutable feature-producer binding for future executions. They are performance/integration changes, not the source of these caches or an additional scientific attempt. Successful consistency checks do not override a failed classification improvement gate or turn reused provider-case development outcomes into an untouched patient test.
