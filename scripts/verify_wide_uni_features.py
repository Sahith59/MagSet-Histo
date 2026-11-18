"""Independently reconstruct the development tensors from archived per-crop features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from magset.data import MAGNIFICATIONS, file_hash


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, role_file, source_cache, original_views):
    receipt = json.loads((run / "feature-receipt.json").read_text())
    check(file_hash(role_file) == receipt["audit_sha256"] and
          file_hash(original_views) == receipt["case_views_sha256"] and
          file_hash(run / "development-views.npz") == receipt["development_views_sha256"],
          "Role file or feature bundle commitment differs")
    roles = json.loads(role_file.read_text())
    ids = np.array(roles["roles"]["train"] + roles["roles"]["validation"], dtype=int)
    check(len(ids) == len(set(ids)) == 66 and not set(ids) & set(roles["roles"]["test"]),
          "Development role identities differ")
    details = {}
    with np.load(run / "development-views.npz", allow_pickle=False) as bundled:
        center, wider = bundled["center_views"], bundled["wide_views"]
        check(np.array_equal(ids, bundled["case_indices"]) and
              np.array_equal(MAGNIFICATIONS, bundled["magnifications"]) and
              center.dtype == wider.dtype == np.float64 and
              center.shape == wider.shape == (66, 4, 1536), "Unexpected tensor axes or precision")
    with np.load(original_views, allow_pickle=False) as original:
        check(np.array_equal(center, original["case_view_means"][ids]), "Original center case mean differs")
    for j, zoom in enumerate(MAGNIFICATIONS):
        selected = [r for r in roles["fields"] if r["case_index"] in set(ids)
                    and int(r["magnification"]) == zoom]
        key = {hashlib.sha256(r["relative_path"].encode()).hexdigest(): r for r in selected}
        official = source_cache / f"uni2-h-v1-{zoom}.npz"
        path = run / f"fields-{zoom}.npz"
        metadata = receipt["shards"][str(zoom)]
        check(file_hash(official) == metadata["center_source"]["npz_sha256"] and
              file_hash(path) == metadata["array_sha256"], "Individual field shard commitment differs")
        with np.load(official, allow_pickle=False) as src, np.load(path, allow_pickle=False) as shard:
            by_path = dict(zip(src["paths"].tolist(), src["features"]))
            hashes, owners, parts = shard["path_sha256"].tolist(), shard["case_indices"], shard["components"]
            check(len(key) == len(hashes) == len(set(hashes)) and set(key) == set(hashes),
                  "Duplicated, missing, excluded or unknown field")
            check(parts.shape == (len(key), 3, 1536) and parts.dtype == np.float32
                  and np.isfinite(parts).all(), "Invalid raw crop features")
            accum = {i: np.zeros(1536, dtype=np.float64) for i in ids}
            counts = {i: 0 for i in ids}
            for n, h in enumerate(hashes):
                r = key[h]
                i = r["case_index"]
                left, center_row, right = parts[n]
                check(owners[n] == i and r["view"] == j and
                      np.array_equal(center_row, by_path[r["relative_path"]]), "Center or ownership differs")
                wide = ((left.astype(np.float64) + center_row + right)/3).astype(np.float32)
                accum[i] += wide
                counts[i] += 1
            check(all(counts.values()) and np.array_equal(
                np.stack([accum[i]/counts[i] for i in ids]), wider[:, j]), "Wide case averages differ")
        details[str(zoom)] = {"verified_fields": len(key), "case_view_pairs": len(counts)}
    check(sum(d["verified_fields"] for d in details.values()) == receipt["fields"] == 6205 and
          receipt["image_forwards"] == 18615 and receipt["followup_fields"] == 0, "Field total differs")
    return {"verified_fields": 6205, "verified_center_case_views": 264, "verified_wide_case_views": 264,
            "followup_fields": 0, "stages": details, "feature_receipt_sha256": file_hash(run / "feature-receipt.json"),
            "verification_source_sha256": file_hash(Path(__file__)), "role_sha256": file_hash(role_file)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "role-file", "source-cache", "original-views"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.role_file, args.source_cache, args.original_views), indent=2))
