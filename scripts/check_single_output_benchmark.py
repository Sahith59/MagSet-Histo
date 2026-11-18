"""Check the retrospective result from field descriptors, saved weights and private answers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.check_field_hierarchy import compare_metrics, digest, fixed_roles, metrics, require, zoom_masks
from scripts.verify_single_output_coverage import decide, reconstruct

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "views", "cache", "checkpoint", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "benchmark.json").read_text())
    plan_path = ROOT / "configs/breakhis-single-output-retrospective-v1.json"
    plan = json.loads(plan_path.read_text())
    config = json.loads((ROOT / "configs/breakhis-single-output-center-v1.json").read_text())
    require(digest(args.run / "predictions.npz") == report["evidence_sha256"] and
            digest(args.checkpoint) == report["checkpoint_sha256"] == plan["source_checkpoint_sha256"] and
            digest(plan_path) == report["plan_sha256"] and report["new_fits"] == 0 and
            digest(args.views) == report["case_views_sha256"] == config["case_views_sha256"] and
            digest(args.audit) == report["audit_sha256"] == config["audit_sha256"], "Input commitment differs")
    pins = json.loads((ROOT / "configs/breakhis-uni2-h-v1-kernel.json").read_text())["cache_provenance"]
    audit = json.loads(args.audit.read_text())
    fixed_roles(audit)
    indices = np.array(audit["roles"]["test"])
    truth = np.array([audit["cases"][i]["truth5"] for i in indices])
    follow_up = set(indices)
    fields = [f for f in audit["fields"] if f["case_index"] in follow_up]
    require(len(fields) == len({f["relative_path"] for f in fields}) == 1704, "Incomplete field roster")
    with np.load(args.views, allow_pickle=False) as source:
        original = source["case_view_means"][indices]
    rebuilt = np.empty_like(original)
    for col, zoom in enumerate(config["magnifications"]):
        path = args.cache / f"uni2-h-v1-{zoom}.npz"
        require(digest(path) == pins[str(zoom)]["npz_sha256"], "Feature shard changed")
        with np.load(path, allow_pickle=False) as shard:
            paths = shard["paths"]
            lookup = {name: i for i, name in enumerate(paths)}
            require(len(lookup) == len(paths), "Field key repeated in feature shard")
            features = shard["features"]
            for row, i in enumerate(indices):
                names = sorted(f["relative_path"] for f in fields if f["case_index"] == i and int(f["magnification"]) == zoom)
                require(bool(names), "Missing required magnification")
                rebuilt[row, col] = features[[lookup[p] for p in names]].mean(axis=0, dtype=np.float64)
    require(np.array_equal(original, rebuilt), "Rebuilt case and zoom features differ")
    weights = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model"]
    replay = reconstruct(rebuilt, weights["mean"].numpy(), weights["scale"].numpy(), zoom_masks(),
                         weights["classifier.weight"].numpy(), weights["classifier.bias"].numpy())
    predicted = decide(replay)
    with np.load(args.run / "predictions.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["indices"], indices) and np.array_equal(saved["masks"], zoom_masks()) and
                np.array_equal(saved["truth"], truth) and np.array_equal(saved["predictions"], predicted),
                "Recorded case, truth or decision differs")
        error = float(np.abs(replay - saved["logits"]).max())
        require(error <= 1.e-5, "Follow-up logits exceed tolerance")
    verified = metrics(truth, predicted)
    compare_metrics(verified, report["matched_follow_up"]["single_output"], "retrospective replay")
    result = {"accepted": True, "fields": len(fields), "distinct_case_zoom_descriptors": len(indices)*4,
              "case_view_max_absolute_error": float(np.abs(original - rebuilt).max()),
              "max_logit_error": error, "reproduced_answers": int(predicted.size),
              "correct_final": verified["correct_final"], "correct_binary": verified["correct_binary"],
              "audit_script_sha256": digest(Path(__file__)), "benchmark_sha256": digest(args.run / "benchmark.json")}
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
