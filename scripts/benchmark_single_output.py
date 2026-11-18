"""Run the frozen BreakHis follow-up without fitting or selecting a model."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.check_field_hierarchy import digest, fixed_roles, metrics, require, zoom_masks
from scripts.train_single_output_coverage import SingleOutputClassifier, final_labels
from scripts.verify_single_output_coverage import decide, reconstruct, verify

ROOT = Path(__file__).resolve().parents[1]


def read_pinned(path, expected):
    require(digest(path) == expected, "Digest mismatch for " + path.name)
    return json.loads(path.read_text())


def aligned_comparator(path, expected_hash, cases, truth, zoom_sets):
    require(digest(path) == expected_hash, "Comparator predictions changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {(r["case_id"], tuple(r["magnifications"])): r for r in rows}
    require(len(by_key) == len(rows) == len(cases) * len(zoom_sets), "Duplicate or missing answers")
    predictions = []
    for case, label in zip(cases, truth):
        answers = [by_key[(case, zooms)] for zooms in zoom_sets]
        require(all(r["role"] == "test" and r["truth5"] == label for r in answers),
                "Different role or diagnosis in comparator")
        predictions.append([r["prediction"] for r in answers])
    return np.array(predictions)


def bootstrap_accuracy(groups, truth, predictions, draws, seed):
    group_names, group_index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(group_names), size=(draws, len(group_names)))
    group_sizes = np.bincount(group_index) * 15
    denominators = group_sizes[sample].sum(axis=1)
    def interval(values):
        return np.quantile(values, [.025, .975]).tolist()
    bands = {}
    for key, pred in predictions.items():
        correct = np.column_stack(((pred == truth[:, None]).sum(1),
                                   ((pred > 0) == (truth[:, None] > 0)).sum(1)))
        totals = np.stack([correct[group_index == g].sum(0) for g in range(len(group_names))])
        bands[key] = totals[sample].sum(1) / denominators[:, None]
    primary = bands["single_output"]
    return {
        "groups": len(group_names), "draws": draws, "seed": seed,
        "final_accuracy_interval": interval(primary[:, 0]),
        "binary_accuracy_interval": interval(primary[:, 1]),
        "paired_difference_single_output_minus": {
            key: {"final": interval(primary[:, 0] - other[:, 0]), "binary": interval(primary[:, 1] - other[:, 1])}
            for key, other in bands.items() if key != "single_output"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "views", "run", "comparators", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / "configs/breakhis-single-output-retrospective-v1.json"
    plan = json.loads(plan_path.read_text())
    args.config = ROOT / "configs/breakhis-single-output-center-v1.json"
    source = (ROOT / ".source-sha").read_text().strip()
    require(source == os.environ["MAGSET_CODE_SHA"], "Executing source differs from the requested snapshot")
    config = read_pinned(args.config, plan["fit_config_sha256"])
    read_pinned(args.run / "result.json", plan["fit_result_sha256"])
    require(digest(args.run / "selected.pt") == plan["source_checkpoint_sha256"], "Wrong selected model")
    replay = verify(args)
    require(replay["accepted"] and replay["selected_epoch"] == plan["original_checkpoint_update"],
            "Selected development checkpoint failed replay")
    audit = read_pinned(args.audit, config["audit_sha256"])
    previous = read_pinned(ROOT / "docs/evidence/MATCHED_THREE_WAY_RESULTS.json", plan["matched_result_sha256"])
    group_report = read_pinned(ROOT / "docs/evidence/SINGLE_OUTPUT_CENTER_GROUP.json", plan["group_result_sha256"])
    require(previous["input_audit_sha256"] == config["audit_sha256"], "Comparison split differs")
    fixed_roles(audit)
    indices = np.asarray(audit["roles"]["test"])
    cases = [audit["cases"][i] for i in indices]
    truth = np.array([c["truth5"] for c in cases])
    masks = zoom_masks()
    zoom_sets = [tuple(np.asarray(config["magnifications"])[mask].tolist()) for mask in masks]
    with np.load(args.views, allow_pickle=False) as cache:
        views = cache["case_view_means"][indices]
    saved = torch.load(args.run / "selected.pt", map_location="cpu", weights_only=True)
    state = saved["model"]
    model = SingleOutputClassifier(state["mean"], state["scale"]).eval()
    model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        logits = model(torch.tensor(views, dtype=torch.float32), torch.from_numpy(masks))
        final = final_labels(logits).numpy()
    rebuilt = reconstruct(views, state["mean"].numpy(), state["scale"].numpy(), masks,
                          state["classifier.weight"].numpy(), state["classifier.bias"].numpy())
    max_diff = float(np.max(np.abs(rebuilt - logits.numpy())))
    require(max_diff <= 1.e-5 and np.array_equal(decide(rebuilt), final), "Independent replay failed")
    comparisons = {"single_output": final}
    for key, commitment in plan["comparator_record_sha256"].items():
        predictions = aligned_comparator(args.comparators / key / "test/records.jsonl", commitment,
                                         [c["case_id"] for c in cases], truth, zoom_sets)
        score = metrics(truth, predictions)
        original = previous["models"][key]["test"]
        for col, zooms in enumerate(zoom_sets):
            subset = original["subsets"]["+".join(map(str, zooms))]
            require(np.array_equal(np.bincount(5*truth + predictions[:, col], minlength=25).reshape(5,5),
                                   subset["five_class"]["confusion_matrix"]), "Comparator confusion differs")
        require(abs(score["mean_subset_macro_f1"] - original["mean_subset_macro_f1"]) < 1.e-10,
                "Comparator F1 differs")
        comparisons[key] = predictions
    summaries = {key: metrics(truth, predicted) for key, predicted in comparisons.items()}
    primary = summaries["single_output"]
    interval = bootstrap_accuracy([case["group"] for case in cases], truth, comparisons,
                                  plan["bootstrap_draws"], plan["bootstrap_seed"])
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "predictions.npz"
    np.savez_compressed(archive, indices=indices, truth=truth, masks=masks, logits=logits.numpy(), predictions=final)
    report = {
        "study": plan["study"], "job": os.environ["SLURM_JOB_ID"], "source": source,
        "archive_sha256": (ROOT / ".archive-sha256").read_text().strip(), "plan_sha256": digest(plan_path),
        "checkpoint_sha256": plan["source_checkpoint_sha256"], "evidence_sha256": digest(archive),
        "case_views_sha256": config["case_views_sha256"], "audit_sha256": config["audit_sha256"],
        "labels": config["labels"], "new_fits": 0, "roles": previous["roles"],
        "validation": replay["validation"], "group_check": group_report["held_out"],
        "matched_follow_up": summaries, "group_bootstrap": interval, "numpy_replay_max_abs": max_diff,
        "four_zoom_binary_correct": int(((final[:, -1] > 0) == (truth > 0)).sum()),
        "both_targets_met": primary["correct_final"] >= plan["final_correct_needed"]
                            and primary["correct_binary"] >= plan["binary_correct_needed"],
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(primary))


if __name__ == "__main__":
    main()
