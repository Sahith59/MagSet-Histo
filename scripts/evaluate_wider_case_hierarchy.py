"""Compare two fixed image-coverage methods using only the original development roles."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.check_field_hierarchy import (digest, fixed_roles, metrics, numpy_logits,
                                          paired_intervals, require, zoom_masks)
from scripts.run_subtype_initialization import fit_and_verify

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (812, 813, 814, 815)
ARMS = ("center", "wider")


def load_inputs(args, config):
    require(digest(args.audit) == config["audit_sha256"] and
            digest(args.original_views) == config["case_views_sha256"], "Historical source digest differs")
    audit = json.loads(args.audit.read_text())
    train, val = fixed_roles(audit)
    case_indices = np.r_[train, val]
    require(audit["manifest_sha256"] == config["manifest_sha256"], "Manifest changed")
    receipt = json.loads((args.features / "feature-receipt.json").read_text())
    bundle_path = args.features / "development-views.npz"
    require(digest(bundle_path) == receipt["development_views_sha256"], "Feature bundle digest differs")
    for key in ("reader_revision", "weights_sha256", "case_views_sha256", "manifest_sha256", "audit_sha256"):
        require(receipt[key] == config[key], "Feature receipt differs at " + key)
    require(receipt["followup_fields"] == 0 and receipt["center_max_abs_error"] == 0,
            "Expected exact center and zero follow-up fields")
    with np.load(bundle_path, allow_pickle=False) as data:
        require(np.array_equal(data["case_indices"], case_indices) and
                np.array_equal(data["magnifications"], config["magnifications"]),
                "Development case order or magnification order differs")
        views = {"center": data["center_views"], "wider": data["wide_views"]}
    for values in views.values():
        require(values.dtype == np.float64 and values.shape == (len(case_indices), 4, 1536) and
                np.isfinite(values).all(), "Invalid development descriptor type, shape or value")
    with np.load(args.original_views, allow_pickle=False) as prior:
        require(np.array_equal(prior["case_indices"], np.arange(82)) and
                np.array_equal(prior["train_case_indices"], train) and
                np.array_equal(prior["masks"], zoom_masks()), "Historical input ordering differs")
        require(np.array_equal(views["center"], prior["case_view_means"][case_indices]),
                "Center descriptors differ from the original experiment")
    labels = np.array([audit["cases"][i]["truth5"] for i in case_indices])
    groups = np.array([audit["cases"][i]["group"] for i in case_indices])
    return views, labels, groups, case_indices, len(train), audit, receipt


def paired_case_changes(truth, center, wider):
    differences = (wider == truth[:, None]).sum(1) - (center == truth[:, None]).sum(1)
    return {"cases_improved": int((differences > 0).sum()),
            "cases_tied": int((differences == 0).sum()),
            "cases_worse": int((differences < 0).sum())}


def compare_role(seed, role, inputs, config, prior_path, prior_digest, output):
    views, labels, groups, case_indices, n_train, audit, _ = inputs
    train = np.arange(n_train)
    if role == "validation":
        rows = np.arange(n_train, len(labels))
        folds = [(train, rows)]
    else:
        require(role == "group_check", "Unexpected evaluation role")
        rows = train
        folds = [(train[groups[train] != group], train[groups[train] == group])
                 for group in np.unique(groups[train])]
    require(digest(prior_path) == prior_digest, "Archived control predictions have changed")
    with np.load(prior_path, allow_pickle=False) as recorded:
        require(np.array_equal(recorded["case_indices"], case_indices[rows]) and
                np.array_equal(recorded["truth"], labels[rows]) and
                np.array_equal(recorded["masks"], zoom_masks()), "Historical predictions use other roles or masks")
        historic = recorded["original"]
    full = {arm: np.full((len(labels), 15), -1, dtype=np.int64) for arm in ARMS}
    subtypes = {}
    worst_error = 0.
    weights = {}
    for fold, (fitting, held) in enumerate(folds):
        require(not set(groups[fitting]) & set(groups[held]), "A held-out group entered fitting")
        for arm, array in views.items():
            checkpoint = output / f"{seed}-{role}-{fold:02d}-{arm}.pt"
            # Both inputs use the unmodified case-first model and original random initialization.
            decisions, error = fit_and_verify("original", array[fitting], labels[fitting], array[held],
                                              {**config, "seed": seed}, checkpoint)
            require((full[arm][held] == -1).all(), "A case was evaluated twice")
            full[arm][held] = decisions
            worst_error = max(worst_error, error)
            weights[checkpoint.name] = digest(checkpoint)
            if role == "validation":
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                logits = numpy_logits("control", {k: v.numpy() for k, v in state.items()},
                                      {"views": array[held].astype(np.float32), "masks": zoom_masks()})
                subtypes[arm + "_if_malignant"] = 1 + logits[..., 1:].argmax(-1)
    rows_by_arm = {arm: matrix[rows] for arm, matrix in full.items()}
    require(np.array_equal(rows_by_arm["center"], historic), "Control disagrees with historical hard predictions")
    path = output / f"{seed}-{role}.npz"
    np.savez_compressed(path, case_indices=case_indices[rows], truth=labels[rows],
                        masks=zoom_masks(), **rows_by_arm, **subtypes)
    with np.load(path, allow_pickle=False) as saved:
        result = {arm: metrics(saved["truth"], saved[arm]) for arm in ARMS}
        result["paired_cases"] = paired_case_changes(saved["truth"], saved["center"], saved["wider"])
        if role == "validation":
            bootstrap = paired_intervals({"audit": audit, "val": saved["case_indices"], "truth": saved["truth"]},
                                         saved["center"], saved["wider"])
            result["paired_group_interval"] = bootstrap["field_minus_control"]
            result["bootstrap"] = {key: bootstrap[key] for key in ("groups", "draws", "seed", "unit")}
            result["conditional_correct_if_truth_is_malignant"] = {
                arm: [int((saved[arm + "_if_malignant"][saved["truth"] == label] == label).sum())
                      for label in range(1, 5)] for arm in ARMS}
    result.update({"folds": len(folds), "max_logit_replay_error": worst_error,
                   "predictions_sha256": digest(path), "historical_control_archive_sha256": prior_digest,
                   "matched_historical_control_decisions": int(historic.size)})
    return result, weights


def progression(training, validation):
    def change(pair, key):
        return pair["wider"][key] - pair["center"][key]
    validated = {
        "validation_five_increases": change(validation, "correct_final") > 0,
        "validation_binary_increases": change(validation, "correct_binary") > 0,
        "validation_macro_f1_holds": change(validation, "mean_subset_macro_f1") >= 0,
        "training_five_increases_in_at_least_three_seeds":
            sum(change(p, "correct_final") > 0 for p in training) >= 3,
        "mean_training_binary_holds": sum(change(p, "correct_binary") for p in training) >= 0,
        "mean_training_macro_f1_holds": sum(change(p, "mean_subset_macro_f1") for p in training) >= 0,
    }
    require(len(training) == 4, "Four paired seeds required")
    return {"criteria": validated, "passes": all(validated.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "audit", "original-views", "previous-runs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / ".source-sha").read_text().strip()
    require(source == os.environ["MAGSET_CODE_SHA"], "Immutable source revision differs")
    config_path = ROOT / "configs/breakhis-shared-hierarchy-v1.json"
    config = json.loads(config_path.read_text())
    inputs = load_inputs(args, config)
    baseline_path = ROOT / "docs/evidence/SUBTYPE_INITIALIZATION.json"
    stability_path = ROOT / "docs/evidence/SUBTYPE_INITIALIZATION_STABILITY.json"
    primary = json.loads(baseline_path.read_text())
    stability = json.loads(stability_path.read_text())
    base_dir = args.previous_runs / primary["slurm_job"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    training = {}
    inventory = {}
    for seed in SEEDS:
        folder = base_dir if seed == 812 else args.previous_runs / stability["slurm_job"] / str(seed)
        ref = primary["group_check"] if seed == 812 else stability["by_seed"][str(seed)]
        training[seed], weights = compare_role(seed, "group_check", inputs, config,
            folder / "group_check.npz", ref["archive_sha256"], args.output)
        inventory.update(weights)
        print("Completed training groups for seed", seed, flush=True)
    validation, weights = compare_role(812, "validation", inputs, config, base_dir / "validation.npz",
                                        primary["validation"]["archive_sha256"], args.output)
    inventory.update(weights)
    require(len(inventory) == 338, "Missing primary or group checkpoint")
    inventory_path = args.output / "weights-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    report = {"study": "BreakHis case-first UNI2-h, center versus three crops",
              "code_sha": source, "slurm_job": os.environ["SLURM_JOB_ID"], "torch": str(torch.__version__),
              "code_archive_sha256": (ROOT / ".archive-sha256").read_text().strip(),
              "protocol_sha256": digest(ROOT / "docs/BREAKHIS_WIDER_FIELD.md"),
              "model_config_sha256": digest(config_path), "reference_primary_sha256": digest(baseline_path),
              "reference_stability_sha256": digest(stability_path),
              "feature_receipt_sha256": digest(args.features / "feature-receipt.json"),
              "development_views_sha256": inputs[-1]["development_views_sha256"],
              "weights_inventory_sha256": digest(inventory_path), "weights_independently_replayed": len(inventory),
              **{k: config[k] for k in ("audit_sha256", "case_views_sha256", "manifest_sha256",
                                        "reader_revision", "weights_sha256")},
              "training_groups": training, "validation_seed_812": validation,
              "updates_per_fit": 10, "followup_predictions": 0,
              "progression": progression(list(training.values()), validation)}
    (args.output / "aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["progression"]), flush=True)


if __name__ == "__main__":
    main()
