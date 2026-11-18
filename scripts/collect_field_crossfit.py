"""Check fixed TRAIN-only group cross-fitting without fitting or selecting models.

Detailed group ownership and held-out predictions stay in the private shards.
Only aggregate results and provenance digests are written to the public receipt.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import torch

from magset.field_hierarchy import FieldHierarchy, PackedFields
from magset.shared_hierarchy import SharedHierarchy


REPO = Path(__file__).resolve().parents[1]
ARMS = ("control", "field")
PIN_KEYS = ("code_sha", "source_archive_sha256", "config_sha256", "plan_sha256",
            "audit_sha256", "train_fields_sha256", "input_receipt_sha256", "case_views_sha256")
MASKS = np.asarray([[i in subset for i in range(4)] for size in range(1, 5)
                    for subset in combinations(range(4), size)], dtype=bool)
LOGIT_TOLERANCE = 1.e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_digest(state):
    result = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().numpy()
        result.update(name.encode())
        result.update(str(array.shape).encode())
        result.update(str(array.dtype).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def integer_array(value, expected, context):
    value = np.asarray(value)
    require(value.dtype.kind in "iu" and value.ndim == 1 and np.array_equal(value, expected),
            context + " ownership/order differs")


def labels_from_logits(logits):
    require(logits.ndim == 3 and logits.shape[1:] == (15, 5)
            and logits.dtype == np.float32 and np.isfinite(logits).all(), "Invalid held-out logits")
    return np.where(logits[..., 0] >= 0, logits[..., 1:].argmax(axis=-1) + 1, 0)


def metrics(truth, predictions):
    require(truth.shape == (49,) and predictions.shape == (49, 15), "Incomplete aggregate TRAIN predictions")
    tables = np.stack([np.bincount(5 * truth + column, minlength=25).reshape(5, 5)
                       for column in predictions.T])
    denominator = tables.sum(axis=1) + tables.sum(axis=2)
    numerator = 2 * tables.diagonal(axis1=1, axis2=2)
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    totals = tables.sum(axis=0)
    correct_final = int(totals.trace())
    correct_binary = int(totals[0, 0] + totals[1:, 1:].sum())
    return {"cases": 49, "events": 735, "folds": 42,
            "class_counts": np.bincount(truth, minlength=5).tolist(),
            "correct_final": correct_final, "accuracy": correct_final / 735,
            "correct_binary": correct_binary, "binary_accuracy": correct_binary / 735,
            "mean_subset_macro_f1": float(f1.mean()),
            "class_recall": (totals.diagonal() / totals.sum(axis=1)).tolist(),
            "aggregate_confusion_true_rows": totals.tolist(),
            "routing_errors": 735 - correct_binary,
            "subtype_errors_after_correct_route": correct_binary - correct_final,
            "all_four_correct": int(tables[-1].trace()),
            "all_four_accuracy": int(tables[-1].trace()) / 49,
            "worst_subset_accuracy": float(tables.trace(axis1=1, axis2=2).min() / 49),
            "worst_subset_macro_f1": float(f1.mean(axis=1).min())}


def load_inputs(args):
    config, plan, audit, receipt = (read_json(path) for path in
                                   (args.config, args.plan, args.audit, args.input_receipt))
    require(digest(args.config) == digest(REPO / "configs/breakhis-field-hierarchy-v1.json")
            and digest(args.plan) == digest(REPO / "configs/breakhis-field-crossfit-v1.json"),
            "Use the exact registered primary and supplemental configurations")
    require(re.fullmatch(r"[0-9a-f]{40}", args.code_sha) is not None, "Invalid scientific source revision")
    for name, expected in ((".source-sha", args.code_sha),):
        marker = REPO / name
        if marker.exists():
            require(marker.read_text().strip() == expected, "Immutable source revision differs")
    require(plan["study"] == "breakhis-field-crossfit-v1" and plan["folds"] == 42
            and plan["shards"] == 6 and plan["updates"] == 10 and plan["seed"] == 812
            and plan["arms"] == list(ARMS)
            and plan["role"] == "train-only leave-one-conservative-group-out", "Supplemental plan differs")
    require(plan["primary_config_sha256"] == digest(args.config)
            and plan["audit_sha256"] == config["audit_sha256"] == digest(args.audit)
            and plan["case_views_sha256"] == config["case_views_sha256"], "Plan/configuration/input pins differ")
    require(config["seed"] == 812 and config["hidden"] == 32 and config["dropout"] == .2
            and config["learning_rate"] == .001 and config["weight_decay"] == .01
            and config["arms"] == list(ARMS), "Paired architecture/optimizer recipe differs")
    expected_receipt = {"train_fields_sha256": digest(args.train_fields), "config_sha256": digest(args.config),
                        "audit_sha256": digest(args.audit), "code_sha": args.code_sha,
                        "case_views_sha256": config["case_views_sha256"]}
    require(receipt == expected_receipt, "Outcome-free TRAIN input receipt differs")
    require(audit["manifest_sha256"] == config["manifest_sha256"] and len(audit["cases"]) == 82,
            "Original dataset audit binding differs")
    train = np.asarray(audit["roles"]["train"])
    require(train.shape == (49,) and train.dtype.kind in "iu" and len(set(train.tolist())) == 49
            and ((train >= 0) & (train < 82)).all(), "Invalid fixed TRAIN case role")
    groups = [audit["cases"][int(i)]["group"] for i in train]
    group_order = sorted(set(groups))
    truth = np.asarray([audit["cases"][int(i)]["truth5"] for i in train], dtype=np.int64)
    require(len(group_order) == 42 and np.array_equal(np.bincount(truth, minlength=5), [14, 24, 2, 5, 4]),
            "TRAIN group/label composition differs")
    for role in ("validation", "test"):
        other = set(audit["roles"][role])
        require(not set(train.tolist()) & other
                and not set(groups) & {audit["cases"][i]["group"] for i in other},
                "TRAIN cases/groups cross a protected role")
    with np.load(args.train_fields, allow_pickle=False) as saved:
        require(set(saved.files) == {"fields", "case", "zoom", "views", "path_sha256", "case_indices"},
                "Unexpected TRAIN bundle arrays")
        arrays = {key: saved[key].copy() for key in saved.files}
    fields, case, zoom, views = (arrays[key] for key in ("fields", "case", "zoom", "views"))
    integer_array(arrays["case_indices"], train, "TRAIN bundle")
    require(fields.shape == (4621, 1536) and fields.dtype == np.float32 and np.isfinite(fields).all()
            and views.shape == (49, 4, 1536) and views.dtype == np.float32 and np.isfinite(views).all()
            and case.shape == zoom.shape == (4621,) and case.dtype == zoom.dtype == np.int64
            and ((case >= 0) & (case < 49)).all() and ((zoom >= 0) & (zoom < 4)).all(),
            "TRAIN-only field bundle shape/type/content differs")
    expected_paths = {hashlib.sha256(row["relative_path"].encode()).hexdigest():
                      (int(row["case_index"]), int(row["view"]))
                      for row in audit["fields"] if row["case_index"] in set(train.tolist())}
    paths = arrays["path_sha256"].astype("U64").tolist()
    require(len(paths) == len(set(paths)) == len(expected_paths) == 4621
            and set(paths) == set(expected_paths), "TRAIN field-path coverage differs")
    for i, path in enumerate(paths):
        require(expected_paths[path] == (int(train[case[i]]), int(zoom[i])), "TRAIN field ownership differs")
    raw_means = np.empty((49, 4, 1536), dtype=np.float64)
    for i in range(49):
        for z in range(4):
            selected = fields[(case == i) & (zoom == z)]
            require(len(selected) > 0, "A TRAIN case/view has no original fields")
            raw_means[i, z] = selected.mean(axis=0, dtype=np.float64)
    require(np.array_equal(raw_means.astype(np.float32), views), "TRAIN bundle raw view means differ")
    pins = {**expected_receipt, "plan_sha256": digest(args.plan),
            "input_receipt_sha256": digest(args.input_receipt)}
    return dict(config=config, plan=plan, audit=audit, train=train, groups=np.asarray(groups),
                group_order=group_order, truth=truth, fields=fields, case=case, zoom=zoom,
                views=views, raw_means=raw_means, pins=pins)


def held_batch(inputs, held):
    lookup = np.full(49, -1, dtype=np.int64)
    lookup[held] = np.arange(len(held))
    keep = lookup[inputs["case"]] >= 0
    return PackedFields(torch.from_numpy(inputs["fields"][keep]),
                        torch.from_numpy(lookup[inputs["case"]][keep]),
                        torch.from_numpy(inputs["zoom"][keep]), torch.from_numpy(inputs["views"][held]))


def check_fold(folder, index, inputs, expected_pins, inventory):
    stem = "fold-%02d" % index
    fold_path = folder / (stem + ".json")
    fold = read_json(fold_path)
    group = inputs["group_order"][index]
    held = np.flatnonzero(inputs["groups"] == group)
    fit = np.flatnonzero(inputs["groups"] != group)
    require(fold["pins"] == expected_pins and fold["fold_index"] == index and fold["held_group"] == group,
            "Fold source/input/group pins differ")
    require(fold["study"] == inputs["plan"]["study"] and fold["role"] == inputs["plan"]["role"]
            and fold["epoch"] == 10 and set(fold["arms"]) == set(ARMS)
            and fold["validation_predictions"] == fold["test_predictions"] == 0
            and fold["held_out_predictions_per_arm"] == len(held) * 15,
            "Fold role, update budget or inference counters differ")
    for key, expected in (("held_local_ordinals", held), ("fit_local_ordinals", fit),
                          ("held_case_indices", inputs["train"][held]), ("fit_case_indices", inputs["train"][fit])):
        integer_array(fold[key], expected, "Fold " + key)
    require(0 < len(held) and len(fit) <= 48 and len(fit) + len(held) == 49
            and not set(inputs["groups"][held]) & set(inputs["groups"][fit])
            and (np.bincount(inputs["truth"][fit], minlength=5) > 0).all(), "Fold fitting/held group leakage")
    archive_path = folder / (stem + ".npz")
    require(digest(archive_path) == fold["archive_sha256"], "Changed private fold prediction archive")
    with np.load(archive_path, allow_pickle=False) as saved:
        archive = {key: saved[key].copy() for key in saved.files}
    require(set(archive) == {"held_local_ordinals", "fit_local_ordinals", "held_case_indices", "fit_case_indices",
                             "truth", "masks", "mean", "scale", "control_logits", "field_logits",
                             "control_predictions", "field_predictions"}, "Unexpected private fold arrays")
    for key, expected in (("held_local_ordinals", held), ("fit_local_ordinals", fit),
                          ("held_case_indices", inputs["train"][held]), ("fit_case_indices", inputs["train"][fit]),
                          ("truth", inputs["truth"][held])):
        integer_array(archive[key], expected, "Archive " + key)
    require(np.array_equal(archive["masks"], MASKS) and archive["masks"].dtype == np.bool_, "Fold subset masks differ")
    mean = inputs["raw_means"][fit].mean(axis=0)
    std = inputs["raw_means"][fit].std(axis=0, ddof=0)
    scale = np.where(std < 1.e-6, 1., std)
    require(archive["mean"].dtype == archive["scale"].dtype == np.float64
            and np.array_equal(archive["mean"], mean) and np.array_equal(archive["scale"], scale),
            "Fold normalizers were not fitted only on TRAIN minus held group")
    masks = torch.from_numpy(MASKS)
    batch = held_batch(inputs, held)
    outputs, errors, exact, initial = {}, {}, {}, []
    for arm, network in (("control", SharedHierarchy), ("field", FieldHierarchy)):
        checkpoint_path = folder / (stem + "-" + arm + ".pt")
        report = fold["arms"][arm]
        require(digest(checkpoint_path) == report["checkpoint_sha256"], "Changed private fold checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(checkpoint["epoch"] == 10 and checkpoint["config"] == inputs["config"]
                and checkpoint["fold_index"] == index and checkpoint["arm"] == arm
                and checkpoint["held_group"] == group and checkpoint["pins"] == expected_pins,
                "Fold checkpoint training/source/ownership context differs")
        integer_array(checkpoint["fit_local_ordinals"], fit, "Checkpoint fitting cases")
        integer_array(checkpoint["held_local_ordinals"], held, "Checkpoint held cases")
        torch.manual_seed(812)
        model = network(torch.from_numpy(mean), torch.from_numpy(scale), 32, .2).eval()
        require(state_digest(model.state_dict()) == report["initial_state_sha256"],
                "Seed812 initial parameters do not reproduce without fitting")
        state = checkpoint["model"]
        expected_state = model.state_dict()
        require(set(state) == set(expected_state), "Unexpected checkpoint parameter keys")
        for key, value in state.items():
            require(value.dtype == torch.float32 and value.shape == expected_state[key].shape
                    and torch.isfinite(value).all(), "Invalid checkpoint parameter " + key)
        require(torch.equal(state["mean"], torch.from_numpy(mean.astype(np.float32)))
                and torch.equal(state["scale"], torch.from_numpy(scale.astype(np.float32))),
                "Checkpoint normalizers include held-out or protected cases")
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            logits = model(batch.views if arm == "control" else batch, masks).numpy()
        saved_logits = archive[arm + "_logits"]
        saved_labels = labels_from_logits(saved_logits)
        require(logits.shape == saved_logits.shape == (len(held), 15, 5), "Held-out output shape differs")
        error = float(np.abs(logits - saved_logits).max())
        require(error <= LOGIT_TOLERANCE, "Saved-weight CPU4 held-out logit replay exceeds fixed tolerance")
        require(archive[arm + "_predictions"].dtype.kind in "iu"
                and archive[arm + "_predictions"].shape == (len(held), 15)
                and np.array_equal(archive[arm + "_predictions"], saved_labels)
                and np.array_equal(labels_from_logits(logits), saved_labels), "Held-out labels do not replay")
        losses = np.asarray(report["training_losses"])
        require(losses.shape == (10,) and np.isfinite(losses).all() and (losses >= 0).all()
                and report["trainable_parameters"] == 49477 and report["reload_max_absolute_error"] == 0.,
                "Fixed fitting budget or checkpoint reload differs")
        require(re.fullmatch(r"[0-9a-f]{64}", report["initial_state_sha256"]) is not None,
                "Missing initial-state digest")
        initial.append(report["initial_state_sha256"])
        outputs[arm], errors[arm], exact[arm] = saved_labels, error, bool(np.array_equal(logits, saved_logits))
        inventory.append({"file": folder.name + "/" + checkpoint_path.name, "sha256": digest(checkpoint_path)})
    require(initial[0] == initial[1], "Arms did not share initial parameter state")
    for path in (fold_path, archive_path):
        inventory.append({"file": folder.name + "/" + path.name, "sha256": digest(path)})
    return held, outputs, errors, exact


def collect(args):
    require(not args.output.exists(), "Refusing to overwrite an aggregate cross-fit receipt")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    inputs = load_inputs(args)
    predictions = {arm: np.full((49, 15), -1, dtype=np.int64) for arm in ARMS}
    covered, folds, inventory = np.zeros(49, dtype=np.int64), set(), []
    max_errors = {arm: 0. for arm in ARMS}
    exact_replays = {arm: True for arm in ARMS}
    source_archive = None
    for shard in range(6):
        folder = args.shards / str(shard)
        root_path = folder / "result.json"
        root = read_json(root_path)
        pins = root["pins"]
        require(set(pins) == set(PIN_KEYS) and all(pins[key] == value for key, value in inputs["pins"].items()),
                "Shard source/configuration/input receipt pins differ")
        require(re.fullmatch(r"[0-9a-f]{64}", pins["source_archive_sha256"]) is not None,
                "Invalid immutable source archive digest")
        source_archive = pins["source_archive_sha256"] if source_archive is None else source_archive
        require(pins["source_archive_sha256"] == source_archive, "Shard source archives differ")
        marker = REPO / ".archive-sha256"
        if marker.exists():
            require(marker.read_text().strip() == source_archive, "Collector immutable archive differs from workers")
        indices = list(range(shard, 42, 6))
        require(root["shard"] == shard and root["fold_indices"] == indices
                and set(root["fold_json_sha256"]) == {"fold-%02d.json" % i for i in indices},
                "Shard/fold assignment differs")
        require(root["study"] == inputs["plan"]["study"] and root["role"] == inputs["plan"]["role"]
                and root["status"] == "complete fixed-update TRAIN-only group cross-fitting shard"
                and root["folds"] == 7 and root["shards"] == 6 and root["updates"] == 10
                and root["arms"] == list(ARMS), "Shard declared role/update/architecture scope differs")
        expected_files = {"result.json"} | {"fold-%02d%s" % (i, suffix) for i in indices
                                            for suffix in (".json", ".npz", "-control.pt", "-field.pt")}
        require({path.name for path in folder.iterdir()} == expected_files,
                "Unexpected files or prediction roles in private cross-fit shard")
        require(root["validation_predictions"] == root["test_predictions"] == 0,
                "Cross-fitting queried protected roles")
        expected_held_cases = 0
        for index in indices:
            filename = "fold-%02d.json" % index
            require(digest(folder / filename) == root["fold_json_sha256"][filename], "Bound private fold receipt changed")
            held, outputs, errors, exact = check_fold(folder, index, inputs, pins, inventory)
            require(index not in folds and (covered[held] == 0).all(), "Repeated held-out TRAIN group/case")
            covered[held] += 1
            folds.add(index)
            expected_held_cases += len(held)
            for arm in ARMS:
                predictions[arm][held] = outputs[arm]
                max_errors[arm] = max(max_errors[arm], errors[arm])
                exact_replays[arm] = exact_replays[arm] and exact[arm]
        require(root["held_out_predictions_per_arm"] == expected_held_cases * 15
                and root["held_out_cases"] == expected_held_cases,
                "Shard held-out prediction count differs")
        inventory.append({"file": str(shard) + "/result.json", "sha256": digest(root_path)})
    require(folds == set(range(42)) and (covered == 1).all(), "Incomplete 42-group/49-case held-out coverage")
    result = {"study": inputs["plan"]["study"], "status": "accepted fixed TRAIN-only group cross-fit replay",
              "role": inputs["plan"]["role"], "code_sha": args.code_sha,
              "source_archive_sha256": source_archive, **{k: v for k, v in inputs["pins"].items() if k != "code_sha"},
              "fold_inventory_sha256": canonical_digest(sorted(inventory, key=lambda row: row["file"])),
              "folds": 42, "training_cases": 49, "training_groups": 42, "fields": 4621,
              "updates_per_fold": 10, "arms": {arm: metrics(inputs["truth"], predictions[arm]) for arm in ARMS},
              "replay_max_absolute_logit_error": max_errors, "bit_exact_saved_weight_logits": exact_replays,
              "saved_weight_logit_tolerance": LOGIT_TOLERANCE, "all_saved_hard_labels_replayed": True,
              "all_held_cases_covered_once": True, "fold_normalizers_exclude_held_groups": True,
              "validation_predictions": 0, "test_predictions": 0,
              "collector_fits": 0, "new_encoder_forwards": 0,
              "inference_scope": "saved-weight replay on held-out TRAIN cases only",
              "interpretation": "descriptive cross-fit corroboration; no checkpoint, architecture or threshold selection; dependent zoom conditions; not fresh clinical confirmation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shards", "train-fields", "audit", "config", "plan", "input-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    print(json.dumps(collect(parser.parse_args()), allow_nan=False))


if __name__ == "__main__":
    main()
