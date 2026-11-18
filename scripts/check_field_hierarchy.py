"""Independently check the registered validation-only field-pooling comparison.

Inference, classification arithmetic and paired intervals use NumPy/SciPy. Torch
is used only to deserialize this run's hash-bound parameter dictionaries. This
checker does not import the producer or fit a model.
"""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import ndtr
import torch


ARMS = ("control", "field")
EPOCHS = [10, 20, 30, 40]
METRIC_TOLERANCE = 1.e-12
LOGIT_TOLERANCE = 1.e-4
REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def zoom_masks():
    return np.asarray([[i in subset for i in range(4)]
                       for size in range(1, 5)
                       for subset in combinations(range(4), size)], dtype=bool)


def predictions(logits):
    require(logits.ndim == 3 and logits.shape[1:] == (15, 5)
            and np.isfinite(logits).all(), "Expected finite case/mask/five-score logits")
    return np.where(logits[..., 0] >= 0, logits[..., 1:].argmax(axis=-1) + 1, 0)


def metrics(truth, predicted):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    require(truth.ndim == 1 and predicted.shape == (len(truth), 15)
            and np.issubdtype(truth.dtype, np.integer)
            and np.issubdtype(predicted.dtype, np.integer)
            and ((truth >= 0) & (truth < 5)).all()
            and ((predicted >= 0) & (predicted < 5)).all(), "Invalid classification rows")
    tables = np.stack([np.bincount(5 * truth + column, minlength=25).reshape(5, 5)
                       for column in predicted.T])
    denominators = tables.sum(axis=1) + tables.sum(axis=2)
    numerators = 2 * tables.diagonal(axis1=1, axis2=2)
    f1 = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                   where=denominators != 0)
    accuracy = tables.trace(axis1=1, axis2=2) / len(truth)
    total = tables.sum(axis=0)
    correct_final = int(total.trace())
    correct_binary = int(total[0, 0] + total[1:, 1:].sum())
    recalls = np.divide(total.diagonal(), total.sum(axis=1),
                        out=np.zeros(5, dtype=float), where=total.sum(axis=1) != 0)
    return {
        "cases": len(truth), "events": predicted.size,
        "correct_final": correct_final, "correct_binary": correct_binary,
        "accuracy": correct_final / predicted.size,
        "binary_accuracy": correct_binary / predicted.size,
        "mean_subset_macro_f1": float(f1.mean(axis=1).mean()),
        "all_four_correct": int(tables[-1].trace()),
        "all_four_accuracy": float(accuracy[-1]),
        "worst_subset_accuracy": float(accuracy.min()),
        "worst_subset_macro_f1": float(f1.mean(axis=1).min()),
        "class_recall": recalls.tolist(),
        "aggregate_confusion_true_rows": total.tolist(),
        "routing_errors": predicted.size - correct_binary,
        "subtype_errors_after_correct_route": correct_binary - correct_final,
    }


def compare_metrics(actual, reported, context, allow_missing_worst=False):
    for key, value in actual.items():
        if allow_missing_worst and key.startswith("worst_subset_") and key not in reported:
            continue
        require(key in reported, context + ": missing " + key)
        other = np.asarray(reported[key])
        expected = np.asarray(value)
        require(other.shape == expected.shape and np.isfinite(other).all()
                and np.allclose(expected, other, rtol=0, atol=METRIC_TOLERANCE),
                context + ": metric mismatch for " + key)


def fixed_roles(audit):
    require(len(audit["cases"]) == 82, "Unexpected audited case count")
    expected = {
        "train": (49, 42, [14, 24, 2, 5, 4], 4621),
        "validation": (17, 15, [5, 7, 2, 2, 1], 1584),
        "test": (16, 13, [5, 7, 1, 2, 1], 1704),
    }
    role_sets, group_sets = [], []
    for role, (n_cases, n_groups, class_counts, n_fields) in expected.items():
        indices = np.asarray(audit["roles"][role])
        require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                and len(np.unique(indices)) == len(indices) == n_cases
                and ((indices >= 0) & (indices < 82)).all(), "Invalid " + role + " role")
        groups = {audit["cases"][int(i)]["group"] for i in indices}
        labels = np.asarray([audit["cases"][int(i)]["truth5"] for i in indices])
        field_count = sum(field["case_index"] in set(indices) for field in audit["fields"])
        require(len(groups) == n_groups and np.array_equal(np.bincount(labels, minlength=5), class_counts)
                and field_count == n_fields, "Audited " + role + " composition differs")
        role_sets.append(set(indices))
        group_sets.append(groups)
    require(set.union(*role_sets) == set(range(82)), "Roles do not cover the audited cases")
    for i, left in enumerate(role_sets):
        for j in range(i + 1, len(role_sets)):
            require(not left & role_sets[j] and not group_sets[i] & group_sets[j],
                    "Related cases cross role boundaries")
    return np.asarray(audit["roles"]["train"]), np.asarray(audit["roles"]["validation"])


def load_inputs(args, config, root):
    require(digest(args.config) == root["config_sha256"]
            and digest(args.audit) == config["audit_sha256"]
            and digest(args.views) == config["case_views_sha256"], "Configuration/input digest mismatch")
    prior_path = REPO / "docs/evidence/BREAKHIS_SHARED_HIERARCHY_VALIDATION.json"
    require(digest(prior_path) == config["previous_result_sha256"], "Pinned prior result changed")
    prior = read_json(prior_path)
    require(config["validation_epochs"] == EPOCHS and config["epochs"] == 40
            and config["seed"] == 812 and config["hidden"] == 32
            and config["dropout"] == .2 and config["learning_rate"] == .001
            and config["weight_decay"] == .01, "Registered fitting/selection budget differs")
    require(config["arms"] == list(ARMS) and config["primary_arm"] == "field"
            and config["selection"] == "first maximum mean per-subset five-category macro F1"
            and config["evaluation_role"] == "validation only"
            and config["magnifications"] == [40, 100, 200, 400]
            and config["labels"] == ["benign", "ductal", "lobular", "mucinous", "papillary"]
            and config["thresholds"] == {"final_five_accuracy_strictly_greater_than": .9,
                                         "binary_accuracy_strictly_greater_than": .95}
            and config["paired_group_bootstrap"] == {"samples": 2000, "seed": 20261117},
            "Registered arms, labels, targets or selection rule differ")
    require(root["test_predictions"] == 0 and root["train_fields"] == 4621
            and root["validation_fields"] == 1584, "Run scope/counters differ")
    require(re.fullmatch(r"[0-9a-f]{40}", root["code_sha"]) is not None
            and re.fullmatch(r"[0-9a-f]{64}", root["source_archive_sha256"]) is not None,
            "Missing source revision/archive binding")
    source_checks = {}
    for name, key in ((".source-sha", "code_sha"), (".archive-sha256", "source_archive_sha256")):
        path = REPO / name
        source_checks[name] = path.exists()
        if path.exists():
            require(path.read_text().strip() == root[key], "Available immutable source binding differs")
    audit = read_json(args.audit)
    require(audit["manifest_sha256"] == config["manifest_sha256"] == prior["inputs"]["manifest_sha256"],
            "Original manifest binding differs")
    for key in ("audit_sha256", "case_views_sha256", "reader", "reader_revision", "weights_sha256"):
        require(config[key] == prior["inputs"][key], "Prior input binding differs: " + key)
    feature_config_path = REPO / "configs/breakhis-uni2-h-v1-kernel.json"
    extraction_path = REPO / "configs/uni2-h-v1.json"
    require(digest(feature_config_path) == config["feature_config_sha256"] == root["feature_config_sha256"]
            and digest(extraction_path) == config["extraction_config_sha256"] == root["extraction_config_sha256"],
            "Pinned field feature/extraction configuration differs")
    feature_config, extraction = read_json(feature_config_path), read_json(extraction_path)
    require(feature_config["cache_provenance"] == root["cache_provenance"]
            and extraction["manifest_sha256"] == config["manifest_sha256"]
            and extraction["encoder"]["repository"] == config["reader"]
            and extraction["encoder"]["revision"] == config["reader_revision"]
            and extraction["encoder"]["weights_sha256"] == config["weights_sha256"],
            "Frozen reader or shard provenance differs")
    train, val = fixed_roles(audit)
    masks = zoom_masks()
    with np.load(args.views, allow_pickle=False) as source:
        require(np.array_equal(source["case_indices"], np.arange(82))
                and np.array_equal(source["train_case_indices"], train)
                and np.array_equal(source["masks"], masks), "Pinned preprocessing identities/masks differ")
        views = source["case_view_means"].copy()
        require(views.shape == (82, 4, 1536) and np.isfinite(views).all(), "Invalid frozen view descriptors")
        mean = views[train].mean(axis=0)
        std = views[train].std(axis=0)
        scale = np.where(std < 1.e-6, 1., std)
        require(np.array_equal(mean, source["mean"]) and np.array_equal(scale, source["scale"]),
                "Pinned preprocessing was not fitted on the fixed TRAIN cases")
    fields_path = args.run / "validation_fields.npz"
    require(digest(fields_path) == root["validation_fields_sha256"], "Private validation field archive changed")
    with np.load(fields_path, allow_pickle=False) as source:
        fields, case, zoom = (source[key].copy() for key in ("fields", "case", "zoom"))
        local_views = source["views"].copy()
        path_hashes = source["path_sha256"].copy()
    require(fields.shape == (1584, 1536) and fields.dtype == np.float32
            and np.isfinite(fields).all() and case.shape == zoom.shape == (1584,)
            and np.issubdtype(case.dtype, np.integer) and np.issubdtype(zoom.dtype, np.integer)
            and ((case >= 0) & (case < 17)).all() and ((zoom >= 0) & (zoom < 4)).all(),
            "Private validation field values/indices differ from scope")
    require(local_views.shape == (17, 4, 1536) and local_views.dtype == np.float32
            and np.array_equal(local_views, views[val].astype(np.float32)), "Private view descriptors differ")
    expected_counts = np.zeros((17, 4), dtype=np.int64)
    ordinal = {int(index): local for local, index in enumerate(val)}
    for field in audit["fields"]:
        if field["case_index"] in ordinal:
            expected_counts[ordinal[field["case_index"]], field["view"]] += 1
    counts = np.bincount(case * 4 + zoom, minlength=68).reshape(17, 4)
    require(np.array_equal(counts, expected_counts) and (counts > 0).all(),
            "Original audited validation field counts/assignments differ")
    expected_fields = sorted((field for field in audit["fields"] if field["case_index"] in ordinal),
                             key=lambda field: (ordinal[field["case_index"]], field["view"], field["relative_path"]))
    expected_hashes = np.asarray([hashlib.sha256(field["relative_path"].encode("utf-8")).hexdigest()
                                  for field in expected_fields], dtype="S64")
    require(path_hashes.dtype == np.dtype("S64") and np.array_equal(path_hashes, expected_hashes)
            and len(np.unique(path_hashes)) == 1584
            and np.array_equal(case, [ordinal[field["case_index"]] for field in expected_fields])
            and np.array_equal(zoom, [field["view"] for field in expected_fields]),
            "Exact original validation field identities/order/role membership differ")
    rebuilt_views = np.empty_like(local_views, dtype=np.float64)
    for i in range(17):
        for z in range(4):
            rebuilt_views[i, z] = fields[(case == i) & (zoom == z)].mean(axis=0, dtype=np.float64)
    mean_error = float(np.abs(rebuilt_views - views[val]).max())
    require(mean_error < 1.e-6, "Private field means do not reproduce pinned frozen descriptors")
    exact_cache = check_field_cache(args.field_cache, feature_config, extraction, audit, expected_fields, fields)
    return dict(audit=audit, prior=prior, train=train, val=val, masks=masks,
                truth=np.asarray([audit["cases"][int(i)]["truth5"] for i in val]),
                mean=mean.astype(np.float32), scale=scale.astype(np.float32), views=local_views,
                fields=fields, case=case, zoom=zoom, counts=counts,
                field_mean_max_abs_error=mean_error, source_checks=source_checks,
                exact_field_cache_check=exact_cache)


def check_field_cache(directory, feature_config, extraction, audit, expected_fields, fields):
    """Optionally attest every packed vector directly against the pinned shards."""
    if directory is None:
        return {"performed": False, "reason": "--field-cache was not supplied"}
    checked = 0
    for zoom, magnification in enumerate((40, 100, 200, 400)):
        pins = feature_config["cache_provenance"][str(magnification)]
        stem = directory / (feature_config["encoder_name"] + "-" + str(magnification))
        npz, metadata_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
        require(digest(npz) == pins["npz_sha256"] and digest(metadata_path) == pins["json_sha256"],
                "Original frozen feature shard digest differs")
        metadata = read_json(metadata_path)
        require(metadata["feature_sha256"] == pins["npz_sha256"]
                and metadata["manifest_sha256"] == audit["manifest_sha256"]
                and metadata["magnification"] == magnification and metadata.get("smoke_limit", 0) == 0
                and metadata["encoder"] == extraction["encoder"], "Original shard metadata differs")
        with np.load(npz, allow_pickle=False) as source:
            paths, vectors = source["paths"].tolist(), source["features"]
            require(vectors.shape == (pins["images"], 1536) and np.isfinite(vectors).all()
                    and metadata["images"] == len(paths) == len(set(paths)) == len(vectors)
                    and set(paths) == {field["relative_path"] for field in audit["fields"] if field["view"] == zoom},
                    "Original frozen shard membership/values differ")
            lookup = {path: i for i, path in enumerate(paths)}
            packed_indices = [i for i, field in enumerate(expected_fields) if field["view"] == zoom]
            shard_indices = [lookup[expected_fields[i]["relative_path"]] for i in packed_indices]
            require(np.array_equal(fields[packed_indices], vectors[shard_indices].astype(np.float32)),
                    "Packed validation vectors differ bytewise from original frozen field descriptors")
            checked += len(packed_indices)
    require(checked == 1584, "Incomplete exact frozen-field comparison")
    return {"performed": True, "vectors_checked": checked, "original_shards_verified": 4,
            "all_packed_float32_vectors_identical": True}


def load_weights(path, config, inputs, selected_epoch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["epoch"] == selected_epoch and checkpoint["config"] == config,
            "Selected checkpoint epoch/configuration mismatch")
    state = {key: value.detach().cpu().numpy() for key, value in checkpoint["model"].items()}
    shapes = {"mean": (4, 1536), "scale": (4, 1536),
              "classifier.0.weight": (32, 1540), "classifier.0.bias": (32,),
              "classifier.3.weight": (5, 32), "classifier.3.bias": (5,)}
    require(set(state) == set(shapes), "Unexpected trainable network/state keys")
    for key, shape in shapes.items():
        require(state[key].shape == shape and state[key].dtype == np.float32
                and np.isfinite(state[key]).all(), "Invalid parameter " + key)
    require(np.array_equal(state["mean"], inputs["mean"])
            and np.array_equal(state["scale"], inputs["scale"]), "Checkpoint TRAIN normalizer differs")
    return state


def numpy_logits(arm, state, inputs):
    standardized_views = (inputs["views"] - state["mean"]) / state["scale"]
    first = state["classifier.0.weight"].astype(np.float64)
    bias = state["classifier.0.bias"].astype(np.float64)
    last = state["classifier.3.weight"].astype(np.float64)
    last_bias = state["classifier.3.bias"].astype(np.float64)
    if arm == "field":
        standardized_fields = ((inputs["fields"] - state["mean"][inputs["zoom"]])
                               / state["scale"][inputs["zoom"]])
        field_projection = np.einsum("ni,hi->nh", standardized_fields, first[:, :1536], dtype=np.float64)
    rows = []
    for mask in inputs["masks"]:
        pooled = standardized_views[:, mask].mean(axis=1)
        norm = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1.e-12)
        factor = np.sqrt(np.float32(1536)) / norm
        flag_bias = np.einsum("hi,i->h", first[:, 1536:], mask.astype(np.float64)) + bias
        if arm == "control":
            hidden = np.einsum("ni,hi->nh", pooled * factor, first[:, :1536], dtype=np.float64) + flag_bias
            latent = hidden * ndtr(hidden)
        else:
            hidden = field_projection * factor[inputs["case"]] + flag_bias
            activated = hidden * ndtr(hidden)
            view_latent = np.zeros((len(standardized_views), 4, 32), dtype=np.float64)
            np.add.at(view_latent, (inputs["case"], inputs["zoom"]), activated)
            view_latent /= inputs["counts"][..., None]
            latent = view_latent[:, mask].mean(axis=1)
        rows.append(np.einsum("nh,oh->no", latent, last) + last_bias)
    return np.stack(rows, axis=1)


def check_arm(arm, args, config, root, inputs):
    directory = args.run / arm
    report_path = directory / "result.json"
    require(digest(report_path) == root["arm_result_sha256"][arm], "Changed bound arm report: " + arm)
    report = read_json(report_path)
    for key in ("code_sha", "source_archive_sha256", "config_sha256", "slurm_job"):
        require(report[key] == root[key], arm + ": source/run provenance differs: " + key)
    require(report["role"] == "validation" and report["test_predictions"] == 0
            and report["train_cases"] == 49 and report["train_class_counts"] == [14, 24, 2, 5, 4]
            and report["validation_class_counts"] == [5, 7, 2, 2, 1]
            and report["trainable_parameters"] == 49477
            and report["reload_max_absolute_error"] == 0., arm + ": training/evaluation scope differs")
    if "inputs" in report:
        for key in ("manifest_sha256", "audit_sha256", "case_views_sha256", "reader", "reader_revision", "weights_sha256"):
            require(report["inputs"][key] == config[key], arm + ": input binding differs: " + key)
    for filename, key in (("selected.pt", "checkpoint_sha256"), ("validation.npz", "validation_sha256")):
        require(digest(directory / filename) == report[key], arm + ": changed artifact " + filename)
    require(len(report["fitting_loss_before_each_update"]) == 40
            and np.isfinite(report["fitting_loss_before_each_update"]).all(), arm + ": incomplete fitting trajectory")
    with np.load(directory / "validation.npz", allow_pickle=False) as archive:
        require(np.array_equal(archive["case_indices"], inputs["val"])
                and np.array_equal(archive["truth"], inputs["truth"])
                and np.array_equal(archive["masks"], inputs["masks"]), arm + ": validation identities/labels/masks differ")
        require(archive["validation_epochs"].tolist() == EPOCHS
                == [row["epoch"] for row in report["history"]]
                and archive["epoch_logits"].shape == (4, 17, 15, 5), arm + ": incomplete checkpoint history")
        histories, scores = [], []
        for row, logits in zip(report["history"], archive["epoch_logits"]):
            rebuilt = metrics(inputs["truth"], predictions(logits))
            compare_metrics(rebuilt, row, arm + " epoch " + str(row["epoch"]))
            histories.append(rebuilt)
            scores.append(rebuilt["mean_subset_macro_f1"])
        best = int(np.argmax(scores))
        require(EPOCHS[best] == report["selected_epoch"], arm + ": first maximum selection differs")
        require(np.array_equal(archive["epoch_logits"][best], archive["logits"]),
                arm + ": selected logits are not the selected checkpoint")
        saved_logits = archive["logits"].copy()
        saved_predictions = archive["predictions"].copy()
        require(np.array_equal(predictions(saved_logits), saved_predictions), arm + ": saved final labels differ")
    state = load_weights(directory / "selected.pt", config, inputs, EPOCHS[best])
    logits = numpy_logits(arm, state, inputs)
    error = float(np.abs(logits - saved_logits).max())
    require(error < LOGIT_TOLERANCE, arm + ": independent NumPy logits exceed tolerance")
    predicted = predictions(logits)
    require(np.array_equal(predicted, saved_predictions), arm + ": independent final labels differ")
    selected = metrics(inputs["truth"], predicted)
    compare_metrics(selected, report["selected"], arm + " selected")
    fitting = report["selected_fitting_diagnostic"]
    require(fitting["cases"] == 49 and fitting["events"] == 735, arm + ": fitting diagnostic scope differs")
    # Fitting predictions are not supplied: check their published internal arithmetic,
    # rather than misrepresenting this as an independent TRAIN inference replay.
    check_fitting_arithmetic(fitting, arm)
    if arm == "control":
        prior = inputs["prior"]
        require(report["selected_epoch"] == prior["selected_epoch"], "Control no longer selects the pinned pilot epoch")
        for actual, previous in zip(histories, prior["history"]):
            compare_metrics(actual, previous, "Pinned control checkpoint", allow_missing_worst=True)
        compare_metrics(selected, prior["selected"], "Pinned selected control", allow_missing_worst=True)
    return {"result_sha256": digest(report_path), "checkpoint_sha256": report["checkpoint_sha256"],
            "validation_sha256": report["validation_sha256"], "verified_epochs": EPOCHS,
            "selected_epoch": EPOCHS[best], "selected": selected,
            "numpy_checkpoint_max_abs_logit_error": error,
            "all_saved_decisions_match": True}, predicted


def check_fitting_arithmetic(row, arm):
    table = np.asarray(row["aggregate_confusion_true_rows"])
    require(table.shape == (5, 5) and np.issubdtype(table.dtype, np.integer) and (table >= 0).all()
            and np.array_equal(table.sum(axis=1), np.asarray([14, 24, 2, 5, 4]) * 15),
            arm + ": fitting confusion/class support differs")
    correct = int(table.trace())
    binary = int(table[0, 0] + table[1:, 1:].sum())
    derived = {"correct_final": correct, "correct_binary": binary, "accuracy": correct / 735,
               "binary_accuracy": binary / 735, "routing_errors": 735 - binary,
               "subtype_errors_after_correct_route": binary - correct,
               "class_recall": (table.diagonal() / table.sum(axis=1)).tolist()}
    for key, value in derived.items():
        require(np.asarray(value).shape == np.asarray(row[key]).shape
                and np.allclose(value, row[key], rtol=0, atol=METRIC_TOLERANCE),
                arm + ": fitting arithmetic mismatch for " + key)
    require(0 <= row["all_four_correct"] <= 49
            and row["all_four_accuracy"] == row["all_four_correct"] / 49
            and all(0 <= row[key] <= 1 for key in ("mean_subset_macro_f1", "worst_subset_accuracy", "worst_subset_macro_f1")),
            arm + ": fitting submetric range/arithmetic differs")


def paired_intervals(inputs, control, field):
    groups = np.asarray([inputs["audit"]["cases"][int(i)]["group"] for i in inputs["val"]])
    unique = sorted(set(groups.tolist()))
    rows = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(20261117)
    keys = ("accuracy", "binary_accuracy", "mean_subset_macro_f1")
    samples = np.empty((2000, len(keys)))
    for draw in range(len(samples)):
        index = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        left = metrics(inputs["truth"][index], control[index])
        right = metrics(inputs["truth"][index], field[index])
        samples[draw] = [right[key] - left[key] for key in keys]
    left, right = metrics(inputs["truth"], control), metrics(inputs["truth"], field)
    return {"interpretation": "descriptive selection-conditional paired conservative-group bootstrap; not fresh confirmation",
            "groups": len(rows), "draws": 2000, "seed": 20261117,
            "unit": "conservative group, retaining every case and all 15 masks",
            "macro_f1_labels": [0, 1, 2, 3, 4], "absent_class_f1": 0,
            "field_minus_control": {key: {"estimate": right[key] - left[key],
                "percentile_95_interval": np.quantile(samples[:, i], [.025, .975]).tolist()}
                for i, key in enumerate(keys)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "config", "audit", "views"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--field-cache", type=Path,
                        help="Optional pinned UNI2-h shard directory for exact per-field vector verification")
    args = parser.parse_args()
    config = read_json(args.config)
    root = read_json(args.run / "result.json")
    inputs = load_inputs(args, config, root)
    checked, predicted = {}, {}
    for arm in ARMS:
        checked[arm], predicted[arm] = check_arm(arm, args, config, root, inputs)
    goals = {arm: {"final_five_over_90": item["selected"]["accuracy"] > .9,
                   "binary_over_95": item["selected"]["binary_accuracy"] > .95,
                   "both_on_same_checkpoint": (item["selected"]["accuracy"] > .9
                                               and item["selected"]["binary_accuracy"] > .95)}
             for arm, item in checked.items()}
    field, control = checked["field"]["selected"], checked["control"]["selected"]
    expected_targets = {"five_above_90": goals["field"]["final_five_over_90"],
                        "binary_above_95": goals["field"]["binary_over_95"],
                        "both_at_selected_checkpoint": goals["field"]["both_on_same_checkpoint"]}
    improved = (field["correct_final"] > control["correct_final"]
                and field["correct_binary"] > control["correct_binary"]
                and field["mean_subset_macro_f1"] >= control["mean_subset_macro_f1"])
    require(root["targets"] == expected_targets and root["improvement_screen_passed"] == improved,
            "Joint target gates or improvement screen differ")
    for arm in ARMS:
        compare_metrics(checked[arm]["selected"], root["selected"][arm], "Root selected " + arm)
    receipt = {"status": "accepted independent validation-only field hierarchy replay",
               "result_sha256": digest(args.run / "result.json"), "config_sha256": digest(args.config),
               "code_sha": root["code_sha"], "source_archive_sha256": root["source_archive_sha256"],
               "input_audit_sha256": digest(args.audit), "case_views_sha256": digest(args.views),
               "validation_fields_sha256": root["validation_fields_sha256"],
               "verified_role": "validation", "verified_cases": 17, "verified_conservative_groups": 15,
               "verified_events": 255, "test_predictions": 0, "verified_original_field_counts": 1584,
               "field_mean_max_abs_error": inputs["field_mean_max_abs_error"],
               "exact_validation_field_identity_order_verified": True,
               "exact_frozen_field_cache_check": inputs["exact_field_cache_check"],
               "available_immutable_source_checks": inputs["source_checks"],
               "arms": checked, "independent_target_gates": goals,
               "improvement_screen_passed": improved,
               "paired_intervals": paired_intervals(inputs, predicted["control"], predicted["field"]),
               "limits": ["All validation cases have prior development exposure; intervals condition on selected checkpoints.",
                          "Private field identities/order and pinned view means are checked; optional --field-cache verifies every vector against original pinned shards. Images are not re-encoded.",
                          "All four checkpoint metrics are recalculated from saved logits; NumPy weight inference replays the selected checkpoint only.",
                          "TRAIN confusion arithmetic is checked; original TRAIN predictions are not supplied for independent inference."]}
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
