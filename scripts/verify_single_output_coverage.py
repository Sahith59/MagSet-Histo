#!/usr/bin/env python3
"""Independently reconstruct the selected five-class linear model on fixed development roles."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from check_field_hierarchy import compare_metrics, digest, fixed_roles, metrics, require, zoom_masks


def decide(scores):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape[-2:] == (15, 5) and np.isfinite(scores).all(), "Invalid five-class scores")
    odds = np.logaddexp.reduce(scores[..., 1:], axis=-1) - scores[..., 0]
    return np.where(odds >= 0, scores[..., 1:].argmax(axis=-1) + 1, 0)


def reconstruct(views, mean, scale, masks, weights, bias):
    features = (views.astype(np.float64) - mean) / scale
    scores = []
    for mask in masks:
        pooled = features[:, mask].mean(axis=1)
        norm = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1.e-12)
        pooled = pooled / norm * np.sqrt(mean.shape[-1])
        flags = np.broadcast_to(mask, (len(views), 4))
        scores.append(np.column_stack((pooled, flags)) @ weights.T + bias)
    return np.stack(scores, axis=1)


def verify(args):
    config = json.loads(args.config.read_text())
    report = json.loads((args.run / "result.json").read_text())
    registered = Path(__file__).resolve().parents[1] / "configs/breakhis-single-output-center-v1.json"
    require(digest(registered) == digest(args.config) == report["config_sha256"], "Config differs")
    for path, key in ((args.audit, "audit_sha256"), (args.views, "case_views_sha256")):
        require(digest(path) == config[key], "Input digest differs: " + key)
    checkpoint = args.run / "selected.pt"
    archive = args.run / "validation.npz"
    require(digest(checkpoint) == report["checkpoint_sha256"] and
            digest(archive) == report["validation_sha256"], "Run artifact hash differs")
    audit = json.loads(args.audit.read_text())
    train, val = fixed_roles(audit)
    require(report["role"] == "validation" and report["test_predictions"] == 0, "Unexpected role")
    masks = zoom_masks()
    with np.load(args.views, allow_pickle=False) as saved:
        require(np.array_equal(saved["case_indices"], np.arange(82)) and
                np.array_equal(saved["train_case_indices"], train) and np.array_equal(saved["masks"], masks),
                "Feature case or mask ordering differs")
        views = saved["case_view_means"]
        require(views.shape == (82, 4, 1536), "Unexpected encoder feature shape")
        train_views, val_views = views[train], views[val]
        mean, std = train_views.mean(axis=0), train_views.std(axis=0)
        scale = np.where(std < 1.e-6, 1., std)
        require(np.array_equal(mean, saved["mean"]) and np.array_equal(scale, saved["scale"]),
                "Frozen normalizers differ from TRAIN statistics")
    selected = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state = {key: value.numpy() for key, value in selected["model"].items()}
    require(set(state) == {"classifier.weight", "classifier.bias", "mean", "scale"} and
            np.array_equal(state["mean"], mean) and np.array_equal(state["scale"], scale) and
            selected["config"] == config, "Unexpected model state")
    weights, bias = state["classifier.weight"], state["classifier.bias"]
    require(weights.shape == (5, 1540) and bias.shape == (5,) and report["trainable_parameters"] == 7705,
            "Expected one five-output linear classifier")
    labels = np.array([row["truth5"] for row in audit["cases"]])
    epochs = config["validation_epochs"]
    require(epochs == [10, 20, 30, 40] and len(report["history"]) == len(epochs), "Unexpected history")
    with np.load(archive, allow_pickle=False) as saved:
        require(np.array_equal(saved["case_indices"], val) and np.array_equal(saved["truth"], labels[val])
                and np.array_equal(saved["masks"], masks) and np.array_equal(saved["validation_epochs"], epochs)
                and saved["epoch_logits"].shape == (4, 17, 15, 5), "Invalid validation event coverage")
        history = []
        for epoch, logits, stated in zip(epochs, saved["epoch_logits"], report["history"]):
            require(stated["epoch"] == epoch, "Checkpoint ordering differs")
            actual = metrics(labels[val], decide(logits))
            compare_metrics(actual, stated, str(epoch), allow_missing_worst=True)
            history.append(actual["mean_subset_macro_f1"])
        best = int(np.argmax(history))
        require(epochs[best] == selected["epoch"] == report["selected_epoch"]
                and np.array_equal(saved["epoch_logits"][best], saved["logits"]), "Selection differs")
        replay = reconstruct(val_views, mean, scale, masks, weights, bias)
        error = float(np.abs(replay - saved["logits"]).max())
        predicted = decide(replay)
        require(error <= 1.e-5 and np.array_equal(predicted, saved["predictions"])
                and np.array_equal(predicted, decide(saved["logits"])), "Selected validation replay differs")
    validation = metrics(labels[val], predicted)
    compare_metrics(validation, report["selected"], "validation", allow_missing_worst=True)
    fitting = metrics(labels[train], decide(reconstruct(train_views, mean, scale, masks, weights, bias)))
    compare_metrics(fitting, report["selected_fitting_diagnostic"], "training", allow_missing_worst=True)
    return {"accepted": True, "selected_epoch": epochs[best], "max_absolute_logit_error": error,
            "validation": validation, "selected_fitting_diagnostic": fitting, "test_predictions": 0,
            "joint_targets_met": validation["correct_final"] >= 230 and validation["correct_binary"] >= 243,
            "checkpoint_sha256": digest(checkpoint), "validation_sha256": digest(archive),
            "result_sha256": digest(args.run / "result.json"), "config_sha256": digest(args.config),
            "verifier_sha256": digest(Path(__file__)), "code_sha": report["code_sha"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "audit", "views", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
