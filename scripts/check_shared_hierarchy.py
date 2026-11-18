"""Rebuild selected validation inference and all checkpoint scores from saved evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise ValueError(description)


def metrics(truth, scores):
    check(np.isfinite(scores).all(), "Nonfinite saved logits")
    preds = np.where(scores[..., 0] < 0, 0, scores[..., 1:].argmax(axis=-1) + 1)
    tables = np.stack([np.bincount(5 * truth + column, minlength=25).reshape(5, 5) for column in preds.T])
    denom = tables.sum(axis=1) + tables.sum(axis=2)
    numer = 2 * tables.diagonal(axis1=1, axis2=2)
    f1 = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)
    total = tables.sum(axis=0)
    binary_correct = int(total[0, 0] + total[1:, 1:].sum())
    return preds, {
        "cases": len(truth), "events": preds.size,
        "correct_final": int(total.trace()), "correct_binary": binary_correct,
        "accuracy": total.trace() / preds.size, "binary_accuracy": binary_correct / preds.size,
        "all_four_accuracy": tables[-1].trace() / len(truth),
        "routing_errors": preds.size - binary_correct, "subtype_errors_after_correct_route": binary_correct - total.trace(),
        "mean_subset_macro_f1": float(f1.mean()), "all_four_correct": int(tables[-1].trace()),
        "class_recall": total.diagonal() / total.sum(axis=1), "aggregate_confusion_true_rows": total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "config", "views", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    config = json.loads(args.config.read_text())
    check(digest(args.config) == result["config_sha256"] and digest(args.views) == config["case_views_sha256"]
          and digest(args.audit) == config["audit_sha256"], "Input hash mismatch")
    for name, key in (("selected.pt", "checkpoint_sha256"), ("validation.npz", "validation_sha256")):
        check(digest(args.run / name) == result[key], "Changed artifact " + name)
    audit = json.loads(args.audit.read_text())
    train, val = audit["roles"]["train"], audit["roles"]["validation"]
    truth = np.array([audit["cases"][i]["truth5"] for i in val])
    checkpoint = torch.load(args.run / "selected.pt", map_location="cpu", weights_only=True)
    state = {k: v.numpy() for k, v in checkpoint["model"].items()}
    with np.load(args.views) as source, np.load(args.run / "validation.npz") as archive:
        check(np.array_equal(archive["case_indices"], val) and np.array_equal(truth, archive["truth"]),
              "Validation rows do not match the original role")
        check(np.array_equal(archive["masks"], source["masks"]), "Mask order mismatch")
        case_views = source["case_view_means"]
        mean, std = case_views[train].mean(axis=0), case_views[train].std(axis=0)
        scale = np.where(std < 1.e-6, 1., std)
        check(np.array_equal(mean.astype("f4"), state["mean"]) and np.array_equal(scale.astype("f4"), state["scale"]),
              "Selected state has a different training normalizer")
        standardized = (case_views[val].astype("f4") - state["mean"]) / state["scale"]
        rows = []
        for mask in archive["masks"]:
            view = standardized[:, mask, :].mean(axis=1)
            norm = np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1.e-12)
            pooled = view / norm * np.sqrt(np.float32(view.shape[-1]))
            features = np.column_stack((pooled, np.broadcast_to(mask, (len(val), 4))))
            hidden = np.einsum("bi,hi->bh", features, state["classifier.0.weight"], dtype=np.float64) + state["classifier.0.bias"]
            rows.append(np.einsum("bh,oh->bo", hidden * ndtr(hidden), state["classifier.3.weight"]) + state["classifier.3.bias"])
        logits = np.stack(rows, axis=1)
        error = float(abs(logits - archive["logits"]).max())
        check(error < 1.e-4, "Independent selected logits differ")
        preds, selected = metrics(truth, logits)
        check(np.array_equal(preds, archive["predictions"]), "Independent final decisions differ")
        epochs = archive["validation_epochs"].tolist()
        check(epochs == config["validation_epochs"] == [v["epoch"] for v in result["history"]], "Epoch list differs")
        check(len(archive["epoch_logits"]) == len(epochs), "Missing epoch logits")
        scores = []
        for row, raw in zip(result["history"], archive["epoch_logits"]):
            _, rebuilt = metrics(truth, raw)
            for k, v in rebuilt.items():
                check(np.allclose(v, row[k], rtol=0, atol=1.e-12), f"Epoch {row['epoch']} mismatch: {k}")
            scores.append(rebuilt["mean_subset_macro_f1"])
        best = int(np.argmax(scores))
        check(epochs[best] == checkpoint["epoch"] == result["selected_epoch"], "First maximum differs")
        check(np.array_equal(archive["epoch_logits"][best], archive["logits"]), "Different selected epoch logits")
    for k, v in selected.items():
        check(np.allclose(v, result["selected"][k], rtol=0, atol=1.e-12), "Selected score mismatch: " + k)
    print(json.dumps({"result_sha256": digest(args.run / "result.json"), "verified_role": "validation",
                      "verified_epochs": epochs, "selected_epoch": epochs[best], "verified_cases": len(val),
                      "verified_final_correct": selected["correct_final"], "verified_binary_correct": selected["correct_binary"],
                      "all_saved_decisions_match": True, "numpy_checkpoint_max_abs_logit_error": error}, indent=2))


if __name__ == "__main__":
    main()
