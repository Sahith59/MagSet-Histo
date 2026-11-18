"""Paired development experiment on the original case-first UNI2-h network."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from magset.shared_hierarchy import SharedHierarchy, final_labels, training_loss, zoom_masks
from scripts.check_field_hierarchy import fixed_roles, metrics, numpy_logits, predictions, require
from scripts.run_shared_hierarchy import digest


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("original", "subtype_initialized")


@torch.no_grad()
def initialize_subtypes(model, fitting_views, labels, masks):
    """Write four cosine similarity scores into the existing GELU network."""
    features = model.classifier_inputs(fitting_views, masks)
    dim = model.mean.shape[-1]
    unit_views = features[..., :dim] / math.sqrt(dim)
    class_means = torch.stack([unit_views[labels == label].mean(dim=(0, 1))
                               for label in range(1, 5)])
    require(bool(torch.isfinite(class_means).all() and (class_means.norm(dim=1) > 0).all()),
            "All four malignant classes need a finite nonzero training mean")
    directions = F.normalize(class_means, dim=1)
    first, last = model.classifier[0], model.classifier[-1]
    first.weight[:8].zero_()
    first.bias[:8].zero_()
    first.weight[:4, :dim].copy_(directions / math.sqrt(dim))
    first.weight[4:8, :dim].copy_(-first.weight[:4, :dim])
    last.weight[1:].zero_()
    last.bias[1:].zero_()
    identity = torch.eye(4, dtype=last.weight.dtype, device=last.weight.device)
    last.weight[1:, :4].copy_(identity)
    last.weight[1:, 4:8].copy_(-identity)
    # GELU(x) - GELU(-x) = x. Check the starting scores before dropout or fitting.
    model.eval()
    torch.testing.assert_close(model(fitting_views, masks)[..., 1:], unit_views @ directions.T,
                               atol=2e-6, rtol=2e-6)


def fit_and_verify(arm, x_train, y_train, x_eval, config, checkpoint):
    center = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scale = np.where(std < 1e-6, 1., std)
    torch.manual_seed(config["seed"])
    model = SharedHierarchy(torch.from_numpy(center), torch.from_numpy(scale),
                            config["hidden"], config["dropout"])
    masks = zoom_masks()
    inputs = torch.tensor(x_train, dtype=torch.float32)
    labels = torch.from_numpy(y_train)
    if arm == "subtype_initialized":
        initialize_subtypes(model, inputs, labels, masks)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    for _ in range(10):
        optimizer.zero_grad()
        loss = training_loss(model(inputs, masks), labels)
        require(bool(torch.isfinite(loss)), "Training loss became nonfinite")
        loss.backward()
        optimizer.step()
    model.eval()
    x_eval = x_eval.astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x_eval), masks)
        predicted = final_labels(logits).numpy()
    torch.save(model.state_dict(), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key: value.numpy() for key, value in saved.items()}
    replay = numpy_logits("control", state, {"views": x_eval, "masks": masks.numpy()})
    error = float(np.abs(logits.numpy() - replay).max())
    require(error <= 1.e-4 and np.array_equal(predictions(replay), predicted),
            "Independent replay of saved model differs")
    return predicted, error


def evaluate(role, views, labels, groups, train, validation, case_indices, config, output):
    rows = train if role == "group_check" else validation
    folds = [(train, validation)]
    if role == "group_check":
        folds = [(train[groups[train] != group], train[groups[train] == group])
                 for group in np.unique(groups[train])]
    all_predictions = {arm: np.full((len(views), 15), -1, dtype=int) for arm in ARMS}
    max_error = 0.
    for fold, (fitting, held) in enumerate(folds):
        require(not set(groups[fitting]) & set(groups[held]), "A held-out group enters fitting")
        for arm in ARMS:
            checkpoint = output / f"{role}-{fold:02d}-{arm}.pt"
            preds, error = fit_and_verify(arm, views[fitting], labels[fitting], views[held],
                                          config, checkpoint)
            require((all_predictions[arm][held] == -1).all(), "Duplicate held-out rows")
            all_predictions[arm][held] = preds
            max_error = max(max_error, error)
    saved = output / f"{role}.npz"
    np.savez_compressed(saved, case_indices=case_indices[rows], truth=labels[rows], masks=zoom_masks().numpy(),
                        **{arm: values[rows] for arm, values in all_predictions.items()})
    with np.load(saved, allow_pickle=False) as archive:
        result = {arm: metrics(archive["truth"], archive[arm]) for arm in ARMS}
    return {**result, "folds": len(folds), "max_logit_replay_error": max_error,
            "archive_sha256": digest(saved)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / ".source-sha").read_text().strip()
    require(source == os.environ["MAGSET_CODE_SHA"], "Immutable source revision differs")
    config_path = ROOT / "configs/breakhis-shared-hierarchy-v1.json"
    config = json.loads(config_path.read_text())
    require(digest(args.audit) == config["audit_sha256"] and
            digest(args.views) == config["case_views_sha256"], "Dataset commitment differs")
    audit = json.loads(args.audit.read_text())
    train, validation = fixed_roles(audit)
    require(audit["manifest_sha256"] == config["manifest_sha256"], "Manifest differs")
    development = np.r_[train, validation]
    with np.load(args.views, allow_pickle=False) as archive:
        require(np.array_equal(archive["case_indices"], np.arange(82)) and
                np.array_equal(archive["train_case_indices"], train) and
                np.array_equal(archive["masks"], zoom_masks().numpy()), "Cache ordering differs")
        views = archive["case_view_means"][development]
    labels = np.array([audit["cases"][i]["truth5"] for i in development])
    groups = np.array([audit["cases"][i]["group"] for i in development])
    # Work in development-local ordinals throughout. No follow-up descriptors are passed to a fit.
    fit_rows = np.arange(len(train))
    val_rows = np.arange(len(train), len(development))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"code_sha": source, "source_archive_sha256": (ROOT / ".archive-sha256").read_text().strip(),
              "slurm_job": os.environ["SLURM_JOB_ID"], "torch_version": str(torch.__version__),
              "reference_config_sha256": digest(config_path),
              **{k: config[k] for k in ("audit_sha256", "case_views_sha256", "manifest_sha256",
                                        "reader_revision", "weights_sha256", "seed")},
              "updates": 10, "followup_predictions": 0}
    for role in ("group_check", "validation"):
        report[role] = evaluate(role, views, labels, groups, fit_rows, val_rows, development, config, args.output)
        print(role, {arm: report[role][arm]["correct_final"] for arm in ARMS}, flush=True)
    prior_dir = ROOT / "docs/evidence"
    references = {
        "group_check": json.loads((prior_dir / "FIELD_HIERARCHY_GROUP_CHECK.json").read_text())["arms"]["control"],
        "validation": json.loads((prior_dir / "FIELD_HIERARCHY_CONTROL.json").read_text())["selected"],
    }
    passed = []
    for role, reference in references.items():
        baseline = report[role]["original"]
        candidate = report[role]["subtype_initialized"]
        require(baseline["aggregate_confusion_true_rows"] == reference["aggregate_confusion_true_rows"] and
                abs(baseline["mean_subset_macro_f1"] - reference["mean_subset_macro_f1"]) < 1e-10,
                "Re-fitted control disagrees with archived results on " + role)
        passed.append(candidate["correct_final"] > baseline["correct_final"] and
                      candidate["correct_binary"] >= baseline["correct_binary"] and
                      candidate["mean_subset_macro_f1"] >= baseline["mean_subset_macro_f1"])
    report["development_screen_passed"] = all(passed)
    (args.output / "aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
