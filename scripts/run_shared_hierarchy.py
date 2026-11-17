"""Fit and score the registered shared model on the matched development roles."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from magset.shared_hierarchy import SharedHierarchy, final_labels, training_loss, zoom_masks


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(audit_path, views_path, config):
    if digest(audit_path) != config["audit_sha256"] or digest(views_path) != config["case_views_sha256"]:
        raise ValueError("Input audit or descriptor file differs from registration")
    audit = json.loads(audit_path.read_text())
    if audit["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Unexpected image manifest")
    train, val = (np.asarray(audit["roles"][role]) for role in ("train", "validation"))
    groups = [{audit["cases"][i]["group"] for i in indices} for indices in (train, val)]
    if (len(train), len(val), len(groups[0]), len(groups[1])) != (49, 17, 42, 15) or groups[0] & groups[1]:
        raise ValueError("Development role sizes or group separation differ")
    with np.load(views_path, allow_pickle=False) as cache:
        if not np.array_equal(train, cache["train_case_indices"]) or not np.array_equal(cache["case_indices"], np.arange(82)):
            raise ValueError("Descriptor case order or training role differs")
        if not np.array_equal(cache["masks"], zoom_masks().numpy()):
            raise ValueError("Zoom mask order differs")
        views = cache["case_view_means"]
        x_train, x_val = views[train], views[val]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        scale = np.where(std < 1.e-6, 1., std)
        if not np.array_equal(mean, cache["mean"]) or not np.array_equal(scale, cache["scale"]):
            raise ValueError("Training-only normalizer differs from source")
    y_train, y_val = (torch.tensor([audit["cases"][i]["truth5"] for i in role]) for role in (train, val))
    tensors = [torch.tensor(x, dtype=torch.float32) for x in (mean, scale, x_train, x_val)]
    return (*tensors, y_train, y_val, val)


def summarize(truth, predictions):
    binary_correct = (predictions > 0) == (truth[:, None] > 0)
    correct = predictions == truth[:, None]
    confusions = [confusion_matrix(truth, col, labels=range(5)) for col in predictions.T]
    total = np.sum(confusions, axis=0)
    return {
        "cases": len(truth), "events": int(correct.size), "correct_final": int(correct.sum()),
        "correct_binary": int(binary_correct.sum()), "accuracy": float(correct.mean()),
        "binary_accuracy": float(binary_correct.mean()),
        "mean_subset_macro_f1": float(np.mean([f1_score(truth, col, labels=range(5), average="macro",
                                                       zero_division=0) for col in predictions.T])),
        "all_four_correct": int(correct[:, -1].sum()),
        "all_four_accuracy": float(correct[:, -1].mean()),
        "class_recall": (total.diagonal() / total.sum(axis=1)).tolist(),
        "aggregate_confusion_true_rows": total.tolist(),
        "routing_errors": int((~binary_correct).sum()),
        "subtype_errors_after_correct_route": int((binary_correct & ~correct).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "audit", "views", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (root / ".source-sha").read_text().strip() != args.code_sha:
        raise ValueError("Run from the immutable source archive for this revision")
    if digest(args.config) != digest(root / "configs/breakhis-shared-hierarchy-v1.json"):
        raise ValueError("Only the registered configuration can be used")
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    mean, scale, x_train, x_val, y_train, y_val, val_indices = load_development(args.audit, args.views, config)
    masks = zoom_masks()
    model_args = (mean, scale, config["hidden"], config["dropout"])
    model = SharedHierarchy(*model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history = []
    validation_scores = []
    fitting_loss = []
    best_f1 = -1.
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = training_loss(model(x_train, masks), y_train)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        fitting_loss.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        if epoch in config["validation_epochs"]:
            model.eval()
            with torch.no_grad():
                scores = model(x_val, masks)
                metrics = summarize(y_val.numpy(), final_labels(scores).numpy())
            history.append({"epoch": epoch, **metrics})
            validation_scores.append(scores.numpy())
            if metrics["mean_subset_macro_f1"] > best_f1:
                best_f1 = metrics["mean_subset_macro_f1"]
                selected = {"epoch": epoch, "model": copy.deepcopy(model.state_dict()), "config": config}
                selected_scores = scores.clone()
    checkpoint = args.output / "selected.pt"
    torch.save(selected, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    replay = SharedHierarchy(*model_args).eval()
    replay.load_state_dict(saved["model"])
    with torch.no_grad():
        logits = replay(x_val, masks)
        if not torch.equal(logits, selected_scores):
            raise ValueError("Reload changed validation scores")
        predictions = final_labels(logits).numpy()
        fitting = summarize(y_train.numpy(), final_labels(replay(x_train, masks)).numpy())
    archive = args.output / "validation.npz"
    np.savez_compressed(archive, case_indices=val_indices, truth=y_val.numpy(), masks=masks.numpy(),
                        logits=logits.numpy(), predictions=predictions,
                        validation_epochs=np.array(config["validation_epochs"]),
                        epoch_logits=np.stack(validation_scores))
    report = {
        "study": config["study"], "code_sha": args.code_sha, "config_sha256": digest(args.config),
        "inputs": {k: config[k] for k in ("manifest_sha256", "audit_sha256", "case_views_sha256",
                                         "reader", "reader_revision", "weights_sha256")},
        "role": "validation", "train_cases": len(y_train), "train_class_counts": torch.bincount(y_train).tolist(),
        "validation_class_counts": torch.bincount(y_val).tolist(), "test_predictions": 0,
        "history": history, "selected_epoch": saved["epoch"], "selected": summarize(y_val.numpy(), predictions),
        "trainable_parameters": sum(p.numel() for p in model.parameters()), "reload_max_absolute_error": 0.,
        "checkpoint_sha256": digest(checkpoint), "validation_sha256": digest(archive),
        "source_archive_sha256": (root / ".archive-sha256").read_text().strip(),
        "fitting_loss_before_each_update": fitting_loss, "selected_fitting_diagnostic": fitting,
        "slurm_job": os.getenv("SLURM_JOB_ID"), "torch": str(torch.__version__),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"epoch": saved["epoch"], **report["selected"]}))


if __name__ == "__main__":
    main()
