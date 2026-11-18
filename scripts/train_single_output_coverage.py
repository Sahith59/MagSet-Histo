"""Fit one five-category linear softmax classifier on frozen center-view descriptors."""
import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from magset.shared_hierarchy import zoom_masks
from scripts.run_shared_hierarchy import digest, load_development, summarize


class SingleOutputClassifier(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        if (mean.ndim != 2 or mean.shape[0] != 4 or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or not (scale > 0).all()):
            raise ValueError("Expected four finite training normalizers with positive scales")
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float())
        self.classifier = nn.Linear(mean.shape[-1] + 4, 5)

    def forward(self, views, masks):
        if views.ndim != 3 or views.shape[1:] != self.mean.shape:
            raise ValueError("Expected one descriptor per canonical zoom")
        if (masks.ndim != 2 or masks.dtype != torch.bool or masks.shape[1] != 4
                or len(masks) == 0 or not masks.any(dim=1).all()):
            raise ValueError("Each mask must select at least one canonical zoom")
        standardized = (views - self.mean) / self.scale
        present = torch.where(masks[None, :, :, None], standardized[:, None], 0.)
        pooled = present.sum(dim=2) / masks.sum(dim=1)[:, None]
        if not torch.isfinite(pooled).all():
            raise ValueError("A selected view has nonfinite features")
        pooled = F.normalize(pooled, dim=-1) * math.sqrt(self.mean.shape[-1])
        flags = masks.to(pooled.dtype).expand(len(views), -1, -1)
        return self.classifier(torch.cat((pooled, flags), dim=-1))


def final_labels(logits):
    if logits.shape[-1] != 5 or not torch.isfinite(logits).all():
        raise ValueError("Expected five finite class logits")
    malignant = logits.softmax(dim=-1)[..., 1:].sum(dim=-1) >= 0.5
    return torch.where(malignant, 1 + logits[..., 1:].argmax(dim=-1), 0)


def fit(config, data, output):
    mean, scale, train, validation, y_train, y_val, val_indices = data
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    model = SingleOutputClassifier(mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    masks = zoom_masks()
    targets = y_train[:, None].expand(-1, len(masks)).reshape(-1)
    counts = torch.bincount(y_train, minlength=5)
    if len(counts) != 5 or not (counts > 0).all():
        raise ValueError("All five classes are required in training")
    weights = counts.float().reciprocal()
    history, epoch_logits, losses = [], [], []
    best_f1 = -1.
    for epoch in range(1, config["epochs"] + 1):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(train, masks).reshape(-1, 5), targets, weight=weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch in config["validation_epochs"]:
            with torch.no_grad():
                scores = model(validation, masks)
            metrics = summarize(y_val.numpy(), final_labels(scores).numpy())
            history.append({"epoch": epoch, **metrics})
            epoch_logits.append(scores.numpy())
            if metrics["mean_subset_macro_f1"] > best_f1:
                best_f1 = metrics["mean_subset_macro_f1"]
                best = {"epoch": epoch, "config": config, "model": copy.deepcopy(model.state_dict())}
                best_scores = scores
    checkpoint = output / "selected.pt"
    torch.save(best, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    replay = SingleOutputClassifier(mean, scale).eval()
    replay.load_state_dict(saved["model"])
    with torch.no_grad():
        logits = replay(validation, masks)
        if not torch.equal(best_scores, logits):
            raise ValueError("Selected checkpoint reload changed validation logits")
        predictions = final_labels(logits).numpy()
        fitting = summarize(y_train.numpy(), final_labels(replay(train, masks)).numpy())
    archive = output / "validation.npz"
    np.savez_compressed(archive, case_indices=val_indices, truth=y_val.numpy(), masks=masks.numpy(),
                        logits=logits.numpy(), predictions=predictions, epoch_logits=np.stack(epoch_logits),
                        validation_epochs=np.array(config["validation_epochs"]))
    return {
        "history": history, "selected_epoch": saved["epoch"], "selected": summarize(y_val.numpy(), predictions),
        "trainable_parameters": sum(p.numel() for p in model.parameters()), "reload_max_absolute_error": 0.,
        "checkpoint_sha256": digest(checkpoint), "validation_sha256": digest(archive),
        "fitting_loss_before_each_update": losses, "selected_fitting_diagnostic": fitting,
        "train_class_counts": counts.tolist(), "validation_class_counts": torch.bincount(y_val, minlength=5).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "audit", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (root / ".source-sha").read_text().strip() != args.code_sha:
        raise ValueError("Source archive revision differs")
    if digest(args.config) != digest(root / "configs/breakhis-single-output-center-v1.json"):
        raise ValueError("Configuration differs from the registered center-view study")
    config = json.loads(args.config.read_text())
    data = load_development(args.audit, args.views, config)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "study": config["study"], "code_sha": args.code_sha, "config_sha256": digest(args.config),
        "inputs": {k: config[k] for k in ("manifest_sha256", "audit_sha256", "case_views_sha256",
                                          "reader", "reader_revision", "weights_sha256")},
        "role": "validation", "train_cases": len(data[4]), "test_predictions": 0,
        "source_archive_sha256": (root / ".archive-sha256").read_text().strip(),
        "slurm_job": os.getenv("SLURM_JOB_ID"), "torch": str(torch.__version__),
        **fit(config, data, args.output),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"epoch": result["selected_epoch"], **result["selected"]}))


if __name__ == "__main__":
    main()
