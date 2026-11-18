"""Fixed-update, TRAIN-only conservative-group cross-fitting of the two arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from magset.field_hierarchy import FieldHierarchy, PackedFields
from magset.shared_hierarchy import SharedHierarchy, final_labels, training_loss, zoom_masks
from scripts.run_shared_hierarchy import digest


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "field")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, expected):
    require(digest(path) == expected, f"Checksum differs for {path.name}")
    return json.loads(path.read_text())


def state_digest(state):
    value = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().numpy()
        value.update(name.encode())
        value.update(str(array.shape).encode())
        value.update(str(array.dtype).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def load_inputs(args):
    """Verify an outcome-free, TRAIN-only bundle before any optimization."""
    require((ROOT / ".source-sha").read_text().strip() == args.code_sha,
            "Run from the requested immutable source snapshot")
    config_sha = digest(ROOT / "configs/breakhis-field-hierarchy-v1.json")
    config = pinned_json(args.config, config_sha)
    plan_sha = digest(ROOT / "configs/breakhis-field-crossfit-v1.json")
    plan = pinned_json(args.plan, plan_sha)
    fixed = {"study": "breakhis-field-crossfit-v1", "primary_config_sha256": config_sha,
             "audit_sha256": config["audit_sha256"], "folds": 42, "shards": 6,
             "updates": 10, "seed": 812, "arms": list(ARMS),
             "role": "train-only leave-one-conservative-group-out"}
    for key, expected in fixed.items():
        require(plan.get(key) == expected, f"Cross-fitting plan differs at {key}")
    require(config["seed"] == 812 and config["hidden"] == 32 and config["dropout"] == .2
            and config["learning_rate"] == .001 and config["weight_decay"] == .01,
            "Primary projection or optimizer recipe differs")
    require(args.input_receipt.resolve().parent == args.train_fields.resolve().parent,
            "Input receipt must accompany the TRAIN bundle")
    receipt = json.loads(args.input_receipt.read_text())
    train_sha = digest(args.train_fields)
    expected_receipt = {"train_fields_sha256": train_sha, "config_sha256": config_sha,
                        "audit_sha256": config["audit_sha256"], "code_sha": args.code_sha,
                        "case_views_sha256": config["case_views_sha256"]}
    for key, expected in expected_receipt.items():
        require(receipt.get(key) == expected, f"Input receipt differs at {key}")
    audit = pinned_json(args.audit, config["audit_sha256"])
    require(audit["manifest_sha256"] == config["manifest_sha256"], "Manifest differs")
    train_indices = np.asarray(audit["roles"]["train"], dtype=np.int64)
    require(train_indices.shape == (49,) and len(np.unique(train_indices)) == 49,
            "Expected the fixed 49 unique TRAIN cases")
    with np.load(args.train_fields, allow_pickle=False) as source:
        required = {"fields", "case", "zoom", "views", "case_indices", "path_sha256"}
        require(set(source.files) == required, "TRAIN bundle members differ")
        fields, case, zoom, views, case_indices, paths = (source[key] for key in
                ("fields", "case", "zoom", "views", "case_indices", "path_sha256"))
    require(case_indices.dtype == np.int64 and np.array_equal(case_indices, train_indices),
            "TRAIN bundle case order differs from the audit")
    require(fields.dtype == np.float32 and fields.shape == (4621, 1536)
            and np.isfinite(fields).all(), "Expected all 4,621 finite UNI2-h TRAIN fields")
    require(case.dtype == np.int64 and zoom.dtype == np.int64
            and case.shape == zoom.shape == (len(fields),)
            and ((case >= 0) & (case < 49)).all() and ((zoom >= 0) & (zoom < 4)).all(),
            "Invalid packed field ownership")
    require(views.dtype == np.float32 and views.shape == (49, 4, 1536)
            and np.isfinite(views).all(), "Invalid packed TRAIN case/view means")
    require(paths.shape == (len(fields),) and paths.dtype.kind in "SU",
            "Expected a SHA256 identity for every TRAIN field")
    path_text = [p.decode("ascii") if isinstance(p, bytes) else str(p) for p in paths]
    require(len(set(path_text)) == len(fields)
            and all(len(p) == 64 and all(c in "0123456789abcdef" for c in p) for p in path_text),
            "Packed field hashes are invalid or duplicated")
    raw_means = np.empty((49, 4, 1536), dtype=np.float64)
    audited_fields = {}
    for field in audit["fields"]:
        if field["case_index"] in set(train_indices.tolist()):
            key = (int(field["case_index"]), int(field["view"]))
            audited_fields.setdefault(key, []).append(hashlib.sha256(field["relative_path"].encode()).hexdigest())
    for local, index in enumerate(train_indices):
        for view in range(4):
            selected = np.flatnonzero((case == local) & (zoom == view))
            require(len(selected) > 0, "Every TRAIN case requires all four zooms")
            require(sorted(path_text[i] for i in selected) == sorted(audited_fields.get((int(index), view), [])),
                    "Packed fields do not match their audited TRAIN case/view")
            raw_means[local, view] = fields[selected].mean(axis=0, dtype=np.float64)
    require(np.array_equal(raw_means.astype(np.float32), views),
            "Stored case means differ from the raw-field FP64 arithmetic")
    truth = np.asarray([audit["cases"][i]["truth5"] for i in train_indices], dtype=np.int64)
    require(np.array_equal(np.bincount(truth, minlength=5), [14, 24, 2, 5, 4]),
            "TRAIN class support differs")
    groups = [audit["cases"][i]["group"] for i in train_indices]
    require(all(isinstance(g, str) and g for g in groups), "Non-string conservative group")
    group_names = sorted(set(groups))
    require(len(group_names) == 42, "Expected 42 conservative TRAIN groups")
    for group in group_names:
        fitting = np.flatnonzero(np.asarray(groups) != group)
        require((np.bincount(truth[fitting], minlength=5) > 0).all(),
                "A leave-group-out fitting role loses a final class")
    pins = {"code_sha": args.code_sha,
            "source_archive_sha256": (ROOT / ".archive-sha256").read_text().strip(),
            "config_sha256": config_sha, "plan_sha256": plan_sha,
            "audit_sha256": config["audit_sha256"], "train_fields_sha256": train_sha,
            "input_receipt_sha256": digest(args.input_receipt),
            "case_views_sha256": config["case_views_sha256"]}
    batch = PackedFields(torch.from_numpy(fields), torch.from_numpy(case),
                         torch.from_numpy(zoom), torch.from_numpy(views))
    return config, plan, pins, batch, train_indices, truth, groups, group_names, raw_means


def subset_batch(batch, indices):
    """Reindex only the requested TRAIN cases without retaining excluded fields."""
    indices = np.asarray(indices, dtype=np.int64)
    require(indices.ndim == 1 and len(indices) > 0 and len(np.unique(indices)) == len(indices)
            and ((indices >= 0) & (indices < len(batch.views))).all(),
            "Unique valid local case ordinals required")
    remap = torch.full((len(batch.views),), -1, dtype=torch.long)
    remap[torch.from_numpy(indices)] = torch.arange(len(indices))
    selected = remap[batch.case] >= 0
    return PackedFields(batch.fields[selected], remap[batch.case[selected]], batch.zoom[selected],
                        batch.views[torch.from_numpy(indices)])


def fit_arm(config, pins, fold_index, held_group, arm, mean, scale, fitting, held,
            fitting_ordinals, held_ordinals, labels, output):
    """Exactly ten full-batch updates; held cases never control fitting or selection."""
    torch.manual_seed(config["seed"])
    network = SharedHierarchy if arm == "control" else FieldHierarchy
    def build():
        return network(torch.from_numpy(mean), torch.from_numpy(scale), config["hidden"], config["dropout"])
    model = build()
    initial_sha = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    masks = zoom_masks()
    train_input = fitting.views if arm == "control" else fitting
    held_input = held.views if arm == "control" else held
    losses = []
    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        loss = training_loss(model(train_input, masks), labels)
        require(bool(torch.isfinite(loss)), "Nonfinite TRAIN-only fitting loss")
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(held_input, masks).clone()
    checkpoint = output / f"fold-{fold_index:02d}-{arm}.pt"
    torch.save({"epoch": 10, "model": model.state_dict(), "config": config,
                "fold_index": fold_index, "arm": arm, "held_group": held_group,
                "fit_local_ordinals": fitting_ordinals.tolist(),
                "held_local_ordinals": held_ordinals.tolist(), "pins": pins}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    replay = build().eval()
    replay.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        reloaded = replay(held_input, masks)
    require(torch.equal(logits, reloaded), "Held-out checkpoint replay changed logits")
    return logits.numpy(), {"checkpoint_sha256": digest(checkpoint),
                            "initial_state_sha256": initial_sha, "training_losses": losses,
                            "trainable_parameters": sum(p.numel() for p in model.parameters()),
                            "reload_max_absolute_error": 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-fields", "input-receipt", "audit", "config", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--shard", type=int, choices=range(6), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config, plan, pins, batch, case_indices, truth, groups, group_names, raw_means = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    fold_hashes, held_count = {}, 0
    fold_indices = [i for i in range(len(group_names)) if i % 6 == args.shard]
    for fold_index in fold_indices:
        group = group_names[fold_index]
        held = np.flatnonzero(np.asarray(groups) == group).astype(np.int64)
        fitting = np.flatnonzero(np.asarray(groups) != group).astype(np.int64)
        require(set(fitting).isdisjoint(held) and len(fitting) + len(held) == 49,
                "Fitting and held cases do not partition TRAIN")
        # Recompute moments from original float32 fields, not rounded stored means.
        mean = raw_means[fitting].mean(axis=0)
        std = raw_means[fitting].std(axis=0, ddof=0)
        scale = np.where(std < 1.e-6, 1., std)
        train_batch, held_batch = subset_batch(batch, fitting), subset_batch(batch, held)
        labels = torch.from_numpy(truth[fitting])
        scores, arms = {}, {}
        for arm in ARMS:
            scores[arm], arms[arm] = fit_arm(config, pins, fold_index, group, arm, mean, scale,
                    train_batch, held_batch, fitting, held, labels, args.output)
        require(arms["control"]["initial_state_sha256"] == arms["field"]["initial_state_sha256"],
                "Paired arms do not start from identical parameters")
        require(arms["control"]["trainable_parameters"] == arms["field"]["trainable_parameters"] == 49477,
                "Paired arm parameter counts differ")
        archive = args.output / f"fold-{fold_index:02d}.npz"
        np.savez_compressed(archive, held_local_ordinals=held, fit_local_ordinals=fitting,
                            held_case_indices=case_indices[held], fit_case_indices=case_indices[fitting],
                            masks=zoom_masks().numpy(), truth=truth[held], mean=mean, scale=scale,
                            control_logits=scores["control"], field_logits=scores["field"],
                            control_predictions=final_labels(torch.from_numpy(scores["control"])).numpy(),
                            field_predictions=final_labels(torch.from_numpy(scores["field"])).numpy())
        fold = {"study": plan["study"], "role": plan["role"], "pins": pins,
                "fold_index": fold_index, "held_group": group,
                "held_local_ordinals": held.tolist(), "fit_local_ordinals": fitting.tolist(),
                "held_case_indices": case_indices[held].tolist(), "fit_case_indices": case_indices[fitting].tolist(),
                "epoch": 10, "archive_sha256": digest(archive), "arms": arms,
                "validation_predictions": 0, "test_predictions": 0,
                "held_out_predictions_per_arm": len(held) * 15}
        fold_path = args.output / f"fold-{fold_index:02d}.json"
        fold_path.write_text(json.dumps(fold, indent=2) + "\n")
        fold_hashes[fold_path.name] = digest(fold_path)
        held_count += len(held)
        print(json.dumps({"fold_index": fold_index, "held_cases": len(held), "updates": 10}), flush=True)
    result = {"study": plan["study"], "role": plan["role"], "pins": pins,
              "status": "complete fixed-update TRAIN-only group cross-fitting shard",
              "shard": args.shard, "fold_indices": fold_indices, "fold_json_sha256": fold_hashes,
              "folds": len(fold_indices), "shards": 6, "updates": 10, "arms": list(ARMS),
              "held_out_cases": held_count, "held_out_predictions_per_arm": held_count * 15,
              "validation_predictions": 0, "test_predictions": 0,
              "slurm_job": os.getenv("SLURM_JOB_ID"), "torch": str(torch.__version__)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
