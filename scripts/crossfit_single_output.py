"""Train and replay the fixed single-output rule with each original training group held out."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from magset.shared_hierarchy import zoom_masks
from scripts.run_shared_hierarchy import digest
from scripts.train_single_output_coverage import SingleOutputClassifier, final_labels
from scripts.check_field_hierarchy import fixed_roles, metrics, require
from scripts.verify_single_output_coverage import reconstruct, decide


def roles_and_views(args, config):
    require(digest(args.audit) == config["audit_sha256"] and
            digest(args.views) == config["case_views_sha256"], "Input commitment differs")
    audit = json.loads(args.audit.read_text())
    train, _ = fixed_roles(audit)
    with np.load(args.views, allow_pickle=False) as cache:
        require(np.array_equal(cache["train_case_indices"], train) and
                np.array_equal(cache["case_indices"], np.arange(82)), "Cached case order differs")
        views = cache["case_view_means"][train]
    truth = np.array([audit["cases"][i]["truth5"] for i in train])
    groups = np.array([audit["cases"][i]["group"] for i in train])
    require(len(np.unique(groups)) == 42, "Expected 42 groups")
    return train, views, truth, groups


def fit_fold(views, truth, fitting, held, config, masks):
    raw = views[fitting]
    center = raw.mean(axis=0)
    std = raw.std(axis=0)
    scale = np.where(std < 1.e-6, 1., std)
    torch.manual_seed(config["seed"])
    model = SingleOutputClassifier(torch.tensor(center), torch.tensor(scale))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    inputs = torch.tensor(raw, dtype=torch.float32)
    labels = torch.from_numpy(truth[fitting])
    counts = torch.bincount(labels, minlength=5)
    require(bool((counts > 0).all()), "All classes must occur in each fitting fold")
    targets = labels[:, None].expand(-1, len(masks)).reshape(-1)
    for _ in range(10):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs, masks).reshape(-1, 5), targets, weight=counts.float().reciprocal())
        require(bool(torch.isfinite(loss)), "Nonfinite fitting loss")
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        scores = model(torch.tensor(views[held], dtype=torch.float32), masks)
        predictions = final_labels(scores)
    return model, scores.numpy(), predictions.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "audit", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    require(digest(args.config) == digest(source / "configs/breakhis-single-output-center-v1.json"),
            "Single-output recipe differs")
    code_sha = (source / ".source-sha").read_text().strip()
    require(code_sha == os.environ["MAGSET_CODE_SHA"], "Source revision mismatch")
    indices, views, truth, groups = roles_and_views(args, config)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    masks = zoom_masks()
    args.output.mkdir(parents=True, exist_ok=False)
    trained = []
    logits = np.empty((49, 15, 5), dtype=np.float32)
    predictions = np.full((49, 15), -1, dtype=np.int64)
    fold_no = np.full(49, -1, dtype=int)
    for number, group in enumerate(np.unique(groups)):
        held = np.flatnonzero(groups == group)
        fitting = np.flatnonzero(groups != group)
        require(set(groups[held]).isdisjoint(groups[fitting]), "Group crossed a role")
        model, group_scores, group_predictions = fit_fold(views, truth, fitting, held, config, masks)
        logits[held] = group_scores
        predictions[held] = group_predictions
        trained.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        fold_no[held] = number
    weights_file = args.output / "fold-weights.pt"
    prediction_file = args.output / "predictions.npz"
    torch.save(trained, weights_file)
    np.savez_compressed(prediction_file, indices=indices, truth=truth, fold=fold_no, logits=logits,
                        predictions=predictions, masks=masks.numpy())
    loaded = torch.load(weights_file, weights_only=True, map_location="cpu")
    max_error = 0.
    with np.load(prediction_file, allow_pickle=False) as recorded:
        require(np.array_equal(recorded["indices"], indices) and np.array_equal(recorded["truth"], truth)
                and np.array_equal(recorded["masks"], masks.numpy()), "Saved case or mask order differs")
        for number, group in enumerate(sorted(set(groups))):
            held = groups == group
            require(np.array_equal(held, recorded["fold"] == number), "Saved fold assignment differs")
            train = views[~held]
            center = train.mean(axis=0).astype(np.float32)
            std = train.std(axis=0)
            scale = np.where(std < 1.e-6, 1., std).astype(np.float32)
            state = {name: value.numpy() for name, value in loaded[number].items()}
            require(np.array_equal(center, state["mean"]) and np.array_equal(scale, state["scale"]),
                    "Fold normalizer differs from its fitting group statistics")
            replayed = reconstruct(views[held].astype(np.float32), center, scale, masks.numpy(),
                                   state["classifier.weight"], state["classifier.bias"])
            max_error = max(max_error, float(np.abs(replayed - recorded["logits"][held]).max()))
            require(np.array_equal(decide(replayed), recorded["predictions"][held]), "Fold replay differs")
        require(max_error <= 1.e-5, "Replay exceeded the fixed logit tolerance")
        performance = metrics(truth, recorded["predictions"])
    report = {"study": "breakhis-single-output-center-group-check", "code_sha": code_sha,
              "archive_sha256": (source / ".archive-sha256").read_text().strip(),
              "slurm_job": os.environ["SLURM_JOB_ID"], "source_roles": "49 original training cases",
              "folds": len(loaded), "updates_per_fold": 10, "validation_predictions": 0, "test_predictions": 0,
              "config_sha256": digest(args.config), "inputs": {k: config[k] for k in
              ("audit_sha256", "case_views_sha256", "reader_revision", "weights_sha256")},
              "weights_archive_sha256": digest(weights_file), "predictions_sha256": digest(prediction_file),
              "max_absolute_replay_error": max_error, "held_out": performance}
    (args.output / "aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(performance))


if __name__ == "__main__":
    main()
