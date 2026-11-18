"""Replay two archived comparators and evaluate one loss change on TRAIN groups only."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from magset.bags import load_bags
from magset.field_hierarchy import FieldHierarchy
from magset.shared_hierarchy import SharedHierarchy, final_labels, zoom_masks
from scripts.check_field_hierarchy import (compare_metrics, digest, fixed_roles, load_weights,
                                          metrics, numpy_logits, predictions, require)
from scripts.crossfit_single_output import fit_fold, roles_and_views
from scripts.run_field_hierarchy import pack_role, read_pinned
from scripts.verify_single_output_coverage import reconstruct, decide

ROOT = Path(__file__).resolve().parents[1]


def older_models(args, plan, masks):
    config = read_pinned(ROOT / "configs/breakhis-field-hierarchy-v1.json", plan["field_config_sha256"])
    audit = read_pinned(args.audit, config["audit_sha256"])
    train, val = fixed_roles(audit)
    pins = read_pinned(ROOT / "configs/breakhis-uni2-h-v1-kernel.json", config["feature_config_sha256"])
    for zoom, receipts in pins["cache_provenance"].items():
        for suffix in ("json", "npz"):
            path = args.field_cache / f"uni2-h-v1-{zoom}.{suffix}"
            require(digest(path) == receipts[suffix+"_sha256"], "Official feature shard changed")
    cases, bags, _ = load_bags(sorted(audit["fields"], key=lambda f: f["relative_path"]),
                               args.field_cache, pins["encoder_name"], config["manifest_sha256"])
    require([c["case_id"] for c in cases] == [c["case_id"] for c in audit["cases"]], "Case order differs")
    require(digest(args.views) == config["case_views_sha256"], "Unexpected case views")
    with np.load(args.views, allow_pickle=False) as cache:
        views = cache["case_view_means"]
    mean = views[train].mean(axis=0).astype(np.float32)
    std = views[train].std(axis=0)
    scale = np.where(std < 1.e-6, 1., std).astype(np.float32)
    result = {}
    for arm, cls in (("control", SharedHierarchy), ("field", FieldHierarchy)):
        receipt = plan["arms"][arm]
        folder = args.old_run / arm
        old = read_pinned(folder/"result.json", receipt["report_sha256"])
        checkpoint = folder/"selected.pt"
        require(digest(checkpoint) == receipt["checkpoint_sha256"], "Historical weights differ")
        state = load_weights(checkpoint, config, {"mean": mean, "scale": scale}, receipt["selected_epoch"])
        model = cls(torch.from_numpy(mean), torch.from_numpy(scale), config["hidden"], config["dropout"]).eval()
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        summary = {"checkpoint_sha256": digest(checkpoint)}
        for role, indices in (("validation", val), ("follow_up", np.array(audit["roles"]["test"]))):
            batch, _ = pack_role(cases, bags, indices, torch.tensor(views[indices], dtype=torch.float32))
            inputs = {k: getattr(batch, k).numpy() for k in ("fields", "case", "zoom", "views")}
            counts = np.bincount(4*inputs["case"] + inputs["zoom"], minlength=len(indices)*4).reshape(-1, 4)
            inputs.update(mean=mean, scale=scale, counts=counts, masks=masks.numpy())
            with torch.inference_mode():
                logits = model(batch.views if arm == "control" else batch, masks)
                output = final_labels(logits).numpy()
            independent = numpy_logits(arm, state, inputs)
            error = float(np.abs(logits.numpy() - independent).max())
            require(error < 1.e-5 and np.array_equal(predictions(independent), output), "Historical replay differs")
            truth = np.array([audit["cases"][i]["truth5"] for i in indices])
            scored = metrics(truth, output)
            if role == "validation":
                compare_metrics(scored, old["selected"], arm+" validation")
            archive = args.output / f"{arm}-{role}.npz"
            np.savez_compressed(archive, case_indices=indices, truth=truth, masks=masks.numpy(), predictions=output)
            summary[role] = dict(scored, numpy_replay_max_error=error, predictions_sha256=digest(archive))
        result[arm] = summary
    return result


def loss_experiment(args, plan, masks):
    config = read_pinned(ROOT / "configs/breakhis-single-output-center-v1.json", plan["single_config_sha256"])
    indices, views, truth, groups = roles_and_views(args, config)
    require(digest(args.prior_group) == plan["old_group_predictions_sha256"], "Baseline predictions changed")
    with np.load(args.prior_group, allow_pickle=False) as archive:
        require(np.array_equal(indices, archive["indices"]) and np.array_equal(truth, archive["truth"]) and
                np.array_equal(masks.numpy(), archive["masks"]), "Control case or mask ordering differs")
        old_predictions, fold_ids = archive["predictions"], archive["fold"]
    base = metrics(truth, old_predictions)
    output = np.full_like(old_predictions, -1)
    states = []
    for fold, group in enumerate(np.unique(groups)):
        held, fitting = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
        require(np.array_equal(held, np.flatnonzero(fold_ids == fold)), "Control folds differ")
        model, _, group_predictions = fit_fold(views, truth, fitting, held, config, masks, plan["training_class_masses"])
        output[held] = group_predictions
        states.append(model.state_dict())
    weights_path = args.output / "candidate-weights.pt"
    torch.save(states, weights_path)
    for fold, state in enumerate(torch.load(weights_path, weights_only=True, map_location="cpu")):
        held = fold_ids == fold
        raw = views[~held]
        center, std = raw.mean(axis=0).astype(np.float32), raw.std(axis=0)
        scale = np.where(std < 1.e-6, 1., std).astype(np.float32)
        require(np.array_equal(center, state["mean"].numpy()) and np.array_equal(scale, state["scale"].numpy()),
                "Held group entered normalization")
        logits = reconstruct(views[held], center, scale, masks.numpy(),
                             state["classifier.weight"].numpy(), state["classifier.bias"].numpy())
        require(np.array_equal(decide(logits), output[held]), "Candidate replay differs")
    archive = args.output / "candidate-training-groups.npz"
    np.savez_compressed(archive, indices=indices, truth=truth, masks=masks.numpy(), fold=fold_ids, predictions=output)
    candidate = metrics(truth, output)
    passed = (candidate["correct_final"] > base["correct_final"] and candidate["correct_binary"] > base["correct_binary"]
              and candidate["mean_subset_macro_f1"] >= base["mean_subset_macro_f1"])
    return {"baseline": base, "candidate": candidate, "progression_passed": passed,
            "weights_sha256": digest(weights_path), "predictions_sha256": digest(archive),
            "folds": len(states), "validation_predictions": 0, "follow_up_predictions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("views", "audit", "old-run", "field-cache", "prior-group", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / "configs/breakhis-matched-scorecard-v1.json"
    plan = json.loads(plan_path.read_text())
    revision = (ROOT / ".source-sha").read_text().strip()
    require(revision == os.environ["MAGSET_CODE_SHA"], "Executing different source")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    masks = zoom_masks()
    args.output.mkdir(parents=True, exist_ok=False)
    # The fitting experiment has no follow-up inputs or outcomes beyond those of the old, global split metadata.
    development = loss_experiment(args, plan, masks)
    reports = older_models(args, plan, masks)
    report = {"study": plan["study"], "code_sha": revision, "slurm_job": os.environ["SLURM_JOB_ID"],
              "archive_sha256": (ROOT / ".archive-sha256").read_text().strip(), "protocol_sha256": digest(plan_path),
              "training_group_loss_comparison": development, "historical_frozen_comparators": reports}
    (args.output/"aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
