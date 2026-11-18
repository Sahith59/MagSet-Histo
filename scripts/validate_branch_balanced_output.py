"""Score the prespecified ten-update class-weight candidate on the 17 validation cases."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from magset.shared_hierarchy import zoom_masks
from scripts.check_field_hierarchy import fixed_roles, metrics, require, digest
from scripts.crossfit_single_output import fit_fold
from scripts.run_field_hierarchy import read_pinned
from scripts.verify_single_output_coverage import reconstruct, decide

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "views", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / "configs/breakhis-matched-scorecard-v1.json"
    plan = json.loads(plan_path.read_text())
    progression_path = ROOT / "docs/evidence/MATCHED_SCORECARD_SUPPLEMENT.json"
    progression = json.loads(progression_path.read_text())
    require(progression["protocol_sha256"] == digest(plan_path)
            and progression["training_group_loss_comparison"]["progression_passed"], "Missing development gate")
    config = read_pinned(ROOT / "configs/breakhis-single-output-center-v1.json", plan["single_config_sha256"])
    audit = read_pinned(args.audit, config["audit_sha256"])
    train, val = fixed_roles(audit)
    require(digest(args.views) == config["case_views_sha256"], "Feature commitment differs")
    indices = np.concatenate([train, val])
    with np.load(args.views, allow_pickle=False) as source:
        require(np.array_equal(source["case_indices"], np.arange(82)), "Unexpected case order")
        views = source["case_view_means"][indices]
    truth = np.array([audit["cases"][i]["truth5"] for i in indices])
    fit = np.arange(len(train))
    held = np.arange(len(train), len(indices))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    masks = zoom_masks()
    model, logits, pred = fit_fold(views, truth, fit, held, config, masks, plan["training_class_masses"])
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output/"weights.pt"
    torch.save(model.state_dict(), weights)
    state = torch.load(weights, weights_only=True, map_location="cpu")
    center = views[fit].mean(axis=0).astype(np.float32)
    std = views[fit].std(axis=0)
    scale = np.where(std < 1.e-6, 1., std).astype(np.float32)
    require(np.array_equal(center, state["mean"].numpy()) and np.array_equal(scale, state["scale"].numpy()),
            "Normalizer differs from fitting-only statistics")
    rebuilt = reconstruct(views[held], center, scale, masks.numpy(),
                          state["classifier.weight"].numpy(), state["classifier.bias"].numpy())
    error = float(np.abs(logits - rebuilt).max())
    require(error < 1.e-5 and np.array_equal(pred, decide(rebuilt)), "Validation replay differs")
    archive = args.output/"validation.npz"
    np.savez_compressed(archive, indices=val, masks=masks.numpy(), truth=truth[held], predictions=pred, logits=logits)
    revision = (ROOT/".source-sha").read_text().strip()
    require(revision == os.environ["MAGSET_CODE_SHA"], "Different source")
    report = {"study":"branch-balanced-single-output-validation","source":revision,"job":os.environ["SLURM_JOB_ID"],
              "archive_sha256":(ROOT/".archive-sha256").read_text().strip(),"progression_sha256":digest(progression_path),
              "plan_sha256":digest(plan_path),"single_config_sha256":plan["single_config_sha256"],
              "audit_sha256":config["audit_sha256"], "views_sha256":config["case_views_sha256"],
              "fixed_updates":10,"training_cases":len(train),"test_predictions":0,"validation":metrics(truth[held],pred),
              "max_replay_logit_error":error,"weights_sha256":digest(weights),"prediction_sha256":digest(archive)}
    (args.output/"aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
