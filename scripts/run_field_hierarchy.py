"""Compare the order of field pooling and the shared hidden layer on BreakHis."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from magset.bags import load_bags
from magset.data import MAGNIFICATIONS
from magset.field_hierarchy import FieldHierarchy, PackedFields
from magset.shared_hierarchy import SharedHierarchy, zoom_masks
from scripts.run_shared_hierarchy import digest, fit_classifier, load_development, summarize


ROOT = Path(__file__).resolve().parents[1]


def read_pinned(path, sha):
    if digest(path) != sha:
        raise ValueError(f"Checksum differs for {path.name}")
    return json.loads(path.read_text())


def pack_role(cases, bags, indices, reference_views):
    fields, case_ordinals, zooms, paths = [], [], [], []
    means = []
    for ordinal, index in enumerate(indices):
        case_means = []
        for zoom, magnification in enumerate(MAGNIFICATIONS):
            data = bags[index][zoom]
            keys = cases[index]["indices"][magnification]
            if len(keys) != len(data) or data.shape[1] != 1536:
                raise ValueError("Field identity count or UNI2-h feature dimension differs")
            fields.append(data)
            case_ordinals.extend([ordinal] * len(data))
            zooms.extend([zoom] * len(data))
            paths.extend(hashlib.sha256(key.encode()).hexdigest() for key in keys)
            case_means.append(data.mean(axis=0, dtype=np.float64))
        means.append(case_means)
    means = torch.from_numpy(np.asarray(means, dtype=np.float32))
    if not torch.equal(means, reference_views):
        raise ValueError("Rebuilt case-view means differ from the accepted pilot")
    batch = PackedFields(fields=torch.from_numpy(np.concatenate(fields)), case=torch.tensor(case_ordinals),
                         zoom=torch.tensor(zooms), views=means)
    return batch, np.asarray(paths, dtype="S64")


def metrics_with_worst(truth, predictions):
    result = summarize(truth, predictions)
    result["worst_subset_accuracy"] = float((truth[:, None] == predictions).mean(axis=0).min())
    result["worst_subset_macro_f1"] = min(float(f1_score(truth, col, labels=range(5), average="macro",
                                                       zero_division=0)) for col in predictions.T)
    return result


def check_control(report, reference):
    if report["selected_epoch"] != reference["selected_epoch"] or len(report["history"]) != len(reference["history"]):
        raise ValueError("Matched control checkpoint schedule or selection differs from pilot")
    for old, replay in zip(reference["history"], report["history"]):
        if any(replay[key] != value for key, value in old.items()):
            raise ValueError("Matched control validation metrics differ from the original")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "audit", "views", "field-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    if (ROOT / ".source-sha").read_text().strip() != args.code_sha:
        raise ValueError("Immutable source revision differs")
    config_sha = digest(ROOT / "configs/breakhis-field-hierarchy-v1.json")
    config = read_pinned(args.config, config_sha)
    control_ref = read_pinned(ROOT / "docs/evidence/BREAKHIS_SHARED_HIERARCHY_VALIDATION.json",
                              config["previous_result_sha256"])
    cache_config = read_pinned(ROOT / "configs/breakhis-uni2-h-v1-kernel.json", config["feature_config_sha256"])
    extraction = read_pinned(ROOT / "configs/uni2-h-v1.json", config["extraction_config_sha256"])
    train_mean, scale, x_train, x_val, y_train, y_val, val_indices = load_development(args.audit, args.views, config)
    audit = json.loads(args.audit.read_text())
    for mag, pins in cache_config["cache_provenance"].items():
        shard = args.field_cache / (cache_config["encoder_name"] + "-" + mag)
        meta = read_pinned(shard.with_suffix(".json"), pins["json_sha256"])
        if (digest(shard.with_suffix(".npz")) != pins["npz_sha256"] or meta["encoder"] != extraction["encoder"]
                or extraction["encoder"]["revision"] != config["reader_revision"]
                or extraction["encoder"]["weights_sha256"] != config["weights_sha256"]):
            raise ValueError("Field shard or pretrained weights differ")
    cases, bags, _ = load_bags(sorted(audit["fields"], key=lambda r: r["relative_path"]), args.field_cache,
                               cache_config["encoder_name"], config["manifest_sha256"])
    if [c["case_id"] for c in cases] != [c["case_id"] for c in audit["cases"]]:
        raise ValueError("Field and case identifiers are misaligned")
    train, train_paths = pack_role(cases, bags, audit["roles"]["train"], x_train)
    val, val_paths = pack_role(cases, bags, val_indices, x_val)
    if (len(train.fields), len(val.fields)) != (4621, 1584):
        raise ValueError("Unexpected development field counts")
    args.output.mkdir(parents=True, exist_ok=False)
    input_hashes = {}
    for role, batch, paths, indices in (("train", train, train_paths, audit["roles"]["train"]),
                                        ("validation", val, val_paths, val_indices)):
        fields_file = args.output / (role + "_fields.npz")
        np.savez_compressed(fields_file, fields=batch.fields.numpy(), case=batch.case.numpy(), zoom=batch.zoom.numpy(),
                            views=batch.views.numpy(), path_sha256=paths, case_indices=indices)
        input_hashes[role + "_fields_sha256"] = digest(fields_file)
    receipt = {"train_fields_sha256": input_hashes["train_fields_sha256"], "config_sha256": config_sha,
               "audit_sha256": config["audit_sha256"], "code_sha": args.code_sha,
               "case_views_sha256": config["case_views_sha256"]}
    (args.output / "train-input-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    common = {"study": config["study"], "role": "validation", "code_sha": args.code_sha,
              "source_archive_sha256": (ROOT / ".archive-sha256").read_text().strip(), "config_sha256": config_sha,
              "train_cases": len(y_train), "train_class_counts": torch.bincount(y_train).tolist(),
              "validation_class_counts": torch.bincount(y_val).tolist(), "test_predictions": 0,
              "inputs": {k: config[k] for k in ("manifest_sha256", "audit_sha256", "case_views_sha256",
                                                "reader", "reader_revision", "weights_sha256")},
              "slurm_job": os.getenv("SLURM_JOB_ID"), "torch": str(torch.__version__)}
    masks = zoom_masks()
    arm_hashes, selected = {}, {}
    for arm, network, train_inputs, val_inputs in (("control", SharedHierarchy, x_train, x_val),
                                                  ("field", FieldHierarchy, train, val)):
        folder = args.output / arm
        folder.mkdir()
        def build_model(network=network):
            return network(train_mean, scale, config["hidden"], config["dropout"])
        fitted = fit_classifier(config, build_model, (train_inputs, masks), (val_inputs, masks),
                                y_train, y_val, val_indices, folder, metrics_with_worst)
        report = {**common, "arm": arm, **fitted}
        if arm == "control":
            check_control(report, control_ref)
        selected[arm] = report["selected"]
        path = folder / "result.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        arm_hashes[arm] = digest(path)
    field, control = selected["field"], selected["control"]
    limits = config["thresholds"]
    targets = {"five_above_90": field["accuracy"] > limits["final_five_accuracy_strictly_greater_than"],
               "binary_above_95": field["binary_accuracy"] > limits["binary_accuracy_strictly_greater_than"]}
    targets["both_at_selected_checkpoint"] = all(targets.values())
    improved = (field["correct_final"] > control["correct_final"] and field["correct_binary"] > control["correct_binary"]
                and field["mean_subset_macro_f1"] >= control["mean_subset_macro_f1"])
    result = {**common, **input_hashes, "arm_result_sha256": arm_hashes,
              "train_fields": len(train.fields), "validation_fields": len(val.fields), "targets": targets,
              "improvement_screen_passed": improved, "selected": selected,
              "feature_config_sha256": config["feature_config_sha256"],
              "extraction_config_sha256": config["extraction_config_sha256"], "cache_provenance": cache_config["cache_provenance"]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"targets": targets, "improvement_screen_passed": improved, "selected": selected}))


if __name__ == "__main__":
    main()
