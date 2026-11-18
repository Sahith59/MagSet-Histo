"""Build three-crop UNI2-h case features from the matched development fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from magset import approved_features as approved
from magset.breakhis_predict import load_checked_image
from magset.data import MAGNIFICATIONS, file_hash
from scripts.check_field_hierarchy import fixed_roles, require

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {40: 1613, 100: 1628, 200: 1555, 400: 1409}


def windows(image, config):
    """Use the official scale and normalization at the left, center and right."""
    import torch
    from torchvision import transforms

    center, geometry = approved.approved_pixels(image, config)
    resized = image.resize(tuple(geometry["resized"]), Image.Resampling.BILINEAR)
    spec = config["encoder"]["transform"]
    require(resized.height == 224 and 224 <= resized.width <= 448 and
            spec["interpolation"] == "bilinear", "Incomplete field coverage")
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])])
    left = normalize(resized.crop((0, 0, 224, 224)))
    middle = normalize(resized.crop(tuple(geometry["crop_box"])))
    right = normalize(resized.crop((resized.width - 224, 0, resized.width, 224)))
    require(torch.equal(middle, center), "Center crop pixels differ from the official input")
    return torch.stack([left, center, right])


def development_rows(audit, development, zoom):
    allowed = set(development.tolist())
    selected = sorted((row for row in audit["fields"] if row["case_index"] in allowed
                       and int(row["magnification"]) == zoom), key=lambda row: row["relative_path"])
    require(len(selected) == EXPECTED_COUNTS[zoom]
            and len({row["relative_path"] for row in selected}) == len(selected)
            and all(row["view"] == MAGNIFICATIONS.index(zoom) for row in selected)
            and {row["case_index"] for row in selected} == allowed, "Unexpected development field identities")
    return selected


def path_digests(rows):
    return np.array([hashlib.sha256(row["relative_path"].encode()).hexdigest() for row in rows])


def center_references(cache, zoom, rows):
    config = json.loads((ROOT / "configs/breakhis-uni2-h-v1-kernel.json").read_text())
    pins = config["cache_provenance"][str(zoom)]
    shard = cache / f"uni2-h-v1-{zoom}.npz"
    require(file_hash(shard) == pins["npz_sha256"] and
            file_hash(shard.with_suffix(".json")) == pins["json_sha256"], "Center source shard differs")
    with np.load(shard, allow_pickle=False) as src:
        paths = src["paths"].tolist()
        require(len(paths) == len(set(paths)), "Duplicate cached center path")
        indices = {p: i for i, p in enumerate(paths)}
        refs = src["features"][[indices[row["relative_path"]] for row in rows]]
    require(refs.shape == (len(rows), 1536) and refs.dtype == np.float32
            and np.isfinite(refs).all(), "Invalid center cache")
    return refs, pins


def extract(args, recipe, rows, source):
    import torch

    config_path = ROOT / "configs/uni2-h-v1.json"
    config = approved.read_config(config_path)
    require(config["encoder"]["revision"] == recipe["reader_revision"] and
            config["encoder"]["weights_sha256"] == recipe["weights_sha256"], "Reader lineage differs")
    require(torch.cuda.is_available() and torch.cuda.get_device_name() == "NVIDIA A40",
            "Use the same A40 numerical recipe as the cached centers")
    versions = approved.base.runtime_versions(config)
    refs, pins = center_references(args.cache, args.zoom, rows)
    model = approved.load_model(args.weights, torch.device("cuda"), config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    components = np.empty((len(rows), 3, 1536), dtype=np.float32)
    with torch.inference_mode():
        for i, row in enumerate(rows):
            image = load_checked_image(args.images, row)
            values = approved.classification_features(model, windows(image, config).cuda(), config).cpu().numpy()
            require(np.array_equal(values[1], refs[i]), "Official center descriptor differs")
            components[i] = values
            if (i + 1) % 200 == 0:
                print(f"{args.zoom}x: {i + 1}/{len(rows)} fields", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"fields-{args.zoom}.npz"
    with target.open("xb") as f:
        np.savez_compressed(f, path_sha256=path_digests(rows), case_indices=[r["case_index"] for r in rows],
                            components=components)
    receipt = {"zoom": args.zoom, "fields": len(rows), "image_forwards": len(rows)*3,
               "center_max_abs_error": 0., "array_sha256": file_hash(target),
               "center_source": pins, "slurm_job": os.environ["SLURM_JOB_ID"],
               "gpu": torch.cuda.get_device_name(), "versions": versions,
               "reader_revision": recipe["reader_revision"], "weights_sha256": recipe["weights_sha256"],
               "encoder_config_sha256": file_hash(config_path), "code_sha": source,
               "source_archive_sha256": (ROOT / ".archive-sha256").read_text().strip()}
    target.with_suffix(".json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def collect(args, recipe, audit, development, source):
    center_views = np.empty((len(development), 4, 1536), dtype=np.float64)
    wide_views = np.empty_like(center_views)
    shards = {}
    for z, zoom in enumerate(MAGNIFICATIONS):
        rows = development_rows(audit, development, zoom)
        ref, pins = center_references(args.cache, zoom, rows)
        names = path_digests(rows)
        cases = np.array([row["case_index"] for row in rows])
        file = args.output / f"fields-{zoom}.npz"
        receipt = json.loads(file.with_suffix(".json").read_text())
        require(receipt["array_sha256"] == file_hash(file) and receipt["code_sha"] == source
                and receipt["center_source"] == pins and receipt["zoom"] == zoom and
                receipt["fields"] == len(rows), "Incomplete feature shard")
        with np.load(file, allow_pickle=False) as archive:
            parts = archive["components"]
            require(parts.dtype == np.float32 and parts.shape == (len(rows), 3, 1536)
                    and np.isfinite(parts).all() and np.array_equal(parts[:, 1], ref)
                    and np.array_equal(archive["case_indices"], cases)
                    and np.array_equal(archive["path_sha256"], names), "Field order or center replay differs")
            pooled_fields = parts.mean(axis=1, dtype=np.float64).astype(np.float32)
        for i, case in enumerate(development):
            center_views[i, z] = ref[cases == case].mean(axis=0, dtype=np.float64)
            wide_views[i, z] = pooled_fields[cases == case].mean(axis=0, dtype=np.float64)
        shards[str(zoom)] = {**receipt, "receipt_sha256": file_hash(file.with_suffix(".json"))}
    require(file_hash(args.views) == recipe["case_views_sha256"], "Original case cache differs")
    with np.load(args.views, allow_pickle=False) as original:
        require(np.array_equal(original["case_indices"], np.arange(82)) and
                np.array_equal(original["case_view_means"][development], center_views),
                "Rebuilt development center means differ from original study")
    output = args.output / "development-views.npz"
    with output.open("xb") as f:
        np.savez_compressed(f, case_indices=development, magnifications=MAGNIFICATIONS,
                            center_views=center_views, wide_views=wide_views)
    report = {k: recipe[k] for k in ("reader_revision", "weights_sha256", "audit_sha256",
                                     "manifest_sha256", "case_views_sha256")}
    report.update(code_sha=source, source_archive_sha256=(ROOT / ".archive-sha256").read_text().strip(),
                  development_views_sha256=file_hash(output), followup_fields=0, fields=6205,
                  image_forwards=18615, train_cases=49, validation_cases=17, center_max_abs_error=0.,
                  slurm_job=os.environ["SLURM_JOB_ID"], shards=shards)
    (args.output / "feature-receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "cache", "views", "weights", "images", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--zoom", type=int, choices=MAGNIFICATIONS, help="Omit to verify and collect all shards")
    args = p.parse_args()
    source = (ROOT / ".source-sha").read_text().strip()
    require(os.environ.get("SLURM_JOB_ID") and source == os.environ["MAGSET_CODE_SHA"],
            "An immutable scheduler run is required")
    recipe = json.loads((ROOT / "configs/breakhis-shared-hierarchy-v1.json").read_text())
    require(file_hash(args.audit) == recipe["audit_sha256"], "Input role audit differs")
    audit = json.loads(args.audit.read_text())
    require(audit["manifest_sha256"] == recipe["manifest_sha256"], "Dataset manifest differs")
    train, val = fixed_roles(audit)
    development = np.r_[train, val]
    if args.zoom is None:
        collect(args, recipe, audit, development, source)
    else:
        extract(args, recipe, development_rows(audit, development, args.zoom), source)


if __name__ == "__main__":
    main()
