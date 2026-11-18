"""Measure frozen UNI2-h crop descriptor similarity on the fixed blinded packet."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from magset import approved_features as approved
from magset.data import file_hash

ZOOMS = (40, 100, 200, 400)
ROOT = Path(__file__).resolve().parents[1]


def packet_rows(path):
    if file_hash(path) != "bb8a6d6ec1654c45ec7ec6942b7c50f08dd5e03222b2ce655b79034c94f3c240":
        raise ValueError("Expected the original blinded packet mapping")
    with path.open(newline="") as handle:
        rows = [{"name": row["box"].lower().replace(" ", "-") + "-" + row["zoom"] + ".png",
                 "zoom": int(row["zoom"]), "path": row["source"], "hash": row["sha256"],
                 "role": row["role"]} for row in csv.DictReader(handle)]
    expected = {f"box-{i:02d}-{zoom}.png" for i in range(1, 22) for zoom in ZOOMS}
    if (len(rows) != 84 or {r["name"] for r in rows} != expected
            or len({r["path"] for r in rows}) != 84
            or any(r["role"] not in ("train", "validation") for r in rows)):
        raise ValueError("Expected the fixed 84 distinct TRAIN/validation fields")
    return rows


def reference_centers(cache, rows):
    pins = json.loads((ROOT / "configs/breakhis-uni2-h-v1-kernel.json").read_text())["cache_provenance"]
    references = {}
    for zoom in ZOOMS:
        file = cache / f"uni2-h-v1-{zoom}.npz"
        if file_hash(file) != pins[str(zoom)]["npz_sha256"]:
            raise ValueError("Center feature shard hash mismatch")
        with np.load(file, allow_pickle=False) as shard:
            indices = {str(path): i for i, path in enumerate(shard["paths"])}
            vectors = shard["features"]
            for row in rows:
                if row["zoom"] == zoom:
                    references[row["name"]] = vectors[indices[row["path"]]].copy()
    return references


def inputs(original, displayed, config):
    import torch
    from torchvision import transforms

    center, geometry = approved.approved_pixels(original, config)
    resized = original.resize(tuple(geometry["resized"]), Image.Resampling.BILINEAR)
    center_image = resized.crop(tuple(geometry["crop_box"]))
    if not np.array_equal(np.asarray(displayed), np.asarray(center_image)):
        raise ValueError("Displayed and reconstructed crop pixels differ")
    spec = config["encoder"]["transform"]
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])])
    if not torch.equal(normalize(center_image), center):
        raise ValueError("Reconstructed center differs from approved normalized pixels")
    right = resized.width - 224
    crops = [resized.crop((0, 0, 224, 224)), center_image, resized.crop((right, 0, resized.width, 224))]
    return torch.stack([normalize(crop) for crop in crops]), geometry


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Undefined cosine")
    return float(np.dot(a, b) / norm)


def summarize(values):
    return dict(zip(("min", "q1", "median", "q3", "max"),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist()), count=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fields", "centers", "key", "config", "weights", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch

    config = approved.read_config(args.config)
    if config["name"] != "uni2-h-v1" or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Expected registered UNI2-h and a scheduler GPU allocation")
    versions = approved.base.runtime_versions(config)
    rows = packet_rows(args.key)
    ref = reference_centers(args.cache, rows)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(812)
    model = approved.load_model(args.weights, torch.device("cuda"), config)
    receipts = []
    other = []
    centers_by_zoom = {zoom: [] for zoom in ZOOMS}
    with torch.inference_mode():
        for row in rows:
            name = row["name"]
            source = args.fields / name
            if file_hash(source) != row["hash"]:
                raise ValueError("Original field hash mismatch")
            with Image.open(source) as original, Image.open(args.centers / name) as displayed:
                pixels, geometry = inputs(original, displayed, config)
            left, center, right = approved.classification_features(model, pixels.cuda(), config).cpu().numpy()
            error = approved.assert_parity(center, ref[name], config)
            centers_by_zoom[row["zoom"]].append(center)
            receipts.append({"name": name, "sha256": row["hash"], "geometry": geometry,
                             "center_max_abs_error": error, "left_center": cosine(left, center),
                             "right_center": cosine(right, center)})
    for vectors in centers_by_zoom.values():
        other.extend(cosine(a, b) for a, b in itertools.combinations(vectors, 2))
    summary = {key: summarize([r[key] for r in receipts]) for key in ("left_center", "right_center")}
    summary.update({
        "different_box_same_zoom_center_pairs": summarize(other), "boxes": 21, "fields": 84,
        "code_sha": (ROOT / ".source-sha").read_text().strip(),
        "source_archive_sha256": (ROOT / ".archive-sha256").read_text().strip(),
        "slurm_job": os.environ["SLURM_JOB_ID"], "image_forwards": 252, "center_pixel_matches": 84, "center_feature_parity_matches": 84,
        "center_feature_max_abs_error": max(r["center_max_abs_error"] for r in receipts),
        "center_feature_parity_tolerance": config["parity"], "encoder_revision": config["encoder"]["revision"],
        "weights_sha256": config["encoder"]["weights_sha256"], "config_sha256": file_hash(args.config),
        "probe_sha256": file_hash(Path(__file__)), "dependencies": versions, "gpu": torch.cuda.get_device_name(),
        "measure": "Cosine between raw FP32 CLS vectors computed in FP64; same-field windows overlap.",
        "scope": "Descriptive representation sensitivity. Distinct-box pairs share fields across comparisons. No clinical cue, predictive benefit or independent patient comparison is established."
    })
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("private.json", "aggregate.json")):
        raise FileExistsError("Outputs already exist")
    (args.output / "private.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
