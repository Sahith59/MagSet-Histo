"""Trace the original case-first network's archived rare-type decisions on development cases."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from scripts.check_field_hierarchy import digest, predictions, require
from scripts.collect_field_crossfit import canonical_digest


ROOT = Path(__file__).resolve().parents[1]
ZOOMS = (40, 100, 200, 400)
SUBSETS = [s for count in range(1, 5) for s in combinations(ZOOMS, count)]
MASKS = np.array([[z in s for z in ZOOMS] for s in SUBSETS])
CLASSES = ("benign", "ductal", "lobular", "mucinous", "papillary")
COMPARATORS = {2: 1, 4: 3}


def fold_archives(folder, public):
    """Authenticate the original scores using the collector's public inventory commitment."""
    rows, archives = [], []
    for shard in range(6):
        directory = folder / str(shard)
        for index in range(shard, 42, 6):
            base = directory / f"fold-{index:02d}"
            receipt_path, scores = base.with_suffix(".json"), base.with_suffix(".npz")
            receipt = json.loads(receipt_path.read_text())
            require(digest(scores) == receipt["archive_sha256"], "Group score archive changed")
            for arm, result in receipt["arms"].items():
                rows.append({"file": f"{shard}/{base.name}-{arm}.pt", "sha256": result["checkpoint_sha256"]})
            for path in (receipt_path, scores):
                rows.append({"file": f"{shard}/{path.name}", "sha256": digest(path)})
            archives.append(scores)
        rows.append({"file": f"{shard}/result.json", "sha256": digest(directory / "result.json")})
    require(canonical_digest(sorted(rows, key=lambda r: r["file"])) == public["fold_inventory_sha256"],
            "Archived decisions do not match the published fold inventory")
    return archives


def similarity_contrast(standardized, held_view, fitting_labels, label):
    """Mean cosine to fitting cases of true type minus mean cosine to cases of its comparator."""
    margins = []
    for mask in MASKS:
        reference = standardized[:, mask].mean(axis=1)
        query = held_view[mask].mean(axis=0)
        lengths = np.linalg.norm(reference, axis=1) * np.linalg.norm(query)
        require((lengths > 0).all(), "Undefined cosine in a selected zoom subset")
        cosine = np.einsum("nd,d->n", reference, query) / lengths
        margins.append(cosine[fitting_labels == label].mean() - cosine[fitting_labels == COMPARATORS[label]].mean())
    return np.asarray(margins)


def audit_role(paths, role, audit, views):
    rare = {label: [] for label in COMPARATORS}
    encountered = []
    group_names = {label: set() for label in COMPARATORS}
    training = np.asarray(audit["roles"]["train"])
    for path in paths:
        held_key, score_key, pred_key = ("held_case_indices", "control_logits", "control_predictions") if (
            role == "train") else ("case_indices", "logits", "predictions")
        with np.load(path, allow_pickle=False) as data:
            fitting = data["fit_case_indices"] if role == "train" else training
            held, scores, truth = data[held_key], data[score_key], data["truth"]
            require(np.array_equal(data["masks"], MASKS) and np.array_equal(predictions(scores), data[pred_key]),
                    "Archived mask order or hard routing differs")
        cases = audit["cases"]
        require(set(fitting).issubset(training) and set(held).issubset(audit["roles"][role]), "Unexpected role")
        require(not ({cases[i]["group"] for i in fitting} & {cases[i]["group"] for i in held}), "Group leakage")
        require([cases[i]["truth5"] for i in held] == truth.tolist(), "Truth differs")
        encountered.extend(held)
        rare_positions = [i for i, y in enumerate(truth) if y in COMPARATORS]
        if not rare_positions:
            continue
        raw = np.stack([views[i] for i in fitting])
        center, std = raw.mean(axis=0), raw.std(axis=0)
        scale = np.where(std < 1.e-6, 1., std)
        reference = (raw - center) / scale
        fit_labels = np.asarray([cases[i]["truth5"] for i in fitting])
        for i in rare_positions:
            label, case = int(truth[i]), held[i]
            sub = 1 + np.argmax(scores[i, :, 1:], axis=1)
            final = np.where(scores[i, :, 0] >= 0, sub, 0)
            margins = similarity_contrast(reference, (views[case] - center) / scale, fit_labels, label)
            rare[label].append((final, sub, margins))
            group_names[label].add(cases[case]["group"])
    require(sorted(encountered) == sorted(audit["roles"][role]), "Cases missing or repeated")
    summary = {}
    for label, rows in rare.items():
        final, sub, margins = (np.stack(column) for column in zip(*rows))
        correct = final == label
        summary[CLASSES[label]] = {
            "cases": len(rows), "groups": len(group_names[label]), "answers": int(final.size),
            "all_five_category_counts": np.bincount(final.ravel(), minlength=5).tolist(),
            "subtype_counts_with_true_malignant_route": np.bincount(sub.ravel(), minlength=5).tolist(),
            "correct_by_zoom_subset": correct.sum(axis=0).tolist(),
            "sorted_counts_of_correct_answers_per_case": sorted(correct.sum(axis=1).tolist()),
            "sorted_counts_if_true_malignant_route_per_case": sorted((sub == label).sum(axis=1).tolist()),
            "full_zoom_predictions_by_category": np.bincount(final[:, -1], minlength=5).tolist(),
            "geometry_comparator": CLASSES[COMPARATORS[label]],
            "true_type_mean_cosine_higher": int((margins > 0).sum()),
            "geometry_ties": int((margins == 0).sum()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "views", "crossfit", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evidence = ROOT / "docs/evidence"
    group = json.loads((evidence / "FIELD_HIERARCHY_GROUP_CHECK.json").read_text())
    validation = json.loads((evidence / "FIELD_HIERARCHY_CONTROL.json").read_text())
    require(digest(args.audit) == group["audit_sha256"] and digest(args.views) == group["case_views_sha256"],
            "Unexpected data or frozen case descriptors")
    require(digest(args.validation) == validation["validation_sha256"], "Unexpected validation score archive")
    audit = json.loads(args.audit.read_text())
    ids = np.asarray(audit["roles"]["train"] + audit["roles"]["validation"])
    with np.load(args.views, allow_pickle=False) as cache:
        require(np.array_equal(cache["case_indices"], np.arange(82)), "Unexpected descriptor order")
        views = dict(zip(ids, cache["case_view_means"][ids]))
    report = {"scientific_source": validation["code_sha"], "audit_sha256": group["audit_sha256"],
              "case_views_sha256": group["case_views_sha256"], "fold_inventory_sha256": group["fold_inventory_sha256"],
              "validation_archive_sha256": validation["validation_sha256"], "class_order": CLASSES,
              "zoom_subsets": SUBSETS, "new_fits": 0, "followup_cases_scored": 0,
              "training_group_holdouts": audit_role(fold_archives(args.crossfit, group), "train", audit, views),
              "validation": audit_role([args.validation], "validation", audit, views)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
