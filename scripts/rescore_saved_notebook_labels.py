"""Describe native image results after mapping eight BreakHis types to five final labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


# LabelEncoder sorts the original subtype directory names alphabetically.
EIGHT = ("adenosis", "ductal_carcinoma", "fibroadenoma", "lobular_carcinoma",
         "mucinous_carcinoma", "papillary_carcinoma", "phyllodes_tumor", "tubular_adenoma")
FIVE = np.array([0, 1, 0, 2, 3, 4, 0, 0])
ORIGINAL = {
    "non-localattention-breakhis.ipynb": "8321e316e03364883b8138901df2630849d3888ee758c694bb3180b3139045bc",
    "spatialsequenceattention-breakhis.ipynb": "b3e2c0f1d1c076519e88df0ab84863171486b8129046cb12c73487ccf908aa58",
}


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected source hash for {path.name}")
    return data


def from_notebook(path, expected):
    cells = json.loads(checked_bytes(path, expected))["cells"]
    text = "".join("".join(output.get("text", [])) for output in cells[11]["outputs"])
    body = re.search(r'Confusion Matrix:\s*\n\s*(\[\[.*?\]\])', text, re.DOTALL)
    if body is None:
        raise ValueError("Missing original confusion matrix")
    matrix = np.array([int(n) for n in re.findall(r'\d+', body.group(1))]).reshape(8, 8)
    if matrix.sum(1).tolist() != [23, 173, 50, 31, 41, 29, 22, 30]:
        raise ValueError("Unexpected class supports")
    five_equal = FIVE[:, None] == FIVE
    binary_equal = (FIVE[:, None] > 0) == (FIVE > 0)
    return {"images": int(matrix.sum()), "correct_eight": int(matrix.trace()),
            "correct_five_derived": int(matrix[five_equal].sum()),
            "correct_binary_derived": int(matrix[binary_equal].sum()), "source_sha256": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root/"docs/evidence/SUPPLIED_CNN_TRANSFORMER_RESULTS.json").read_text())
    original = {name: from_notebook(args.originals/name, sha) for name, sha in ORIGINAL.items()}
    variants = {}
    cancer_columns = np.flatnonzero(FIVE > 0)
    for name, evidence in source["variants"].items():
        path = args.predictions/(name+"-predictions.npz")
        sha = evidence["artifacts"]["predictions.npz"]
        checked_bytes(path, sha)
        with np.load(path, allow_pickle=False) as z:
            truth, truth_binary = z["truth_subtype"], z["truth_binary"]
            pred_binary = z["binary_scores"].argmax(1)
            subtype = z["subtype_scores"]
            if not np.array_equal(truth_binary, FIVE[truth] > 0):
                raise ValueError("Subtype order does not match the binary ground truth")
            five = np.where(pred_binary == 0, 0, FIVE[cancer_columns[subtype[:, cancer_columns].argmax(1)]])
            correct_binary = int(np.sum(pred_binary == truth_binary))
            correct_eight = int(np.sum(subtype.argmax(1) == truth))
            if (len(truth) != 1582 or correct_binary != evidence["metrics"]["binary"]["correct"]
                    or correct_eight != evidence["metrics"]["eight_class"]["correct"]):
                raise ValueError("Original metrics failed replay")
            variants[name] = {"images": len(truth), "correct_eight": correct_eight, "correct_binary": correct_binary,
                              "correct_five_derived": int(np.sum(five == FIVE[truth])), "source_sha256": sha}
    result = {"unit": "single image on each original evaluation split", "eight_class_order": EIGHT,
              "mapped_five_index": FIVE.tolist(), "historical_40x_notebooks": original, "supplied_cnn": variants,
              "cnn_decoding": "Use the saved binary argmax. If malignant, choose the highest of the four malignant subtype scores.",
              "historical_decoding": "Map each saved eight-class hard decision and true label to the corresponding five or two labels.",
              "new_fits": 0, "new_encoder_calls": 0}
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
