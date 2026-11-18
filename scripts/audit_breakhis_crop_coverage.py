"""Quantify omitted color in the blinded fixed review packet."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit(fields, centers):
    names = {f"box-{case:02d}-{zoom}.png" for case in range(1, 22)
             for zoom in (40, 100, 200, 400)}
    for folder in (fields, centers):
        if {file.name for file in folder.glob("*.png")} != names:
            raise ValueError("Expected four fields per box across 21 boxes")
    fractions = []
    areas = []
    for name in sorted(names):
        with Image.open(fields / name) as image:
            if image.mode != "RGB" or image.size not in ((700, 460), (700, 456)):
                raise ValueError("Unexpected original geometry")
            width = image.width * 224 // image.height
            rgb = np.asarray(image.resize((width, 224), Image.Resampling.BILINEAR))
        left = round((width - 224) / 2)
        center = rgb[:, left:left + 224]
        with Image.open(centers / name) as saved:
            if not np.array_equal(center, np.asarray(saved)):
                raise ValueError("Displayed center differs from reconstructed center")
        # Simple color proxy. Includes artifacts and excludes pale tissue.
        colored = (np.ptp(rgb, axis=2) >= 20) & (rgb.min(axis=2) <= 220)
        count = np.count_nonzero(colored)
        if not count:
            raise ValueError("Color fraction is undefined for a colorless field")
        kept = np.count_nonzero(colored[:, left:left + 224])
        fractions.append(float((count - kept) / count))
        areas.append((width - 224) / width)
    return {
        "fields": len(names), "boxes": 21, "exact_center_pixel_matches": len(names),
        "image_coordinates": "Bilinear short-side 224 resize, round-to-even center offset",
        "proxy": "8-bit RGB max minus min >= 20 and min <= 220 on the resized full field",
        "outside_area_fraction_range": [min(areas), max(areas)],
        "outside_proxy_fraction_per_field": dict(zip(
            ("min", "q1", "median", "q3", "max"),
            np.quantile(fractions, [0, .25, .5, .75, 1]).tolist())),
        "interpretation": "Color can reflect tissue or artifacts. These measurements cannot identify lost diagnostic cues.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", type=Path)
    parser.add_argument("centers", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.fields, args.centers), indent=2))
