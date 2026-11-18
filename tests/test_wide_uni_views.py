import unittest

import numpy as np
from PIL import Image


class WiderWindowTest(unittest.TestCase):
    def test_overlapping_windows_cover_all_columns_and_match_official_center(self):
        try:
            import torch
            from torchvision.transforms import Normalize, ToTensor
        except ImportError:
            self.skipTest("Neural image dependencies are optional")

        import json
        from pathlib import Path
        from scripts.extract_wide_uni_views import windows

        config = json.loads((Path(__file__).resolve().parents[1] / "configs/uni2-h-v1.json").read_text())
        spec = config["encoder"]["transform"]
        for original_height in (456, 460):
            with self.subTest(original_height=original_height):
                image = Image.fromarray(np.random.default_rng(17).integers(
                    0, 256, size=(original_height, 700, 3), dtype=np.uint8))
                width = 224 * 700 // original_height
                pixels = image.resize((width, 224), Image.Resampling.BILINEAR)
                scaled = Normalize(spec["mean"], spec["std"])(ToTensor()(pixels))
                actual = windows(image, config)
                visited = np.zeros(width, dtype=bool)
                for pos, start in enumerate((0, round((width - 224)/2), width - 224)):
                    self.assertTrue(torch.equal(actual[pos], scaled[:, :, start:start+224]))
                    visited[start:start+224] = True
                self.assertTrue(visited.all())
