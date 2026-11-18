import unittest
from importlib.util import find_spec

import numpy as np
if find_spec("torch") is None:
    raise unittest.SkipTest("These tests require PyTorch")

import torch

from magset.field_hierarchy import PackedFields
from scripts.crossfit_field_hierarchy import subset_batch


class CrossfitIsolationTests(unittest.TestCase):
    def test_excluding_a_case_removes_every_one_of_its_fields_and_zooms(self):
        views = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        cases = torch.arange(3).repeat_interleave(4)
        zooms = torch.arange(4).repeat(3)
        fields = views.reshape(12, 2)
        original = PackedFields(fields, cases, zooms, views)
        fitting = subset_batch(original, np.array([2, 0]))
        self.assertEqual(fitting.case.tolist(), [1] * 4 + [0] * 4)
        self.assertTrue(torch.equal(fitting.fields, torch.cat([fields[:4], fields[8:]])))
        self.assertTrue(torch.equal(fitting.views, views[[2, 0]]))
        held = subset_batch(original, np.array([1]))
        self.assertTrue(torch.equal(held.fields, fields[4:8]))
        for invalid in ([-1], [3], [1, 1], []):
            with self.assertRaises(ValueError):
                subset_batch(original, np.array(invalid))
