import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional")
class WiderCaseHierarchyTests(unittest.TestCase):
    def test_advance_requires_both_validation_counts_and_three_training_seeds(self):
        from scripts.evaluate_wider_case_hierarchy import progression

        def pair(exact, binary, f1):
            return {"center": {"correct_final": 10, "correct_binary": 12, "mean_subset_macro_f1": .5},
                    "wider": {"correct_final": exact, "correct_binary": binary, "mean_subset_macro_f1": f1}}

        good = pair(11, 13, .5)
        tie = pair(10, 12, .5)
        self.assertTrue(progression([good, good, good, tie], good)["passes"])
        self.assertFalse(progression([good, good, tie, tie], good)["passes"])
        self.assertFalse(progression([good] * 4, pair(11, 12, .6))["passes"])
        self.assertFalse(progression([good] * 4, pair(10, 13, .6))["passes"])
        self.assertFalse(progression([good] * 4, pair(11, 13, .49))["passes"])
        self.assertFalse(progression([pair(11, 11, .5)] * 4, good)["passes"])
        self.assertFalse(progression([pair(11, 12, .49)] * 4, good)["passes"])
        with self.assertRaises(ValueError):
            progression([good] * 3, good)

    def test_case_change_counts_keep_all_mask_answers_together(self):
        import numpy as np
        from scripts.evaluate_wider_case_hierarchy import paired_case_changes

        truth = np.array([0, 1, 2])
        center = np.array([[1, 1], [1, 1], [1, 2]])
        wider = np.array([[0, 1], [0, 0], [2, 3]])
        self.assertEqual(paired_case_changes(truth, center, wider),
                         {"cases_improved": 1, "cases_tied": 1, "cases_worse": 1})
