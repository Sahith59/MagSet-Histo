"""Regression for class-complete fitting-only smoke summaries; no fitting."""
import unittest

import numpy as np

from magset import specialist_five as runner, specialist_collect as collector
from magset.five_metrics import MAP8_TO5, summarize_records
from scripts.review_specialist_smoke import fitting_queries


class SpecialistSmokeReviewTests(unittest.TestCase):
    def test_five_class_selector_supports_all_four_arm_summaries(self):
        labels = np.repeat(np.arange(5), [11, 25, 2, 5, 3])
        train = np.arange(len(labels))
        selected = fitting_queries(train, labels)
        np.testing.assert_array_equal(selected, [0, 11, 36, 38, 43])
        np.testing.assert_array_equal(labels[selected], np.arange(5))
        original = [0, 1, 3, 4, 5]
        self.assertEqual([MAP8_TO5[i] for i in original], list(range(5)))
        cases = [{'case_id': 'synthetic%d'%i, 'label': original[label]} for i, label in enumerate(labels)]
        rng = np.random.default_rng(391)
        bags = [[rng.normal(size=(2, 7)).astype(np.float32) for _ in range(4)] for _ in train]
        factors = {'binary': np.exp(np.linspace(-.2, .2, 7)), 'malignant': np.exp(np.linspace(.2, -.2, 7))}
        total = 0
        for arm in runner.ARMS:
            names = () if arm == 'identity' else ('binary', 'malignant') if arm == 'both' else (arm.removesuffix('_only'),)
            active = {head: factors[head] for head in names}
            rows = collector.replay_records(bags, labels, train, selected, cases, 'train', .1, active)
            summary = summarize_records(rows, 'dual')
            self.assertEqual(summary['pooled']['records'], 75)
            self.assertEqual({row['truth5'] for row in rows}, set(range(5)))
            total += len(rows)
        self.assertEqual(total, 300)
        incomplete = collector.replay_records(bags, labels, train, train[:2], cases, 'train', .1, {})
        with self.assertRaises(ValueError):
            summarize_records(incomplete, 'dual')

    def test_missing_fitting_class_rejected_before_replay(self):
        with self.assertRaisesRegex(ValueError, 'five final labels'):
            fitting_queries(np.arange(4), np.arange(4))


if __name__ == '__main__':
    unittest.main()
