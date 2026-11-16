"""Analytic frozen-field fixtures; no cohort data, encoder work or fitted heads."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from magset import kernel_model as kernel
from magset.field_five import pool_log_probabilities


def fixture():
    labels = np.repeat(np.arange(5), 2)
    bags = []
    for i in range(11):
        views = []
        for view in range(4):
            views.append(np.asarray([[i + 1., view + 1., (i % 3) - 1.],
                                     [i + 2., view + .5, (i % 2) + .25]], dtype=np.float32))
        bags.append(views)
    return bags, np.append(labels, 0), np.arange(10)


class KernelModelTests(unittest.TestCase):
    def test_class_case_and_field_mass(self):
        labels = np.asarray([0, 0, 1, 2, 3, 4])
        counts = np.asarray([1, 9, 2, 3, 4, 5])
        for head in kernel.HEAD_SLICES:
            weights = kernel.class_weights(labels, counts, head)
            np.testing.assert_allclose(weights.sum(0), 1, atol=1e-15)
        direct = kernel.class_weights(labels, counts)
        self.assertAlmostEqual(direct[0, 0], .5)
        self.assertAlmostEqual(direct[1:10, 0].sum(), .5)
        malignant = kernel.class_weights(labels, counts, 'malignant')
        self.assertEqual(float(malignant[:10].sum()), 0)
        binary = kernel.class_weights(labels, counts, 'binary')
        self.assertAlmostEqual(binary[:1, 0].sum(), .5)
        self.assertAlmostEqual(binary[1:10, 0].sum(), .5)
        self.assertAlmostEqual(binary[10:12, 1].sum(), .25)

    def test_density_matches_explicit_logsumexp_oracle(self):
        bags, labels, support = fixture()
        prepared = kernel.prepare_case(bags, labels, support, 10)
        for temperature in kernel.TEMPERATURES:
            logs = kernel.density_logs(prepared, temperature)
            for view_index, view in enumerate(prepared.views):
                for head, columns in kernel.HEAD_SLICES.items():
                    weights = view.weights[:, columns]
                    explicit = []
                    for column in range(weights.shape[1]):
                        selected = weights[:, column] > 0
                        explicit.append(logsumexp(view.cosine[:, selected] / temperature
                                                  + np.log(weights[selected, column]), axis=1))
                    explicit = np.column_stack(explicit)
                    explicit -= logsumexp(explicit, axis=1, keepdims=True)
                    np.testing.assert_allclose(logs[head][view_index], explicit, atol=3e-14, rtol=1e-13)

    def test_uniform_field_duplication_and_permutation(self):
        bags, labels, support = fixture()
        original = kernel.prepare_case(bags, labels, support, 10)
        altered = copy.deepcopy(bags)
        altered[0][0] = np.repeat(altered[0][0], 3, axis=0)
        altered[2][3] = altered[2][3][::-1]
        changed = kernel.prepare_case(altered, labels, support[::-1], 10)
        for arm in ('direct', 'dual'):
            first = kernel.field_log_probabilities(original, .03, arm)
            second = kernel.field_log_probabilities(changed, .03, arm)
            for x, y in zip(first, second):
                np.testing.assert_allclose(x, y, atol=1e-12, rtol=1e-12)

    def test_whole_query_case_excluded_from_center_and_keys(self):
        bags, labels, support = fixture()
        prepared = kernel.prepare_case(bags, labels, support, 0, leave_query_case_out=True)
        self.assertNotIn(0, prepared.support_indices)
        self.assertEqual(len(prepared.support_indices), 9)
        expected = np.mean([bags[i][0].astype(np.float64).mean(0) for i in range(1, 10)], axis=0)
        np.testing.assert_array_equal(prepared.centers[0], expected)
        altered = copy.deepcopy(bags)
        altered[0] = [np.full_like(view, 1e10) for view in altered[0]]
        changed = kernel.prepare_case(altered, labels, support, 0, leave_query_case_out=True)
        for a, b in zip(prepared.centers, changed.centers):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(prepared.views[0].cosine.shape[1], 18)
        self.assertEqual(kernel.class_weights(labels[1:10], np.full(9, 2)).shape, (18, 5))
        with self.assertRaises(ValueError):
            kernel.prepare_case(bags, labels, support, 0)
        with self.assertRaises(ValueError):
            kernel.prepare_case(bags, labels, support, 10, leave_query_case_out=True)

    def test_center_equal_case_mass_not_equal_field_mass(self):
        bags, labels, support = fixture()
        bags[0][0] = np.repeat(np.asarray([[100., 0., 0.]], dtype=np.float32), 20, axis=0)
        prepared = kernel.prepare_case(bags, labels, support, 10)
        expected = np.mean([bags[i][0].astype(np.float64).mean(0) for i in support], axis=0)
        field_mean = np.concatenate([bags[i][0] for i in support]).mean(0)
        np.testing.assert_array_equal(prepared.centers[0], expected)
        self.assertFalse(np.allclose(prepared.centers[0], field_mean))

    def test_same_magnification_only(self):
        bags, labels, support = fixture()
        prepared = kernel.prepare_case(bags, labels, support, 10)
        bags[2][1] *= 100
        changed = kernel.prepare_case(bags, labels, support, 10)
        np.testing.assert_array_equal(prepared.views[0].cosine, changed.views[0].cosine)
        np.testing.assert_array_equal(kernel.density_logs(prepared, .1)['direct'][0],
                                      kernel.density_logs(changed, .1)['direct'][0])

    def test_zero_nearzero_extremes_and_missing_class(self):
        zero = kernel.normalize_centered(np.zeros((2, 3)), np.zeros(3))
        np.testing.assert_array_equal(zero, 0)
        near = kernel.normalize_centered(np.asarray([[1e-14, 0.]]), np.zeros(2))
        np.testing.assert_allclose(near, [[.01, 0]], atol=1e-17)
        extreme = kernel.normalize_centered(np.asarray([[3e38, -3e38]], dtype=np.float32), np.zeros(2))
        self.assertTrue(np.isfinite(extreme).all())
        self.assertAlmostEqual(float(np.linalg.norm(extreme)), 1.)
        bags = [[np.zeros((1, 2), dtype=np.float32) for _ in range(4)] for _ in range(6)]
        labels = np.asarray([0, 1, 2, 3, 4, 0])
        prepared = kernel.prepare_case(bags, labels, np.arange(5), 5)
        for temperature in kernel.TEMPERATURES:
            heads = kernel.density_logs(prepared, temperature)
            np.testing.assert_allclose(np.exp(heads['direct'][0]), .2)
            np.testing.assert_allclose(np.exp(heads['binary'][0]), .5)
            np.testing.assert_allclose(np.exp(heads['malignant'][0]), .25)
        with self.assertRaises(ValueError):
            kernel.prepare_case(bags, labels, np.arange(5), 2, leave_query_case_out=True)
        with self.assertRaises(ValueError):
            kernel.density_logs(prepared, .02)

    def test_joint_composition_before_pooling(self):
        binary = np.log(np.asarray([[.1, .9], [.9, .1]]))
        malignant = np.log(np.asarray([[.89, .09, .01, .01], [.09, .89, .01, .01]]))
        heads = {'direct': (), 'binary': (binary,) * 4, 'malignant': (malignant,) * 4}
        with patch.object(kernel, 'density_logs', return_value=heads):
            joint = kernel.field_log_probabilities(None, .1, 'dual')
        pooled = pool_log_probabilities(joint, (0,), 'dual')
        expected = np.asarray([.5, .405, .085, .005, .005])
        np.testing.assert_allclose(pooled['probabilities5'], expected, atol=1e-15)
        independent = np.exp(binary).mean(0)[1] * np.exp(malignant).mean(0)
        self.assertFalse(np.allclose(pooled['probabilities5'][1:], independent))

    def test_hard_route_tie_and_summary_json(self):
        bags = [[np.zeros((1, 2), dtype=np.float32) for _ in range(4)] for _ in range(11)]
        labels = np.append(np.repeat(np.arange(5), 2), 0)
        prepared = kernel.prepare_case(bags, labels, np.arange(10), 10)
        direct = pool_log_probabilities(kernel.field_log_probabilities(prepared, .01, 'direct'), (0, 2), 'direct')
        dual = pool_log_probabilities(kernel.field_log_probabilities(prepared, .01, 'dual'), (0, 2), 'dual')
        self.assertEqual(direct['prediction'], 0)
        self.assertAlmostEqual(dual['binary_score'], .5, places=15)
        self.assertEqual(dual['prediction'], 1)
        summary = kernel.support_summary(bags, labels, np.arange(10))
        self.assertEqual(json.loads(json.dumps(summary)), summary)
        self.assertEqual(summary['class_support'], [2] * 5)
        self.assertEqual(summary['head_case_counts']['binary'], [2, 8])
        self.assertEqual(summary['field_counts_by_view'], [10] * 4)


if __name__ == '__main__':
    unittest.main()
