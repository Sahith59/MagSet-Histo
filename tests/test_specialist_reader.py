"""Independent scalar KDE, parent identity and fit-only specialist checks."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from magset import specialist_reader as m
from magset import prior_centered_model as parent


def fixture():
    rng = np.random.RandomState(83)
    labels = np.array([0, 0, 1, 1, 1, 2, 3, 4, 4, 0], dtype=np.int64)
    pbags = []; ubags = []
    for case, label in enumerate(labels):
        pbags.append([rng.normal(size=(1 + (case + view) % 3, 7)).astype(np.float32) + label
                      for view in range(4)])
        ubags.append([rng.normal(size=(1 + (case + view) % 3, 5)).astype(np.float32) + label
                      for view in range(4)])
    return pbags, ubags, labels, list(range(8)), 8


def scalar_normalize(value, center):
    delta = [float(x) - float(c) for x, c in zip(value, center)]
    norm = max(math.sqrt(sum(x * x for x in delta)), 1e-12)
    return [x / norm for x in delta]


class SpecialistReaderTests(unittest.TestCase):
    def test_center_equal_fitting_cases_and_malignant_support_class_mass(self):
        _, bags, labels, fit, _ = fixture()
        support = m.prepare_support(bags, labels, fit)
        for view in range(4):
            expected = np.mean([np.asarray(bags[i][view], np.float64).mean(0) for i in fit], axis=0)
            np.testing.assert_array_equal(support.centers[view], expected)
            np.testing.assert_allclose(support.weights[view].sum(0), 1, rtol=0, atol=1e-14)
            self.assertEqual(len(support.features[view]), sum(len(bags[i][view]) for i in fit if labels[i] > 0))
            self.assertFalse(support.centers[view].flags.writeable)
        self.assertEqual(support.malignant_indices, (2, 3, 4, 5, 6, 7))
        summary = m.support_summary(support)
        self.assertEqual(summary['fitting_class_counts'], [2, 3, 1, 1, 1])
        self.assertEqual(summary, m.support_summary(m.prepare_support(bags, labels, fit[::-1])))

    def test_uni_outputs_match_independent_scalar_class_case_field_density(self):
        pbags, ubags, labels, fit, query = fixture()
        prepared = m.prepare_query(pbags, ubags, labels, fit, query)
        for temperature in m.TEMPERATURES:
            actual = m.field_probabilities(prepared, temperature, 'uni2_c')
            for view in range(4):
                # Independent loops define equal-case center and class-balanced density.
                center = [sum(sum(float(row[d]) for row in ubags[i][view]) / len(ubags[i][view]) for i in fit) / len(fit)
                          for d in range(5)]
                for row, raw in enumerate(ubags[query][view]):
                    q = scalar_normalize(raw, center); density = []
                    for label in range(1, 5):
                        cases = [i for i in fit if labels[i] == label]
                        value = 0.
                        for i in cases:
                            field_density = 0.
                            for point in ubags[i][view]:
                                x = scalar_normalize(point, center)
                                cosine = min(1., max(-1., sum(a * b for a, b in zip(q, x))))
                                field_density += math.exp(cosine / temperature)
                            value += field_density / len(ubags[i][view])
                        density.append(value / len(cases))
                    conditional = np.array(density) / sum(density)
                    binary = np.exp(prepared.fixed_binary_logs[view][row])
                    expected = np.r_[binary[0], binary[1] * conditional]
                    np.testing.assert_allclose(actual[view][row], expected, rtol=0, atol=5e-14)

    def test_control_identity_preserves_parent_probability_arithmetic_exactly(self):
        pbags, ubags, labels, fit, query = fixture()
        prepared = m.prepare_query(pbags, ubags, labels, fit, query)
        bundle = parent.prepare_case(pbags, labels, fit, query, leave_query_case_out=False)
        cache = parent.density_cache(bundle, .1)
        logs = parent.field_log_probabilities(bundle, .1, 'dual', 'head_specific', cache=cache)
        actual = m.field_probabilities(prepared, .1, 'p_control')
        for left, right in zip(actual, logs):
            expected = np.exp(right - logsumexp(right, axis=1, keepdims=True))
            np.testing.assert_array_equal(left, expected)

    def test_router_marginal_is_fixed_for_every_arm_and_temperature(self):
        pbags, ubags, labels, fit, query = fixture()
        prepared = m.prepare_query(pbags, ubags, labels, fit, query)
        for arm in m.ARMS:
            for temperature in m.TEMPERATURES:
                joint = m.field_log_probabilities(prepared, temperature, arm)
                binary, malignant = m.decompose_joint_logs(joint)
                for b, reference, subtype in zip(binary, prepared.fixed_binary_logs, malignant):
                    np.testing.assert_allclose(b, reference, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(logsumexp(subtype, axis=1), 0, rtol=0, atol=1e-12)
        changed = [values.copy() for values in prepared.fixed_binary_logs]
        changed[0][:, 0] -= .01
        with self.assertRaisesRegex(ValueError, 'binary router'):
            m.field_log_probabilities(prepared, .03, 'uni2_c', changed)

    def test_query_truth_and_other_held_case_features_do_not_fit_statistics_or_predictions(self):
        pbags, ubags, labels, fit, query = fixture()
        support = m.prepare_support(ubags, labels, fit)
        original = m.prepare_query(pbags, ubags, labels, fit, query, support)
        changed_labels = labels.copy(); changed_labels[query] = 0; changed_labels[9] = 3
        changed_p = copy.deepcopy(pbags); changed_u = copy.deepcopy(ubags)
        for view in range(4):
            changed_p[9][view] += 1000.; changed_u[9][view] -= 1000.
        changed_support = m.prepare_support(changed_u, changed_labels, fit)
        self.assertEqual(m.support_summary(support), m.support_summary(changed_support))
        changed = m.prepare_query(changed_p, changed_u, changed_labels, fit, query, changed_support)
        for arm in m.ARMS:
            for a, b in zip(m.field_probabilities(original, .03, arm), m.field_probabilities(changed, .03, arm)):
                np.testing.assert_array_equal(a, b)

    def test_zero_norm_support_stays_finite_uniform_and_query_permutations_preserved(self):
        labels = np.arange(10, dtype=np.int64) % 5
        bags = [[np.ones((2, 4), np.float32) for _ in range(4)] for _ in range(10)]
        prepared = m.prepare_query(bags, bags, labels, list(range(5)), 5)
        for arm in m.ARMS:
            for temperature in m.TEMPERATURES:
                for values in m.field_probabilities(prepared, temperature, arm):
                    np.testing.assert_allclose(values, [[.5, .125, .125, .125, .125]] * 2, rtol=0, atol=1e-14)
        pbags, ubags, labels, fit, query = fixture()
        before = m.prepare_query(pbags, ubags, labels, fit, query)
        for view in range(4):
            pbags[query][view] = pbags[query][view][::-1].copy()
            ubags[query][view] = ubags[query][view][::-1].copy()
        after = m.prepare_query(pbags, ubags, labels, fit, query)
        for arm in m.ARMS:
            for a, b in zip(m.field_probabilities(before, .01, arm), m.field_probabilities(after, .01, arm)):
                np.testing.assert_allclose(a[::-1], b, rtol=0, atol=1e-12)

    def test_alignment_rejects_path_order_byte_provider_and_view_changes(self):
        rows = [dict(relative_path='field%d' % i, sha256=('%064x' % i),
                     provider_key='specimen%d' % i, view=i, magnification=str(mag))
                for i, mag in enumerate((40, 100, 200, 400))]
        self.assertEqual(m.assert_field_alignment(rows, copy.deepcopy(rows)), 4)
        for key, value in [('relative_path', 'other'), ('sha256', 'f' * 64),
                           ('provider_key', 'other'), ('magnification', '100')]:
            changed = copy.deepcopy(rows); changed[0][key] = value
            with self.assertRaises(ValueError):
                m.assert_field_alignment(rows, changed)
        with self.assertRaises(ValueError): m.assert_field_alignment(rows, rows[::-1])
        with self.assertRaises(ValueError): m.assert_field_alignment(rows + rows[:1], rows + rows[:1])

    def test_rejects_fitting_queries_missing_classes_and_unregistered_choices(self):
        pbags, ubags, labels, fit, query = fixture()
        with self.assertRaisesRegex(ValueError, 'outside'):
            m.prepare_query(pbags, ubags, labels, fit, 0)
        with self.assertRaisesRegex(ValueError, 'all five'):
            m.prepare_support(ubags, labels, fit[:-1])
        prepared = m.prepare_query(pbags, ubags, labels, fit, query)
        for temperature, arm in [(.02, 'uni2_c'), (True, 'uni2_c'), (.03, 'other')]:
            with self.assertRaises(ValueError): m.field_log_probabilities(prepared, temperature, arm)

    def test_canonical_reference_skips_unselected_parent_specialist_query(self):
        pbags, ubags, labels, fit, query = fixture()
        original = m.prepare_query(pbags, ubags, labels, fit, query)
        canonical = original.reference_joint_logs
        with patch.object(parent, 'density_cache', wraps=parent.density_cache) as density:
            prepared = m.prepare_query(pbags, ubags, labels, fit, query,
                                      parent_reference_joint_logs=canonical)
            values = m.field_probabilities(prepared, .03, 'uni2_c')
            density.assert_not_called()
        for actual, fixed in zip(values, prepared.fixed_binary_logs):
            marginal = np.column_stack((actual[:, 0], actual[:, 1:].sum(1)))
            np.testing.assert_allclose(marginal, np.exp(fixed), rtol=0, atol=1e-12)
        with patch.object(parent, 'density_cache', wraps=parent.density_cache) as density:
            m.field_probabilities(prepared, .03, 'p_control')
            self.assertEqual([call.args[1] for call in density.call_args_list], [.03])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            m.prepare_query(pbags, ubags, labels, fit, query, parent_reference_joint_logs=canonical[:-1])


if __name__ == '__main__':
    unittest.main()
