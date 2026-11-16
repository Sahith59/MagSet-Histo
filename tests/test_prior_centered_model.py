"""Scalar centering/density oracles on synthetic fields; no actual cohort or fitting."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from magset import prior_centered_model as m
from magset import kernel_model as old
from magset.field_five import pool_log_probabilities


def fixture():
    labels = np.repeat(np.arange(5), [11, 25, 2, 5, 3])
    origin = np.array([[8, 20, 31], [12, 20, 31], [10, 22, 31], [10, 18, 31], [12, 22, 31]], float)
    bags = []
    for i, label in enumerate(labels):
        views = []
        for view in range(4):
            n = 1+(i+view)%3
            offsets = (np.arange(n)-(n-1)/2)[:, None]*np.array([.02, -.03, .01])
            views.append((origin[label]+[.007*i, -.003*i, .001*i]+view*.02+offsets).astype(np.float32))
        bags.append(views)
    bags.append([np.array([[11.6, 20.1, 30.9], [10.3, 20.5, 31.2]], np.float32)+v*.02 for v in range(4)])
    return bags, np.r_[labels, 2], np.arange(46), 46


def scalar_center(bags, labels, indices, head, view):
    groups = range(5) if head == 'direct' else range(2) if head == 'binary' else range(1, 5)
    result = []
    for group in groups:
        selected = [i for i in indices if (int(labels[i] > 0) if head == 'binary' else int(labels[i])) == group]
        value = np.array([sum(sum(float(x[d]) for x in bags[i][view])/len(bags[i][view]) for i in selected)/len(selected) for d in range(3)])
        result.append(value)
    return np.sum(result, axis=0)/len(result)


def scalar_logs(bags, labels, indices, query, center_head, density_head, view, temperature):
    mu = scalar_center(bags, labels, indices, center_head, view)
    classes = range(5) if density_head == 'direct' else range(2) if density_head == 'binary' else range(1, 5)
    result = []
    def normalized(x):
        a = np.array([float(v) for v in x])-mu
        return a/max(float(np.sqrt(sum(float(v*v) for v in a))), 1e-12)
    for field in bags[query][view]:
        q = normalized(field); values = []
        for label in classes:
            chosen = [i for i in indices if (int(labels[i] > 0) if density_head == 'binary' else int(labels[i])) == label]
            contributions = []
            for i in chosen:
                for x in bags[i][view]:
                    cosine = np.clip(sum(float(a*b) for a, b in zip(q, normalized(x))), -1, 1)
                    contributions.append(cosine/temperature-np.log(len(chosen))-np.log(len(bags[i][view])))
            values.append(logsumexp(contributions))
        result.append(np.asarray(values)-logsumexp(values))
    return np.asarray(result)


class PriorCenteredModelTests(unittest.TestCase):
    def setUp(self): self.bags, self.labels, self.support, self.query = fixture()

    def test_canonical_two_stage_centers_match_scalar_case_population_means(self):
        prepared = m.prepare_case(self.bags, self.labels, self.support, self.query)
        for head in m.HEADS:
            for view in range(4):
                np.testing.assert_allclose(prepared.heads[head].centers[view], scalar_center(self.bags, self.labels, self.support, head, view), rtol=0, atol=1e-13)
        shared = prepared.heads['direct'].centers[0]
        flat_fields = np.concatenate([x[0] for x in self.bags[:46]]).astype(float).mean(0)
        self.assertGreater(np.max(np.abs(shared-flat_fields)), .01)
        self.assertEqual(m.NORMALIZATION, json.loads(Path('configs/breakhis-prior-centered-v1.json').read_text())['normalization'])

    def test_reused_old_densities_match_independent_scalar_case_class_field_oracle(self):
        bundle = m.prepare_case(self.bags, self.labels, self.support, self.query)
        combinations = [('direct', 'direct'), ('direct', 'binary'), ('direct', 'malignant'), ('binary', 'binary'), ('malignant', 'malignant')]
        for temperature in m.TEMPERATURES:
            with patch.object(old, 'density_logs', wraps=old.density_logs) as density:
                cache = m.density_cache(bundle, temperature)
                for mode in m.MODES:
                    for arm in ('direct', 'dual'): m.field_log_probabilities(bundle, temperature, arm, mode, cache=cache)
                self.assertEqual(density.call_count, 3)
            for center_head, density_head in combinations:
                for view in range(4):
                    expected = scalar_logs(self.bags, self.labels, self.support, self.query, center_head, density_head, view, temperature)
                    # Scalar and vectorized FP64 means/norms reduce in different orders.
                    # At T=.01 their ~1e-14 cosine rounding is amplified in large
                    # negative logs; retain the absolute bound near zero and add
                    # a 64-epsilon, magnitude-scaled roundoff allowance.
                    np.testing.assert_allclose(cache.logs[center_head][density_head][view], expected,
                                               rtol=64*np.finfo(np.float64).eps, atol=1e-12)
            for head in m.HEADS:
                for view in bundle.heads[head].views:
                    np.testing.assert_allclose(view.weights.sum(0), np.ones(11), rtol=0, atol=1e-14)

    def test_whole_case_exclusion_precedes_centers_and_query_truth_is_unused(self):
        for query in self.support:
            bundle = m.prepare_case(self.bags, self.labels, self.support, int(query), leave_query_case_out=True)
            self.assertNotIn(query, bundle.support_indices)
            self.assertEqual(len(bundle.support_indices), 45)
            for head in m.HEADS:
                np.testing.assert_allclose(bundle.heads[head].centers[0], scalar_center(self.bags, self.labels, bundle.support_indices, head, 0), rtol=0, atol=1e-13)
        for query, loo in [(self.query, False), (36, True)]:
            before = m.prepare_case(self.bags, self.labels, self.support, query, leave_query_case_out=loo)
            changed = self.labels.copy(); changed[query] = (changed[query]+2)%5
            after = m.prepare_case(self.bags, changed, self.support, query, leave_query_case_out=loo)
            for mode in m.MODES:
                for arm in ('direct', 'dual'):
                    for a, b in zip(m.field_log_probabilities(before, .1, arm, mode), m.field_log_probabilities(after, .1, arm, mode)):
                        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError): m.prepare_case(self.bags, self.labels, self.support, 0)
        with self.assertRaises(ValueError): m.prepare_case(self.bags, self.labels, self.support, self.query, leave_query_case_out=True)

    def test_conditional_expert_ignores_benign_values_but_shared_expert_can_change(self):
        original = m.prepare_case(self.bags, self.labels, self.support, self.query)
        changed = copy.deepcopy(self.bags)
        for i in self.support[self.labels[self.support] == 0]:
            changed[i] = [np.repeat(x+[100+i, -60-i, 40], 2, axis=0) for x in changed[i]]
        modified = m.prepare_case(changed, self.labels, self.support, self.query)
        for a, b in zip(original.heads['malignant'].centers, modified.heads['malignant'].centers): np.testing.assert_array_equal(a, b)
        for temperature in m.TEMPERATURES:
            before = m.density_cache(original, temperature); after = m.density_cache(modified, temperature)
            for a, b in zip(before.logs['malignant']['malignant'], after.logs['malignant']['malignant']):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        difference = max(np.max(np.abs(a-b)) for a, b in zip(m.density_cache(original, .1).logs['direct']['malignant'], m.density_cache(modified, .1).logs['direct']['malignant']))
        self.assertGreater(difference, 1e-4)

    def test_direct_identity_duplication_permutation_and_available_subset_invariance(self):
        bundle = m.prepare_case(self.bags, self.labels, self.support, self.query); cache = m.density_cache(bundle, .1)
        direct = [m.field_log_probabilities(bundle, .1, 'direct', mode, cache=cache) for mode in m.MODES]
        self.assertIs(direct[0], direct[1])
        changed = copy.deepcopy(self.bags); changed[0][0] = np.repeat(changed[0][0], 3, axis=0)
        changed[self.query] = [x[::-1] for x in changed[self.query]]
        other = m.prepare_case(changed, self.labels, self.support[::-1], self.query)
        for mode in m.MODES:
            for arm in ('direct', 'dual'):
                logs = m.field_log_probabilities(bundle, .1, arm, mode)
                expected = pool_log_probabilities(logs, (0, 2), arm)
                actual = pool_log_probabilities(m.field_log_probabilities(other, .1, arm, mode), (0, 2), arm)
                np.testing.assert_allclose(expected['probabilities5'], actual['probabilities5'], rtol=0, atol=1e-12)
                missing = list(logs); missing[1] = np.full((1, 5), np.nan); missing[3] = None
                np.testing.assert_array_equal(expected['probabilities5'], pool_log_probabilities(missing, (2, 0), arm)['probabilities5'])

    def test_old_zero_policy_four_temperatures_cache_guards_and_serializable_support(self):
        bundle = m.prepare_case(self.bags, self.labels, self.support, self.query)
        cache = m.density_cache(bundle, .1)
        with self.assertRaises(ValueError): m.field_log_probabilities(bundle, .3, 'dual', 'head_specific', cache=cache)
        other = m.prepare_case(self.bags, self.labels, self.support, self.query)
        with self.assertRaises(ValueError): m.field_log_probabilities(other, .1, 'dual', 'head_specific', cache=cache)
        with self.assertRaises(ValueError): m.density_cache(bundle, .5)
        with self.assertRaises(ValueError): m.field_log_probabilities(bundle, .1, 'dual', 'unregistered')
        zero = old.normalize_centered(np.array([[0., 0., 0.], [1e-14, 0., 0.]], np.float32), np.zeros(3))
        np.testing.assert_array_equal(zero[0], np.zeros(3)); self.assertAlmostEqual(zero[1, 0], .01, places=8)
        summary = m.support_summary(self.bags, self.labels, self.support)
        self.assertEqual(summary['class_support'], [11, 25, 2, 5, 3])
        self.assertEqual(summary['head_case_counts']['binary'], [11, 35])
        self.assertEqual(set(summary['centers_sha256']), set(m.HEADS))
        self.assertEqual(summary['mode_used_center_mapping']['class_shared']['malignant'], 'direct')
        self.assertEqual(m.support_summary(self.bags, self.labels, self.support, 'head_specific')['mode_used_center_mapping']['malignant'], 'malignant')
        json.dumps(summary, allow_nan=False)


if __name__ == '__main__': unittest.main()
