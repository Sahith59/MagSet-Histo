"""Analytic metric fixtures; no cohort images, cached cohort features or jobs."""
import io
import unittest

import numpy as np
from scipy.special import logsumexp

from magset import specialist_metric as metric

try:
    import torch
except ImportError:
    torch = None


def fixture():
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    rng = np.random.default_rng(41)
    x = rng.normal(size=(len(y), 4, 7))
    x[:, :, 0] += y[:, None] * .4
    return x, y


class SpecialistMetricNumpyTests(unittest.TestCase):
    def test_identity_preserves_original_normalized_field_geometry(self):
        rng = np.random.default_rng(4)
        fields = rng.normal(size=(9, 7))
        fields /= np.linalg.norm(fields, axis=1, keepdims=True)
        transformed = metric.transform_vectors(fields, np.ones(7))
        np.testing.assert_array_equal(transformed, fields)
        np.testing.assert_array_equal(transformed @ transformed.T, fields @ fields.T)
        self.assertIsNot(transformed, fields)

    def test_bounded_diagonal_map_and_zero_vectors(self):
        a, factors = metric.factors_from_theta(np.asarray([-1e5, 0., 1e5]))
        np.testing.assert_array_equal(a, [-.5, 0., .5])
        np.testing.assert_array_equal(factors, np.exp(a))
        z = metric.transform_vectors(np.asarray([[0., 0., 0.], [1., 2., 3.]]), factors)
        np.testing.assert_array_equal(z[0], np.zeros(3))
        self.assertAlmostEqual(float(np.linalg.norm(z[1])), 1.)

    def test_class_density_matches_explicit_cross_case_oracle_for_both_heads(self):
        x, y = fixture()
        theta = np.linspace(-.3, .4, x.shape[-1])
        for head in ('binary', 'malignant'):
            value = metric.numpy_objective(x, y, .03, head=head, theta=theta)
            indices = np.arange(len(y)) if head == 'binary' else np.flatnonzero(y)
            labels = (y > 0).astype(int) if head == 'binary' else y[indices] - 1
            center = x.mean(0)
            np.testing.assert_array_equal(value['centers'], center)
            centered = x[indices] - center
            u = centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)
            a = .5 * np.tanh(theta)
            z = u * np.exp(a)
            z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
            expected = np.empty_like(value['head_log_probabilities'])
            for c in range(len(indices)):
                for view in range(4):
                    densities = []
                    for label in range(expected.shape[-1]):
                        support = [j for j in range(len(indices)) if j != c and labels[j] == label]
                        cosines = np.asarray([np.clip(z[c, view] @ z[j, view], -1., 1.) for j in support])
                        densities.append(logsumexp(cosines / .03) - np.log(len(support)))
                    expected[c, view] = densities - logsumexp(densities)
            np.testing.assert_allclose(value['head_log_probabilities'], expected, rtol=1e-13, atol=1e-13)
            ce_by_case = -expected[np.arange(len(indices)), :, labels].mean(1)
            expected_ce = np.mean([ce_by_case[labels == k].mean() for k in range(expected.shape[-1])])
            self.assertAlmostEqual(value['ce'], expected_ce, places=13)
            self.assertAlmostEqual(value['penalty'], 10 * np.mean(a ** 2), places=14)

    def test_case_and_view_permutations_preserve_class_balanced_objective(self):
        x, y = fixture()
        p = np.random.default_rng(5).permutation(len(y))
        for head in ('binary', 'malignant'):
            before = metric.numpy_objective(x, y, .1, head=head, theta=np.arange(7) * .04)
            after = metric.numpy_objective(x[p][:, [2, 0, 3, 1]], y[p], .1, head=head, theta=np.arange(7) * .04)
            self.assertAlmostEqual(before['loss'], after['loss'], places=13)

    def test_invalid_input_and_singleton_positive_support_fail(self):
        x, y = fixture()
        for bad_t in (True, .2, np.nan):
            with self.assertRaises(ValueError):
                metric.numpy_objective(x, y, bad_t)
        bad = x.copy(); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            metric.numpy_objective(bad, y, .1)
        with self.assertRaises(ValueError):
            metric.numpy_objective(x[:-1], y[:-1], .1)
        with self.assertRaises(ValueError):
            metric.numpy_objective(x, y, .1, head='direct')
        with self.assertRaises(ValueError):
            metric.transform_vectors(x, np.full(7, 2.))
        with self.assertRaises(ValueError):
            metric.numpy_objective(x, y, .1, theta=np.zeros(7), factors=np.ones(7))

    def test_exact_serialized_theta_recreates_float64_metric(self):
        theta = np.linspace(-.45, .35, 7).astype(np.float32)
        a, factors = metric.factors_from_theta(theta)
        buffer = io.BytesIO()
        np.savez(buffer, theta=theta, log_factors=a, factors=factors)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as state:
            re_a, re_factors = metric.factors_from_theta(state['theta'])
            np.testing.assert_array_equal(re_a, state['log_factors'])
            np.testing.assert_array_equal(re_factors, state['factors'])


@unittest.skipIf(torch is None, 'Torch is required for supervised metric fitting checks')
class SpecialistMetricTorchTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_finite_gradients_match_independent_numerical_oracle(self):
        x, y = fixture()
        for head in ('binary', 'malignant'):
            u, labels, _, _, _ = metric._inputs(x, y, .1, head)
            theta = torch.tensor(np.linspace(-.15, .1, 7), dtype=torch.float64, requires_grad=True)
            result = metric.torch_objective(torch.from_numpy(u), torch.from_numpy(labels), .1, theta, head=head)
            expected = metric.numpy_objective(x, y, .1, theta=theta.detach().numpy(), head=head)
            self.assertAlmostEqual(float(result['loss']), expected['loss'], places=12)
            np.testing.assert_allclose(result['head_log_probabilities'].detach().numpy(), expected['head_log_probabilities'], atol=3e-14, rtol=1e-13)
            result['loss'].backward()
            self.assertTrue(torch.isfinite(theta.grad).all())
            numerical = []
            for column in range(7):
                plus = theta.detach().numpy().copy(); minus = plus.copy()
                plus[column] += 1e-5; minus[column] -= 1e-5
                numerical.append((metric.numpy_objective(x, y, .1, theta=plus, head=head)['loss']
                                  - metric.numpy_objective(x, y, .1, theta=minus, head=head)['loss']) / 2e-5)
            np.testing.assert_allclose(theta.grad.numpy(), numerical, rtol=2e-7, atol=2e-9)

    def test_fixed_fit_is_reproducible_bounded_and_replayable_for_both_heads(self):
        x, y = fixture()
        for head in ('binary', 'malignant'):
            first = metric.fit_metric(x, y, .1, head=head)
            second = metric.fit_metric(x, y, .1, head=head)
            self.assertEqual(first['theta'].dtype, np.float32)
            self.assertEqual(first['factors'].dtype, np.float64)
            np.testing.assert_array_equal(first['theta'], second['theta'])
            self.assertEqual(first['history'], second['history'])
            self.assertEqual(len(first['history']), 200)
            self.assertEqual([v['step'] for v in first['history']], list(range(1, 201)))
            self.assertTrue(all(np.isfinite(v['gradient_l2']) for v in first['history']))
            self.assertTrue(np.all(first['factors'] >= np.exp(-.5)))
            self.assertTrue(np.all(first['factors'] <= np.exp(.5)))
            final = metric.numpy_objective(x, y, .1, head=head, factors=first['factors'])
            np.testing.assert_array_equal(first['factors'], first['factors'].astype(np.float32).astype(np.float64))
            a, d = metric.factors_from_theta(first['theta'])
            np.testing.assert_allclose(first['log_factors'], a, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(first['factors'], d, atol=2e-7, rtol=1e-6)
            self.assertEqual(first['receipt']['final_numpy_objective']['loss'], final['loss'])
            self.assertLess(final['loss'], first['receipt']['initial_numpy_objective']['loss'])
            self.assertEqual(first['receipt']['head'], head)
            self.assertEqual(first['receipt']['fitting_diagnostic'], metric.FITTING_DIAGNOSTIC)
            self.assertEqual(first['receipt']['validation_selection_opportunities'], 0)
            self.assertAlmostEqual(first['receipt']['final_torch_objective']['loss'], final['loss'], places=5)
            buffer = io.BytesIO()
            np.savez(buffer, theta=first['theta'], log_factors=first['log_factors'], factors=first['factors'])
            buffer.seek(0)
            with np.load(buffer, allow_pickle=False) as saved:
                replay = metric.numpy_objective(x, y, .1, head=head, factors=saved['factors'])
                self.assertEqual(replay['loss'], final['loss'])
                np.testing.assert_array_equal(replay['head_log_probabilities'], final['head_log_probabilities'])
                np.testing.assert_array_equal(saved['theta'], first['theta'])

    def test_zero_vectors_have_finite_loss_gradient_and_identity_state(self):
        x, y = fixture(); x[:] = 1.
        for head in ('binary', 'malignant'):
            fitted = metric.fit_metric(x, y, .01, head=head)
            np.testing.assert_array_equal(fitted['theta'], np.zeros(7, dtype=np.float32))
            np.testing.assert_array_equal(fitted['factors'], np.ones(7))
            self.assertAlmostEqual(fitted['receipt']['final_numpy_objective']['loss'], np.log(2 if head == 'binary' else 4))

    def test_fitting_restores_callers_torch_determinism_and_random_state(self):
        x, y = fixture()
        random_state = torch.random.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        metric.fit_metric(x, y, .1)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)
        torch.testing.assert_close(torch.random.get_rng_state(), random_state, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
