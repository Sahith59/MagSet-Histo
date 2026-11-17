"""Synthetic case-SVC preprocessing, weights, scores and serialization checks."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.metrics import f1_score

from magset import uni_case_svm as m


def fixture():
    rng = np.random.RandomState(71)
    labels = np.arange(20, dtype=np.int64) % 5
    z = rng.normal(size=(20, 4, 6)).astype(np.float64)
    z[:, :, 0] += 3 * labels[:, None]
    z[:, :, 5] = 7.
    train = np.arange(10, dtype=np.int64)
    return z, labels, train


class UniCaseSVMTests(unittest.TestCase):
    def test_equal_field_means_and_population_training_only_statistics(self):
        z, _, train = fixture()
        bags = [[np.stack((value - 1, value + 1)).astype(np.float32) for value in views] for views in z]
        means = m.case_view_means(bags)
        np.testing.assert_allclose(means, z, rtol=0, atol=5e-7)
        p = m.fit_preprocessing(z, train)
        np.testing.assert_array_equal(p['mean'], z[train].mean(0))
        np.testing.assert_array_equal(p['std'], z[train].std(0, ddof=0))
        np.testing.assert_array_equal(p['scale'][:, 5], np.ones(4))
        changed = z.copy(); changed[10:] += 10000.
        q = m.fit_preprocessing(changed, train)
        for name in ('mean', 'std', 'scale'): np.testing.assert_array_equal(p[name], q[name])

    def test_active_mean_then_l2_then_half_flags_without_final_normalization(self):
        z, _, train = fixture(); p = m.fit_preprocessing(z, train)
        actual = m.vectorize(z, p, np.array([10], np.int64))[0]
        for index, subset in enumerate(m.SUBSETS):
            v = ((z[10, list(subset)] - p['mean'][list(subset)]) / p['scale'][list(subset)]).mean(0)
            v /= max(np.linalg.norm(v), 1e-12)
            np.testing.assert_allclose(actual[index, :6], v, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(actual[index, 6:], .5 * m.MASKS[index])
        self.assertGreater(np.linalg.norm(actual[0]), 1.)
        changed = z.copy(); changed[10, 1:] += 1e9
        np.testing.assert_array_equal(m.vectorize(changed, p, np.array([10], np.int64))[0, 0], actual[0])
        self.assertEqual(m.SUBSETS[:5], ((0,), (1,), (2,), (3,), (0, 1)))

    def test_balanced_mask_weights_train_supports_and_separate_gamma(self):
        z, labels, train = fixture(); p = m.fit_preprocessing(z, train)
        bundle = m.fit_bundle(z, labels, p, 'rbf', 1.)
        for name, expected_cases in [('binary', train), ('malignant', train[labels[train] > 0])]:
            model = bundle[name]; receipt = bundle['training_receipts'][name]
            X = m.vectorize(z, p, expected_cases).reshape(len(expected_cases) * 15, 10)
            self.assertEqual(model.get_params()['class_weight'], 'balanced')
            self.assertFalse(model.get_params()['probability'])
            self.assertEqual(receipt['rows'], len(expected_cases) * 15)
            self.assertAlmostEqual(receipt['sample_weight_sum'], len(expected_cases), places=12)
            self.assertAlmostEqual(receipt['effective_weight_sum'], len(expected_cases), places=12)
            self.assertEqual(receipt['training_features_sha256'], m.array_hash(X))
            self.assertEqual(model._gamma, 1. / (X.shape[1] * X.var()))
            self.assertEqual(model.fit_status_, 0)
            np.testing.assert_array_equal(model.support_vectors_, X[model.support_])
        arrays = m.state_arrays(bundle)
        for name in ('binary', 'malignant'):
            np.testing.assert_array_equal(arrays[name + '_dual_coef_'], bundle[name].dual_coef_)
            self.assertTrue(set(arrays[name + '_support_case_indices'].tolist()) <= set(train.tolist()))
            self.assertFalse(set(arrays[name + '_support_case_indices'].tolist()) & set(range(10, 20)))

    def test_both_kernels_joblib_reload_all_validation_scores(self):
        z, labels, train = fixture(); p = m.fit_preprocessing(z, train)
        with tempfile.TemporaryDirectory() as tmp:
            for kernel in m.ARMS:
                bundle = m.fit_bundle(z, labels, p, kernel, .1)
                before = m.predict(bundle, z, np.arange(10, 15, dtype=np.int64))
                path = Path(tmp) / (kernel + '.joblib'); joblib.dump(bundle, path)
                restored = joblib.load(path); after = m.predict(restored, z, np.arange(10, 15, dtype=np.int64))
                for name in before: np.testing.assert_array_equal(before[name], after[name])

    def test_margin_zero_and_subtype_ties_are_canonical_and_tiny_rounding_fails(self):
        margin = np.zeros((3, 15), np.float64); margin[0] = -1.; margin[2] = 1.
        scores = np.zeros((3, 15, 4), np.float64)
        joint, pred = m.score_arrays(margin, scores)
        np.testing.assert_array_equal(pred[:, 0], [0, 1, 1])
        np.testing.assert_allclose(joint.sum(-1), 1., rtol=0, atol=1e-14)
        margin[0, 0] = -1e-20
        with self.assertRaisesRegex(ValueError, 'authoritative binary margin'): m.score_arrays(margin, scores)
        extreme = np.full((1, 15), 1000.)
        self.assertTrue(np.isfinite(m.score_arrays(extreme, np.zeros((1, 15, 4)))[0]).all())

    def test_records_and_summary_count_cases_masks_and_explicit_routes(self):
        truth8 = (0, 1, 3, 4, 5)
        cases = [dict(case_id='synthetic%d' % i, truth5=i, truth8=truth8[i]) for i in range(5)]
        margin = np.ones((5, 15)); margin[0] = -1.
        scores = np.zeros((5, 15, 4))
        for i in range(1, 5): scores[i, :, i - 1] = 5.
        joint, prediction = m.score_arrays(margin, scores)
        archive = {'case_indices': np.arange(5, dtype=np.int64), 'masks': m.MASKS.copy(), 'binary_margin': margin,
                   'malignant_scores': scores, 'joint_probabilities5': joint, 'predictions': prediction}
        records = m.case_records(cases, archive, 'validation'); report = m.summary(cases, archive)
        self.assertEqual(len(records), 75); self.assertEqual(records[15]['case_id'], 'synthetic1')
        self.assertEqual(records[14]['magnifications'], [40, 100, 200, 400])
        self.assertEqual(report['cases'], 5); self.assertEqual(report['mean_subset_accuracy'], 1.)
        self.assertEqual(report['mean_subset_macro_f1'], 1.)
        scores[3, :, 0] = 10.; archive['malignant_scores'] = scores
        archive['joint_probabilities5'], archive['predictions'] = m.score_arrays(margin, scores)
        expected = f1_score(np.arange(5), archive['predictions'][:, 0], labels=np.arange(5), average='macro')
        self.assertAlmostEqual(m.summary(cases, archive)['mean_subset_macro_f1'], expected, places=14)

    def test_invalid_candidate_and_missing_fitting_class_rejected(self):
        z, labels, train = fixture(); p = m.fit_preprocessing(z, train)
        for kernel, C in [('other', 1.), ('linear', .3), ('rbf', True)]:
            with self.assertRaises(ValueError): m.fit_bundle(z, labels, p, kernel, C)
        bad = labels.copy(); bad[train] = 0
        with self.assertRaisesRegex(ValueError, 'All five'): m.fit_bundle(z, bad, p, 'linear', 1.)


if __name__ == '__main__':
    unittest.main()
