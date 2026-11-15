"""Bounded coefficient/role fixtures; no real-data or synthetic classifier fitting."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from magset import five_class as m
from magset.bags import fit_normalization
from magset.five_metrics import summarize_records


def coefficient_models(arm, dimension=3):
    """Assign coefficients directly; sklearn is used only as a prediction oracle."""
    names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
    result = {}
    for name in names:
        model = LogisticRegression()
        count = 1 if name == 'binary' else 5 if name == 'direct' else 4
        model.classes_ = np.arange(2) if name == 'binary' else np.arange(5) if name == 'direct' else np.arange(1, 5)
        model.coef_ = np.arange(count*(dimension+4), dtype=np.float64).reshape(count, dimension+4)/100
        model.intercept_ = np.linspace(-.3, .3, count, dtype=np.float64)
        model.n_features_in_ = dimension+4
        result[name] = model
    return result


class FiveClassRunnerTests(unittest.TestCase):
    def normalizer(self):
        return np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)

    def test_saved_direct_and_binary_heads_replay_sklearn_without_fitting(self):
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]
        z = np.arange(8*4*3, dtype=np.float32).reshape(8, 4, 3)/100
        mean, std = self.normalizer()
        with tempfile.TemporaryDirectory() as tmp, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')):
            for arm in ('direct', 'dual'):
                models = coefficient_models(arm)
                path = Path(tmp)/(arm+'.npz')
                m.save_model(path, mean, std, models)
                saved = m.load_model(path, arm, dimension=3)
                records, error = m.score_records(saved, z, np.arange(8), cases, 'validation', arm, models)
                self.assertEqual(len(records), 120)
                self.assertLess(error, 1e-12)
                summary = summarize_records(records, arm)
                self.assertEqual(summary['distinct_class_support'], [4, 1, 1, 1, 1])
                if arm == 'dual':
                    self.assertEqual(saved['binary_coef'].shape, (1, 7))
                    self.assertEqual(saved['binary_classes'].tolist(), [0, 1])

    def test_strict_saved_schema_rejects_wrong_binary_shape_float_classes_and_normalizer(self):
        mean, std = self.normalizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            m.save_model(path, mean, std, coefficient_models('dual'))
            with np.load(path, allow_pickle=False) as source:
                original = {key: source[key] for key in source.files}
            changes = [('binary_coef', np.zeros((2, 7))),
                       ('binary_classes', np.array([0., 1.])),
                       ('std', np.zeros((4, 3), np.float32)),
                       ('mean', np.zeros((4, 3), np.float64)),
                       ('malignant_classes', np.array([1, 3, 2, 4]))]
            for key, value in changes:
                with self.subTest(key=key):
                    arrays = dict(original); arrays[key] = value
                    np.savez(path, **arrays)
                    with self.assertRaises(ValueError):
                        m.load_model(path, 'dual', dimension=3)

    def test_hard_route_can_disagree_with_composed_argmax_and_ties_route_malignant(self):
        arrays = {'binary_coef': np.zeros((1, 7)), 'binary_intercept': np.array([np.log(.6/.4)]),
                  'malignant_coef': np.zeros((4, 7)), 'malignant_intercept': np.zeros(4)}
        scores = m.scores_from_arrays(arrays, np.zeros((1, 7)), 'dual')
        self.assertEqual(int(scores['prediction'][0]), 1)
        self.assertEqual(int(scores['probabilities5'][0].argmax()), 0)
        arrays['binary_intercept'] = np.zeros(1)
        self.assertEqual(int(m.scores_from_arrays(arrays, np.zeros((1, 7)), 'dual')['prediction'][0]), 1)
        with self.assertRaises(ValueError):
            m.scores_from_arrays(arrays, np.zeros((1, 7)), 'unknown')

    def test_stable_extreme_scores_and_coefficient_mutation_replay_failure(self):
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]
        z = np.zeros((8, 4, 3), np.float32); mean, std = self.normalizer()
        models = coefficient_models('direct')
        models['direct'].coef_[:] = 0
        models['direct'].intercept_[:] = [0, -1000, -1000, -1000, -1000]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'; m.save_model(path, mean, std, models)
            arrays = m.load_model(path, 'direct', dimension=3)
            records, _ = m.score_records(arrays, z, np.arange(8), cases, 'validation', 'direct', models)
            self.assertEqual(records[1]['probabilities5'][1], 0)
            self.assertEqual(records[1]['log_probabilities5'][1], -1000)
            self.assertEqual(summarize_records(records, 'direct')['mean_subset_cross_entropy'], 500)
            arrays['direct_intercept'] = np.zeros(5)
            with self.assertRaisesRegex(ValueError, 'replay differs'):
                m.score_records(arrays, z, np.arange(8), cases, 'validation', 'direct', models)

    def test_forbidden_roles_duplicate_negative_or_out_of_range_indices_fail(self):
        mean, std = self.normalizer()
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]
        z = np.zeros((8, 4, 3), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'; m.save_model(path, mean, std, coefficient_models('direct'))
            arrays = m.load_model(path, 'direct', dimension=3)
            for role in ('test', 'excluded', 'anything'):
                with self.subTest(role=role), self.assertRaises(ValueError):
                    m.score_records(arrays, z, [0], cases, role, 'direct')
            for indices in ([0, 0], [-1], [8], [], [0.5]):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    m.score_records(arrays, z, indices, cases, 'validation', 'direct')

    def test_specialist_uses_only_true_malignant_rows_with_shared_C(self):
        x = np.arange(6*7).reshape(6, 7); y = np.array([0, 1, 2, 3, 4, 0]); config = {}
        with patch.object(m, 'head_fit', side_effect=[object(), object()]) as fit:
            m.fit_arm(x, y, 'dual', .1, config)
        self.assertEqual(fit.call_count, 2)
        np.testing.assert_array_equal(fit.call_args_list[0].args[0], x)
        np.testing.assert_array_equal(fit.call_args_list[0].args[1], [0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(fit.call_args_list[1].args[0], x[1:5])
        np.testing.assert_array_equal(fit.call_args_list[1].args[1], [1, 2, 3, 4])
        self.assertEqual([call.args[2] for call in fit.call_args_list], [.1, .1])

    def test_head_weighting_and_convergence_warnings_are_not_silently_accepted(self):
        estimator = Mock(); config = {'tol': 1e-4, 'max_iter': 1000, 'training_seed': 812}
        with patch.object(m, 'LogisticRegression', return_value=estimator) as constructor:
            m.head_fit(np.zeros((30, 7)), np.tile([0, 1], 15), .01, config)
        self.assertEqual(constructor.call_args.kwargs['class_weight'], 'balanced')
        self.assertEqual(constructor.call_args.kwargs['solver'], 'lbfgs')
        np.testing.assert_array_equal(estimator.fit.call_args.kwargs['sample_weight'], np.full(30, 1/15))
        def warn(*args, **kwargs):
            warnings.warn('fixture convergence failure', ConvergenceWarning)
        estimator.fit.side_effect = warn
        with patch.object(m, 'LogisticRegression', return_value=estimator), self.assertRaises(ConvergenceWarning):
            m.head_fit(np.zeros((30, 7)), np.tile([0, 1], 15), .01, config)

    def test_normalizer_is_unchanged_by_validation_or_excluded_fields(self):
        bags = [[np.array([[float(i), 1, 2], [float(i)+2, 3, 4]], np.float32) for _ in range(4)] for i in range(4)]
        before = fit_normalization(bags, [0, 1])
        bags[2] = [np.full((20, 3), 1e8, np.float32) for _ in range(4)]
        bags[3] = [np.full((30, 3), -1e8, np.float32) for _ in range(4)]
        after = fit_normalization(bags, [0, 1])
        for left, right in zip(before, after):
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_immutable_source_receipts_and_registered_config_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'magset').mkdir()
            for name in m.DEPENDENCIES:
                (root/'magset'/name).write_text(name)
            (root/'.source-sha').write_text('a'*40)
            (root/'.archive-sha256').write_text('b'*64)
            with patch.object(m, '__file__', str(root/'magset/five_class.py')):
                context = m.source_context('a'*40)
                self.assertEqual(set(context['implementation_hashes']), set(m.DEPENDENCIES))
                with self.assertRaises(ValueError):
                    m.source_context('c'*40)
            config = root/'different.json'; config.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'config bytes'):
                m.load_inputs(None, None, None, config, 'a'*40)


if __name__ == '__main__':
    unittest.main()
