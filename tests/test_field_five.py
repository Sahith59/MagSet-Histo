"""Field weighting/pooling fixtures; no real or synthetic classifier fits."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from magset import field_five as m
from magset.bags import fit_normalization
from magset.five_metrics import summarize_records
from test_five_class import coefficient_models


def bags_fixture(count):
    bags = []
    for case in range(count):
        views = []
        for view in range(4):
            n = 1+(case+view) % 3
            views.append(np.column_stack((np.arange(n)/10+case/20,
                                          np.full(n, view/4), np.full(n, .5))).astype(np.float32))
        bags.append(views)
    return bags


class FieldFiveTests(unittest.TestCase):
    def normalizer(self):
        return np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)

    def saved(self, path, arm):
        mean, std = self.normalizer()
        models = coefficient_models(arm)
        m.save_model(path, mean, std, models)
        return m.load_model(path, arm, dimension=3), models

    def test_field_class_case_view_mass_uses_cases_not_field_counts(self):
        y = np.repeat(np.arange(5), [11, 25, 2, 5, 3])
        bags = bags_fixture(len(y)); mean, std = self.normalizer()
        for head, mass, classes in [('direct', 46, range(5)), ('binary', 46, range(2)), ('malignant', 35, range(1, 5))]:
            with self.subTest(head=head):
                x, target, weights = m.field_training_rows(bags, np.arange(46), y, mean, std, head)
                self.assertAlmostEqual(weights.sum(), mass)
                self.assertEqual(x.shape[1], 7)
                self.assertEqual(set(target), set(classes))
                for label in classes:
                    self.assertAlmostEqual(weights[target == label].sum(), mass/len(classes))
                selected = np.arange(46)[y > 0] if head == 'malignant' else np.arange(46)
                labels = (y > 0).astype(int) if head == 'binary' else y
                cursor = 0
                for index in selected:
                    nclass = np.sum(labels[selected] == labels[index])
                    for view in range(4):
                        n = len(bags[index][view]); rows = slice(cursor, cursor+n)
                        np.testing.assert_array_equal(x[rows, 3:], np.tile(np.eye(4)[view], (n, 1)))
                        self.assertAlmostEqual(weights[rows].sum(), mass/(len(classes)*nclass*4))
                        cursor += n
                self.assertEqual(cursor, len(x))

    def test_fitting_rows_and_normalizer_ignore_other_cases_entirely(self):
        y = np.array([0, 1, 2, 3, 4, 0, 4]); bags = bags_fixture(7)
        train = np.arange(5); mean, std = fit_normalization(bags, train)
        before = m.field_training_rows(bags, train, y, mean, std, 'direct')
        bags[5] = [np.full((50, 3), np.nan, np.float32) for _ in range(4)]
        bags[6] = [np.full((70, 3), 1e20, np.float32) for _ in range(4)]
        after_normalizer = fit_normalization(bags, train)
        for old, new in zip((mean, std), after_normalizer):
            self.assertEqual(old.tobytes(), new.tobytes())
        for old, new in zip(before, m.field_training_rows(bags, train, y, mean, std, 'direct')):
            np.testing.assert_array_equal(old, new)

    def test_specialist_true_malignant_only_same_C_no_extra_subset_factor(self):
        bags = bags_fixture(7); y = np.array([0, 1, 2, 3, 4, 0, 4]); mean, std = self.normalizer()
        with patch.object(m, 'head_fit', side_effect=[object(), object()]) as fit:
            models = m.fit_arm(bags, np.arange(6), y, mean, std, 'dual', .1, {})
        self.assertEqual(set(models), {'binary', 'malignant'})
        self.assertEqual(fit.call_count, 2)
        self.assertEqual([call.args[-2] for call in fit.call_args_list], [.1, .1])
        self.assertEqual(set(fit.call_args_list[1].args[1]), {1, 2, 3, 4})
        self.assertAlmostEqual(fit.call_args_list[0].args[2].sum(), 6)
        self.assertAlmostEqual(fit.call_args_list[1].args[2].sum(), 4)
        expected_fields = sum(len(view) for case in bags[1:5] for view in case)
        self.assertEqual(len(fit.call_args_list[1].args[0]), expected_fields)

    def test_manual_sample_weights_and_solver_warning_are_enforced(self):
        config = {'tol': 1e-4, 'max_iter': 2000, 'training_seed': 812, 'solver': 'lbfgs', 'penalty': 'l2'}
        estimator = Mock(); weights = np.array([.2, .8])
        with patch.object(m, 'LogisticRegression', return_value=estimator) as constructor:
            m.head_fit(np.zeros((2, 7)), np.array([0, 1]), weights, .1, config)
        self.assertIsNone(constructor.call_args.kwargs['class_weight'])
        self.assertEqual(constructor.call_args.kwargs['max_iter'], 2000)
        np.testing.assert_array_equal(estimator.fit.call_args.kwargs['sample_weight'], weights)
        def warn(*args, **kwargs):
            warnings.warn('fixture convergence failure', ConvergenceWarning)
        estimator.fit.side_effect = warn
        with patch.object(m, 'LogisticRegression', return_value=estimator), self.assertRaises(ConvergenceWarning):
            m.head_fit(np.zeros((2, 7)), np.array([0, 1]), weights, .1, config)

    def test_joint_first_pooling_differs_from_independent_router_expert_means(self):
        r = np.array([.9, .1]); q = np.array([[.97, .01, .01, .01], [.01, .97, .01, .01]])
        joint = np.column_stack((1-r, r[:, None]*q))
        fields = [np.log(joint), None, None, None]
        values = m.pool_log_probabilities(fields, (0,), 'dual')
        np.testing.assert_allclose(values['probabilities5'], joint.mean(0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(values['malignant_probabilities4'], joint.mean(0)[1:]/r.mean(), atol=1e-15)
        self.assertGreater(abs(values['probabilities5'][1]-r.mean()*q.mean(0)[0]), .1)
        self.assertEqual(values['prediction'], 1)

    def test_probability_pooling_can_change_route_from_mean_feature_head(self):
        bags = bags_fixture(1); bags[0][0] = np.array([[-10, 0, 0], [2, 0, 0], [2, 0, 0]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            arrays, _ = self.saved(Path(tmp)/'model.npz', 'dual')
        arrays['binary_coef'][:] = 0; arrays['binary_coef'][0, 0] = 1
        arrays['binary_intercept'][:] = 0; arrays['malignant_coef'][:] = 0; arrays['malignant_intercept'][:] = 0
        rows, _ = m.score_records(arrays, bags, [0], [{'case_id': 'fixture', 'label': 0}], 'validation', 'dual')
        first = next(row for row in rows if row['magnifications'] == [40])
        self.assertAlmostEqual(first['binary_score'], .5872131846081557)
        self.assertEqual(first['prediction'], 1)
        self.assertLess(1/(1+np.exp(2)), .5)

    def test_unequal_field_counts_preserve_equal_view_mass_and_all_subset_invariances(self):
        a = np.array([.8, .05, .05, .05, .05]); b = np.array([.2, .2, .2, .2, .2])
        logs = [np.log(np.tile(a, (17, 1))), np.log(b[None]), None, None]
        pooled = m.pool_log_probabilities(logs, (0, 1), 'direct')
        np.testing.assert_allclose(pooled['probabilities5'], (a+b)/2, atol=1e-15)
        reversed_logs = [logs[0][::-1], logs[1], np.full((10, 5), np.nan), None]
        other = m.pool_log_probabilities(reversed_logs, (1, 0), 'direct')
        for key in pooled:
            np.testing.assert_allclose(pooled[key], other[key], rtol=0, atol=1e-15)
        self.assertFalse(np.allclose(pooled['probabilities5'], (17*a+b)/18))
        for subset in ((), (0, 0), (4,)):
            with self.assertRaises(ValueError):
                m.pool_log_probabilities(logs, subset, 'direct')

    def test_all_subsets_saved_reload_replay_and_field_permutation(self):
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]; bags = bags_fixture(8)
        permuted = [[fields[::-1] for fields in case] for case in bags]
        with tempfile.TemporaryDirectory() as tmp, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')):
            for arm in m.ARMS:
                arrays, models = self.saved(Path(tmp)/(arm+'.npz'), arm)
                records, error = m.score_records(arrays, bags, np.arange(8), cases, 'validation', arm, models)
                shuffled, _ = m.score_records(arrays, permuted, np.arange(8), cases, 'validation', arm)
                self.assertEqual(len(records), 120); self.assertLess(error, 1e-12)
                self.assertEqual(summarize_records(records, arm)['distinct_class_support'], [4, 1, 1, 1, 1])
                for left, right in zip(records, shuffled):
                    self.assertEqual(left['prediction'], right['prediction'])
                    np.testing.assert_allclose(left['log_probabilities5'], right['log_probabilities5'], atol=1e-12)

    def test_extreme_logits_finite_conditional_even_if_malignant_mass_underflows(self):
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]; bags = bags_fixture(8)
        with tempfile.TemporaryDirectory() as tmp:
            arrays, _ = self.saved(Path(tmp)/'direct.npz', 'direct')
            arrays['direct_coef'][:] = 0; arrays['direct_intercept'][:] = [0, -1000, -1000, -1000, -1000]
            rows, _ = m.score_records(arrays, bags, np.arange(8), cases, 'validation', 'direct')
            self.assertEqual(rows[0]['probabilities5'][1], 0)
            self.assertAlmostEqual(summarize_records(rows, 'direct')['mean_subset_cross_entropy'], 500)
            arrays, _ = self.saved(Path(tmp)/'dual.npz', 'dual')
            arrays['binary_coef'][:] = 0; arrays['binary_intercept'][:] = -1000
            arrays['malignant_coef'][:] = 0; arrays['malignant_intercept'][:] = [1000, -1000, -1000, -1000]
            rows, _ = m.score_records(arrays, bags, np.arange(8), cases, 'validation', 'dual')
            self.assertEqual(rows[0]['binary_score'], 0)
            self.assertTrue(np.isfinite(rows[0]['malignant_log_probabilities']).all())
            self.assertEqual(rows[0]['prediction'], 0)
            summarize_records(rows, 'dual')

    def test_half_tie_routes_malignant_even_when_joint_argmax_is_benign(self):
        logs = [np.log(np.array([[.5, .125, .125, .125, .125]])) for _ in range(4)]
        for subset in m.INDEX_SUBSETS:
            values = m.pool_log_probabilities(logs, subset, 'dual')
            self.assertAlmostEqual(values['binary_score'], .5)
            self.assertEqual(values['prediction'], 1)
            self.assertEqual(int(values['probabilities5'].argmax()), 0)
        exact = [np.array([[np.log(.5), np.log(.5), -1000, -1000, -1000]])]*4
        values = m.pool_log_probabilities(exact, (0,), 'dual')
        self.assertEqual(values['binary_score'], .5)
        self.assertEqual(values['prediction'], 1)

    def test_forbidden_roles_indices_empty_view_and_mutated_head_fail(self):
        cases = [{'case_id': 'fixture%d'%i, 'label': i} for i in range(8)]; bags = bags_fixture(8)
        with tempfile.TemporaryDirectory() as tmp:
            arrays, models = self.saved(Path(tmp)/'model.npz', 'direct')
            for role in ('test', 'excluded', 'anything'):
                with self.assertRaises(ValueError):
                    m.score_records(arrays, bags, [0], cases, role, 'direct')
            for indices in ([0, 0], [-1], [8], [], [.5]):
                with self.assertRaises(ValueError):
                    m.score_records(arrays, bags, indices, cases, 'validation', 'direct')
            invalid = copy.deepcopy(bags); invalid[0][0] = np.empty((0, 3), np.float32)
            with self.assertRaises(ValueError):
                m.score_records(arrays, invalid, [0], cases, 'validation', 'direct')
            arrays['direct_intercept'] = np.zeros(5)
            with self.assertRaisesRegex(ValueError, 'replay differs'):
                m.score_records(arrays, bags, [0], cases, 'validation', 'direct', models)

    def test_objective_gradient_diagnostics_are_finite_without_fitting(self):
        bags = bags_fixture(5); y = np.arange(5); mean, std = self.normalizer()
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')):
            for arm in m.ARMS:
                models = coefficient_models(arm)
                receipts, objectives = m.fitting_diagnostics(models, bags, np.arange(5), y, mean, std, .1)
                for name, model in models.items():
                    objective = objectives[name]
                    self.assertTrue(all(np.isfinite(value) for value in objective.values()))
                    self.assertAlmostEqual(objective['objective_sum'], objective['data_loss_sum']+objective['penalty'])
                    self.assertAlmostEqual(receipts['weight_mass_per_head'][name], 4 if name == 'malignant' else 5)
                    self.assertEqual(len(receipts['class_mass_per_head'][name]), len(model.classes_))

    def test_field_config_bytes_checked_before_loading_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'different.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'field config bytes'):
                m.load_inputs(SimpleNamespace(config=path))

    def test_load_extends_original_context_and_retains_base_contract(self):
        config_path = Path('configs/breakhis-phikon-field-v1.json').resolve()
        base_path = Path('configs/breakhis-five-class-v1.json').resolve()
        original = json.loads(base_path.read_text())
        context = {'config_sha256': m.base.CONFIG_SHA256, 'implementation_hashes': {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'magset').mkdir()
            for name in m.DEPENDENCIES:
                (root/'magset'/name).write_text(name)
            args = SimpleNamespace(config=config_path, base_config=base_path,
                                   manifest=None, protocol=None, cache=None, code_sha='a'*40)
            accepted = (original, context, {}, [], [], None, np.array([], dtype=int), {})
            with patch.object(m, '__file__', str(root/'magset/field_five.py')), patch.object(m.base, 'load_inputs', return_value=accepted):
                config, bound, *_ = m.load_inputs(args)
            self.assertEqual(config['expected_role_counts'], [46, 8, 28])
            self.assertEqual(config['max_iter'], 2000)
            self.assertEqual(bound['config_sha256'], m.base.CONFIG_SHA256)
            self.assertEqual(bound['field_config_sha256'], m.FIELD_CONFIG_SHA256)
            self.assertEqual(set(bound['implementation_hashes']), set(m.DEPENDENCIES))

    def test_audit_and_complete_run_budget_earliest_tie_without_any_classifier_fit(self):
        labels = ([0]*11+[1]*25+[3]*2+[4]*5+[5]*3
                  +[0, 2, 6, 7, 1, 3, 4, 5]
                  +[0]*9+[1]*12+[3]*2+[4]*3+[5]*2)
        cases = [{'case_id': 'fixture%d'%i, 'label': label} for i, label in enumerate(labels)]
        bags = bags_fixture(82); y5 = np.array([m.MAP8_TO5[label] for label in labels])
        config = json.loads(Path('configs/breakhis-five-class-v1.json').read_text())
        config.update(json.loads(Path('configs/breakhis-phikon-field-v1.json').read_text()))
        config['encoder'] = dict(config['encoder'], dimension=3); config['encoder_dimension'] = 3
        roles = {(seed, trial): (np.arange(46), np.arange(46, 54), np.arange(54, 82))
                 for seed in config['seeds'] for trial in config['trials']}
        context = {'config_sha256': m.base.CONFIG_SHA256, 'field_config_sha256': m.FIELD_CONFIG_SHA256}
        loaded = (config, context, {}, cases, bags, None, y5, roles)
        def mock_head(x, y, weights, C, conf):
            labels = set(y)
            name = 'direct' if labels == set(range(5)) else 'binary' if labels == {0, 1} else 'malignant'
            arm = 'direct' if name == 'direct' else 'dual'
            model = coefficient_models(arm)[name]; model.n_iter_ = np.array([1])
            return model
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'load_inputs', return_value=loaded), patch.object(m, 'head_fit', side_effect=mock_head) as fit, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fitting')):
            root = Path(tmp)
            args = SimpleNamespace(output=root/'audit')
            m.audit_input(args)
            self.assertEqual(fit.call_count, 0)
            audit = json.loads((args.output/'audit.json').read_text())
            self.assertEqual(len(audit['checks']), 10)
            args = SimpleNamespace(output=root/'run', audit=root/'audit', seed=812, trial=1)
            m.run(args)
            self.assertEqual(fit.call_count, 12)
            report = json.loads((args.output/'run.json').read_text())
            self.assertEqual(report['status'], 'complete field-supervised five-class validation run')
            self.assertEqual(report['role_counts'], [46, 8, 28])
            self.assertEqual(report['head_fits'], 12)
            self.assertEqual(report['outer_predictions'], 0)
            for arm in m.ARMS:
                self.assertEqual(report['arms'][arm]['selected_index'], 0)
                self.assertEqual(report['arms'][arm]['selected_C'], .01)
                self.assertEqual(len(report['arms'][arm]['candidates']), 4)
            self.assertEqual(len(list(args.output.rglob('model.npz'))), 8)
            self.assertEqual(len(list(args.output.rglob('*-predictions.jsonl'))), 16)
            with self.assertRaisesRegex(ValueError, 'existing output'):
                m.run(args)
            self.assertEqual(fit.call_count, 12)


if __name__ == '__main__':
    unittest.main()
