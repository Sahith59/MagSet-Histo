"""Independent head-factorial, exact-router, precision and artifact checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import specialist_five as runner, specialist_collect as collector
from magset import kernel_model, kernel_five
from magset.five_metrics import MAP8_TO5


def fixture():
    rng = np.random.default_rng(991)
    labels = np.append(np.repeat(np.arange(5), 2), [0, 4])
    bags = [[rng.normal(size=(2+(i+v)%3, 7)).astype(np.float32) for v in range(4)] for i in range(len(labels))]
    cases = [{'case_id': 'synthetic%d'%i, 'label': [0, 1, 3, 4, 5][label]} for i, label in enumerate(labels)]
    assert [MAP8_TO5[c['label']] for c in cases] == labels.tolist()
    return bags, labels, cases, np.arange(10), np.asarray([10, 11])


class SpecialistScoringTests(unittest.TestCase):
    def test_identity_exactly_reuses_existing_ordinary_kernel_path(self):
        bags, labels, cases, train, val = fixture()
        for role, indices in [('train', train), ('validation', val)]:
            expected = kernel_five.score_records(kernel_five.prepare_queries(bags, labels, train, val)[role], cases, role, 'dual', .1)
            actual = runner.score_records(bags, labels, train, indices, cases, role, .1, {})
            self.assertEqual(expected, actual)

    def test_factorial_numpy_replay_and_malignant_only_preserves_exact_binary(self):
        bags, labels, cases, train, val = fixture()
        first = np.exp(np.linspace(-.35, .35, 7)).astype(np.float64)
        second = first[::-1].copy()
        old = runner.score_records(bags, labels, train, val, cases, 'validation', .1, {})
        for factors in ({'binary': first}, {'malignant': second}, {'binary': first, 'malignant': second}):
            actual = runner.score_records(bags, labels, train, val, cases, 'validation', .1, factors)
            replay = collector.replay_records(bags, labels, train, val, cases, 'validation', .1, factors)
            for left, right in zip(actual, replay):
                self.assertEqual(left['prediction'], right['prediction'])
                for key in ('probabilities5', 'log_probabilities5', 'malignant_probabilities4', 'binary_log_probabilities'):
                    np.testing.assert_allclose(left[key], right[key], rtol=0, atol=1e-12)
            if 'binary' not in factors:
                for left, original in zip(actual, old):
                    self.assertEqual(left['binary_score'], original['binary_score'])
                    self.assertEqual(left['binary_log_probabilities'], original['binary_log_probabilities'])
                    self.assertEqual(left['prediction'] == 0, original['prediction'] == 0)

    def test_omitted_view_and_field_case_permutation_invariance(self):
        bags, labels, cases, train, val = fixture()
        factors = {'malignant': np.exp(np.linspace(-.2, .2, 7))}
        old = runner.score_records(bags, labels, train, val, cases, 'validation', .03, factors)
        changed = copy.deepcopy(bags)
        for case in changed:
            for view in range(4): case[view] = case[view][::-1].copy()
        replay = runner.score_records(changed, labels, train[::-1], val, cases, 'validation', .03, factors)
        for a, b in zip(old, replay):
            np.testing.assert_allclose(a['probabilities5'], b['probabilities5'], rtol=0, atol=1e-12)
        changed = copy.deepcopy(bags)
        for case in changed: case[3] *= 1e4
        replay = runner.score_records(changed, labels, train, val, cases, 'validation', .03, factors)
        for a, b in zip(old, replay):
            if 400 not in a['magnifications']:
                self.assertEqual(a, b)

    def test_fitting_whole_query_excluded_and_held_queries_rejected_as_train(self):
        bags, labels, cases, train, val = fixture()
        query = kernel_model.prepare_case(bags, labels, train, 0, leave_query_case_out=True)
        self.assertNotIn(0, query.support_indices)
        with self.assertRaises(ValueError):
            runner.score_records(bags, labels, train, val, cases, 'train', .1, {})

    def test_exact_uniform_half_router_tie_replays_without_rounding_route_change(self):
        bags, labels, cases, train, val = fixture()
        bags = [[np.zeros_like(view) for view in case] for case in bags]
        for factors in ({}, {'malignant': np.exp(np.linspace(-.2, .2, 7))}):
            actual = runner.score_records(bags, labels, train, val, cases, 'validation', .1, factors)
            replay = collector.replay_records(bags, labels, train, val, cases, 'validation', .1, factors)
            for left, right in zip(actual, replay):
                self.assertEqual(left['prediction'], right['prediction'])
                self.assertEqual(left['binary_score'], right['binary_score'])

    def test_precision_uses_float64_before_field_means(self):
        bags, labels, cases, train, val = fixture()
        bags[0][0] = np.asarray([[1e8]*7, [1.]*7, [-1e8]*7], dtype=np.float32)
        values = runner.fitting_centroids(bags, train)
        np.testing.assert_array_equal(values[0, 0], np.full(7, 1./3))
        self.assertFalse(np.array_equal(values[0, 0], bags[0][0].mean(0)))
        np.testing.assert_array_equal(values.mean(0), np.stack(kernel_model._centers(bags, train)))

    def test_saved_actual_float32_factors_accept_replay_and_reject_tampering(self):
        theta = np.linspace(-.6, .6, 7).astype(np.float32)
        logs = (.5*np.tanh(theta)).astype(np.float32)
        factors = np.exp(logs).astype(np.float32)
        state = {'theta': theta, 'log_factors': logs.astype(np.float64), 'factors': factors.astype(np.float64)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metric.npz'; runner.save_factors(path, state)
            loaded = runner.load_factors(path, 7)
            for key in state: np.testing.assert_array_equal(loaded[key], state[key])
            state['factors'][0] += .01; runner.save_factors(path, state)
            with self.assertRaisesRegex(ValueError, 'parameterization'): runner.load_factors(path, 7)

    def test_canonical_config_and_inherited_temperatures_are_fixed(self):
        path = Path('configs/breakhis-specialist-metric-v1.json')
        if not path.exists(): self.skipTest('Root registration not yet present')
        config = json.loads(path.read_text())
        self.assertIs(runner.validate_config(config), config)
        changed = copy.deepcopy(config); changed['metric']['steps'] = 201
        with self.assertRaises(ValueError): runner.validate_config(changed)


class SpecialistSavedRunTests(unittest.TestCase):
    def test_actual_synthetic_optimizers_reload_and_numpy_collector_rejects_metric_tampering(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional torch unavailable')
        rng = np.random.default_rng(819)
        labels = np.repeat(np.arange(5), [11, 25, 2, 5, 3]).tolist()+[0, 0, 0, 0, 1, 2, 3, 4]+[0]*28
        labels = np.asarray(labels, dtype=int)
        bags = [[rng.normal(size=(1, 1536)).astype(np.float32) for _ in range(4)] for _ in labels]
        cases = [{'case_id': 'synthetic%d'%i, 'label': [0, 1, 3, 4, 5][label]} for i, label in enumerate(labels)]
        train, val, excluded = np.arange(46), np.arange(46, 54), np.arange(54, 82)
        roles = {(812, 1): (train, val, excluded)}
        z = np.asarray([[view.mean(0) for view in case] for case in bags], dtype=np.float32)
        identity = runner.score_records(bags, labels, train, val, cases, 'validation', .1, {})
        candidate = {'temperature': .1, 'validation': runner.summarize_records(identity, 'dual')}
        reference = {'runs': [{'seed': 812, 'trial': 1, 'arms': {'dual': {
            'selected_index': 2, 'selected_temperature': .1, 'candidates': [{}, {}, candidate, {}]}}}]}
        config = {}; context = {'code_sha': 'a'*40}
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); audit = root/'audit'; audit.mkdir(); runner.write_json(audit/'audit.json', {'synthetic': True})
                args = SimpleNamespace(output=root/'run', audit=audit, seed=812, trial=1, device='cpu')
                inputs = (config, context, {}, cases, bags, z, labels, roles, reference)
                with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(runner, 'verify_input_audit'), patch.object(runner, 'verify_identity'):
                    report = runner.run(args)
                from magset.data import file_hash
                from magset.specialist_metric import fit_metric
                with patch('magset.specialist_metric.fit_metric', side_effect=AssertionError('Collector must not fit')):
                    verified = collector.verify_run(args.output, 812, 1, config, context, cases, bags, z, labels, roles, reference, file_hash(audit/'audit.json'))
                self.assertEqual(verified['metric_fits'], 2)
                self.assertEqual(set(verified['arms']), set(runner.ARMS))
                self.assertEqual(verified['arms']['both']['validation']['pooled']['records'], 120)
                # Reuse these synthetic trained states to exercise the full smoke
                # entry point without an additional optimization trajectory.
                states = {}
                for head in ('binary', 'malignant'):
                    states[head] = runner.load_factors(args.output/(head+'-metric.npz'))
                    training = runner.read_json(args.output/(head+'-training.json'))
                    states[head].update(training)
                def fitted_state(*unused, head, **kwargs):
                    return states[head]
                smoke_args = SimpleNamespace(**vars(args)); smoke_args.output = root/'smoke'
                with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(runner, 'verify_input_audit'), patch.object(runner, 'verify_identity', side_effect=AssertionError('Smoke must not score validation')), patch('magset.specialist_metric.fit_metric', side_effect=fitted_state):
                    smoke = runner.run(smoke_args, smoke=True)
                self.assertEqual(smoke['status'], 'complete specialist fitting-only technical smoke')
                self.assertEqual(runner.smoke_query_indices(labels, train).tolist(), [0, 11, 36, 38, 43])
                for arm in runner.ARMS:
                    self.assertEqual(smoke['arms'][arm]['fitting']['pooled']['records'], 75)
                    self.assertNotIn('validation', smoke['arms'][arm])
                    path = smoke_args.output/arm/'train-predictions.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual({row['truth5'] for row in rows}, set(range(5)))
                    self.assertFalse((smoke_args.output/arm/'validation-predictions.jsonl').exists())
                report['arms']['both']['validation']['mean_subset_accuracy'] += .1
                runner.write_json(args.output/'run.json', report)
                with self.assertRaisesRegex(ValueError, 'factorial both/validation'):
                    collector.verify_run(args.output, 812, 1, config, context, cases, bags, z, labels, roles, reference, file_hash(audit/'audit.json'))
        finally:
            torch.set_num_threads(previous_threads)
        changed = copy.deepcopy(config); changed['arms'] = ['identity', 'both']
        with self.assertRaises(ValueError): runner.validate_config(changed)


if __name__ == '__main__': unittest.main()
