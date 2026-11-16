"""Fixed-mixture math and hash/role/selection corruption fixtures; no real inputs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from magset import fixed_ensemble as m
from magset.data import CLASS_NAMES, file_hash
from magset.five_metrics import FINAL5, MAP8_TO5, summarize_records
from test_five_collect import fixture as original_fixture


def record(case, subset, probability, arm, logs=None):
    p = np.asarray(probability, float); logs = np.log(p) if logs is None else np.asarray(logs, float)
    log_r = logsumexp(logs[1:]); score = float(p[1:].sum()); log_q = logs[1:]-log_r
    q = p[1:]/score if score else np.exp(log_q)
    pred = int(p.argmax()) if arm == 'direct' else 1+int(q.argmax()) if score >= .5 else 0
    return {'case_id': case['case_id'], 'role': 'validation', 'truth': int(case['label']), 'truth5': MAP8_TO5[case['label']],
            'magnifications': list(subset), 'probabilities5': p.tolist(), 'log_probabilities5': logs.tolist(),
            'binary_score': score, 'binary_log_probabilities': [float(logs[0]), float(log_r)],
            'malignant_probabilities4': q.tolist(), 'malignant_log_probabilities': log_q.tolist(), 'prediction': pred}


def fixture(root):
    old = root/'old'; old.mkdir(); f = original_fixture(old)
    config = {**f.config, **json.loads(Path('configs/breakhis-fixed-ensemble-v1.json').read_text())}
    config['midnight_reference_file'] = str(root/'midnight-reference.json')
    config['kernel_reference_file'] = str(root/'kernel-reference.json')
    context = dict(f.context, ensemble_config_sha256='9'*64)
    args = SimpleNamespace(midnight_runs=root/'midnight', kernel_runs=root/'kernel', audit=root/'audit', output=root/'audit')
    references = {}
    for source in ('midnight', 'kernel'):
        public_runs = []
        for (seed, trial), (train, val, excluded) in sorted(f.roles.items()):
            source_root = args.midnight_runs/'case' if source == 'midnight' else args.kernel_runs
            folder = source_root/('split%d'%seed)/('trial%d'%trial); folder.mkdir(parents=True)
            arms, public_arms = {}, {}
            for arm in m.ARMS:
                rows = []
                for subset in m.SUBSETS:
                    for index in val:
                        truth5 = MAP8_TO5[f.cases[index]['label']]
                        p = np.full(5, .0875 if source == 'midnight' else .1125)
                        p[truth5] = .65 if source == 'midnight' else .55
                        rows.append(record(f.cases[index], subset, p, arm))
                if source == 'kernel': rows = rows[::-1]  # Native writer orders may differ.
                summary = summarize_records(rows, arm); candidates = []; public_candidates = []
                for index in range(4):
                    candidate = {'index': index, 'validation': summary, 'fitting' if source == 'midnight' else 'fitting_leave_case_out': f.report['arms'][arm]['fitting']}
                    candidate['C' if source == 'midnight' else 'temperature'] = ([.01, .1, 1., 10.] if source == 'midnight' else [.01, .03, .1, .3])[index]
                    directory = folder/arm/('candidate%d'%index); directory.mkdir(parents=True)
                    path = directory/'validation-predictions.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    candidate['validation_predictions_sha256'] = file_hash(path)
                    m.base.write_json(directory/'metrics.json', candidate); candidates.append(candidate)
                    public_candidates.append(dict(candidate, metrics_sha256=file_hash(directory/'metrics.json')))
                key = 'selected_C' if source == 'midnight' else 'selected_temperature'
                arms[arm] = {'selected_index': 0, key: .01, 'candidates': candidates, 'validation': summary,
                             'fitting' if source == 'midnight' else 'fitting_leave_case_out': f.report['arms'][arm]['fitting']}
                public_arms[arm] = dict(arms[arm], candidates=public_candidates)
            private = {**f.context, 'seed': seed, 'trial': trial, 'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
                       'role_counts': [46, 8, 28], 'arms': arms,
                       'case_ids': {r: [f.cases[i]['case_id'] for i in ix] for r, ix in [('train', train), ('validation', val), ('excluded', excluded)]}}
            m.base.write_json(folder/'run.json', private)
            public_runs.append({'seed': seed, 'trial': trial, 'readout': 'case', 'arms': public_arms, 'run_sha256': file_hash(folder/'run.json')})
        document = {**f.context, 'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES), 'outer_predictions': 0,
                    'verified_runs': 20 if source == 'midnight' else 10, 'runs': public_runs}
        summaries = {str(seed): {arm: {'validation': m.aggregate_summaries([r['arms'][arm]['validation'] for r in public_runs if r['seed'] == seed])} for arm in m.ARMS} for seed in (812, 813)}
        document['summary'] = {'case': summaries} if source == 'midnight' else summaries
        path = Path(config[source+'_reference_file']); m.base.write_json(path, document)
        config[source+'_reference_sha256'] = file_hash(path); references[source] = document
    f.config, f.context = config, context
    f.inputs = (config, context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
    f.args, f.references = args, references
    return f


class FixedEnsembleMathTests(unittest.TestCase):
    def test_registered_weights_and_inherited_budget_cannot_change(self):
        own = json.loads(Path('configs/breakhis-fixed-ensemble-v1.json').read_text())
        def digest(path):
            return own['base_input_config_sha256'] if Path(path).name == 'breakhis-midnight-five-v1.json' else 'a'*64
        empty = ({}, {'implementation_hashes': {}}, {}, [], [], None, np.array([], int), {})
        args = SimpleNamespace(config=Path('fixture-alias.json'))
        with patch.object(m, 'file_hash', side_effect=digest), patch.object(m, 'read_json', return_value=own), patch.object(m.base, 'load_inputs', return_value=empty):
            config, context, *_ = m.load_inputs(args)
        self.assertEqual(context['ensemble_config_sha256'], 'a'*64)
        self.assertEqual(config['weight_midnight'], .5)
        for key, value in [('weight_midnight', .6), ('new_weight_selections', 1), ('inherited_selection_opportunities_per_role_per_arm', {'midnight_case_C': 4})]:
            changed = dict(own, **{key: value})
            with patch.object(m, 'file_hash', side_effect=digest), patch.object(m, 'read_json', return_value=changed), patch.object(m.base, 'load_inputs') as load:
                with self.subTest(key=key), self.assertRaises(ValueError): m.load_inputs(args)
                load.assert_not_called()

    def test_joint_mixture_is_not_separate_router_expert_average_and_ties_route_malignant(self):
        case = {'case_id': 'fixture', 'label': 1}; q1 = np.array([.60, .38, .01, .01]); q2 = np.array([.01, .97, .01, .01])
        for arm in m.ARMS:
            left = record(case, [40], np.r_[.1, .9*q1], arm)
            right = record(case, [40], np.r_[.9, .1*q2], arm)
            actual = m.blend_records([left], [right], arm)[0]
            np.testing.assert_allclose(actual['probabilities5'], [.5, .2705, .2195, .005, .005], atol=1e-15)
            self.assertEqual(actual['prediction'], 0 if arm == 'direct' else 1)
            self.assertEqual(actual['binary_score'], .5)
            self.assertNotEqual(int(((q1+q2)/2).argmax())+1, 1)

    def test_one_thousand_half_boundary_pairs_match_independent_scalar_replay(self):
        rng = np.random.default_rng(42); case = {'case_id': 'fixture', 'label': 1}
        for _ in range(1000):
            r = rng.uniform(.001, .999); q1 = rng.dirichlet(np.ones(4)); q2 = rng.dirichlet(np.ones(4))
            a = record(case, [40], np.r_[1-r, r*q1], 'dual'); b = record(case, [40], np.r_[r, (1-r)*q2], 'dual')
            actual = m.blend_records([a], [b], 'dual')[0]; expected = m.blend_records([a], [b], 'dual', independent=True)[0]
            self.assertEqual(actual['prediction'], expected['prediction'])
            self.assertEqual(actual['binary_score'], expected['binary_score'])
            np.testing.assert_array_equal(actual['probabilities5'], expected['probabilities5'])
            np.testing.assert_allclose(actual['log_probabilities5'], expected['log_probabilities5'], atol=2e-15)

    def test_underflow_retains_finite_logs_and_conditional_distribution(self):
        case = {'case_id': 'fixture', 'label': 1}
        logs = np.array([0., -1000., -1001., -1002., -1003.])
        row = record(case, [40], np.exp(logs), 'dual', logs)
        actual = m.blend_records([row], [row], 'dual')[0]
        self.assertTrue(np.isfinite(actual['log_probabilities5']).all())
        self.assertTrue(np.isfinite(actual['malignant_log_probabilities']).all())
        self.assertAlmostEqual(sum(actual['malignant_probabilities4']), 1)
        self.assertEqual(actual['binary_score'], 0)
        self.assertEqual(actual['prediction'], 0)


class FixedEnsembleArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)
        with patch.object(m, 'load_inputs', return_value=cls.f.inputs), patch.object(m, 'blend_records', side_effect=AssertionError('Audit cannot compute ensemble performance')):
            m.audit_input(cls.f.args)
        cls.run_root = cls.root/'ensemble'
        for seed, trial in cls.f.roles:
            args = SimpleNamespace(**vars(cls.f.args)); args.seed, args.trial = seed, trial; args.output = cls.run_root/('split%d'%seed)/('trial%d'%trial)
            with patch.object(m, 'load_inputs', return_value=cls.f.inputs): m.run(args)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self): self.original = {}; self.config = copy.deepcopy(self.f.config)
    def tearDown(self):
        for p, data in self.original.items(): p.write_bytes(data)
        self.f.config.clear(); self.f.config.update(self.config)
    def write(self, path, value, jsonl=False):
        self.original.setdefault(path, path.read_bytes())
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value))

    def test_audit_binds_all_sources_and_canonical_join_ignores_native_order(self):
        f = self.f; audit = m.read_json(f.args.audit/'audit.json')
        self.assertEqual(len(audit['source_bindings']), 40)
        refs, sources, bindings = m.load_components(f.args, f.config, f.context, f.cases, f.roles)
        for seed, trial in f.roles:
            for arm in m.ARMS:
                a, b = sources['midnight', seed, trial, arm], sources['kernel', seed, trial, arm]
                self.assertEqual([{k: r[k] for k in m.IDENTITY} for r in a], [{k: r[k] for k in m.IDENTITY} for r in b])
        self.assertEqual(sum(b['records'] for b in bindings), 4800)

    def test_complete_collection_is_independent_zero_fit_and_has_no_fitting_metric(self):
        args = SimpleNamespace(**vars(self.f.args)); args.runs = self.run_root
        with tempfile.TemporaryDirectory() as tmp:
            args.output = Path(tmp)/'comparison'
            with patch.object(m, 'load_inputs', return_value=self.f.inputs), patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No ensemble fitting')):
                result = m.collect(args)
            self.assertEqual(result['verified_prediction_archives'], 20)
            self.assertEqual(result['verified_case_subset_records'], 2400)
            self.assertEqual(result['new_weight_selections'], 0)
            self.assertFalse(result['gate_passed'])
            self.assertNotIn('fitting', result['summary']['812']['dual'])
            self.assertEqual(result['inherited_selection_opportunities_per_role_per_arm'], {'midnight_case_C': 4, 'kernel_temperature': 4})
            m.assert_public_safe(result)

    def test_private_run_hash_and_missing_matrix_fail_before_any_blend(self):
        path = self.f.args.midnight_runs/'case/split812/trial1/run.json'
        self.original[path] = path.read_bytes(); path.write_text(path.read_text()+' ')
        with patch.object(m, 'blend_records', side_effect=AssertionError('Do not blend corrupted inputs')):
            with self.assertRaisesRegex(ValueError, 'run hash'): m.load_components(self.f.args, self.f.config, self.f.context, self.f.cases, self.f.roles)
        path.write_bytes(self.original[path]); self.original.pop(path)
        out = self.run_root/'split813/trial5/run.json'; self.original[out] = out.read_bytes(); out.unlink()
        args = SimpleNamespace(**vars(self.f.args)); args.runs = self.run_root; args.output = self.root/'partial-rejected'
        with patch.object(m, 'load_inputs', return_value=self.f.inputs):
            with self.assertRaisesRegex(ValueError, 'artifact matrix'): m.collect(args)
        self.assertFalse(args.output.exists())

    def test_coordinated_rehash_cannot_change_earliest_component_selection(self):
        f = self.f; run_path = f.args.midnight_runs/'case/split812/trial1/run.json'
        private = m.read_json(run_path)
        private['arms']['dual']['selected_index'] = 1
        private['arms']['dual']['selected_C'] = .1
        self.write(run_path, private)
        reference_path = Path(f.config['midnight_reference_file']); public = m.read_json(reference_path)
        run = next(r for r in public['runs'] if r['seed'] == 812 and r['trial'] == 1)
        run['run_sha256'] = file_hash(run_path)
        run['arms']['dual']['selected_index'] = 1
        run['arms']['dual']['selected_C'] = .1
        self.write(reference_path, public)
        f.config['midnight_reference_sha256'] = file_hash(reference_path)
        with self.assertRaisesRegex(ValueError, 'earliest component selection'):
            m.load_components(f.args, f.config, f.context, f.cases, f.roles)

    def test_duplicate_missing_and_trusted_label_mismatches_are_rejected(self):
        folder = self.f.args.kernel_runs/'split812/trial1/dual/candidate0'
        path = folder/'validation-predictions.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
        for mode in ('duplicate', 'missing', 'truth', 'excluded'):
            changed = copy.deepcopy(rows)
            if mode == 'duplicate': changed[0] = changed[1]
            elif mode == 'missing': changed.pop()
            elif mode == 'truth': changed[0]['truth'] = (changed[0]['truth']+1)%8
            else: changed[0]['case_id'] = self.f.cases[self.f.roles[812, 1][2][0]]['case_id']
            self.write(path, changed, True)
            with self.subTest(mode=mode), self.assertRaises(ValueError): m.checked_records(path, self.f.cases, self.f.roles[812, 1][1], 'dual')

    def test_rehashed_incorrect_ensemble_output_cannot_change_the_fixed_rule(self):
        folder = self.run_root/'split812/trial1/dual'; path = folder/'validation-predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0] = record({'case_id': rows[0]['case_id'], 'label': rows[0]['truth']}, rows[0]['magnifications'], [.4, .3, .1, .1, .1], 'dual')
        self.write(path, rows, True)
        metric = {'validation': summarize_records(rows, 'dual'), 'prediction_sha256': file_hash(path)}
        self.write(folder/'metrics.json', metric)
        run_path = folder.parent/'run.json'; run = m.read_json(run_path); run['arms']['dual'] = metric; self.write(run_path, run)
        args = SimpleNamespace(**vars(self.f.args)); args.runs = self.run_root; args.output = self.root/'replay-rejected'
        with patch.object(m, 'load_inputs', return_value=self.f.inputs):
            with self.assertRaisesRegex(ValueError, 'independent linear'): m.collect(args)
        self.assertFalse(args.output.exists())


if __name__ == '__main__': unittest.main()
