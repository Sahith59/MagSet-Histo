"""Collector tamper/leakage fixtures; coefficients are assigned and never fitted."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from magset import five_class as runner
from magset import five_collect as m
from magset.bags import fit_normalization
from magset.baseline import stratified_inner_split
from magset.data import CLASS_NAMES, file_hash
from magset.five_metrics import FINAL5, MAP8_TO5, five_labels, summarize_records
from test_five_class import coefficient_models


def fixture(root):
    config = json.loads(Path('configs/breakhis-five-class-v1.json').read_text())
    config['encoder']['dimension'] = 3
    cases = []; bags = []; outer = {}
    counts = [4, 38, 10, 5, 9, 6, 3, 7]; excluded = [2, 12, 3, 2, 3, 2, 1, 3]
    for label, (count, held) in enumerate(zip(counts, excluded)):
        for j in range(count):
            cid = 'fixture-%d-%02d'%(label, j)
            cases.append({'case_id': cid, 'label': label})
            outer[cid] = 'test' if j < held else 'train'
            fields = np.array([[label/8, j/20, .2], [label/8+.1, j/20+.2, .3]], np.float32)
            bags.append([fields.copy() for _ in range(4)])
    y5 = five_labels(np.array([case['label'] for case in cases]))
    z = np.array([[x.mean(0) for x in case] for case in bags], np.float32)
    roles = {(seed, trial): stratified_inner_split(cases, outer, seed)
             for seed in config['seeds'] for trial in config['trials']}
    context = {'code_sha': 'a'*40, 'source_archive_sha256': 'b'*64,
               'implementation_hashes': {'five_class.py': 'c'*64, 'five_collect.py': 'd'*64},
               'config_sha256': runner.CONFIG_SHA256, 'manifest_sha256': 'e'*64,
               'protocol_sha256': 'f'*64, 'unit': 'synthetic provider-case fixture',
               'dependencies': config['dependencies'], 'python': 'fixture',
               'encoder': config['encoder'], 'cache_provenance': config['cache_provenance']}
    checks = []
    for (seed, trial), (train, val, held) in roles.items():
        mean, std = fit_normalization(bags, train)
        checks.append({'seed': seed, 'trial': trial, 'role_counts': [len(train), len(val), len(held)],
                       'fitting_support': np.bincount(y5[train], minlength=5).tolist(),
                       'validation_support': np.bincount(y5[val], minlength=5).tolist(),
                       'normalization_sha256': hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()})
    audit = {'status': 'zero-fit inputs accepted', 'new_fits': 0, 'new_forwards': 0,
             'outer_predictions': 0, 'checks': checks, **context}
    audit_dir = root/'audit'; audit_dir.mkdir(); runner.write_json(audit_dir/'audit.json', audit)
    folder = root/'runs/split812/trial1'; folder.mkdir(parents=True)
    train, val, held = roles[812, 1]; mean, std = fit_normalization(bags, train)
    arms = {}
    for arm in runner.ARMS:
        models = coefficient_models(arm)
        candidates = []
        records = {}; summaries = {}
        for index, C in enumerate(config['C_grid']):
            directory = folder/arm/('candidate%d'%index); directory.mkdir(parents=True)
            runner.save_model(directory/'model.npz', mean, std, models)
            arrays = runner.load_model(directory/'model.npz', arm, dimension=3)
            if not records:
                for role, indices in (('train', train), ('validation', val)):
                    rows, _ = runner.score_records(arrays, z, indices, cases, role, arm)
                    records[role] = ''.join(json.dumps(row)+'\n' for row in rows)
                    summaries[role] = summarize_records(rows, arm)
            for role, text in records.items():
                (directory/(role+'-predictions.jsonl')).write_text(text)
            candidate = {'index': index, 'C': C, 'head_fits': len(models),
                         'n_iter': {name: [2] for name in models}, 'sklearn_replay_max_abs': 0.,
                         'elapsed_seconds': .1, 'model_sha256': file_hash(directory/'model.npz'),
                         'train_predictions_sha256': file_hash(directory/'train-predictions.jsonl'),
                         'validation_predictions_sha256': file_hash(directory/'validation-predictions.jsonl'),
                         'fitting': summaries['train'], 'validation': summaries['validation']}
            runner.write_json(directory/'metrics.json', candidate); candidates.append(candidate)
        arms[arm] = {'selected_index': 0, 'selected_C': config['C_grid'][0],
                     'candidates': candidates, 'fitting': summaries['train'], 'validation': summaries['validation']}
    report = {'status': 'complete five-class validation run', 'seed': 812, 'trial': 1,
              'class_names': list(FINAL5), 'original_class_names': list(CLASS_NAMES),
              'case_ids': {role: [cases[i]['case_id'] for i in ix] for role, ix in
                           (('train', train), ('validation', val), ('excluded', held))},
              'role_counts': [46, 8, 28], 'head_fits': 12, 'outer_predictions': 0,
              'arms': arms, 'audit_sha256': file_hash(audit_dir/'audit.json'), **context}
    runner.write_json(folder/'run.json', report)
    return SimpleNamespace(config=config, context=context, cases=cases, bags=bags, z=z, y5=y5,
                           roles=roles, audit=audit, audit_dir=audit_dir, folder=folder, report=report)


class FiveCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.f = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.original = {}

    def tearDown(self):
        for path, content in self.original.items():
            path.write_bytes(content)

    def write(self, path, content):
        path = Path(path)
        if path not in self.original:
            self.original[path] = path.read_bytes()
        path.write_text(content)

    def write_candidate(self, candidate, arm='direct', index=0):
        folder = self.f.folder/arm/('candidate%d'%index)
        self.write(folder/'metrics.json', json.dumps(candidate))
        report = m.read_json(self.f.folder/'run.json')
        report['arms'][arm]['candidates'][index] = candidate
        self.write(self.f.folder/'run.json', json.dumps(report))

    def verify(self):
        f = self.f
        return m.verify_run(f.folder, 812, 1, f.config, f.context, f.cases, f.bags,
                            f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))

    def test_complete_candidate_replay_preserves_all_original_roles_and_never_fits(self):
        with patch.object(runner, 'head_fit', side_effect=AssertionError('Collector must not fit')):
            result = self.verify()
        self.assertEqual(result['head_fits'], 12)
        self.assertEqual(set(result['arms']), {'direct', 'dual'})
        for arm in runner.ARMS:
            self.assertEqual(len(result['arms'][arm]['candidates']), 4)
            self.assertEqual(result['arms'][arm]['selected_index'], 0)
            self.assertEqual(result['arms'][arm]['fitting']['distinct_class_support'], [11, 25, 2, 5, 3])
            self.assertEqual(result['arms'][arm]['validation']['distinct_class_support'], [4, 1, 1, 1, 1])
        m.assert_public_safe(result)
        self.assertNotIn('fixture-', json.dumps(result))

    def test_zero_fit_audit_rejects_missing_checks_source_changes_and_normalizer_poison(self):
        f = self.f
        m.verify_audit(f.audit, f.context, f.config, f.bags, f.y5, f.roles)
        for mutation in ('missing', 'context', 'normalizer', 'counter'):
            audit = copy.deepcopy(f.audit)
            if mutation == 'missing':
                audit['checks'].pop()
            elif mutation == 'context':
                audit['source_archive_sha256'] = '0'*64
            elif mutation == 'normalizer':
                audit['checks'][0]['normalization_sha256'] = '0'*64
            else:
                audit['outer_predictions'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.verify_audit(audit, f.context, f.config, f.bags, f.y5, f.roles)

    def test_coordinated_hash_tamper_cannot_change_original_truth_or_admit_excluded_case(self):
        path = self.f.folder/'direct/candidate0/train-predictions.jsonl'
        original = path.read_text(); candidate = m.read_json(path.with_name('metrics.json'))
        rows = [json.loads(line) for line in original.splitlines()]
        old = rows[0]['truth']; rows[0]['truth'] = (old+1)%8
        rows[0]['truth5'] = MAP8_TO5[rows[0]['truth']]
        self.write(path, ''.join(json.dumps(row)+'\n' for row in rows))
        candidate['train_predictions_sha256'] = file_hash(path)
        self.write_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'trusted role/original truth'):
            self.verify()
        rows = [json.loads(line) for line in original.splitlines()]
        excluded = self.f.roles[812, 1][2][0]
        rows[0]['case_id'] = self.f.cases[excluded]['case_id']
        self.write(path, ''.join(json.dumps(row)+'\n' for row in rows))
        candidate['train_predictions_sha256'] = file_hash(path)
        self.write_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'trusted role/original truth'):
            self.verify()

    def test_normalizer_must_match_fitting_only_moments_bitwise_even_if_hashes_updated(self):
        path = self.f.folder/'direct/candidate0/model.npz'
        self.original[path] = path.read_bytes()
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['mean'][0, 0] = np.nextafter(arrays['mean'][0, 0], np.float32(np.inf))
        np.savez_compressed(path, **arrays)
        candidate = m.read_json(path.with_name('metrics.json'))
        candidate['model_sha256'] = file_hash(path); self.write_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'normalizer differs bitwise'):
            self.verify()

    def test_candidate_summary_and_earliest_tie_selection_are_recomputed(self):
        candidate = m.read_json(self.f.folder/'direct/candidate0/metrics.json')
        candidate['validation']['mean_subset_macro_f1'] += .1
        self.write_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'candidate/validation'):
            self.verify()
        # Restore the candidate before testing selection: all four fixtures tie.
        for path, content in self.original.items():
            path.write_bytes(content)
        report = m.read_json(self.f.folder/'run.json')
        report['arms']['direct']['selected_index'] = 1
        report['arms']['direct']['selected_C'] = .1
        self.write(self.f.folder/'run.json', json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'earliest maximum'):
            self.verify()

    def test_public_leak_guard_and_duplicate_json_keys(self):
        for value in ({'case_ids': []}, {'text': 'SOB_M_DC-14-99999'}, {'text': '/private/root'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.assert_public_safe(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'duplicate.json'; path.write_text('{"a": 1, "a": 2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                m.read_json(path)

    def test_complete_matrix_controller_writes_safe_aggregates_and_no_partial_ranking(self):
        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'
            for seed, trial in f.roles:
                path = runs/('split%d'%seed)/('trial%d'%trial)/'run.json'
                path.parent.mkdir(parents=True); path.write_text('{}')
                for arm in runner.ARMS:
                    for index in range(4):
                        directory = path.parent/arm/('candidate%d'%index); directory.mkdir(parents=True)
                        for name in ('model.npz', 'train-predictions.jsonl', 'validation-predictions.jsonl'):
                            (directory/name).write_text('fixture: verified by the mocked per-run verifier')
            args = SimpleNamespace(runs=runs, audit=f.audit_dir, output=root/'comparison',
                manifest=None, protocol=None, cache=None, config=None, code_sha='a'*40)
            inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
            gate_fixture = {'mode': 'original'}
            def verified(folder, seed, trial, *unused):
                # This test isolates complete-matrix/gate aggregation. Actual
                # coefficient, label and metric replay is exercised above.
                arms = copy.deepcopy(f.report['arms'])
                if gate_fixture['mode'] != 'original':
                    for arm in runner.ARMS:
                        value = arms[arm]['validation']
                        value['mean_subset_macro_f1'] = .4 if arm == 'direct' else .5
                        value['mean_subset_accuracy'] = .5 if arm == 'direct' else .6
                        value['pooled']['confusion_matrix'] = (np.eye(5)*15).tolist()
                    if gate_fixture['mode'] == 'accuracy failure' and seed == 813:
                        arms['dual']['validation']['mean_subset_accuracy'] = .4
                return {'seed': seed, 'trial': trial, 'head_fits': 12, 'run_sha256': '0'*64, 'arms': arms}
            with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(m, 'verify_run', side_effect=verified):
                result = m.collect(args)
            self.assertEqual(result['production_head_fits'], 120)
            self.assertEqual(result['collector_new_fits'], 0)
            self.assertEqual(result['verified_runs'], 10)
            self.assertFalse(result['gate_passed'])
            self.assertEqual(len(result['runs']), 10)
            self.assertEqual(set(result['summary']), {'812', '813'})
            m.assert_public_safe(m.read_json(args.output/'comparison.json'))
            self.assertNotIn('fixture-', (args.output/'comparison.json').read_text())
            self.assertIn('minimum_trial_worst_subset_macro_f1', result['summary']['812']['direct']['validation'])
            for mode, passed in (('positive', True), ('accuracy failure', False)):
                gate_fixture['mode'] = mode; args.output = root/mode
                with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(m, 'verify_run', side_effect=verified):
                    self.assertEqual(m.collect(args)['gate_passed'], passed)
            (runs/'split813/trial5/run.json').unlink(); args.output = root/'rejected'
            with patch.object(runner, 'load_inputs', return_value=inputs), patch.object(m, 'verify_run') as verify:
                with self.assertRaisesRegex(ValueError, 'Complete registered run matrix'):
                    m.collect(args)
                verify.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_collector_requires_runs_and_audit(self):
        with self.assertRaisesRegex(ValueError, '--runs'):
            m.collect(SimpleNamespace(runs=None, audit=None))


if __name__ == '__main__':
    unittest.main()
