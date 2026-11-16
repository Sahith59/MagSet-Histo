"""Small-state and cached-feature tamper fixtures, with no encoder construction."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from magset import lora_collect as m
    from magset import lora_io as io
except ImportError:
    torch = None

from magset.bags import fit_normalization
from magset.data import MAGNIFICATIONS, file_hash
from magset.field_five import score_records
from magset.five_class import load_model, write_json
from magset.five_metrics import summarize_records
from test_five_collect import fixture as old_fixture


def fixture(root):
    old = root/'old'; old.mkdir(); f = old_fixture(old)
    f.config.update(json.loads(Path('configs/breakhis-midnight-lora-v1.json').read_text()))
    f.config['encoder_dimension'] = 3072
    f.bags = [[np.pad(fields, ((0, 0), (0, 3069))) for fields in views] for views in f.bags]
    for case, views in zip(f.cases, f.bags):
        case['indices'] = {view: ['fixture/%s/%d/%d.png'%(case['case_id'], view, index) for index in range(len(fields))]
                           for view, fields in zip(MAGNIFICATIONS, views)}
    f.arm = 'dual'; f.seed = 812; f.trial = 1
    train, val, excluded = f.roles[f.seed, f.trial]
    mean, std = fit_normalization(f.bags, train)
    arrays = {'mean': mean, 'std': std}
    for head, outputs, classes in [('binary', 1, np.arange(2)), ('malignant', 4, np.arange(1, 5))]:
        arrays[head+'_coef'] = np.zeros((outputs, 3076), np.float64)
        arrays[head+'_coef'][:, :3] = (np.arange(outputs*3).reshape(outputs, 3)/100).astype(np.float32).astype(np.float64)
        arrays[head+'_intercept'] = np.linspace(-.3, .3, outputs).astype(np.float32).astype(np.float64)
        arrays[head+'_classes'] = classes
    f.common = dict(f.context, lora_config_sha256='f'*64)
    initial = f.config['initial_model_sha256']['812/1/dual']
    f.context = dict(f.common, arm=f.arm, seed=f.seed, trial=f.trial, initial_model_sha256=initial)
    f.folder = root/'runs/dual/split812/trial1'; f.folder.mkdir(parents=True)
    f.audit_sha = 'a'*64
    state = {}
    for name, shape in m._state_schema(f.arm, f.config).items():
        value = arrays[name] if name in ('mean', 'std') else np.full(shape, .01 if name.endswith('lora_A') else 0, np.float32)
        if name.startswith('heads.'):
            _, head, parameter = name.split('.')
            value = arrays[head+('_coef' if parameter == 'weight' else '_intercept')]
        state[name] = torch.tensor(value, dtype=torch.float32)
    f.state = state
    probes = [path for view in MAGNIFICATIONS for path in sorted(path for index in train for path in f.cases[index]['indices'][view])[:2]]
    parity = dict(f.config['parity'], passed=True, feature_max_abs=0., probability_max_abs=0., probe_paths=probes)
    candidates = []
    for epoch in f.config['candidate_epochs']:
        directory = f.folder/('epoch%d'%epoch); directory.mkdir()
        torch.save({'context': f.context, 'epoch': epoch, 'small_state': state}, directory/'state.pt')
        np.savez_compressed(directory/'model.npz', **arrays)
        paths = io.expected_paths(f.cases, val)
        values = np.concatenate([f.bags[index][view] for index in val for view in range(4)])
        np.savez_compressed(directory/'validation-features.npz', paths=np.asarray(paths), features=values)
        rows = score_records(arrays, f.bags, val, f.cases, 'validation', f.arm)[0]
        (directory/'validation-predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metric = {'epoch': epoch, 'validation': summarize_records(rows, f.arm),
                  'state_sha256': file_hash(directory/'state.pt'), 'model_sha256': file_hash(directory/'model.npz'),
                  'validation_features_sha256': file_hash(directory/'validation-features.npz'),
                  'validation_predictions_sha256': file_hash(directory/'validation-predictions.jsonl')}
        write_json(directory/'metrics.json', metric); candidates.append(metric)
    selected = f.folder/'selected'; selected.mkdir()
    torch.save({'context': f.context, 'epoch': 2, 'small_state': state}, selected/'state.pt')
    np.savez_compressed(selected/'model.npz', **arrays)
    paths = io.expected_paths(f.cases, train)
    values = np.concatenate([f.bags[index][view] for index in train for view in range(4)])
    np.savez_compressed(selected/'train-features.npz', paths=np.asarray(paths), features=values)
    rows = score_records(arrays, f.bags, train, f.cases, 'train', f.arm)[0]
    (selected/'train-predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    selected_metrics = {'epoch': 2, 'fitting': summarize_records(rows, f.arm),
                        'state_sha256': file_hash(selected/'state.pt'), 'model_sha256': file_hash(selected/'model.npz'),
                        'train_features_sha256': file_hash(selected/'train-features.npz'),
                        'train_predictions_sha256': file_hash(selected/'train-predictions.jsonl')}
    write_json(selected/'metrics.json', selected_metrics)
    history = []
    for epoch in range(1, 21):
        order = io.case_order(train, io.run_seed(812, 1), epoch)
        receipts = [{'case_index': index, 'draws': io.sampled_paths(f.cases[index], io.run_seed(812, 1), epoch)} for index in order]
        history.append({'epoch': epoch, 'adapting': epoch > 2, 'cases': 46, 'optimizer_updates': 12, 'microbatches': 46,
                        'case_order': order, 'sampling_sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
                        'learning_rates': [1e-4*.5*(1+math.cos(math.pi*((epoch-1)*12+step)/240)) for step in range(12)],
                        'gradient_steps': [{'A': .1 if epoch > 2 else 0, 'B': .1 if epoch > 2 else 0, 'heads': .1} for _ in range(12)],
                        'loss_sum': 1., 'small_digests': io.small_digests(state), 'elapsed_seconds': .1})
    schema = m._state_schema(f.arm, f.config)
    names = [name for name in schema if name not in ('mean', 'std')]
    parameters = [{'name': name, 'shape': list(schema[name]), 'count': math.prod(schema[name]),
                   'requires_grad': name.startswith('heads.')} for name in names]
    manifest = {'parameters': parameters, 'adapter_parameters': 98304, 'head_parameters': 15385,
                'total_parameters': 113689, 'active_parameters': 15385}
    malignant_steps = sum(any(f.y5[index] > 0 for index in row['case_order'][start:start+4]) for row in history for start in range(0, 46, 4))
    optimizer = {'param_groups': [dict(f.config['optimizer'], params=list(range(len(names))))], 'state': {}}
    for index, name in enumerate(names):
        step = 216 if name.startswith('backbone.') else malignant_steps if name.startswith('heads.malignant.') else 240
        optimizer['state'][index] = {'step': torch.tensor(float(step)), 'exp_avg': torch.zeros(schema[name]), 'exp_avg_sq': torch.zeros(schema[name])}
    last = {'context': f.context, 'epoch': 20, 'small_state': state, 'optimizer': optimizer,
            'scheduler': {'last_epoch': 240, 'T_max': 240}, 'rng': io.rng_state(), 'history': history,
            'candidates': candidates, 'best_epoch': 2, 'initial_parity': parity, 'frozen_digest_initial': 'b'*64,
            'updates': 240, 'initial_small_digests': io.small_digests(state)}
    torch.save(last, f.folder/'last.pt')
    report = {'status': 'complete Midnight LoRA validation run', **f.context, 'class_names': list(m.FINAL5),
              'original_class_names': list(m.CLASS_NAMES), 'role_counts': [46, 8, 28], 'outer_predictions': 0,
              'case_ids': {role: [f.cases[index]['case_id'] for index in indices] for role, indices in
                           [('train', train), ('validation', val), ('excluded', excluded)]},
              'audit_sha256': f.audit_sha, 'history': history, 'candidates': candidates, 'selected_epoch': 2,
              'selected_validation': candidates[0]['validation'], 'selected_fitting': selected_metrics,
              'initial_parity': parity, 'reload_parity': parity, 'frozen_digest_initial': 'b'*64,
              'frozen_digest_final': 'b'*64, 'trainable_manifest': manifest,
              'last_checkpoint_sha256': file_hash(f.folder/'last.pt')}
    write_json(f.folder/'run.json', report)
    f.report = report
    return f


@unittest.skipIf(torch is None, 'torch is required for small-state collector fixtures')
class LoraCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name); cls.f = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.original = {}

    def tearDown(self):
        for path, value in self.original.items():
            path.write_bytes(value)

    def remember(self, path):
        if path not in self.original: self.original[path] = path.read_bytes()

    def report(self, value):
        path = self.f.folder/'run.json'; self.remember(path); write_json(path, value)

    def candidate(self, epoch, change, filename=None, content=None):
        folder = self.f.folder/('epoch%d'%epoch)
        if filename:
            path = folder/filename; self.remember(path)
            if filename.endswith('.pt'): torch.save(content, path)
            elif filename.endswith('.npz'): np.savez_compressed(path, **content)
            else: path.write_text(content)
        metric_path = folder/'metrics.json'; self.remember(metric_path); metric = json.loads(metric_path.read_text())
        metric.update(change); write_json(metric_path, metric)
        report = copy.deepcopy(self.f.report); index = self.f.config['candidate_epochs'].index(epoch)
        report['candidates'][index] = metric; self.report(report)

    def verify(self):
        f = self.f
        return m.verify_run(f.folder, f.arm, f.seed, f.trial, f.config, f.common, f.cases, f.bags, f.y5, f.roles, f.audit_sha)

    def test_complete_replay_uses_tiny_states_and_cannot_build_encoder_or_fit(self):
        from sklearn.linear_model import LogisticRegression
        from magset import midnight_features
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No fit')), patch.object(midnight_features, 'load_model', side_effect=AssertionError('No encoder')):
            result = self.verify()
        self.assertEqual(result['selected_epoch'], 2)
        self.assertEqual(len(result['candidates']), 4)
        self.assertEqual(result['trainable_manifest']['total_parameters'], 113689)
        self.assertNotIn('probe_paths', result['initial_parity'])

    def test_head_state_mismatch_fails_even_with_updated_state_and_json_hashes(self):
        path = self.f.folder/'epoch2/state.pt'; saved = torch.load(path, weights_only=True)
        saved['small_state']['heads.binary.bias'] += .1
        self.remember(path); torch.save(saved, path)
        self.candidate(2, {'state_sha256': file_hash(path)})
        with self.assertRaisesRegex(ValueError, 'head differs'):
            self.verify()

    def test_adapted_feature_forbidden_path_fails_after_rehash(self):
        path = self.f.folder/'epoch2/validation-features.npz'
        with np.load(path) as saved: arrays = {key: saved[key] for key in saved.files}
        excluded = self.f.roles[812, 1][2][0]
        paths = arrays['paths'].tolist(); paths[0] = self.f.cases[excluded]['indices'][40][0]; arrays['paths'] = np.asarray(paths)
        self.remember(path); np.savez_compressed(path, **arrays)
        self.candidate(2, {'validation_features_sha256': file_hash(path)})
        with self.assertRaisesRegex(ValueError, 'feature paths'):
            self.verify()

    def test_feature_dtype_and_normalizer_mutations_fail(self):
        path = self.f.folder/'epoch2/validation-features.npz'
        with np.load(path) as saved: arrays = {key: saved[key] for key in saved.files}
        arrays['features'] = arrays['features'].astype(np.float64)
        self.remember(path); np.savez_compressed(path, **arrays)
        self.candidate(2, {'validation_features_sha256': file_hash(path)})
        with self.assertRaises(ValueError): self.verify()

    def test_metric_selection_manifest_and_parity_changes_fail(self):
        mutations = [lambda r: r.update(selected_epoch=20),
                     lambda r: r['trainable_manifest'].update(total_parameters=1),
                     lambda r: r['reload_parity'].update(passed=False),
                     lambda r: r['history'][0].update(sampling_sha256='f'*64),
                     lambda r: r.update(frozen_digest_final='f'*64)]
        for change in mutations:
            report = copy.deepcopy(self.f.report); change(report); self.report(report)
            with self.subTest(change=change), self.assertRaises(ValueError): self.verify()

    def test_optimizer_update_count_cannot_be_hidden_by_new_last_hash(self):
        path = self.f.folder/'last.pt'; self.remember(path)
        saved = torch.load(path, weights_only=True); saved['optimizer']['state'][0]['step'] = torch.tensor(240.)
        torch.save(saved, path); report = copy.deepcopy(self.f.report); report['last_checkpoint_sha256'] = file_hash(path); self.report(report)
        with self.assertRaisesRegex(ValueError, 'Optimizer update count'): self.verify()

    def test_extra_role_scores_prevent_run_acceptance(self):
        path = self.f.folder/'excluded-predictions.jsonl'; path.write_text('{}\n')
        try:
            with self.assertRaisesRegex(ValueError, 'Unexpected run artifacts'): self.verify()
        finally:
            path.unlink()

    def test_public_aggregation_compares_case_field_and_kernel(self):
        value = self.verify(); f = self.f
        runs = [dict(value, seed=seed, trial=trial, arm=arm) for arm in m.ARMS for seed, trial in f.roles]
        summary = {str(seed): {'dual': {'validation': value['validation']}} for seed in f.config['seeds']}
        reference = {'code_sha': 'b'*40, 'summary': {'case': summary, 'field': summary}}
        kernel = {'summary': summary}
        result = m.aggregate(runs, f.config, f.common, f.cases, f.roles, reference, kernel, f.audit_sha)
        self.assertEqual(result['verified_runs'], 20)
        self.assertIn('comparison_against_kernel_dual', result)
        self.assertEqual(result['comparison_against_kernel_dual']['812']['dual']['delta_mean_subset_accuracy'], 0)
        self.assertFalse(result['gate_passed'])
        self.assertNotIn('probe_paths', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
