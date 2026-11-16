"""Collector corruption and independent numerical fixtures; no classifier fitting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import field_collect as m
from magset import five_class as base
from magset.data import file_hash
from magset.five_metrics import summarize_records
from test_five_collect import fixture as base_fixture


def fixture(root):
    f = base_fixture(root)
    f.config.update(json.loads(Path('configs/breakhis-phikon-field-v1.json').read_text()))
    f.context['field_config_sha256'] = 'b70bf70726dd733c2ac357198e5047ed27aa8e35454e172b5797af3300349e15'
    f.audit.update(f.context); base.write_json(f.audit_dir/'audit.json', f.audit)
    f.report.update(f.context); f.report['status'] = 'complete field-supervised five-class validation run'
    f.report['audit_sha256'] = file_hash(f.audit_dir/'audit.json')
    train, val, _ = f.roles[812, 1]
    for arm in base.ARMS:
        candidates = []
        for index, C in enumerate(f.config['C_grid']):
            folder = f.folder/arm/('candidate%d'%index)
            candidate = m.read_json(folder/'metrics.json')
            arrays = base.load_model(folder/'model.npz', arm, dimension=3)
            names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
            rows = {h: m.training_rows(f.bags, train, f.y5, arrays['mean'], arrays['std'], h) for h in names}
            candidate['field_receipts'] = m.field_receipts(rows)
            candidate['head_objectives'] = m.head_objectives(arrays, rows, C, f.config)
            for role, indices, metric in [('train', train, 'fitting'), ('validation', val, 'validation')]:
                records = m.replay_records(arrays, f.bags, indices, f.cases, role, arm)
                p = folder/(role+'-predictions.jsonl'); p.write_text(''.join(json.dumps(r)+'\n' for r in records))
                candidate[role+'_predictions_sha256'] = file_hash(p)
                candidate[metric] = summarize_records(records, arm)
            base.write_json(folder/'metrics.json', candidate); candidates.append(candidate)
        f.report['arms'][arm] = {'selected_index': 0, 'selected_C': .01, 'candidates': candidates,
                                 'fitting': candidates[0]['fitting'], 'validation': candidates[0]['validation']}
    base.write_json(f.folder/'run.json', f.report)
    return f


class FieldCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.f = fixture(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.saved = {}

    def tearDown(self):
        for p, content in self.saved.items():
            p.write_bytes(content)

    def write(self, p, value):
        p = Path(p)
        self.saved.setdefault(p, p.read_bytes())
        p.write_text(json.dumps(value))

    def change_candidate(self, candidate, arm='direct', index=0):
        self.write(self.f.folder/arm/('candidate%d'%index)/'metrics.json', candidate)
        report = m.read_json(self.f.folder/'run.json'); report['arms'][arm]['candidates'][index] = candidate
        self.write(self.f.folder/'run.json', report)

    def verify(self):
        f = self.f
        return m.verify_run(f.folder, 812, 1, f.config, f.context, f.cases, f.bags,
                            f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))

    def test_complete_replay_is_public_safe_and_cannot_fit(self):
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('Collector must not fit')):
            result = self.verify()
        self.assertEqual(result['head_fits'], 12)
        self.assertEqual(result['arms']['dual']['selected_index'], 0)
        m.assert_public_safe(result)
        self.assertNotIn('fixture-', json.dumps(result))

    def test_head_weights_have_registered_class_and_case_mass(self):
        f = self.f; train = f.roles[812, 1][0]
        mean, std = m.fit_normalization(f.bags, train)
        for head, n, k in [('direct', 46, 5), ('binary', 46, 2), ('malignant', 35, 4)]:
            x, y, w = m.training_rows(f.bags, train, f.y5, mean, std, head)
            self.assertAlmostEqual(float(w.sum()), n)
            for label in np.unique(y):
                self.assertAlmostEqual(float(w[y == label].sum()), n/k)
            self.assertTrue(np.all(x[:, -4:].sum(1) == 1))
            self.assertTrue(np.all(w > 0))

    def test_manual_joint_pool_matches_explicit_arithmetic_and_not_product_of_means(self):
        f = self.f; folder = f.folder/'dual/candidate0/model.npz'
        arrays = base.load_model(folder, 'dual', dimension=3)
        records = m.replay_records(arrays, f.bags, [0], f.cases, 'validation', 'dual')
        r = next(r for r in records if r['magnifications'] == [40])
        fields = np.asarray(f.bags[0][0], np.float32); mask = np.tile([1, 0, 0, 0], (len(fields), 1))
        x = np.column_stack(((fields-arrays['mean'][0])/arrays['std'][0], mask))
        b = np.exp(m.log_probabilities(m.logits(x, arrays, 'binary')))
        q = np.exp(m.log_probabilities(m.logits(x, arrays, 'malignant')))
        expected = np.column_stack((b[:, 0], b[:, 1, None]*q)).mean(0)
        np.testing.assert_allclose(r['probabilities5'], expected, atol=1e-14)
        np.testing.assert_allclose(r['malignant_probabilities4'], expected[1:]/expected[1:].sum(), atol=1e-14)
        self.assertGreater(np.max(np.abs(expected[1:]-b[:, 1].mean()*q.mean(0))), 1e-10)

    def test_objective_receipts_and_weights_cannot_be_coordinately_rehashed(self):
        candidate = m.read_json(self.f.folder/'direct/candidate0/metrics.json')
        candidate['head_objectives']['direct']['objective_sum'] += 1
        self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'head objective'):
            self.verify()
        for p, content in self.saved.items(): p.write_bytes(content)
        candidate = m.read_json(self.f.folder/'direct/candidate0/metrics.json')
        candidate['field_receipts']['weight_mass_per_head']['direct'] /= 15
        self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'field receipt'):
            self.verify()

    def test_selection_metric_context_and_normalizer_tampering_fail(self):
        for mode in ['selection', 'metric', 'context', 'normalizer']:
            for p, content in self.saved.items(): p.write_bytes(content)
            report = m.read_json(self.f.folder/'run.json')
            if mode == 'selection': report['arms']['dual']['selected_index'] = 1
            elif mode == 'context': report['field_config_sha256'] = '0'*64
            elif mode == 'metric':
                candidate = copy.deepcopy(report['arms']['direct']['candidates'][0])
                candidate['validation']['mean_subset_macro_f1'] += .1
                self.change_candidate(candidate)
            else:
                p = self.f.folder/'direct/candidate0/model.npz'; self.saved.setdefault(p, p.read_bytes())
                with np.load(p, allow_pickle=False) as data: arrays = dict(data)
                arrays['mean'][0, 0] += 1
                np.savez_compressed(p, **arrays)
                candidate = copy.deepcopy(report['arms']['direct']['candidates'][0]); candidate['model_sha256'] = file_hash(p)
                self.change_candidate(candidate)
            if mode in ['selection', 'context']: self.write(self.f.folder/'run.json', report)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.verify()

    def test_reference_requires_exact_hash_and_matching_task(self):
        config = dict(self.f.config); config['reference_comparison_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'reference hash'):
            m.reference_comparison(config, self.f.context)
        with self.assertRaisesRegex(ValueError, 'reference/manifest'):
            m.reference_comparison(self.f.config, self.f.context)

    def test_independent_replay_agrees_with_producer_fields_weights_and_objectives(self):
        from magset import field_five
        f = self.f; train, val, _ = f.roles[812, 1]
        for arm in base.ARMS:
            arrays = base.load_model(f.folder/arm/'candidate0/model.npz', arm, dimension=3)
            names = ('direct',) if arm == 'direct' else ('binary', 'malignant')
            rows = {}; models = {}
            for head in names:
                rows[head] = m.training_rows(f.bags, train, f.y5, arrays['mean'], arrays['std'], head)
                produced = field_five.field_training_rows(f.bags, train, f.y5, arrays['mean'], arrays['std'], head)
                for actual, expected in zip(rows[head], produced):
                    np.testing.assert_array_equal(actual, expected)
                models[head] = SimpleNamespace(coef_=arrays[head+'_coef'], intercept_=arrays[head+'_intercept'], classes_=arrays[head+'_classes'])
            receipt, objective = field_five.fitting_diagnostics(models, f.bags, train, f.y5, arrays['mean'], arrays['std'], .01)
            m.require_equal(m.field_receipts(rows), receipt, 'independent field receipts', 1e-10)
            m.require_equal(m.head_objectives(arrays, rows, .01, f.config), objective, 'independent objective', 1e-8)
            expected = m.replay_records(arrays, f.bags, val, f.cases, 'validation', arm)
            actual, _ = field_five.score_records(arrays, f.bags, val, f.cases, 'validation', arm)
            key = lambda r: (r['case_id'], tuple(r['magnifications']))
            actual = sorted(actual, key=key); expected = sorted(expected, key=key)
            m.require_equal(actual, expected, 'independent scorer', 1e-12)

    def test_full_matrix_gate_is_against_fixed_old_dual_on_both_arrangements(self):
        from magset import field_five
        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'
            for seed, trial in f.roles:
                folder = runs/('split%d'%seed)/('trial%d'%trial)
                folder.mkdir(parents=True); (folder/'run.json').write_text('{}')
                for arm in base.ARMS:
                    for index in range(4):
                        directory = folder/arm/('candidate%d'%index); directory.mkdir(parents=True)
                        for name in ('model.npz', 'train-predictions.jsonl', 'validation-predictions.jsonl'):
                            (directory/name).write_text('coefficient replay covered by separate tests')
            inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
            reference = {'code_sha': '1'*40, 'summary': {str(s): {'dual': {'validation': {'mean_subset_macro_f1': .2, 'mean_subset_accuracy': .5}}} for s in (812, 813)}}
            failing = [False]
            def verified(folder, seed, trial, *unused):
                arms = copy.deepcopy(f.report['arms'])
                for arm in base.ARMS:
                    value = arms[arm]['validation']
                    value['mean_subset_macro_f1'] = .3 if arm == 'direct' else .4
                    value['mean_subset_accuracy'] = .6
                    value['pooled']['confusion_matrix'] = (np.eye(5)*15).tolist()
                if failing[0] and seed == 813: arms['dual']['validation']['mean_subset_accuracy'] = .4
                return {'seed': seed, 'trial': trial, 'head_fits': 12, 'run_sha256': '0'*64, 'arms': arms}
            for fail in (False, True):
                failing[0] = fail
                args = SimpleNamespace(runs=runs, audit=f.audit_dir, output=root/str(fail))
                with patch.object(field_five, 'load_inputs', return_value=inputs), patch.object(m, 'reference_comparison', return_value=reference), patch.object(m, 'verify_run', side_effect=verified):
                    result = m.collect(args)
                self.assertEqual(result['gate_passed'], not fail)
                self.assertEqual(result['head_fits'], 120)
                self.assertEqual(result['candidate_pipelines'], 80)
                self.assertEqual(result['collector_new_fits'], 0)
                self.assertEqual(result['new_encoder_forwards'], 0)
                m.assert_public_safe(result)
                self.assertTrue((args.output/'comparison.json').exists())

    def test_partial_matrix_never_publishes_or_ranks(self):
        from magset import field_five
        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(runs=Path(tmp)/'runs', audit=f.audit_dir, output=Path(tmp)/'output')
            inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
            with patch.object(field_five, 'load_inputs', return_value=inputs), patch.object(m, 'reference_comparison', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Complete registered run matrix'):
                    m.collect(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
