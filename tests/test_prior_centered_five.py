"""Paired centering/controller checks using small synthetic feature dimensions."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import prior_centered_five as m, prior_centered_model as core
from magset.data import file_hash
from magset.five_class import write_json
from test_kernel_five import fixture as kernel_fixture


def fixture(root):
    f = kernel_fixture(root)
    own = m.read_json(Path('configs/breakhis-prior-centered-v1.json'))
    f.config.update(own)
    f.context.update(prior_centered_config_sha256='1'*64, prior_centered_base_config_sha256='2'*64,
                     coverage_config_sha256='2'*64, feature_audit_sha256='3'*64,
                     center_cache_provenance=copy.deepcopy(f.context['cache_provenance']))
    for name in ('data.py', 'baseline.py', 'five_metrics.py', 'bags.py', 'kernel_model.py', 'field_five.py'):
        f.context['implementation_hashes'][name] = '4'*64
    f.config['base_input_config_sha256'] = f.context['coverage_config_sha256']
    f.audit.update(f.context, head_center_checks=m.center_checks(f.bags, f.y5, f.roles))
    write_json(f.audit_dir/'audit.json', f.audit)
    f.inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
    return f


class PriorCenteredRunnerTests(unittest.TestCase):
    def test_loader_preserves_accepted_coverage_data_and_binds_new_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fixture(Path(tmp)); own = m.read_json('configs/breakhis-prior-centered-v1.json')
            own['base_input_config_sha256'] = 'b'*64
            args = SimpleNamespace(config=Path('registered.json'))
            with patch.object(m, 'file_hash', return_value='b'*64), patch.object(m, 'read_json', return_value=own), patch.object(m.source, 'load_inputs', return_value=f.inputs) as load:
                config, context, _, cases, bags, _, _, roles = m.load_inputs(args)
            self.assertEqual(load.call_args.args[0].config.name, 'breakhis-coverage-kernel-v1.json')
            self.assertIs(bags, f.bags); self.assertIs(cases, f.cases); self.assertIs(roles, f.roles)
            self.assertEqual(context['prior_centered_config_sha256'], 'b'*64)
            self.assertIn('prior_centered_collect.py', context['implementation_hashes'])
            self.assertEqual(config['normalization'], core.NORMALIZATION)

    def test_centers_remove_query_before_all_means_and_ignore_validation_truth(self):
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 2])
        bags = [[np.full((1+i%2, 3), [i, i*i, v], np.float32) for v in range(4)] for i in range(12)]
        prepared = m.prepare_queries(bags, labels, np.arange(11), [11])
        for q in prepared['train']:
            self.assertNotIn(q.query_index, q.support_indices)
            eligible = np.asarray(q.support_indices); means = np.array([bags[i][0].astype(float).mean(0) for i in eligible])
            perclass = np.array([means[labels[eligible] == k].mean(0) for k in range(5)])
            np.testing.assert_array_equal(q.heads['direct'].centers[0], perclass.mean(0))
            np.testing.assert_array_equal(q.heads['malignant'].centers[0], perclass[1:].mean(0))
        cases = [{'case_id': 'toy%d'%i, 'label': [0, 1, 3, 4, 5][k]} for i, k in enumerate(labels)]
        q1 = prepared['validation'][0]; changed = labels.copy(); changed[11] = 0
        q2 = core.prepare_case(bags, changed, np.arange(11), 11)
        for mode in m.MODES:
            c1, c2 = core.density_cache(q1, .1), core.density_cache(q2, .1)
            r1 = m.score_records([q1], cases, 'validation', 'dual', .1, mode, [c1])
            r2 = m.score_records([q2], cases, 'validation', 'dual', .1, mode, [c2])
            self.assertEqual(r1, r2); self.assertEqual(len(r1), 15)

    def test_audit_binds_actual_fit_centers_separately_from_input_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fixture(Path(tmp)); args = SimpleNamespace(output=Path(tmp)/'newaudit')
            with patch.object(m, 'load_inputs', return_value=f.inputs): m.audit_input(args)
            audit = m.read_json(args.output/'audit.json')
            m.verify_input_audit(audit, f.context, f.config, f.bags, f.y5, f.roles)
            self.assertEqual(len(audit['head_center_checks']), 10)
            self.assertEqual(audit['new_fits'], 0)
            audit['head_center_checks'][0]['support_summary']['centers_sha256']['binary'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'audit/head centers'):
                m.verify_input_audit(audit, f.context, f.config, f.bags, f.y5, f.roles)

    def test_paired_run_reuses_query_temperature_caches_and_duplicate_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = fixture(root)
            args = SimpleNamespace(output=root/'paired', audit=f.audit_dir, seed=812, trial=1)
            with patch.object(m, 'load_inputs', return_value=f.inputs), patch.object(core, 'density_cache', wraps=core.density_cache) as cache, patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No coefficient optimization')):
                m.run(args)
            self.assertEqual(cache.call_count, 54*4)
            report = m.read_json(args.output/'run.json')
            self.assertEqual(report['optimized_head_fits'], 0)
            self.assertEqual(len(list(args.output.rglob('model.json'))), 16)
            self.assertEqual(len(list(args.output.rglob('*predictions.jsonl'))), 32)
            for mode in m.MODES:
                for arm in m.ARMS:
                    saved = report['modes'][mode]['arms'][arm]
                    wanted = max(range(4), key=lambda i: saved['candidates'][i]['validation']['mean_subset_macro_f1'])
                    self.assertEqual(saved['selected_index'], wanted)
                    model = m.read_json(args.output/mode/arm/'candidate0/model.json')
                    self.assertEqual(model['normalization'], core.NORMALIZATION)
                    self.assertTrue(model['fitting_labels_used_for_centers'])
            for index in range(4):
                for role in ('train', 'validation'):
                    paths = [args.output/mode/'direct'/('candidate%d'%index)/(role+'-predictions.jsonl') for mode in m.MODES]
                    self.assertEqual(file_hash(paths[0]), file_hash(paths[1]))
            with self.assertRaisesRegex(ValueError, 'Refusing existing'): m.run(args)


if __name__ == '__main__': unittest.main()
