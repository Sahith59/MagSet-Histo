"""Tiny support/controller fixtures; no actual cohort or optimized fitting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import kernel_five as m
from magset import kernel_model as core
from magset.five_metrics import five_labels, summarize_records
from magset.five_class import write_json
from test_five_collect import fixture as base_fixture


def fixture(root):
    old = root/'old'; old.mkdir(); f = base_fixture(old)
    f.config.update(name='breakhis-midnight-kernel-v1', schema_version=1, temperatures=m.TEMPERATURES,
        candidate_pipelines=80, support_classifier_instances=30, optimized_head_fits=0,
        encoder_name='midnight-12k', dimension=3072, arms=['direct', 'dual'], normalization=m.NORMALIZATION,
        training_diagnostic=m.FITTING_POLICY, reference_readout='case',
        support_instances_scope='three full-fit support heads per role, shared across temperatures; additional46 leave-one-case-out support evaluations per role are fitting diagnostics',
        new_encoder_forwards=0, outer_predictions=0,
        reference_comparison_sha256='a'*64,
        improvement_screen={'each_arrangement_macro_f1_delta': .02, 'accuracy_not_lower': True, 'positive_recall_all_five_classes': True})
    f.context['kernel_config_sha256'] = 'c'*64
    f.audit.update(f.context); write_json(f.audit_dir/'audit.json', f.audit)
    f.inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
    return f


class KernelRunnerTests(unittest.TestCase):
    def test_training_queries_remove_whole_case_from_support_and_center(self):
        bags = [[np.full((1+(i%2), 3), i+v, np.float32) for v in range(4)] for i in range(11)]
        labels = np.repeat(np.arange(5), 2).tolist()+[0]
        prepared = m.prepare_queries(bags, np.array(labels), np.arange(10), [10])
        self.assertEqual(len(prepared['train']), 10)
        for query in prepared['train']:
            self.assertNotIn(query.query_index, query.support_indices)
            self.assertEqual(len(query.support_indices), 9)
            expected = np.mean([bags[i][0].astype(float).mean(0) for i in range(10) if i != query.query_index], axis=0)
            np.testing.assert_array_equal(query.centers[0], expected)
        self.assertEqual(prepared['validation'][0].support_indices, tuple(range(10)))
        self.assertEqual(m.NORMALIZATION, core.NORMALIZATION)

    def test_records_match_manual_uniform_kde_joint_pool_and_all_subsets(self):
        bags = [[np.array([[i/5, 1., v/3]], np.float32) for v in range(4)] for i in range(11)]
        labels = np.array(list(np.repeat(np.arange(5), 2))+[1])
        cases = [{'case_id': 'fixture-%d'%i, 'label': [0, 1, 3, 4, 5][labels[i]]} for i in range(11)]
        query = core.prepare_case(bags, labels, np.arange(10), 10)
        records = m.score_records([query], cases, 'validation', 'dual', .1)
        row = next(r for r in records if r['magnifications'] == [40])
        cosine = query.views[0].cosine[0]; w = query.views[0].weights
        mass = np.exp(cosine/.1-cosine.max()/.1) @ w
        binary = mass[5:7]/mass[5:7].sum(); malignant = mass[7:]/mass[7:].sum()
        expected = np.r_[binary[0], binary[1]*malignant]
        np.testing.assert_allclose(row['probabilities5'], expected, rtol=0, atol=1e-14)
        self.assertEqual(len(records), 15)
        self.assertEqual(len({tuple(r['magnifications']) for r in records}), 15)
        self.assertEqual(row['prediction'], 0 if binary[1] < .5 else 1+int(malignant.argmax()))

    def test_mocked_base_loader_preserves_cache_and_binds_new_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fixture(Path(tmp))
            kernel = {k: f.config[k] for k in ('name', 'schema_version', 'temperatures', 'production_runs', 'candidate_pipelines', 'support_classifier_instances', 'optimized_head_fits', 'new_encoder_forwards', 'outer_predictions', 'encoder_name', 'dimension', 'arms', 'normalization', 'training_diagnostic', 'reference_readout', 'routing_threshold', 'routing_tie', 'improvement_screen')}
            kernel.update(base_input_config='configs/breakhis-midnight-five-v1.json', base_input_config_sha256='b'*64)
            args = SimpleNamespace(config=Path('registered.json'))
            with patch.object(m, 'file_hash', return_value='b'*64), patch.object(m, 'read_json', return_value=kernel), patch.object(m.base, 'load_inputs', return_value=f.inputs) as load:
                config, context, *rest = m.load_inputs(args)
            self.assertEqual(context['kernel_config_sha256'], 'b'*64)
            self.assertIn('kernel_model.py', context['implementation_hashes'])
            self.assertEqual(load.call_args.args[0].config.name, 'breakhis-midnight-five-v1.json')
            self.assertIs(rest[2], f.bags)
            self.assertEqual(config['temperatures'], [.01, .03, .1, .3])

    def test_synthetic_run_labels_loco_diagnostics_and_cannot_optimize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = fixture(root)
            args = SimpleNamespace(output=root/'kernel', audit=f.audit_dir, seed=812, trial=1)
            with patch.object(m, 'load_inputs', return_value=f.inputs), patch.object(LogisticRegression, 'fit', side_effect=AssertionError('KDE must not optimize')):
                m.run(args)
            report = json.loads((args.output/'run.json').read_text())
            self.assertEqual(report['optimized_head_fits'], 0)
            self.assertEqual(report['support_classifier_instances'], 3)
            for arm in m.ARMS:
                self.assertNotIn('fitting', report['arms'][arm])
                self.assertIn('fitting_leave_case_out', report['arms'][arm])
                self.assertEqual(len(report['arms'][arm]['candidates']), 4)
                selected = max(range(4), key=lambda i: report['arms'][arm]['candidates'][i]['validation']['mean_subset_macro_f1'])
                self.assertEqual(report['arms'][arm]['selected_index'], selected)
            self.assertEqual(len(list(args.output.rglob('model.json'))), 8)
            self.assertEqual(len(list(args.output.rglob('*predictions.jsonl'))), 16)
            self.assertFalse(list(args.output.rglob('*.npz')))


if __name__ == '__main__': unittest.main()
