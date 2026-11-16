"""Support, role, metric and complete-matrix tampering fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import kernel_collect as m
from magset import kernel_five as runner
from magset.data import file_hash
from magset.five_class import write_json
from test_kernel_five import fixture


class KernelCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)
        cls.folder = cls.root/'kernel'
        args = SimpleNamespace(output=cls.folder, audit=cls.f.audit_dir, seed=812, trial=1)
        with patch.object(runner, 'load_inputs', return_value=cls.f.inputs): runner.run(args)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self): self.saved = {}
    def tearDown(self):
        for path, data in self.saved.items(): path.write_bytes(data)

    def write(self, path, data):
        self.saved.setdefault(path, path.read_bytes()); write_json(path, data)

    def verify(self):
        f = self.f
        return m.verify_run(self.folder, 812, 1, f.config, f.context, f.cases, f.bags,
                            f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))

    def change_candidate(self, candidate, arm='direct', index=0):
        self.write(self.folder/arm/('candidate%d'%index)/'metrics.json', candidate)
        report = m.read_json(self.folder/'run.json'); report['arms'][arm]['candidates'][index] = candidate
        self.write(self.folder/'run.json', report)

    def test_complete_replay_constructs_supports_but_never_optimizes(self):
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('Collector must not optimize')):
            result = self.verify()
        self.assertEqual(result['optimized_head_fits'], 0)
        self.assertEqual(result['support_classifier_instances'], 3)
        m.assert_public_safe(result)
        self.assertNotIn('support_case_ids', json.dumps(result))
        self.assertIn('fitting_leave_case_out', result['arms']['dual'])

    def test_changed_support_case_or_center_policy_fails_despite_updated_hashes(self):
        path = self.folder/'direct/candidate0/model.json'
        model = m.read_json(path); model['support_case_ids'][0] = 'excluded-fixture'
        self.write(path, model)
        candidate = m.read_json(path.with_name('metrics.json')); candidate['model_sha256'] = file_hash(path)
        self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'support model'): self.verify()
        for p, data in self.saved.items(): p.write_bytes(data)
        model = m.read_json(path); model['normalization']['centering'] = 'include query statistics'
        self.write(path, model); candidate['model_sha256'] = file_hash(path); self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'support model'): self.verify()

    def test_metric_selection_and_support_fit_count_tampering_fail(self):
        candidate = m.read_json(self.folder/'direct/candidate0/metrics.json')
        candidate['fitting_leave_case_out']['mean_subset_accuracy'] += .1
        self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'candidate/fitting_leave_case_out'): self.verify()
        for p, data in self.saved.items(): p.write_bytes(data)
        report = m.read_json(self.folder/'run.json'); report['optimized_head_fits'] = 1
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'optimized_head_fits'): self.verify()
        for p, data in self.saved.items(): p.write_bytes(data)
        report = m.read_json(self.folder/'run.json'); report['arms']['direct']['selected_index'] = (report['arms']['direct']['selected_index']+1)%4
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'earliest temperature'): self.verify()

    def reference(self):
        f = self.f; accepted = m.read_json(self.folder/'run.json')
        runs = []
        for s, t in f.roles:
            arms = copy.deepcopy(accepted['arms'])
            for a in runner.ARMS: arms[a]['fitting'] = arms[a].pop('fitting_leave_case_out')
            runs.append({'readout': 'case', 'seed': s, 'trial': t, 'arms': arms})
        return {**copy.deepcopy(f.context), 'class_names': accepted['class_names'], 'original_class_names': accepted['original_class_names'],
                'verified_runs': 20, 'head_fits': 240, 'readouts': ['case', 'field'], 'outer_predictions': 0,
                'runs': runs, 'summary': {'case': {str(s): {'dual': {'validation': {'mean_subset_macro_f1': .2, 'mean_subset_accuracy': .5}}} for s in (812, 813)}}}

    def test_reference_must_be_same_midnight_cache_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reference.json'; ref = self.reference(); write_json(path, ref)
            config = dict(self.f.config, reference_comparison_sha256=file_hash(path))
            self.assertEqual(m.reference_comparison(config, self.f.context, path)['verified_runs'], 20)
            ref['encoder']['dimension'] = 999; write_json(path, ref); config['reference_comparison_sha256'] = file_hash(path)
            context = copy.deepcopy(self.f.context); context['encoder']['dimension'] = 3
            with self.assertRaisesRegex(ValueError, 'reference/encoder'): m.reference_comparison(config, context, path)

    def test_full_matrix_gates_and_partial_matrix_rejection(self):
        f = self.f; report = m.read_json(self.folder/'run.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'
            for s, t in f.roles:
                folder = runs/('split%d'%s)/('trial%d'%t); folder.mkdir(parents=True); (folder/'run.json').write_text('{}')
                for a in runner.ARMS:
                    for i in range(4):
                        directory = folder/a/('candidate%d'%i); directory.mkdir(parents=True)
                        for name in ('model.json', 'train-predictions.jsonl', 'validation-predictions.jsonl'): (directory/name).write_text('{}')
            failing = [False]
            def verified(folder, s, t, *unused):
                arms = copy.deepcopy(report['arms'])
                for a in runner.ARMS:
                    arms[a]['validation']['mean_subset_macro_f1'] = .5
                    arms[a]['validation']['mean_subset_accuracy'] = .7
                    arms[a]['validation']['pooled']['confusion_matrix'] = (np.eye(5)*15).tolist()
                if failing[0] and s == 813: arms['dual']['validation']['mean_subset_accuracy'] = .1
                return {'seed': s, 'trial': t, 'arms': arms, 'optimized_head_fits': 0, 'support_classifier_instances': 3}
            args = SimpleNamespace(runs=runs, audit=f.audit_dir, output=root/'accepted')
            for fail in (False, True):
                failing[0] = fail; args.output = root/str(fail)
                with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(m, 'reference_comparison', return_value=self.reference()), patch.object(m, 'verify_run', side_effect=verified): result = m.collect(args)
                self.assertEqual(result['gate_passed'], not fail)
                self.assertEqual(result['support_classifier_instances'], 30)
                self.assertEqual(result['optimized_head_fits'], 0)
                self.assertEqual(result['verified_case_subset_records'], 64800)
                self.assertEqual(result['verified_prediction_archives'], 160)
                self.assertEqual(result['candidate_pipelines'], 80)
                m.assert_public_safe(result)
            (runs/'split813/trial5/run.json').unlink(); args.output = root/'partial'
            with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(m, 'reference_comparison', return_value=self.reference()), patch.object(m, 'verify_run') as verify:
                with self.assertRaisesRegex(ValueError, 'Complete kernel artifact'): m.collect(args)
                verify.assert_not_called()
            self.assertFalse(args.output.exists())


if __name__ == '__main__': unittest.main()
