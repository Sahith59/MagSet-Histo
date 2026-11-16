"""Two-readout controller and delegated replay fixtures; no model fitting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from magset import midnight_collect as m
from magset.data import file_hash
from magset.five_class import write_json
from test_five_collect import fixture as case_fixture
from test_field_collect import fixture as field_fixture


def config_from(f):
    config = copy.deepcopy(f.config)
    config.update(readouts=['case', 'field'], total_production_runs=20, total_head_fits=240,
                  reference_comparison_sha256='bb97c03d22e4299da5ebd7a629cfd9d8c4693d3454feb05211607f7b80a4a795',
                  readout_definitions={'case': 'mean frozen fields then equal available-view mean plus availability indicators; balanced case heads',
                                       'field': 'balanced field heads then mean joint probabilities within and across available views'},
                  improvement_screen={'each_arrangement_macro_f1_delta': .02,
                                      'accuracy_not_lower': True, 'positive_recall_all_five_classes': True})
    return config


class MidnightCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); root = Path(cls.temp.name)
        cls.fixtures = {}
        for readout, builder in [('case', case_fixture), ('field', field_fixture)]:
            directory = root/readout; directory.mkdir()
            f = builder(directory); f.config = config_from(f)
            f.context['readout'] = readout
            f.report.update(f.context)
            write_json(f.folder/'run.json', f.report)
            cls.fixtures[readout] = f

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_delegated_case_and_field_replay_require_readout_context_without_fitting(self):
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No collector fits')):
            for readout, f in self.fixtures.items():
                common = {k: v for k, v in f.context.items() if k != 'readout'}
                result = m.verify_run(f.folder, readout, 812, 1, f.config, common, f.cases,
                                      f.bags, f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))
                self.assertEqual(result['readout'], readout)
                self.assertEqual(result['head_fits'], 12)
                m.assert_public_safe(result)
                wrong = 'field' if readout == 'case' else 'case'
                with self.assertRaises(ValueError):
                    m.verify_run(f.folder, wrong, 812, 1, f.config, common, f.cases,
                                 f.bags, f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))

    def test_3072_dimensional_saved_heads_replay_both_readouts(self):
        from magset import five_class, field_collect
        from test_five_class import coefficient_models
        cases = [{'case_id': 'tiny-dimensional-fixture', 'label': 1}]
        bags = [[np.zeros((1, 3072), np.float32) for _ in range(4)]]
        z = np.zeros((1, 4, 3072), np.float32)
        mean = np.zeros((4, 3072), np.float32); std = np.ones_like(mean)
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('direct', 'dual'):
                path = Path(tmp)/(arm+'.npz')
                models = coefficient_models(arm, dimension=3072)
                five_class.save_model(path, mean, std, models)
                arrays = five_class.load_model(path, arm, dimension=3072)
                case_records, _ = five_class.score_records(arrays, z, [0], cases, 'validation', arm)
                field_records = field_collect.replay_records(arrays, bags, [0], cases, 'validation', arm)
                self.assertEqual(len(case_records), 15)
                self.assertEqual(len(field_records), 15)
                for row in field_records:
                    self.assertTrue(np.isfinite(row['log_probabilities5']).all())
                    self.assertAlmostEqual(sum(row['probabilities5']), 1)

    def test_reference_deliberately_allows_new_encoder_and_cache_but_checks_old_roles(self):
        f = self.fixtures['case']; reference_path = Path('docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json')
        reference = m.read_json(reference_path)
        context = {'manifest_sha256': reference['manifest_sha256'], 'protocol_sha256': reference['protocol_sha256'],
                   'encoder': {'repository': 'kaiko-ai/midnight', 'dimension': 3072}, 'cache_provenance': {'new': 'frozen'}}
        self.assertEqual(m.reference_comparison(f.config, context)['code_sha'], reference['code_sha'])
        context['protocol_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'reference/protocol'):
            m.reference_comparison(f.config, context)
        context['protocol_sha256'] = reference['protocol_sha256']
        for mode in ['role', 'support', 'labels']:
            changed = copy.deepcopy(reference)
            if mode == 'role': changed['runs'][0]['trial'] = 999
            elif mode == 'support': changed['runs'][0]['arms']['dual']['validation']['distinct_class_support'] = [1]*5
            else: changed['class_names'] = changed['class_names'][::-1]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'changed.json'; write_json(path, changed)
                config = dict(f.config, reference_comparison_sha256=file_hash(path))
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    m.reference_comparison(config, context, path)

    def test_matrix_contract_rejects_undeclared_partial_budgets_or_common_readout(self):
        f = self.fixtures['case']; context = {'encoder': {'dimension': 3072}}
        m.validate_matrix_contract(f.config, context)
        for key, value in [('total_head_fits', 120), ('readouts', ['case']), ('C_grid', [.01])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_matrix_contract(dict(f.config, **{key: value}), context)
        with self.assertRaises(ValueError):
            m.validate_matrix_contract(f.config, dict(context, readout='case'))

    def make_controller(self, root):
        f = self.fixtures['case']; config = config_from(f)
        context = {k: copy.deepcopy(v) for k, v in f.context.items() if k != 'readout'}
        context['encoder'] = {'repository': 'kaiko-ai/midnight', 'dimension': 3072}
        audit = copy.deepcopy(f.audit); audit.update(context)
        audit_dir = root/'audit'; audit_dir.mkdir(); write_json(audit_dir/'audit.json', audit)
        runs_root = root/'runs'
        paths, models, scores = m.expected_inventory(runs_root, config)
        for path in paths | models | scores:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
        args = SimpleNamespace(runs=runs_root, audit=audit_dir, output=root/'output')
        reference = m.read_json('docs/evidence/BREAKHIS_FIVE_CLASS_RESULTS.json')
        inputs = (config, context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
        return args, inputs, reference

    def test_complete_matrix_publishes_both_readouts_and_separate_primary_gates(self):
        from magset import midnight_five
        f = self.fixtures['case']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args, inputs, reference = self.make_controller(root)
            fail = [False]
            def verified(folder, readout, seed, trial, *unused):
                arms = copy.deepcopy(f.report['arms'])
                for arm in ('direct', 'dual'):
                    value = arms[arm]['validation']
                    value['mean_subset_macro_f1'] = .9
                    value['mean_subset_accuracy'] = .95
                    value['pooled']['confusion_matrix'] = (np.eye(5)*15).tolist()
                if fail[0] and readout == 'field' and seed == 813:
                    arms['dual']['validation']['mean_subset_accuracy'] = .1
                return {'readout': readout, 'seed': seed, 'trial': trial, 'head_fits': 12,
                        'run_sha256': '0'*64, 'arms': arms}
            for failing in (False, True):
                fail[0] = failing; args.output = root/str(failing)
                with patch.object(midnight_five, 'load_inputs', return_value=inputs), patch.object(m, 'reference_comparison', return_value=reference), patch.object(m, 'verify_run', side_effect=verified):
                    result = m.collect(args)
                self.assertEqual(result['head_fits'], 240)
                self.assertEqual(result['candidate_pipelines'], 160)
                self.assertEqual(result['verified_prediction_archives'], 320)
                self.assertEqual(result['verified_case_subset_records'], 129600)
                self.assertEqual(result['verified_runs_per_readout'], {'case': 10, 'field': 10})
                self.assertEqual(result['gate_passed_by_readout'], {'case': True, 'field': not failing})
                self.assertEqual(set(result['summary']), {'case', 'field'})
                self.assertEqual(result['collector_new_fits'], 0)
                self.assertNotIn('gate_passed', result)
                self.assertTrue((args.output/'comparison.json').exists())
                m.assert_public_safe(result)

    def test_missing_readout_or_extra_role_file_prevents_all_publication(self):
        from magset import midnight_five
        for extra in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args, inputs, reference = self.make_controller(root)
                if extra:
                    path = args.runs/'field/split812/trial1/dual/candidate0/excluded-predictions.jsonl'
                    path.write_text('{}')
                else:
                    (args.runs/'field/split813/trial5/run.json').unlink()
                with patch.object(midnight_five, 'load_inputs', return_value=inputs), patch.object(m, 'reference_comparison', return_value=reference), patch.object(m, 'verify_run') as verify:
                    with self.assertRaises(ValueError): m.collect(args)
                    verify.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_corrupt_field_objective_is_rejected_through_midnight_delegation(self):
        f = self.fixtures['field']; path = f.folder/'dual/candidate0/metrics.json'; report_path = f.folder/'run.json'
        original, report_original = path.read_bytes(), report_path.read_bytes()
        try:
            candidate = m.read_json(path); candidate['head_objectives']['binary']['penalty'] += .25
            write_json(path, candidate)
            report = m.read_json(report_path); report['arms']['dual']['candidates'][0] = candidate; write_json(report_path, report)
            common = {k: v for k, v in f.context.items() if k != 'readout'}
            with self.assertRaisesRegex(ValueError, 'head objective'):
                m.verify_run(f.folder, 'field', 812, 1, f.config, common, f.cases, f.bags, f.z,
                             f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))
        finally:
            path.write_bytes(original); report_path.write_bytes(report_original)


if __name__ == '__main__':
    unittest.main()
