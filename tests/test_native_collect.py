"""Complete tiny native support matrix and unchanged-method reference guards."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magset import native_collect as m, native_five as runner
from magset.data import CLASS_NAMES, file_hash
from magset.five_class import write_json
from magset.five_metrics import FINAL5
from test_kernel_five import fixture
from test_native_five import bound_config


class NativeCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)
        f = cls.f; f.config.update(bound_config())
        f.context['implementation_hashes'].update({n: file_hash(Path('magset')/n) for n in m.METHOD_FILES})
        f.context['native_config_sha256'] = '9'*64
        cls.native = [[x*2+.3 for x in case] for case in f.bags]
        cls.inputs = (f.config, f.context, {}, f.cases, cls.native, f.z, f.y5, f.roles)
        cls.audit = cls.root/'native-audit'; cls.runs_root = cls.root/'native-runs'
        args = SimpleNamespace(output=cls.audit)
        with patch.object(runner, 'load_inputs', return_value=cls.inputs): runner.audit_input(args)
        cls.reports = []
        for seed, trial in f.roles:
            folder = cls.runs_root/('split%d'%seed)/('trial%d'%trial)
            args = SimpleNamespace(output=folder, audit=cls.audit, seed=seed, trial=trial)
            with patch.object(runner, 'load_inputs', return_value=cls.inputs): runner.run(args)
            cls.reports.append(m.read_json(folder/'run.json'))
        summary = {str(seed): {a: {role: m.aggregate_summaries([r['arms'][a][role] for r in cls.reports if r['seed'] == seed])
                                  for role in ('fitting_leave_case_out', 'validation')} for a in runner.kernel.ARMS} for seed in (812, 813)}
        for secondary, name in [(False, 'primary'), (True, 'secondary')]:
            doc = dict(f.context, class_names=list(FINAL5), original_class_names=list(CLASS_NAMES), verified_runs=10,
                       candidate_pipelines=80, outer_predictions=0, runs=cls.reports, summary=summary,
                       encoder={'features': 'single-center' if secondary else 'three-crop'})
            path = cls.root/(name+'.json'); write_json(path, doc)
            f.config['secondary_reference_file' if secondary else 'reference_file'] = str(path)
            f.config['secondary_reference_comparison_sha256' if secondary else 'reference_comparison_sha256'] = file_hash(path)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_both_references_allow_encoder_change_but_protect_all_six_method_files(self):
        f = self.f
        for secondary in (False, True):
            ref = m.reference_comparison(f.config, f.context, f.roles, secondary)
            self.assertNotEqual(ref['encoder'], f.context['encoder'])
            for name in m.METHOD_FILES:
                bad = copy.deepcopy(f.context); bad['implementation_hashes'][name] = '0'*64
                with self.subTest(secondary=secondary, file=name), self.assertRaisesRegex(ValueError, 'unchanged method/split'):
                    m.reference_comparison(f.config, bad, f.roles, secondary)
            bad = dict(f.context, protocol_sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'protocol'): m.reference_comparison(f.config, bad, f.roles, secondary)

    def test_actual_full_synthetic_collect_replays_all_native_candidates_without_fitting(self):
        from sklearn.linear_model import LogisticRegression
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(runs=self.runs_root, audit=self.audit, output=Path(tmp)/'comparison')
            with patch.object(runner, 'load_inputs', return_value=self.inputs), patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No optimized fitting')):
                result = m.collect(args)
            self.assertEqual(result['verified_runs'], 10)
            self.assertEqual(result['verified_prediction_archives'], 160)
            self.assertEqual(result['verified_case_subset_records'], 64800)
            self.assertEqual(result['support_classifier_instances'], 30)
            self.assertEqual(result['optimized_head_fits'], 0)
            self.assertEqual(result['new_encoder_forwards'], 0)
            self.assertEqual(result['native_feature_forwards'], 7909)
            self.assertEqual(result['feature_token_shape'], [1601, 1536])
            self.assertIn('not comparable', result['compute_scope'])
            self.assertIn('comparison_against_threecrop_kernel_dual', result)
            self.assertIn('comparison_against_single_center_kernel_dual', result)
            self.assertFalse(result['gate_passed'])
            self.assertFalse(result['secondary_gate_passed'])
            m.assert_public_safe(result)
            self.assertTrue((args.output/'comparison.json').exists())

    def test_missing_or_corrupted_native_matrix_does_not_publish_partial_scores(self):
        f = self.f; missing = self.runs_root/'split813/trial5/run.json'; original = missing.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(runs=self.runs_root, audit=self.audit, output=Path(tmp)/'partial')
            try:
                missing.unlink()
                with patch.object(runner, 'load_inputs', return_value=self.inputs), patch.object(m.replay, 'verify_run') as verify:
                    with self.assertRaisesRegex(ValueError, 'artifact matrix'): m.collect(args)
                    verify.assert_not_called()
                self.assertFalse(args.output.exists())
            finally: missing.write_bytes(original)
            path = self.runs_root/'split812/trial1/direct/candidate0/model.json'; data = path.read_bytes()
            try:
                model = m.read_json(path); model['support_summary']['centers_sha256'] = '0'*64; write_json(path, model)
                with patch.object(runner, 'load_inputs', return_value=self.inputs):
                    with self.assertRaisesRegex(ValueError, 'hash differs'): m.collect(args)
                self.assertFalse(args.output.exists())
            finally: path.write_bytes(data)


if __name__ == '__main__': unittest.main()
