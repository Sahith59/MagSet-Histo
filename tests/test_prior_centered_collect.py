"""Independent replay, metadata tampering and complete paired-matrix fixtures."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression

from magset import prior_centered_collect as m, prior_centered_five as runner
from magset.data import file_hash
from magset.five_class import write_json
from test_prior_centered_five import fixture


class PriorCenteredCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)
        cls.folder = cls.root/'paired'
        args = SimpleNamespace(output=cls.folder, audit=cls.f.audit_dir, seed=812, trial=1)
        with patch.object(runner, 'load_inputs', return_value=cls.f.inputs): runner.run(args)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self): self.saved = {}

    def tearDown(self):
        for path, data in self.saved.items(): path.write_bytes(data)

    def write(self, path, value):
        self.saved.setdefault(path, path.read_bytes()); write_json(path, value)

    def verify(self):
        f = self.f
        return m.verify_run(self.folder, 812, 1, f.config, f.context, f.cases, f.bags,
                            f.z, f.y5, f.roles, file_hash(f.audit_dir/'audit.json'))

    def change_candidate(self, candidate, mode='class_shared', arm='direct', index=0):
        self.write(self.folder/mode/arm/('candidate%d'%index)/'metrics.json', candidate)
        report = m.read_json(self.folder/'run.json'); report['modes'][mode]['arms'][arm]['candidates'][index] = candidate
        self.write(self.folder/'run.json', report)

    def test_full_replay_public_summary_and_no_optimization(self):
        with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('No collector fitting')):
            result = self.verify()
        self.assertEqual(result['logical_full_fit_head_definitions'], 6)
        self.assertEqual(result['structurally_unique_full_fit_head_definitions'], 5)
        self.assertEqual(result['optimized_head_fits'], 0)
        m.assert_public_safe(result)
        self.assertNotIn('support_case_ids', json.dumps(result))
        m.verify_direct_duplicates(result['modes'])

    def test_rehashed_changed_used_center_or_support_is_rejected(self):
        path = self.folder/'head_specific/dual/candidate0/model.json'
        model = m.read_json(path); model['used_centers_sha256']['binary'] = '0'*64
        self.write(path, model)
        candidate = m.read_json(path.with_name('metrics.json')); candidate['model_sha256'] = file_hash(path)
        self.change_candidate(candidate, 'head_specific', 'dual')
        with self.assertRaisesRegex(ValueError, 'support model'): self.verify()

    def test_missing_mode_and_changed_earliest_selection_fail(self):
        report = m.read_json(self.folder/'run.json'); del report['modes']['head_specific']
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'Incomplete mode matrix'): self.verify()
        for p, data in self.saved.items(): p.write_bytes(data)
        report = m.read_json(self.folder/'run.json')
        saved = report['modes']['head_specific']['arms']['dual']; saved['selected_index'] = (saved['selected_index']+1)%4
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'earliest selection/selected_index'): self.verify()

    def test_direct_duplicate_detector_requires_exact_records_and_metrics(self):
        modes = m.read_json(self.folder/'run.json')['modes']
        m.verify_direct_duplicates(modes)
        modes['head_specific']['arms']['direct']['candidates'][0]['validation_predictions_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'duplicated direct candidate'): m.verify_direct_duplicates(modes)
        modes = m.read_json(self.folder/'run.json')['modes']
        modes['head_specific']['arms']['direct']['validation']['mean_subset_macro_f1'] += 1e-14
        with self.assertRaisesRegex(ValueError, 'duplicated direct/validation'): m.verify_direct_duplicates(modes)

    def reference(self):
        f = self.f; report = m.read_json(self.folder/'run.json')
        summary = {a: {'validation': m.aggregate_summaries([report['modes']['class_shared']['arms'][a]['validation']])} for a in runner.ARMS}
        return dict(copy.deepcopy(f.context), class_names=report['class_names'], original_class_names=report['original_class_names'],
                    verified_runs=10, candidate_pipelines=80, outer_predictions=0,
                    runs=[dict(seed=s, trial=t, arms=copy.deepcopy(report['modes']['class_shared']['arms'])) for s, t in f.roles],
                    summary={str(s): copy.deepcopy(summary) for s in (812, 813)})

    def test_references_bind_same_features_primary_and_original_features_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); primary = self.reference(); primary['config_sha256'] = self.f.config['base_input_config_sha256']
            secondary = self.reference()
            p, q = root/'primary.json', root/'secondary.json'; write_json(p, primary); write_json(q, secondary)
            cfg = dict(self.f.config, reference_file=str(p), reference_comparison_sha256=file_hash(p),
                       secondary_reference_file=str(q), secondary_reference_comparison_sha256=file_hash(q))
            self.assertEqual(set(m.references(cfg, self.f.context, self.f.roles)), {'coverage', 'single_center'})
            primary['cache_provenance']['40']['npz_sha256'] = '0'*64
            write_json(p, primary); cfg['reference_comparison_sha256'] = file_hash(p)
            with self.assertRaisesRegex(ValueError, 'same coverage features/cache_provenance'):
                m.references(cfg, self.f.context, self.f.roles)

    def test_complete_matrix_counts_and_partial_matrix_rejection(self):
        f = self.f; report = m.read_json(self.folder/'run.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'
            for s, t in f.roles:
                folder = runs/('split%d'%s)/('trial%d'%t); folder.mkdir(parents=True); (folder/'run.json').write_text('{}')
                for mode in runner.MODES:
                    for arm in runner.ARMS:
                        for i in range(4):
                            directory = folder/mode/arm/('candidate%d'%i); directory.mkdir(parents=True)
                            for name in ('model.json', 'metrics.json', 'train-predictions.jsonl', 'validation-predictions.jsonl'):
                                (directory/name).write_text('{}')
            def verified(folder, s, t, *unused):
                return dict(seed=s, trial=t, modes=copy.deepcopy(report['modes']), optimized_head_fits=0,
                            logical_full_fit_head_definitions=6, structurally_unique_full_fit_head_definitions=5)
            args = SimpleNamespace(runs=runs, audit=f.audit_dir, output=root/'accepted')
            refs = {'coverage': self.reference(), 'single_center': self.reference()}
            with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(m, 'references', return_value=refs), patch.object(m, 'verify_run', side_effect=verified): result = m.collect(args)
            for key, value in runner.COUNTS.items(): self.assertEqual(result[key], value)
            self.assertEqual(result['verified_prediction_archives'], 320)
            self.assertEqual(result['verified_case_subset_records'], 129600)
            self.assertEqual(result['structurally_nonduplicated_candidate_records'], 97200)
            self.assertEqual(set(result['gate_passed_by_mode']), set(runner.MODES))
            m.assert_public_safe(result)
            (runs/'split813/trial5/head_specific/dual/candidate3/model.json').unlink(); args.output = root/'partial'
            with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(m, 'references', return_value=refs), patch.object(m, 'verify_run') as verify:
                with self.assertRaisesRegex(ValueError, 'artifact matrix'): m.collect(args)
                verify.assert_not_called()
            self.assertFalse(args.output.exists())


if __name__ == '__main__': unittest.main()
