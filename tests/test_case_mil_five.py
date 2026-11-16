"""Small cached-feature trajectories only; no image reader or real cohort fitting."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from magset import case_mil_five as m
from magset.five_class import write_json
from test_five_collect import fixture as base_fixture


def fixture(root):
    f = base_fixture(root); own = m.read_json('configs/breakhis-case-mil-v1.json')
    f.config.update(own); f.config.update(dimension=3, hidden=4, attention_hidden=2, epochs=4, candidate_epochs=[1, 2, 3, 4])
    f.context.update(case_mil_config_sha256='1'*64, coverage_config_sha256='2'*64,
                     case_mil_base_config_sha256='2'*64, feature_audit_sha256='3'*64,
                     center_cache_provenance=copy.deepcopy(f.context['cache_provenance']))
    f.audit.update(f.context); write_json(f.audit_dir/'audit.json', f.audit)
    f.inputs = (f.config, f.context, {}, f.cases, f.bags, f.z, f.y5, f.roles)
    return f


def smokes(root, f):
    for pool in m.POOLS:
        for arm in m.ARMS:
            args = SimpleNamespace(output=root/pool/arm, audit=f.audit_dir, seed=812, trial=1, pool=pool, arm=arm)
            m.execute(args, f.inputs, torch.device('cpu'), smoke=True)


@unittest.skipIf(torch is None, 'Optional Torch unavailable')
class CaseMILRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup(); torch.set_num_threads(cls.threads)

    def test_loader_binds_new_source_and_preserves_F_cache(self):
        own = m.read_json('configs/breakhis-case-mil-v1.json'); own['base_input_config_sha256'] = 'b'*64
        with patch.object(m, 'file_hash', return_value='b'*64), patch.object(m, 'read_json', return_value=own), patch.object(m.source, 'load_inputs', return_value=self.f.inputs) as load:
            config, context, _, cases, bags, _, _, roles = m.load_inputs(SimpleNamespace(config=Path('registered.json')))
        self.assertEqual(load.call_args.args[0].config.name, 'breakhis-coverage-kernel-v1.json')
        self.assertIs(bags, self.f.bags); self.assertIs(roles, self.f.roles)
        self.assertIn('case_mil_collect.py', context['implementation_hashes'])
        self.assertEqual(config['production_runs'], 40)

    def test_normalizer_audit_and_LR_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp)/'audit')
            with patch.object(m, 'load_inputs', return_value=self.f.inputs): m.audit_input(args)
            m.verify_input_audit(m.read_json(args.output/'audit.json'), self.f.context, self.f.config, self.f.bags, self.f.y5, self.f.roles)
        self.assertEqual(m.learning_rate(self.f.config, 1), .0003)
        self.assertAlmostEqual(m.learning_rate(self.f.config, 4), .00003, places=17)

    def test_independent_numpy_matches_CPU64_all_subsets_both_pools_and_arms(self):
        from magset.case_mil_collect import numpy_outputs
        from magset.bags import fit_normalization
        f = self.f; train, val, _ = f.roles[812, 1]; mean, std = fit_normalization(f.bags, train)
        for pool in m.POOLS:
            for arm in m.ARMS:
                torch.manual_seed(77); model = m.build_model(f.config, pool, arm)
                saved = m.payload(model, {}, 0, mean, std)
                actual = numpy_outputs(saved, f.config, f.bags, val, pool, arm)
                wanted = m.infer_checkpoint(saved, f.config, f.bags, val, pool, arm)
                for key in wanted: np.testing.assert_allclose(actual[key], wanted[key], atol=5e-15, rtol=2e-14)

    def test_all_four_smokes_are_fitting_only_and_pool_initialization_is_paired(self):
        from magset import case_mil_collect as c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); smokes(root, self.f)
            with patch.object(c, '_runtime'):
                receipts = c.verify_smokes(root, self.f.config, self.f.context, self.f.cases, self.f.bags, self.f.y5, self.f.roles, m.file_hash(self.f.audit_dir/'audit.json'))
            self.assertEqual(len(receipts), 4); self.assertFalse(list(root.rglob('*predictions*.jsonl')))
            for path in root.rglob('technical.json'):
                report = m.read_json(path)
                self.assertEqual(len(report['fitting_indices']), 5)
                self.assertEqual(report['updates'], 6)
                self.assertTrue(report['reload_parity']['passed'])

    def test_tiny_production_saves_every_candidate_and_preserves_training_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args = SimpleNamespace(output=root/'run', audit=self.f.audit_dir, seed=812, trial=1, pool='attention', arm='dual')
            m.execute(args, self.f.inputs, torch.device('cpu'), smoke_receipts={})
            report = m.read_json(args.output/'run.json')
            self.assertEqual(len(report['history']), 4); self.assertEqual(len(report['candidates']), 4)
            self.assertEqual(len(list(args.output.rglob('*predictions.jsonl'))), 8)
            self.assertTrue(all(row['optimizer_updates'] == 12 for row in report['history']))
            wanted = max(range(4), key=lambda i: report['candidates'][i]['validation']['mean_subset_macro_f1'])
            self.assertEqual(report['selected_epoch'], wanted+1)
            again = SimpleNamespace(**vars(args)); again.output = root/'again'
            m.execute(again, self.f.inputs, torch.device('cpu'), smoke_receipts={})
            a = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
            b = torch.load(again.output/'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(m.state_digest(a['model_state']), m.state_digest(b['model_state']))
            self.assertTrue(torch.equal(a['rng'], b['rng']))

    def test_stable_hard_tie_and_extreme_router_use_case_logits(self):
        cases = [{'case_id': 'tiny', 'label': 1}]
        for value, expected in [(0., 1), (-10000., 0), (10000., 1)]:
            q = np.zeros((15, 1, 4)); a = np.full((15, 1), value)
            joint = np.concatenate((-np.logaddexp(0, a)[..., None], -np.logaddexp(0, -a)[..., None]+q-np.log(4)), axis=-1)
            rows = m.records_from_outputs(dict(joint_log_probabilities=joint, binary_logits=a, malignant_logits=q), [0], cases, 'validation', 'dual')
            self.assertEqual(rows[0]['prediction'], expected); self.assertTrue(np.isfinite(rows[0]['log_probabilities5']).all())
        with self.assertRaises(ValueError): m.records_from_outputs({}, [], cases, 'excluded', 'dual')


if __name__ == '__main__': unittest.main()
