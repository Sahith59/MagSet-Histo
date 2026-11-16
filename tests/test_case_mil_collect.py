"""Complete checkpoint replay and adversarial artifact fixtures; synthetic fields only."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from magset import case_mil_collect as c, case_mil_five as runner
from magset.five_class import write_json
from test_case_mil_five import fixture, smokes


@unittest.skipIf(torch is None, 'Optional Torch unavailable')
class CaseMILCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name); cls.f = fixture(cls.root)
        cls.smokes = cls.root/'smokes'; smokes(cls.smokes, cls.f)
        with patch.object(c, '_runtime'):
            cls.receipts = c.verify_smokes(cls.smokes, cls.f.config, cls.f.context, cls.f.cases, cls.f.bags, cls.f.y5, cls.f.roles, runner.file_hash(cls.f.audit_dir/'audit.json'))
        cls.folder = cls.root/'production'
        args = SimpleNamespace(output=cls.folder, audit=cls.f.audit_dir, seed=812, trial=1, pool='attention', arm='dual')
        runner.execute(args, cls.f.inputs, torch.device('cpu'), smoke_receipts=cls.receipts)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup(); torch.set_num_threads(cls.threads)

    def setUp(self): self.saved = {}

    def tearDown(self):
        for path, value in self.saved.items(): path.write_bytes(value)

    def write(self, path, value, tensor=False):
        self.saved.setdefault(path, path.read_bytes())
        if tensor: torch.save(value, path)
        else: write_json(path, value)

    def verify(self):
        f = self.f
        with patch.object(c, '_runtime'):
            return c.verify_run(self.folder, 'attention', 'dual', 812, 1, f.config, f.context,
                                f.cases, f.bags, f.y5, f.roles, runner.file_hash(f.audit_dir/'audit.json'), self.receipts)

    def change_candidate(self, candidate, index=0):
        self.write(self.folder/('epoch%d'%(index+1))/'metrics.json', candidate)
        report = c.read_json(self.folder/'run.json'); report['candidates'][index] = candidate
        self.write(self.folder/'run.json', report)

    def test_full_checkpoint_replay_does_not_call_model_or_optimizer(self):
        from magset.case_mil import CaseMIL
        with patch.object(CaseMIL, 'forward', side_effect=AssertionError('No collector neural forward')), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No collector fitting')):
            result = self.verify()
        self.assertEqual(len(result['candidates']), 4); c.assert_public_safe(result)
        self.assertNotIn('case_ids', str(result))

    def test_rehashed_normalizer_or_parameter_tampering_is_rejected(self):
        path = self.folder/'epoch1/state.pt'; saved = torch.load(path, weights_only=True)
        saved['mean'][0, 0] += .1; self.write(path, saved, True)
        candidate = c.read_json(path.with_name('metrics.json')); candidate['state_sha256'] = runner.file_hash(path)
        self.change_candidate(candidate)
        with self.assertRaisesRegex(ValueError, 'fitting-only normalizer'): self.verify()
        for p, value in self.saved.items(): p.write_bytes(value)
        saved = torch.load(path, weights_only=True); saved['model_state']['heads.binary.bias'] += 1
        self.write(path, saved, True); candidate = c.read_json(path.with_name('metrics.json')); candidate['state_sha256'] = runner.file_hash(path)
        self.change_candidate(candidate)
        with self.assertRaises(ValueError): self.verify()

    def test_optimizer_count_and_checkpoint_selection_tampering_fail(self):
        path = self.folder/'last.pt'; saved = torch.load(path, weights_only=True)
        saved['optimizer']['state'][0]['step'] += 1; self.write(path, saved, True)
        report = c.read_json(self.folder/'run.json'); report['last_checkpoint_sha256'] = runner.file_hash(path)
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'update count'): self.verify()
        for p, value in self.saved.items(): p.write_bytes(value)
        report = c.read_json(self.folder/'run.json'); report['selected_epoch'] = report['selected_epoch']%4+1
        self.write(self.folder/'run.json', report)
        with self.assertRaisesRegex(ValueError, 'earliest best'): self.verify()

    def test_production_schema_counts_and_forbidden_parameters(self):
        cfg = dict(self.f.config, dimension=3072, hidden=128, attention_hidden=32)
        self.assertEqual(sum(__import__('math').prod(s) for s in c.state_schema(cfg, 'mean', 'dual').values()), 394009)
        self.assertEqual(sum(__import__('math').prod(s) for s in c.state_schema(cfg, 'attention', 'direct').values()), 402297)
        saved = torch.load(self.folder/'epoch1/state.pt', weights_only=True)
        saved['model_state']['backbone.weight'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'whitelist'): c.check_model_state(saved['model_state'], self.f.config, 'attention', 'dual')

    def test_actual_F_C_P_reference_bindings_accept_and_reject_cache_change(self):
        cfg = c.read_json('configs/breakhis-case-mil-v1.json'); ref = c.read_json(cfg['reference_file'])
        roles = {(r['seed'], r['trial']): None for r in ref['runs']}
        self.assertEqual(set(c.references(cfg, ref, roles)), {'coverage', 'single_center', 'head_specific_center'})
        changed = copy.deepcopy(ref); changed['cache_provenance']['40']['npz_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'same coverage features'): c.references(cfg, changed, roles)

    def test_full_forty_matrix_counts_and_extra_role_file_fail_before_replay(self):
        f = self.f; sample = self.verify()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root/'runs'
            for pool in runner.POOLS:
                for arm in runner.ARMS:
                    for seed, trial in f.roles:
                        folder = runs/pool/arm/('split%d'%seed)/('trial%d'%trial); folder.mkdir(parents=True)
                        (folder/'run.json').write_text('{}'); (folder/'last.pt').write_text('{}')
                        for epoch in f.config['candidate_epochs']:
                            d = folder/('epoch%d'%epoch); d.mkdir()
                            for name in ('state.pt', 'metrics.json', 'train-predictions.jsonl', 'validation-predictions.jsonl'): (d/name).write_text('{}')
            values = {str(s): {'dual': {'validation': c.aggregate_summaries([sample['validation']])}} for s in (812, 813)}
            refs = {'coverage': {'summary': values}, 'single_center': {'summary': values}, 'head_specific_center': {'summary': {'head_specific': values}}}
            def verified(folder, pool, arm, seed, trial, *unused):
                return dict(copy.deepcopy(sample), pool=pool, arm=arm, seed=seed, trial=trial, initial_shared_sha256='a'*64)
            args = SimpleNamespace(output=root/'accepted', runs=runs, audit=f.audit_dir, smokes=self.smokes)
            with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(c, '_runtime'), patch.object(c, 'references', return_value=refs), patch.object(c, 'verify_run', side_effect=verified): result = c.collect(args)
            self.assertEqual(result['verified_runs'], 40); self.assertEqual(result['candidate_pipelines'], 160)
            self.assertEqual(result['verified_prediction_archives'], 320); self.assertEqual(result['verified_case_subset_records'], 129600)
            self.assertEqual(result['collector_new_fits'], 0); c.assert_public_safe(result)
            extra = runs/'mean/direct/split812/trial1/epoch1/excluded-predictions.jsonl'; extra.write_text('{}'); args.output = root/'bad'
            with patch.object(runner, 'load_inputs', return_value=f.inputs), patch.object(c, '_runtime'), patch.object(c, 'references', return_value=refs), patch.object(c, 'verify_run') as verify:
                with self.assertRaisesRegex(ValueError, 'forty-trajectory artifact matrix'): c.collect(args)
                verify.assert_not_called()


if __name__ == '__main__': unittest.main()
