"""CPU-only tiny trainer/sampling fixtures; no external weights or actual images."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from magset import lora_five as m
    from magset import lora_io as io
    from magset import midnight_lora as core
    from magset.bags import fit_normalization
    from magset.data import file_hash
    from magset.five_class import load_model
    from test_midnight_lora import ToyBackbone, arrays, toy_features
except ImportError:
    torch = None


def fixture():
    cases, rows, bags = [], [], []
    for i in range(10):
        indices = {}; views = []
        for view, mag in enumerate((40, 100, 200, 400)):
            paths = ['fixture/%d-%d-%d.png'%(i, mag, j) for j in range(2)]
            indices[mag] = paths
            for j, path in enumerate(paths): rows.append({'relative_path': path, 'payload': [i/10+.1*j, view/5+.2]})
            views.append(np.array([[i/10, view/5, i/10, view/5], [i/10+.1, view/5+.2, i/10+.1, view/5+.2]], np.float32))
        cases.append({'case_id': 'fixture-%d'%i, 'label': [0, 1, 3, 4, 5][i//2], 'indices': indices})
        bags.append(views)
    return rows, cases, bags, np.repeat(np.arange(5), 2)


@unittest.skipIf(torch is None, 'Torch is required for tiny LoRA trainer checks')
class LoRARunnerTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.config = json.loads(Path('configs/breakhis-midnight-lora-v1.json').read_text())
        self.rows, self.cases, self.bags, self.y5 = fixture()

    def tearDown(self): torch.set_num_threads(self.threads)

    def model(self, arm='dual'):
        torch.manual_seed(29)
        return core.AdapterClassifier(ToyBackbone(), arrays(arm), arm)

    def test_arm_independent_case_and_field_draws_ignore_global_rng(self):
        seed = io.run_seed(812, 1); before = io.case_order(list(range(46)), seed, 1)
        draws = io.sampled_paths(self.cases[0], seed, 1)
        torch.manual_seed(999); np.random.seed(999)
        self.assertEqual(io.case_order(list(range(46)), seed, 1), before)
        self.assertEqual(sorted(before), list(range(46)))
        self.assertEqual(io.sampled_paths(self.cases[0], seed, 1), draws)
        self.assertEqual(len(draws), 8)
        self.assertEqual([v for v, _, _ in draws], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertNotEqual(before, io.case_order(list(range(46)), seed, 2))

    def test_augmented_pixels_are_reproducible_and_restore_cpu_rng(self):
        from PIL import Image
        transform = io.build_augmentation(); image = Image.fromarray(np.arange(80*100*3, dtype=np.uint8).reshape(80, 100, 3))
        torch.manual_seed(4); before = torch.get_rng_state().clone()
        first = transform(image, 77)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.manual_seed(18); state = torch.get_rng_state().clone()
        torch.testing.assert_close(first, transform(image, 77), rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(first.shape, (3, 224, 224))

    def train(self, output, smoke=True, resume=False, fail_save=False):
        model = self.model(); args = SimpleNamespace(output=output, seed=812, trial=1, arm='dual', images_root=Path('unused'), resume=resume)
        config = copy.deepcopy(self.config)
        if not smoke: config.update(epochs=3, candidate_epochs=[])
        real_save = io.atomic_save
        def saving(value, path):
            real_save(value, path)
            if fail_save and value.get('epoch') == 1: raise InterruptedError('fixture interruption after committed epoch')
        with patch.object(core, 'classification_features', side_effect=toy_features), patch.object(m, 'load_checked_image', side_effect=lambda root, row: np.array(row['payload'], np.float32)), patch.object(io, 'build_augmentation', return_value=lambda image, seed: torch.as_tensor(image)), patch.object(io, 'atomic_save', side_effect=saving):
            result = m.train_epochs(model, self.rows, self.cases, np.arange(10), np.array([], int), self.y5, args, config, {}, torch.device('cpu'), None, {}, model.frozen_digest(), smoke=smoke)
        return model, result

    def test_three_epoch_smoke_exercises_heads_then_two_adapter_updates_without_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, (history, candidates, _) = self.train(Path(tmp))
            self.assertEqual([r['adapting'] for r in history], [False, False, True])
            self.assertEqual([r['optimizer_updates'] for r in history], [2, 2, 2])
            self.assertEqual([r['cases'] for r in history], [8, 8, 8])
            self.assertEqual(candidates, [])
            for row in history[:2]:
                self.assertEqual(len(row['gradient_steps']), 2)
                self.assertTrue(all(g['A'] == 0 and g['B'] == 0 and g['heads'] > 0 for g in row['gradient_steps']))
            self.assertEqual(history[2]['gradient_steps'][0]['A'], 0)
            self.assertGreater(history[2]['gradient_steps'][0]['B'], 0)
            self.assertGreater(history[2]['gradient_steps'][1]['A'], 0)
            self.assertGreater(history[2]['gradient_steps'][1]['B'], 0)
            self.assertEqual(history[0]['small_digests']['A'], history[1]['small_digests']['A'])
            self.assertNotEqual(history[1]['small_digests']['A'], history[2]['small_digests']['A'])
            self.assertNotEqual(history[1]['small_digests']['B'], history[2]['small_digests']['B'])
            self.assertFalse(list(Path(tmp).glob('epoch*')))
            checkpoint = torch.load(Path(tmp)/'last.pt', weights_only=True)
            self.assertEqual(checkpoint['updates'], 6)
            self.assertEqual(set(checkpoint['small_state']), set(model.small_state()))
            self.assertEqual(model.frozen_digest(), self.model().frozen_digest())

    def test_partial_last_batch_keeps_fixed_four_case_denominator(self):
        original = core.weighted_case_loss; denominators = []
        def checked(*args, **kwargs): denominators.append(kwargs['batch_denominator']); return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'weighted_case_loss', side_effect=checked):
            _, (history, _, _) = self.train(Path(tmp), smoke=False)
        self.assertEqual([r['optimizer_updates'] for r in history], [3, 3, 3])
        self.assertEqual(denominators, [4]*30)

    def test_interrupted_small_checkpoint_resumes_exact_training_state(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as partial:
            uninterrupted, _ = self.train(Path(full))
            with self.assertRaises(InterruptedError): self.train(Path(partial), fail_save=True)
            resumed, (history, _, _) = self.train(Path(partial), resume=True)
            self.assertEqual(len(history), 3)
            for name, value in uninterrupted.small_state().items(): torch.testing.assert_close(value, resumed.small_state()[name], rtol=0, atol=0)

    def test_feature_archives_export_canonical_heads_and_saved_probability_replay(self):
        model = self.model('direct')
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'classification_features', side_effect=toy_features), patch.object(m, 'load_checked_image', side_effect=lambda root, row: np.array(row['payload'], np.float32)):
            summary, hashes = m.save_scores(Path(tmp), model, self.rows, self.cases, np.arange(10), Path('unused'), self.config, torch.device('cpu'), torch.as_tensor, 'validation')
            loaded = load_model(Path(tmp)/'model.npz', 'direct', dimension=4)
            with np.load(Path(tmp)/'validation-features.npz', allow_pickle=False) as archive:
                records = io.records_from_features(archive['paths'].tolist(), archive['features'], loaded, self.cases, np.arange(10), 'validation', 'direct')
            self.assertEqual(summary['pooled']['records'], 150)
            self.assertEqual(len(records), 150)
            self.assertEqual(loaded['direct_coef'].dtype, np.float64)
            self.assertEqual(hashes['validation_predictions_sha256'], file_hash(Path(tmp)/'validation-predictions.jsonl'))

    def test_tiny_full_run_publishes_checkpoints_selected_features_and_fresh_reload_receipts(self):
        rows, cases, bags, y5 = fixture()
        # Eight synthetic validation cases with the declared pooled-five support.
        for offset, label in enumerate([0, 0, 0, 0, 1, 2, 3, 4]):
            original = cases[offset]
            index = len(cases); fields = {mag: ['validation/%d-%d-%d.png'%(index, mag, j) for j in range(2)] for mag in (40, 100, 200, 400)}
            cases.append({'case_id': 'validation-%d'%index, 'label': [0, 1, 3, 4, 5][label], 'indices': fields})
            for view, mag in enumerate((40, 100, 200, 400)):
                for j, path in enumerate(fields[mag]): rows.append({'relative_path': path, 'payload': [.2+offset/10, view/5+.1*j]})
            y5 = np.r_[y5, label]
        train, val = np.arange(10), np.arange(10, 18)
        initial = arrays('dual')
        def factory(args, config, initial_arrays, device):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(29)
                return core.AdapterClassifier(ToyBackbone(), initial_arrays, args.arm)
        config = copy.deepcopy(self.config); config.update(encoder_dimension=4, epochs=3, candidate_epochs=[2, 3])
        model = factory(SimpleNamespace(arm='dual'), config, initial, torch.device('cpu'))
        with patch.object(core, 'classification_features', side_effect=toy_features), torch.inference_mode():
            by_path = {r['relative_path']: r for r in rows}; bags = []
            for case in cases:
                views = []
                for mag in (40, 100, 200, 400):
                    pixels = torch.tensor([by_path[p]['payload'] for p in case['indices'][mag]], dtype=torch.float32)
                    views.append(model.encode(pixels).numpy())
                bags.append(views)
        initial['mean'], initial['std'] = fit_normalization(bags, train)
        config['production_trainable_parameters_per_run'] = model.trainable_manifest()['total_parameters']
        common = {'config_sha256': 'b'*64, 'lora_config_sha256': 'c'*64}
        inputs = (config, common, {}, cases, bags, None, y5, {(812, 1): (train, val, np.array([], int))})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit = root/'audit'; audit.mkdir(); (audit/'audit.json').write_text('{}')
            initial_path = m.initial_model_path(root/'initial', 812, 1, 'dual'); initial_path.parent.mkdir(parents=True); np.savez_compressed(initial_path, **initial)
            args = SimpleNamespace(output=root/'run', audit=audit, arm='dual', seed=812, trial=1, initial_runs=root/'initial', manifest=root/'manifest', images_root=root, resume=False)
            with patch.object(m, 'load_inputs', return_value=inputs), patch.object(m, 'verify_input_audit'), patch.object(m, 'runtime', return_value=torch.device('cpu')), patch.object(m, 'build_adapter', side_effect=factory), patch.object(m, 'read_manifest', return_value=rows), patch.object(m, 'load_checked_image', side_effect=lambda root, row: np.array(row['payload'], np.float32)), patch.object(core, 'classification_features', side_effect=toy_features), patch.object(io, 'build_augmentation', return_value=lambda image, seed: torch.as_tensor(image)), patch('magset.midnight_features.build_transform', return_value=torch.as_tensor), patch.object(torch.cuda, 'empty_cache'), patch.object(torch.cuda, 'max_memory_allocated', return_value=0), patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                m.execute(args)
            report = json.loads((args.output/'run.json').read_text())
            self.assertEqual([c['epoch'] for c in report['candidates']], [2, 3])
            self.assertTrue(report['initial_parity']['passed'])
            self.assertTrue(report['reload_parity']['passed'])
            self.assertEqual(report['frozen_digest_initial'], report['frozen_digest_final'])
            self.assertTrue((args.output/'selected/train-features.npz').exists())
            self.assertTrue((args.output/'selected/train-predictions.jsonl').exists())
            self.assertEqual(report['selected_fitting']['fitting']['pooled']['records'], 150)
            for candidate in report['candidates']:
                folder = args.output/('epoch%d'%candidate['epoch'])
                self.assertEqual(candidate['state_sha256'], file_hash(folder/'state.pt'))
                self.assertTrue((folder/'validation-features.npz').exists())

    def test_fixed_c1_binding_rejects_poisoned_fit_normalizer_even_after_rehash(self):
        mean, std = fit_normalization(self.bags, np.arange(10)); initial = arrays('direct'); initial.update(mean=mean, std=std)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = m.initial_model_path(root, 812, 1, 'direct'); path.parent.mkdir(parents=True)
            np.savez_compressed(path, **initial)
            dual = arrays('dual'); dual.update(mean=mean, std=std)
            other = m.initial_model_path(root, 812, 1, 'dual'); other.parent.mkdir(parents=True); np.savez_compressed(other, **dual)
            config = {'encoder_dimension': 4, 'initial_model_sha256': {'812/1/direct': file_hash(path), '812/1/dual': file_hash(other)}}
            roles = {(812, 1): (np.arange(10), [], [])}
            self.assertEqual(len(m.initial_bindings(config, self.cases, self.bags, roles, root)), 2)
            initial['mean'][0, 0] += 1; np.savez_compressed(path, **initial); config['initial_model_sha256']['812/1/direct'] = file_hash(path)
            with self.assertRaisesRegex(ValueError, 'normalizer'): m.initial_bindings(config, self.cases, self.bags, roles, root)


if __name__ == '__main__': unittest.main()
