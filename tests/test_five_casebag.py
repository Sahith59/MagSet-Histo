"""Case-bag contracts on synthetic images and a tiny CNN, with no downloads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn
    from magset import five_casebag as bags
    from test_adaptation import rows_and_cases
except ImportError:
    torch = None


if torch is not None:
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_stem = nn.Conv2d(3, 4, 1)
            self.bn1 = nn.BatchNorm2d(4)
            self.blocks = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.SiLU()) for _ in range(6)])
            self.conv_head = nn.Conv2d(4, 4, 1)
            self.bn2 = nn.BatchNorm2d(4)
            self.num_features = 4

        def forward(self, image):
            features = self.blocks(self.bn1(self.conv_stem(image)))
            return self.bn2(self.conv_head(features)).mean((2, 3))


def config():
    return {
        'training_seed': 812,
        'model': {'num_features': 4},
        'preprocessing': {'resize': [4, 4]},
        'sampling': {'fields_per_view': 2},
        'optimization': {'epochs': 3, 'warmup_epochs': 1, 'case_batch_size': 4,
                         'inference_image_batch_size': 7, 'backbone_lr': .001, 'head_lr': .01,
                         'cosine_min_factor': .1, 'weight_decay': .01, 'gradient_clip_norm': 1.,
                         'adamw_betas': [.9, .999], 'adamw_eps': 1e-8},
        'budget': {'technical_smoke_epochs': 2, 'technical_smoke_batches_per_epoch': 2},
        'selection': {'epochs': [1, 2, 3]},
    }


@unittest.skipIf(torch is None, 'Torch is required for synthetic case-bag checks')
class FiveCaseBagTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self, arm='dual'):
        torch.manual_seed(9)
        return bags.CaseBagModel(TinyCNN(), 4, arm)

    def test_epoch_and_field_draws_are_deterministic_and_arm_independent(self):
        _, cases = rows_and_cases()
        case = copy.deepcopy(cases[0])
        case['indices'][40] = ['third.png', 'first.png', 'second.png']
        first = bags.sampled_paths(case, 812, 3)
        torch.manual_seed(999)
        np.random.seed(444)
        self.assertEqual(first, bags.sampled_paths(case, 812, 3))
        self.assertEqual(len(set(first[0])), 2)
        self.assertEqual(first[1][0], first[1][1])
        order = bags.case_order(46, 812, 3)
        self.assertEqual(sorted(order), list(range(46)))
        self.assertEqual(order, bags.case_order(46, 812, 3))
        self.assertNotEqual(order, bags.case_order(46, 812, 4))

    def test_pixel_bags_and_allowed_role_check_precede_image_access(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, cases = rows_and_cases(directory)
            dataset = bags.CaseBags(rows, directory, cases, config(), 1)
            _, label, pixels, paths = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(pixels.shape, (4, 2, 3, 4, 4))
            torch.manual_seed(400)
            np.testing.assert_array_equal(pixels, dataset[0][2])
            self.assertEqual(paths, dataset[0][3])
            bad = copy.deepcopy(rows)
            bad[0]['case_id_raw'] = 'excluded'
            with patch.object(bags, 'load_checked_image', side_effect=AssertionError('Forbidden open')):
                with self.assertRaisesRegex(ValueError, 'allowed-role'):
                    bags.CaseBags(bad, directory, cases, config(), 1)

    def test_worker_count_preserves_case_order_paths_and_augmented_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, cases = rows_and_cases(directory)
            first = list(bags.stream_cases(rows, directory, cases, config(), 2, workers=0))
            other = list(bags.stream_cases(rows, directory, cases, config(), 2, workers=1))
            for left, right in zip(first, other):
                for index in range(3):
                    torch.testing.assert_close(left[index], right[index], rtol=0, atol=0)
                self.assertEqual(left[3], right[3])

    def test_all_masks_share_one_encoder_execution_and_field_mean_is_invariant(self):
        model = self.model('direct')
        model.eval()
        fields = torch.randn(2, 4, 2, 3, 4, 4)
        calls = []
        hook = model.backbone.register_forward_hook(lambda module, inputs, outputs: calls.append(len(outputs)))
        first = model(fields)['direct']
        hook.remove()
        second = model(fields.flip(2))['direct']
        self.assertEqual(calls, [16])
        self.assertEqual(first.shape, (2, 15, 5))
        torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
        views = torch.randn(2, 4, 4)
        expected = bags.fuse_views(views, (0, 2))
        views[:, [1, 3]] = float('nan')
        torch.testing.assert_close(expected, bags.fuse_views(views, (2, 0)), rtol=0, atol=0)
        for subset in ((), (0, 0), (4,)):
            with self.assertRaises(ValueError):
                bags.fuse_views(views, subset)

    def test_global_weight_mass_and_fixed_short_batch_denominator(self):
        labels = np.repeat(np.arange(5), [11, 25, 2, 5, 3])
        weights = bags.case_weights(labels)
        for name, value in weights.items():
            target = (labels > 0).astype(int) if name == 'binary' else labels
            self.assertAlmostEqual(float(np.sum(value[target])), 46)
        y = torch.tensor([0, 1, 2, 4])
        logits = torch.randn(4, 15, 5, requires_grad=True)
        whole, _ = bags.case_loss({'direct': logits}, y, weights, 4)
        left, _ = bags.case_loss({'direct': logits[:2]}, y[:2], weights, 4)
        right, _ = bags.case_loss({'direct': logits[2:]}, y[2:], weights, 4)
        torch.testing.assert_close(whole, left + right)
        zeros = torch.zeros(2, 15, 5)
        loss, _ = bags.case_loss({'direct': zeros}, torch.tensor([0, 4]), weights, 4)
        self.assertAlmostEqual(float(loss), np.log(5) * (weights['direct'][0] + weights['direct'][4]) / 4, places=6)

    def test_all_benign_batch_preserves_expert_adamw_state_and_parameters(self):
        model = self.model()
        bags.set_phase(model, False)
        weights = bags.case_weights(np.repeat(np.arange(5), [11, 25, 2, 5, 3]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
        features = torch.randn(2, 4, 4)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = bags.case_loss(model.head_logits(features), torch.tensor([1, 2]), weights)
        loss.backward()
        optimizer.step()
        before = copy.deepcopy(model.heads['malignant'].state_dict())
        state_before = copy.deepcopy([optimizer.state[parameter] for parameter in model.heads['malignant'].parameters()])
        optimizer.zero_grad(set_to_none=True)
        loss, components = bags.case_loss(model.head_logits(features), torch.tensor([0, 0]), weights)
        self.assertEqual(float(components['malignant']), 0)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.heads['malignant'].parameters()))
        optimizer.step()
        for name, value in model.heads['malignant'].state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for parameter, previous in zip(model.heads['malignant'].parameters(), state_before):
            for name, value in optimizer.state[parameter].items():
                torch.testing.assert_close(value, previous[name], rtol=0, atol=0)

    def test_phase_whitelist_bn_buffers_and_late_bn_affine(self):
        model = self.model('direct')
        initial = bags.state_hashes(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        weights = bags.case_weights(np.arange(5))
        for adapting in (False, True):
            allowed = bags.set_phase(model, adapting)
            self.assertFalse(model.backbone.bn1.training)
            self.assertFalse(model.backbone.bn2.training)
            self.assertFalse(model.backbone.bn1.weight.requires_grad)
            self.assertEqual(model.backbone.bn2.weight.requires_grad, adapting)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = bags.case_loss(model(torch.randn(4, 4, 2, 3, 4, 4)), torch.tensor([0, 1, 2, 4]), weights)
            loss.backward()
            self.assertFalse(any(parameter.grad is not None for name, parameter in model.named_parameters()
                                 if not name.startswith('heads.') and name not in allowed))
            optimizer.step()
            hashes = bags.state_hashes(model)
            self.assertEqual(hashes['bn'], initial['bn'])
            self.assertEqual(hashes['frozen'], initial['frozen'])
            if adapting:
                self.assertNotEqual(hashes['adaptive'], initial['adaptive'])
                self.assertIsNotNone(model.backbone.bn2.weight.grad)
            else:
                self.assertEqual(hashes['backbone'], initial['backbone'])

    def test_learning_rate_endpoints_and_single_adaptation_smoke(self):
        cfg = config()
        cfg['optimization'].update(epochs=50, warmup_epochs=5, head_lr=.001, backbone_lr=.00001)
        self.assertEqual(bags.learning_rates(cfg, 1), (.00001, .001))
        self.assertEqual(bags.learning_rates(cfg, 6), (.00001, .001))
        self.assertAlmostEqual(bags.learning_rates(cfg, 50)[0], .000001)
        self.assertAlmostEqual(bags.learning_rates(cfg, 50)[1], .0001)
        self.assertEqual(bags.learning_rates(cfg, 2, smoke=True), (.00001, .001))

    def test_declared_numpy_head_path_hard_route_and_extreme_logits(self):
        descriptor = np.zeros((1, 8), dtype=np.float32)
        arrays = {'binary_coef': np.zeros((1, 8), dtype=np.float32),
                  'binary_intercept': np.asarray([np.log(.6 / .4)], dtype=np.float32),
                  'malignant_coef': np.zeros((4, 8), dtype=np.float32),
                  'malignant_intercept': np.log([.3, .25, .25, .2]).astype(np.float32)}
        result = bags.scores_from_heads(arrays, descriptor, 'dual')
        self.assertEqual(result['prediction'][0], 1)
        self.assertEqual(result['probabilities5'][0].argmax(), 0)
        arrays['binary_intercept'][0] = -2000
        arrays['malignant_intercept'] = np.asarray([1000, 0, -1000, -2000], dtype=np.float32)
        result = bags.scores_from_heads(arrays, descriptor, 'dual')
        self.assertTrue(np.isfinite(result['log_probabilities5']).all())
        self.assertEqual(result['prediction'][0], 0)
        self.assertEqual(result['probabilities5'].dtype, np.float64)
        for near_zero in (-1e-20, 0., 1e-20):
            arrays['binary_intercept'][0] = near_zero
            result = bags.scores_from_heads(arrays, descriptor, 'dual')
            self.assertEqual(result['binary_score'][0], .5)
            self.assertEqual(result['prediction'][0], 1)

    def test_full_field_scores_chunking_replay_and_outer_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, cases = rows_and_cases(directory)
            model = self.model()
            cfg = config()
            archive, records, metric = bags.score_cases(model, rows, directory, cases, cfg, torch.device('cpu'), 'validation')
            self.assertEqual(archive['view_features'].dtype, np.float64)
            self.assertEqual(archive['descriptors'].dtype, np.float32)
            self.assertEqual(archive['descriptors'].shape, (8, 15, 8))
            self.assertEqual(archive['field_counts'].tolist(), [[1] * 4] * 8)
            self.assertEqual(len(records), 120)
            self.assertEqual(metric['distinct_class_support'], [4, 1, 1, 1, 1])
            cfg['optimization']['inference_image_batch_size'] = 11
            other, other_records, _ = bags.score_cases(model, list(reversed(rows)), directory, cases, cfg, torch.device('cpu'), 'validation')
            np.testing.assert_allclose(archive['view_features'], other['view_features'], rtol=0, atol=1e-6)
            np.testing.assert_allclose([row['probabilities5'] for row in records],
                                       [row['probabilities5'] for row in other_records], rtol=0, atol=1e-6)
            with patch.object(bags, 'load_checked_image', side_effect=AssertionError('Forbidden open')):
                with self.assertRaises(ValueError):
                    bags.score_cases(model, rows, directory, cases, cfg, torch.device('cpu'), 'excluded')

    def test_smoke_is_fitting_only_and_reload_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, cases = rows_and_cases(root)
            output = root / 'smoke'
            output.mkdir()
            model = self.model()
            context = {'source': 'synthetic', 'gpu_family': 'cpu', 'arm': 'dual'}
            with patch.object(bags, 'score_cases', side_effect=AssertionError('Smoke cannot validate')):
                proof = bags.train(model, rows, [], cases, [], root, config(), context,
                                   output, torch.device('cpu'), smoke=True)
            self.assertEqual([row['steps'] for row in proof['history']], [2, 2])
            self.assertEqual([row['backbone_adapting'] for row in proof['history']], [False, True])
            self.assertEqual(proof['candidates'], [])
            self.assertEqual(proof['smoke_reloads'], [1])
            self.assertEqual(proof['completed_epoch_training_field_forwards'], 128)
            saved = bags.bound_checkpoint(output / 'last.pt', context)
            restored = self.model()
            restored.load_state_dict(saved['model'], strict=True)
            self.assertEqual(bags.probe_reload(model, restored, rows, root, config(), torch.device('cpu')),
                             {'feature_max_abs': 0., 'probability_max_abs': 0.})
            with self.assertRaises(ValueError):
                bags.bound_checkpoint(output / 'last.pt', {**context, 'gpu_family': 'different'})
            interrupted = root / 'interrupted-smoke'
            interrupted.mkdir()
            original_save = bags.atomic_save

            def stop_after_epoch_one_commit(value, path):
                original_save(value, path)
                if value.get('epoch') == 1 and 'optimizer' in value and not value.get('smoke_reloads'):
                    raise RuntimeError('Stopped after the authoritative commit')

            with patch.object(bags, 'atomic_save', side_effect=stop_after_epoch_one_commit):
                with self.assertRaisesRegex(RuntimeError, 'authoritative commit'):
                    bags.train(self.model(), rows, [], cases, [], root, config(), context,
                               interrupted, torch.device('cpu'), smoke=True)
            recovered = bags.train(self.model(), rows, [], cases, [], root, config(), context,
                                   interrupted, torch.device('cpu'), smoke=True, resume=True)
            self.assertEqual(recovered['smoke_reloads'], [1])
            self.assertEqual(recovered['final'], proof['final'])

    def test_authoritative_epoch_resume_matches_uninterrupted_and_recovers_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, cases = rows_and_cases(root)
            complete = root / 'complete'
            interrupted = root / 'interrupted'
            complete.mkdir()
            interrupted.mkdir()
            cfg = config()
            context = {'source': 'synthetic', 'roles': 'same', 'arm': 'direct'}
            first = self.model('direct')
            expected = bags.train(first, rows, rows, cases, cases, root, cfg, context,
                                  complete, torch.device('cpu'))
            second = self.model('direct')
            original_candidate = bags._candidate

            def interrupt(output, epoch, *args):
                if epoch == 2:
                    raise RuntimeError('Synthetic interruption')
                return original_candidate(output, epoch, *args)

            with patch.object(bags, '_candidate', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                    bags.train(second, rows, rows, cases, cases, root, cfg, context,
                               interrupted, torch.device('cpu'))
            # A stale/corrupt selected convenience file is rebuilt from last.pt.
            (interrupted / 'selected.pt').write_bytes(b'stale')
            second = self.model('direct')
            actual = bags.train(second, rows, rows, cases, cases, root, cfg, context,
                                interrupted, torch.device('cpu'), resume=True)
            self.assertEqual(expected['final'], actual['final'])
            self.assertEqual(expected['best']['epoch'], actual['best']['epoch'])
            for left, right in zip(expected['history'], actual['history']):
                self.assertEqual(left['field_draw_sha256'], right['field_draw_sha256'])
                self.assertEqual(left['case_order'], right['case_order'])
                self.assertEqual(left['loss'], right['loss'])
            saved = bags.bound_checkpoint(interrupted / 'selected.pt', context)
            self.assertEqual(saved['epoch'], actual['best']['epoch'])
            json.dumps(actual, allow_nan=False)
            candidate = interrupted / 'candidate-epoch1-features.npz'
            candidate.write_bytes(candidate.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact hashes'):
                bags.train(self.model('direct'), rows, rows, cases, cases, root, cfg, context,
                           interrupted, torch.device('cpu'), resume=True)

    def test_resume_rejects_optimizer_and_sampling_corruption_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, cases = rows_and_cases(root)
            output = root / 'smoke'
            output.mkdir()
            cfg = config()
            context = {'source': 'synthetic', 'arm': 'dual'}
            bags.train(self.model(), rows, [], cases, [], root, cfg, context,
                       output, torch.device('cpu'), smoke=True)
            original = bags.bound_checkpoint(output / 'last.pt', context)
            changes = ('moment_nan', 'moment_dtype', 'step_count', 'case_order', 'learning_rate')
            for change in changes:
                modified = copy.deepcopy(original)
                states = modified['optimizer']['state']
                state = next(iter(states.values()))
                if change == 'moment_nan':
                    state['exp_avg'].flatten()[0] = float('nan')
                elif change == 'moment_dtype':
                    state['exp_avg'] = state['exp_avg'].double()
                elif change == 'step_count':
                    state['step'] += 1
                elif change == 'case_order':
                    modified['history'][0]['case_order'] = list(reversed(modified['history'][0]['case_order']))
                else:
                    modified['optimizer']['param_groups'][0]['lr'] *= 2
                bags.atomic_save(modified, output / 'last.pt')
                with self.subTest(change=change), patch.object(torch.optim.AdamW, 'load_state_dict',
                                                              side_effect=AssertionError('Invalid state must fail first')):
                    with self.assertRaises(ValueError):
                        bags.train(self.model(), rows, [], cases, [], root, cfg, context,
                                   output, torch.device('cpu'), smoke=True, resume=True)

    def test_draft_production_gate_config_drift_and_nan_replay(self):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'breakhis-casebag-draft.json'
        declared = json.loads(path.read_text())
        bags.validate_config(declared)
        draft = copy.deepcopy(declared)
        draft['production_ready'] = False
        with self.assertRaisesRegex(ValueError, 'production-ready'):
            bags.execution_gate('run', draft, torch.device('cuda'))
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            bags.execution_gate('smoke', declared, torch.device('cpu'))
        changed = copy.deepcopy(declared)
        changed['optimization']['epochs'] = 51
        with self.assertRaises(ValueError):
            bags.validate_config(changed)
        with self.assertRaises(ValueError):
            bags.maximum_difference([float('nan')], [0.])

    def test_two_smoke_receipts_bind_actual_checkpoints_and_source_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, cases = rows_and_cases(root)
            shared = {'code_sha': 'synthetic-source', 'config_sha256': 'synthetic-config',
                      'weights_sha256': 'synthetic-weights', 'manifest_sha256': 'synthetic-manifest',
                      'protocol_sha256': 'synthetic-protocol', 'source_archive_sha256': 'synthetic-archive',
                      'membership_fingerprint_sha256': 'synthetic-memberships', 'environment': {'gpu_family': 'A40'},
                      'implementation_sha256': {'five_casebag.py': 'synthetic-implementation'},
                      'seed': 812, 'trial': 1}
            entries = []
            for arm in ('direct', 'dual'):
                output = root / arm
                output.mkdir()
                context = {**shared, 'arm': arm, 'smoke': True}
                proof = bags.train(self.model(arm), rows, [], cases, [], root, config(), context,
                                   output, torch.device('cpu'), smoke=True)
                proof['reload'] = {'feature_max_abs': 0., 'probability_max_abs': 0.}
                path = output / 'metrics.json'
                bags.write_json(path, {'status': 'fitting-only case-bag smoke', 'context': context,
                                       'proof': proof, 'validation_predictions': 0, 'outer_predictions': 0})
                entries.append({'arm': arm, 'path': str(path), 'sha256': bags.file_hash(path)})
            receipt = root / 'accepted.json'
            bags.write_json(receipt, {'status': 'accepted case-bag technical smokes', 'reports': entries})
            self.assertEqual(bags.accepted_smokes(receipt, {**shared, 'arm': 'dual', 'smoke': False}), bags.file_hash(receipt))
            bags.write_json(receipt, {'status': 'accepted case-bag technical smokes', 'reports': entries[:1]})
            with self.assertRaises(ValueError):
                bags.accepted_smokes(receipt, shared)
            bags.write_json(receipt, {'status': 'accepted case-bag technical smokes', 'reports': entries})
            with self.assertRaisesRegex(ValueError, 'source, runtime'):
                bags.accepted_smokes(receipt, {**shared, 'config_sha256': 'changed'})


if __name__ == '__main__':
    unittest.main()
