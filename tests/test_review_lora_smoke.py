"""Negative synthetic smoke receipts; no model construction, data or jobs."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location('review_lora_smoke_fixture', Path(__file__).resolve().parents[1] / 'scripts/review_lora_smoke.py')
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)


@unittest.skipIf(torch is None, 'PyTorch is absent in the lightweight environment')
class SmokeReviewTests(unittest.TestCase):
    def fixture(self):
        config = {'optimizer': {'lr': 1e-4}, 'parity': {'atol': 1e-4, 'rtol': 1e-4}}
        cases = [{'case_id': 'fixture%d' % i, 'indices': {mag: ['fixture/%d/%d/a.png' % (i, mag), 'fixture/%d/%d/b.png' % (i, mag)]
                  for mag in review.MAGNIFICATIONS}} for i in range(8)]
        initial = {name: str(i) * 64 for i, name in enumerate(('A', 'B', 'heads', 'normalization'), 1)}
        history = []
        for epoch in range(1, 4):
            order = review.lora_io.case_order(list(range(8)), 812, epoch)
            receipts = [{'case_index': i, 'draws': review.lora_io.sampled_paths(cases[i], 812, epoch)} for i in order]
            current = dict(initial, heads='5' * 64)
            if epoch == 3:
                current.update(A='6' * 64, B='7' * 64)
            history.append({'epoch': epoch, 'adapting': epoch == 3, 'cases': 8, 'microbatches': 8,
                'optimizer_updates': 2, 'case_order': order,
                'sampling_sha256': hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
                'learning_rates': [1e-4 * (1 + math.cos(math.pi * s / 240)) / 2 for s in range((epoch-1)*2, epoch*2)],
                'elapsed_seconds': .1, 'loss_sum': 1., 'small_digests': current,
                'gradient_steps': [{'A': 0., 'B': .1 if epoch == 3 else 0., 'heads': .2},
                                   {'A': .01 if epoch == 3 else 0., 'B': .1 if epoch == 3 else 0., 'heads': .2}]})
        return config, cases, initial, history

    def test_stage_and_actual_gradient_proof(self):
        config, cases, initial, history = self.fixture()
        paired, gradients = review.check_history(history, initial, config, cases, list(range(8)), 812)
        self.assertEqual(len(paired), 3)
        self.assertEqual(len(gradients), 6)
        bad = copy.deepcopy(history)
        bad[2]['gradient_steps'][1]['A'] = 0.  # A state may change from AdamW decay alone.
        with self.assertRaises(ValueError):
            review.check_history(bad, initial, config, cases, list(range(8)), 812)
        bad = copy.deepcopy(history); bad[0]['gradient_steps'][0]['B'] = .1
        with self.assertRaises(ValueError):
            review.check_history(bad, initial, config, cases, list(range(8)), 812)

    def test_wrong_sampling_learning_rate_and_nonfinite_rejected(self):
        config, cases, initial, history = self.fixture()
        for field, value in [('sampling_sha256', '0'*64), ('learning_rates', [.001, .001]), ('loss_sum', float('nan'))]:
            bad = copy.deepcopy(history); bad[0][field] = value
            with self.assertRaises(ValueError):
                review.check_history(bad, initial, config, cases, list(range(8)), 812)

    def test_probe_roles_and_small_state_guards(self):
        config, _, _, _ = self.fixture()
        paths = ['fixture/%d.png' % i for i in range(8)]
        receipt = {'passed': True, 'probe_paths': paths, 'atol': 1e-4, 'rtol': 1e-4,
                   'feature_max_abs': 1e-5, 'probability_max_abs': 1e-6}
        self.assertTrue(review.check_parity(receipt, paths, config)['passed'])
        with self.assertRaises(ValueError):
            review.check_parity(dict(receipt, probe_paths=paths[:-1] + ['forbidden.png']), paths, config)
        state = {'mean': torch.zeros(4, 2), 'std': torch.ones(4, 2)}
        arrays = {k: v.numpy() for k, v in state.items()}
        schema = {k: {'shape': [4, 2], 'dtype': 'float32'} for k in state}
        review.check_state(state, arrays, schema)
        with self.assertRaises(ValueError):
            review.check_state(dict(state, extra=torch.zeros(1)), arrays, schema)
        with self.assertRaises(ValueError):
            review.check_state(dict(state, mean=state['mean'] + 1), arrays, schema)


if __name__ == '__main__':
    unittest.main()
