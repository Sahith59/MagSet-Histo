"""Tiny CPU LoRA fixtures; no real Midnight model, data or pretrained weights."""
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn
    from magset import midnight_lora as lora
except ImportError:
    torch = None


def arrays(arm, dimension=4):
    result = {'mean': np.arange(4 * dimension, dtype=np.float32).reshape(4, dimension) / 100,
              'std': np.ones((4, dimension), dtype=np.float32) * 2}
    heads = ('direct',) if arm == 'direct' else ('binary', 'malignant')
    for name in heads:
        count = 5 if name == 'direct' else 1 if name == 'binary' else 4
        result[name + '_coef'] = np.arange(count * (dimension + 4), dtype=np.float64).reshape(count, dimension + 4) / 100
        result[name + '_intercept'] = np.arange(count, dtype=np.float64) / 10
        result[name + '_classes'] = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
    return result


if torch is not None:
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList()
            for _ in range(40):
                layer = nn.Module()
                layer.attention = nn.Module()
                layer.attention.attention = nn.Module()
                layer.attention.attention.query = nn.Linear(2, 2)
                layer.attention.attention.key = nn.Linear(2, 2)
                layer.attention.attention.value = nn.Linear(2, 2)
                self.encoder.layer.append(layer)
            self.register_buffer('fixed_buffer', torch.ones(1))

        def forward(self, pixel_values):
            x = pixel_values
            for layer in self.encoder.layer[36:]:
                attention = layer.attention.attention
                x = x + .05 * (attention.query(x) + attention.value(x))
            return SimpleNamespace(last_hidden_state=x[:, None, :].expand(-1, 257, -1))


def toy_features(tokens):
    return torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=1)


@unittest.skipIf(torch is None, 'PyTorch is absent in the lightweight environment')
class MidnightLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)

    def test_unmerged_zero_initialization_and_eval(self):
        base = nn.Linear(3, 2)
        before = base.weight.detach().clone()
        wrapper = lora.LoRALinear(base, rank=4, alpha=8, dropout=.1)
        x = torch.randn(5, 3)
        torch.testing.assert_close(wrapper(x), base(x), rtol=0, atol=0)
        self.assertFalse(base.weight.requires_grad)
        with torch.no_grad():
            wrapper.lora_B.fill_(.3)
        wrapper.eval()
        expected = base(x) + 2 * (x @ wrapper.lora_A.T @ wrapper.lora_B.T)
        torch.testing.assert_close(wrapper(x), expected)
        wrapper.train(); wrapper.eval(); wrapper.train(); wrapper.eval()
        torch.testing.assert_close(base.weight, before, rtol=0, atol=0)

    def test_real_dimension_schema_counts_without_backbone_construction(self):
        for arm in ('direct', 'dual'):
            schema = lora.small_state_schema(arrays(arm, 3072), arm)
            adapter = sum(np.prod(v['shape']) for k, v in schema.items() if k.startswith('backbone.'))
            heads = sum(np.prod(v['shape']) for k, v in schema.items() if k.startswith('heads.'))
            self.assertEqual(adapter, 98304)
            self.assertEqual(heads, 15385)
            self.assertEqual(len(lora.ADAPTER_TARGETS), 8)

    def test_fixed_head_initialization_parity_and_flags(self):
        for arm in ('direct', 'dual'):
            initial = arrays(arm)
            model = lora.AdapterClassifier(ToyBackbone(), initial, arm)
            features = torch.randn(8, 4)
            views = torch.arange(8) % 4
            actual = model.logits(features, views)
            x = np.concatenate(((features.numpy() - initial['mean'][views]) / initial['std'][views],
                                np.eye(4, dtype=np.float32)[views]), axis=1)
            for name in actual:
                expected = x @ initial[name + '_coef'].T + initial[name + '_intercept']
                if name == 'binary':
                    expected = expected[:, 0]
                np.testing.assert_allclose(actual[name].detach().numpy(), expected, atol=2e-7, rtol=2e-7)
            exported = model.head_arrays()
            self.assertEqual(set(exported), set(initial))
            self.assertEqual(exported['mean'].dtype, np.float32)
            self.assertEqual(exported['std'].dtype, np.float32)
            for name in exported:
                np.testing.assert_allclose(exported[name], initial[name], atol=1e-7)
                if name.endswith(('_coef', '_intercept')):
                    self.assertEqual(exported[name].dtype, np.float64)

    def test_stages_gradients_and_frozen_invariance(self):
        model = lora.AdapterClassifier(ToyBackbone(), arrays('dual'), 'dual')
        optimizer = torch.optim.AdamW(model.small_parameters(), lr=.01, weight_decay=0)
        digest = model.frozen_digest()
        adapter = next(iter(model._adapters.values()))
        original_a = adapter.lora_A.detach().clone()
        original_b = adapter.lora_B.detach().clone()
        images, views = torch.randn(8, 2), torch.arange(8) % 4
        with patch.object(lora, 'classification_features', side_effect=toy_features):
            _, logits = model(images, views)
            loss = lora.weighted_case_loss(logits, torch.tensor([2]), [11, 25, 2, 5, 3], 'dual')
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.testing.assert_close(adapter.lora_B, original_b, rtol=0, atol=0)
            self.assertFalse(adapter.lora_A.requires_grad)
            self.assertFalse(adapter.dropout.training)
            model.set_stage(True)
            self.assertFalse(model.backbone.training)
            self.assertTrue(adapter.dropout.training)
            for step in range(2):
                _, logits = model(images, views)
                lora.weighted_case_loss(logits, torch.tensor([2]), [11, 25, 2, 5, 3], 'dual').backward()
                if step == 0:
                    self.assertEqual(float(adapter.lora_A.grad.abs().sum()), 0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(adapter.lora_B, original_b))
        self.assertFalse(torch.equal(adapter.lora_A, original_a))
        self.assertEqual(model.frozen_digest(), digest)
        model.eval()
        self.assertFalse(adapter.dropout.training)
        self.assertFalse(model.backbone.training)
        model.train()
        self.assertTrue(adapter.dropout.training)
        self.assertFalse(model.backbone.training)
        manifest = model.trainable_manifest()
        self.assertEqual(manifest['adapter_parameters'], 128)
        self.assertEqual(manifest['head_parameters'], 45)
        self.assertEqual(manifest['total_parameters'], manifest['active_parameters'])

    def test_small_state_npz_roundtrip_and_strict_guards(self):
        backbone = ToyBackbone()
        second_backbone = copy.deepcopy(backbone)
        model = lora.AdapterClassifier(backbone, arrays('direct'), 'direct')
        second = lora.AdapterClassifier(second_backbone, arrays('direct'), 'direct')
        with torch.no_grad():
            next(iter(model._adapters.values())).lora_B.fill_(.3)
            model.heads['direct'].weight.add_(.2)
        state = model.small_state()
        self.assertEqual(set(state), set(lora.small_state_schema(arrays('direct'), 'direct')))
        self.assertFalse(any('.base.' in name for name in state))
        stream = io.BytesIO()
        np.savez(stream, **{name: value.numpy() for name, value in state.items()})
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as saved:
            restored = {name: torch.from_numpy(saved[name]) for name in saved.files}
        second.load_small_state(restored)
        model.eval(); second.eval()
        with patch.object(lora, 'classification_features', side_effect=toy_features):
            images, views = torch.randn(8, 2), torch.arange(8) % 4
            for first, other in zip(model(images, views), second(images, views)):
                if isinstance(first, dict):
                    torch.testing.assert_close(first['direct'], other['direct'], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(first, other, rtol=0, atol=0)
        bad = dict(restored); bad.pop('mean')
        with self.assertRaises(ValueError):
            second.load_small_state(bad)
        bad = dict(restored); bad['mean'] = bad['mean'] + 1
        with self.assertRaises(ValueError):
            second.load_small_state(bad)
        bad = dict(restored); bad['heads.direct.bias'] = bad['heads.direct.bias'].double()
        with self.assertRaises(ValueError):
            second.load_small_state(bad)

    def test_class_weights_expert_scale_and_fixed_last_batch_denominator(self):
        counts = [11, 25, 2, 5, 3]
        logits = {'direct': torch.zeros(16, 5, requires_grad=True)}
        actual = lora.weighted_case_loss(logits, torch.tensor([0, 2]), counts, 'direct')
        expected = np.log(5) * (46 / (5 * 11) + 46 / (5 * 2)) / 4
        self.assertAlmostEqual(float(actual.detach()), expected, places=6)
        dual = {'binary': torch.zeros(8, requires_grad=True), 'malignant': torch.zeros(8, 4, requires_grad=True)}
        benign = lora.weighted_case_loss(dual, torch.tensor([0]), counts, 'dual')
        self.assertAlmostEqual(float(benign.detach()), .5 * np.log(2) * 46 / (2 * 11) / 4, places=6)
        benign.backward()
        self.assertIsNone(dual['malignant'].grad)
        rare = lora.weighted_case_loss(dual, torch.tensor([2]), counts, 'dual')
        expected = .5 * (np.log(2) * 46 / (2 * 35) + np.log(4) * 46 / (4 * 2)) / 4
        self.assertAlmostEqual(float(rare.detach()), expected, places=6)
        with self.assertRaises(ValueError):
            lora.weighted_case_loss(logits, torch.tensor([0, 2]), counts, 'direct', batch_denominator=2)
        with self.assertRaises(ValueError):
            lora.weighted_case_loss({'direct': torch.zeros(7, 5)}, torch.tensor([2]), counts, 'direct')


if __name__ == '__main__':
    unittest.main()
