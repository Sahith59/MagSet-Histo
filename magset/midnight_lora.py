"""Unmerged last-four-layer Q/V LoRA and fixed-normalizer field classifiers."""
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .midnight_features import classification_features

ADAPTER_TARGETS = tuple('encoder.layer.%d.attention.attention.%s' % (layer, name)
                        for layer in range(36, 40) for name in ('query', 'value'))


def _array_contract(arrays, arm):
    heads = ('direct',) if arm == 'direct' else ('binary', 'malignant') if arm == 'dual' else ()
    expected = {'mean', 'std'} | {h + '_' + suffix for h in heads for suffix in ('coef', 'intercept', 'classes')}
    if not heads or set(arrays) != expected:
        raise ValueError('Canonical field-head array keys/arm differ')
    mean, std = np.asarray(arrays['mean']), np.asarray(arrays['std'])
    if (mean.ndim != 2 or mean.shape[0] != 4 or not mean.shape[1] or mean.shape[1] % 2
            or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Invalid original fixed per-view normalizers')
    dimension = mean.shape[1]
    for head in heads:
        classes = np.arange(5) if head == 'direct' else np.arange(2) if head == 'binary' else np.arange(1, 5)
        outputs = 1 if head == 'binary' else len(classes)
        if (np.asarray(arrays[head + '_coef']).shape != (outputs, dimension + 4)
                or np.asarray(arrays[head + '_intercept']).shape != (outputs,)
                or not np.array_equal(arrays[head + '_classes'], classes)
                or not np.isfinite(arrays[head + '_coef']).all()
                or not np.isfinite(arrays[head + '_intercept']).all()):
            raise ValueError('Invalid fixed-C head shape, values or canonical class order')
    return heads, dimension


def small_state_schema(arrays, arm, rank=4):
    heads, dimension = _array_contract(arrays, arm)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError('Positive integer LoRA rank required')
    schema = {'mean': {'shape': [4, dimension], 'dtype': 'float32'},
              'std': {'shape': [4, dimension], 'dtype': 'float32'}}
    for path in ADAPTER_TARGETS:
        schema['backbone.' + path + '.lora_A'] = {'shape': [rank, dimension // 2], 'dtype': 'float32'}
        schema['backbone.' + path + '.lora_B'] = {'shape': [dimension // 2, rank], 'dtype': 'float32'}
    for head in heads:
        count = 5 if head == 'direct' else 1 if head == 'binary' else 4
        schema['heads.' + head + '.weight'] = {'shape': [count, dimension + 4], 'dtype': 'float32'}
        schema['heads.' + head + '.bias'] = {'shape': [count], 'dtype': 'float32'}
    return schema


class LoRALinear(nn.Module):
    """Frozen affine base plus dropout(x) A^T B^T alpha/r; never merged."""
    def __init__(self, base, rank=4, alpha=8, dropout=.1):
        super().__init__()
        if (not isinstance(base, nn.Linear) or base.weight.dtype != torch.float32
                or not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0
                or not np.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1):
            raise ValueError('Invalid FP32 linear module or LoRA settings')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling


class AdapterClassifier(nn.Module):
    def __init__(self, backbone, arrays, arm, rank=4, alpha=8, dropout=.1):
        super().__init__()
        heads, dimension = _array_contract(arrays, arm)
        self.arm, self.feature_dimension, self.rank = arm, dimension, rank
        self.backbone = backbone.requires_grad_(False)
        self._adapters = {}
        # Validate every target before making any structural change.
        for path in ADAPTER_TARGETS:
            base = backbone.get_submodule(path)
            if (not isinstance(base, nn.Linear) or base.in_features != dimension // 2
                    or base.out_features != dimension // 2):
                raise ValueError('Exact Q/V projection target shape differs: ' + path)
        device = backbone.get_submodule(ADAPTER_TARGETS[0]).weight.device
        for path in ADAPTER_TARGETS:
            parent, name = path.rsplit('.', 1)
            adapter = LoRALinear(backbone.get_submodule(path), rank, alpha, dropout)
            setattr(backbone.get_submodule(parent), name, adapter)
            self._adapters[path] = adapter
        self.register_buffer('mean', torch.as_tensor(np.array(arrays['mean'], copy=True), dtype=torch.float32, device=device))
        self.register_buffer('std', torch.as_tensor(np.array(arrays['std'], copy=True), dtype=torch.float32, device=device))
        self.heads = nn.ModuleDict()
        for head in heads:
            count = 5 if head == 'direct' else 1 if head == 'binary' else 4
            module = nn.Linear(dimension + 4, count, device=device, dtype=torch.float32)
            with torch.no_grad():
                module.weight.copy_(torch.as_tensor(arrays[head + '_coef'], device=device))
                module.bias.copy_(torch.as_tensor(arrays[head + '_intercept'], device=device))
            self.heads[head] = module
        self._adapt = False
        self.set_stage(False)

    def set_stage(self, adapt, training=True):
        self._adapt, self.training = bool(adapt), bool(training)
        self.backbone.eval()
        for adapter in self._adapters.values():
            adapter.lora_A.requires_grad_(self._adapt)
            adapter.lora_B.requires_grad_(self._adapt)
            adapter.dropout.train(self._adapt and self.training)
        self.heads.train(self.training)
        self.heads.requires_grad_(True)
        return self

    def train(self, mode=True):
        return self.set_stage(self._adapt, mode)

    def encode(self, images):
        features = classification_features(self.backbone(pixel_values=images).last_hidden_state)
        if features.shape[1] != self.feature_dimension:
            raise ValueError('Encoded feature dimension differs from fixed head')
        return features

    def logits(self, features, views):
        if (features.ndim != 2 or features.shape[1] != self.feature_dimension
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or views.ndim != 1 or len(views) != len(features)
                or views.dtype not in (torch.int32, torch.int64)
                or torch.any((views < 0) | (views > 3))):
            raise ValueError('Expected finite raw FP32 fields and integer per-field view indices')
        if features.device != self.mean.device or views.device != features.device:
            raise ValueError('Feature/view/model devices differ')
        normalized = (features - self.mean[views]) / self.std[views]
        flags = F.one_hot(views.long(), num_classes=4).to(dtype=torch.float32)
        x = torch.cat((normalized, flags), dim=1)
        return {name: module(x).squeeze(-1) if name == 'binary' else module(x)
                for name, module in self.heads.items()}

    def forward(self, images, views):
        features = self.encode(images)
        return features, self.logits(features, views)

    def _named_small(self):
        for path, adapter in self._adapters.items():
            for name in ('lora_A', 'lora_B'):
                yield 'backbone.' + path + '.' + name, getattr(adapter, name)
        yield from (('heads.' + name, value) for name, value in self.heads.named_parameters())

    def small_parameters(self):
        return [value for _, value in self._named_small()]

    def trainable_manifest(self):
        items = [{'name': name, 'shape': list(value.shape), 'count': value.numel(),
                  'requires_grad': value.requires_grad} for name, value in self._named_small()]
        adapter = sum(item['count'] for item in items if item['name'].startswith('backbone.'))
        total = sum(item['count'] for item in items)
        return {'parameters': items, 'adapter_parameters': adapter, 'head_parameters': total - adapter,
                'total_parameters': total,
                'active_parameters': sum(item['count'] for item in items if item['requires_grad'])}

    def small_state(self):
        return {name: value.detach().cpu().clone() for name, value in
                [('mean', self.mean), ('std', self.std)] + list(self._named_small())}

    def load_small_state(self, state):
        destinations = dict([('mean', self.mean), ('std', self.std)] + list(self._named_small()))
        if set(state) != set(destinations):
            raise ValueError('Small-state keys differ from exact adapter/head/normalizer whitelist')
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != destinations[name].shape
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()
                    or (name == 'std' and torch.any(value <= 0))):
                raise ValueError('Invalid small-state shape, dtype or values: ' + name)
        # The reference normalizers are immutable even across checkpoint loading.
        if any(not torch.equal(state[name].cpu(), destinations[name].detach().cpu()) for name in ('mean', 'std')):
            raise ValueError('Small state attempts to change fixed reference normalizers')
        with torch.no_grad():
            for name, value in state.items():
                destinations[name].copy_(value.to(device=destinations[name].device))
        return self

    def head_arrays(self):
        result = {'mean': self.mean.detach().cpu().numpy().copy(), 'std': self.std.detach().cpu().numpy().copy()}
        for name, module in self.heads.items():
            # Preserve the canonical saved-head schema used by five_class.load_model.
            result[name + '_coef'] = module.weight.detach().cpu().numpy().astype(np.float64)
            result[name + '_intercept'] = module.bias.detach().cpu().numpy().astype(np.float64)
            result[name + '_classes'] = np.arange(5) if name == 'direct' else np.arange(2) if name == 'binary' else np.arange(1, 5)
        return result

    def frozen_digest(self):
        excluded = {path + '.' + name for path in self._adapters for name in ('lora_A', 'lora_B')}
        digest = hashlib.sha256()
        for kind, values in [('parameter', self.backbone.named_parameters()), ('buffer', self.backbone.named_buffers())]:
            for name, value in values:
                if name in excluded:
                    continue
                header = json.dumps([kind, name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
                digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
                raw = memoryview(value.detach().cpu().contiguous().numpy()).cast('B')
                for offset in range(0, len(raw), 1024 * 1024):
                    digest.update(raw[offset:offset + 1024 * 1024])
        return digest.hexdigest()


def weighted_case_loss(logits, label, support_counts, arm, batch_denominator=4):
    """Eight fields/case; fixed denominator preserves equal case mass in last batch."""
    expected = {'direct'} if arm == 'direct' else {'binary', 'malignant'} if arm == 'dual' else set()
    if not expected or set(logits) != expected or batch_denominator != 4:
        raise ValueError('Expected registered arm, heads and fixed four-case denominator')
    first = next(iter(logits.values()))
    labels = torch.as_tensor(label, device=first.device)
    counts = torch.as_tensor(support_counts, device=first.device, dtype=torch.float32)
    if (labels.ndim != 1 or not len(labels) or len(labels) > 4
            or labels.dtype not in (torch.int32, torch.int64) or torch.any((labels < 0) | (labels > 4))
            or counts.shape != (5,) or not torch.isfinite(counts).all()
            or torch.any(counts <= 0) or torch.any(counts != counts.round())):
        raise ValueError('Invalid case labels or fitting-only class counts')
    labels = labels.long(); batch = len(labels); number = counts.sum()
    def shaped(name, outputs):
        value = logits[name]
        allowed = ((batch * 8,), (batch, 8)) if outputs == 1 else ((batch * 8, outputs), (batch, 8, outputs))
        if value.shape not in allowed or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Expected eight finite FP32 field logits per case')
        return value.reshape(batch, 8) if outputs == 1 else value.reshape(batch, 8, outputs)
    if arm == 'direct':
        value = shaped('direct', 5)
        loss = F.cross_entropy(value.reshape(-1, 5), labels[:, None].expand(-1, 8).reshape(-1), reduction='none').reshape(batch, 8).mean(1)
        return (loss * number / (5 * counts[labels])).sum() / batch_denominator
    binary = shaped('binary', 1); malignant = shaped('malignant', 4)
    target = (labels > 0).to(dtype=torch.float32)
    broad_counts = torch.stack((counts[0], counts[1:].sum()))
    binary_loss = F.binary_cross_entropy_with_logits(binary, target[:, None].expand(-1, 8), reduction='none').mean(1)
    binary_sum = (binary_loss * number / (2 * broad_counts[target.long()])).sum()
    selected = labels > 0
    # With no malignant cases, leave expert gradients absent so AdamW also skips
    # that head's moments/decay rather than updating it on benign-only batches.
    expert_sum = binary_sum.new_zeros(())
    if selected.any():
        truth = labels[selected] - 1
        expert = F.cross_entropy(malignant[selected].reshape(-1, 4), truth[:, None].expand(-1, 8).reshape(-1), reduction='none').reshape(-1, 8).mean(1)
        expert_sum = (expert * number / (4 * counts[labels[selected]])).sum()
    return .5 * (binary_sum + expert_sum) / batch_denominator
