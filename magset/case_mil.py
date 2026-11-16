"""Whole-case supervision on frozen fields with matched mean and gated pooling.

Normalizers are external fitting-only artifacts. No field correspondence between
magnifications is assumed. All subset losses share one projected field sample.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import SUBSETS


def _requested_views(subset, device):
    if (not isinstance(subset, (tuple, list)) or not subset
            or any(isinstance(v, bool) or not isinstance(v, int) or v not in range(4) for v in subset)
            or len(set(subset)) != len(subset)):
        raise ValueError('Expected a nonempty unique subset of four view indices')
    requested = torch.zeros(4, device=device, dtype=torch.bool)
    requested[list(subset)] = True
    return requested


class CaseMIL(nn.Module):
    def __init__(self, dimension=3072, hidden=128, attention_hidden=32,
                 dropout=.25, pool='mean', arm='direct'):
        super().__init__()
        if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in (dimension, hidden, attention_hidden))
                or not 0 <= dropout < 1 or pool not in ('mean', 'attention')
                or arm not in ('direct', 'dual')):
            raise ValueError('Valid frozen-field dimensions, dropout, pooling and arm required')
        self.dimension, self.hidden = dimension, hidden
        self.attention_hidden, self.dropout = attention_hidden, float(dropout)
        self.pool, self.arm = pool, arm
        # Construct all shared parameters first for exact seed-paired initial states.
        self.project = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(dropout))
        outputs = {'direct': 5} if arm == 'direct' else {'binary': 1, 'malignant': 4}
        self.heads = nn.ModuleDict({name: nn.Linear(hidden+4, size) for name, size in outputs.items()})
        if pool == 'attention':
            self.field_value = nn.Linear(hidden, attention_hidden)
            self.field_gate = nn.Linear(hidden, attention_hidden)
            self.field_score = nn.Linear(attention_hidden, 1, bias=False)

    def encode_views(self, x, field_mask):
        """Project only actual finite fields; padded values never enter learned ops."""
        if (x.ndim != 4 or x.shape[1] != 4 or x.shape[3] != self.dimension
                or x.shape[0] == 0 or x.shape[2] == 0
                or field_mask.shape != x.shape[:3] or field_mask.dtype != torch.bool
                or x.device != field_mask.device or x.device != self.project[0].weight.device
                or x.dtype != self.project[0].weight.dtype):
            raise ValueError('Expected matching floating B x 4 x fields x D inputs and boolean field mask')
        present = field_mask.any(-1)
        if not present.any(-1).all() or not torch.isfinite(x[field_mask]).all():
            raise ValueError('Every case needs an observed view and finite actual fields')
        actual = self.project(x[field_mask])
        h = actual.new_zeros((*field_mask.shape, self.hidden))
        h[field_mask] = actual
        if self.pool == 'mean':
            pooled = h.sum(2)/field_mask.sum(2).clamp_min(1).unsqueeze(-1)
        else:
            score = self.field_score(torch.tanh(self.field_value(h))*torch.sigmoid(self.field_gate(h))).squeeze(-1)
            score = score.masked_fill(~field_mask, -torch.inf)
            # An absent view has zero mass, without an all-negative-infinity softmax.
            score = torch.where(present.unsqueeze(-1), score, torch.zeros_like(score))
            weights = torch.softmax(score, -1)*field_mask
            pooled = (h*weights.unsqueeze(-1)).sum(2)
        return pooled, present

    def outputs_from_views(self, views, present, subset=None):
        if subset is None:
            available = present
        else:
            available = present & _requested_views(subset, present.device)
        if not available.any(-1).all():
            raise ValueError('Requested subset must contain an observed view for every case')
        pooled = torch.where(available.unsqueeze(-1), views, torch.zeros_like(views)).sum(1)/available.sum(1).unsqueeze(-1)
        value = torch.cat((pooled, available.to(dtype=pooled.dtype)), -1)
        if self.arm == 'direct':
            logits = self.heads['direct'](value)
            return {'direct_logits': logits, 'joint_log_probabilities': F.log_softmax(logits, -1)}
        binary = self.heads['binary'](value).squeeze(-1)
        malignant = self.heads['malignant'](value)
        joint = torch.cat((F.logsigmoid(-binary).unsqueeze(-1),
                           F.logsigmoid(binary).unsqueeze(-1)+F.log_softmax(malignant, -1)), -1)
        return {'binary_logits': binary, 'malignant_logits': malignant, 'joint_log_probabilities': joint}

    def forward(self, x, field_mask, subset=None):
        if subset is not None:
            if field_mask.ndim != 3 or field_mask.shape[1] != 4 or field_mask.dtype != torch.bool:
                raise ValueError('Expected a four-view field mask')
            field_mask = field_mask & _requested_views(subset, field_mask.device)[None, :, None]
        return self.outputs_from_views(*self.encode_views(x, field_mask), subset=subset)


def all_subset_outputs(model, x, field_mask):
    """Canonical 15 x batch outputs with one shared projection/dropout draw."""
    views, present = model.encode_views(x, field_mask)
    if not present.all():
        raise ValueError('All 15 subsets require every case to have all four views')
    rows = [model.outputs_from_views(views, present, subset) for subset in SUBSETS]
    return {name: torch.stack([row[name] for row in rows]) for name in rows[0]}


def fixed_batch_case_loss(outputs, y5, fitting_counts, denominator=4):
    """Exact registered class-balanced loss, averaged equally over all 15 subsets.

The last two cases retain denominator four. An all-benign batch never depends
on expert logits, so its expert parameters retain absent gradients with set_to_none.
"""
    if denominator != 4 or not isinstance(outputs, dict):
        raise ValueError('The fixed four-case loss denominator is required')
    names = set(outputs)
    direct = names == {'joint_log_probabilities', 'direct_logits'}
    if not direct and names != {'joint_log_probabilities', 'binary_logits', 'malignant_logits'}:
        raise ValueError('Expected exact direct or dual subset outputs')
    first = outputs['joint_log_probabilities']
    labels = torch.as_tensor(y5, device=first.device)
    counts = torch.as_tensor(fitting_counts, dtype=first.dtype, device=first.device)
    if (labels.ndim != 1 or not 1 <= len(labels) <= 4 or labels.dtype not in (torch.int32, torch.int64)
            or torch.any((labels < 0) | (labels > 4)) or counts.shape != (5,)
            or not torch.isfinite(counts).all() or torch.any(counts <= 0) or torch.any(counts != counts.round())
            or first.shape != (len(SUBSETS), len(labels), 5) or not first.is_floating_point()
            or not all(value.device == first.device and value.dtype == first.dtype and torch.isfinite(value).all()
                       for value in outputs.values())):
        raise ValueError('Finite 15-subset logits, case labels and positive fitting class counts required')
    labels = labels.long(); number = counts.sum(); subsets, batch = first.shape[:2]
    if direct:
        logits = outputs['direct_logits']
        if logits.shape != first.shape:
            raise ValueError('Direct logits must have 15 x batch x five shape')
        loss = F.cross_entropy(logits.reshape(-1, 5), labels.repeat(subsets), reduction='none').reshape(subsets, batch).mean(0)
        return (loss*number/(5*counts[labels])).sum()/denominator
    binary, malignant = outputs['binary_logits'], outputs['malignant_logits']
    if binary.shape != (subsets, batch) or malignant.shape != (subsets, batch, 4):
        raise ValueError('Binary and expert logits have incorrect subset/batch shapes')
    target = (labels > 0).to(dtype=first.dtype)
    broad_counts = torch.stack((counts[0], counts[1:].sum()))
    losses = F.binary_cross_entropy_with_logits(binary, target.unsqueeze(0).expand_as(binary), reduction='none').mean(0)
    binary_sum = (losses*number/(2*broad_counts[target.long()])).sum()
    expert_sum = binary_sum.new_zeros(())
    selected = labels > 0
    if selected.any():
        truth = labels[selected]-1
        losses = F.cross_entropy(malignant[:, selected].reshape(-1, 4), truth.repeat(subsets), reduction='none').reshape(subsets, -1).mean(0)
        expert_sum = (losses*number/(4*counts[labels[selected]])).sum()
    return .5*(binary_sum+expert_sum)/denominator
