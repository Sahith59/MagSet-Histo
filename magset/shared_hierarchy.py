"""One jointly fitted classifier for all available zooms and both decisions."""
from itertools import combinations
import math

import torch
from torch import nn
from torch.nn import functional as F


def zoom_masks():
    subsets = [s for n in range(1, 5) for s in combinations(range(4), n)]
    return torch.tensor([[i in s for i in range(4)] for s in subsets])


class SharedHierarchy(nn.Module):
    """Input axes are case, canonical zoom (40/100/200/400), UNI2-h feature."""

    def __init__(self, mean, scale, hidden, dropout):
        super().__init__()
        if (mean.ndim != 2 or mean.shape[0] != 4 or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or not (scale > 0).all()):
            raise ValueError("Expected four finite training normalizers with positive scales")
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float())
        self.classifier = nn.Sequential(
            nn.Linear(mean.shape[-1] + 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 5),
        )

    def classifier_inputs(self, case_views, masks):
        if case_views.ndim != 3 or case_views.shape[1:] != self.mean.shape:
            raise ValueError("Case descriptors must match the four canonical zooms")
        if masks.dtype != torch.bool or masks.ndim != 2 or masks.shape[1] != 4 or not masks.any(dim=1).all() or len(masks) == 0:
            raise ValueError("Each requested mask must select at least one of four zooms")
        standardized = (case_views - self.mean) / self.scale
        included = torch.where(masks[None, :, :, None], standardized[:, None, :, :], 0.0)
        pooled = included.sum(dim=2) / masks.sum(dim=1)[:, None]
        if not torch.isfinite(pooled).all():
            raise ValueError("A selected view has nonfinite features")
        pooled = F.normalize(pooled, dim=-1) * math.sqrt(self.mean.shape[-1])
        flags = masks.to(pooled.dtype).expand(len(case_views), -1, -1)
        return torch.cat((pooled, flags), dim=-1)

    def forward(self, case_views, masks):
        return self.classifier(self.classifier_inputs(case_views, masks))


def final_labels(scores):
    if scores.shape[-1] != 5 or not torch.isfinite(scores).all():
        raise ValueError("Five finite decision scores are required")
    return torch.where(scores[..., 0] >= 0, 1 + scores[..., 1:].argmax(dim=-1), 0)


def training_loss(scores, labels):
    """Give each class equal loss weight inside its decision, over all masks."""
    counts = torch.bincount(labels, minlength=5)
    if len(counts) != 5 or not (counts > 0).all():
        raise ValueError("All five classes are required in the full training batch")
    truth = labels[:, None].expand_as(scores[..., 0])
    malignant = truth > 0
    binary = F.binary_cross_entropy_with_logits(scores[..., 0], malignant.float(), reduction="none")
    binary_loss = (binary[~malignant].mean() + binary[malignant].mean()) / 2
    subtype_loss = F.cross_entropy(scores[..., 1:][malignant], truth[malignant] - 1,
                                   weight=counts[1:].to(scores.dtype).reciprocal())
    return (binary_loss + subtype_loss) / 2
