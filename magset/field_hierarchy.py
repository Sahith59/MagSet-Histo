"""Shared nonlinear field evidence before equal field/view aggregation."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .shared_hierarchy import SharedHierarchy


@dataclass(frozen=True)
class PackedFields:
    """Unpadded fields and audited raw case/view means in canonical zoom order.

    ``views`` contains the arithmetic field mean for each case and zoom. The
    loader verifies this relationship before packing; forward uses those same
    means to preserve the normalization of the raw-mean control.
    """

    fields: torch.Tensor
    case: torch.Tensor
    zoom: torch.Tensor
    views: torch.Tensor


class FieldHierarchy(SharedHierarchy):
    """Use exactly the control's parameters, with GELU before aggregation.

    Each case/mask uses the control's one raw-bag normalization factor and
    availability flags for every included field. Only the placement of GELU
    relative to uniform field/view averaging changes at evaluation time.
    """

    def forward(self, batch, masks):
        if not isinstance(batch, PackedFields):
            raise ValueError("Expected packed fields with audited case/view means")
        fields, case, zoom, views = batch.fields, batch.case, batch.zoom, batch.views
        if (views.ndim != 3 or views.shape[1:] != self.mean.shape or len(views) == 0
                or views.dtype != torch.float32 or fields.ndim != 2
                or fields.shape[1] != self.mean.shape[-1] or len(fields) == 0
                or fields.dtype != torch.float32):
            raise ValueError("Expected nonempty float32 fields and four-view case means")
        if (case.dtype != torch.long or zoom.dtype != torch.long
                or case.shape != (len(fields),) or zoom.shape != (len(fields),)
                or not ((case >= 0) & (case < len(views))).all()
                or not ((zoom >= 0) & (zoom < 4)).all()):
            raise ValueError("Each field requires valid long case and canonical zoom indices")
        if (masks.dtype != torch.bool or masks.ndim != 2 or masks.shape[1] != 4
                or len(masks) == 0 or not masks.any(dim=1).all()):
            raise ValueError("Each requested mask must select at least one of four zooms")
        if any(value.device != self.mean.device for value in (fields, case, zoom, views, masks)):
            raise ValueError("Packed fields, masks and model must share a device")

        # Exclude unused views before projecting. Even nonfinite fields in a
        # wholly unavailable view must have no effect on the requested answers.
        used_zooms = masks.any(dim=0)
        selected = used_zooms[zoom]
        fields, case, zoom = fields[selected], case[selected], zoom[selected]
        groups = case * 4 + zoom
        counts = torch.bincount(groups, minlength=len(views) * 4).reshape(len(views), 4)
        if not (counts[:, used_zooms] > 0).all():
            raise ValueError("Every selected case/view must contain at least one field")
        if not torch.isfinite(fields).all():
            raise ValueError("A selected field has nonfinite features")

        standardized_views = (views - self.mean) / self.scale
        included = torch.where(masks[None, :, :, None], standardized_views[:, None, :, :], 0.0)
        pooled = included.sum(dim=2) / masks.sum(dim=1)[:, None]
        if not torch.isfinite(pooled).all():
            raise ValueError("A selected view has nonfinite features")
        norms = torch.linalg.vector_norm(pooled, dim=-1).clamp_min(1.e-12)
        factors = math.sqrt(self.mean.shape[-1]) / norms

        first = self.classifier[0]
        standardized_fields = (fields - self.mean[zoom]) / self.scale[zoom]
        projected = F.linear(standardized_fields, first.weight[:, :-4])
        flags = F.linear(masks.to(projected.dtype), first.weight[:, -4:], first.bias)
        active = masks[:, zoom].T
        masked_projection = torch.where(active[..., None], projected[:, None, :], 0.0)
        hidden = self.classifier[1](masked_projection * factors[case, :, None] + flags[None, :, :])
        hidden = torch.where(active[..., None], hidden, 0.0)
        if not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite selected field activations")

        # Equal field mass inside each zoom, then equal mass across active zooms.
        # Dividing before accumulation also avoids dependence on other bag sizes.
        weighted = hidden / counts[case, zoom, None, None]
        view_hidden = hidden.new_zeros((len(views) * 4, len(masks), hidden.shape[-1]))
        view_hidden.index_add_(0, groups, weighted)
        bag_hidden = view_hidden.reshape(len(views), 4, len(masks), -1).sum(dim=1)
        bag_hidden = bag_hidden / masks.sum(dim=1)[None, :, None]
        scores = self.classifier[3](self.classifier[2](bag_hidden))
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite case/mask scores")
        return scores
