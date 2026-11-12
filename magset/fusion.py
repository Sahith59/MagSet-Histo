"""Small hierarchical field/view fusion baselines; no physical field pairing assumed."""
import torch
from torch import nn


class HierarchicalFusion(nn.Module):
    def __init__(self, dimension, hidden=128, field_pool='mean', view_fusion='mean', dropout=.2, heads=4):
        super().__init__()
        if field_pool not in ('mean','attention') or view_fusion not in ('mean','gated','attention'):
            raise ValueError('Unknown field pooling or view fusion')
        self.field_pool = field_pool; self.view_fusion = view_fusion
        self.project = nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Dropout(dropout))
        if field_pool == 'attention':
            self.field_value = nn.Linear(hidden,hidden)
            self.field_gate = nn.Linear(hidden,hidden)
            self.field_score = nn.Linear(hidden,1,bias=False)
        self.scale_embedding = nn.Embedding(4,hidden)
        nn.init.normal_(self.scale_embedding.weight,std=.02)
        if view_fusion == 'gated':
            self.view_gate = nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        elif view_fusion == 'attention':
            self.attention = nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True)
            self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
            self.feedforward = nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*hidden,hidden))
            self.residual_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,8))

    def pool_fields(self, x, field_mask):
        # x: B x views x fields x encoder_dimension. Padding is removed before any learned operation.
        if x.ndim != 4 or field_mask.shape != x.shape[:3] or field_mask.dtype != torch.bool:
            raise ValueError('Expected field bags and a Boolean field mask')
        clean = torch.where(field_mask.unsqueeze(-1),x,0.)
        h = self.project(clean)
        h = torch.where(field_mask.unsqueeze(-1),h,0.)
        count = field_mask.sum(-1,keepdim=True).clamp_min(1)
        if self.field_pool == 'mean':
            pooled = h.sum(-2)/count
        else:
            logits = self.field_score(torch.tanh(self.field_value(h))*torch.sigmoid(self.field_gate(h))).squeeze(-1)
            # A missing view has no fields: finite dummy logits avoid all-masked softmax NaNs.
            logits = logits.masked_fill(~field_mask,-torch.finfo(logits.dtype).max)
            weights = torch.softmax(logits,-1)*field_mask
            weights = weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            pooled = (weights.unsqueeze(-1)*h).sum(-2)
        return pooled

    def fuse_views(self, pooled, observed, scale_ids=None):
        # The scale IDs travel with their tokens, so reordering available views changes no prediction.
        if observed.shape != pooled.shape[:2] or observed.dtype != torch.bool or not observed.any(-1).all():
            raise ValueError('Every case requires at least one observed view')
        if scale_ids is None:
            scale_ids = torch.arange(pooled.shape[1],device=pooled.device)
        if scale_ids.ndim == 1:
            scale_ids = scale_ids.expand(pooled.shape[0],-1)
        if scale_ids.shape != observed.shape or torch.any(scale_ids < 0) or torch.any(scale_ids > 3):
            raise ValueError('Invalid scale IDs')
        if torch.any(torch.sort(scale_ids,dim=-1).values[:,1:] == torch.sort(scale_ids,dim=-1).values[:,:-1]):
            raise ValueError('Duplicate scale IDs')
        h = torch.where(observed.unsqueeze(-1),pooled,0.) + self.scale_embedding(scale_ids)
        h = torch.where(observed.unsqueeze(-1),h,0.)
        count = observed.sum(-1,keepdim=True)
        if self.view_fusion == 'gated':
            context = h.sum(1)/count
            logits = self.view_gate(torch.cat([h,context.unsqueeze(1).expand_as(h)],dim=-1)).squeeze(-1)
            weights = torch.softmax(logits.masked_fill(~observed,-torch.finfo(logits.dtype).max),dim=1)
            fused = (h*weights.unsqueeze(-1)).sum(1)
        else:
            if self.view_fusion == 'attention':
                attended,_ = self.attention(h,h,h,key_padding_mask=~observed,need_weights=True)
                h = self.norm1(h+self.residual_dropout(attended))
                h = self.norm2(h+self.residual_dropout(self.feedforward(h)))
                h = torch.where(observed.unsqueeze(-1),h,0.)
            fused = h.sum(1)/count
        return self.classifier(fused)

    def forward(self, x, field_mask, observed, scale_ids=None):
        if observed.shape != field_mask.shape[:2] or observed.dtype != torch.bool:
            raise ValueError('Observed-view mask shape/type differs')
        if torch.any(observed & ~field_mask.any(-1)):
            raise ValueError('An observed view must contain at least one field')
        effective = field_mask & observed.unsqueeze(-1)
        return self.fuse_views(self.pool_fields(x,effective),observed,scale_ids)
