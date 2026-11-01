"""Minimal PyTorch multi-head self-attention for teaching shape and masking.

No third-party model APIs. This implementation materializes the attention matrix,
so it is intentionally unsuitable for memory-efficient long-context inference.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Time O(B*L*D^2 + B*L^2*D), extra attention memory O(B*H*L^2).

    Query/key/value are learned projections; output has shape [B,L,D].
    'causal=True' prevents query position i from reading keys j > i.
    """
    def __init__(self, d_model: int, n_heads: int, *, causal: bool = False):
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be divisible by positive n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *, return_weights: bool = False):
        if x.ndim != 3 or x.shape[-1] != self.n_heads * self.d_head:
            raise ValueError("expected [batch, sequence, d_model]")
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().reshape(b, length, dim)
        result = self.out(context)
        return (result, weights) if return_weights else result
