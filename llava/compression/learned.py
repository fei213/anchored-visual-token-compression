"""Minimal learned latent compressor (new phase, perry 2026-09-25 spec).

Learned Representation Compression, NOT learned token selection:

    X in R^{T x N x D}  --C_theta-->  Z in R^{T x K_t x D}

Z is produced by cross-attention from K learnable latent queries onto the visual
tokens, so it need NOT be a subset of X. There is deliberately no MLP+TopK score
path -- that would just be ST-Rare's heuristic with a learned score.

Design choices kept deliberately small:
  * one shared set of queries across frames + a learnable per-frame temporal
    embedding (temporal position must stay encodable; D3 showed temporal order
    matters and v0's one-way gating was the only cross-frame channel);
  * per-frame output counts follow the same quota rule as the other methods, so
    the total content budget matches Uniform exactly at any ratio;
  * the module runs in float32 (training a small head on a frozen bf16 backbone)
    and casts its output back to the backbone's dtype at the boundary.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LatentCompressor(nn.Module):
    """K learnable queries + cross-attention over each frame's visual tokens."""

    def __init__(self, dim: int, num_frames: int = 16, max_tokens_per_frame: int = 64,
                 num_heads: int = 8, dropout: float = 0.0, init_scale: float = 0.02):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.max_tokens_per_frame = max_tokens_per_frame
        # a query bank sized for the largest per-frame quota; smaller quotas slice it
        self.queries = nn.Parameter(torch.randn(max_tokens_per_frame, dim) * init_scale)
        self.frame_emb = nn.Parameter(torch.randn(num_frames, dim) * init_scale)
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, N, D] (any float dtype) -> z: [T, max_tokens_per_frame, D] float32."""
        t, n, d = x.shape
        if t > self.num_frames:
            raise ValueError(f"got {t} frames, module built for {self.num_frames}")
        if d != self.dim:
            raise ValueError(f"dim mismatch: got {d}, expected {self.dim}")
        xf = x.float()
        kv = xf + self.frame_emb[:t].unsqueeze(1)                     # [T, N, D]
        q = self.queries.unsqueeze(0).expand(t, -1, -1)               # [T, K, D]
        z, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)  # [T, K, D]
        return self.norm_out(z)


def frame_quotas(content: int, num_frames: int) -> list[int]:
    """Same quota rule as the other methods: spread `content` over frames evenly."""
    base = content // num_frames
    rem = content - base * num_frames
    return [base + (1 if f < rem else 0) for f in range(num_frames)]


def compress_and_pack(module: LatentCompressor, features: torch.Tensor,
                      newline: torch.Tensor, content: int, return_z: bool = False):
    """Run the compressor and pack to the exact content budget.

    Returns (packed [content + T, D] in features.dtype, meta). Per-frame outputs
    are sliced to their quota, so a smaller quota simply uses the first K_t
    queries -- the budget is exact for every ratio.
    """
    t, n, d = features.shape
    quotas = frame_quotas(content, t)
    z = module(features)                                   # [T, Kmax, D] float32
    if max(quotas) > z.shape[1]:
        raise ValueError(f"quota {max(quotas)} exceeds query bank {z.shape[1]}")
    packed_tokens = []
    for f in range(t):
        packed_tokens.append(z[f, :quotas[f]].to(features.dtype))
        packed_tokens.append(newline.unsqueeze(0))
    packed = torch.cat(packed_tokens, dim=0)
    assert packed.shape == (content + t, d), (packed.shape, content + t, d)
    meta = dict(module="latent_cross_attn", queries=module.max_tokens_per_frame,
                num_heads=module.attn.num_heads, per_frame_count=quotas,
                content_tokens=content, separator_tokens=t)
    return (packed, meta, z) if return_z else (packed, meta)


def trainable_parameters(module: LatentCompressor):
    """Parameters the first version trains (everything in the compressor)."""
    return [p for p in module.parameters() if p.requires_grad]


def count_trainable(module: LatentCompressor) -> int:
    return sum(p.numel() for p in trainable_parameters(module))


def uniform_indices(n: int, k: int, device) -> torch.Tensor:
    """Uniform spatial grid of k indices over [0, n) -- mirrors the `uniform`
    baseline's per-frame rule so the anchor is exactly that selection."""
    if k >= n:
        return torch.arange(n, device=device)
    step = n / k
    return torch.tensor([int((j + 0.5) * step) for j in range(k)],
                        dtype=torch.long, device=device)


class ResidualAnchoredCompressor(nn.Module):
    """Z = A + alpha * C_theta(X), with A the Uniform anchors and alpha ~ 0 at init.

    Motivation: a free latent bottleneck replaces the visual tokens outright, and
    v0 collapsed to rank ~1, sending the LLM degenerate input. Anchoring to a
    selection that is known to work ("uniform") gives an identity / safe path:
    at alpha = 0 the compressor IS the uniform baseline, so it can only improve
    from there rather than falling off a cliff.
    """

    def __init__(self, dim: int, num_frames: int = 16, max_tokens_per_frame: int = 64,
                 num_heads: int = 8, dropout: float = 0.0, init_scale: float = 0.02,
                 alpha_init: float = 0.0):
        super().__init__()
        self.core = LatentCompressor(dim=dim, num_frames=num_frames,
                                     max_tokens_per_frame=max_tokens_per_frame,
                                     num_heads=num_heads, dropout=dropout,
                                     init_scale=init_scale)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.dim = dim

    @property
    def queries(self):
        return self.core.queries

    @property
    def frame_emb(self):
        return self.core.frame_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


def anchors_for(features: torch.Tensor, quotas: list[int]) -> torch.Tensor:
    """A[t] = the uniform-selected tokens of frame t, stacked to the max quota."""
    t, n, d = features.shape
    k_max = max(quotas)
    out = []
    for f in range(t):
        idx = uniform_indices(n, quotas[f], features.device)
        row = features[f, idx]
        if quotas[f] < k_max:                      # pad by repeating last token
            pad = row[-1:].expand(k_max - quotas[f], d)
            row = torch.cat([row, pad], dim=0)
        out.append(row)
    return torch.stack(out)                        # [T, k_max, D]


def compress_and_pack_anchored(module: ResidualAnchoredCompressor, features: torch.Tensor,
                               newline: torch.Tensor, content: int, return_z: bool = False):
    """Anchor-residual pack. At alpha == 0 the output equals the uniform baseline exactly."""
    t, n, d = features.shape
    quotas = frame_quotas(content, t)
    a = anchors_for(features, quotas)               # [T, max(quotas), D]
    k_needed = a.shape[1]
    if k_needed > module.core.max_tokens_per_frame:
        raise ValueError(f"per-frame quota {k_needed} exceeds the query bank "
                         f"{module.core.max_tokens_per_frame}; enlarge the bank")
    delta = module(features)[:, :k_needed]          # slice the bank to what is needed
    z = (a.float() + module.alpha * delta).to(features.dtype)
    packed_tokens = []
    for f in range(t):
        packed_tokens.append(z[f, :quotas[f]])
        packed_tokens.append(newline.unsqueeze(0))
    packed = torch.cat(packed_tokens, dim=0)
    assert packed.shape == (content + t, d), (packed.shape, content + t, d)
    meta = dict(module="uniform_anchor_residual", alpha=float(module.alpha.detach()),
                queries=module.core.max_tokens_per_frame, per_frame_count=quotas,
                content_tokens=content, separator_tokens=t)
    return (packed, meta, z) if return_z else (packed, meta)
