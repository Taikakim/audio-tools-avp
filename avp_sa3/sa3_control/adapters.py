"""Decoupled cross-attention control adapter for SA3 medium-base.

MuseControlLite/IP-Adapter pattern, written against this fork's `Attention`
(models/transformer.py:542). Each adapter branch:
  - reuses the FROZEN base cross-attn query projection (`base.to_q`) + attention
    kernel (`base.apply_attn`) + qk-norm + head config, so the math matches SA3;
  - has its OWN K/V (`to_k`/`to_v`) over the control tokens;
  - has a ZERO-INIT output (`to_out`), so at init the branch is a no-op and the
    wrapped model is byte-identical to the base — safe to train against a frozen base.

Control tokens reach the wrapped modules via a ContextVar (no change to the fork's
forward signature, no change to generate()): set `use_control_context(...)` around
the DiT forward; each `ControlledCrossAttention` reads `current_control_context()`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ── control-token channel (ContextVar; thread/async-safe, no module state) ──────

@dataclass
class ControlContext:
    control_tokens: Optional[torch.Tensor] = None   # (B, T_ctrl, control_dim)
    gain: float = 1.0                               # generation-time control strength (1.0 = as trained)


# Module-global holder for the active control tokens. We deliberately DON'T use a
# ContextVar: the DiT gradient-checkpoints every block, and torch.utils.checkpoint does
# NOT preserve ContextVar state into the backward RECOMPUTE, so the adapter branch would
# be skipped on recompute (saved-tensor count mismatch -> CheckpointError). A plain module
# global is read live during recompute, so the original forward and the recompute agree.
_ACTIVE = {"tokens": None, "gain": 1.0}


def current_control_context() -> Optional[ControlContext]:
    tok = _ACTIVE["tokens"]
    return ControlContext(tok, _ACTIVE["gain"]) if tok is not None else None


@contextmanager
def use_control_context(ctx: Optional[ControlContext]):
    _ACTIVE["tokens"] = ctx.control_tokens if ctx is not None else None
    _ACTIVE["gain"] = ctx.gain if ctx is not None else 1.0
    try:
        yield
    finally:
        _ACTIVE["tokens"] = None
        _ACTIVE["gain"] = 1.0


# ── head reshape (mirrors the fork's layout) ────────────────────────────────────

def _shape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, s, w = x.shape
    d = w // heads
    return x.view(b, s, heads, d).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, s, h * d)


def add_fractional_positions(tokens: torch.Tensor) -> torch.Tensor:
    """Deterministic time positions for the control tokens (keeps them ordered)."""
    if tokens.shape[1] <= 1:
        return tokens
    b, s, d = tokens.shape
    half = d // 2
    if half == 0:
        return tokens
    pos = torch.linspace(0.0, 1.0, s, device=tokens.device, dtype=tokens.dtype)
    freq = torch.exp(torch.linspace(0.0, 8.0, half, device=tokens.device, dtype=tokens.dtype))
    ang = pos[:, None] * freq[None, :] * torch.pi
    pe = tokens.new_zeros(s, d)
    pe[:, :half] = torch.sin(ang)
    pe[:, half:half + half] = torch.cos(ang)
    return tokens + pe.unsqueeze(0)


# ── the adapter branch + the wrapper ────────────────────────────────────────────

class DecoupledControlAdapter(nn.Module):
    """One decoupled cross-attention branch over the control tokens."""

    def __init__(self, base_attention: nn.Module, control_dim: int,
                 position_encoding: bool = True):
        super().__init__()
        self.control_dim = int(control_dim)
        self.position_encoding = bool(position_encoding)
        self.inner_dim = int(base_attention.dim)            # confirmed: Attention.dim
        self.dim_heads = int(base_attention.dim_heads)
        self.num_heads = int(base_attention.num_heads)
        self.kv_heads = int(base_attention.kv_heads)
        self.to_k = nn.Linear(self.control_dim, self.kv_heads * self.dim_heads, bias=False)
        self.to_v = nn.Linear(self.control_dim, self.kv_heads * self.dim_heads, bias=False)
        self.to_out = zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False))

    def forward(self, query_input: torch.Tensor, base_attention: nn.Module,
                control_tokens: torch.Tensor) -> torch.Tensor:
        adapter_dtype = self.to_k.weight.dtype
        ct = control_tokens.to(device=query_input.device, dtype=adapter_dtype)
        if self.position_encoding:
            ct = add_fractional_positions(ct)

        q = base_attention.to_q(query_input)
        if getattr(base_attention, "differential", False):
            q = q.chunk(2, dim=-1)[0]                        # differential keeps first half
        q = _shape_heads(q, self.num_heads).to(dtype=adapter_dtype)
        k = _shape_heads(self.to_k(ct), self.kv_heads)
        v = _shape_heads(self.to_v(ct), self.kv_heads)

        qk_norm = getattr(base_attention, "qk_norm", "none")
        if qk_norm == "l2":
            eps = getattr(base_attention, "qk_norm_eps", 1e-6)
            q = F.normalize(q, dim=-1, eps=eps)
            k = F.normalize(k, dim=-1, eps=eps)
        elif qk_norm != "none" and hasattr(base_attention, "apply_qk_layernorm"):
            q, k = base_attention.apply_qk_layernorm(q, k)

        attended = base_attention.apply_attn(
            q, k, v, causal=False,
            flex_attention_block_mask=None, flex_attention_score_mod=None,
            flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None,
        )
        return self.to_out(_merge_heads(attended)).to(dtype=query_input.dtype)


class ControlledCrossAttention(nn.Module):
    """Drop-in wrapper for an SA3 cross-attention module. Runs the (frozen) base
    cross-attn unchanged, then adds the control branch iff a control context is set."""

    def __init__(self, base_attention: nn.Module, control_dim: int,
                 position_encoding: bool = True):
        super().__init__()
        self.base_attention = base_attention
        self.adapter = DecoupledControlAdapter(base_attention, control_dim, position_encoding)

    def forward(self, x, context=None, **kwargs):
        base = self.base_attention(x, context=context, **kwargs)
        ctx = current_control_context()
        if ctx is not None and ctx.control_tokens is not None:
            base = base + ctx.gain * self.adapter(x, self.base_attention, ctx.control_tokens)
        return base
