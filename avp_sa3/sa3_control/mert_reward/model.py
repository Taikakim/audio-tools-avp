"""Tiny MLP reward head over MERT features (Part 5, scaffold).

The model consumes the MERT mid+upper embeddings of the *previous* window and a
*candidate* continuation and predicts the Part-4 ``composite_reward`` (and,
optionally, its components ce / rhythm_sim / melody_sim as auxiliary heads).

Input layout (matches ``dataset.build_reward_dataset``)::

    prev_emb = concat(prev_mid, prev_upper)   -> (B, 2 * emb_dim)
    cand_emb = concat(cand_mid, cand_upper)   -> (B, 2 * emb_dim)
    x        = concat(prev_emb, cand_emb)      -> (B, 4 * emb_dim)

The MLP is CPU-friendly (~1-2 M params at the defaults). No MERT model is loaded
here — embeddings are produced upstream (mir venv, ``mert_score.MERTEmbedder``);
this module only sees the pre-computed feature vectors, so it imports and runs on
CPU with no GPU dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MERTRewardModel", "AUX_FIELDS"]

# Auxiliary regression targets, in fixed output order.
AUX_FIELDS = ("ce", "rhythm_sim", "melody_sim")


class MERTRewardModel(nn.Module):
    """Small MLP over ``[prev_mid, prev_upper, cand_mid, cand_upper]`` -> reward.

    Parameters
    ----------
    emb_dim:
        Per-layer-group MERT embedding dimension (MERT-v1-330M = 1024). Each of
        prev/cand contributes ``mid`` + ``upper`` = ``2 * emb_dim``; the full
        input is ``4 * emb_dim``.
    hidden:
        Hidden width of the trunk MLP.
    aux:
        If True, attach a 3-way auxiliary head predicting (ce, rhythm_sim,
        melody_sim) in :data:`AUX_FIELDS` order.
    """

    def __init__(self, emb_dim: int = 1024, hidden: int = 512, aux: bool = True) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.in_dim = 4 * emb_dim
        self.aux = aux

        self.trunk = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.reward_head = nn.Linear(hidden, 1)
        self.aux_head = nn.Linear(hidden, len(AUX_FIELDS)) if aux else None

    def _check_emb(self, name: str, emb: torch.Tensor) -> None:
        if emb.dim() != 2 or emb.shape[-1] != 2 * self.emb_dim:
            raise ValueError(
                f"{name} must have shape (B, 2*emb_dim={2 * self.emb_dim}); "
                f"got {tuple(emb.shape)}"
            )

    def trunk_features(self, prev_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """Shared trunk features for ``(prev_emb, cand_emb)`` -> ``(B, hidden)``."""
        self._check_emb("prev_emb", prev_emb)
        self._check_emb("cand_emb", cand_emb)
        x = torch.cat([prev_emb, cand_emb], dim=-1)
        return self.trunk(x)

    def forward(self, prev_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """Predict reward for each row. Returns ``(B,)``.

        ``prev_emb`` / ``cand_emb`` are each ``(B, 2 * emb_dim)`` =
        ``concat(mid, upper)``.
        """
        h = self.trunk_features(prev_emb, cand_emb)
        return self.reward_head(h).squeeze(-1)

    def forward_aux(
        self, prev_emb: torch.Tensor, cand_emb: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Return ``(reward (B,), aux (B, 3) | None)`` in a single trunk pass."""
        h = self.trunk_features(prev_emb, cand_emb)
        reward = self.reward_head(h).squeeze(-1)
        aux = self.aux_head(h) if self.aux_head is not None else None
        return reward, aux
