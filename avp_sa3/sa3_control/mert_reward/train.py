"""Train entrypoint for the MERT continuation reward head (Part 5, scaffold).

Fits :class:`~sa3_control.mert_reward.model.MERTRewardModel` on the output of
:func:`~sa3_control.mert_reward.dataset.build_reward_dataset`.

Loss::

    L = MSE(pred_reward, reward) [+ aux_weight * MSE(pred_aux, [ce, rhythm_sim, melody_sim])]

The TARGET reward is the Part-4 ``composite_reward`` (same weights), so the
learned model reproduces the selector and can then drop in as a re-ranker.

**Scaffold note (NO training run this pass):** there is no ``bestof_log`` corpus
yet (needs Part-4 runs). The loss (:func:`reward_loss`) and the single-step
update (:func:`train_step`) are pure / CPU-testable. :func:`train` and
:func:`main` are wired but call out to data/IO; a real run is deferred. The RL
fine-tune of the generator against this reward is explicitly deferred (named, not
designed here).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sa3_control.mert_reward.model import MERTRewardModel

__all__ = ["TrainConfig", "reward_loss", "train_step", "train", "main"]


@dataclass
class TrainConfig:
    npz: str = ""
    out_pt: str = "mert_reward_head.pt"
    emb_dim: int = 1024
    hidden: int = 512
    aux: bool = True
    aux_weight: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 256
    val_frac: float = 0.1
    device: str = "cpu"
    seed: int = 0


def reward_loss(
    pred_reward: torch.Tensor,
    target_reward: torch.Tensor,
    pred_aux: Optional[torch.Tensor] = None,
    target_aux: Optional[torch.Tensor] = None,
    *,
    aux_weight: float = 0.1,
) -> torch.Tensor:
    """Reward-regression loss = MSE(reward) [+ aux_weight * MSE(aux)].

    Pure: takes tensors, returns a scalar tensor. ``pred_reward`` /
    ``target_reward`` are ``(B,)``; the optional aux tensors are ``(B, 3)``.
    """
    loss = nn.functional.mse_loss(pred_reward, target_reward)
    if pred_aux is not None and target_aux is not None:
        loss = loss + aux_weight * nn.functional.mse_loss(pred_aux, target_aux)
    return loss


def _split_emb(x: torch.Tensor, emb_dim: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Split a ``(B, 4*emb_dim)`` row into ``(prev_emb, cand_emb)`` halves."""
    half = 2 * emb_dim
    return x[:, :half], x[:, half:]


def train_step(
    model: MERTRewardModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_target: Optional[torch.Tensor] = None,
    *,
    aux_weight: float = 0.1,
) -> float:
    """One optimization step. Returns the scalar loss value.

    ``x`` is ``(B, 4*emb_dim)``, ``y`` is ``(B,)``, ``aux_target`` is ``(B, 3)``.
    Pure enough to unit-test on a couple of random rows (CPU).
    """
    model.train()
    prev_emb, cand_emb = _split_emb(x, model.emb_dim)
    optimizer.zero_grad()
    pred_reward, pred_aux = model.forward_aux(prev_emb, cand_emb)
    loss = reward_loss(
        pred_reward, y, pred_aux, aux_target, aux_weight=aux_weight
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train(cfg: TrainConfig) -> MERTRewardModel:  # pragma: no cover - no corpus yet
    """Fit the reward head on a ``build_reward_dataset`` npz. Deferred (no corpus).

    Importable and wired, but not exercised this pass: there is no ``bestof_log``
    corpus to build the npz from yet. Once Part-4 runs exist, this runs end-to-end
    on CPU (the model is tiny).
    """
    torch.manual_seed(cfg.seed)
    data = np.load(cfg.npz)
    X = torch.from_numpy(data["X"].astype(np.float32))
    y = torch.from_numpy(data["y"].astype(np.float32))
    aux_target = None
    if cfg.aux:
        aux_target = torch.stack(
            [torch.from_numpy(data[k].astype(np.float32)) for k in ("ce", "rhythm_sim", "melody_sim")],
            dim=-1,
        )

    n = X.shape[0]
    n_val = max(1, int(round(cfg.val_frac * n)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = MERTRewardModel(emb_dim=cfg.emb_dim, hidden=cfg.hidden, aux=cfg.aux).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _epoch in range(cfg.epochs):
        for start in range(0, len(train_idx), cfg.batch_size):
            b = train_idx[start : start + cfg.batch_size]
            xb = X[b].to(cfg.device)
            yb = y[b].to(cfg.device)
            ab = aux_target[b].to(cfg.device) if aux_target is not None else None
            train_step(model, optimizer, xb, yb, ab, aux_weight=cfg.aux_weight)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(cfg),
        },
        cfg.out_pt,
    )
    return model


def _parse_args(argv: Optional[list] = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the MERT continuation reward head (Part 5).")
    p.add_argument("npz", help="dataset npz from build_reward_dataset")
    p.add_argument("--out-pt", default="mert_reward_head.pt")
    p.add_argument("--emb-dim", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--no-aux", action="store_true")
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    return TrainConfig(
        npz=a.npz,
        out_pt=a.out_pt,
        emb_dim=a.emb_dim,
        hidden=a.hidden,
        aux=not a.no_aux,
        aux_weight=a.aux_weight,
        lr=a.lr,
        weight_decay=a.weight_decay,
        epochs=a.epochs,
        batch_size=a.batch_size,
        val_frac=a.val_frac,
        device=a.device,
        seed=a.seed,
    )


def main() -> None:  # pragma: no cover - CLI entry, deferred (no corpus yet)
    """argv: <dataset.npz> [opts] -> fit and save .pt for the render-side swap-in."""
    cfg = _parse_args()
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
