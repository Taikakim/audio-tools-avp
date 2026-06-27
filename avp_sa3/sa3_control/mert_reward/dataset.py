"""Reward-dataset builder (Part 5, scaffold).

Consumes Part-4 best-of-N selector logs (``bestof_log``) and turns them into a
training table for :class:`~sa3_control.mert_reward.model.MERTRewardModel`.

Each log entry describes one scored candidate window::

    {"prev_wav": ".../win2_best.wav" | null,   # null for the first window (k==0)
     "cand_wav": ".../win3_cand0.wav",
     "ce": 7.21, "rhythm_sim": 0.83, "melody_sim": 0.71, "reward": 8.93}

For each entry we MERT-embed ``prev_wav`` and ``cand_wav`` (mid + upper layer
groups, mean-pooled + L2-normalized), and write an ``.npz``::

    X       : (n, 4 * emb_dim)  = concat(prev_mid, prev_upper, cand_mid, cand_upper)
    y       : (n,)              = reward (the composite_reward target)
    ce      : (n,)              \
    rhythm_sim : (n,)            > component columns, for auxiliary heads
    melody_sim : (n,)           /

**Scaffold note:** the MERT embedding step needs the model (mir venv / GPU). The
builder therefore takes an injectable ``embedder`` — a callable
``embed(wav_path) -> {"mid": (D,), "upper": (D,)}`` — so the npz **schema** is
unit-testable on CPU with a fake embedder. When ``embedder is None`` the real
``mert_score.MERTEmbedder`` is loaded lazily (untested in this pass: requires the
mir venv with transformers + the MERT checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = ["build_reward_dataset", "load_bestof_logs", "Embedder"]

# An embedder maps a wav path -> {"mid": (D,) float array, "upper": (D,) float array}.
# ``None`` path (the first window's missing prev_wav) must be tolerated by callers,
# handled here by substituting a zero vector.
Embedder = Callable[[str], "dict[str, np.ndarray]"]

# Required component columns carried alongside the reward target (aux heads).
_COMPONENT_FIELDS = ("ce", "rhythm_sim", "melody_sim")


def load_bestof_logs(bestof_logs: Sequence[str]) -> "list[dict[str, Any]]":
    """Flatten one or more bestof_log JSON files into a single list of entries.

    Each file is expected to be a JSON list of candidate dicts (or a dict with a
    ``"log"`` key holding that list). Missing files raise ``FileNotFoundError``.
    """
    entries: list[dict[str, Any]] = []
    for path in bestof_logs:
        with open(path, "r") as f:
            obj = json.load(f)
        rows = obj["log"] if isinstance(obj, dict) else obj
        entries.extend(rows)
    return entries


def _embed_or_zero(
    embedder: Embedder, wav_path: Optional[str], emb_dim: int
) -> np.ndarray:
    """Embed a wav -> concat(mid, upper) (2*emb_dim,); zeros if ``wav_path`` is None."""
    if wav_path is None:
        return np.zeros(2 * emb_dim, dtype=np.float32)
    emb = embedder(wav_path)
    mid = np.asarray(emb["mid"], dtype=np.float32).reshape(-1)
    upper = np.asarray(emb["upper"], dtype=np.float32).reshape(-1)
    return np.concatenate([mid, upper], axis=0)


def build_reward_dataset(
    bestof_logs: "list[str]",
    out_npz: str,
    *,
    embedder: Optional[Embedder] = None,
    emb_dim: int = 1024,
    layers_mid: Sequence[int] = (3, 4, 5, 6),
    layers_upper: Sequence[int] = (23,),
) -> str:
    """Build the reward-regression dataset npz from Part-4 selector logs.

    Parameters
    ----------
    bestof_logs:
        Paths to ``bestof_log`` JSON files (each a list of candidate dicts).
    out_npz:
        Destination ``.npz`` path.
    embedder:
        Callable ``embed(wav_path) -> {"mid": (D,), "upper": (D,)}``. If ``None``,
        the real ``sa3_control.mert_score.MERTEmbedder`` is loaded lazily (mir
        venv only — untested in this scaffold pass).
    emb_dim:
        MERT per-group embedding dim (used to size the zero vector for the first
        window's missing ``prev_wav``).
    layers_mid / layers_upper:
        MERT hidden-state layer groups; only used when constructing the default
        embedder.

    Returns the written ``out_npz`` path.
    """
    if embedder is None:  # pragma: no cover - needs mir venv + MERT checkpoint
        embedder = _default_mert_embedder(layers_mid, layers_upper)

    entries = load_bestof_logs(bestof_logs)
    if not entries:
        raise ValueError("no entries found in bestof_logs")

    X_rows: list[np.ndarray] = []
    y: list[float] = []
    components: dict[str, list[float]] = {k: [] for k in _COMPONENT_FIELDS}

    for e in entries:
        prev_vec = _embed_or_zero(embedder, e.get("prev_wav"), emb_dim)
        cand_vec = _embed_or_zero(embedder, e["cand_wav"], emb_dim)
        X_rows.append(np.concatenate([prev_vec, cand_vec], axis=0))
        y.append(float(e["reward"]))
        for k in _COMPONENT_FIELDS:
            components[k].append(float(e[k]))

    X = np.asarray(X_rows, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "X": X,
        "y": np.asarray(y, dtype=np.float32),
    }
    for k in _COMPONENT_FIELDS:
        arrays[k] = np.asarray(components[k], dtype=np.float32)

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return str(out)


def _default_mert_embedder(
    layers_mid: Sequence[int], layers_upper: Sequence[int]
) -> Embedder:  # pragma: no cover - needs mir venv + MERT checkpoint
    """Lazily construct the real MERT embedder (mir venv). Untested in this pass."""
    import torchaudio  # type: ignore

    from sa3_control.mert_score import MERTEmbedder  # type: ignore

    mert = MERTEmbedder(layers_mid=tuple(layers_mid), layers_upper=tuple(layers_upper))

    def _embed(wav_path: str) -> "dict[str, np.ndarray]":
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0).numpy()  # mono
        return mert.embed(wav, sr)

    return _embed
