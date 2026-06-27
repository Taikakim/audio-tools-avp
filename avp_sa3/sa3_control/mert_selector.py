"""MERT + Audiobox continuation scorer (the best-of-N reward, mir venv).

Scores how well a *candidate* next-window continues a *previous* window of audio,
balancing three signals:

  * **ce**          -- Audiobox Content Enjoyment (quality north-star, 1-10 scale).
  * **rhythm_sim**  -- cosine(MERT-mid(cand), MERT-mid(prev)); MID layers (3,4,5,6)
                       carry groove/rhythm. HIGH = same groove (continuity). Rewarded.
  * **melody_sim**  -- cosine(MERT-upper(cand), MERT-upper(prev)); UPPER layer (23)
                       carries melodic/harmonic content. This is a **target BAND**, not
                       a maximum -- maximizing it *is* the loop (literal repeat). We
                       reward proximity to ``band_center`` and penalize deviation.

Composite::

    score = w_ce*ce + w_rhythm*rhythm_sim - w_melody*abs(melody_sim - band_center)

Public interface
----------------
    score_continuation(prev_wav, cand_wav, *, sr, embedder=None, ce_fn=None,
                       weights=None) -> {"ce","rhythm_sim","melody_sim","score"}
    best_of(prev_wav, cand_wavs, *, ...) -> best_index (int)

Pure-math helpers (``cosine_sim``, ``composite_score``, ``score_from_embeddings``)
are model-free and unit-tested. The model path (``MERTEmbedder``, ``audiobox_ce``)
is importable but exercises the GPU/heavy models, so it is left untested here.

Run as a CLI for the cross-venv best-of-N seam (see the longform design spec, Part 4)::

    mir/bin/python -m sa3_control.mert_selector job.json scores.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

MERT_ID = "m-a-p/MERT-v1-330M"
MERT_SR = 24000
LAYERS_MID = (3, 4, 5, 6)
LAYERS_UPPER = (23,)

# A waveform is either an in-memory mono float array, or a path to an audio file.
WavLike = Union[np.ndarray, str]


# --------------------------------------------------------------------------- #
# Reward weights
# --------------------------------------------------------------------------- #
@dataclass
class RewardWeights:
    """Weights + melody-sim band for the composite continuation reward.

    The melody band (``band_lo``/``band_hi``) is a *target window*: below it the
    candidate is unrelated/incoherent, above it the candidate is a literal repeat
    (the loop). ``band_center`` is the deviation anchor for the penalty term.
    """

    w_ce: float = 1.0
    w_rhythm: float = 1.0
    w_melody: float = 1.0
    band_center: float = 0.675  # midpoint of the melody-sim band
    band_lo: float = 0.55
    band_hi: float = 0.80

    def in_band(self, melody_sim: float) -> bool:
        return self.band_lo <= melody_sim <= self.band_hi


# --------------------------------------------------------------------------- #
# Pure scoring math (model-free, unit-tested)
# --------------------------------------------------------------------------- #
def cosine_sim(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 if either is ~zero."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"cosine_sim shape mismatch: {a.shape} vs {b.shape}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def melody_band_penalty(melody_sim: float, band_center: float) -> float:
    """Deviation of melody-sim from the band center (always >= 0). This is the term
    that turns 'more melodic similarity' from an objective into a *band* target."""
    return abs(float(melody_sim) - float(band_center))


def composite_score(
    ce: float,
    rhythm_sim: float,
    melody_sim: float,
    *,
    weights: RewardWeights,
) -> float:
    """The single source of truth for the continuation reward.

        score = w_ce*ce + w_rhythm*rhythm_sim - w_melody*|melody_sim - band_center|
    """
    return (
        weights.w_ce * float(ce)
        + weights.w_rhythm * float(rhythm_sim)
        - weights.w_melody * melody_band_penalty(melody_sim, weights.band_center)
    )


def score_from_embeddings(
    prev_mid: Optional[np.ndarray],
    prev_upper: Optional[np.ndarray],
    cand_mid: np.ndarray,
    cand_upper: np.ndarray,
    ce: float,
    *,
    weights: RewardWeights,
) -> dict:
    """Assemble a continuation score dict from already-computed MERT embeddings + CE.

    If the reference embeddings (``prev_*``) are None (the first window has no
    predecessor), rhythm_sim and melody_sim fall back to neutral values
    (rhythm_sim=0, melody_sim=band_center => zero melody penalty), so the score
    reduces to a CE-only ranking. This keeps the first-window selection well-defined.
    """
    if prev_mid is None or prev_upper is None:
        rhythm_sim = 0.0
        melody_sim = weights.band_center
    else:
        rhythm_sim = cosine_sim(cand_mid, prev_mid)
        melody_sim = cosine_sim(cand_upper, prev_upper)
    score = composite_score(ce, rhythm_sim, melody_sim, weights=weights)
    return {
        "ce": float(ce),
        "rhythm_sim": float(rhythm_sim),
        "melody_sim": float(melody_sim),
        "score": float(score),
    }


def best_index(scores: Sequence[dict]) -> int:
    """argmax over a list of score dicts by their 'score' field. Ties -> lowest index."""
    if not scores:
        raise ValueError("best_index: empty score list")
    return int(max(range(len(scores)), key=lambda i: scores[i]["score"]))


# --------------------------------------------------------------------------- #
# Model path: MERT embedder + Audiobox CE (importable, GPU/model-heavy)
# --------------------------------------------------------------------------- #
class MERTEmbedder:
    """Loads MERT-v1-330M once; embeds a mono waveform into mean-pooled, L2-normalized
    ``{'mid': (D,), 'upper': (D,)}`` over the requested hidden-state layer groups.

    Reference: jobs/b1ee18e7/tmp/mert_embed.py. Heavyweight (transformers + GPU) ->
    not exercised by the unit tests; constructed lazily on first ``embed``.
    """

    def __init__(
        self,
        layers_mid: Sequence[int] = LAYERS_MID,
        layers_upper: Sequence[int] = LAYERS_UPPER,
        device: str = "cuda",
        model_id: str = MERT_ID,
    ) -> None:
        self.layers_mid = tuple(layers_mid)
        self.layers_upper = tuple(layers_upper)
        self.device = device
        self.model_id = model_id
        self._proc = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (local import keeps the module import torch-free)
        from transformers import AutoModel, AutoProcessor

        self._proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = (
            AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def embed(self, wav: np.ndarray, sr: int) -> dict:
        """Resample to 24 kHz mono, run MERT, mean-pool over time, average the mid and
        upper layer groups separately, L2-normalize each. Returns {'mid','upper'}."""
        import torch
        import torchaudio

        self._ensure_loaded()
        x = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
        if x.ndim == 2:  # (C, T) or (T, C) -> mono
            x = x.mean(0) if x.shape[0] <= x.shape[1] else x.mean(1)
        if sr != MERT_SR:
            x = torchaudio.functional.resample(x, sr, MERT_SR)
        inp = self._proc(x.numpy(), sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._model(
                input_values=inp["input_values"].to(self.device),
                output_hidden_states=True,
            )
        hs = out.hidden_states  # tuple[(1, T', D)]

        def pool(layers: Sequence[int]) -> np.ndarray:
            stacked = torch.stack([hs[i].mean(dim=1).squeeze(0) for i in layers], 0)
            v = stacked.mean(0).cpu().float().numpy()
            n = np.linalg.norm(v)
            return v / n if n > 1e-8 else v

        return {"mid": pool(self.layers_mid), "upper": pool(self.layers_upper)}


def audiobox_ce(path: str) -> float:
    """Audiobox Content Enjoyment for a wav path (mir venv; needs mir/src on PYTHONPATH).
    Importable but model-heavy -> not exercised by the unit tests."""
    from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics

    return float(analyze_audiobox_aesthetics(path)["content_enjoyment"])


# --------------------------------------------------------------------------- #
# Top-level scoring API
# --------------------------------------------------------------------------- #
# A CE scorer maps a wav-like to a CE float; an embedder maps (wav, sr) -> {mid,upper}.
CEFn = Callable[[WavLike], float]


def _default_embedder() -> MERTEmbedder:
    global _EMBEDDER_SINGLETON
    try:
        return _EMBEDDER_SINGLETON  # type: ignore[name-defined]
    except NameError:
        _EMBEDDER_SINGLETON = MERTEmbedder()  # noqa: F841
        return _EMBEDDER_SINGLETON


def _load_wav(wav: WavLike, target_sr: int = MERT_SR) -> tuple[np.ndarray, int]:
    """Resolve a WavLike to (mono float ndarray, sr). Arrays are returned as-is with
    the caller-declared sr; paths are loaded via torchaudio."""
    if isinstance(wav, np.ndarray):
        return wav, target_sr
    import torchaudio

    x, sr = torchaudio.load(wav)
    return x.float().mean(0).numpy(), int(sr)


def score_continuation(
    prev_wav: Optional[WavLike],
    cand_wav: WavLike,
    *,
    sr: int = MERT_SR,
    embedder: Optional[MERTEmbedder] = None,
    ce_fn: Optional[CEFn] = None,
    weights: Optional[RewardWeights] = None,
) -> dict:
    """Score a single candidate continuation against a reference (previous) window.

    Returns ``{"ce", "rhythm_sim", "melody_sim", "score"}``.

    ``prev_wav`` may be None (first window) -> CE-only ranking (see
    ``score_from_embeddings``). ``embedder`` and ``ce_fn`` are injectable so the
    scoring math can be unit-tested with synthetic embeddings/CE and no model. By
    default they use the real MERT model + Audiobox (requires the mir venv + GPU).
    """
    weights = weights or RewardWeights()
    embedder = embedder or _default_embedder()
    ce_fn = ce_fn or audiobox_ce

    ce = float(ce_fn(cand_wav))

    cw, cw_sr = _load_wav(cand_wav, sr)
    cand_emb = embedder.embed(cw, cw_sr)

    if prev_wav is None:
        prev_mid = prev_upper = None
    else:
        pw, pw_sr = _load_wav(prev_wav, sr)
        prev_emb = embedder.embed(pw, pw_sr)
        prev_mid, prev_upper = prev_emb["mid"], prev_emb["upper"]

    return score_from_embeddings(
        prev_mid,
        prev_upper,
        cand_emb["mid"],
        cand_emb["upper"],
        ce,
        weights=weights,
    )


def best_of(
    prev_wav: Optional[WavLike],
    cand_wavs: Sequence[WavLike],
    *,
    sr: int = MERT_SR,
    embedder: Optional[MERTEmbedder] = None,
    ce_fn: Optional[CEFn] = None,
    weights: Optional[RewardWeights] = None,
    return_scores: bool = False,
):
    """Score every candidate against ``prev_wav`` and return the argmax index.

    The reference is embedded **once** and reused across candidates. With
    ``return_scores=True`` returns ``(best_index, scores)`` where ``scores`` is the
    list of per-candidate dicts (aligned to ``cand_wavs``)."""
    if not cand_wavs:
        raise ValueError("best_of: empty candidate list")
    weights = weights or RewardWeights()
    embedder = embedder or _default_embedder()
    ce_fn = ce_fn or audiobox_ce

    if prev_wav is None:
        prev_mid = prev_upper = None
    else:
        pw, pw_sr = _load_wav(prev_wav, sr)
        prev_emb = embedder.embed(pw, pw_sr)
        prev_mid, prev_upper = prev_emb["mid"], prev_emb["upper"]

    scores: list[dict] = []
    for cand in cand_wavs:
        ce = float(ce_fn(cand))
        cw, cw_sr = _load_wav(cand, sr)
        cand_emb = embedder.embed(cw, cw_sr)
        s = score_from_embeddings(
            prev_mid, prev_upper, cand_emb["mid"], cand_emb["upper"], ce, weights=weights
        )
        s = {"path": cand if isinstance(cand, str) else None, **s}
        scores.append(s)

    bi = best_index(scores)
    return (bi, scores) if return_scores else bi


# --------------------------------------------------------------------------- #
# CLI: cross-venv best-of-N seam (job.json -> scores.json)
# --------------------------------------------------------------------------- #
def score_candidates_from_job(job: dict) -> list[dict]:
    """Run the real MERT + Audiobox scorer described by a job dict (see Part 4 schema).

    job: {"candidates": [path...], "reference": path|null, "sample_rate": int,
          "weights": {...}, "mert_layers_mid": [...], "mert_layers_upper": [...]}
    Returns a list of score dicts aligned to ``candidates`` (each with a "path").
    """
    weights = RewardWeights(**{k: v for k, v in (job.get("weights") or {}).items()
                               if k in RewardWeights.__dataclass_fields__})
    embedder = MERTEmbedder(
        layers_mid=tuple(job.get("mert_layers_mid", LAYERS_MID)),
        layers_upper=tuple(job.get("mert_layers_upper", LAYERS_UPPER)),
    )
    _, scores = best_of(
        job.get("reference"),
        list(job["candidates"]),
        embedder=embedder,
        weights=weights,
        return_scores=True,
    )
    return scores


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m sa3_control.mert_selector job.json scores.json", file=sys.stderr)
        return 2
    job_path, out_path = argv
    with open(job_path) as f:
        job = json.load(f)
    scores = score_candidates_from_job(job)
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
