"""Part 1 — Continuation modes for the steered long-form renderer.

A thin layer over ``stable_audio_3.inference.longform`` that selects the
prefix/stitch behaviour by ``--continuation {clamp,sdedit,crossfade}`` WITHOUT
editing ``longform.py`` (subclass + wrap only — MASTER §5 hard invariant).

- **clamp**     — current behaviour. ``prefix_frames=overlap``, the inner
  ``InpaintContinuationGenerator`` clamps the prefix; stitched with
  ``continuation_join`` (slerp ``blend_frames``). Coherent, loopy. Regression-safe
  default: ``render_latents`` delegates to the unchanged base loop, so a
  ``clamp`` run byte-matches today's output.
- **sdedit**    — decaying audio-init: seed each window FROM the previous tail and
  re-noise→denoise at a decaying ``init_noise_level`` (high early → low late) so
  early windows develop freely and later windows stay anchored. Same prefix/stitch
  decisions as clamp (anchoring comes from the init, not a clamp), driven by
  ``SDEditContinuationGenerator`` instead of ``InpaintContinuationGenerator``.
- **crossfade** — fresh seed per window (``prefix_frames=0``) + slerp the overlap
  (``transition_join`` over ``crossfade_overlap_frac * overlap`` frames) EVERY
  window. Maximum development; coherence comes only from the slerp.

Public interface:
    ContinuationMode = Literal["clamp", "sdedit", "crossfade"]
    SDEditContinuationGenerator(model, *, steps, cfg_scale,
                                init_noise_start, init_noise_end, total_windows)
    DevelopmentRenderer(LongFormRenderer)(generator, channels, fps,
        window_frames, overlap_frames, *, continuation_mode,
        crossfade_overlap_frac, blend_frames, stitcher, monitor)

GPU note: the *quality* of sdedit/crossfade needs a launch session; the stitching
and seed/noise bookkeeping here are model-free and unit-tested with
``longform.FakeChunkGenerator``.
"""
from __future__ import annotations

import math
import warnings
from typing import Literal, Optional

import torch

from stable_audio_3.inference.longform import (
    LongFormRenderer,
    ChunkGenerator,
    InpaintContinuationGenerator,
    SDEditReanchor,
    CrossfadeStitcher,  # re-exported for callers/tests; used implicitly via the renderer
)

ContinuationMode = Literal["clamp", "sdedit", "crossfade"]

__all__ = [
    "ContinuationMode",
    "SDEditContinuationGenerator",
    "DevelopmentRenderer",
]


class SDEditContinuationGenerator(ChunkGenerator):
    """sdedit-mode generator: init each window FROM the prefix and denoise at a
    decaying ``init_noise_level``.

    Window 0 (``prefix_frames == 0`` / no prefix) falls back to a fresh clamp
    generate (a pure ``InpaintContinuationGenerator`` window). Subsequent windows
    build ``init_data`` by tiling/continuing the prefix to ``n_frames`` and run
    ``SDEditReanchor.reanchor`` at ``sigma_peak = init_noise_level(window_index)``::

        init_noise_level(k) = init_noise_end
                              + (init_noise_start - init_noise_end)
                                * max(0, 1 - k / max(1, total_windows - 1))

    The model-dependent halves (the clamp generator + the reanchor) are built
    lazily on first ``generate`` so the pure bookkeeping (``init_noise_level``,
    ``_build_init_data``) is unit-testable without a model/GPU. Tests may inject
    fakes by assigning ``.clamp`` / ``.reanchor`` before the first call.
    """

    def __init__(self, model, *, steps: int = 50, cfg_scale: float = 6.0,
                 init_noise_start: float = 0.85, init_noise_end: float = 0.55,
                 total_windows: Optional[int] = None) -> None:
        self.model = model
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.init_noise_start = float(init_noise_start)
        self.init_noise_end = float(init_noise_end)
        self.total_windows = total_windows
        # built lazily (or injected by tests)
        self.clamp: Optional[ChunkGenerator] = None
        self.reanchor: Optional[SDEditReanchor] = None
        self._call = 0
        # diagnostics: (window_index, sigma_peak or None for fresh window)
        self.sigma_log: list[tuple[int, Optional[float]]] = []

    def set_total_windows(self, n: int) -> None:
        """Renderer calls this once it knows the window count (drives the decay)."""
        self.total_windows = int(n)

    def init_noise_level(self, k: int) -> float:
        """Decaying re-noise level for window ``k`` (0-indexed)."""
        tw = self.total_windows
        denom = max(1, (int(tw) - 1)) if tw else 1
        frac = max(0.0, 1.0 - (k / denom))
        return self.init_noise_end + (self.init_noise_start - self.init_noise_end) * frac

    def _build_init_data(self, prefix_latents: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Tile/continue ``prefix_latents`` (1, C, prefix_frames) to (1, C, n_frames)."""
        pf = prefix_latents.shape[-1]
        if pf >= n_frames:
            return prefix_latents[..., :n_frames]
        reps = math.ceil(n_frames / pf)
        tiled = prefix_latents.repeat(1, 1, reps)
        return tiled[..., :n_frames]

    def _ensure_built(self) -> None:
        if self.clamp is None:
            self.clamp = InpaintContinuationGenerator(
                self.model, steps=self.steps, cfg_scale=self.cfg_scale)
        if self.reanchor is None:
            self.reanchor = SDEditReanchor(
                self.model, steps=self.steps, cfg_scale=self.cfg_scale)

    def generate(self, prompt: str, prefix_latents: Optional[torch.Tensor],
                 prefix_frames: int, n_frames: int, seed: int) -> torch.Tensor:
        k = self._call
        self._call += 1
        self._ensure_built()
        if prefix_latents is None or prefix_frames <= 0:
            self.sigma_log.append((k, None))
            return self.clamp.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
        init_data = self._build_init_data(prefix_latents, n_frames)
        sigma = self.init_noise_level(k)
        self.sigma_log.append((k, sigma))
        out = self.reanchor.reanchor(init_data, sigma_peak=sigma, prompt=prompt, seed=seed)
        return out.float()


class DevelopmentRenderer(LongFormRenderer):
    """``LongFormRenderer`` subclass that selects prefix/stitch behaviour by mode.

    clamp / sdedit -> identical prefix/stitch to the base loop (delegates to
        ``super().render_latents`` so clamp is byte-for-byte regression-safe);
        the only addition is a ``set_total_windows`` call if the generator exposes it.
    crossfade -> ``prefix_frames`` forced to 0 (fresh seed each window); stitch via
        ``transition_join`` over ``round(crossfade_overlap_frac * overlap)`` frames
        every window. Keeps the base drift_log + non-finite retry semantics.
    """

    def __init__(self, generator, channels, fps, window_frames, overlap_frames, *,
                 continuation_mode: ContinuationMode = "clamp",
                 crossfade_overlap_frac: float = 1.0,
                 blend_frames: int = 3, stitcher=None, monitor=None) -> None:
        super().__init__(generator, channels, fps, window_frames, overlap_frames,
                         blend_frames=blend_frames, stitcher=stitcher, monitor=monitor)
        if continuation_mode not in ("clamp", "sdedit", "crossfade"):
            raise ValueError(
                f"continuation_mode must be clamp|sdedit|crossfade, got {continuation_mode!r}")
        self.continuation_mode: ContinuationMode = continuation_mode
        self.crossfade_overlap_frac = float(crossfade_overlap_frac)

    def estimate_total_windows(self, total_frames: int) -> int:
        """ceil((total_frames - window) / (window - overlap)) + 1 (clamp/sdedit advance).

        Used to drive the sdedit decay; an estimate (the last window is trimmed),
        not an exact frame budget.
        """
        step = self.window - self.overlap
        return math.ceil(max(0, int(total_frames) - self.window) / step) + 1

    def render_latents(self, schedule, total_frames, base_seed: int = 0):
        if hasattr(self.gen, "set_total_windows"):
            self.gen.set_total_windows(self.estimate_total_windows(total_frames))
        if self.continuation_mode in ("clamp", "sdedit"):
            # Identical prefix/stitch decisions to the base loop -> delegate unchanged.
            return super().render_latents(schedule, total_frames, base_seed)
        return self._render_crossfade(schedule, total_frames, base_seed)

    # -- crossfade: fresh seed each window + slerp join every window --------------
    def _render_crossfade(self, schedule, total_frames, base_seed: int):
        n_join = max(1, min(int(round(self.crossfade_overlap_frac * self.overlap)),
                            self.overlap))
        out = None
        k = 0
        while out is None or out.shape[-1] < total_frames:
            t_sec = (out.shape[-1] / self.fps) if out is not None else 0.0
            prompt, _is_transition, _xf_sec = schedule.resolve(t_sec)
            prev_tail = out[..., -self.overlap:] if out is not None else None
            chunk = self.gen.generate(
                prompt, prefix_latents=prev_tail, prefix_frames=0,
                n_frames=self.window, seed=base_seed + k)
            tries = 0
            while not torch.isfinite(chunk).all() and tries < 3:
                tries += 1
                chunk = self.gen.generate(
                    prompt, prefix_latents=prev_tail, prefix_frames=0,
                    n_frames=self.window, seed=base_seed + k + 1000 * tries)
            if not torch.isfinite(chunk).all():
                raise RuntimeError(
                    f"DevelopmentRenderer(crossfade): chunk {k} (prompt={prompt!r}) "
                    f"non-finite after {tries} retries")
            stats = self.monitor.observe(chunk)
            self.drift_log.append(stats)
            if self.monitor.should_reanchor(stats):
                warnings.warn(
                    f"DriftMonitor canary: RMS collapse at chunk {k} (t={t_sec:.1f}s, "
                    f"rms={stats['rms']:.4f})",
                    RuntimeWarning, stacklevel=2)
            if out is None:
                out = chunk
            else:
                n = min(n_join, out.shape[-1], chunk.shape[-1])
                joined = self.stitcher.transition_join(out, chunk, n)
                out = torch.cat([out[..., :-n], joined, chunk[..., n:]], dim=-1)
            k += 1
        return out[..., :total_frames]
