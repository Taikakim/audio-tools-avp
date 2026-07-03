"""Sonar — radial loss probes along the applied update ("throw rocks and listen").

Forward-only trust-scaling for any optimizer (designed for FusionOpt): every `every`
steps, capture params before the step, and after it evaluate the loss at a small ladder
of multiples of the APPLIED update Δθ (the post-momentum/post-orthogonalization step —
probing the raw gradient measures the wrong direction for Muon-family updates). Fit a
parabola over (scale, loss), derive the suggested step multiple, and fold it into a
log-space EMA `scale` that the trainer writes to `optimizer.gamma_scale`.

Design per the 2026-07 research pass (docs/superpowers/specs/
2026-07-02-perceptual-signal-optimizer-directions.md #3):
- SALSA (arXiv:2407.20650): smooth the decision (EMA), conservative growth, cadence
  amortization (overhead ~3% at every<=10; we default every=25).
- Critical sharpness (arXiv:2601.16979): <10 forward passes along Δθ; the ladder is 3
  extra forwards here.
- Distance-Aware Muon (arXiv:2605.18999): trust radius on the update scale, not the LR
  schedule.
- Probes MUST reuse the same batch + RNG (the trainer's loss closure must be
  deterministic for the current batch) and restore params bit-exactly.
- Schedule-Free: probes displace the live (fast/eval-point) params — that is the right
  iterate to probe (never the averaged x). Restore is bit-exact via saved clones.
- Per-event scale change clamped to [0.5, 2]; absolute scale clamped to [0.1, 10].

Usage in a train loop:
    sonar = SonarProbe(every=25)
    ...
    sonar.before_step(params)          # cheap no-op except on probe steps
    loss.backward(); opt.step()
    if sonar.after_step(params, loss_fn):   # loss_fn: () -> loss on the SAME batch, no_grad-safe
        opt.gamma_scale = sonar.scale
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch


class SonarProbe:
    def __init__(self, every: int = 25, ladder: Sequence[float] = (0.5, 2.0, 4.0),
                 ema_beta: float = 0.9, event_clamp: tuple = (0.5, 2.0),
                 scale_clamp: tuple = (0.1, 10.0)):
        self.every = max(1, int(every))
        self.ladder = tuple(float(x) for x in ladder)   # probed step-multiples besides 1.0
        self.ema_beta = ema_beta
        self.event_clamp = event_clamp
        self.scale_clamp = scale_clamp
        self.scale = 1.0                 # the trust multiple (trainer copies to gamma_scale)
        self.last_suggestion = None      # raw parabola suggestion, for telemetry
        self.last_profile = None         # [(multiple, loss)] of the last probe
        self._step = 0
        self._saved = None               # params snapshot on probe steps

    # -- trainer hooks -----------------------------------------------------

    def before_step(self, params: Iterable[torch.nn.Parameter]) -> None:
        self._step += 1
        if self._step % self.every == 0:
            self._saved = [p.detach().clone() for p in params]
        else:
            self._saved = None

    @torch.no_grad()
    def after_step(self, params: Iterable[torch.nn.Parameter],
                   loss_fn: Callable[[], torch.Tensor]) -> bool:
        """Probe (if scheduled). Returns True when `scale` was updated this call."""
        if self._saved is None:
            return False
        params = list(params)
        theta1 = [p.detach().clone() for p in params]           # post-step point
        delta = [t1 - t0 for t1, t0 in zip(theta1, self._saved)]
        if sum(float(d.abs().sum()) for d in delta) == 0.0:
            self._saved = None
            return False

        # loss profile along the applied update: multiples 1.0 (already there) + ladder.
        prof = [(1.0, float(loss_fn().float()))]                # fp32 accumulation
        for m in self.ladder:
            for p, t0, d in zip(params, self._saved, delta):
                p.data.copy_(t0 + m * d)
            prof.append((float(m), float(loss_fn().float())))
        for p, t1 in zip(params, theta1):                       # bit-exact restore
            p.data.copy_(t1)
        self._saved = None
        self.last_profile = sorted(prof)

        suggestion = self._parabola_argmin(self.last_profile)
        self.last_suggestion = suggestion
        lo, hi = self.event_clamp
        event = min(max(suggestion, lo), hi)
        # log-space EMA toward the (clamped) suggestion — SALSA-style smoothed decision
        new = math.exp(self.ema_beta * math.log(self.scale)
                       + (1 - self.ema_beta) * math.log(self.scale * event))
        slo, shi = self.scale_clamp
        self.scale = min(max(new, slo), shi)
        return True

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _parabola_argmin(profile) -> float:
        """Fit L(m) = a m^2 + b m + c; return argmin/1.0 as the suggested multiple.

        Guards (Edge-of-Stability bumps break naive fits — arXiv:2601.16979):
        - non-convex fit (a <= 0): if the far end still descends, suggest the max probed
          multiple (go further); else suggest the best probed point.
        - suggestion capped to the probed range (never extrapolate past the ladder).
        """
        import numpy as np
        m = np.array([p[0] for p in profile]); L = np.array([p[1] for p in profile])
        A = np.stack([m ** 2, m, np.ones_like(m)], axis=1)
        (a, b, _), *_ = np.linalg.lstsq(A, L, rcond=None)
        m_best = float(m[np.argmin(L)])
        if a <= 1e-12:
            return m_best
        m_star = float(-b / (2 * a))
        return min(max(m_star, float(m.min())), float(m.max()))
