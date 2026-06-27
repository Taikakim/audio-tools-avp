"""Pure reward math for the MERT continuation re-ranker (Part 5).

This is the **single source of truth** for the composite reward used by the
Part-4 best-of-N selector (``mert_score.composite_reward``) AND the Part-5
trainer below — so the learned model reproduces the selector exactly, with no
divergence between scorer and target.

Reward (development without losing identity)::

    reward = w_ce     * CE                              # Audiobox content_enjoyment, want HIGH
           + w_rhythm  * rhythm_sim                     # MERT MID cosine vs prev window, want HIGH
           - w_melody  * |melody_sim - band_center|     # MERT UPPER cosine, want a BAND not a max

``melody_sim`` is rewarded toward a *band* (centred on ``band_center``), not a
maximum: below the band the continuation is unrelated/incoherent, above it the
continuation is a literal repeat (the loop). The penalty is the absolute
deviation from the band centre, so it is zero at the centre and grows linearly
either side.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RewardSpec", "composite_reward", "composite_reward_spec"]


@dataclass(frozen=True)
class RewardSpec:
    """Reward specification: component weights + the melody-similarity band.

    Mirrors the Part-4 ``RewardWeights`` (``bestof_selector.RewardWeights``) so a
    selector log can be replayed through the trainer with identical weights.

    - ``w_ce`` / ``w_rhythm`` / ``w_melody`` — component weights.
    - ``band_center`` — midpoint of the rewarded melody-sim band.
    - ``band_lo`` / ``band_hi`` — the band edges (informational: used for
      calibration / clamping; ``band_center`` is what enters the penalty). By
      default ``band_center`` is the midpoint of ``[band_lo, band_hi]``.
    """

    w_ce: float = 1.0
    w_rhythm: float = 1.0
    w_melody: float = 1.0
    band_center: float = 0.675  # midpoint of [0.55, 0.80]
    band_lo: float = 0.55
    band_hi: float = 0.80

    @property
    def band_width(self) -> float:
        """Full width of the melody-sim band (``band_hi - band_lo``)."""
        return self.band_hi - self.band_lo

    @classmethod
    def from_band(
        cls,
        band_lo: float,
        band_hi: float,
        *,
        w_ce: float = 1.0,
        w_rhythm: float = 1.0,
        w_melody: float = 1.0,
    ) -> "RewardSpec":
        """Build a spec from band edges, deriving ``band_center`` as the midpoint."""
        return cls(
            w_ce=w_ce,
            w_rhythm=w_rhythm,
            w_melody=w_melody,
            band_center=0.5 * (band_lo + band_hi),
            band_lo=band_lo,
            band_hi=band_hi,
        )


def composite_reward(
    ce: float,
    rhythm_sim: float,
    melody_sim: float,
    *,
    w_ce: float,
    w_rhythm: float,
    w_melody: float,
    band_center: float,
) -> float:
    """Composite continuation reward (pure scalar).

    Single source of truth shared by the Part-4 selector and the Part-5 trainer.
    See module docstring for the formula. Identical to
    ``mert_score.composite_reward`` (Part 4) by construction.
    """
    return (
        w_ce * ce
        + w_rhythm * rhythm_sim
        - w_melody * abs(melody_sim - band_center)
    )


def composite_reward_spec(
    ce: float,
    rhythm_sim: float,
    melody_sim: float,
    spec: RewardSpec,
) -> float:
    """Convenience wrapper: :func:`composite_reward` driven by a :class:`RewardSpec`."""
    return composite_reward(
        ce,
        rhythm_sim,
        melody_sim,
        w_ce=spec.w_ce,
        w_rhythm=spec.w_rhythm,
        w_melody=spec.w_melody,
        band_center=spec.band_center,
    )
