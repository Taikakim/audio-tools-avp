# Owner: KUANG (created 2026-10-11)
"""Inference-time scoring of rendered candidates with the v4 measures (Stage 1 of the plan; review B1).

Glue only: `inference_v3.predict_and_rerank_candidates` renders flow candidates and scores them; this adds the v4
components as extra, separately visible terms. It never trains anything and never needs a gradient.

  candidate_scores   raw components for one rendered candidate against the target audio
  rank_candidates    order candidates by a weighted sum of the components' absolute values

Weights are NOT tuned. The default uses the two-anchor score alone. A composite of timbre, resonance and buzz terms needs
evidence first (probe tables, then a blind A/B on the 24 stems); choose the weights from that, not from this file. Use
`explain=True` to see every component for every candidate and judge them yourself.

Components (all in the units of their module):
  two_anchor         objectives_v4.TwoAnchorObjective (dB, onset-aligned attack + sustain + global)
  harmonic_env       objectives_v4.harmonic_envelope_distance (dB at the harmonics)
  height_delta_db    |resonance height of the candidate - of the target| (resonance_v4); NaN if neither has a peak
  centre_delta_oct   |log2(candidate centre / target centre)| when both have a peak
  comb_delta         |comb residual of the candidate - of the target| (buzz_v4)
  q_censored         True when either resonance is narrower than the harmonic spacing (height/centre still valid)
"""
import numpy as np

from buzz_v4 import comb_residual
from objectives_v4 import TwoAnchorObjective, harmonic_envelope_distance
from resonance_v4 import resonance_card

_DEFAULT = TwoAnchorObjective()


def candidate_scores(cand, target, f0: float, sr: int = 44100, expected_onset_s: float = 0.0, cand_onset_s: float = 0.0,
                     objective: TwoAnchorObjective = None) -> dict:
    obj = objective or _DEFAULT
    out = {"two_anchor": float(obj(cand, target, expected_onset_s, cand_onset_s)),
           "harmonic_env": float(harmonic_envelope_distance(cand, target, f0, sr))}
    rc, rt = resonance_card(cand, sr, f0), resonance_card(target, sr, f0)
    if rc.has_peak or rt.has_peak:
        out["height_delta_db"] = abs(rc.height_db - rt.height_db)
    else:
        out["height_delta_db"] = float("nan")
    if rc.has_peak and rt.has_peak:
        out["centre_delta_oct"] = abs(float(np.log2(rc.peak_freq_hz / rt.peak_freq_hz)))
    else:
        out["centre_delta_oct"] = float("nan")
    out["comb_delta"] = abs(comb_residual(cand, sr, f0)["residual"] - comb_residual(target, sr, f0)["residual"])
    out["q_censored"] = bool(rc.q_censored or rt.q_censored)
    return out


def rank_candidates(cands, target, f0: float, weights: dict = None, explain: bool = False, **kw):
    """Returns the candidate indices, best first (and the per-candidate component dicts if explain=True). NaN components
    count as 0, so a term with no resonance peak does not decide the order."""
    w = weights or {"two_anchor": 1.0}
    unknown = set(w) - {"two_anchor", "harmonic_env", "height_delta_db", "centre_delta_oct", "comb_delta"}
    if unknown:
        raise ValueError(f"unknown score terms: {sorted(unknown)}")
    rows = [candidate_scores(c, target, f0, **kw) for c in cands]
    totals = [sum(wt * (0.0 if not np.isfinite(r[k]) else r[k]) for k, wt in w.items()) for r in rows]
    order = [int(i) for i in np.argsort(totals, kind="stable")]
    return (order, rows) if explain else order
