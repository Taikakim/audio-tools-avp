"""Renderer-in-the-loop refinement of the perceptually salient axes after flow / JEPA proposals.

Why: the flow (and the JEPA search) return a patch from a parameter-space objective; on a single
note several parameter combinations trade off (cutoff vs FEG amount vs keytrack vs drive), so the
proposal is often close but not on the sound. Synth-JEPA (arXiv:2609.31024, Fig. 3) shows that a
rendered best-of-k pass helps a flow model most -- its posterior contains good solutions it does
not reliably sample. This goes one step further: starting from the best proposal, bracket each
salient axis with real Surge renders and keep what the AUDIO says is closer.

Search, per axis (Kim's bracketing idea, as a coordinate search):
  1. a coarse ladder of `n_coarse` renders across [x - w, x + w] (clipped to the plugin range);
  2. a fine ladder of `n_fine` renders between the best coarse point's two neighbours, i.e.
     inside the bracket that holds the optimum;
  3. accept the best point only if it beats the current objective.
Axes are visited in order of audible weight (cutoff first, then the filter and amp envelopes),
in `passes` sweeps with the bracket half-width shrinking each pass, because cutoff and FEG
interact.

Objective (lower is better), chosen by human-timbre agreement (timbre_alignment_check.py, 21
listening-test datasets): log-mel L1 on the model's own 0.8 s mel was best (mean Spearman 0.473),
paper MSS 0.470, our MultiScaleSTFT 0.437. We use
    J = mss_ref (multi-scale log-mel: the 100 ms scale resolves the low bass, the 10 ms scale the
        attack)  +  env_weight * (1 - rms envelope cosine)   [envelope term, ~0..0.3]
Both are level-invariant; renders and target are peak-normalised.

Cost: ~11 renders per axis per pass; 8 axes x 2 passes ~ 150 renders ~ 3-5 s CPU per note.
"""
import json

import numpy as np

import reference_metrics as rm
from surge_spec import LP_FILTERS, SAMPLE_RATE, WAVESHAPER_TYPES, render_patch

# (patch key, coarse half-width as a fraction of [0, 1]) -- order = audible weight
DEFAULT_AXES = [
    ("cutoff", 0.15),
    ("feg_amount", 0.15),
    ("feg_decay", 0.15),
    ("aeg_decay", 0.15),
    ("aeg_sustain", 0.20),
    ("feg_sustain", 0.20),
    ("resonance", 0.15),
    ("aeg_release", 0.15),
]


def patch_from_json(path_or_dict) -> dict:
    """Saved .json (filter_circuit / waveshaper names) -> the dict apply_patch() expects."""
    d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
    p = dict(d.get("patch", d))
    if "filter_idx" not in p:
        p["filter_idx"] = [n for n, _ in LP_FILTERS].index(p.pop("filter_circuit"))
    if "ws_idx" not in p:
        p["ws_idx"] = [n for n, _ in WAVESHAPER_TYPES].index(p.pop("waveshaper"))
    p.pop("filter_circuit", None)
    p.pop("waveshaper", None)
    return p


def objective(cand, target, env_weight=10.0):
    return rm.mss(target, cand) + env_weight * (1.0 - rm.rms_env_cos(target, cand))


class Refiner:
    """render_fn(patch) -> peak-normalised mono audio, compared against `target` (same length)."""

    def __init__(self, render_fn, target, env_weight=10.0, score_slice=slice(None)):
        self.render_fn = render_fn
        self.target = np.asarray(target, np.float32)
        self.target = self.target / (np.max(np.abs(self.target)) + 1e-7)
        self.env_weight = env_weight
        self.sl = score_slice           # e.g. only the first half of a phrase (hold the rest out)
        self.n_renders = 0
        self.trace = []

    def score(self, patch):
        self.n_renders += 1
        a = self.render_fn(patch)
        return objective(a[self.sl], self.target[self.sl], self.env_weight)

    def _ladder(self, patch, key, lo, hi, n):
        best = []
        for v in np.linspace(lo, hi, n):
            q = dict(patch, **{key: float(v)})
            best.append((self.score(q), float(v)))
        return best

    def refine(self, patch, axes=DEFAULT_AXES, passes=2, n_coarse=7, n_fine=4, shrink=0.4,
               cand_bounds=None, probe_out=0.0, min_span=0.03):
        """cand_bounds: {key: (lo, hi)} -- first-pass search range per axis from the spread of the model's
        top candidates (bounds_from_candidates) instead of the fixed half-width around the start value.
        probe_out: also probe this fraction of the range beyond each end, in case the candidates share
        a bias (both ideas from bracket_refiner.py, 2026-10-06). Later passes shrink around the current
        value as before."""
        patch = dict(patch)
        cur = self.score(patch)
        self.trace.append(("start", None, cur))
        for ps in range(passes):
            for key, half in axes:
                x = patch[key]
                if ps == 0 and cand_bounds and key in cand_bounds:
                    lo, hi = cand_bounds[key]
                    lo, hi = min(lo, x), max(hi, x)
                    if hi - lo < min_span:
                        c = 0.5 * (lo + hi)
                        lo, hi = c - min_span / 2, c + min_span / 2
                    lo, hi = max(0.0, lo), min(1.0, hi)
                else:
                    w = half * (shrink ** ps)
                    lo, hi = max(0.0, x - w), min(1.0, x + w)
                coarse = self._ladder(patch, key, lo, hi, n_coarse)
                if ps == 0 and probe_out > 0:
                    span = hi - lo
                    for v in (lo - probe_out * span, hi + probe_out * span):
                        if 0.0 <= v <= 1.0:
                            coarse += [(self.score(dict(patch, **{key: float(v)})), float(v))]
                    coarse.sort(key=lambda sv: sv[1])
                vals = [v for _, v in coarse]
                i = int(np.argmin([s for s, _ in coarse]))
                # bracket that holds the optimum: the neighbours of the best coarse point
                flo, fhi = vals[max(0, i - 1)], vals[min(len(vals) - 1, i + 1)]
                fine = self._ladder(patch, key, flo, fhi, n_fine + 2)[1:-1] if fhi > flo else []
                s_best, v_best = min(coarse + fine)
                if s_best < cur - 1e-6:
                    self.trace.append((key, (x, v_best), s_best))
                    patch[key], cur = v_best, s_best
        return patch, cur


def bounds_from_candidates(candidates, axes=DEFAULT_AXES, top_k=20):
    """{key: (min, max)} over the model's top-k candidate patches (sorted best first)."""
    top = candidates[:top_k]
    return {k: (min(c[k] for c in top), max(c[k] for c in top)) for k, _ in axes}


def refine_patch(synth, patch, target, midi_note, note_dur, duration=0.8, **kw):
    """Single-note mode. -> (refined patch, final objective, start objective, n_renders, trace)."""
    r = Refiner(lambda p: render_patch(synth, p, int(midi_note), float(note_dur), duration=duration),
                target, env_weight=kw.pop("env_weight", 10.0))
    out, final = r.refine(patch, **kw)
    return out, final, r.trace[0][2], r.n_renders, r.trace


def refine_patch_phrase(render_phrase_fn, patch, target_phrase, fit_fraction=0.5, **kw):
    """Phrase mode: score only the first `fit_fraction` of the phrase (real note lengths and
    overlaps, several notes), leaving the rest as a held-out check. Same return as refine_patch."""
    n_fit = int(len(target_phrase) * fit_fraction)
    r = Refiner(render_phrase_fn, target_phrase, env_weight=kw.pop("env_weight", 10.0),
                score_slice=slice(0, n_fit))
    out, final = r.refine(patch, **kw)
    return out, final, r.trace[0][2], r.n_renders, r.trace
