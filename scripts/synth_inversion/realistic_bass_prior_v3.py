"""Patch prior built from real Surge XT bass presets (extract_real_bass_manifold.py, format_version 2).

Each draw (v2, 2026-10-02 review):
  1. picks one preset of the requested split ('train' / 'val' / 'all') as the archetype;
  2. keeps its three categorical choices (filter circuit, unison, waveshaper) as they are;
  3. keeps every control that is exactly 0 in that preset at 0 (sync off, no sub, no drive, ...) on
     the dimensions where 0 is a real "off" state (>= 5 % of presets), so the on/off pattern of a
     preset survives — v1 re-rolled each on/off switch independently, which kept the marginal rates
     but discarded the joint structure that was the reason to sample from presets;
  4. perturbs the remaining continuous controls with Gaussian noise shaped like the presets'
     covariance (scaled by `jitter`), and REFLECTS the result into the presets' own range, so no
     probability piles up at a clip boundary (v1 clipped, creating spikes at 0.05/0.08/0.75);
  5. draws the MIDI note uniformly over E1..D3 and the note length from surge_spec.NOTE_DUR_RANGE.
The FX dimensions are always 0 (presets' FX are not projected).

support_bounds() gives the per-parameter box this prior can produce (training split), which the
trainer stores in its checkpoints so the renderer-free search stays inside it.
"""
import hashlib
import os

import numpy as np

from surge_spec_v3 import CATEGORICAL, NOTE_DUR_RANGE, NOTE_HIGH, NOTE_LOW, PARAM_INDEX, PARAM_NAMES, vector_to_patch

MANIFOLD_PATH = "/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz"
MANIFOLD_FORMAT = 2
SPIKE_MIN_ZERO_FRAC = 0.05
FX_NAMES = ("chorus_mix", "delay_mix", "delay_fb")
_FIXED = {"midi_note", *CATEGORICAL, *FX_NAMES}
JITTER_IDX = np.array([i for i, n in enumerate(PARAM_NAMES) if n not in _FIXED])


def manifold_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _reflect(x, lo, hi):
    w = hi - lo
    y = np.mod(x - lo, 2.0 * np.where(w > 0, w, 1.0))
    y = np.where(y > w, 2.0 * w - y, y)
    return np.where(w > 0, lo + y, lo)


class RealisticBassPrior:
    def __init__(self, manifold_path=MANIFOLD_PATH, split="train", jitter=0.08):
        if not os.path.exists(manifold_path):
            raise FileNotFoundError(f"Manifold file not found: {manifold_path} (run extract_real_bass_manifold.py)")
        data = np.load(manifold_path, allow_pickle=False)
        version = int(data["format_version"]) if "format_version" in data.files else 1
        if version < MANIFOLD_FORMAT:
            raise ValueError(f"{manifold_path} is a v{version} manifold, built with guessed unit conversions "
                             "(see extract_real_bass_manifold.py docstring); rebuild it with the v2 extractor.")
        if list(data["param_names"]) != list(PARAM_NAMES):
            raise ValueError(f"{manifold_path}: parameter layout differs from surge_spec.PARAM_NAMES")
        all_params = data["params"].astype(np.float64)
        is_val = data["is_val"].astype(bool)
        train = all_params[~is_val]
        self.calibrated = bool(data["calibrated"])
        self.path = manifold_path
        self.split = split
        self.params = {"train": train, "val": all_params[is_val], "all": all_params}[split]
        if len(self.params) == 0:
            raise ValueError(f"{manifold_path}: split '{split}' is empty")
        self.N = len(self.params)
        self.jitter = float(jitter)

        # Statistics always from the TRAINING split, so a val draw cannot leak into the box/noise shape
        j = JITTER_IDX
        self.spike = np.mean(train[:, j] == 0.0, axis=0) >= SPIKE_MIN_ZERO_FRAC
        on = np.where(train[:, j] > 0.0, train[:, j], np.nan)
        lo_all, hi = train[:, j].min(axis=0), train[:, j].max(axis=0)
        lo_on = np.nanmin(np.where(np.isnan(on), np.inf, on), axis=0)
        lo_on = np.where(np.isfinite(lo_on), lo_on, lo_all)
        # "on" values of a spike dimension stay strictly above 0 by reflecting into [min_on, max]
        self.lo = np.where(self.spike, lo_on, lo_all)
        self.hi = hi
        cov = np.cov(train[:, j], rowvar=False) if len(train) > 1 else np.zeros((len(j), len(j)))
        w, V = np.linalg.eigh(np.atleast_2d(cov))
        self._noise = V * np.sqrt(np.clip(w, 0.0, None))  # noise = _noise @ z, z ~ N(0, I)

    def support_bounds(self) -> dict:
        """{continuous parameter name: (lo, hi)} in vector units that this prior can produce."""
        b = {n: (0.0, 0.0) for n in FX_NAMES}
        b["midi_note"] = (0.0, 1.0)
        for k, i in enumerate(JITTER_IDX):
            lo = 0.0 if self.spike[k] else float(self.lo[k])
            b[PARAM_NAMES[i]] = (min(lo, float(self.lo[k])), float(self.hi[k]))
        return b

    def sample_vector(self, rng=None) -> np.ndarray:
        rng = np.random if rng is None else rng
        base = self.params[rng.randint(0, self.N)].copy()
        x = base[JITTER_IDX]
        noisy = x + np.sqrt(self.jitter) * (self._noise @ rng.standard_normal(len(JITTER_IDX)))
        noisy = _reflect(noisy, self.lo, self.hi)
        off = self.spike & (x == 0.0)
        base[JITTER_IDX] = np.where(off, 0.0, noisy)
        for n in FX_NAMES:
            base[PARAM_INDEX[n]] = 0.0
        if base[PARAM_INDEX["unison"]] < 0.5:
            base[PARAM_INDEX["unison_detune"]] = 0.0
        if base[PARAM_INDEX["waveshaper_type"]] < 0.5 / (CATEGORICAL["waveshaper_type"] - 1):
            base[PARAM_INDEX["drive"]] = 0.0
        midi_note = rng.randint(NOTE_LOW, NOTE_HIGH + 1)
        base[PARAM_INDEX["midi_note"]] = (midi_note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW)
        return np.clip(base, 0.0, 1.0).astype(np.float32)

    def sample_patch_and_midi(self, rng=None):
        """-> (patch dict, 23-d vector, midi_note, note_dur)."""
        rng = np.random if rng is None else rng
        vec = self.sample_vector(rng)
        patch = vector_to_patch(vec)
        note_dur = float(rng.uniform(*NOTE_DUR_RANGE))
        patch["note_dur"] = note_dur
        return patch, vec, patch["midi_note"], note_dur


_priors = {}


def get_prior(split="train", manifold_path=MANIFOLD_PATH):
    key = (split, manifold_path)
    if key not in _priors:
        _priors[key] = RealisticBassPrior(manifold_path, split=split)
    return _priors[key]


def draw_realistic_patch(seed: int = None, split: str = "train", manifold_path: str = MANIFOLD_PATH):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    return get_prior(split, manifold_path).sample_patch_and_midi(rng)


if __name__ == "__main__":
    prior = get_prior()
    p, v, note, dur = draw_realistic_patch(42)
    print(f"{prior.N} training presets (calibrated={prior.calibrated}); MIDI {note}, {dur:.2f} s")
    print({k: (round(a, 3), round(b, 3)) for k, (a, b) in prior.support_bounds().items()})
