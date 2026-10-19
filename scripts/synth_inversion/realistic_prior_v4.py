"""Soundset prior generator for v4: 30% bass manifold, 10% all-presets manifold, 60% pure-random.

Rules:
- 10% in all groups with at least some waveshaping, with >50% of waveshaping loud above 18 dB.
- Full unconstrained resonance [0.0, 1.0].
- 1 to 6 unison voices.
- Classic and Modern subtractive oscillator engines for Osc 1, 2, 3.
- Filter feedback across [-1.0, 1.0].
- Microtuning (+-1 st) and drift.
"""
import hashlib
import os
import numpy as np

from surge_spec_v4 import (
    CATEGORICAL, CONT_BOUNDS, CONT_INDICES, LP_FILTERS, NOTE_DUR_RANGE, NOTE_HIGH, NOTE_LOW,
    NUM_PARAMS, PARAM_INDEX, PARAM_NAMES, WAVESHAPER_TYPES, vector_to_patch
)

BASS_MANIFOLD_PATH = "/home/kim/sao_runs/real_manifold_v4_bass.npz"
ALL_MANIFOLD_PATH = "/home/kim/sao_runs/real_manifold_v4_all.npz"

FX_NAMES = ("chorus_mix", "delay_mix", "delay_fb")
_FIXED = {"midi_note", *CATEGORICAL, *FX_NAMES}
JITTER_IDX = np.array([i for i, n in enumerate(PARAM_NAMES) if n not in _FIXED])


def manifold_md5(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()



def _reflect(x, lo, hi):
    w = hi - lo
    y = np.mod(x - lo, 2.0 * np.where(w > 0, w, 1.0))
    y = np.where(y > w, 2.0 * w - y, y)
    return np.where(w > 0, lo + y, lo)


class SubManifoldPrior:
    """Manages sampling from a single manifold file (bass or all) with covariance jitter."""
    def __init__(self, path: str, split: str = "train", jitter: float = 0.08):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifold not found: {path}")
        data = np.load(path, allow_pickle=False)
        all_params = data["params"].astype(np.float64)
        is_val = data["is_val"].astype(bool)
        train = all_params[~is_val]
        self.params = {"train": train, "val": all_params[is_val], "all": all_params}[split]
        if len(self.params) == 0:
            raise ValueError(f"{path}: split '{split}' is empty")
        self.N = len(self.params)
        self.jitter = float(jitter)

        j = JITTER_IDX
        self.spike = np.mean(train[:, j] == 0.0, axis=0) >= 0.05
        on = np.where(train[:, j] > 0.0, train[:, j], np.nan)
        lo_all, hi = train[:, j].min(axis=0), train[:, j].max(axis=0)
        lo_on = np.nanmin(np.where(np.isnan(on), np.inf, on), axis=0)
        lo_on = np.where(np.isfinite(lo_on), lo_on, lo_all)
        self.lo = np.where(self.spike, lo_on, lo_all)
        self.hi = hi
        cov = np.cov(train[:, j], rowvar=False) if len(train) > 1 else np.zeros((len(j), len(j)))
        w, V = np.linalg.eigh(np.atleast_2d(cov))
        self._noise = V * np.sqrt(np.clip(w, 0.0, None))

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

        # Ensure FX stay zeroed
        for n in FX_NAMES:
            base[PARAM_INDEX[n]] = 0.0

        # Enforce 10% waveshaper rule across manifold draws if not already shaped
        if rng.rand() < 0.10:
            if base[PARAM_INDEX["waveshaper_type"]] < 0.5 / (len(WAVESHAPER_TYPES) - 1):
                ws_idx = int(rng.randint(1, len(WAVESHAPER_TYPES)))
                base[PARAM_INDEX["waveshaper_type"]] = ws_idx / (len(WAVESHAPER_TYPES) - 1)
            # Ensure >50% loud drive > 18 dB (raw 0.75..1.0)
            if rng.rand() < 0.55:
                base[PARAM_INDEX["drive"]] = float(rng.uniform(0.75, 1.0))
            else:
                base[PARAM_INDEX["drive"]] = float(rng.uniform(0.08, 0.75))
        else:
            base[PARAM_INDEX["waveshaper_type"]] = 0.0
            base[PARAM_INDEX["drive"]] = 0.0

        if base[PARAM_INDEX["unison"]] < (0.5 / 5.0):
            base[PARAM_INDEX["unison_detune"]] = 0.0

        midi_note = rng.randint(NOTE_LOW, NOTE_HIGH + 1)
        base[PARAM_INDEX["midi_note"]] = (midi_note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW)
        return np.clip(base, 0.0, 1.0).astype(np.float32), midi_note

    def sample_patch_and_midi(self, rng=None):
        rng = np.random if rng is None else rng
        vec, midi = self.sample_vector(rng)
        dur = float(rng.uniform(*NOTE_DUR_RANGE))
        return vector_to_patch(vec), vec, midi, dur


def sample_unconstrained_random_patch(rng=None):
    rng = np.random if rng is None else rng
    p = np.zeros(NUM_PARAMS, dtype=np.float32)

    midi = int(rng.randint(NOTE_LOW, NOTE_HIGH + 1))
    p[PARAM_INDEX["midi_note"]] = (midi - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW)

    # Both subtractive families active (Classic 0 vs Modern 1)
    p[PARAM_INDEX["osc_1_type"]] = float(rng.choice([0, 1]))
    p[PARAM_INDEX["osc_2_type"]] = float(rng.choice([0, 1]))
    p[PARAM_INDEX["osc_3_type"]] = float(rng.choice([0, 1]))

    p[PARAM_INDEX["shape"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["width"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["sub_mix"]] = float(rng.uniform(0.0, 0.85)) if rng.rand() < 0.6 else 0.0
    p[PARAM_INDEX["sync"]] = float(rng.uniform(0.0, 0.50)) if rng.rand() < 0.3 else 0.0
    p[PARAM_INDEX["fm_depth"]] = float(rng.uniform(0.0, 0.50)) if rng.rand() < 0.35 else 0.0

    p[PARAM_INDEX["osc_1_width_2"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["osc_2_shape"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["osc_2_width_1"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["osc_2_width_2"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["osc_3_shape"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["osc_3_width_1"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["osc_3_width_2"]] = float(rng.uniform(0.05, 0.95))

    p[PARAM_INDEX["osc_1_octave"]] = float(rng.uniform(0.25, 0.75))
    p[PARAM_INDEX["osc_2_octave"]] = float(rng.uniform(0.25, 0.75))
    p[PARAM_INDEX["osc_3_octave"]] = float(rng.uniform(0.25, 0.75))
    p[PARAM_INDEX["noise_color"]] = float(rng.uniform(0.0, 1.0))

    p[PARAM_INDEX["osc_1_volume"]] = float(rng.uniform(0.6, 1.0))
    p[PARAM_INDEX["osc_2_volume"]] = float(rng.uniform(0.0, 1.0)) if rng.rand() < 0.6 else 0.0
    p[PARAM_INDEX["osc_3_volume"]] = float(rng.uniform(0.0, 0.8)) if rng.rand() < 0.4 else 0.0
    p[PARAM_INDEX["ring_1x2_volume"]] = float(rng.uniform(0.0, 0.8)) if rng.rand() < 0.2 else 0.0
    p[PARAM_INDEX["ring_2x3_volume"]] = float(rng.uniform(0.0, 0.8)) if rng.rand() < 0.15 else 0.0
    p[PARAM_INDEX["noise_volume"]] = float(rng.uniform(0.0, 0.25)) if rng.rand() < 0.25 else 0.0

    # Unison: 1 to 6 voices
    if rng.rand() < 0.35:
        u_cls = int(rng.randint(1, 6)) # 2 to 6 voices
        p[PARAM_INDEX["unison"]] = u_cls / 5.0
        p[PARAM_INDEX["unison_detune"]] = float(rng.uniform(0.05, 0.40))
    else:
        p[PARAM_INDEX["unison"]] = 0.0
        p[PARAM_INDEX["unison_detune"]] = 0.0

    # Microtuning and drift
    p[PARAM_INDEX["osc_2_pitch"]] = float(rng.uniform(0.35, 0.65)) if rng.rand() < 0.5 else 0.5
    p[PARAM_INDEX["osc_3_pitch"]] = float(rng.uniform(0.35, 0.65)) if rng.rand() < 0.5 else 0.5
    p[PARAM_INDEX["drift"]] = float(rng.uniform(0.0, 0.5)) if rng.rand() < 0.4 else 0.0

    # Filters: full resonance allowed
    p[PARAM_INDEX["filter_type"]] = int(rng.randint(0, len(LP_FILTERS))) / (len(LP_FILTERS) - 1)
    p[PARAM_INDEX["cutoff"]] = float(rng.uniform(0.08, 0.95))
    p[PARAM_INDEX["resonance"]] = float(rng.uniform(0.0, 1.00))
    p[PARAM_INDEX["keytrack"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["filter_feedback"]] = float(rng.uniform(0.1, 0.9)) if rng.rand() < 0.35 else 0.5

    # Routing
    p[PARAM_INDEX["filter_configuration"]] = int(rng.randint(0, 7)) / 6.0
    p[PARAM_INDEX["fm_routing"]] = int(rng.randint(0, 4)) / 3.0
    p[PARAM_INDEX["filter_balance"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["filter_2_type"]] = int(rng.randint(0, len(LP_FILTERS))) / (len(LP_FILTERS) - 1)
    p[PARAM_INDEX["filter_2_cutoff"]] = float(rng.uniform(0.08, 1.0))
    p[PARAM_INDEX["filter_2_resonance"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["filter_2_keytrack"]] = float(rng.uniform(0.0, 1.0))
    p[PARAM_INDEX["filter_2_feg_amount"]] = float(rng.uniform(0.0, 1.0))

    # Envelopes
    p[PARAM_INDEX["feg_amount"]] = float(rng.uniform(0.1, 0.95))
    p[PARAM_INDEX["feg_decay"]] = float(rng.uniform(0.05, 0.95))
    p[PARAM_INDEX["feg_sustain"]] = float(rng.uniform(0.0, 0.65))
    p[PARAM_INDEX["aeg_decay"]] = float(rng.uniform(0.08, 0.70))
    p[PARAM_INDEX["aeg_sustain"]] = float(rng.uniform(0.10, 0.85))
    p[PARAM_INDEX["aeg_release"]] = float(rng.uniform(0.05, 0.40))
    p[PARAM_INDEX["feg_release"]] = float(rng.uniform(0.05, 0.50))

    p[PARAM_INDEX["amp_eg_mode"]] = float(rng.choice([0, 1]))
    p[PARAM_INDEX["filter_eg_mode"]] = float(rng.choice([0, 1]))
    p[PARAM_INDEX["amp_eg_decay_shape"]] = float(rng.choice([0, 1, 2])) / 2.0
    p[PARAM_INDEX["filter_eg_decay_shape"]] = float(rng.choice([0, 1, 2])) / 2.0

    # Waveshaping: 10% of ALL patches have waveshaping; >50% loud > 18 dB
    if rng.rand() < 0.10:
        ws_idx = int(rng.randint(1, len(WAVESHAPER_TYPES)))
        p[PARAM_INDEX["waveshaper_type"]] = ws_idx / (len(WAVESHAPER_TYPES) - 1)
        if rng.rand() < 0.55:
            p[PARAM_INDEX["drive"]] = float(rng.uniform(0.75, 1.0)) # 18..24 dB
        else:
            p[PARAM_INDEX["drive"]] = float(rng.uniform(0.08, 0.75)) # 2..18 dB
    else:
        p[PARAM_INDEX["waveshaper_type"]] = 0.0
        p[PARAM_INDEX["drive"]] = 0.0

    p[PARAM_INDEX["chorus_mix"]] = 0.0
    p[PARAM_INDEX["delay_mix"]] = 0.0
    p[PARAM_INDEX["delay_fb"]] = 0.0

    return np.clip(p, 0.0, 1.0).astype(np.float32), midi


class UnifiedSoundsetPrior:
    """Combines 30% bass prior, 10% all-presets prior, 60% pure-random."""
    def __init__(self, bass_manifold=BASS_MANIFOLD_PATH, all_manifold=ALL_MANIFOLD_PATH, split="train"):
        self.bass_prior = SubManifoldPrior(bass_manifold, split=split)
        self.all_prior = SubManifoldPrior(all_manifold, split=split)
        self.N = self.bass_prior.N + self.all_prior.N
        self.calibrated = True

    def support_bounds(self) -> dict:
        """{continuous parameter name: (lo, hi)} in vector units that this prior can produce."""
        b = {n: (0.0, 0.0) for n in FX_NAMES}
        b["midi_note"] = (0.0, 1.0)
        for i in CONT_INDICES:
            name = PARAM_NAMES[i]
            if name not in FX_NAMES and name != "midi_note":
                b[name] = (0.0, 1.0)
        return b

    def sample_vector(self, rng=None):
        rng = np.random if rng is None else rng
        u = rng.rand()
        if u < 0.60:
            # 60% pure random
            vec, midi = sample_unconstrained_random_patch(rng)
            source = "random"
        elif u < 0.90:
            # 30% prior-bass-bracketed
            vec, midi = self.bass_prior.sample_vector(rng)
            source = "bass_prior"
        else:
            # 10% prior-all
            vec, midi = self.all_prior.sample_vector(rng)
            source = "all_prior"
        return vec, midi, source

    def sample_patch_and_midi(self, rng=None):
        rng = np.random if rng is None else rng
        vec, midi, source = self.sample_vector(rng)
        dur = float(rng.uniform(*NOTE_DUR_RANGE))
        return vector_to_patch(vec), vec, midi, dur
