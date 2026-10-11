"""Stand-in subtractive bass synth for CPU tests and probes. NOT Surge XT.

Why it exists: the v4 components (onset alignment, transient descriptors, resonance card, buzz measures,
objectives, encoders) must be testable on a machine without Surge, Essentia or a GPU. This synth is additive:
a sawtooth (1/k harmonics) shaped by a two-pole resonant low-pass whose cutoff has an exponential envelope,
with an attack / decay / sustain / release amplitude envelope and optional phase modulation ("FM") for buzz tests.
Every control is analytic, so a probe knows the true parameter it sweeps.

Results obtained with this synth are labeled "stand-in" everywhere. They show that a measure follows a
parameter in a clean, controlled setting; they say nothing about Surge or real stems.
"""
from dataclasses import dataclass

import numpy as np

SR = 44100


@dataclass
class StandinPatch:
    f0: float = 55.0
    fc_hz: float = 1200.0        # resting cutoff
    fc_env_oct: float = 0.0      # octaves the cutoff is raised at the attack (decays with fc_decay_s)
    fc_decay_s: float = 0.1
    q: float = 0.7               # resonance (two-pole low-pass Q)
    attack_s: float = 0.003
    amp_decay_s: float = 0.3
    amp_sustain: float = 0.6
    release_s: float = 0.05
    detune_cents: float = 0.0
    fm_ratio: float = 0.0        # phase-modulator frequency as a ratio of f0 (non-integer = inharmonic)
    fm_depth: float = 0.0        # radians
    fm_decay_s: float = 0.0      # 0 = static FM; >0 = FM depth decays exponentially (enveloped FM)
    phase_seed: int = -1         # -1: all harmonics start at phase 0 (a saw whose first cycle ramps from 0).
                                 # >= 0: seeded random harmonic phases (a free-running oscillator proxy); use this
                                 # whenever attack time is what is being measured


def _lowpass_gain(f, fc, q):
    x = f / fc
    return 1.0 / np.sqrt((1.0 - x * x) ** 2 + (x / q) ** 2)


def render(patch: StandinPatch, dur: float = 0.8, note_dur: float = 0.5, onset_s: float = 0.0,
           sr: int = SR, peak_norm: bool = True, max_harm: int = 200) -> np.ndarray:
    """Render `dur` seconds of mono audio. The note starts at `onset_s` and is released at onset_s + note_dur."""
    n = int(round(dur * sr))
    t = np.arange(n, dtype=np.float64) / sr
    tn = t - onset_s                                   # time since note start (negative before the note)
    f0 = patch.f0 * 2.0 ** (patch.detune_cents / 1200.0)
    k_max = int(min(max_harm, np.floor(0.45 * sr / f0)))
    ks = np.arange(1, k_max + 1, dtype=np.float64)

    # filter gain per harmonic on a coarse time grid (2 ms), linearly interpolated to samples
    grid_t = np.arange(0.0, dur + 0.002, 0.002)
    fc_t = patch.fc_hz * 2.0 ** (patch.fc_env_oct * np.exp(-np.maximum(grid_t - onset_s, 0.0) / max(patch.fc_decay_s, 1e-4)))
    gains = _lowpass_gain((ks * f0)[:, None], fc_t[None, :], patch.q) / ks[:, None]      # [K, M]

    amp = np.zeros(n)
    on = tn >= 0.0
    ta = tn[on]
    a = np.minimum(ta / max(patch.attack_s, 1e-5), 1.0)
    d = patch.amp_sustain + (1.0 - patch.amp_sustain) * np.exp(-np.maximum(ta - patch.attack_s, 0.0) / max(patch.amp_decay_s, 1e-4))
    rel = np.where(ta > note_dur, np.exp(-(ta - note_dur) / max(patch.release_s, 1e-4)), 1.0)
    amp[on] = a * d * rel

    mod = None
    if patch.fm_depth > 0.0 and patch.fm_ratio > 0.0:
        depth = patch.fm_depth * (np.exp(-np.maximum(tn, 0.0) / patch.fm_decay_s) if patch.fm_decay_s > 0 else 1.0)
        mod = depth * np.sin(2.0 * np.pi * patch.fm_ratio * f0 * tn)

    ph0 = (np.random.default_rng(patch.phase_seed).uniform(0.0, 2.0 * np.pi, k_max)
           if patch.phase_seed >= 0 else np.zeros(k_max))
    out = np.zeros(n)
    chunk = 20
    for i in range(0, k_max, chunk):
        kk = ks[i:i + chunk]
        g = np.stack([np.interp(t, grid_t, gains[i + j]) for j in range(len(kk))])        # [chunk, N]
        ph = 2.0 * np.pi * (kk[:, None] * f0) * tn[None, :] + ph0[i:i + chunk][:, None]
        if mod is not None:
            ph = ph + mod[None, :]
        out += (g * np.sin(ph)).sum(axis=0)
    out *= amp
    if peak_norm:
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak
    return out.astype(np.float32)


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)
