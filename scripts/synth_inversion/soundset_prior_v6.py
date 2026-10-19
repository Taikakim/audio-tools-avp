# Owner: KUANG (created 2026-10-19)
"""V6 training prior with a simplicity bias (V6_SPECIFICATION.md section 3.2): in an ambiguous case the flow's
posterior follows the prior, so the prior should say "one clean oscillator" unless the sound says otherwise.

Mixture (defaults, per draw):
  70 % sparse:   a realistic_prior_v4 random patch for the continuous ranges, then every optional feature is
                 switched off unless drawn on with the probability in SPARSE (osc 2 20 %, three quarters of those
                 as a sub an octave down; osc 3 5 %; waveshaper 25 %; unison 15 %; detune 20 %; complex routing 15 %;
                 feedback 10 %; ...). "Filter 2 off" is approximated by a transparent filter 2 (Serial 1, cutoff
                 fully open, no resonance, no envelope): Surge's filter 2 list here has no Off entry.
  20 % presets:  the soundset manifolds (bass 15 %, all 5 %), converted to v6 (surge_spec_v6.from_v5).
  10 % random:   realistic_prior_v4's unconstrained random patch, converted (coverage).
The MIDI note is drawn uniformly in NOTE_LOW..NOTE_HIGH for sparse and random draws; preset draws keep their own
(moved by osc 1's octave and folded, see from_v5).

Same interface as realistic_prior_v4.UnifiedSoundsetPrior: sample_vector(rng) -> (vec [59], midi, source),
sample_patch_and_midi(rng) -> (patch, vec, midi, dur), N, bass_prior / all_prior, support_bounds(), calibrated.
Verify the result on Surge with ablation_v6.py labels --layout v6: the share of features that are on but inaudible
should drop well below the v5 prior's (osc 2: 75 % on, 20 % of those audible).
"""
import numpy as np

import surge_spec_v6 as S
from realistic_prior_v4 import SubManifoldPrior, manifold_md5, sample_unconstrained_random_patch  # noqa: F401

SPARSE = {
    "osc2": 0.20, "osc2_sub_share": 0.75, "osc3": 0.05, "sub_mix": 0.10, "waveshaper": 0.25, "unison": 0.15,
    "detune": 0.20, "complex_routing": 0.15, "feedback": 0.10, "ring": 0.05, "noise": 0.10, "fm": 0.10,
    "sync": 0.10, "drift": 0.20,
}
MIX = {"sparse": 0.70, "bass_prior": 0.15, "all_prior": 0.05, "random": 0.10}


def sparsify(v, rng, p=SPARSE):
    """Switch optional features off unless drawn on. v: v6 storage (modified copy returned)."""
    I = S.PARAM_INDEX
    v = np.array(v, dtype=np.float32, copy=True)

    def on(key):
        return rng.rand() < p[key]
    v[I["osc_1_volume"]] = float(rng.uniform(0.7, 1.0))
    if on("osc2"):
        v[I["osc_2_volume"]] = float(rng.uniform(0.3, 1.0))
        rel = -1 if rng.rand() < p["osc2_sub_share"] else int(rng.choice([-2, 0, 1, 2]))
        v[I["osc_2_octave"]] = (rel + 2) / 4.0
    else:
        v[I["osc_2_volume"]], v[I["osc_2_octave"]] = 0.0, 0.5
    if on("osc3"):
        v[I["osc_3_volume"]] = float(rng.uniform(0.2, 0.8))
        v[I["osc_3_octave"]] = int(rng.randint(0, 5)) / 4.0
    else:
        v[I["osc_3_volume"]], v[I["osc_3_octave"]] = 0.0, 0.5
    v[I["sub_mix"]] = float(rng.uniform(0.1, 0.7)) if on("sub_mix") else 0.0
    if on("waveshaper"):
        v[I["waveshaper_type"]] = int(rng.randint(1, len(S.WAVESHAPER_TYPES))) / (len(S.WAVESHAPER_TYPES) - 1)
        v[I["drive"]] = float(rng.uniform(0.08, 1.0))
    else:
        v[I["waveshaper_type"]], v[I["drive"]] = 0.0, 0.0
    if on("unison"):
        v[I["unison"]] = int(rng.randint(1, 6)) / 5.0
        v[I["unison_detune"]] = float(rng.uniform(0.05, 0.40))
    else:
        v[I["unison"]], v[I["unison_detune"]] = 0.0, 0.0
    if on("detune"):
        v[I["osc_2_pitch"]], v[I["osc_3_pitch"]] = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
    else:
        v[I["osc_2_pitch"]] = v[I["osc_3_pitch"]] = 0.5
    if not on("complex_routing"):
        v[I["filter_configuration"]] = 0.0                      # Serial 1
        v[I["filter_balance"]] = 0.0
        v[I["filter_2_cutoff"]] = 1.0                           # transparent filter 2
        v[I["filter_2_resonance"]] = 0.0
        v[I["filter_2_feg_amount"]] = 0.5                       # 0.5 = no envelope (bipolar), as the v5 snappy branch
    v[I["filter_feedback"]] = float(rng.uniform(0.1, 0.9)) if on("feedback") else 0.5
    v[I["ring_1x2_volume"]] = float(rng.uniform(0.0, 0.8)) if on("ring") else 0.0
    v[I["ring_2x3_volume"]] = 0.0
    v[I["noise_volume"]] = float(rng.uniform(0.0, 0.25)) if on("noise") else 0.0
    if on("fm"):
        v[I["fm_depth"]] = float(rng.uniform(0.05, 0.45))
        v[I["fm_routing"]] = int(rng.randint(1, 4)) / 3.0
    else:
        v[I["fm_depth"]], v[I["fm_routing"]] = 0.0, 0.0
    v[I["sync"]] = float(rng.uniform(0.0, 0.4)) if on("sync") else 0.0
    v[I["drift"]] = float(rng.uniform(0.0, 0.3)) if on("drift") else 0.0
    for fx in ("chorus_mix", "delay_mix", "delay_fb"):
        v[I[fx]] = 0.0
    return S.canonicalize_vector(v)


class SoundsetPriorV6:
    def __init__(self, bass_manifold, all_manifold, split="train", mix=None, sparse=None):
        self.bass_prior = SubManifoldPrior(bass_manifold, split=split)
        self.all_prior = SubManifoldPrior(all_manifold, split=split)
        self.N = self.bass_prior.N + self.all_prior.N
        self.calibrated = True
        self.mix = dict(MIX if mix is None else mix)
        self.sparse = dict(SPARSE if sparse is None else sparse)
        if abs(sum(self.mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"mix shares must sum to 1, got {self.mix}")

    def support_bounds(self):
        return {S.PARAM_NAMES[i]: (0.0, 1.0) for i in S.CONT_INDICES}

    def _note_unit(self, midi):
        return (midi - S.NOTE_LOW) / (S.NOTE_HIGH - S.NOTE_LOW)

    def sample_vector(self, rng=None):
        rng = np.random if rng is None else rng
        u, acc, source = rng.rand(), 0.0, "sparse"
        for name, share in self.mix.items():
            acc += share
            if u < acc:
                source = name
                break
        if source in ("bass_prior", "all_prior"):
            sub = self.bass_prior if source == "bass_prior" else self.all_prior
            v5, midi = sub.sample_vector(rng)
            vec, midi = S.from_v5(v5, midi)
            return S.canonicalize_vector(vec), int(midi), source
        v5, _ = sample_unconstrained_random_patch(rng)
        vec, _ = S.from_v5(v5)
        midi = int(rng.randint(S.NOTE_LOW, S.NOTE_HIGH + 1))
        vec[S.PARAM_INDEX["midi_note"]] = self._note_unit(midi)
        if source == "sparse":
            vec = sparsify(vec, rng, self.sparse)
        return S.canonicalize_vector(vec), midi, source

    def sample_patch_and_midi(self, rng=None):
        rng = np.random if rng is None else rng
        vec, midi, _ = self.sample_vector(rng)
        dur = float(rng.uniform(*S.NOTE_DUR_RANGE))
        return S.vector_to_patch(vec), vec, midi, dur
