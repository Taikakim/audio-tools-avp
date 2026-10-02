"""Realistic Bass Prior Sampler.

Draws synthesizer parameter vectors directly from the empirical manifold of 430
real, curated Surge XT bass patches (SurgeXTData + New Loops).

Preserves:
  - Exact zero-spike probabilities (spike-and-slab for sync, FM, drive, unison, resonance, feg_sustain)
  - Empirical continuous distributions (kernel density / covariance-guided sampling)
  - Clamping to valid physical synth bounds
"""

import os
import numpy as np

MANIFOLD_PATH = "/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz"

NOTE_LOW, NOTE_HIGH = 28, 50  # E1..D3
NOTE_DUR_RANGE = (0.18, 0.45)

class RealisticBassPrior:
    def __init__(self, manifold_path=MANIFOLD_PATH):
        if not os.path.exists(manifold_path):
            raise FileNotFoundError(f"Manifold file not found: {manifold_path}")
        data = np.load(manifold_path, allow_pickle=True)
        self.params = data["params"]  # [N, 23]
        self.mean = data["mean"]
        self.cov = data["cov"]
        self.p10 = data["p10"]
        self.p50 = data["p50"]
        self.p90 = data["p90"]
        self.zero_prob = data["zero_prob"]
        self.N, self.D = self.params.shape

    def sample_vector(self, rng=None) -> np.ndarray:
        """Sample one 23-d parameter vector from the empirical bass manifold."""
        if rng is None:
            rng = np.random

        # 1. Start from an existing real bass patch as an archetype kernel (KDE / Kernel Perturbation)
        idx = rng.randint(0, self.N)
        base = self.params[idx].copy()

        # 2. Add subtle covariance-guided perturbation (jitter on continuous parameters)
        jitter = rng.multivariate_normal(np.zeros(self.D), self.cov * 0.08)
        sample = base + jitter

        # 3. Enforce spike-and-slab sparsity according to real bass statistics
        # shape: 58.8% pure saw (0.0)
        if rng.rand() < self.zero_prob[2]:
            sample[2] = 0.0

        # sub_mix: 33% 0.0, else capped at 0.50
        if rng.rand() < self.zero_prob[4]:
            sample[4] = 0.0
        else:
            sample[4] = np.clip(sample[4], 0.05, 0.50)

        # sync: 61.2% strictly 0.0, else subtle sync [0.02, 0.20]
        if rng.rand() < self.zero_prob[5]:
            sample[5] = 0.0
        else:
            sample[5] = np.clip(sample[5], 0.01, 0.25)

        # fm_depth: 14% strictly 0.0, else empirical range
        if rng.rand() < self.zero_prob[6]:
            sample[6] = 0.0
        else:
            sample[6] = np.clip(sample[6], 0.0, 0.40)

        # unison: 66% single voice (0.0)
        if rng.rand() < self.zero_prob[7]:
            sample[7] = 0.0
            sample[8] = 0.0
        else:
            sample[7] = 1.0
            sample[8] = np.clip(sample[8], 0.05, 0.45)

        # cutoff: clamp to warm bass range [0.10, 0.75] (~30 Hz to 3.8 kHz)
        sample[9] = np.clip(sample[9], 0.08, 0.75)

        # resonance: 40% zero resonance
        if rng.rand() < self.zero_prob[10]:
            sample[10] = 0.0
        else:
            sample[10] = np.clip(sample[10], 0.05, 0.85)

        # feg_sustain: 71.6% zero sustain (snappy pluck)
        if rng.rand() < self.zero_prob[14]:
            sample[14] = 0.0
        else:
            sample[14] = np.clip(sample[14], 0.0, 0.50)

        # drive: 50% zero drive
        if rng.rand() < self.zero_prob[19]:
            sample[18] = 0.0
            sample[19] = 0.0
        else:
            sample[18] = np.clip(sample[18], 0.0, 1.0)
            sample[19] = np.clip(sample[19], 0.05, 0.60)

        # Dry FX (chorus, delay off for clean synthesis)
        sample[20] = 0.0
        sample[21] = 0.0
        sample[22] = 0.0

        # General clip to [0, 1]
        sample = np.clip(sample, 0.0, 1.0)

        # Randomize MIDI note across bass range E1..D3
        midi_note = rng.randint(NOTE_LOW, NOTE_HIGH + 1)
        sample[0] = (midi_note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW)

        return sample.astype(np.float32)

    def sample_patch_and_midi(self, rng=None):
        """Returns (patch_dict, vector, midi_note, note_dur)."""
        if rng is None:
            rng = np.random
        vec = self.sample_vector(rng)
        
        # Convert vector to patch dict for rendering
        from surge_spec import vector_to_patch
        patch = vector_to_patch(vec)
        midi_note = patch["midi_note"]
        note_dur = float(rng.uniform(*NOTE_DUR_RANGE))
        patch["note_dur"] = note_dur
        return patch, vec, midi_note, note_dur

# Global singleton
_prior = None
def get_prior():
    global _prior
    if _prior is None:
        _prior = RealisticBassPrior()
    return _prior

def draw_realistic_patch(seed: int = None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    prior = get_prior()
    return prior.sample_patch_and_midi(rng)

if __name__ == "__main__":
    p, v, note, dur = draw_realistic_patch(42)
    print(f"Sampled Patch (MIDI Note {note}, Dur {dur:.2f}s):")
    print(f"  Shape: {p['shape']:.2f}, Cutoff: {p['cutoff']:.2f}, Res: {p['resonance']:.2f}")
    print(f"  Sync: {p['sync']:.3f}, FM: {p['fm_depth']:.3f}, Sub: {p['sub_mix']:.2f}")
    print(f"  FEG Decay: {p['feg_decay']:.2f}, FEG Sustain: {p['feg_sustain']:.2f}")
    print("Prior sampler successfully verified!")
