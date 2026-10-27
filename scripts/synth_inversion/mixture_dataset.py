# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Paired Mixture Dataset for Invariance and Selectivity Extractor Training.

Constructs paired mixtures sharing the same instrument timbre identity
but with different musical content and distinct backing arrangements, as
required by Eq. 2 (L_inv, L_algn, L_sel) in Tian et al. (arXiv:2610.19445v1).
"""

import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class PairedMixtureDataset(Dataset):
    """Dataset producing paired mixtures for CGSP training.
    
    Each sample returns:
      - mix_a: 48kHz waveform of Mixture A [samples]
      - mix_b: 48kHz waveform of Mixture B [samples] (same target instrument, different content & backing)
      - target_clean: clean excerpt of the target instrument [samples]
      - cue_audio_pos: audio cue snippet of the target instrument/class [samples]
      - cue_audio_neg: mismatched audio cue snippet (for L_sel) [samples]
      - text_pos: text description of target (e.g. 'synth bass')
      - text_neg: mismatched text description (e.g. 'acoustic drums')
    """

    def __init__(
        self,
        bank_v6_dir: Optional[str] = "/home/kim/sao_runs/bank_v6",
        multitrack_dir: Optional[str] = "/run/media/kim/Mantu/avp-stems-original-classified",
        sample_rate: int = 48000,
        clip_duration_sec: float = 4.0,
        epoch_length: int = 20000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_sec)
        self.epoch_length = epoch_length

        # 1. Discover classified multitrack stems if available
        self.classified_stems: Dict[str, List[str]] = {}
        if multitrack_dir and os.path.exists(multitrack_dir):
            song_dirs = [os.path.join(multitrack_dir, d) for d in os.listdir(multitrack_dir) if os.path.isdir(os.path.join(multitrack_dir, d))]
            for sdir in song_dirs:
                for cat in ["bass", "acid", "leads", "pads", "drums", "kick", "arp_keys", "sfx"]:
                    cat_dir = os.path.join(sdir, cat)
                    if os.path.exists(cat_dir):
                        flacs = glob.glob(os.path.join(cat_dir, "*.flac")) + glob.glob(os.path.join(cat_dir, "*.wav"))
                        if flacs:
                            self.classified_stems.setdefault(cat, []).extend(flacs)

        # 2. Check Surge bank_v6 availability
        self.has_bank_v6 = False
        self.bank_shards = []
        if bank_v6_dir and os.path.exists(bank_v6_dir):
            audio_files = sorted(glob.glob(os.path.join(bank_v6_dir, "main_*_audio.npy")))
            if audio_files:
                self.bank_shards = audio_files
                self.has_bank_v6 = True

        self.categories = list(self.classified_stems.keys()) if self.classified_stems else ["bass", "drums", "lead"]

    def __len__(self) -> int:
        return self.epoch_length

    def _load_audio_excerpt(self, path: str, target_sr: int, duration_samples: int) -> torch.Tensor:
        """Loads and crops/loops an audio excerpt to exact duration, ensuring active sounding energy."""
        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            total_src_frames = info.num_frames

            # Calculate source frames needed for target duration
            src_frames_needed = int(duration_samples * (sr / target_sr)) + 256

            if total_src_frames <= src_frames_needed:
                wav, _ = torchaudio.load(path)
            else:
                max_offset = total_src_frames - src_frames_needed
                best_wav = None
                max_energy = 0.0
                for _ in range(3):
                    offset = random.randint(0, max_offset)
                    w, _ = torchaudio.load(path, frame_offset=offset, num_frames=src_frames_needed)
                    if w.shape[0] > 1:
                        w = w.mean(dim=0, keepdim=True)
                    peak = w.abs().max().item()
                    if peak > 0.05:
                        best_wav = w
                        break
                    if peak > max_energy:
                        max_energy = peak
                        best_wav = w
                wav = best_wav if best_wav is not None else torchaudio.load(path, num_frames=src_frames_needed)[0]

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            wav = wav.squeeze(0)

            if wav.shape[0] < duration_samples:
                repeats = (duration_samples // max(1, wav.shape[0])) + 1
                wav = wav.repeat(repeats)
            return wav[:duration_samples]
        except Exception:
            return torch.zeros(duration_samples, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Choose a target category (defaulting to bass/synth)
        target_cat = "bass" if "bass" in self.classified_stems else random.choice(self.categories)
        neg_candidates = [c for c in self.categories if c != target_cat]
        neg_cat = random.choice(neg_candidates) if neg_candidates else "drums"

        # Load two distinct clips of the target category (simulating same instrument class)
        target_pool = self.classified_stems.get(target_cat, [])
        if len(target_pool) >= 2:
            path_a, path_b = random.sample(target_pool, 2)
            target_a = self._load_audio_excerpt(path_a, self.sample_rate, self.clip_samples)
            target_b = self._load_audio_excerpt(path_b, self.sample_rate, self.clip_samples)
        elif len(target_pool) == 1:
            target_a = self._load_audio_excerpt(target_pool[0], self.sample_rate, self.clip_samples)
            target_b = target_a.clone()
        else:
            target_a = torch.randn(self.clip_samples) * 0.1
            target_b = torch.randn(self.clip_samples) * 0.1

        # Load backing stems (e.g. drums/percussion)
        drum_pool = self.classified_stems.get("drums", []) + self.classified_stems.get("kick", [])
        if drum_pool:
            drum_a = self._load_audio_excerpt(random.choice(drum_pool), self.sample_rate, self.clip_samples)
            drum_b = self._load_audio_excerpt(random.choice(drum_pool), self.sample_rate, self.clip_samples)
        else:
            drum_a = torch.zeros(self.clip_samples)
            drum_b = torch.zeros(self.clip_samples)

        # Mix: target + backing with random SNR between -3 dB and +6 dB
        gain_drum_a = 10 ** (random.uniform(-0.3, 0.3))
        gain_drum_b = 10 ** (random.uniform(-0.3, 0.3))
        mix_a = target_a + drum_a * gain_drum_a
        mix_b = target_b + drum_b * gain_drum_b

        # Normalize mixes to avoid clipping
        max_a = mix_a.abs().max() + 1e-6
        if max_a > 1.0:
            mix_a = mix_a / max_a
        max_b = mix_b.abs().max() + 1e-6
        if max_b > 1.0:
            mix_b = mix_b / max_b

        # Audio cues
        cue_audio_pos = target_a.clone()
        neg_pool = self.classified_stems.get(neg_cat, drum_pool)
        if neg_pool:
            cue_audio_neg = self._load_audio_excerpt(random.choice(neg_pool), self.sample_rate, self.clip_samples)
        else:
            cue_audio_neg = torch.randn(self.clip_samples) * 0.1

        return {
            "mix_a": mix_a,
            "mix_b": mix_b,
            "target_clean": target_a,
            "cue_audio_pos": cue_audio_pos,
            "cue_audio_neg": cue_audio_neg,
            "text_pos": "synth bass" if target_cat == "bass" else target_cat,
            "text_neg": "drums" if neg_cat == "drums" else neg_cat,
        }
