#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Musical Dataset V8: Musically-Grounded Dynamic Training Data Augmentation.

Pillar 1 of the V8 Architecture:
1. Loads 40 multitracks from the AVP studio archive, holding out 6 songs completely.
2. Dual Target Distribution:
   - 50% known Surge renders (with ground-truth parameter vectors)
   - 50% authentic synth stems from the archive (unlabeled for parameters, but real analog timbre)
3. Live Pedalboard FX Rack applied to target:
   - Chorus, Delay, Reverb, Distortion
4. Backing Arrangement:
   - Authentic Goa/psytrance drum loops + isolated kick hit substitution
   - Dynamic mixing at realistic SNRs (-6 dB to +6 dB)
5. Returns (mix, target, cue_pos, cue_neg) for Asymmetric TSE training.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
import pedalboard
from pedalboard import Chorus, Delay, Distortion, HighpassFilter, LowpassFilter, Reverb

DEFAULT_ARCHIVE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "/run/media/kim/Mantu/avp-stems-original-classified"))
HOLDOUT_SONGS = [
    "Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance",
    "Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance",
    "Aavepyörä - Dance of Tomorrow driving rhythmic hypnotic  goatrance",
    "Aavepyörä - Commodore Shiva impulse tracker soft adventure trance",
    "Summamutikka - Lohikäärmelaulu - furious goa trance",
    "Summamutikka - Naidenie Flowers furious oldschool goa trance",
]


class MusicalDatasetV8(Dataset):
    """Generates authentic polyphonic electronic music mixtures for TSE training."""

    def __init__(
        self,
        archive_root: Path = DEFAULT_ARCHIVE_ROOT,
        bank_v6_dir: Optional[str] = "/home/kim/sao_runs/bank_v6",
        sample_rate: int = 44100,
        clip_duration_s: float = 4.0,
        epoch_length: int = 10000,
        split: str = "train",  # 'train' or 'val'
        cue_duration_s: float = 1.0,
    ):
        super().__init__()
        self.archive_root = Path(archive_root)
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_s)
        self.cue_samples = int(sample_rate * cue_duration_s)
        self.epoch_length = epoch_length
        self.split = split

        # 1. Discover multitrack stems
        self.kicks: List[Path] = []
        self.drums: List[Path] = []
        self.bass_stems: List[Path] = []
        self.other_synths: List[Path] = []

        if self.archive_root.exists():
            for sdir in sorted(self.archive_root.iterdir()):
                if not sdir.is_dir() or sdir.name.startswith("."):
                    continue

                is_holdout = sdir.name in HOLDOUT_SONGS
                if self.split == "train" and is_holdout:
                    continue
                if self.split == "val" and not is_holdout:
                    continue

                # Kicks
                kick_dir = sdir / "kick"
                if kick_dir.exists():
                    self.kicks.extend(list(kick_dir.glob("*.flac")) + list(kick_dir.glob("*.wav")))

                # Full drums
                drum_dir = sdir / "drums"
                if drum_dir.exists():
                    self.drums.extend(list(drum_dir.glob("*.flac")) + list(drum_dir.glob("*.wav")))

                # Bass stems
                bass_dir = sdir / "bass"
                if bass_dir.exists():
                    self.bass_stems.extend(list(bass_dir.glob("*.flac")) + list(bass_dir.glob("*.wav")))

                # Melodic synths (acid, leads, pads, arp_keys)
                for cat in ["acid", "leads", "pads", "arp_keys"]:
                    cdir = sdir / cat
                    if cdir.exists():
                        self.other_synths.extend(list(cdir.glob("*.flac")) + list(cdir.glob("*.wav")))

        # 2. Check Surge bank_v6 availability
        self.has_bank_v6 = False
        if bank_v6_dir and os.path.exists(bank_v6_dir):
            audio_path = os.path.join(bank_v6_dir, "main_00000_audio.npy")
            meta_path = os.path.join(bank_v6_dir, "main_00000_meta.npz")
            if os.path.exists(audio_path) and os.path.exists(meta_path):
                self.bank_audio = np.load(audio_path, mmap_mode="r")
                self.bank_meta = np.load(meta_path)
                self.has_bank_v6 = True
                self.num_bank_patches = len(self.bank_audio)

        # 3. Check fast NVMe stem cache
        cache_dir = Path("/home/kim/sao_runs/v8_stem_cache")
        self.use_cache = False
        if cache_dir.exists() and (cache_dir / "kicks.npy").exists() and self.split == "train":
            try:
                self.cached_kicks = np.load(cache_dir / "kicks.npy", mmap_mode="r")
                self.cached_drums = np.load(cache_dir / "drums.npy", mmap_mode="r")
                self.cached_bass = np.load(cache_dir / "bass.npy", mmap_mode="r")
                self.cached_synths = np.load(cache_dir / "synths.npy", mmap_mode="r")
                self.use_cache = True
            except Exception:
                self.use_cache = False

    def _random_crop(self, path: Path) -> np.ndarray:
        """Reads a random 4.0s mono slice from an audio file."""
        try:
            info = sf.info(path)
            total_frames = info.frames
            needed_frames = int(self.clip_samples * (info.samplerate / self.sample_rate)) + 256
            if total_frames <= needed_frames:
                data, sr = sf.read(path, dtype="float32")
            else:
                max_start = total_frames - needed_frames
                start = random.randint(0, max_start)
                data, sr = sf.read(path, start=start, frames=needed_frames, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != self.sample_rate:
                data = torchaudio.functional.resample(
                    torch.from_numpy(data), sr, self.sample_rate
                ).numpy()
            # Ensure exact length
            if len(data) < self.clip_samples:
                data = np.pad(data, (0, self.clip_samples - len(data)))
            else:
                data = data[:self.clip_samples]
            return data
        except Exception:
            return np.zeros(self.clip_samples, dtype=np.float32)

    def _apply_live_fx(self, wav: np.ndarray) -> np.ndarray:
        """Applies a randomized Pedalboard effects chain on the fly."""
        board = pedalboard.Pedalboard()

        # 1. Modulation (Chorus / Flanger)
        if random.random() < 0.4:
            board.append(Chorus(rate_hz=random.uniform(0.5, 2.5), depth=random.uniform(0.1, 0.5)))

        # 2. Time-domain (Delay / Reverb)
        if random.random() < 0.45:
            board.append(
                Delay(
                    delay_seconds=random.choice([0.107, 0.160, 0.214, 0.320]),  # 16th, 8th dot sync
                    feedback=random.uniform(0.15, 0.45),
                    mix=random.uniform(0.15, 0.40),
                )
            )
        if random.random() < 0.35:
            board.append(Reverb(room_size=random.uniform(0.1, 0.4), damping=0.5, wet_level=0.2))

        # 3. Saturation
        if random.random() < 0.3:
            board.append(Distortion(drive_db=random.uniform(2.0, 8.0)))

        try:
            effected = board(wav, self.sample_rate)
            return effected.astype(np.float32)
        except Exception:
            return wav

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Target Selection
        is_surge = self.has_bank_v6 and (random.random() < 0.5)
        if is_surge:
            patch_idx = random.randint(0, self.num_bank_patches - 1)
            raw_audio = self.bank_audio[patch_idx].astype(np.float32) / 32768.0  # 35280 samples (0.8s)
            repeats = (self.clip_samples // len(raw_audio)) + 1
            target_clean = np.tile(raw_audio, repeats)[:self.clip_samples]
        elif self.use_cache:
            if random.random() < 0.7 and len(self.cached_bass) > 0:
                idx_b = random.randint(0, len(self.cached_bass) - 1)
                target_clean = self.cached_bass[idx_b].copy()
            elif len(self.cached_synths) > 0:
                idx_s = random.randint(0, len(self.cached_synths) - 1)
                target_clean = self.cached_synths[idx_s].copy()
            else:
                target_clean = np.zeros(self.clip_samples, dtype=np.float32)
        else:
            if self.bass_stems and (random.random() < 0.7):
                stem_path = random.choice(self.bass_stems)
            elif self.other_synths:
                stem_path = random.choice(self.other_synths)
            else:
                stem_path = random.choice(self.kicks) if self.kicks else None
            
            target_clean = self._random_crop(stem_path) if stem_path else np.zeros(self.clip_samples, dtype=np.float32)

        # 2. Apply Live Pedalboard FX to target in mix
        target_in_mix = self._apply_live_fx(target_clean)

        # 3. Backing Arrangement
        if self.use_cache:
            if random.random() < 0.6 and len(self.cached_drums) > 0:
                drum_stem = self.cached_drums[random.randint(0, len(self.cached_drums) - 1)].copy()
            elif len(self.cached_kicks) > 0:
                drum_stem = self.cached_kicks[random.randint(0, len(self.cached_kicks) - 1)].copy()
            else:
                drum_stem = np.zeros(self.clip_samples, dtype=np.float32)

            if random.random() < 0.6 and len(self.cached_synths) > 0:
                acc_stem = self.cached_synths[random.randint(0, len(self.cached_synths) - 1)].copy()
            else:
                acc_stem = np.zeros(self.clip_samples, dtype=np.float32)
        else:
            # Drum stem
            if self.drums and (random.random() < 0.6):
                drum_stem = self._random_crop(random.choice(self.drums))
            elif self.kicks:
                drum_stem = self._random_crop(random.choice(self.kicks))
            else:
                drum_stem = np.zeros(self.clip_samples, dtype=np.float32)

            # Melodic accompaniment
            if self.other_synths and (random.random() < 0.6):
                acc_stem = self._random_crop(random.choice(self.other_synths))
            else:
                acc_stem = np.zeros(self.clip_samples, dtype=np.float32)

        # 4. Mixing at target SNR
        snr_db = random.uniform(-6.0, 6.0)
        t_rms = np.sqrt(np.mean(target_in_mix ** 2)) + 1e-6
        d_rms = np.sqrt(np.mean(drum_stem ** 2)) + 1e-6
        a_rms = np.sqrt(np.mean(acc_stem ** 2)) + 1e-6

        noise_stem = drum_stem + (acc_stem * (d_rms / a_rms) * 0.5)
        n_rms = np.sqrt(np.mean(noise_stem ** 2)) + 1e-6

        desired_scale = (t_rms / (n_rms + 1e-6)) * (10.0 ** (-snr_db / 20.0))
        scaled_noise = noise_stem * desired_scale
        mixture = target_in_mix + scaled_noise

        # Peak normalization
        max_val = max(1.0, float(np.abs(mixture).max()))
        mixture = mixture / max_val
        target_clean = target_clean / max_val

        # 5. Positive Audio Cue (1.0s slice of clean target)
        cue_start = random.randint(0, max(0, self.clip_samples - self.cue_samples))
        cue_pos = target_clean[cue_start : cue_start + self.cue_samples]

        # 6. Negative Audio Cue (mismatched)
        if self.use_cache and len(self.cached_synths) > 0:
            cue_neg = self.cached_synths[random.randint(0, len(self.cached_synths) - 1)][:self.cue_samples].copy()
        elif self.other_synths:
            neg_path = random.choice(self.other_synths)
            cue_neg = self._random_crop(neg_path)[:self.cue_samples]
        else:
            cue_neg = np.zeros(self.cue_samples, dtype=np.float32)

        return {
            "mix": torch.from_numpy(mixture).float().unsqueeze(0),
            "target": torch.from_numpy(target_clean).float().unsqueeze(0),
            "cue_pos": torch.from_numpy(cue_pos).float().unsqueeze(0),
            "cue_neg": torch.from_numpy(cue_neg).float().unsqueeze(0),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


if __name__ == "__main__":
    ds = MusicalDatasetV8(split="train", epoch_length=10)
    print(f"MusicalDatasetV8 initialized. Total training kicks: {len(ds.kicks)}, drums: {len(ds.drums)}, bass: {len(ds.bass_stems)}, synths: {len(ds.other_synths)}")
    sample = ds[0]
    print(f"Sample shapes - Mix: {sample['mix'].shape}, Target: {sample['target'].shape}, Cue Pos: {sample['cue_pos'].shape}, SNR: {sample['snr_db'].item():+.1f} dB")
