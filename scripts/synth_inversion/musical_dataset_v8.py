#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Musical Dataset V8: Musically-Grounded Dynamic Training Data Augmentation.

Pillar 1 of the V8 Architecture:
1. Loads 40 multitracks from the AVP studio archive, holding out 6 songs completely.
2. Target distribution:
   - authentic synth stems from the archive (unlabeled for parameters, but real analog timbre)
   - optionally (surge_frac > 0) known Surge renders from bank_v6, tiled to 4 s
3. Live Pedalboard FX Rack applied to target:
   - Chorus, Delay, Reverb, Distortion
4. Backing Arrangement:
   - Authentic Goa/psytrance drum loops or isolated kick stems (random crops; no shared grid with the target yet)
   - Dynamic mixing at realistic SNRs (-6 dB to +6 dB)
5. Returns (mix, target, cue_pos, cue_neg) for Asymmetric TSE training.

KUANG 2026-10-29 (docs/synth-v8-code-review-2026-10-29.md, C4):
- The positive cue never overlaps the target window. With the stem cache it is a 1 s window of another clip of the
  same stem (cache_v8_stems.py stores stem ids); from files it is a 1 s window outside the target crop. At use time
  the cue comes from elsewhere in the track, so a cue cut from the target taught a shortcut the model cannot use.
- Surge bank items are off by default (surge_frac = 0.0): a bank item is a 0.8 s loop, so any cue taken from it is
  the target itself. With surge_frac > 0 that overlap is kept and flagged (cue_disjoint = False) for ablations.
- Targets and cues that are near-silent are redrawn.
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
DEFAULT_CACHE_DIR = Path("/home/kim/sao_runs/v8_stem_cache")
HOLDOUT_SONGS = [
    "Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance",
    "Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance",
    "Aavepyörä - Dance of Tomorrow driving rhythmic hypnotic  goatrance",
    "Aavepyörä - Commodore Shiva impulse tracker soft adventure trance",
    "Summamutikka - Lohikäärmelaulu - furious goa trance",
    "Summamutikka - Naidenie Flowers furious oldschool goa trance",
]
SILENCE_RMS = 0.003  # cue / target windows quieter than this are redrawn
MAX_DRAWS = 16


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def loudest_window(clip: np.ndarray, length: int, tries: int = 4) -> np.ndarray:
    """A random length-sample window of clip, the loudest of a few tries (cues should hold the sound)."""
    if len(clip) <= length:
        return np.pad(clip, (0, length - len(clip)))
    best, best_rms = None, -1.0
    for _ in range(tries):
        s = random.randint(0, len(clip) - length)
        w = clip[s:s + length]
        r = _rms(w)
        if r > best_rms:
            best, best_rms = w, r
    return np.array(best, dtype=np.float32, copy=True)


def pick_start(total: int, length: int, avoid: Optional[Tuple[int, int]] = None) -> Optional[int]:
    """A uniform random start s with [s, s + length) inside [0, total) and, given avoid = (a0, a1), not overlapping
    [a0, a1). None when no such start exists."""
    last = total - length
    if last < 0:
        return None
    if avoid is None:
        return random.randint(0, last)
    a0, a1 = avoid
    ranges = [(0, min(last, a0 - length)), (max(0, a1), last)]  # inclusive start ranges
    ranges = [(lo, hi) for lo, hi in ranges if hi >= lo]
    if not ranges:
        return None
    sizes = [hi - lo + 1 for lo, hi in ranges]
    k = random.choices(range(len(ranges)), weights=sizes)[0]
    return random.randint(*ranges[k])


def group_by_stem(stem_ids: np.ndarray) -> Tuple[Dict[int, List[int]], List[int]]:
    """{stem id: clip indices} and the clip indices whose stem has at least one other clip (usable as cued targets)."""
    groups: Dict[int, List[int]] = {}
    for i, s in enumerate(np.asarray(stem_ids).tolist()):
        groups.setdefault(int(s), []).append(i)
    eligible = [i for idx in groups.values() if len(idx) > 1 for i in idx]
    return groups, eligible


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
        surge_frac: float = 0.0,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    ):
        super().__init__()
        self.archive_root = Path(archive_root)
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_s)
        self.cue_samples = int(sample_rate * cue_duration_s)
        self.epoch_length = epoch_length
        self.split = split
        self.surge_frac = float(surge_frac)

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

        # 2. Check Surge bank_v6 availability (only read when surge_frac > 0)
        self.has_bank_v6 = False
        if self.surge_frac > 0 and bank_v6_dir and os.path.exists(bank_v6_dir):
            audio_path = os.path.join(bank_v6_dir, "main_00000_audio.npy")
            meta_path = os.path.join(bank_v6_dir, "main_00000_meta.npz")
            if os.path.exists(audio_path) and os.path.exists(meta_path):
                self.bank_audio = np.load(audio_path, mmap_mode="r")
                self.bank_meta = np.load(meta_path)
                self.has_bank_v6 = True
                self.num_bank_patches = len(self.bank_audio)

        # 3. Check fast NVMe stem cache (needs stem ids for disjoint cues)
        self.use_cache = False
        prefix = "val_" if self.split == "val" else ""
        cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir is not None and (cache_dir / f"{prefix}kicks.npy").exists():
            if not (cache_dir / f"{prefix}bass_stem_ids.npy").exists():
                raise RuntimeError(
                    f"{cache_dir} holds a stem cache without stem ids, so cues cannot be kept apart from targets. "
                    f"Rebuild it: python cache_v8_stems.py --split {self.split}")
            self.cached_kicks = np.load(cache_dir / f"{prefix}kicks.npy", mmap_mode="r")
            self.cached_drums = np.load(cache_dir / f"{prefix}drums.npy", mmap_mode="r")
            self.cached_bass = np.load(cache_dir / f"{prefix}bass.npy", mmap_mode="r")
            self.cached_synths = np.load(cache_dir / f"{prefix}synths.npy", mmap_mode="r")
            self.bass_ids = np.load(cache_dir / f"{prefix}bass_stem_ids.npy")
            self.synth_ids = np.load(cache_dir / f"{prefix}synths_stem_ids.npy")
            self.bass_groups, self.bass_eligible = group_by_stem(self.bass_ids)
            self.synth_groups, self.synth_eligible = group_by_stem(self.synth_ids)
            if not self.bass_eligible and not self.synth_eligible:
                raise RuntimeError("No cached stem has two or more clips; disjoint cues are impossible.")
            self.use_cache = True

    def _random_crop(self, path: Path, avoid: Optional[Tuple[int, int]] = None,
                     length: Optional[int] = None) -> Tuple[Optional[np.ndarray], int]:
        """Reads a random mono slice (length samples, default the clip length) from an audio file.

        avoid: (start, end) in output-rate samples that the slice must not overlap. Returns (audio, start) with
        audio None when the file cannot give such a slice.
        """
        length = self.clip_samples if length is None else length
        try:
            info = sf.info(path)
            ratio = info.samplerate / self.sample_rate
            total = int(info.frames / ratio)
            if total < length:
                if avoid is not None:
                    return None, 0
                data, sr = sf.read(path, dtype="float32")
                start = 0
            else:
                start = pick_start(total, length, avoid)
                if start is None:
                    return None, 0
                data, sr = sf.read(path, start=int(start * ratio), frames=int(length * ratio) + 256, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != self.sample_rate:
                data = torchaudio.functional.resample(torch.from_numpy(data), sr, self.sample_rate).numpy()
            # Ensure exact length
            if len(data) < length:
                data = np.pad(data, (0, length - len(data)))
            return data[:length].astype(np.float32), start
        except Exception:
            return None, 0

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

    def _target_and_cue(self) -> Tuple[np.ndarray, np.ndarray, bool, Optional[int]]:
        """(target [clip], cue [cue], cue_disjoint, target stem id or None)."""
        if self.has_bank_v6 and random.random() < self.surge_frac:
            patch_idx = random.randint(0, self.num_bank_patches - 1)
            raw_audio = self.bank_audio[patch_idx].astype(np.float32) / 32768.0  # 35280 samples (0.8s)
            repeats = (self.clip_samples // len(raw_audio)) + 1
            target = np.tile(raw_audio, repeats)[:self.clip_samples]
            return target, loudest_window(target, self.cue_samples), False, None

        for _ in range(MAX_DRAWS):
            if self.use_cache:
                use_bass = self.bass_eligible and (random.random() < 0.7 or not self.synth_eligible)
                bank, ids, groups, eligible = (
                    (self.cached_bass, self.bass_ids, self.bass_groups, self.bass_eligible) if use_bass else
                    (self.cached_synths, self.synth_ids, self.synth_groups, self.synth_eligible))
                t_idx = random.choice(eligible)
                stem = int(ids[t_idx])
                c_idx = random.choice([i for i in groups[stem] if i != t_idx])
                target = np.array(bank[t_idx], dtype=np.float32, copy=True)
                cue = loudest_window(np.asarray(bank[c_idx], dtype=np.float32), self.cue_samples)
            else:
                pool = self.bass_stems if (self.bass_stems and random.random() < 0.7) else self.other_synths
                if not pool:
                    raise RuntimeError("No bass or synth stems found for the V8 dataset.")
                path = random.choice(pool)
                target, start = self._random_crop(path)
                if target is None:
                    continue
                cue, _ = self._random_crop(path, avoid=(start, start + self.clip_samples), length=self.cue_samples)
                if cue is None:
                    continue
                stem = None
            if _rms(target) > SILENCE_RMS and _rms(cue) > SILENCE_RMS:
                return target, cue, True, stem
        raise RuntimeError(f"No audible target with a disjoint cue in {MAX_DRAWS} draws.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Target selection, with a positive cue that does not overlap the target window
        target_clean, cue_pos, cue_disjoint, target_stem = self._target_and_cue()

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

            acc_stem = np.zeros(self.clip_samples, dtype=np.float32)
            if random.random() < 0.6 and len(self.cached_synths) > 0:
                a_idx = random.randint(0, len(self.cached_synths) - 1)
                if target_stem is None or int(self.synth_ids[a_idx]) != target_stem:
                    acc_stem = self.cached_synths[a_idx].copy()
        else:
            # Drum stem
            drum_stem = None
            if self.drums and (random.random() < 0.6):
                drum_stem, _ = self._random_crop(random.choice(self.drums))
            elif self.kicks:
                drum_stem, _ = self._random_crop(random.choice(self.kicks))
            if drum_stem is None:
                drum_stem = np.zeros(self.clip_samples, dtype=np.float32)

            # Melodic accompaniment
            acc_stem = None
            if self.other_synths and (random.random() < 0.6):
                acc_stem, _ = self._random_crop(random.choice(self.other_synths))
            if acc_stem is None:
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

        # Peak normalization (the cue keeps the target's scale)
        max_val = max(1.0, float(np.abs(mixture).max()))
        mixture = mixture / max_val
        target_clean = target_clean / max_val
        cue_pos = cue_pos / max_val

        # 5. Negative Audio Cue (mismatched)
        if self.use_cache and len(self.cached_synths) > 0:
            n_idx = random.randint(0, len(self.cached_synths) - 1)
            if target_stem is not None and int(self.synth_ids[n_idx]) == target_stem and len(self.cached_synths) > 1:
                n_idx = (n_idx + 1) % len(self.cached_synths)
            cue_neg = self.cached_synths[n_idx][:self.cue_samples].copy()
        elif self.other_synths:
            cue_neg, _ = self._random_crop(random.choice(self.other_synths), length=self.cue_samples)
            if cue_neg is None:
                cue_neg = np.zeros(self.cue_samples, dtype=np.float32)
        else:
            cue_neg = np.zeros(self.cue_samples, dtype=np.float32)

        return {
            "mix": torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0),
            "target": torch.from_numpy(np.asarray(target_clean, dtype=np.float32)).unsqueeze(0),
            "cue_pos": torch.from_numpy(np.asarray(cue_pos, dtype=np.float32)).unsqueeze(0),
            "cue_neg": torch.from_numpy(np.asarray(cue_neg, dtype=np.float32)).unsqueeze(0),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "cue_disjoint": torch.tensor(cue_disjoint),
        }


if __name__ == "__main__":
    ds = MusicalDatasetV8(split="train", epoch_length=10)
    print(f"MusicalDatasetV8 initialized. Total training kicks: {len(ds.kicks)}, drums: {len(ds.drums)}, bass: {len(ds.bass_stems)}, synths: {len(ds.other_synths)}")
    sample = ds[0]
    print(f"Sample shapes - Mix: {sample['mix'].shape}, Target: {sample['target'].shape}, Cue Pos: {sample['cue_pos'].shape}, SNR: {sample['snr_db'].item():+.1f} dB")
