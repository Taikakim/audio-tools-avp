#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 Dynamic Dataset: Generates 4.0s phrase audio with dynamic parameter automation.

Synthesizes diverse dynamic sweeps across the 4 automation lanes:
- Lane 0: Filter 1 Cutoff (Linear rising, falling, sinusoidal, exponential, static)
- Lane 1: Filter 1 Resonance
- Lane 2: Waveshaper Drive
- Lane 3: FM Depth
Renders audio via Surge XT block-by-block and extracts 128-band log-mel spectrograms.
"""

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import ExactGpuMel
import surge_spec_v5 as S5
import surge_spec_v6 as S6

SAMPLE_RATE = 44100
DURATION_S = 4.0
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION_S)  # 176400
NUM_FRAMES = 43  # 10.76 Hz (92.9 ms per frame)
FRAME_DUR = DURATION_S / NUM_FRAMES
FRAME_SAMPLES = int(round(SAMPLE_RATE * FRAME_DUR))


def generate_sweep_curve(sweep_type: str, num_frames: int = 43, low: float = 0.15, high: float = 0.85) -> np.ndarray:
    """Generates smooth automation trajectories for a single lane."""
    t = np.linspace(0.0, 1.0, num_frames)
    if sweep_type == "rising":
        curve = low + (high - low) * t
    elif sweep_type == "falling":
        curve = high - (high - low) * t
    elif sweep_type == "sinusoidal":
        cycles = random.choice([0.5, 1.0, 2.0])
        mid = (low + high) / 2.0
        amp = (high - low) / 2.0
        curve = mid + amp * np.sin(2.0 * np.pi * cycles * t + random.uniform(0, np.pi))
    elif sweep_type == "exponential":
        # Snappy exponential rise or decay
        if random.random() < 0.5:
            curve = low + (high - low) * (np.exp(3.0 * t) - 1.0) / (np.exp(3.0) - 1.0)
        else:
            curve = high - (high - low) * (np.exp(3.0 * t) - 1.0) / (np.exp(3.0) - 1.0)
    else:  # static
        val = random.uniform(low, high)
        curve = np.full(num_frames, val, dtype=np.float32)

    return np.clip(curve, 0.0, 1.0).astype(np.float32)


def render_dynamic_patch(
    plugin,
    patch: dict,
    midi_note: int,
    cutoff_lane: np.ndarray,
    res_lane: np.ndarray,
    drive_lane: np.ndarray,
    fm_lane: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Renders 4.0s audio chunk-by-chunk automating the 4 dynamic lanes at 10.76 Hz."""
    S5.apply_patch(plugin, patch)
    plugin.reset()

    # Create a 16-note bassline phrase across 4.0s (1 note every 0.25s)
    note_times = np.arange(0.0, DURATION_S, 0.25)
    all_events = []
    for nt in note_times:
        all_events.append(("note_on", midi_note, 105, nt))
        all_events.append(("note_off", midi_note, 0, nt + 0.18))

    chunks = []
    for t in range(NUM_FRAMES):
        t_start = t * FRAME_DUR
        t_end = (t + 1) * FRAME_DUR

        # Automate dynamic parameters
        plugin.parameters["a_filter_1_cutoff"].raw_value = float(cutoff_lane[t])
        plugin.parameters["a_filter_1_resonance"].raw_value = float(res_lane[t])
        plugin.parameters["a_waveshaper_drive"].raw_value = float(drive_lane[t])
        plugin.parameters["a_fm_depth"].raw_value = float(fm_lane[t])

        # Gather MIDI events within this frame
        ev_frame = []
        for etype, enote, evel, etime in all_events:
            if t_start <= etime < t_end:
                rel_time = etime - t_start
                ev_frame.append(mido.Message(etype, note=enote, velocity=evel, time=rel_time))

        audio_chunk = plugin.process(ev_frame, duration=FRAME_DUR, sample_rate=sample_rate, num_channels=2)
        chunks.append(audio_chunk.mean(axis=0))

    full_mono = np.concatenate(chunks).astype(np.float32)
    if len(full_mono) < TOTAL_SAMPLES:
        full_mono = np.pad(full_mono, (0, TOTAL_SAMPLES - len(full_mono)))
    else:
        full_mono = full_mono[:TOTAL_SAMPLES]

    # Peak normalization
    peak = float(np.abs(full_mono).max())
    if peak > 1e-4:
        full_mono = full_mono / max(1.0, peak)

    return full_mono


class DynamicSweepDataset(Dataset):
    """Memory-mapped or pre-rendered bank of 4.0s dynamic filter sweeps."""

    def __init__(self, cache_file: Path):
        super().__init__()
        data = np.load(cache_file)
        self.mels = torch.from_numpy(data["mels"]).float()          # [N, 128, 401]
        self.statics = torch.from_numpy(data["statics"]).float()    # [N, 59]
        self.dynamics = torch.from_numpy(data["dynamics"]).float()  # [N, 4, 43]

    def __len__(self) -> int:
        return len(self.mels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "mel": self.mels[idx],
            "target_static": self.statics[idx],
            "target_dynamic": self.dynamics[idx],
        }


def build_dynamic_sweep_cache(
    out_dir: Path,
    num_train: int = 500,
    num_val: int = 50,
    seed: int = 42,
):
    """Pre-renders dynamic sweep training and validation sets to NVMe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "dynamic_train.npz"
    val_path = out_dir / "dynamic_val.npz"
    if train_path.exists() and val_path.exists():
        print(f"Dynamic sweep caches already exist at {out_dir}. Skipping build.")
        return

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print(f"Building Dynamic Sweep Caches ({num_train} train, {num_val} val) to {out_dir}...")
    print("=" * 80)

    # 1. Load base patches from bank_v6
    bank_dir = "/home/kim/sao_runs/bank_v6"
    meta = np.load(os.path.join(bank_dir, "main_00000_meta.npz"))
    base_vecs = meta["vec"]
    base_midis = meta["midi"]
    num_patches = len(base_vecs)

    # 2. Init Synth & Mel Processor
    plugin = S5.init_synth()
    mel_fn = ExactGpuMel(device="cpu")

    sweep_types = ["rising", "falling", "sinusoidal", "exponential", "static"]

    for split_name, count, target_file in [("train", num_train, train_path), ("val", num_val, val_path)]:
        t0 = time.time()
        mels = []
        statics = []
        dynamics = []

        print(f"Generating {count} {split_name} dynamic sweeps...")
        for i in range(count):
            p_idx = random.randint(0, num_patches - 1)
            v6_vec = base_vecs[p_idx].copy()
            midi_note = int(base_midis[p_idx])

            # Convert to patch dict
            v5_vec = S6.to_v5(v6_vec)
            patch = S5.vector_to_patch(v5_vec)

            # Ensure filter 1 is active lowpass
            patch["filter_idx"] = random.choice([0, 1, 2, 4, 5])
            patch["filter_balance"] = 0.0

            # Generate dynamic trajectories
            stype = random.choice(sweep_types)
            cutoff_lane = generate_sweep_curve(stype, NUM_FRAMES, low=0.10, high=0.90)
            res_lane = generate_sweep_curve(random.choice(["static", "sinusoidal"]), NUM_FRAMES, low=0.10, high=0.70)
            drive_lane = generate_sweep_curve(random.choice(["static", "rising"]), NUM_FRAMES, low=0.0, high=0.60)
            fm_lane = generate_sweep_curve(random.choice(["static", "falling"]), NUM_FRAMES, low=0.0, high=0.40)

            lanes = np.stack([cutoff_lane, res_lane, drive_lane, fm_lane], axis=0)  # [4, 43]

            # Render 4.0s phrase
            wav = render_dynamic_patch(plugin, patch, midi_note, cutoff_lane, res_lane, drive_lane, fm_lane)

            # Compute Mel spectrogram
            with torch.no_grad():
                mel = mel_fn(torch.from_numpy(wav)).numpy()  # [128, 401]

            mels.append(mel)
            statics.append(v6_vec.astype(np.float32))
            dynamics.append(lanes.astype(np.float32))

            if (i + 1) % 100 == 0 or (i + 1) == count:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{i+1:03d}/{count}] in {time.time()-t0:.1f}s ({rate:.1f} renders/s)")

        np.savez_compressed(
            target_file,
            mels=np.stack(mels),
            statics=np.stack(statics),
            dynamics=np.stack(dynamics),
        )
        print(f"Saved {target_file} ({target_file.stat().st_size / (1024**2):.1f} MB)")

    print("Dynamic sweep cache build complete!")


if __name__ == "__main__":
    cache_dir = Path("/home/kim/sao_runs/v8_dynamic_cache")
    build_dynamic_sweep_cache(cache_dir, num_train=500, num_val=50)
