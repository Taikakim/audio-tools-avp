#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 Dynamic Dataset: Generates 4.0s phrase audio with dynamic parameter automation.

Synthesizes diverse dynamic sweeps across the 4 automation lanes:
- Lane 0: Filter 1 Cutoff (Linear rising, falling, sinusoidal, exponential, static)
- Lane 1: Filter 1 Resonance
- Lane 2: Waveshaper Drive
- Lane 3: FM Depth
Renders audio via Surge XT block-by-block and extracts 128-band log-mel spectrograms.

Fixes (KUANG 2026-10-29, docs/synth-v8-code-review-2026-10-29.md in avp-audio-craft):
- C1: the synth is reset once per phrase, not once per 93 ms chunk. pedalboard's process() defaults to reset=True,
  which cleared every voice at each chunk boundary (notes cut off, chunks without a note-on silent). Chunks after the
  first now pass reset=False, and chunk lengths follow the plugin's actual output so the 43 chunks join without gaps.
- C2: the stored static vector is the one rendered: the filter type and balance the render sets, and the four lane
  parameters at their lane means. The lane parameters are left out of the static loss (STATIC_LOSS_MASK in
  dynamic_inverter_v8), and lanes the patch cannot hear (drive with the waveshaper off, FM depth with FM routing off)
  are masked through lane_active.
- C3: lanes are stored in the v6 vector's units. The drive lane maps to Surge's raw drive as the static vector does
  (raw = 0.5 + 0.5 * drive, 0 dB .. +24 dB); before, lane values were written as raw values, mostly below 0 dB.
- Train / val / test draw disjoint bank items; the test split is for the gate, val for checkpoint selection.
- Phrases vary pitch and velocity by default (phrase="varied"), so keytracking and velocity are in the data.
"""

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

LANE_NAMES = ("cutoff", "resonance", "drive", "fm_depth")
LANE_PLUGIN_PARAMS = ("a_filter_1_cutoff", "a_filter_1_resonance", "a_waveshaper_drive", "a_fm_depth")
LANE_V6_INDEX = tuple(S6.PARAM_INDEX[n] for n in LANE_NAMES)
SWEEP_TYPES = ("rising", "falling", "sinusoidal", "exponential", "static")
FILTER_CHOICES = (0, 1, 2, 4, 5)  # LP_FILTERS indices the sweeps use


def lane_to_raw(lane_idx: int, values) -> np.ndarray:
    """Lane values (v6 vector units) -> Surge raw values. Only drive differs: raw = 0.5 + 0.5 * drive."""
    v = np.asarray(values, dtype=np.float32)
    return (0.5 + 0.5 * v).astype(np.float32) if LANE_NAMES[lane_idx] == "drive" else v


def lane_active(patch: dict) -> np.ndarray:
    """[4] bool: whether each lane can be heard with this patch (drive needs a waveshaper, FM depth an FM routing)."""
    return np.array([True, True, int(patch["ws_idx"]) != 0, int(patch["fm_routing"]) != 0], dtype=bool)


def repeated_phrase(midi_note: int, step_s: float = 0.25, gate_s: float = 0.18, velocity: int = 105):
    """The original Phase 2 phrase: one note every step_s seconds. Notes are (note, velocity, start_s, dur_s)."""
    return [(int(midi_note), velocity, float(t), gate_s) for t in np.arange(0.0, DURATION_S, step_s)]


def varied_phrase(rng: random.Random, root: int, step_s: float = 0.25):
    """A phrase on the same grid with pitch (within +-7 semitones of root, kept in NOTE_LOW..NOTE_HIGH), velocity
    (70..120) and gate (0.10..0.22 s) varying per note, so keytracking and velocity show up in the audio."""
    notes = []
    for t in np.arange(0.0, DURATION_S, step_s):
        n = int(np.clip(root + rng.choice([-7, -5, 0, 0, 0, 3, 5, 7]), S6.NOTE_LOW, S6.NOTE_HIGH))
        notes.append((n, rng.randint(70, 120), float(t), rng.uniform(0.10, 0.22)))
    return notes


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
    notes: Optional[Sequence[Tuple[int, int, float, float]]] = None,
) -> np.ndarray:
    """Renders 4.0s audio chunk-by-chunk automating the 4 dynamic lanes at 10.76 Hz.

    Lanes are in v6 vector units (lane_to_raw converts). notes: (note, velocity, start_s, dur_s); default
    repeated_phrase(midi_note). The plugin is reset once; every later chunk continues the same voices (reset=False).
    """
    S5.apply_patch(plugin, patch)
    plugin.reset()

    if notes is None:
        notes = repeated_phrase(midi_note)
    all_events = []
    for note, vel, start, dur in notes:
        all_events.append(("note_on", int(note), int(vel), float(start)))
        all_events.append(("note_off", int(note), 0, float(start + dur)))
    all_events.sort(key=lambda e: (e[3], 0 if e[0] == "note_off" else 1))

    lanes = (cutoff_lane, res_lane, drive_lane, fm_lane)
    total = int(round(sample_rate * DURATION_S))
    bounds = np.round(np.linspace(0, total, NUM_FRAMES + 1)).astype(np.int64)

    chunks = []
    produced = 0  # samples the plugin has actually returned so far
    for t in range(NUM_FRAMES):
        for k, name in enumerate(LANE_PLUGIN_PARAMS):
            plugin.parameters[name].raw_value = float(lane_to_raw(k, lanes[k][t]))

        n_want = int(bounds[t + 1] - produced)
        if n_want <= 0:
            continue
        t0, t1 = produced / sample_rate, (produced + n_want) / sample_rate
        ev_frame = [mido.Message(etype, note=enote, velocity=evel, time=etime - t0)
                    for etype, enote, evel, etime in all_events if t0 <= etime < t1]

        # pedalboard truncates duration * sample_rate: a quarter-sample cushion yields exactly n_want samples
        audio_chunk = plugin.process(ev_frame, duration=(n_want + 0.25) / sample_rate, sample_rate=sample_rate,
                                     num_channels=2, reset=(t == 0))
        mono = np.asarray(audio_chunk, dtype=np.float32).mean(axis=0)
        chunks.append(mono)
        produced += len(mono)

    full_mono = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    if len(full_mono) < TOTAL_SAMPLES:
        full_mono = np.pad(full_mono, (0, TOTAL_SAMPLES - len(full_mono)))
    else:
        full_mono = full_mono[:TOTAL_SAMPLES]

    # Peak normalization
    peak = float(np.abs(full_mono).max())
    if peak > 1e-4:
        full_mono = full_mono / max(1.0, peak)

    return full_mono


def rendered_static_vector(base_vec: np.ndarray, filter_idx: int, lanes: np.ndarray) -> np.ndarray:
    """The v6 vector the phrase is rendered from: the base patch with filter 1 type set, filter balance 0 (filter 1
    only) and the four lane parameters at their lane means (they are masked out of the static loss)."""
    v = np.array(base_vec, dtype=np.float32, copy=True)
    v[S6.PARAM_INDEX["filter_type"]] = filter_idx / (len(S5.LP_FILTERS) - 1)
    v[S6.PARAM_INDEX["filter_balance"]] = 0.0
    for k, idx in enumerate(LANE_V6_INDEX):
        v[idx] = float(np.mean(lanes[k]))
    return v


class DynamicSweepDataset(Dataset):
    """Memory-mapped or pre-rendered bank of 4.0s dynamic filter sweeps."""

    def __init__(self, cache_file: Path):
        super().__init__()
        data = np.load(cache_file)
        self.mels = torch.from_numpy(data["mels"]).float()          # [N, 128, 401]
        self.statics = torch.from_numpy(data["statics"]).float()    # [N, 59]
        self.dynamics = torch.from_numpy(data["dynamics"]).float()  # [N, 4, 43]
        n = len(self.mels)
        # Caches built before the fixes have neither field: every lane counts as active, family unknown (-1)
        self.lane_active = torch.from_numpy(data["lane_active"]).bool() if "lane_active" in data \
            else torch.ones(n, len(LANE_NAMES), dtype=torch.bool)
        self.cutoff_family = torch.from_numpy(data["cutoff_family"]).long() if "cutoff_family" in data \
            else torch.full((n,), -1, dtype=torch.long)
        self.fixed_renderer = bool(data["fixed_renderer"]) if "fixed_renderer" in data else False

    def __len__(self) -> int:
        return len(self.mels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "mel": self.mels[idx],
            "target_static": self.statics[idx],
            "target_dynamic": self.dynamics[idx],
            "lane_active": self.lane_active[idx],
            "cutoff_family": self.cutoff_family[idx],
        }


def _split_indices(num_patches: int, counts: Sequence[int], seed: int) -> List[np.ndarray]:
    """Disjoint bank-item index sets for the splits (cycled with a warning when the shard is too small)."""
    perm = np.random.RandomState(seed).permutation(num_patches)
    need = int(sum(counts))
    if need > num_patches:
        print(f"Warning: {need} items requested from {num_patches} bank items; splits will share patches.")
        perm = np.resize(perm, need)
    out, start = [], 0
    for c in counts:
        out.append(perm[start:start + c])
        start += c
    return out


def build_dynamic_sweep_cache(
    out_dir: Path,
    num_train: int = 500,
    num_val: int = 50,
    num_test: int = 100,
    seed: int = 42,
    phrase: str = "varied",
    bank_dir: str = "/home/kim/sao_runs/bank_v6",
):
    """Pre-renders dynamic sweep train / val / test sets to NVMe (val: checkpoint selection; test: the gate)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {s: out_dir / f"dynamic_{s}.npz" for s in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()):
        print(f"Dynamic sweep caches already exist at {out_dir}. Skipping build (delete them to rebuild).")
        return

    print("=" * 80)
    print(f"Building Dynamic Sweep Caches ({num_train} train, {num_val} val, {num_test} test, phrase={phrase}) "
          f"to {out_dir}...")
    print("=" * 80)

    # 1. Load base patches from bank_v6
    meta = np.load(os.path.join(bank_dir, "main_00000_meta.npz"))
    base_vecs = meta["vec"]
    base_midis = meta["midi"]
    split_idx = _split_indices(len(base_vecs), (num_train, num_val, num_test), seed)

    # 2. Init Synth & Mel Processor
    plugin = S5.init_synth()
    mel_fn = ExactGpuMel(device="cpu")

    for s_i, (split_name, items) in enumerate(zip(("train", "val", "test"), split_idx)):
        rng = random.Random(seed + 1 + s_i)
        random.seed(seed + 1 + s_i)  # generate_sweep_curve draws from the module RNG
        t0 = time.time()
        mels, statics, dynamics, actives, families = [], [], [], [], []

        print(f"Generating {len(items)} {split_name} dynamic sweeps...")
        for i, p_idx in enumerate(items):
            v6_vec = base_vecs[int(p_idx)].astype(np.float32)
            midi_note = int(base_midis[int(p_idx)])

            # Generate dynamic trajectories (v6 vector units)
            stype = rng.choice(SWEEP_TYPES)
            cutoff_lane = generate_sweep_curve(stype, NUM_FRAMES, low=0.10, high=0.90)
            res_lane = generate_sweep_curve(rng.choice(["static", "sinusoidal"]), NUM_FRAMES, low=0.10, high=0.70)
            drive_lane = generate_sweep_curve(rng.choice(["static", "rising"]), NUM_FRAMES, low=0.0, high=0.60)
            fm_lane = generate_sweep_curve(rng.choice(["static", "falling"]), NUM_FRAMES, low=0.0, high=0.40)
            lanes = np.stack([cutoff_lane, res_lane, drive_lane, fm_lane], axis=0)  # [4, 43]

            # Filter 1 as an active lowpass; the stored vector is the one rendered
            filter_idx = rng.choice(FILTER_CHOICES)
            vec = rendered_static_vector(v6_vec, filter_idx, lanes)
            patch = S6.vector_to_patch(vec)

            notes = varied_phrase(rng, midi_note) if phrase == "varied" else repeated_phrase(midi_note)
            wav = render_dynamic_patch(plugin, patch, midi_note, *lanes, notes=notes)

            with torch.no_grad():
                mel = mel_fn(torch.from_numpy(wav)).numpy()  # [128, 401]

            mels.append(mel)
            statics.append(vec)
            dynamics.append(lanes.astype(np.float32))
            actives.append(lane_active(patch))
            families.append(SWEEP_TYPES.index(stype))

            if (i + 1) % 100 == 0 or (i + 1) == len(items):
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{i+1:03d}/{len(items)}] in {time.time()-t0:.1f}s ({rate:.1f} renders/s)")

        np.savez_compressed(
            paths[split_name],
            mels=np.stack(mels),
            statics=np.stack(statics),
            dynamics=np.stack(dynamics),
            lane_active=np.stack(actives),
            cutoff_family=np.array(families, dtype=np.int64),
            bank_items=np.asarray(items, dtype=np.int64),
            fixed_renderer=np.array(True),
        )
        print(f"Saved {paths[split_name]} ({paths[split_name].stat().st_size / (1024**2):.1f} MB)")

    print("Dynamic sweep cache build complete!")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the V8 Phase 2 dynamic sweep caches")
    ap.add_argument("--out-dir", default="/home/kim/sao_runs/v8_dynamic_cache_v2",
                    help="a new directory: caches from before the renderer fix must not be mixed in")
    ap.add_argument("--num-train", type=int, default=500)
    ap.add_argument("--num-val", type=int, default=50)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--phrase", choices=["varied", "repeated"], default="varied")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build_dynamic_sweep_cache(Path(a.out_dir), a.num_train, a.num_val, a.num_test, a.seed, a.phrase)
