#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Pre-extracts and caches 4.0s audio slices onto fast NVMe.

Eliminates the 535 ms USB FLAC seek bottleneck by saving memory-mapped
NumPy banks for instant O(1) sampling during training.

KUANG 2026-10-29 (docs/synth-v8-code-review-2026-10-29.md, C4 / M3):
- Every category also gets <prefix><cat>_stem_ids.npy: one id per clip naming the stem file it came from, so the
  dataset can take the cue from another clip of the same stem instead of from the target window.
- --split val builds the val_*.npy caches of the six held-out songs (before, nothing in the repo built them).
- --max-clips / --start-s / --read-s set how much of each stem is cached (defaults as before: 8 clips from 60 s
  starting at 30 s).
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

ARCHIVE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "/run/media/kim/Mantu/avp-stems-original-classified"))
CACHE_DIR = Path("/home/kim/sao_runs/v8_stem_cache")
SR = 44100
CLIP_DUR_S = 4.0
CLIP_SAMPLES = int(SR * CLIP_DUR_S)  # 176,400 samples
MAX_CLIPS_PER_STEM = 8

HOLDOUT_SONGS = {
    "Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance",
    "Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance",
    "Aavepyörä - Dance of Tomorrow driving rhythmic hypnotic  goatrance",
    "Aavepyörä - Commodore Shiva impulse tracker soft adventure trance",
    "Summamutikka - Lohikäärmelaulu - furious goa trance",
    "Summamutikka - Naidenie Flowers furious oldschool goa trance",
}

CATEGORY_DIRS = {
    "kicks": ["kick"],
    "drums": ["drums"],
    "bass": ["bass"],
    "synths": ["acid", "leads", "pads", "arp_keys"],
}


def extract_clips_from_stem(stem_path: Path, max_clips: int = MAX_CLIPS_PER_STEM, start_s: float = 30.0,
                            read_s: float = 60.0) -> list:
    clips = []
    try:
        info = sf.info(stem_path)
        total_frames = info.frames
        if total_frames < CLIP_SAMPLES:
            return clips

        # Read up to read_s of active audio
        read_frames = min(total_frames, int(read_s * info.samplerate))
        start_frame = int(min(start_s * info.samplerate, max(0, total_frames - read_frames)))
        data, file_sr = sf.read(stem_path, start=start_frame, frames=read_frames, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != SR:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=SR)

        step = CLIP_SAMPLES
        for s in range(0, len(data) - CLIP_SAMPLES + 1, step):
            chunk = data[s : s + CLIP_SAMPLES]
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms > 0.005:  # skip near-silence
                clips.append(chunk.astype(np.float32))
            if len(clips) >= max_clips:
                break
    except Exception as e:
        print(f"Warning: could not read {stem_path}: {e}", file=sys.stderr)
    return clips


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache 4 s stem clips (with stem ids) for the V8 dataset")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--archive-root", default=str(ARCHIVE_ROOT))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--max-clips", type=int, default=MAX_CLIPS_PER_STEM)
    ap.add_argument("--start-s", type=float, default=30.0)
    ap.add_argument("--read-s", type=float, default=60.0)
    a = ap.parse_args(argv)

    archive_root, cache_dir = Path(a.archive_root), Path(a.cache_dir)
    prefix = "val_" if a.split == "val" else ""
    print("=" * 80)
    print(f"Building Fast NVMe Stem Cache for V8 ({a.split})")
    print(f"Source: {archive_root} --> Destination: {cache_dir}")
    print("=" * 80)

    cache_dir.mkdir(parents=True, exist_ok=True)

    clips = {c: [] for c in CATEGORY_DIRS}
    ids = {c: [] for c in CATEGORY_DIRS}
    stem_names = []

    start_t = time.time()
    song_dirs = [d for d in sorted(archive_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if a.split == "val":
        songs = [d for d in song_dirs if d.name in HOLDOUT_SONGS]
    else:
        songs = [d for d in song_dirs if d.name not in HOLDOUT_SONGS]
    print(f"Processing {len(songs)} {a.split} songs (held-out list: {len(HOLDOUT_SONGS)} songs)...")

    for sdir in songs:
        for cat, subdirs in CATEGORY_DIRS.items():
            for sub in subdirs:
                cdir = sdir / sub
                if not cdir.exists():
                    continue
                for p in sorted(list(cdir.glob("*.flac")) + list(cdir.glob("*.wav"))):
                    got = extract_clips_from_stem(p, a.max_clips, a.start_s, a.read_s)
                    if not got:
                        continue
                    stem_id = len(stem_names)
                    stem_names.append(f"{sdir.name}/{sub}/{p.name}")
                    clips[cat].extend(got)
                    ids[cat].extend([stem_id] * len(got))

    # Save to memory-mapped npy files
    for cat_name, clip_list in clips.items():
        if clip_list:
            arr = np.stack(clip_list, axis=0)  # [N, 176400]
            out_file = cache_dir / f"{prefix}{cat_name}.npy"
            np.save(out_file, arr)
            np.save(cache_dir / f"{prefix}{cat_name}_stem_ids.npy", np.asarray(ids[cat_name], dtype=np.int64))
            size_mb = os.path.getsize(out_file) / (1024**2)
            n_multi = sum(1 for s in set(ids[cat_name]) if ids[cat_name].count(s) > 1)
            print(f"Saved {prefix}{cat_name:<8}: {arr.shape[0]:4d} clips from {len(set(ids[cat_name]))} stems "
                  f"({n_multi} with 2+ clips, usable as cued targets) ({size_mb:6.1f} MB) -> {out_file}")
    (cache_dir / f"{prefix}stem_names.txt").write_text("\n".join(stem_names), encoding="utf-8")

    dur = time.time() - start_t
    print("=" * 80)
    print(f"NVMe Stem Cache successfully built in {dur:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
