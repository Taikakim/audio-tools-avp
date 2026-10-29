#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Pre-extracts and caches 4.0s audio slices onto fast NVMe.

Eliminates the 535 ms USB FLAC seek bottleneck by saving memory-mapped
NumPy banks for instant O(1) sampling during training.
"""

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


def extract_clips_from_stem(stem_path: Path) -> list:
    clips = []
    try:
        info = sf.info(stem_path)
        total_frames = info.frames
        if total_frames < CLIP_SAMPLES:
            return clips
        
        # Read up to 60s of active audio
        read_frames = min(total_frames, int(60.0 * info.samplerate))
        start_frame = int(min(30.0 * info.samplerate, max(0, total_frames - read_frames)))
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
            if len(clips) >= MAX_CLIPS_PER_STEM:
                break
    except Exception as e:
        pass
    return clips


def main():
    print("=" * 80)
    print("Building Fast NVMe Stem Cache for V8")
    print(f"Source: {ARCHIVE_ROOT} --> Destination: {CACHE_DIR}")
    print("=" * 80)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    categories = {
        "kicks": [],
        "drums": [],
        "bass": [],
        "synths": [],
    }

    start_t = time.time()
    song_dirs = [d for d in sorted(ARCHIVE_ROOT.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    train_songs = [d for d in song_dirs if d.name not in HOLDOUT_SONGS]
    print(f"Processing {len(train_songs)} training songs (holding out {len(HOLDOUT_SONGS)})...")

    for sdir in train_songs:
        # Kicks
        kdir = sdir / "kick"
        if kdir.exists():
            for p in list(kdir.glob("*.flac")) + list(kdir.glob("*.wav")):
                categories["kicks"].extend(extract_clips_from_stem(p))

        # Drums
        ddir = sdir / "drums"
        if ddir.exists():
            for p in list(ddir.glob("*.flac")) + list(ddir.glob("*.wav")):
                categories["drums"].extend(extract_clips_from_stem(p))

        # Bass
        bdir = sdir / "bass"
        if bdir.exists():
            for p in list(bdir.glob("*.flac")) + list(bdir.glob("*.wav")):
                categories["bass"].extend(extract_clips_from_stem(p))

        # Synths
        for cat in ["acid", "leads", "pads", "arp_keys"]:
            cdir = sdir / cat
            if cdir.exists():
                for p in list(cdir.glob("*.flac")) + list(cdir.glob("*.wav")):
                    categories["synths"].extend(extract_clips_from_stem(p))

    # Save to memory-mapped npy files
    for cat_name, clip_list in categories.items():
        if clip_list:
            arr = np.stack(clip_list, axis=0)  # [N, 176400]
            out_file = CACHE_DIR / f"{cat_name}.npy"
            np.save(out_file, arr)
            size_mb = os.path.getsize(out_file) / (1024**2)
            print(f"Saved {cat_name:<8}: {arr.shape[0]:4d} clips ({size_mb:6.1f} MB) -> {out_file}")

    dur = time.time() - start_t
    print("=" * 80)
    print(f"NVMe Stem Cache successfully built in {dur:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
