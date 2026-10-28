#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Pre-Flight Check R1: Contested-Bin Census on the AVP Multitrack Archive.

Measures where the extraction challenge actually lives:
1. Kick vs Bass/Acid/Lead energy within +-30 ms of kick onsets in:
   - Sub band: < 120 Hz
   - Body band: 120 - 2500 Hz
2. Masking across the whole song in the body band (120 - 2500 Hz):
   - Percentage of bass STFT cells where another synth (acid, lead, pad) is within 6 dB.
3. Groups and reports results by subgenre (Old-School Goa, Modern Psytrance, Acid, Other).
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

ARCHIVE_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "/run/media/kim/Mantu/avp-stems-original-classified"))
SR = 44100
MAX_ANALYSIS_DUR_S = 60.0  # Analyze representative 60-second slice per track to run fast on CPU
ANALYSIS_OFFSET_S = 45.0   # Skip intro, drop straight into main groove


def classify_subgenre(song_name: str) -> str:
    lower = song_name.lower()
    if "goa" in lower:
        return "Old-School Goa"
    elif "acid" in lower:
        return "Acid Trance"
    elif "psy" in lower:
        return "Modern Psytrance"
    elif "suomisoundi" in lower:
        return "Suomisoundi"
    elif "techno" in lower:
        return "Techno"
    else:
        return "Melodic / Other"


def load_stem_slice(stem_path: Path, offset_s: float, dur_s: float, sr: int = SR) -> np.ndarray:
    if not stem_path.exists():
        return None
    try:
        info = sf.info(stem_path)
        start_frame = int(min(offset_s, max(0, info.duration - dur_s)) * info.samplerate)
        num_frames = int(min(dur_s, info.duration) * info.samplerate)
        data, file_sr = sf.read(stem_path, start=start_frame, frames=num_frames, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        return data
    except Exception as e:
        print(f"Warning: could not read {stem_path}: {e}", file=sys.stderr)
        return None


def bandpass_energy(y: np.ndarray, low_f: float, high_f: float, sr: int = SR) -> float:
    """Computes RMS energy in a frequency band using Butterworth SOS filter."""
    if y is None or len(y) == 0:
        return 0.0
    nyq = 0.5 * sr
    low = max(10.0, low_f) / nyq
    high = min(nyq - 100.0, high_f) / nyq
    if low >= high:
        return 0.0
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    filtered = sosfilt(sos, y)
    return float(np.sqrt(np.mean(filtered ** 2) + 1e-12))


def analyze_song(song_dir: Path) -> dict:
    song_name = song_dir.name
    genre = classify_subgenre(song_name)
    
    # Locate key stem downmixes
    kick_file = song_dir / "kick" / "KICK_downmix.flac"
    bass_file = song_dir / "bass" / "BASS_downmix.flac"
    acid_file = song_dir / "acid" / "ACID_downmix.flac"
    lead_file = song_dir / "leads" / "LEADS_downmix.flac"
    pad_file = song_dir / "pads" / "PADS_downmix.flac"
    
    if not kick_file.exists() or not bass_file.exists():
        return None

    kick = load_stem_slice(kick_file, ANALYSIS_OFFSET_S, MAX_ANALYSIS_DUR_S)
    bass = load_stem_slice(bass_file, ANALYSIS_OFFSET_S, MAX_ANALYSIS_DUR_S)
    acid = load_stem_slice(acid_file, ANALYSIS_OFFSET_S, MAX_ANALYSIS_DUR_S)
    lead = load_stem_slice(lead_file, ANALYSIS_OFFSET_S, MAX_ANALYSIS_DUR_S)
    pad = load_stem_slice(pad_file, ANALYSIS_OFFSET_S, MAX_ANALYSIS_DUR_S)
    
    if kick is None or bass is None:
        return None

    # Length align
    min_len = min(len(kick), len(bass))
    kick = kick[:min_len]
    bass = bass[:min_len]
    if acid is not None: acid = acid[:min_len]
    if lead is not None: lead = lead[:min_len]
    if pad is not None: pad = pad[:min_len]

    # Detect kick onsets from kick stem
    onset_frames = librosa.onset.onset_detect(
        y=kick, sr=SR, hop_length=256, backtrack=False, units="samples"
    )
    
    # Exclude onsets too close to borders
    half_win = int(0.030 * SR)  # 30 ms
    valid_onsets = [o for o in onset_frames if o >= half_win and o + half_win < min_len]
    if len(valid_onsets) < 8:
        return None

    # 1. Onset window energy analysis
    sub_kick_energies, sub_bass_energies = [], []
    body_kick_energies, body_bass_energies = [], []
    body_acid_energies, body_lead_energies = [], []

    for o in valid_onsets:
        w_slice = slice(o - half_win, o + half_win)
        k_win = kick[w_slice]
        b_win = bass[w_slice]

        sub_k = bandpass_energy(k_win, 20.0, 120.0)
        sub_b = bandpass_energy(b_win, 20.0, 120.0)
        body_k = bandpass_energy(k_win, 120.0, 2500.0)
        body_b = bandpass_energy(b_win, 120.0, 2500.0)

        sub_kick_energies.append(sub_k)
        sub_bass_energies.append(sub_b)
        body_kick_energies.append(body_k)
        body_bass_energies.append(body_b)

        if acid is not None:
            body_acid_energies.append(bandpass_energy(acid[w_slice], 120.0, 2500.0))
        if lead is not None:
            body_lead_energies.append(bandpass_energy(lead[w_slice], 120.0, 2500.0))

    # Average kick-frame metrics
    mean_sub_k = float(np.mean(sub_kick_energies))
    mean_sub_b = float(np.mean(sub_bass_energies))
    # Bass-to-Kick ratio in sub under the kick
    sub_bass_to_kick_ratio_db = 20.0 * np.log10((mean_sub_b + 1e-6) / (mean_sub_k + 1e-6))
    
    mean_body_k = float(np.mean(body_kick_energies))
    mean_body_b = float(np.mean(body_bass_energies))

    # 2. Body-band masking analysis across whole excerpt
    # STFT parameters: 2048 n_fft (~46 ms), hop 512 (~11.6 ms)
    n_fft = 2048
    hop_length = 512
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    body_mask = (freqs >= 120.0) & (freqs <= 2500.0)

    S_bass = np.abs(librosa.stft(bass, n_fft=n_fft, hop_length=hop_length))[body_mask, :]
    
    # Combined other synths in body band
    S_others = np.zeros_like(S_bass)
    for stem in [acid, lead, pad]:
        if stem is not None:
            S_stem = np.abs(librosa.stft(stem, n_fft=n_fft, hop_length=hop_length))[body_mask, :]
            S_others = np.maximum(S_others, S_stem)

    # Active bass cells (above -40 dB relative to bass peak)
    bass_thresh = S_bass.max() * 0.01
    active_bass_cells = S_bass > bass_thresh
    num_active = int(np.sum(active_bass_cells))

    if num_active > 0:
        # Masked condition: other synth is within 6 dB of bass (S_others / S_bass >= 0.5)
        masked_cells = active_bass_cells & (S_others / (S_bass + 1e-6) >= 0.5)
        masking_pct = float(np.sum(masked_cells) / num_active * 100.0)
    else:
        masking_pct = 0.0

    return {
        "song": song_name,
        "genre": genre,
        "num_kicks_analyzed": len(valid_onsets),
        "sub_bass_to_kick_db": round(sub_bass_to_kick_ratio_db, 2),
        "mean_sub_kick_rms": round(mean_sub_k, 5),
        "mean_sub_bass_rms": round(mean_sub_b, 5),
        "mean_body_bass_rms": round(mean_body_b, 5),
        "body_masking_pct": round(masking_pct, 1),
    }


def main():
    print("=" * 80)
    print("Pre-Flight Check R1: Contested-Bin Census on AVP Archive")
    print(f"Archive Root: {ARCHIVE_ROOT}")
    print("=" * 80)

    if not ARCHIVE_ROOT.exists():
        print(f"Error: Archive path {ARCHIVE_ROOT} does not exist!", file=sys.stderr)
        sys.exit(1)

    song_dirs = [p for p in sorted(ARCHIVE_ROOT.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    print(f"Found {len(song_dirs)} candidate song folders. Running acoustic census...")

    results = []
    for d in song_dirs:
        res = analyze_song(d)
        if res:
            results.append(res)
            print(f"[{res['genre']}] {d.name[:35]:35s} | Sub Bass/Kick: {res['sub_bass_to_kick_db']:+6.1f} dB | Body Masking: {res['body_masking_pct']:4.1f}%")

    print("\n" + "=" * 80)
    print("CENSUS SUMMARY BY SUBGENRE")
    print("=" * 80)

    genres = {}
    for r in results:
        g = r["genre"]
        if g not in genres:
            genres[g] = []
        genres[g].append(r)

    print(f"{'Subgenre':<22} | {'Songs':<5} | {'Sub Bass/Kick (dB)':<20} | {'Body Synth Masking (%)':<22}")
    print("-" * 80)

    for g, song_res in sorted(genres.items()):
        sub_dbs = [s["sub_bass_to_kick_db"] for s in song_res]
        mask_pcts = [s["body_masking_pct"] for s in song_res]
        mean_sub = np.mean(sub_dbs)
        mean_mask = np.mean(mask_pcts)
        print(f"{g:<22} | {len(song_res):<5} | {mean_sub:+6.2f} dB (std {np.std(sub_dbs):4.1f})   | {mean_mask:5.1f}% (std {np.std(mask_pcts):4.1f})")

    overall_sub = np.mean([r["sub_bass_to_kick_db"] for r in results])
    overall_mask = np.mean([r["body_masking_pct"] for r in results])
    print("-" * 80)
    print(f"{'OVERALL ARCHIVE':<22} | {len(results):<5} | {overall_sub:+6.2f} dB               | {overall_mask:5.1f}%")
    print("=" * 80)

    # Save artifact
    out_dir = Path("/home/kim/Projects/SAO/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "r1_contested_bin_census_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
