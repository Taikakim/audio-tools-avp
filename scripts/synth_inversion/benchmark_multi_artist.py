#!/usr/bin/env python3
"""Multi-artist rolling phrase inversion benchmark.

Tests Surge XT filter circuits against extracted bass stems from:
1. Astral Projection - Mahadeva (F2, ~140 BPM)
2. MFG - Shape the Future (A1, 143.5 BPM)
3. Cosmosis - Alien Disco (D#1, ~140 BPM)

Saves matched audio, comparison plots, and Steinberg .vstpreset files
directly into ~/Documents/Surge XT/Patches/AI Inversions/.
"""

import os
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
from scipy.optimize import differential_evolution

sys.path.append("/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from surge_spec import init_synth, apply_patch, LP_FILTERS
from audio_utils import save_patch, MultiScaleSTFTLoss
from envelope_extractor import compute_bass_match_loss, lowpass_300hz, zolzer_envelope_follower

DAW_DIR = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")
os.makedirs(DAW_DIR, exist_ok=True)
BENCH_DIR = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/multi_artist_benchmarks"
os.makedirs(BENCH_DIR, exist_ok=True)

TARGETS = [
    {
        "artist": "Astral Projection",
        "track": "Mahadeva",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/astral_mahadeva/bass.wav",
        "root_midi": 41,  # F2
        "bpm": 140.0,
        "bars": 2,
        "prefix": "astral_mahadeva",
    },
    {
        "artist": "MFG",
        "track": "Shape the Future",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/mfg_shape_future/bass.wav",
        "root_midi": 33,  # A1
        "bpm": 143.55,
        "bars": 2,
        "prefix": "mfg_shape_future",
    },
    {
        "artist": "Cosmosis",
        "track": "Alien Disco",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/cosmosis_alien_disco/bass.wav",
        "root_midi": 39,  # D#2 / D#1 register
        "bpm": 140.0,
        "bars": 2,
        "prefix": "cosmosis_alien_disco",
    },
]

SEARCH_SPACE = [
    ("shape", 0.0, 1.0),
    ("width", 0.0, 1.0),
    ("sub_mix", 0.0, 1.0),
    ("cutoff", 0.1, 0.95),
    ("resonance", 0.0, 0.70),
    ("feg_amount", 0.0, 0.8),
    ("feg_decay", 0.05, 0.6),
    ("aeg_decay", 0.1, 0.8),
    ("aeg_release", 0.01, 0.3),
    ("drive_raw", 0.50, 0.75),
]

BASE_PATCH = dict(
    filter_idx=0,
    shape=0.35,
    width=0.50,
    sub_mix=0.40,
    sync=0.0,
    fm_depth=0.0,
    unison=False,
    unison_detune=0.0,
    cutoff=0.65,
    resonance=0.20,
    keytrack_raw=1.0,
    feg_amount=0.30,
    feg_decay=0.25,
    feg_sustain=0.0,
    aeg_decay=0.40,
    aeg_sustain=0.0,
    aeg_release=0.05,
    ws_idx=0,
    drive_raw=0.55,
    chorus_mix=0.0,
    delay_mix=0.0,
    delay_fb=0.0,
)


def build_rolling_midi(root_note: int, bpm: float, bars: int = 2):
    beat_dur = 60.0 / bpm
    sixteenth_dur = beat_dur / 4.0
    phrase_dur = bars * 4.0 * beat_dur
    num_steps = bars * 16

    events = []
    t = 0.0
    for step in range(num_steps):
        beat_step = step % 4
        is_bass = beat_step in (1, 2, 3)
        note_len = sixteenth_dur * 0.85
        if is_bass:
            events.append(mido.Message("note_on", note=root_note, velocity=105, time=t))
            events.append(mido.Message("note_off", note=root_note, velocity=0, time=t + note_len))
        t += sixteenth_dur
    return events, phrase_dur


def invert_track(target: dict):
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {target['artist']} - {target['track']}")
    print("=" * 70)

    sr = 44100
    events, phrase_dur = build_rolling_midi(target["root_midi"], target["bpm"], target["bars"])

    y_raw, _ = librosa.load(target["stem_path"], sr=sr, mono=True, duration=phrase_dur)
    y_target = y_raw.astype(np.float32)
    y_target = y_target / (np.max(np.abs(y_target)) + 1e-7)

    synth = init_synth(sample_rate=sr)
    synth.process(np.zeros((2, 1024), dtype=np.float32), 1024 / sr, sr, 2)

    loss_stft = MultiScaleSTFTLoss()
    filter_candidates = [0, 1, 4, 7]  # LP 12dB, LP 24dB, OB-Xd 12dB, Diode Ladder

    bounds = [(lo, hi) for _, lo, hi in SEARCH_SPACE]
    param_names = [name for name, _, _ in SEARCH_SPACE]

    best_loss = float("inf")
    best_patch = None
    best_audio = None
    best_filter_name = None

    for f_idx in filter_candidates:
        f_name = LP_FILTERS[f_idx][0]
        print(f"Evaluating filter: {f_name}...")

        def objective(x):
            p = dict(BASE_PATCH)
            p["midi_note"] = target["root_midi"]
            p["filter_idx"] = f_idx
            for k, val in zip(param_names, x):
                p[k] = float(val)

            apply_patch(synth, p)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            mono = mono / (np.max(np.abs(mono)) + 1e-7)

            l_stft = float(loss_stft(mono, y_target))
            metrics = compute_bass_match_loss(y_target, mono, fs=sr, cutoff_hz=300.0)
            return l_stft + 0.35 * metrics["loss_bass"]

        res = differential_evolution(objective, bounds, maxiter=8, popsize=6, seed=42, workers=1)
        print(f"  {f_name:20s} -> Loss: {res.fun:.3f}")

        if res.fun < best_loss:
            best_loss = res.fun
            best_filter_name = f_name
            p_opt = dict(BASE_PATCH)
            p_opt["midi_note"] = target["root_midi"]
            p_opt["filter_idx"] = f_idx
            for k, val in zip(param_names, res.x):
                p_opt[k] = float(val)
            best_patch = p_opt

            apply_patch(synth, p_opt)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            best_audio = mono / (np.max(np.abs(mono)) + 1e-7)

    stft_val = float(loss_stft(best_audio, y_target))
    bass_metrics = compute_bass_match_loss(y_target, best_audio, fs=sr, cutoff_hz=300.0)

    print(f"\nWINNER for {target['artist']}: {best_filter_name} (Loss: {best_loss:.3f})")
    print(f"  Cutoff: {best_patch['cutoff']:.3f}, Res: {best_patch['resonance']:.3f}, Shape: {best_patch['shape']:.3f}")
    print(f"  Target <300Hz Ratio: {bass_metrics['bass_ratio_target']*100:.1f}%, Synth: {bass_metrics['bass_ratio_synth']*100:.1f}%")

    out_prefix = f"{BENCH_DIR}/{target['prefix']}"
    sf.write(f"{out_prefix}_synth.wav", best_audio, sr)
    sf.write(f"{out_prefix}_target.wav", y_target, sr)

    save_patch(
        synth,
        best_patch,
        out_prefix,
        extra={
            "artist": target["artist"],
            "track": target["track"],
            "bpm": target["bpm"],
            "bars": target["bars"],
            "stft_loss": stft_val,
            "bass_metrics": bass_metrics,
        },
        copy_to_user_dir=True,
    )

    daw_preset_path = f"{DAW_DIR}/{target['prefix']}_inversion.vstpreset"
    if os.path.exists(f"{out_prefix}.vstpreset"):
        shutil.copy(f"{out_prefix}.vstpreset", daw_preset_path)

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    t_axis = np.linspace(0, phrase_dur, len(y_target))

    axs[0].plot(t_axis, y_target, label=f"Real Stem ({target['artist']} - {target['track']})", color="royalblue", alpha=0.75)
    axs[0].plot(t_axis, best_audio, label=f"Surge XT Match ({best_filter_name})", color="crimson", alpha=0.75)
    axs[0].set_title(f"{target['artist']} - {target['track']} ({target['bars']} Bars, {target['bpm']:.1f} BPM, STFT Loss: {stft_val:.2f})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    lp_tgt = lowpass_300hz(y_target, fs=sr, cutoff_hz=300.0)
    lp_syn = lowpass_300hz(best_audio, fs=sr, cutoff_hz=300.0)
    env_tgt = zolzer_envelope_follower(lp_tgt, fs=sr)
    env_syn = zolzer_envelope_follower(lp_syn, fs=sr)

    tgt_r = bass_metrics["bass_ratio_target"] * 100.0
    syn_r = bass_metrics["bass_ratio_synth"] * 100.0
    axs[1].plot(t_axis, env_tgt, label=f"Target Bass Env (<300Hz, Ratio: {tgt_r:.1f}%)", color="blue", linewidth=1.5)
    axs[1].plot(t_axis, env_syn, label=f"Synth Bass Env (<300Hz, Ratio: {syn_r:.1f}%)", color="darkgreen", linewidth=1.5, linestyle="--")
    axs[1].set_title(f"Dynamic Bass Envelope Alignment (MSE: {bass_metrics['bass_env_mse']:.6f})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Level")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    S_syn = librosa.amplitude_to_db(np.abs(librosa.stft(best_audio, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(S_syn, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[2], cmap="magma", vmin=-50, vmax=0)
    axs[2].set_ylim(0, 4000)
    axs[2].set_title(f"Synth Spectrogram ({best_filter_name}, Cutoff: {best_patch['cutoff']:.3f}, Res: {best_patch['resonance']:.3f})")

    plt.tight_layout()
    plot_path = f"{out_prefix}_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")
    print(f"Saved DAW preset to: {daw_preset_path}")


def main():
    for target in TARGETS:
        invert_track(target)
    print("\n" + "=" * 70)
    print("ALL MULTI-ARTIST BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
