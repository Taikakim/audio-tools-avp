#!/usr/bin/env python3
"""Warm Log-Mel & Envelope-Calibrated Rolling Bass Inversion.

Fixes the bright / thin sound syndrome:
1. Logarithmic Mel-Scale STFT (80 bands, 30 Hz - 3000 Hz) emphasizing sub-bass (<300 Hz).
2. Explicit Log Spectral Centroid penalty to lock the harmonic warmth to the reference track.
3. Realistic Cutoff Bounds: clamps cutoff raw_value in [0.20, 0.60] (60 Hz to 1250 Hz).
4. Octave-aware 16th note rolling patterns extracted from the stems.
5. Strict zero-FX dry pipeline with native .vstpreset export.
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
import torch
from scipy.optimize import differential_evolution

sys.path.append("/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from surge_spec import init_synth, apply_patch, LP_FILTERS
from audio_utils import save_patch
from envelope_extractor import compute_bass_match_loss, lowpass_300hz, zolzer_envelope_follower

DAW_DIR = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")
os.makedirs(DAW_DIR, exist_ok=True)
BENCH_DIR = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/multi_artist_benchmarks"
os.makedirs(BENCH_DIR, exist_ok=True)


class WarmMelSTFTLoss:
    """Log-Mel STFT Loss with sub-bass emphasis and log-centroid warmth penalty."""
    def __init__(self, sr=44100):
        self.sr = sr
        fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=30, fmax=3000, norm="slaney")
        self.fb = torch.from_numpy(fb).float()
        self.mel_freqs = librosa.mel_frequencies(n_mels=80, fmin=30, fmax=3000)
        
        # Sub-bass (<300 Hz) given 4.0x priority; high frequencies down-weighted
        self.weights = torch.ones(80)
        self.weights[self.mel_freqs < 300] = 4.0
        self.weights[self.mel_freqs > 1200] = 0.5
        
    def __call__(self, x_np, y_np):
        x = torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0)
        y = torch.as_tensor(y_np, dtype=torch.float32).unsqueeze(0)
        n = min(x.shape[1], y.shape[1])
        x, y = x[:, :n], y[:, :n]
        
        window = torch.hann_window(2048)
        X = torch.stft(x, n_fft=2048, hop_length=256, window=window, return_complex=True).abs() + 1e-7
        Y = torch.stft(y, n_fft=2048, hop_length=256, window=window, return_complex=True).abs() + 1e-7
        
        mel_X = torch.matmul(self.fb, X)
        mel_Y = torch.matmul(self.fb, Y)
        
        # Log-magnitude L1 weighted
        log_diff = torch.abs(torch.log(mel_Y) - torch.log(mel_X))
        weighted_mel_loss = (log_diff * self.weights.unsqueeze(0).unsqueeze(-1)).mean()
        
        # Spectral convergence
        sc_loss = torch.norm(mel_Y - mel_X, p="fro") / (torch.norm(mel_Y, p="fro") + 1e-7)
        
        # Log Spectral Centroid penalty: penalizes high-frequency brightness
        cent_x = np.mean(librosa.feature.spectral_centroid(y=x_np, sr=self.sr))
        cent_y = np.mean(librosa.feature.spectral_centroid(y=y_np, sr=self.sr))
        cent_loss = (np.log10(max(30.0, cent_x)) - np.log10(max(30.0, cent_y))) ** 2
        
        return float(weighted_mel_loss + 0.5 * sc_loss + 6.0 * cent_loss)


TARGETS = [
    {
        "artist": "Astral Projection",
        "track": "Mahadeva",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/astral_mahadeva/bass.wav",
        "bpm": 140.0,
        "bars": 2,
        "prefix": "astral_mahadeva_warm",
        # Octave-pumping F1/F2 Goa pattern
        "notes": [0, 41, 41, 0, 29, 41, 41, 0, 0, 41, 41, 29, 41, 41, 0, 0] * 2,
    },
    {
        "artist": "MFG",
        "track": "Shape the Future",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/mfg_shape_future/bass.wav",
        "bpm": 143.55,
        "bars": 2,
        "prefix": "mfg_shape_future_warm",
        # Deep sub-bass A1 pattern
        "notes": [33, 0, 34, 33, 33, 33, 0, 33, 33, 0, 0, 33, 33, 32, 0, 33] * 2,
    },
    {
        "artist": "Cosmosis",
        "track": "Alien Disco",
        "stem_path": f"{BENCH_DIR}/demucs_out/htdemucs/cosmosis_alien_disco/bass.wav",
        "bpm": 140.0,
        "bars": 2,
        "prefix": "cosmosis_alien_disco_warm",
        # Driving D1/D2 acid rolling groove
        "notes": [38, 0, 26, 26, 37, 38, 26, 26, 0, 0, 0, 37, 39, 27, 27, 38] * 2,
    },
]

# Strictly bounded search space: cutoff constrained to bass frequencies (max 1.2 kHz)
SEARCH_SPACE = [
    ("shape", 0.0, 0.60),        # Saw / soft square
    ("width", 0.3, 0.70),
    ("sub_mix", 0.40, 1.0),      # Sub oscillator prioritized for body
    ("cutoff", 0.20, 0.58),      # 61 Hz to 1100 Hz strictly! Prevents thin brightness
    ("resonance", 0.0, 0.55),
    ("feg_amount", 0.0, 0.50),   # Controlled filter envelope sweep
    ("feg_decay", 0.05, 0.35),
    ("aeg_decay", 0.15, 0.65),
    ("aeg_release", 0.02, 0.15),
    ("drive_raw", 0.50, 0.68),   # Mild saturation
]

BASE_PATCH = dict(
    filter_idx=0,
    shape=0.15,
    width=0.50,
    sub_mix=0.60,
    sync=0.0,
    fm_depth=0.0,
    unison=False,
    unison_detune=0.0,
    cutoff=0.38,                 # ~230 Hz warm default
    resonance=0.20,
    keytrack_raw=1.0,
    feg_amount=0.20,
    feg_decay=0.18,
    feg_sustain=0.0,
    aeg_decay=0.40,
    aeg_sustain=0.0,
    aeg_release=0.05,
    ws_idx=0,
    drive_raw=0.54,
    chorus_mix=0.0,
    delay_mix=0.0,
    delay_fb=0.0,
)


def build_phrase_midi(note_pattern, bpm: float):
    beat_dur = 60.0 / bpm
    sixteenth_dur = beat_dur / 4.0
    phrase_dur = len(note_pattern) * sixteenth_dur

    events = []
    t = 0.0
    for note in note_pattern:
        if note > 0:
            events.append(mido.Message("note_on", note=int(note), velocity=105, time=t))
            events.append(mido.Message("note_off", note=int(note), velocity=0, time=t + sixteenth_dur * 0.85))
        t += sixteenth_dur
    return events, phrase_dur


def invert_track_warm(target: dict):
    print("\n" + "=" * 70)
    print(f"WARM INVERSION: {target['artist']} - {target['track']}")
    print("=" * 70)

    sr = 44100
    events, phrase_dur = build_phrase_midi(target["notes"], target["bpm"])

    y_raw, _ = librosa.load(target["stem_path"], sr=sr, mono=True, duration=phrase_dur)
    y_target = y_raw.astype(np.float32)
    y_target = y_target / (np.max(np.abs(y_target)) + 1e-7)

    synth = init_synth(sample_rate=sr)
    synth.process(np.zeros((2, 1024), dtype=np.float32), 1024 / sr, sr, 2)

    loss_fn = WarmMelSTFTLoss(sr=sr)
    filter_candidates = [0, 1, 4, 7]  # LP 12dB, LP 24dB, OB-Xd 12dB, Diode Ladder

    bounds = [(lo, hi) for _, lo, hi in SEARCH_SPACE]
    param_names = [name for name, _, _ in SEARCH_SPACE]

    best_loss = float("inf")
    best_patch = None
    best_audio = None
    best_filter_name = None

    for f_idx in filter_candidates:
        f_name = LP_FILTERS[f_idx][0]
        print(f"Optimizing filter: {f_name}...")

        def objective(x):
            p = dict(BASE_PATCH)
            p["midi_note"] = target["notes"][1] if target["notes"][1] > 0 else 36
            p["filter_idx"] = f_idx
            for k, val in zip(param_names, x):
                p[k] = float(val)

            apply_patch(synth, p)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            mono = mono / (np.max(np.abs(mono)) + 1e-7)

            l_warm = loss_fn(mono, y_target)
            metrics = compute_bass_match_loss(y_target, mono, fs=sr, cutoff_hz=300.0)
            return l_warm + 0.40 * metrics["loss_bass"]

        res = differential_evolution(objective, bounds, maxiter=9, popsize=7, seed=42, workers=1)
        print(f"  {f_name:20s} -> Warm Loss: {res.fun:.3f}")

        if res.fun < best_loss:
            best_loss = res.fun
            best_filter_name = f_name
            p_opt = dict(BASE_PATCH)
            p_opt["midi_note"] = target["notes"][1] if target["notes"][1] > 0 else 36
            p_opt["filter_idx"] = f_idx
            for k, val in zip(param_names, res.x):
                p_opt[k] = float(val)
            best_patch = p_opt

            apply_patch(synth, p_opt)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            best_audio = mono / (np.max(np.abs(mono)) + 1e-7)

    # Detailed metrics
    bass_metrics = compute_bass_match_loss(y_target, best_audio, fs=sr, cutoff_hz=300.0)
    cent_tgt = np.mean(librosa.feature.spectral_centroid(y=y_target, sr=sr))
    cent_syn = np.mean(librosa.feature.spectral_centroid(y=best_audio, sr=sr))

    print(f"\nCHAMPION for {target['artist']}: {best_filter_name} (Loss: {best_loss:.3f})")
    print(f"  Cutoff: {best_patch['cutoff']:.3f}, Res: {best_patch['resonance']:.3f}, Sub: {best_patch['sub_mix']:.3f}")
    print(f"  Spectral Centroid: Target {cent_tgt:.1f} Hz vs Synth {cent_syn:.1f} Hz (Ratio: {cent_syn/cent_tgt:.2f}x)")
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
            "warm_loss": best_loss,
            "centroid_target_hz": float(cent_tgt),
            "centroid_synth_hz": float(cent_syn),
            "bass_metrics": bass_metrics,
        },
        copy_to_user_dir=True,
    )

    daw_preset_path = f"{DAW_DIR}/{target['prefix']}_inversion.vstpreset"
    if os.path.exists(f"{out_prefix}.vstpreset"):
        shutil.copy(f"{out_prefix}.vstpreset", daw_preset_path)

    # Visualizations
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    t_axis = np.linspace(0, phrase_dur, len(y_target))

    axs[0].plot(t_axis, y_target, label=f"Real Stem ({target['artist']})", color="royalblue", alpha=0.75)
    axs[0].plot(t_axis, best_audio, label=f"Warm Synth Match ({best_filter_name})", color="crimson", alpha=0.75)
    axs[0].set_title(f"{target['artist']} - {target['track']} (Warmth Alignment: {cent_tgt:.0f}Hz Target vs {cent_syn:.0f}Hz Synth)")
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
    axs[1].set_title(f"Dynamic Bass Envelope Alignment (Env MSE: {bass_metrics['bass_env_mse']:.6f})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Level")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # Frequency Spectrum FFT
    fft_t = np.abs(np.fft.rfft(y_target))
    fft_s = np.abs(np.fft.rfft(best_audio))
    freqs = np.fft.rfftfreq(len(y_target), 1/sr)
    
    def smooth(x, w=80):
        return np.convolve(x, np.ones(w)/w, mode="same")
    
    db_t = 20 * np.log10(smooth(fft_t) + 1e-6)
    db_s = 20 * np.log10(smooth(fft_s) + 1e-6)
    db_t -= np.max(db_t)
    db_s -= np.max(db_s)
    
    mask = (freqs >= 30) & (freqs <= 4000)
    axs[2].semilogx(freqs[mask], db_t[mask], label="Target Stem (Demucs)", color="royalblue", lw=1.8)
    axs[2].semilogx(freqs[mask], db_s[mask], label=f"Warm Synth ({best_filter_name})", color="crimson", lw=1.8, linestyle="--")
    axs[2].set_title(f"Spectral Match: Cutoff {best_patch['cutoff']:.3f}, Res {best_patch['resonance']:.3f}, Sub {best_patch['sub_mix']:.3f}")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("dB Magnitude")
    axs[2].set_ylim(-60, 5)
    axs[2].legend(loc="upper right")
    axs[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plot_path = f"{out_prefix}_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved warm comparison plot to: {plot_path}")
    print(f"Saved DAW preset to: {daw_preset_path}")


def main():
    for target in TARGETS:
        invert_track_warm(target)
    print("\n" + "=" * 70)
    print("WARM MULTI-ARTIST BENCHMARKS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
