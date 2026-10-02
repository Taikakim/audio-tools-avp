"""Rolling phrase inverter: inverts full multi-bar bassline phrases directly against stems.

Unlike single-note matching, the rolling phrase evaluates:
  1. Multi-scale STFT loss across the continuous phrase (capturing the full harmonic spectrum).
  2. Continuous lowpassed (<300 Hz) dynamic envelope matching across all 16th-note hits,
     ensuring the synth's decay, release, and filter sweep reproduce the exact rhythmic
     groove without inter-note mud or premature choking.
  3. Strict musical constraints: dry (delay_mix = 0, chorus_mix = 0), mono (unison = False,
     sync = 0, fm = 0), safe pulse width (no needle spikes).

Usage:
  python match_rolling_phrase.py --bars 2 --maxiter 12 --popsize 8
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import mido
import numpy as np
import soundfile as sf
import torch
from scipy.optimize import differential_evolution

from audio_utils import MultiScaleSTFTLoss, save_patch
from envelope_extractor import compute_bass_match_loss, lowpass_300hz, zolzer_envelope_follower
from surge_spec import DEFAULT_PLUGIN_PATH, LP_FILTERS, apply_patch, init_synth

DEFAULT_STEM = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/muscriptor_full_track_test/real_stem_60_90s.wav"
DEFAULT_MIDI = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/muscriptor_full_track_test/bass.mid"
OUT_DIR = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/rolling_phrase_match"
DAW_DIR = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")

# Base patch ensuring 100% dry, mono, musical bassline
BASE_PATCH = dict(
    filter_idx=0,
    shape=0.35,
    width=0.50,
    sub_mix=0.15,
    sync=0.0,
    fm_depth=0.0,
    unison=False,
    unison_detune=0.0,
    cutoff=0.45,
    resonance=0.35,
    keytrack_raw=0.77,
    feg_amount=0.65,
    feg_decay=0.30,
    feg_sustain=0.0,
    aeg_decay=0.30,
    aeg_sustain=0.0,
    aeg_release=0.12,
    ws_idx=0,
    drive_raw=0.50,
    chorus_mix=0.0,
    delay_mix=0.0,
    delay_fb=0.0,
)

SEARCH_SPACE = [
    ("shape", 0.0, 1.0),        # 0: 0=saw, 0.5=square, 1=pulse
    ("width", 0.15, 0.85),      # 1: symmetric safe duty cycle
    ("sub_mix", 0.0, 0.65),     # 2: sub-oscillator reinforcement
    ("cutoff", 0.18, 0.82),     # 3: filter cutoff
    ("resonance", 0.05, 0.80),  # 4: filter resonance / Q
    ("feg_amount", 0.20, 0.95), # 5: filter envelope modulation depth
    ("feg_decay", 0.05, 0.55),  # 6: filter decay time
    ("aeg_decay", 0.10, 0.60),  # 7: amp envelope decay
    ("aeg_release", 0.05, 0.35),# 8: amp envelope release
    ("drive_raw", 0.50, 0.75),  # 9: waveshaper analog drive
]


def load_phrase_midi(midi_path: str, duration_s: float):
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    current_tempo = 500000

    events = []
    current_time_s = 0.0
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            current_tempo = msg.tempo
        current_time_s += mido.tick2second(msg.time, ticks_per_beat, current_tempo)
        if current_time_s >= duration_s:
            break
        if msg.type in ["note_on", "note_off"]:
            events.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=current_time_s))

    return events


def match_rolling_phrase(stem_path: str = DEFAULT_STEM,
                         midi_path: str = DEFAULT_MIDI,
                         bars: int = 2,
                         bpm: float = 142.857,
                         maxiter: int = 12,
                         popsize: int = 8,
                         filter_candidates: list = None):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DAW_DIR, exist_ok=True)

    sr = 44100
    bar_dur = (60.0 / bpm) * 4.0
    phrase_dur = bars * bar_dur
    n_samples = int(phrase_dur * sr)

    print(f"=== Rolling Phrase Inversion ===")
    print(f"BPM: {bpm:.3f} | Bars: {bars} | Duration: {phrase_dur:.3f}s ({n_samples} samples)")
    print(f"Stem: {stem_path}")
    print(f"MIDI: {midi_path}")

    # 1. Load target stem
    y_full, _ = sf.read(stem_path)
    if y_full.ndim > 1:
        y_full = np.mean(y_full, axis=1)
    y_target = y_full[:n_samples].astype(np.float32)
    y_target = y_target / (np.max(np.abs(y_target)) + 1e-7)

    # 2. Load phrase MIDI
    events = load_phrase_midi(midi_path, phrase_dur)
    print(f"Loaded {len(events)} MIDI events over {phrase_dur:.2f}s.")

    # 3. Setup Synth & Loss
    synth = init_synth(verify=False)
    # Prime synth to prevent block-0 muting
    synth.process(np.zeros((2, 1024), dtype=np.float32), sr)

    loss_stft = MultiScaleSTFTLoss()

    if filter_candidates is None:
        # Benchmark top Goa trance lowpass circuits:
        # idx 0: LP 24 dB, idx 1: LP OB-Xd 12 dB, idx 4: LP Vintage Ladder, idx 7: LP Diode Ladder
        filter_candidates = [0, 1, 4, 7]

    bounds = [(lo, hi) for _, lo, hi in SEARCH_SPACE]
    param_names = [name for name, _, _ in SEARCH_SPACE]

    best_overall_loss = float("inf")
    best_overall_patch = None
    best_overall_audio = None
    best_filter_name = None

    for f_idx in filter_candidates:
        filter_name = LP_FILTERS[f_idx][0]
        print(f"\n--- Optimizing Filter: {filter_name} (index {f_idx}) ---")

        def objective(x):
            p = dict(BASE_PATCH)
            p["filter_idx"] = f_idx
            for k, val in zip(param_names, x):
                p[k] = float(val)

            apply_patch(synth, p)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            mono = mono / (np.max(np.abs(mono)) + 1e-7)

            l_stft = float(loss_stft(mono, y_target))
            b_res = compute_bass_match_loss(y_target, mono, fs=sr, cutoff_hz=300.0)
            l_bass = b_res["loss_bass"]

            # Combined objective: STFT spectral match + 2.0x lowpassed envelope/RMS tracking
            total_loss = l_stft + 2.0 * l_bass
            return total_loss

        t_start = time.time()
        res = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            polish=False,
            workers=1,
            disp=False,
        )
        elapsed = time.time() - t_start
        print(f"Filter {filter_name} finished in {elapsed:.1f}s | Best Loss: {res.fun:.3f}")

        if res.fun < best_overall_loss:
            best_overall_loss = res.fun
            best_filter_name = filter_name
            best_p = dict(BASE_PATCH)
            best_p["filter_idx"] = f_idx
            for k, val in zip(param_names, res.x):
                best_p[k] = float(val)
            best_overall_patch = best_p

            # Render best audio
            apply_patch(synth, best_p)
            synth.reset()
            audio = synth.process(events, duration=phrase_dur, sample_rate=sr, num_channels=2)
            mono = np.mean(audio, axis=0).astype(np.float32)
            best_overall_audio = mono / (np.max(np.abs(mono)) + 1e-7)

    # Detailed metrics on champion patch
    stft_final = float(loss_stft(best_overall_audio, y_target))
    bass_metrics = compute_bass_match_loss(y_target, best_overall_audio, fs=sr, cutoff_hz=300.0)

    print("\n" + "=" * 70)
    print("CHAMPION ROLLING PHRASE INVERSION")
    print("=" * 70)
    print(f"Circuit:            {best_filter_name}")
    print(f"Combined Loss:      {best_overall_loss:.3f}")
    print(f"Multi-Scale STFT:   {stft_final:.3f}")
    print(f"Bass Envelope MSE:  {bass_metrics['bass_env_mse']:.6f}")
    print(f"Target <300Hz Ratio: {bass_metrics['bass_ratio_target']*100:.1f}%")
    print(f"Synth <300Hz Ratio:  {bass_metrics['bass_ratio_synth']*100:.1f}%")
    print(f"Optimized Patch:")
    for k in param_names:
        print(f"  {k:15s}: {best_overall_patch[k]:.4f}")

    # Save audio files
    stem_crop_path = f"{OUT_DIR}/rolling_phrase_target.wav"
    synth_crop_path = f"{OUT_DIR}/rolling_phrase_synth.wav"
    sf.write(stem_crop_path, y_target, sr)
    sf.write(synth_crop_path, best_overall_audio, sr)

    # Save patch JSON and native VST preset
    save_patch(
        synth,
        best_overall_patch,
        f"{OUT_DIR}/rolling_phrase_best",
        extra={
            "stft_loss": stft_final,
            "bass_metrics": bass_metrics,
            "bars": bars,
            "bpm": bpm,
            "duration_s": phrase_dur,
        },
        copy_to_user_dir=True,
    )

    # Also copy preset to DAW AI Inversions directory
    daw_preset_path = f"{DAW_DIR}/rolling_phrase_inversion.vstpreset"
    import shutil
    shutil.copy(f"{OUT_DIR}/rolling_phrase_best.vstpreset", daw_preset_path)
    print(f"\nSaved DAW preset to: {daw_preset_path}")
    print(f"Saved audio renders to: {OUT_DIR}/")

    # Generate Comparison Plot: Waveform, Envelope Contour, and Spectrogram
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    t_axis = np.linspace(0, phrase_dur, len(y_target))

    # Panel 1: Waveforms
    axs[0].plot(t_axis, y_target, label="Real Stem (Final Mission)", color="royalblue", alpha=0.75, linewidth=0.9)
    axs[0].plot(t_axis, best_overall_audio, label=f"Inverted Synth ({best_filter_name})", color="crimson", alpha=0.7, linewidth=0.9)
    axs[0].set_title(f"Rolling Phrase Waveform Alignment ({bars} Bars, {bpm:.1f} BPM, STFT Loss: {stft_final:.2f})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: <300 Hz Bass Envelopes (Zölzer Envelope Follower)
    lp_tgt = lowpass_300hz(y_target, fs=sr, cutoff_hz=300.0)
    lp_syn = lowpass_300hz(best_overall_audio, fs=sr, cutoff_hz=300.0)
    env_tgt = zolzer_envelope_follower(lp_tgt, fs=sr)
    env_syn = zolzer_envelope_follower(lp_syn, fs=sr)

    axs[1].plot(t_axis, env_tgt, label=f"Target Bass Env (<300Hz, Ratio: {bass_metrics['bass_ratio_target']*100:.1f}%)", color="blue", linewidth=1.5)
    axs[1].plot(t_axis, env_syn, label=f"Synth Bass Env (<300Hz, Ratio: {bass_metrics['bass_ratio_synth']*100:.1f}%)", color="darkgreen", linewidth=1.5, linestyle="--")
    axs[1].set_title(f"Dynamic Bass Envelope Alignment (<300Hz, MSE: {bass_metrics['bass_env_mse']:.6f})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Envelope Level")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Time-Frequency Spectrogram of Synth Match
    S_syn = librosa.amplitude_to_db(np.abs(librosa.stft(best_overall_audio, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(S_syn, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[2], cmap="magma", vmin=-50, vmax=0)
    axs[2].set_ylim(0, 4000)
    axs[2].set_title(f"Synth Spectrogram ({best_filter_name}, Cutoff: {best_overall_patch['cutoff']:.3f}, Res: {best_overall_patch['resonance']:.3f})")

    plt.tight_layout()
    plot_path = f"{OUT_DIR}/rolling_phrase_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")

    return best_overall_patch, best_overall_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--midi", default=DEFAULT_MIDI)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--bpm", type=float, default=142.857)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--popsize", type=int, default=8)
    args = parser.parse_args()

    match_rolling_phrase(
        stem_path=args.stem,
        midi_path=args.midi,
        bars=args.bars,
        bpm=args.bpm,
        maxiter=args.maxiter,
        popsize=args.popsize,
    )
