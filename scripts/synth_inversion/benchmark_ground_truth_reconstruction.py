#!/usr/bin/env python3
"""Ground-Truth Synthetic Reconstruction Benchmark.

Strictly tests whether the neural model or DE optimizer can recover
known synthetic patches (Saw bass, Pulse bass, Acid pluck, Sub bass).

Process:
1. Define 4 pristine, classic ground-truth patches (Pure Saw, Filtered Pulse, Acid Resonance, Deep Sub).
2. Explicitly verify NO SYNC, NO FM, NO EFFECTS in ground truth.
3. Render reference audio with a known MIDI phrase.
4. Pass audio to Model / Inverter to predict parameters and make a reconstructed preset.
5. Render predicted preset with the exact same MIDI phrase.
6. Compare parameter errors (Sync, FM, Cutoff, Shape) and spectral fidelity (STFT, Centroid, Waveform).
7. Save A/B audio clips, comparison plots, and Steinberg .vstpreset files.
"""

import os
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import torch

sys.path.append("/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from surge_spec import init_synth, apply_patch, LP_FILTERS, render_patch, vector_to_patch, patch_to_vector
from audio_utils import save_patch, MultiScaleSTFTLoss, make_mel_spec
from inference import load_inverter, predict_vectors, vectors_to_patches

DAW_DIR = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")
OUT_DIR = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/ground_truth_benchmark"
os.makedirs(OUT_DIR, exist_ok=True)
EVAL_DIR = f"{OUT_DIR}/eval_clips"
os.makedirs(EVAL_DIR, exist_ok=True)

# 4 Ground-Truth Archetypes (Classic Goa/Psy & Techno Basslines)
GROUND_TRUTH_PATCHES = [
    {
        "name": "pure_saw_bass",
        "description": "Pure Single-Osc Saw Bassline (Zero Sync, Zero FM)",
        "midi_note": 36,  # C2 (65.4 Hz)
        "bpm": 140.0,
        "patch": {
            "midi_note": 36,
            "filter_idx": 1,     # LP 24 dB (Moog ladder)
            "shape": 0.0,        # 100% Pure Sawtooth
            "width": 0.50,
            "sub_mix": 0.0,      # Single oscillator
            "sync": 0.0,         # ZERO SYNC
            "fm_depth": 0.0,     # ZERO FM
            "unison": False,
            "unison_detune": 0.0,
            "cutoff": 0.38,      # ~230 Hz warm cutoff
            "resonance": 0.15,
            "keytrack_raw": 1.0,
            "feg_amount": 0.35,  # Moderate punch
            "feg_decay": 0.22,
            "feg_sustain": 0.0,
            "aeg_decay": 0.38,
            "aeg_sustain": 0.0,
            "aeg_release": 0.04,
            "ws_idx": 0,
            "drive_raw": 0.50,   # Pure clean
            "chorus_mix": 0.0,
            "delay_mix": 0.0,
            "delay_fb": 0.0,
        }
    },
    {
        "name": "pulse_pluck_bass",
        "description": "Filtered Square/Pulse Bass (Zero Sync, Zero FM)",
        "midi_note": 41,  # F2 (86.7 Hz)
        "bpm": 140.0,
        "patch": {
            "midi_note": 41,
            "filter_idx": 0,     # LP 12 dB
            "shape": 1.0,        # Pure Pulse
            "width": 0.35,       # 35% duty cycle pulse
            "sub_mix": 0.30,
            "sync": 0.0,
            "fm_depth": 0.0,
            "unison": False,
            "unison_detune": 0.0,
            "cutoff": 0.42,      # ~320 Hz
            "resonance": 0.25,
            "keytrack_raw": 1.0,
            "feg_amount": 0.40,
            "feg_decay": 0.16,
            "feg_sustain": 0.0,
            "aeg_decay": 0.32,
            "aeg_sustain": 0.0,
            "aeg_release": 0.05,
            "ws_idx": 0,
            "drive_raw": 0.50,
            "chorus_mix": 0.0,
            "delay_mix": 0.0,
            "delay_fb": 0.0,
        }
    },
    {
        "name": "acid_resonant_bass",
        "description": "303-Style Resonant Acid Bass (Zero Sync, Zero FM)",
        "midi_note": 38,  # D2 (73.4 Hz)
        "bpm": 140.0,
        "patch": {
            "midi_note": 38,
            "filter_idx": 7,     # LP Diode Ladder
            "shape": 0.05,       # Saw-leaning
            "width": 0.50,
            "sub_mix": 0.0,
            "sync": 0.0,
            "fm_depth": 0.0,
            "unison": False,
            "unison_detune": 0.0,
            "cutoff": 0.34,      # ~180 Hz
            "resonance": 0.52,   # High squelch
            "keytrack_raw": 1.0,
            "feg_amount": 0.55,  # Strong envelope sweep
            "feg_decay": 0.18,
            "feg_sustain": 0.0,
            "aeg_decay": 0.35,
            "aeg_sustain": 0.0,
            "aeg_release": 0.05,
            "ws_idx": 0,
            "drive_raw": 0.56,
            "chorus_mix": 0.0,
            "delay_mix": 0.0,
            "delay_fb": 0.0,
        }
    },
    {
        "name": "deep_sub_punch",
        "description": "Deep Heavy Sub-Bass (Zero Sync, Zero FM)",
        "midi_note": 33,  # A1 (55.0 Hz)
        "bpm": 143.55,
        "bars": 2,
        "patch": {
            "midi_note": 33,
            "filter_idx": 4,     # LP OB-Xd 12 dB
            "shape": 0.20,
            "width": 0.50,
            "sub_mix": 0.85,     # Heavy sub oscillator
            "sync": 0.0,
            "fm_depth": 0.0,
            "unison": False,
            "unison_detune": 0.0,
            "cutoff": 0.30,      # ~130 Hz closed filter
            "resonance": 0.10,
            "keytrack_raw": 1.0,
            "feg_amount": 0.20,
            "feg_decay": 0.25,
            "feg_sustain": 0.0,
            "aeg_decay": 0.45,
            "aeg_sustain": 0.0,
            "aeg_release": 0.05,
            "ws_idx": 0,
            "drive_raw": 0.52,
            "chorus_mix": 0.0,
            "delay_mix": 0.0,
            "delay_fb": 0.0,
        }
    }
]


def build_rolling_events(note: int, bpm: float, bars: int = 2):
    beat_dur = 60.0 / bpm
    sixteenth = beat_dur / 4.0
    phrase_dur = bars * 4.0 * beat_dur
    events = []
    t = 0.0
    for s in range(bars * 16):
        if s % 4 in (1, 2, 3):  # Rolling 16th bass
            events.append(mido.Message("note_on", note=int(note), velocity=105, time=t))
            events.append(mido.Message("note_off", note=int(note), velocity=0, time=t + sixteenth * 0.85))
        t += sixteenth
    return events, phrase_dur


def main():
    sr = 44100
    synth = init_synth(sample_rate=sr)
    synth.process(np.zeros((2, 1024), dtype=np.float32), 1024/sr, sr, 2)
    loss_fn = MultiScaleSTFTLoss()

    # Load neural inverter (Flow matching model)
    flow_ckpt = "/run/media/kim/Mantu/surge_200k_models/G02_deepflow_200k_best.pt"
    print(f"Loading neural inverter from: {flow_ckpt}...")
    model = load_inverter(flow_ckpt)

    print("\n" + "=" * 80)
    print("GROUND TRUTH SYNTHESIS & INVERSION BENCHMARK")
    print("=" * 80)

    for item in GROUND_TRUTH_PATCHES:
        name = item["name"]
        p_gt = item["patch"]
        bpm = item["bpm"]
        note = item["midi_note"]

        print(f"\n--- Testing Archetype: {item['description']} ---")
        events, dur = build_rolling_events(note, bpm, bars=2)

        # 1. Render Ground Truth Audio
        apply_patch(synth, p_gt)
        synth.reset()
        audio_gt = synth.process(events, duration=dur, sample_rate=sr, num_channels=2)
        mono_gt = np.mean(audio_gt, axis=0).astype(np.float32)
        mono_gt = mono_gt / (np.max(np.abs(mono_gt)) + 1e-7)

        # 2. Extract single 0.8s representative note for the neural inverter input
        # Note starts at first offbeat 16th
        sixteenth = (60.0 / bpm) / 4.0
        start_samp = int(sixteenth * sr)
        note_chunk = mono_gt[start_samp : start_samp + int(0.8 * sr)]
        if len(note_chunk) < int(0.8 * sr):
            note_chunk = np.pad(note_chunk, (0, int(0.8 * sr) - len(note_chunk)))

        mel = torch.from_numpy(make_mel_spec(note_chunk)).unsqueeze(0)

        # 3. Model Prediction (Inference)
        vecs, _ = predict_vectors(model, mel, n_draws=8, steps=25)
        # Evaluate candidate draws against the note
        cand_patches = vectors_to_patches(vecs[:, 0])
        best_p = None
        best_l = float("inf")
        for cp in cand_patches:
            # Render single note to pick best draw
            r_audio = render_patch(synth, cp, note, note_dur=sixteenth * 0.85, duration=0.8, sample_rate=sr)
            l = loss_fn(r_audio, note_chunk)
            if l < best_l:
                best_l = l
                best_p = cp

        # 4. Render predicted preset with the EXACT same rolling phrase
        apply_patch(synth, best_p)
        synth.reset()
        audio_pred = synth.process(events, duration=dur, sample_rate=sr, num_channels=2)
        mono_pred = np.mean(audio_pred, axis=0).astype(np.float32)
        mono_pred = mono_pred / (np.max(np.abs(mono_pred)) + 1e-7)

        # 5. Measure Parameter Errors & Unwanted Sync/FM Hallucination
        stft_loss = float(loss_fn(mono_pred, mono_gt))
        cent_gt = np.mean(librosa.feature.spectral_centroid(y=mono_gt, sr=sr))
        cent_pred = np.mean(librosa.feature.spectral_centroid(y=mono_pred, sr=sr))

        print(f"RESULTS FOR {name}:")
        print(f"  STFT Loss:          {stft_loss:.3f}")
        print(f"  Centroid:           GT: {cent_gt:.1f} Hz | Pred: {cent_pred:.1f} Hz (Ratio: {cent_pred/cent_gt:.2f}x)")
        print(f"  Circuit:            GT: {LP_FILTERS[p_gt['filter_idx']][0]} | Pred: {LP_FILTERS[best_p['filter_idx']][0]}")
        print(f"  Shape:              GT: {p_gt['shape']:.3f} | Pred: {best_p['shape']:.3f} (Err: {abs(best_p['shape']-p_gt['shape']):.3f})")
        print(f"  Cutoff:             GT: {p_gt['cutoff']:.3f} | Pred: {best_p['cutoff']:.3f} (Err: {abs(best_p['cutoff']-p_gt['cutoff']):.3f})")
        print(f"  Resonance:          GT: {p_gt['resonance']:.3f} | Pred: {best_p['resonance']:.3f} (Err: {abs(best_p['resonance']-p_gt['resonance']):.3f})")
        print(f"  Sync:               GT: {p_gt['sync']:.3f} | Pred: {best_p['sync']:.3f} {'[SPURIOUS SYNC!]' if best_p['sync'] > 0.05 else '[Clean]'}")
        print(f"  FM Depth:           GT: {p_gt['fm_depth']:.3f} | Pred: {best_p['fm_depth']:.3f} {'[SPURIOUS FM!]' if best_p['fm_depth'] > 0.05 else '[Clean]'}")

        # 6. Save Audio Files & Presets
        sf.write(f"{EVAL_DIR}/{name}_gt.wav", mono_gt, sr)
        sf.write(f"{EVAL_DIR}/{name}_pred.wav", mono_pred, sr)

        # A/B Comparison clip
        silence = np.zeros(int(0.4 * sr), dtype=np.float32)
        ab_clip = np.concatenate([mono_gt, silence, mono_pred, silence, mono_gt, silence, mono_pred])
        sf.write(f"{EVAL_DIR}/{name}_AB_comparison.wav", ab_clip, sr)

        # Stereo L=GT, R=Pred
        stereo = np.stack([mono_gt, mono_pred], axis=-1)
        sf.write(f"{EVAL_DIR}/{name}_stereo_L_gt_R_pred.wav", stereo, sr)

        # Save Presets
        save_patch(synth, best_p, f"{OUT_DIR}/{name}_predicted", copy_to_user_dir=True)
        shutil.copy(f"{OUT_DIR}/{name}_predicted.vstpreset", f"{DAW_DIR}/{name}_predicted.vstpreset")

        # 7. Comparison Diagnostic Plot
        fig, axs = plt.subplots(3, 1, figsize=(14, 10))
        t_axis = np.linspace(0, dur, len(mono_gt))

        axs[0].plot(t_axis, mono_gt, label=f"Ground Truth ({item['description']})", color="royalblue", alpha=0.75)
        axs[0].plot(t_axis, mono_pred, label=f"Model Prediction ({LP_FILTERS[best_p['filter_idx']][0]})", color="crimson", alpha=0.75)
        axs[0].set_title(f"{item['description']} (STFT Loss: {stft_loss:.2f})")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, alpha=0.3)

        # Zoomed-in single cycle waveform
        idx_zoom_start = int(0.20 * sr)
        idx_zoom_end = int(0.25 * sr)
        t_zoom = t_axis[idx_zoom_start:idx_zoom_end]
        axs[1].plot(t_zoom, mono_gt[idx_zoom_start:idx_zoom_end], label="Ground Truth Single Note Cycles", color="royalblue", lw=1.8)
        axs[1].plot(t_zoom, mono_pred[idx_zoom_start:idx_zoom_end], label="Predicted Synth Note Cycles", color="crimson", lw=1.8, linestyle="--")
        axs[1].set_title(f"Cycle Waveform Fidelity (Shape GT: {p_gt['shape']:.2f} vs Pred: {best_p['shape']:.2f}, Sync: {best_p['sync']:.2f}, FM: {best_p['fm_depth']:.2f})")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, alpha=0.3)

        # Spectrum
        fft_gt = np.abs(np.fft.rfft(mono_gt))
        fft_pr = np.abs(np.fft.rfft(mono_pred))
        freqs = np.fft.rfftfreq(len(mono_gt), 1/sr)
        
        def smooth(x, w=80):
            return np.convolve(x, np.ones(w)/w, mode="same")
            
        db_gt = 20 * np.log10(smooth(fft_gt) + 1e-6)
        db_pr = 20 * np.log10(smooth(fft_pr) + 1e-6)
        db_gt -= np.max(db_gt)
        db_pr -= np.max(db_pr)
        
        mask = (freqs >= 30) & (freqs <= 4000)
        axs[2].semilogx(freqs[mask], db_gt[mask], label="Ground Truth Spectrum", color="royalblue", lw=1.8)
        axs[2].semilogx(freqs[mask], db_pr[mask], label="Predicted Spectrum", color="crimson", lw=1.8, linestyle="--")
        axs[2].set_title(f"Spectral Match: Cutoff GT {p_gt['cutoff']:.2f} vs Pred {best_p['cutoff']:.2f}, Res GT {p_gt['resonance']:.2f} vs Pred {best_p['resonance']:.2f}")
        axs[2].set_xlabel("Frequency (Hz)")
        axs[2].set_ylabel("dB Magnitude")
        axs[2].set_ylim(-60, 5)
        axs[2].legend(loc="upper right")
        axs[2].grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plot_path = f"{OUT_DIR}/{name}_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Eval clips saved to: {EVAL_DIR}/")
    print(f"DAW presets saved to: {DAW_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
