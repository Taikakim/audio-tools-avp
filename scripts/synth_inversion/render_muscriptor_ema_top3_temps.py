#!/usr/bin/env python3
"""Render top 3 predictions per stem across temperatures (tau = 0.3, 0.7, 1.0) using Flow EMA weights.

Features:
1. Uses Flow EMA model (flow_ema_best.pt).
2. Sweeps condition noise / sampling temperatures: tau = 0.3, 0.7, 1.0.
3. No phrase refinement stage (evaluates raw flow predictions directly).
4. Uses curated MuScriptor MIDIs for accurate musical notes and phrasing.
5. For each candidate (cand1, cand2, cand3), renders both:
   - Raw model prediction
   - Harmonic snapped version (oscillator ratio locked to {2:1, 1:1, 1:2, 1:3}, static sync/FM zeroed)
6. Outputs clean WAV files, presets, and comprehensive metrics JSON.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from models_v3 import load_inverter
from surge_spec_v3 import (
    init_synth, apply_patch, flush_synth, DEFAULT_PLUGIN_PATH,
    SAMPLE_RATE, AUDIO_LEN
)
from audio_utils import make_mel_spec, save_patch, MultiScaleSTFTLoss
from inference_v3 import predict_and_rerank_candidates
from invert_stem_collection_v3_cpu import (
    STEM_CATALOG, resolve_mantu, find_active_bass_phrase, load_pedalboard_events_from_midi
)
from render_muscriptor_top3_harmonic import snap_to_harmonic_ratio, apply_patch_with_pitch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_ckpt", type=str,
                        default="/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a0/flow_ema_best.pt")
    parser.add_argument("--out_dir", type=str,
                        default="/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a0/eval_cpu_muscriptor_ema_temps")
    parser.add_argument("--midi_dir", type=str,
                        default="/run/media/kim/Mantu/surge_200k_models/real_stems_eval/midi_muscriptor")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    args = parser.parse_args()

    flow_ckpt = resolve_mantu(args.flow_ckpt)
    out_dir = resolve_mantu(args.out_dir)
    midi_dir = resolve_mantu(args.midi_dir)

    audio_dir = os.path.join(out_dir, "audio")
    presets_dir = os.path.join(out_dir, "vstpresets")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(presets_dir, exist_ok=True)

    print("=" * 80)
    print("SURGE XT REAL-STEM EVALUATION: EMA MODEL SWEEP (tau = 0.3, 0.7, 1.0)")
    print(f"Model: {flow_ckpt}")
    print(f"Output: {out_dir}")
    print(f"Temperatures: {args.temps}")
    print(f"Playback MIDI: MuScriptor from {midi_dir}")
    print("Refinement: NONE (Top 3 direct predictions + harmonic snapped)")
    print("=" * 80)

    print("Loading Flow EMA model onto CPU...")
    model = load_inverter(flow_ckpt, device="cpu")
    synth = init_synth(DEFAULT_PLUGIN_PATH)
    loss_fn = MultiScaleSTFTLoss()

    summary_records = []
    t_start_all = time.time()

    for idx, item in enumerate(STEM_CATALOG):
        stem_id = item["id"]
        name = item["name"]
        path = resolve_mantu(item["path"])
        bpm = item["bpm"]
        style = item["style"]

        if not os.path.exists(path):
            print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] Missing audio: {path}")
            continue

        # Load audio stem
        y, sr = sf.read(path)
        if y.ndim > 1:
            y_raw = y.T
            y_mono = np.mean(y, axis=1)
        else:
            y_raw = np.stack([y, y])
            y_mono = y

        if sr != SAMPLE_RATE:
            y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=SAMPLE_RATE)
            y_raw = librosa.resample(y_raw, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Slice active phrase
        bars = 4 if len(y_mono) / sr > 10.0 else 2
        p_start, p_len = find_active_bass_phrase(y_mono, sr, bpm, bars=bars)
        phrase_dur = p_len / sr

        gt_raw_chunk = y_raw[:, p_start : p_start + p_len]
        gt_mono_chunk = y_mono[p_start : p_start + p_len]
        if gt_raw_chunk.shape[1] < p_len:
            gt_raw_chunk = np.pad(gt_raw_chunk, ((0, 0), (0, p_len - gt_raw_chunk.shape[1])))
        if len(gt_mono_chunk) < p_len:
            gt_mono_chunk = np.pad(gt_mono_chunk, (0, p_len - len(gt_mono_chunk)))

        gt_norm = (gt_raw_chunk / (np.max(np.abs(gt_raw_chunk)) + 1e-7)).astype(np.float32)

        # Save reference stem audio once
        ref_wav_path = os.path.join(audio_dir, f"{stem_id}_real.wav")
        if not os.path.exists(ref_wav_path):
            sf.write(ref_wav_path, gt_norm.T, SAMPLE_RATE)

        # Load MuScriptor MIDI
        mu_midi_path = os.path.join(midi_dir, f"{stem_id}_muscriptor.mid")
        if not os.path.exists(mu_midi_path):
            print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] Missing MuScriptor MIDI: {mu_midi_path}")
            continue

        events = load_pedalboard_events_from_midi(mu_midi_path, max_dur=phrase_dur)
        note_on_pitches = [m.note for m in events if m.type == 'note_on' and m.velocity > 0]
        root_midi = max(set(note_on_pitches), key=note_on_pitches.count) if note_on_pitches else 36

        # Extract target single note for inversion
        sixteenth_s = (60.0 / bpm) / 4.0
        note_dur = sixteenth_s * 0.85
        slice_onsets = librosa.onset.onset_detect(y=gt_mono_chunk, sr=sr, units='samples')
        note_start = slice_onsets[0] if len(slice_onsets) > 0 else 0
        target_note = np.zeros(AUDIO_LEN, dtype=np.float32)
        chunk = gt_mono_chunk[note_start : note_start + int(sr * sixteenth_s)]
        fade_len = int(sr * 0.005)
        if len(chunk) > fade_len:
            chunk[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        target_note[:len(chunk)] = chunk
        mel = torch.from_numpy(make_mel_spec(target_note)).unsqueeze(0)

        stem_record = {
            "index": idx + 1,
            "id": stem_id,
            "name": name,
            "style": style,
            "bpm": bpm,
            "root_midi": root_midi,
            "phrase_dur_s": round(phrase_dur, 2),
            "num_notes": len(note_on_pitches),
            "temperatures": {}
        }

        print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] {name:30s} | Notes: {len(note_on_pitches):2d} | Root: {root_midi}")

        for tau in args.temps:
            t_str = f"t{int(tau*10):02d}" if tau < 1.0 else "t10"
            t0 = time.time()
            best, desc, scored_cands = predict_and_rerank_candidates(
                model, synth, target_note, mel, root_midi, note_dur=note_dur,
                n_candidates=20, steps=25, seed=42 + idx, tau=tau
            )
            top3 = scored_cands[:3]
            cands_data = []

            for rank_i, cand in enumerate(top3):
                rank = rank_i + 1
                raw_patch = cand["patch"]
                harm_patch, ratio_name = snap_to_harmonic_ratio(raw_patch)

                # 1. Render RAW candidate playback
                flush_synth(synth)
                apply_patch_with_pitch(synth, raw_patch)
                pred_raw = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                flush_synth(synth)
                pred_raw_norm = (pred_raw / (np.max(np.abs(pred_raw)) + 1e-7)).astype(np.float32)

                raw_wav = os.path.join(audio_dir, f"{stem_id}_{t_str}_cand{rank}_raw.wav")
                sf.write(raw_wav, pred_raw_norm.T, SAMPLE_RATE)
                save_patch(synth, raw_patch, os.path.join(presets_dir, f"{stem_id}_{t_str}_cand{rank}_raw"), copy_to_user_dir=False)

                # 2. Render HARMONIC SNAPPED candidate playback
                flush_synth(synth)
                apply_patch_with_pitch(synth, harm_patch)
                pred_harm = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                flush_synth(synth)
                pred_harm_norm = (pred_harm / (np.max(np.abs(pred_harm)) + 1e-7)).astype(np.float32)

                safe_ratio = ratio_name.replace(":", "_")
                harm_wav = os.path.join(audio_dir, f"{stem_id}_{t_str}_cand{rank}_harm_{safe_ratio}.wav")
                sf.write(harm_wav, pred_harm_norm.T, SAMPLE_RATE)
                save_patch(synth, harm_patch, os.path.join(presets_dir, f"{stem_id}_{t_str}_cand{rank}_harm_{safe_ratio}"), copy_to_user_dir=False)

                # Calculate phrase STFT losses
                raw_mono = np.mean(pred_raw_norm, axis=0)
                harm_mono = np.mean(pred_harm_norm, axis=0)
                min_len = min(len(gt_mono_chunk), len(raw_mono))
                stft_raw = float(loss_fn(raw_mono[:min_len], gt_mono_chunk[:min_len]))
                stft_harm = float(loss_fn(harm_mono[:min_len], gt_mono_chunk[:min_len]))

                cands_data.append({
                    "rank": rank,
                    "stft_phrase_raw": round(stft_raw, 3),
                    "stft_phrase_harmonic": round(stft_harm, 3),
                    "harmonic_ratio": ratio_name,
                    "cutoff": round(float(raw_patch.get("cutoff", 0)), 3),
                    "resonance": round(float(raw_patch.get("resonance", 0)), 3),
                    "feg_decay": round(float(raw_patch.get("feg_decay", 0)), 3),
                    "aeg_decay": round(float(raw_patch.get("aeg_decay", 0)), 3),
                    "raw_wav": raw_wav,
                    "harm_wav": harm_wav
                })

            stem_record["temperatures"][str(tau)] = cands_data
            c1 = cands_data[0]
            print(f"    tau={tau:.1f} -> Cand1 STFT: raw {c1['stft_phrase_raw']:.2f} | harm ({c1['harmonic_ratio']}) {c1['stft_phrase_harmonic']:.2f} ({time.time()-t0:.1f}s)")

        summary_records.append(stem_record)

    total_time = time.time() - t_start_all
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    print("=" * 80)
    print(f"ALL 24 STEMS COMPLETE across tau in [0.3, 0.7, 1.0] in {total_time:.1f}s ({total_time/60.0:.1f} min)!")
    print(f"Audio files:   {audio_dir}")
    print(f"Presets:       {presets_dir}")
    print(f"Summary:       {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
