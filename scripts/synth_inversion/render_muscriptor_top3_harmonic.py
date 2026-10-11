#!/usr/bin/env python3
"""Render top 3 raw model predictions per stem with MuScriptor MIDI playback.

Features:
1. No phrase refinement stage (evaluates the raw model predictions directly, blazingly fast).
2. Uses curated MuScriptor MIDIs for accurate musical notes and phrasing.
3. For each candidate (1, 2, 3), renders:
   - Raw model prediction.
   - Harmonic snapped version: Oscillator ratios locked to {2:1, 1:1, 1:2, 1:3},
     with unmodulated static Sync, FM, and Ring Mod zeroed to eliminate static inharmonic clang.
4. Generates side-by-side WAV files, Steinberg VST3 presets, and detailed JSON summary.
"""
import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from models_v3 import load_inverter
from surge_spec_v3 import (
    init_synth, apply_patch, flush_synth, DEFAULT_PLUGIN_PATH,
    SAMPLE_RATE, AUDIO_LEN, LP_FILTERS, WAVESHAPER_TYPES
)
from audio_utils import make_mel_spec, save_patch, extract_pitch_and_envelope, MultiScaleSTFTLoss
from inference_v3 import predict_and_rerank_candidates
from invert_stem_collection_v3_cpu import (
    STEM_CATALOG, resolve_mantu, find_active_bass_phrase, load_pedalboard_events_from_midi
)

RATIO_PRESETS = {
    "1:1": {"osc_1_octave": 0.5, "osc_2_octave": 0.5, "osc_2_pitch_raw": 0.5, "semitones": 0.0},
    "2:1": {"osc_1_octave": 0.5, "osc_2_octave": 0.66667, "osc_2_pitch_raw": 0.5, "semitones": 12.0},
    "1:2": {"osc_1_octave": 0.5, "osc_2_octave": 0.33333, "osc_2_pitch_raw": 0.5, "semitones": -12.0},
    "1:3": {"osc_1_octave": 0.5, "osc_2_octave": 0.33333, "osc_2_pitch_raw": 0.0, "semitones": -19.0},
}

def snap_to_harmonic_ratio(patch: dict) -> tuple:
    """Return a modified copy of patch with oscillator ratios locked to {2:1, 1:1, 1:2, 1:3}
    and unmodulated static sync/FM/ring mod zeroed."""
    p = dict(patch)
    # Osc 1 is fundamental: lock to 0.5 (octave 0)
    p["osc_1_octave"] = 0.5

    # Estimate current relative semitone offset of Osc 2
    # osc_octave: 0.0=-3, 0.5=0, 1.0=+3 => octave = (raw - 0.5) * 6
    pred_oct_2 = (float(p.get("osc_2_octave", 0.5)) - 0.5) * 6.0
    pred_semi = pred_oct_2 * 12.0

    # Find closest ratio
    best_name = "1:1"
    min_dist = float("inf")
    for name, r in RATIO_PRESETS.items():
        dist = abs(pred_semi - r["semitones"])
        if dist < min_dist:
            min_dist = dist
            best_name = name

    target = RATIO_PRESETS[best_name]
    p["osc_2_octave"] = target["osc_2_octave"]
    p["osc_2_pitch_raw"] = target["osc_2_pitch_raw"]
    p["harmonic_ratio"] = best_name

    # Tame unmodulated static sync, FM depth, and ring modulation to avoid harsh non-musical wash
    p["sync"] = 0.0
    p["fm_depth"] = 0.0
    p["ring_1x2_volume"] = 0.0
    p["ring_2x3_volume"] = 0.0

    return p, best_name


def apply_patch_with_pitch(synth, p: dict):
    apply_patch(synth, p)
    if "osc_2_pitch_raw" in p and "a_osc_2_pitch" in synth.parameters:
        synth.parameters["a_osc_2_pitch"].raw_value = p["osc_2_pitch_raw"]


def main():
    flow_ckpt = resolve_mantu("/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a0/flow_best.pt")
    out_dir = resolve_mantu("/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a0/eval_cpu_muscriptor_top3")
    muscriptor_midi_dir = resolve_mantu("/run/media/kim/Mantu2/surge_200k_models/real_stems_eval/midi_muscriptor")

    audio_dir = os.path.join(out_dir, "audio")
    presets_dir = os.path.join(out_dir, "vstpresets")
    for d in [audio_dir, presets_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print("SURGE XT REAL-STEM EVALUATION: TOP 3 RAW PREDICTIONS + HARMONIC RATIOS")
    print(f"Model: {flow_ckpt}")
    print(f"Output: {out_dir}")
    print(f"Playback MIDI: MuScriptor from {muscriptor_midi_dir}")
    print("Refinement: NONE (raw model predictions)")
    print("=" * 80)

    print("Loading Flow model onto CPU...")
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

        # Slice 4-bar or 2-bar phrase
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

        # Save reference stem audio
        ref_wav_path = os.path.join(audio_dir, f"{stem_id}_real.wav")
        sf.write(ref_wav_path, gt_norm.T, SAMPLE_RATE)

        # Load MuScriptor MIDI
        mu_midi_path = os.path.join(muscriptor_midi_dir, f"{stem_id}_muscriptor.mid")
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

        # Invert via Flow matching on CPU (20 candidates, top 3 extracted)
        t_inv0 = time.time()
        best, desc, scored_cands = predict_and_rerank_candidates(
            model, synth, target_note, mel, root_midi, note_dur=note_dur,
            n_candidates=20, steps=25, seed=42 + idx
        )
        inv_elapsed = time.time() - t_inv0

        stem_record = {
            "index": idx + 1,
            "id": stem_id,
            "name": name,
            "style": style,
            "bpm": bpm,
            "root_midi": root_midi,
            "phrase_dur_s": round(phrase_dur, 2),
            "num_notes": len(note_on_pitches),
            "candidates": []
        }

        top3 = scored_cands[:3]
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

            raw_wav = os.path.join(audio_dir, f"{stem_id}_cand{rank}_raw.wav")
            sf.write(raw_wav, pred_raw_norm.T, SAMPLE_RATE)
            save_patch(synth, raw_patch, os.path.join(presets_dir, f"{stem_id}_cand{rank}_raw"), copy_to_user_dir=False)

            # 2. Render HARMONIC candidate playback
            flush_synth(synth)
            apply_patch_with_pitch(synth, harm_patch)
            pred_harm = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
            flush_synth(synth)
            pred_harm_norm = (pred_harm / (np.max(np.abs(pred_harm)) + 1e-7)).astype(np.float32)

            harm_wav = os.path.join(audio_dir, f"{stem_id}_cand{rank}_harmonic_{ratio_name.replace(':', '_')}.wav")
            sf.write(harm_wav, pred_harm_norm.T, SAMPLE_RATE)
            save_patch(synth, harm_patch, os.path.join(presets_dir, f"{stem_id}_cand{rank}_harmonic"), copy_to_user_dir=False)

            # Metrics against reference
            min_l = min(gt_norm.shape[1], pred_raw_norm.shape[1])
            stft_raw = loss_fn(np.mean(pred_raw_norm[:, :min_l], axis=0), np.mean(gt_norm[:, :min_l], axis=0))
            stft_harm = loss_fn(np.mean(pred_harm_norm[:, :min_l], axis=0), np.mean(gt_norm[:, :min_l], axis=0))

            cand_info = {
                "rank": rank,
                "stft_single_note": round(float(cand["stft_loss"]), 3),
                "stft_phrase_raw": round(float(stft_raw), 3),
                "stft_phrase_harmonic": round(float(stft_harm), 3),
                "harmonic_ratio": ratio_name,
                "cutoff": round(float(raw_patch["cutoff"]), 3),
                "resonance": round(float(raw_patch["resonance"]), 3),
                "feg_decay": round(float(raw_patch["feg_decay"]), 3),
                "aeg_decay": round(float(raw_patch["aeg_decay"]), 3),
                "raw_wav": raw_wav,
                "harm_wav": harm_wav,
            }
            stem_record["candidates"].append(cand_info)

        summary_records.append(stem_record)
        c1 = stem_record["candidates"][0]
        print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] {name:30s} | Notes: {len(note_on_pitches):2d} | C1 Cut: {c1['cutoff']:.2f} Res: {c1['resonance']:.2f} Ratio: {c1['harmonic_ratio']} | STFT: {c1['stft_phrase_raw']:.2f} (Harm: {c1['stft_phrase_harmonic']:.2f}) | {inv_elapsed:.1f}s")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    total_time = time.time() - t_start_all
    print("=" * 80)
    print(f"ALL 24 STEMS COMPLETE in {total_time:.1f}s (~{total_time/60:.1f} min)!")
    print(f"Audio files:   {audio_dir}/")
    print(f"Presets:       {presets_dir}/")
    print(f"Summary:       {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
