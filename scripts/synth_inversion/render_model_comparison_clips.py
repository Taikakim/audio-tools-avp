#!/usr/bin/env python3
"""Render side-by-side A/B comparison audio clips for today's model.

Evaluates today's best model (flow_best.pt / jepa_best.pt from overnight_realistic_bass_v2):
1. Held-out Surge XT real bass presets (unseen during training)
2. Random empirical prior bass sounds
3. 2-bar 140 BPM rolling 16th-note MIDI bassline phrases comparing ground-truth to predicted

Output directory:
/run/media/kim/Mantu/surge_200k_models/today_comparison_clips/
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import torch

from surge_spec import (
    init_synth, apply_patch, render_patch, vector_to_patch, patch_to_vector,
    DEFAULT_PLUGIN_PATH, SAMPLE_RATE, PARAM_NAMES
)
from audio_utils import MultiScaleSTFTLoss, make_mel_spec, save_patch, extract_pitch_and_envelope
from models import load_inverter
from inference import predict_vectors, vectors_to_patches
from realistic_bass_prior import RealisticBassPrior
from benchmark_ground_truth_reconstruction import GROUND_TRUTH_PATCHES, build_rolling_events

OUT_DIR = "/run/media/kim/Mantu/surge_200k_models/today_comparison_clips"
CLIPS_DIR = os.path.join(OUT_DIR, "audio")
PRESETS_DIR = os.path.join(OUT_DIR, "vstpresets")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

FLOW_CKPT = "/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2/flow_best.pt"

def main():
    print(f"Loading today's model from: {FLOW_CKPT}...")
    model = load_inverter(FLOW_CKPT, device="cpu")
    synth = init_synth(DEFAULT_PLUGIN_PATH)
    loss_fn = MultiScaleSTFTLoss()
    
    val_prior = RealisticBassPrior(split="val")
    manifold_data = np.load("/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz")
    all_names = manifold_data["preset_names"]
    is_val = manifold_data["is_val"]
    val_names = all_names[is_val]
    val_params = manifold_data["params"][is_val]
    
    records = []
    
    # =========================================================================
    # Part 1: Real Surge Held-Out Presets (Target Isolated Note + Rolling Phrase)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: Unseen Surge XT Real Bass Presets (A/B Reconstruction)")
    print("="*80)
    
    # Pick 4 diverse held-out presets: Acid, Saw, Moog, Detune
    selected_indices = [2, 4, 11, 15]  # Acid Bassline 2, Moogaroo, Sequenced Psy Bass, Lord Sawtooth
    for sel_i in selected_indices:
        preset_name = val_names[sel_i]
        clean_name = os.path.splitext(os.path.basename(preset_name))[0].replace(" ", "_").lower()
        true_vec = val_params[sel_i].copy()
        
        # Test note C2 (36)
        midi_note = 36
        note_dur = 0.22  # 16th note gate
        true_patch = vector_to_patch(true_vec)
        true_patch["midi_note"] = midi_note
        
        # 1. Render Target Single Note
        audio_gt_note = render_patch(synth, true_patch, midi_note, note_dur, duration=0.8)
        
        # 2. Invert using today's model
        mel = torch.from_numpy(make_mel_spec(audio_gt_note)).unsqueeze(0)
        vecs, _ = predict_vectors(model, mel, n_draws=8, steps=25, seed=42)
        cands = vectors_to_patches(vecs[:, 0])
        
        # Pick best draw against single note
        best_pred_patch = None
        best_l = float("inf")
        for cp in cands:
            r_aud = render_patch(synth, cp, midi_note, note_dur, duration=0.8)
            l = loss_fn(r_aud, audio_gt_note)
            if l < best_l:
                best_l = l
                best_pred_patch = cp
                
        # 3. Render Rolling 16th phrase (2 bars @ 140 BPM) for both
        events, phrase_dur = build_rolling_events(midi_note, bpm=140.0, bars=2)
        
        apply_patch(synth, true_patch)
        synth.reset()
        audio_gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_gt_phrase = np.mean(audio_gt_phrase, axis=0).astype(np.float32)
        mono_gt_phrase /= (np.max(np.abs(mono_gt_phrase)) + 1e-7)
        
        apply_patch(synth, best_pred_patch)
        synth.reset()
        audio_pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_pred_phrase = np.mean(audio_pred_phrase, axis=0).astype(np.float32)
        mono_pred_phrase /= (np.max(np.abs(mono_pred_phrase)) + 1e-7)
        
        phrase_stft = float(loss_fn(mono_pred_phrase, mono_gt_phrase))
        
        # Save Audio Files
        gt_wav = os.path.join(CLIPS_DIR, f"real_preset_{clean_name}_original.wav")
        pred_wav = os.path.join(CLIPS_DIR, f"real_preset_{clean_name}_predicted.wav")
        sf.write(gt_wav, mono_gt_phrase, SAMPLE_RATE)
        sf.write(pred_wav, mono_pred_phrase, SAMPLE_RATE)
        
        # Save VSTPresets
        save_patch(synth, best_pred_patch, os.path.join(PRESETS_DIR, f"pred_{clean_name}"), copy_to_user_dir=True)
        
        # Extract F0, envelope, and sub/harmonic metrics
        desc_gt = extract_pitch_and_envelope(mono_gt_phrase, SAMPLE_RATE)
        desc_pred = extract_pitch_and_envelope(mono_pred_phrase, SAMPLE_RATE)
        
        print(f"Preset: {preset_name}")
        print(f"  Single note STFT: {best_l:.3f} | Rolling phrase STFT: {phrase_stft:.3f}")
        print(f"  GT:   F0={desc_gt['f0_hz']} Hz ({desc_gt['f0_note']}), Active Env={desc_gt['active_dur_s']}s, Sub={desc_gt['sub_ratio']}, Harmonics={desc_gt['harmonic_richness']}")
        print(f"  PRED: F0={desc_pred['f0_hz']} Hz ({desc_pred['f0_note']}), Active Env={desc_pred['active_dur_s']}s, Sub={desc_pred['sub_ratio']}, Harmonics={desc_pred['harmonic_richness']}")
        print(f"  Original audio:  {gt_wav}")
        print(f"  Predicted audio: {pred_wav}")
        records.append({
            "category": "Real Surge Preset",
            "name": preset_name,
            "gt_wav": gt_wav,
            "pred_wav": pred_wav,
            "stft_loss": phrase_stft,
            "cutoff_gt": true_patch["cutoff"],
            "cutoff_pred": best_pred_patch["cutoff"],
            "sync_gt": true_patch["sync"],
            "sync_pred": best_pred_patch["sync"],
            "gt_descriptors": desc_gt,
            "pred_descriptors": desc_pred,
            "f0_err_hz": abs(desc_gt["f0_hz"] - desc_pred["f0_hz"]),
            "env_dur_err_s": abs(desc_gt["active_dur_s"] - desc_pred["active_dur_s"])
        })

    # =========================================================================
    # Part 2: Random Bass Sounds (Empirical Prior Draws)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: Random Bass Sounds from Empirical Prior")
    print("="*80)
    rng = np.random.RandomState(1337)
    for i in range(3):
        patch_rnd, vec_rnd, note_rnd, dur_rnd = val_prior.sample_patch_and_midi(rng)
        
        # 1. Render Target Single Note
        audio_rnd_note = render_patch(synth, patch_rnd, note_rnd, dur_rnd, duration=0.8)
        
        # 2. Invert with today's model
        mel = torch.from_numpy(make_mel_spec(audio_rnd_note)).unsqueeze(0)
        vecs, _ = predict_vectors(model, mel, n_draws=8, steps=25, seed=100 + i)
        cands = vectors_to_patches(vecs[:, 0])
        
        best_pred_patch = None
        best_l = float("inf")
        for cp in cands:
            r_aud = render_patch(synth, cp, note_rnd, dur_rnd, duration=0.8)
            l = loss_fn(r_aud, audio_rnd_note)
            if l < best_l:
                best_l = l
                best_pred_patch = cp
                
        # 3. Render Rolling Phrase
        events, phrase_dur = build_rolling_events(note_rnd, bpm=140.0, bars=2)
        
        apply_patch(synth, patch_rnd)
        synth.reset()
        audio_gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_gt_phrase = np.mean(audio_gt_phrase, axis=0).astype(np.float32)
        mono_gt_phrase /= (np.max(np.abs(mono_gt_phrase)) + 1e-7)
        
        apply_patch(synth, best_pred_patch)
        synth.reset()
        audio_pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_pred_phrase = np.mean(audio_pred_phrase, axis=0).astype(np.float32)
        mono_pred_phrase /= (np.max(np.abs(mono_pred_phrase)) + 1e-7)
        
        phrase_stft = float(loss_fn(mono_pred_phrase, mono_gt_phrase))
        
        gt_wav = os.path.join(CLIPS_DIR, f"random_bass_{i+1}_original.wav")
        pred_wav = os.path.join(CLIPS_DIR, f"random_bass_{i+1}_predicted.wav")
        sf.write(gt_wav, mono_gt_phrase, SAMPLE_RATE)
        sf.write(pred_wav, mono_pred_phrase, SAMPLE_RATE)
        
        save_patch(synth, best_pred_patch, os.path.join(PRESETS_DIR, f"pred_random_bass_{i+1}"), copy_to_user_dir=True)
        
        desc_gt = extract_pitch_and_envelope(mono_gt_phrase, SAMPLE_RATE)
        desc_pred = extract_pitch_and_envelope(mono_pred_phrase, SAMPLE_RATE)
        
        print(f"Random Sound #{i+1} (MIDI Note {note_rnd}):")
        print(f"  Single note STFT: {best_l:.3f} | Rolling phrase STFT: {phrase_stft:.3f}")
        print(f"  GT:   F0={desc_gt['f0_hz']} Hz ({desc_gt['f0_note']}), Active Env={desc_gt['active_dur_s']}s, Sub={desc_gt['sub_ratio']}, Harmonics={desc_gt['harmonic_richness']}")
        print(f"  PRED: F0={desc_pred['f0_hz']} Hz ({desc_pred['f0_note']}), Active Env={desc_pred['active_dur_s']}s, Sub={desc_pred['sub_ratio']}, Harmonics={desc_pred['harmonic_richness']}")
        print(f"  Original audio:  {gt_wav}")
        print(f"  Predicted audio: {pred_wav}")
        records.append({
            "category": "Random Empirical Bass",
            "name": f"Random Sound #{i+1} (Note {note_rnd})",
            "gt_wav": gt_wav,
            "pred_wav": pred_wav,
            "stft_loss": phrase_stft,
            "cutoff_gt": patch_rnd["cutoff"],
            "cutoff_pred": best_pred_patch["cutoff"],
            "sync_gt": patch_rnd["sync"],
            "sync_pred": best_pred_patch["sync"],
            "gt_descriptors": desc_gt,
            "pred_descriptors": desc_pred,
            "f0_err_hz": abs(desc_gt["f0_hz"] - desc_pred["f0_hz"]),
            "env_dur_err_s": abs(desc_gt["active_dur_s"] - desc_pred["active_dur_s"])
        })

    # =========================================================================
    # Part 3: Classic Archetypes (Pure Saw, Acid Resonance, Deep Sub)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 3: Classic Bass Archetypes")
    print("="*80)
    for arch in GROUND_TRUTH_PATCHES[:3]:
        arch_name = arch["name"]
        p_gt = arch["patch"]
        note = arch["midi_note"]
        bpm = arch["bpm"]
        
        # Single note
        sixteenth = (60.0 / bpm) / 4.0
        audio_note_gt = render_patch(synth, p_gt, note, note_dur=sixteenth * 0.85, duration=0.8)
        
        mel = torch.from_numpy(make_mel_spec(audio_note_gt)).unsqueeze(0)
        vecs, _ = predict_vectors(model, mel, n_draws=8, steps=25, seed=42)
        cands = vectors_to_patches(vecs[:, 0])
        
        best_p = None
        best_l = float("inf")
        for cp in cands:
            r_aud = render_patch(synth, cp, note, note_dur=sixteenth * 0.85, duration=0.8)
            l = loss_fn(r_aud, audio_note_gt)
            if l < best_l:
                best_l = l
                best_p = cp
                
        events, phrase_dur = build_rolling_events(note, bpm=bpm, bars=2)
        apply_patch(synth, p_gt)
        synth.reset()
        audio_gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_gt_phrase = np.mean(audio_gt_phrase, axis=0).astype(np.float32)
        mono_gt_phrase /= (np.max(np.abs(mono_gt_phrase)) + 1e-7)
        
        apply_patch(synth, best_p)
        synth.reset()
        audio_pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        mono_pred_phrase = np.mean(audio_pred_phrase, axis=0).astype(np.float32)
        mono_pred_phrase /= (np.max(np.abs(mono_pred_phrase)) + 1e-7)
        
        phrase_stft = float(loss_fn(mono_pred_phrase, mono_gt_phrase))
        
        gt_wav = os.path.join(CLIPS_DIR, f"archetype_{arch_name}_original.wav")
        pred_wav = os.path.join(CLIPS_DIR, f"archetype_{arch_name}_predicted.wav")
        sf.write(gt_wav, mono_gt_phrase, SAMPLE_RATE)
        sf.write(pred_wav, mono_pred_phrase, SAMPLE_RATE)
        
        save_patch(synth, best_p, os.path.join(PRESETS_DIR, f"pred_{arch_name}"), copy_to_user_dir=True)
        
        desc_gt = extract_pitch_and_envelope(mono_gt_phrase, SAMPLE_RATE)
        desc_pred = extract_pitch_and_envelope(mono_pred_phrase, SAMPLE_RATE)
        
        print(f"Archetype: {arch_name}")
        print(f"  Single note STFT: {best_l:.3f} | Rolling phrase STFT: {phrase_stft:.3f}")
        print(f"  GT:   F0={desc_gt['f0_hz']} Hz ({desc_gt['f0_note']}), Active Env={desc_gt['active_dur_s']}s, Sub={desc_gt['sub_ratio']}, Harmonics={desc_gt['harmonic_richness']}")
        print(f"  PRED: F0={desc_pred['f0_hz']} Hz ({desc_pred['f0_note']}), Active Env={desc_pred['active_dur_s']}s, Sub={desc_pred['sub_ratio']}, Harmonics={desc_pred['harmonic_richness']}")
        print(f"  Original audio:  {gt_wav}")
        print(f"  Predicted audio: {pred_wav}")
        records.append({
            "category": "Classic Archetype",
            "name": arch["description"],
            "gt_wav": gt_wav,
            "pred_wav": pred_wav,
            "stft_loss": phrase_stft,
            "cutoff_gt": p_gt["cutoff"],
            "cutoff_pred": best_p["cutoff"],
            "sync_gt": p_gt.get("sync", 0.0),
            "sync_pred": best_p.get("sync", 0.0),
            "gt_descriptors": desc_gt,
            "pred_descriptors": desc_pred,
            "f0_err_hz": abs(desc_gt["f0_hz"] - desc_pred["f0_hz"]),
            "env_dur_err_s": abs(desc_gt["active_dur_s"] - desc_pred["active_dur_s"])
        })

    # Save summary metadata
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(records, f, indent=2)
        
    print("\n" + "="*80)
    print(f"ALL COMPARISON CLIPS RENDERED SUCCESSFULLY!")
    print(f"Clips directory:   {CLIPS_DIR}/")
    print(f"Presets directory: {PRESETS_DIR}/")
    print(f"User DAW folder:   ~/Documents/Surge XT/Patches/AI Inversions/")
    print("="*80)

if __name__ == "__main__":
    main()
