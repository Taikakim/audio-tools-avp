#!/usr/bin/env python3
import sys
import os
import time
import json
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from models import load_inverter
from surge_spec import init_synth, apply_patch, render_patch, vector_to_patch, DEFAULT_PLUGIN_PATH, SAMPLE_RATE
from audio_utils import make_mel_spec, save_patch, extract_pitch_and_envelope
from inference import predict_and_rerank_candidates
from benchmark_ground_truth_reconstruction import GROUND_TRUTH_PATCHES, build_rolling_events
from realistic_bass_prior import RealisticBassPrior

OUT_DIR = "/run/media/kim/Mantu/surge_200k_models/batch_cpu_reranked_eval"
CLIPS_DIR = os.path.join(OUT_DIR, "audio")
PRESETS_DIR = os.path.join(OUT_DIR, "vstpresets")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

print("Loading model onto CPU for multi-clip benchmark...")
model = load_inverter("/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64/flow_latest.pt", device="cpu")
synth = init_synth(DEFAULT_PLUGIN_PATH)

manifold_data = np.load("/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz")
all_names = manifold_data["preset_names"]
is_val = manifold_data["is_val"]
val_names = all_names[is_val]
val_params = manifold_data["params"][is_val]

val_prior = RealisticBassPrior(split="val")
records = []

# 1. Classic Archetypes
print("\n=== Part 1: Classic Archetypes ===")
for arch in GROUND_TRUTH_PATCHES[:3]:
    name = arch["name"]
    p_gt = arch["patch"]
    note = arch["midi_note"]
    bpm = arch["bpm"]
    sixteenth = (60.0 / bpm) / 4.0
    note_dur = sixteenth * 0.85

    target_note = render_patch(synth, p_gt, note, note_dur=note_dur, duration=0.8)
    mel = torch.from_numpy(make_mel_spec(target_note)).unsqueeze(0)

    t0 = time.time()
    best, t_desc, _ = predict_and_rerank_candidates(
        model, synth, target_note, mel, note, note_dur=note_dur,
        n_candidates=20, steps=25, seed=42
    )
    t_rank = time.time() - t0

    events, phrase_dur = build_rolling_events(note, bpm=bpm, bars=2)
    apply_patch(synth, p_gt)
    synth.reset()
    gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    gt_norm = (gt_phrase / (np.max(np.abs(gt_phrase)) + 1e-7)).astype(np.float32)

    apply_patch(synth, best["patch"])
    synth.reset()
    pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    pred_norm = (pred_phrase / (np.max(np.abs(pred_phrase)) + 1e-7)).astype(np.float32)

    gt_wav = os.path.join(CLIPS_DIR, f"arch_{name}_gt.wav")
    pred_wav = os.path.join(CLIPS_DIR, f"arch_{name}_pred.wav")
    sf.write(gt_wav, gt_norm.T, SAMPLE_RATE)
    sf.write(pred_wav, pred_norm.T, SAMPLE_RATE)
    save_patch(synth, best["patch"], os.path.join(PRESETS_DIR, f"pred_arch_{name}"), copy_to_user_dir=True)

    desc_gt = extract_pitch_and_envelope(gt_norm, SAMPLE_RATE)
    desc_pr = extract_pitch_and_envelope(pred_norm, SAMPLE_RATE)

    stft_val = best["stft_loss"]
    tot_score = best["total_score"]
    f0_gt, f0_pr = desc_gt["f0_hz"], desc_pr["f0_hz"]
    env_gt, env_pr = desc_gt["active_dur_s"], desc_pr["active_dur_s"]
    w_gt, w_pr = desc_gt["stereo_width"], desc_pr["stereo_width"]
    mod_gt, mod_pr = desc_gt["mod_speed_hz"], desc_pr["mod_speed_hz"]

    rec = {
        "name": name,
        "category": "archetype",
        "gt_wav": gt_wav,
        "pred_wav": pred_wav,
        "stft_loss": stft_val,
        "total_score": tot_score,
        "f0_gt": f0_gt,
        "f0_pred": f0_pr,
        "env_gt_s": env_gt,
        "env_pred_s": env_pr,
        "stereo_gt": w_gt,
        "stereo_pred": w_pr,
        "mod_gt_hz": mod_gt,
        "mod_pred_hz": mod_pr
    }
    records.append(rec)
    print(f"[{name:20s}] STFT: {stft_val:.3f} | F0: {f0_gt}->{f0_pr} Hz | Env: {env_gt}s->{env_pr}s | Stereo: {w_gt}->{w_pr} | Mod: {mod_gt}->{mod_pr} Hz")

# 2. All 17 Held-out presets
print(f"\n=== Part 2: Rendering {len(val_names)} Held-out Real Presets ===")
for idx, (p_name, vec) in enumerate(zip(val_names, val_params)):
    clean = os.path.splitext(os.path.basename(p_name))[0].replace(" ", "_").lower()
    true_patch = vector_to_patch(vec.copy())
    midi_note = 36
    note_dur = 0.22

    target_note = render_patch(synth, true_patch, midi_note, note_dur, duration=0.8)
    mel = torch.from_numpy(make_mel_spec(target_note)).unsqueeze(0)

    best, t_desc, _ = predict_and_rerank_candidates(
        model, synth, target_note, mel, midi_note, note_dur=note_dur,
        n_candidates=20, steps=25, seed=100 + idx
    )

    events, phrase_dur = build_rolling_events(midi_note, bpm=140.0, bars=2)
    apply_patch(synth, true_patch)
    synth.reset()
    gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    gt_norm = (gt_phrase / (np.max(np.abs(gt_phrase)) + 1e-7)).astype(np.float32)

    apply_patch(synth, best["patch"])
    synth.reset()
    pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    pred_norm = (pred_phrase / (np.max(np.abs(pred_phrase)) + 1e-7)).astype(np.float32)

    gt_wav = os.path.join(CLIPS_DIR, f"heldout_{idx+1:02d}_{clean}_gt.wav")
    pred_wav = os.path.join(CLIPS_DIR, f"heldout_{idx+1:02d}_{clean}_pred.wav")
    sf.write(gt_wav, gt_norm.T, SAMPLE_RATE)
    sf.write(pred_wav, pred_norm.T, SAMPLE_RATE)
    save_patch(synth, best["patch"], os.path.join(PRESETS_DIR, f"pred_heldout_{idx+1:02d}_{clean}"), copy_to_user_dir=True)

    desc_gt = extract_pitch_and_envelope(gt_norm, SAMPLE_RATE)
    desc_pr = extract_pitch_and_envelope(pred_norm, SAMPLE_RATE)

    stft_val = best["stft_loss"]
    tot_score = best["total_score"]
    f0_gt, f0_pr = desc_gt["f0_hz"], desc_pr["f0_hz"]
    env_gt, env_pr = desc_gt["active_dur_s"], desc_pr["active_dur_s"]
    w_gt, w_pr = desc_gt["stereo_width"], desc_pr["stereo_width"]
    mod_gt, mod_pr = desc_gt["mod_speed_hz"], desc_pr["mod_speed_hz"]

    rec = {
        "index": idx + 1,
        "name": clean,
        "category": "heldout_preset",
        "gt_wav": gt_wav,
        "pred_wav": pred_wav,
        "stft_loss": stft_val,
        "total_score": tot_score,
        "f0_gt": f0_gt,
        "f0_pred": f0_pr,
        "env_gt_s": env_gt,
        "env_pred_s": env_pr,
        "stereo_gt": w_gt,
        "stereo_pred": w_pr,
        "mod_gt_hz": mod_gt,
        "mod_pred_hz": mod_pr
    }
    records.append(rec)
    print(f"[{idx+1:02d}/{len(val_names)}] {clean:25s} | STFT: {stft_val:.3f} | F0: {f0_gt}->{f0_pr} Hz | Env: {env_gt}s->{env_pr}s | Stereo: {w_gt}->{w_pr} | Mod: {mod_gt}->{mod_pr} Hz")

# 3. 10 Random Empirical Bass sounds
print("\n=== Part 3: Rendering 10 Random Empirical Bass Sounds ===")
rng = np.random.RandomState(42)
for i in range(10):
    p_rnd, v_rnd, n_rnd, d_rnd = val_prior.sample_patch_and_midi(rng)
    target_note = render_patch(synth, p_rnd, n_rnd, d_rnd, duration=0.8)
    mel = torch.from_numpy(make_mel_spec(target_note)).unsqueeze(0)

    best, t_desc, _ = predict_and_rerank_candidates(
        model, synth, target_note, mel, n_rnd, note_dur=d_rnd,
        n_candidates=20, steps=25, seed=200 + i
    )

    events, phrase_dur = build_rolling_events(n_rnd, bpm=140.0, bars=2)
    apply_patch(synth, p_rnd)
    synth.reset()
    gt_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    gt_norm = (gt_phrase / (np.max(np.abs(gt_phrase)) + 1e-7)).astype(np.float32)

    apply_patch(synth, best["patch"])
    synth.reset()
    pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
    pred_norm = (pred_phrase / (np.max(np.abs(pred_phrase)) + 1e-7)).astype(np.float32)

    gt_wav = os.path.join(CLIPS_DIR, f"random_{i+1:02d}_gt.wav")
    pred_wav = os.path.join(CLIPS_DIR, f"random_{i+1:02d}_pred.wav")
    sf.write(gt_wav, gt_norm.T, SAMPLE_RATE)
    sf.write(pred_wav, pred_norm.T, SAMPLE_RATE)
    save_patch(synth, best["patch"], os.path.join(PRESETS_DIR, f"pred_random_{i+1:02d}"), copy_to_user_dir=True)

    desc_gt = extract_pitch_and_envelope(gt_norm, SAMPLE_RATE)
    desc_pr = extract_pitch_and_envelope(pred_norm, SAMPLE_RATE)

    stft_val = best["stft_loss"]
    tot_score = best["total_score"]
    f0_gt, f0_pr = desc_gt["f0_hz"], desc_pr["f0_hz"]
    env_gt, env_pr = desc_gt["active_dur_s"], desc_pr["active_dur_s"]
    w_gt, w_pr = desc_gt["stereo_width"], desc_pr["stereo_width"]
    mod_gt, mod_pr = desc_gt["mod_speed_hz"], desc_pr["mod_speed_hz"]

    rec = {
        "index": i + 1,
        "name": f"random_bass_{i+1:02d}",
        "category": "random_empirical",
        "gt_wav": gt_wav,
        "pred_wav": pred_wav,
        "stft_loss": stft_val,
        "total_score": tot_score,
        "f0_gt": f0_gt,
        "f0_pred": f0_pr,
        "env_gt_s": env_gt,
        "env_pred_s": env_pr,
        "stereo_gt": w_gt,
        "stereo_pred": w_pr,
        "mod_gt_hz": mod_gt,
        "mod_pred_hz": mod_pr
    }
    records.append(rec)
    print(f"[RND {i+1:02d}/10] Note {n_rnd:2d} | STFT: {stft_val:.3f} | F0: {f0_gt}->{f0_pr} Hz | Env: {env_gt}s->{env_pr}s | Stereo: {w_gt}->{w_pr} | Mod: {mod_gt}->{mod_pr} Hz")

with open(os.path.join(OUT_DIR, "eval_summary.json"), "w") as f:
    json.dump(records, f, indent=2)

print("\n" + "="*80)
print(f"BATCH EVALUATION COMPLETE: {len(records)} pairs rendered to {CLIPS_DIR}/")
print("="*80)
