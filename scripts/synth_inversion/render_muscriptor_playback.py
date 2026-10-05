#!/usr/bin/env python3
"""Synthesize inverted Surge XT patches using MuScriptor transcribed MIDIs.

Compares MuScriptor playback vs Kim's MIR pipeline (BassMidiPipeline) playback:
- Note count and root note agreement
- Multi-scale STFT loss against the real audio stem
- Spectral centroid error
"""
import os
import sys
import json
import glob
from collections import Counter
import soundfile as sf
import numpy as np
import librosa
import mido

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from surge_spec import init_synth, apply_patch, DEFAULT_PLUGIN_PATH, SAMPLE_RATE
from audio_utils import MultiScaleSTFTLoss
import torch

BASE_DIR = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
PRESETS_DIR = os.path.join(BASE_DIR, "vstpresets")
MUSCRIPTOR_MIDI_DIR = os.path.join(BASE_DIR, "midi_muscriptor")
MIR_MIDI_DIR = os.path.join(BASE_DIR, "midi")

synth = init_synth(DEFAULT_PLUGIN_PATH)
stft_loss_fn = MultiScaleSTFTLoss()

real_wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*_real.wav")))

print(f"Comparing MuScriptor vs MIR Bass Pipeline Playback across {len(real_wavs)} stems...")
print("=" * 95)
print(f"{'Stem ID':32s} | {'Notes (Mu / MIR)':18s} | {'STFT (Mu / MIR)':18s} | {'Centroid (Real/Mu/MIR)':22s}")
print("-" * 95)

comparison_results = []

for wav_path in real_wavs:
    stem_file = os.path.basename(wav_path)
    stem_id = stem_file.replace("_real.wav", "")
    
    # 1. Load real audio
    y_real, sr = sf.read(wav_path)
    dur = len(y_real) / sr
    if y_real.ndim > 1:
        y_real_mono = np.mean(y_real, axis=1)
    else:
        y_real_mono = y_real
    sc_real = float(np.mean(librosa.feature.spectral_centroid(y=y_real_mono, sr=sr)))
    
    # 2. Load inverted patch state
    state_file = os.path.join(PRESETS_DIR, f"Inverted_{stem_id}.pedalboard_state")
    if not os.path.exists(state_file):
        continue
    with open(state_file, "rb") as f:
        synth.raw_state = f.read()
        
    # 3. Load MuScriptor MIDI
    muscriptor_midi_path = os.path.join(MUSCRIPTOR_MIDI_DIR, f"{stem_id}_muscriptor.mid")
    mu_events = []
    mu_pitches = []
    if os.path.exists(muscriptor_midi_path):
        mid_mu = mido.MidiFile(muscriptor_midi_path)
        t = 0.0
        for msg in mid_mu:
            t += msg.time
            if t > dur:
                break
            if msg.type in ('note_on', 'note_off'):
                mu_events.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=t))
                if msg.type == 'note_on' and msg.velocity > 0:
                    mu_pitches.append(msg.note)
        mu_events.sort(key=lambda m: m.time)
        
    # 4. Load MIR playback audio for direct metric comparison
    mir_wav_path = os.path.join(AUDIO_DIR, f"{stem_id}_midi_playback.wav")
    y_mir, _ = sf.read(mir_wav_path)
    if y_mir.ndim > 1: y_mir_mono = np.mean(y_mir, axis=1)
    else: y_mir_mono = y_mir
    sc_mir = float(np.mean(librosa.feature.spectral_centroid(y=y_mir_mono, sr=sr)))
    
    # 5. Render MuScriptor playback
    synth.reset()
    if mu_events:
        mu_audio = synth.process(mu_events, duration=dur, sample_rate=SAMPLE_RATE, num_channels=2)
    else:
        mu_audio = np.zeros_like(y_real.T if y_real.ndim > 1 else np.stack([y_real, y_real]))
    mu_norm = (mu_audio / (np.max(np.abs(mu_audio)) + 1e-7)).astype(np.float32)
    
    mu_wav_path = os.path.join(AUDIO_DIR, f"{stem_id}_muscriptor_playback.wav")
    sf.write(mu_wav_path, mu_norm.T, SAMPLE_RATE)
    
    mu_mono = np.mean(mu_norm, axis=0)
    sc_mu = float(np.mean(librosa.feature.spectral_centroid(y=mu_mono, sr=sr)))
    
    # 6. Compute STFT distance to real stem for both playbacks
    min_len = min(len(y_real_mono), len(mu_mono), len(y_mir_mono))
    t_real = torch.from_numpy(y_real_mono[:min_len]).unsqueeze(0).unsqueeze(0).float()
    t_mu = torch.from_numpy(mu_mono[:min_len]).unsqueeze(0).unsqueeze(0).float()
    t_mir = torch.from_numpy(y_mir_mono[:min_len]).unsqueeze(0).unsqueeze(0).float()
    
    stft_mu = float(stft_loss_fn(t_mu, t_real))
    stft_mir = float(stft_loss_fn(t_mir, t_real))
    
    # 7. Note counts
    mir_midi_path = os.path.join(MIR_MIDI_DIR, f"{stem_id}_transcribed.mid")
    mir_pitches = []
    if os.path.exists(mir_midi_path):
        mid_mir = mido.MidiFile(mir_midi_path)
        mir_pitches = [m.note for m in mid_mir if m.type == 'note_on' and m.velocity > 0]
        
    n_mu = len(mu_pitches)
    n_mir = len(mir_pitches)
    root_mu = Counter(mu_pitches).most_common(1)[0][0] if mu_pitches else 0
    root_mir = Counter(mir_pitches).most_common(1)[0][0] if mir_pitches else 0
    
    record = {
        "stem_id": stem_id,
        "notes_muscriptor": n_mu,
        "notes_mir": n_mir,
        "root_muscriptor": root_mu,
        "root_mir": root_mir,
        "stft_loss_muscriptor": round(stft_mu, 3),
        "stft_loss_mir": round(stft_mir, 3),
        "centroid_real": round(sc_real, 1),
        "centroid_muscriptor": round(sc_mu, 1),
        "centroid_mir": round(sc_mir, 1),
        "audio_muscriptor_playback": mu_wav_path
    }
    comparison_results.append(record)
    
    notes_str = f"{n_mu:2d}(N{root_mu:2d}) / {n_mir:2d}(N{root_mir:2d})"
    stft_str = f"{stft_mu:6.3f} / {stft_mir:6.3f}"
    sc_str = f"{sc_real:4.0f} / {sc_mu:4.0f} / {sc_mir:4.0f}"
    print(f"{stem_id:32s} | {notes_str:18s} | {stft_str:18s} | {sc_str:22s}")

with open(os.path.join(BASE_DIR, "muscriptor_vs_mir_comparison.json"), "w") as f:
    json.dump(comparison_results, f, indent=2)

print("\nComparison saved to /run/media/kim/Mantu/surge_200k_models/real_stems_eval/muscriptor_vs_mir_comparison.json")
