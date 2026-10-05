#!/usr/bin/env python3
"""Transcribe active bass stem slices using MuScriptor (medium model on CUDA).

Saves MIDI files to:
/run/media/kim/Mantu/surge_200k_models/real_stems_eval/midi_muscriptor/<stem_id>_muscriptor.mid
"""
import os
import sys
import time
import json
import glob
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import torch
from muscriptor import TranscriptionModel

AUDIO_DIR = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval/audio"
OUT_MIDI_DIR = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval/midi_muscriptor"
os.makedirs(OUT_MIDI_DIR, exist_ok=True)

real_wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*_real.wav")))

print(f"Loading MuScriptor medium model on CUDA (VRAM usage ~1.2 GB)...")
t0 = time.time()
model = TranscriptionModel.load_model("medium", device="cuda")
print(f"Model loaded in {time.time() - t0:.2f}s! GPU VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

print(f"\nTranscribing {len(real_wavs)} active bass stem slices with MuScriptor...")
print("-" * 80)

records = []
for idx, wav_path in enumerate(real_wavs):
    stem_file = os.path.basename(wav_path)
    stem_id = stem_file.replace("_real.wav", "")
    out_midi = os.path.join(OUT_MIDI_DIR, f"{stem_id}_muscriptor.mid")
    
    t_start = time.time()
    try:
        midi_bytes = model.transcribe_to_midi(wav_path)
        with open(out_midi, "wb") as f:
            f.write(midi_bytes)
        elapsed = time.time() - t_start
        
        # Count notes in generated MIDI
        import mido, io
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        notes = [m.note for m in mid if m.type == 'note_on' and m.velocity > 0]
        n_notes = len(notes)
        root_note = max(set(notes), key=notes.count) if notes else 0
        
        print(f"[{idx+1:02d}/{len(real_wavs)}] {stem_id:32s} | Notes: {n_notes:2d} (Root: {root_note:2d}) | {elapsed:.2f}s")
        records.append({
            "stem_id": stem_id,
            "midi_path": out_midi,
            "num_notes": n_notes,
            "root_note": root_note,
            "elapsed_s": round(elapsed, 2)
        })
    except Exception as e:
        print(f"[{idx+1:02d}/{len(real_wavs)}] {stem_id:32s} | ERROR: {e}")
        records.append({
            "stem_id": stem_id,
            "error": str(e)
        })

print("\nCleaning up MuScriptor GPU memory...")
del model
torch.cuda.empty_cache()
print(f"GPU VRAM after cleanup: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

with open(os.path.join(OUT_MIDI_DIR, "muscriptor_transcription_summary.json"), "w") as f:
    json.dump(records, f, indent=2)

print(f"\nMuScriptor transcription complete! Saved {len(records)} MIDIs to {OUT_MIDI_DIR}/")
