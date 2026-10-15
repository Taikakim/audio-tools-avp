#!/usr/bin/env python3
# Owner: ANTIGRAVITY (created 2026-10-12)
"""Unified multi-temperature evaluation renderer for Real Bass Inversion Arms (A2, A3).

Features:
- Models: Both EMA (flow_ema_best.pt) and Online (flow_online_best.pt)
- Encoders: Supports models_v4 TemporalAudioEncoderV4 (ASP)
- Temperatures: tau in [0.30, 0.50, 0.70, 0.85, 1.00] (0.15 dropped, 0.85 added per Kim's request)
- MIDI Filtering: Strictly strips Channel 9 percussion (fixes hi-hat root hijack on stems 12 & 14)
- Playback: Curated MuScriptor MIDI playback across 24 real bass stems
- Musical Snapping: Renders both Raw and Musical Snapped (musical_snapping.py)
- Device: CPU-only (safe, deterministic)
"""

import os
import sys
import time
import json
import argparse
import subprocess
import mido
import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from models_v4 import load_inverter_v4
from surge_spec_v3 import (
    init_synth, apply_patch, flush_synth, DEFAULT_PLUGIN_PATH,
    SAMPLE_RATE, AUDIO_LEN
)
from audio_utils import make_mel_spec, save_patch, MultiScaleSTFTLoss
from inference_v3 import predict_and_rerank_candidates
from invert_stem_collection_v3_cpu import (
    STEM_CATALOG, resolve_mantu, find_active_bass_phrase
)
from musical_snapping import apply_musical_snapping


def load_clean_pedalboard_events_from_midi(midi_path, max_dur=None):
    """Parse MIDI file and convert to pedalboard-compatible mido Messages with absolute seconds.
    CRITICAL FIX: Strips Channel 9 (General MIDI drum/percussion channel) so hi-hats never
    pollute pitch estimation or voice allocation.
    """
    mid = mido.MidiFile(midi_path)
    events = []
    t = 0.0
    active_notes = {}
    for msg in mid:
        t += msg.time
        if max_dur is not None and t >= max_dur:
            break
        # Ignore MIDI Channel 9 (drums/percussion)
        if getattr(msg, 'channel', 0) == 9:
            continue
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append(mido.Message('note_on', note=msg.note, velocity=msg.velocity, time=t))
            active_notes[msg.note] = t
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            events.append(mido.Message('note_off', note=msg.note, velocity=0, time=t))
            active_notes.pop(msg.note, None)

    cutoff_time = max_dur - 0.005 if max_dur is not None else t
    for note in list(active_notes.keys()):
        events.append(mido.Message('note_off', note=note, velocity=0, time=cutoff_time))

    events.sort(key=lambda m: m.time)
    return events


def snap_to_harmonic_ratio(patch: dict) -> tuple:
    """Applies Kim's musical snapping rules:
    - RM ratios locked to integer octaves {-24, -12, 0, +12, +24} st.
    - FM ratios locked to whole number (octave), 5th (3:2), or 4th (4:3).
    - Sync couples Filter EG into sync with sustain=0% unless exact octave.
    - For sync > 7 st, scales FEG decay length with sync strength.
    """
    p, report = apply_musical_snapping(patch)
    mode_str = ", ".join(f"{k}={v}" for k, v in report.items()) if report else "clean"
    p["harmonic_ratio"] = mode_str
    return p, mode_str


def apply_patch_with_pitch(synth, p: dict):
    apply_patch(synth, p)
    if "osc_2_pitch_raw" in p and "a_osc_2_pitch" in synth.parameters:
        synth.parameters["a_osc_2_pitch"].raw_value = p["osc_2_pitch_raw"]
    if "osc_3_pitch_raw" in p and "a_osc_3_pitch" in synth.parameters:
        synth.parameters["a_osc_3_pitch"].raw_value = p["osc_3_pitch_raw"]
    if p.get("sync_static_floor") == 0.0 and "a_osc_1_sync" in synth.parameters:
        synth.parameters["a_osc_1_sync"].raw_value = 0.0


def format_tau_tag(tau: float) -> str:
    if tau == 1.0:
        return "t10"
    elif tau == 0.85:
        return "t085"
    elif tau == 0.7:
        return "t07"
    elif tau == 0.5:
        return "t05"
    elif tau == 0.3:
        return "t03"
    elif tau == 0.15:
        return "t015"
    else:
        return f"t{int(round(tau * 100)):03d}"


def render_arm(arm_dir: str, out_dir: str, midi_dir: str, temps: list, stem_filter=None):
    arm_dir = resolve_mantu(arm_dir)
    out_dir = resolve_mantu(out_dir)
    midi_dir = resolve_mantu(midi_dir)

    audio_dir = os.path.join(out_dir, "audio")
    presets_dir = os.path.join(out_dir, "vstpresets")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(presets_dir, exist_ok=True)

    ckpt_specs = [
        ("ema", os.path.join(arm_dir, "flow_ema_best.pt")),
        ("online", os.path.join(arm_dir, "flow_online_best.pt"))
    ]

    print("=" * 80)
    print(f"RENDERING ARM EVALUATION: {os.path.basename(arm_dir.rstrip('/'))}")
    print(f"Models: {[s[0] for s in ckpt_specs]} from {arm_dir}")
    print(f"Output: {out_dir}")
    print(f"Temperatures: {temps}")
    print("Candidates: Top 1 Best Prediction per setting (raw + harmonic snapped)")
    print("Execution: CPU-only")
    print("=" * 80)

    synth = init_synth(DEFAULT_PLUGIN_PATH)
    loss_fn = MultiScaleSTFTLoss()

    all_results = {}
    t_start_all = time.time()

    for kind, ckpt_path in ckpt_specs:
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n>>> Loading {kind.upper()} model ({ckpt_path}) onto CPU...")
        model = load_inverter_v4(ckpt_path, device="cpu")
        all_results[kind] = {}

        stems_to_render = STEM_CATALOG
        if stem_filter:
            stems_to_render = [s for s in STEM_CATALOG if stem_filter in s["id"]]

        for idx, item in enumerate(stems_to_render):
            stem_id = item["id"]
            name = item["name"]
            path = resolve_mantu(item["path"])
            bpm = item["bpm"]
            style = item["style"]

            if not os.path.exists(path):
                print(f"[{idx+1:02d}/{len(stems_to_render)}] Missing audio: {path}")
                continue

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

            ref_wav_path = os.path.join(audio_dir, f"{stem_id}_real.wav")
            if not os.path.exists(ref_wav_path):
                sf.write(ref_wav_path, gt_norm.T, SAMPLE_RATE)

            mu_midi_path = os.path.join(midi_dir, f"{stem_id}_muscriptor.mid")
            if not os.path.exists(mu_midi_path):
                print(f"[{idx+1:02d}/{len(stems_to_render)}] Missing MuScriptor MIDI: {mu_midi_path}")
                continue

            events = load_clean_pedalboard_events_from_midi(mu_midi_path, max_dur=phrase_dur)
            note_on_pitches = [m.note for m in events if m.type == 'note_on' and m.velocity > 0]
            root_midi = max(set(note_on_pitches), key=note_on_pitches.count) if note_on_pitches else 36

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

            if stem_id not in all_results[kind]:
                all_results[kind][stem_id] = {
                    "name": name,
                    "style": style,
                    "bpm": bpm,
                    "root_midi": root_midi,
                    "temps": {}
                }

            print(f"[{kind.upper():6s} {idx+1:02d}/{len(stems_to_render)}] {name:28s} | Root: {root_midi}")
            synth = init_synth(DEFAULT_PLUGIN_PATH)

            for tau in temps:
                t_tag = format_tau_tag(tau)
                raw_wav_name = f"{stem_id}_{kind}_{t_tag}_raw.wav"
                harm_wav_name = f"{stem_id}_{kind}_{t_tag}_harm.wav"
                raw_wav = os.path.join(audio_dir, raw_wav_name)
                harm_wav = os.path.join(audio_dir, harm_wav_name)

                if os.path.exists(raw_wav) and os.path.exists(harm_wav):
                    print(f"   [{t_tag} tau={tau:.2f}] Exists, skipping render.")
                    continue

                t0 = time.time()
                try:
                    best, desc, scored_cands = predict_and_rerank_candidates(
                        model, synth, target_note, mel, root_midi, note_dur=note_dur,
                        n_candidates=20, steps=25, seed=42 + idx, tau=tau
                    )
                    cand = scored_cands[0]
                    raw_patch = cand["patch"]
                    harm_patch, ratio_name = snap_to_harmonic_ratio(raw_patch)

                    # 1. Render RAW candidate playback
                    flush_synth(synth)
                    apply_patch_with_pitch(synth, raw_patch)
                    pred_raw = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                    flush_synth(synth)
                    pred_raw_norm = (pred_raw / (np.max(np.abs(pred_raw)) + 1e-7)).astype(np.float32)

                    sf.write(raw_wav, pred_raw_norm.T, SAMPLE_RATE)
                    save_patch(synth, raw_patch, os.path.join(presets_dir, f"{stem_id}_{kind}_{t_tag}_raw"), copy_to_user_dir=False)

                    # 2. Render HARMONIC SNAPPED candidate playback
                    flush_synth(synth)
                    apply_patch_with_pitch(synth, harm_patch)
                    pred_harm = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                    flush_synth(synth)
                    pred_harm_norm = (pred_harm / (np.max(np.abs(pred_harm)) + 1e-7)).astype(np.float32)

                    sf.write(harm_wav, pred_harm_norm.T, SAMPLE_RATE)
                    save_patch(synth, harm_patch, os.path.join(presets_dir, f"{stem_id}_{kind}_{t_tag}_harm"), copy_to_user_dir=False)

                    # MultiScaleSTFT against ground truth
                    raw_mono = np.mean(pred_raw_norm, axis=0)
                    harm_mono = np.mean(pred_harm_norm, axis=0)
                    min_len = min(len(gt_mono_chunk), len(raw_mono))
                    stft_raw = float(loss_fn(raw_mono[:min_len], gt_mono_chunk[:min_len]))
                    stft_harm = float(loss_fn(harm_mono[:min_len], gt_mono_chunk[:min_len]))

                    elapsed_tau = time.time() - t0
                    all_results[kind][stem_id]["temps"][t_tag] = {
                        "tau": tau,
                        "elapsed_s": round(elapsed_tau, 2),
                        "stft_phrase_raw": round(stft_raw, 3),
                        "stft_phrase_harm": round(stft_harm, 3),
                        "snapping_report": ratio_name,
                        "cutoff": round(float(raw_patch.get("cutoff", 0)), 3),
                        "resonance": round(float(raw_patch.get("resonance", 0)), 3),
                        "feg_decay": round(float(raw_patch.get("feg_decay", 0)), 3),
                        "aeg_decay": round(float(raw_patch.get("aeg_decay", 0)), 3),
                        "raw_wav": raw_wav_name,
                        "harm_wav": harm_wav_name,
                    }
                except Exception as e:
                    print(f"   [{t_tag} tau={tau:.2f}] Error rendering: {e}")

    out_json = os.path.join(out_dir, "results.json")
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                existing = json.load(f)
            for k in ("ema", "online"):
                if k in existing:
                    if k not in all_results:
                        all_results[k] = {}
                    for sid, sval in existing[k].items():
                        if sid not in all_results[k]:
                            all_results[k][sid] = sval
        except Exception:
            pass
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - t_start_all
    print("=" * 80)
    print(f"RENDER COMPLETE for {os.path.basename(arm_dir.rstrip('/'))} in {total_time:.1f}s ({total_time/60.0:.1f}m)")
    print(f"WAV Audio Clips: {audio_dir}")
    print(f"VST3 Presets:    {presets_dir}")
    print(f"Results JSON:    {out_json}")
    print("=" * 80)


def resolve_arm_dir(base_dir: str, arm: str) -> str:
    candidates = [
        os.path.join(base_dir, f"real_bass_v3_{arm}_seeded"),
        os.path.join(base_dir, f"real_bass_v3_{arm}_prime"),
        os.path.join(base_dir, f"real_bass_v3_{arm}"),
        os.path.join(base_dir, arm)
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "run_meta.json")):
            return c
    return os.path.join(base_dir, f"real_bass_v3_{arm}")


def render_arm_isolated(arm: str, base_dir: str, midi_dir: str, temps: list, stem_filter=None):
    from invert_stem_collection_v3_cpu import STEM_CATALOG
    stems = STEM_CATALOG
    if stem_filter:
        stems = [s for s in STEM_CATALOG if stem_filter in s["id"]]

    arm_dir = resolve_arm_dir(base_dir, arm)
    print("=" * 80)
    print(f"RUNNING ISOLATED RENDER FOR ARM {arm.upper()} -> {os.path.basename(arm_dir)} ({len(stems)} stems)")
    print("=" * 80)

    audio_dir = os.path.join(arm_dir, "eval_cpu_muscriptor_temps", "audio")

    for idx, s in enumerate(stems):
        stem_id = s["id"]
        # Pre-check: if all clips exist, skip worker invocation entirely
        all_exist = True
        for kind in ("ema", "online"):
            for t in temps:
                t_tag = format_tau_tag(t)
                h_p = os.path.join(audio_dir, f"{stem_id}_{kind}_{t_tag}_harm.wav")
                r_p = os.path.join(audio_dir, f"{stem_id}_{kind}_{t_tag}_raw.wav")
                if not (os.path.exists(h_p) and os.path.exists(r_p)):
                    all_exist = False
                    break
            if not all_exist:
                break
        if all_exist:
            print(f"[{idx+1:02d}/{len(stems)}] {stem_id}: All {len(temps)*2} clips exist, skipping.")
            continue

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--arms", arm,
            "--stem", stem_id,
            "--base_dir", base_dir,
            "--midi_dir", midi_dir,
            "--temps", *[str(t) for t in temps],
            "--worker"
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[WARNING] Stem {stem_id} exited with code {ret.returncode}, continuing to next stem...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["a2", "a3"], help="Arms to render (a2, a3)")
    parser.add_argument("--base_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models")
    parser.add_argument("--midi_dir", type=str,
                        default="/run/media/kim/Mantu/surge_200k_models/real_stems_eval/midi_muscriptor")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.85, 1.0])
    parser.add_argument("--stem", type=str, default=None, help="Filter to specific stem id substring")
    parser.add_argument("--worker", action="store_true", help="Run single worker pass (internal)")
    args = parser.parse_args()

    if args.worker:
        for arm in args.arms:
            arm_dir = resolve_arm_dir(args.base_dir, arm)
            out_dir = os.path.join(arm_dir, "eval_cpu_muscriptor_temps")
            render_arm(arm_dir, out_dir, args.midi_dir, args.temps, stem_filter=args.stem)
    else:
        for arm in args.arms:
            render_arm_isolated(arm, args.base_dir, args.midi_dir, args.temps, stem_filter=args.stem)


if __name__ == "__main__":
    main()
