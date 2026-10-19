# Owner: KUANG / KIM (created 2026-10-18)
"""Multi-note consensus evaluation on 24 real bass stems for V5 synthesis inversion.

Key Innovations:
1. Multi-Note Extraction: Extracts individual clean notes from Muscriptor MIDI instead of a single 16th crop.
2. Pitch-Aware Conditioning: Conditions the flow inverter on each note's actual audio and true MIDI pitch.
3. Consensus Engine:
   - Continuous Parameters: Median across all note predictions (resistant to up to 50% noisy/corrupted notes).
   - Categorical Parameters: Mode (majority vote) across note predictions (filter type, waveshaper, unison, osc type).
4. Full-Phrase Reranking: Validates the consensus patch and top candidates against the entire real stem playback.
5. Dual-Codec Support: Dynamically evaluates 99-d (S0/S1, v3) and 108-d (Stage 3, v4) flow inverters.
"""
import argparse
import json
import math
import os
import sys
import time

# Set conservative thread limits to avoid starving training run
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

import librosa
import mido
import numpy as np
from scipy import stats
import soundfile as sf

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
sys.path.insert(0, "/home/kim/.gemini/antigravity/brain/62fc5213-e94b-4c13-86bf-69147f02f03c/scratch")

import param_codec_v3 as codec_v3
import param_codec_v4 as codec_v4
import param_codec_v5 as codec_v5
import surge_spec_v3 as spec_v3
import surge_spec_v4 as spec_v4
import surge_spec_v5 as spec_v5
from models_v4 import load_inverter_v4
from audio_utils import MultiScaleSTFTLoss, make_mel_spec, extract_pitch_and_envelope
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu
import audio_metrics_v5 as am

DEFAULT_PLUGIN_PATH = spec_v4.DEFAULT_PLUGIN_PATH
SAMPLE_RATE = 44100
AUDIO_LEN = 35280  # 0.8s


def load_clean_pedalboard_events(midi_path, max_dur=None):
    """Parse MIDI file and convert to pedalboard-compatible mido Messages.
    Strips Channel 9 (drums/percussion) to avoid voice hijacking.
    """
    mid = mido.MidiFile(midi_path)
    events = []
    t = 0.0
    active_notes = {}
    for msg in mid:
        t += msg.time
        if max_dur is not None and t >= max_dur:
            break
        if getattr(msg, "channel", 0) == 9:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            events.append(mido.Message("note_on", note=msg.note, velocity=msg.velocity, time=t))
            active_notes[msg.note] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            events.append(mido.Message("note_off", note=msg.note, velocity=0, time=t))
            active_notes.pop(msg.note, None)

    cutoff_time = max_dur - 0.005 if max_dur is not None else t
    for note in list(active_notes.keys()):
        events.append(mido.Message("note_off", note=note, velocity=0, time=cutoff_time))

    events.sort(key=lambda m: m.time)
    return events


def extract_notes_from_midi(midi_path, max_dur=None):
    """Extract distinct note intervals (start, duration, pitch, velocity) from MIDI."""
    mid = mido.MidiFile(midi_path)
    notes = []
    t = 0.0
    active = {}
    for msg in mid:
        t += msg.time
        if getattr(msg, "channel", 0) == 9:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on") and (msg.type == "note_off" or msg.velocity == 0):
            if msg.note in active:
                t0, vel = active.pop(msg.note)
                d = t - t0
                if d >= 0.07:  # filter sub-70ms glitches
                    if max_dur is None or t0 < max_dur:
                        notes.append({"note": msg.note, "start": t0, "dur": d, "vel": vel})
    return notes


def select_representative_notes(notes, max_notes=6):
    """Select diverse, clean notes covering unique pitches and solid durations."""
    if not notes:
        return []
    by_pitch = {}
    for n in notes:
        p = n["note"]
        if p not in by_pitch or n["dur"] > by_pitch[p]["dur"]:
            by_pitch[p] = n
    
    selected = list(by_pitch.values())
    if len(selected) > max_notes:
        selected.sort(key=lambda x: x["dur"], reverse=True)
        selected = selected[:max_notes]
    elif len(selected) < max_notes:
        remaining = [n for n in notes if n not in selected]
        remaining.sort(key=lambda x: x["dur"], reverse=True)
        selected.extend(remaining[:max_notes - len(selected)])
    
    selected.sort(key=lambda x: x["start"])
    return selected


def build_conditioning_targets(chunk, sixteenth_s, sr=SAMPLE_RATE):
    """Build matched 2-note octave phrase and rhythmic rolling pattern to fill 0.8s buffer."""
    s2 = int(0.250 * sr)
    l1 = min(len(chunk), s2)
    t_phrase = np.zeros(AUDIO_LEN, dtype=np.float32)
    t_phrase[:l1] = chunk[:l1]
    
    chunk_oct = librosa.effects.pitch_shift(chunk[:l1], sr=sr, n_steps=12)
    chunk_oct /= (np.max(np.abs(chunk_oct)) + 1e-7)
    l2 = min(len(chunk_oct), AUDIO_LEN - s2)
    t_phrase[s2 : s2 + l2] = chunk_oct[:l2]
    t_phrase /= (np.max(np.abs(t_phrase)) + 1e-7)

    t_rolling = np.zeros(AUDIO_LEN, dtype=np.float32)
    step_samples = max(int(0.05 * sr), int(sixteenth_s * sr))
    for step in range(int(0.8 / max(0.05, sixteenth_s))):
        pos = int(step * step_samples)
        if pos >= AUDIO_LEN:
            break
        to_copy = min(l1, AUDIO_LEN - pos)
        t_rolling[pos : pos + to_copy] = chunk[:to_copy]
    t_rolling /= (np.max(np.abs(t_rolling)) + 1e-7)

    return t_phrase, t_rolling


def get_centroid(y, sr=SAMPLE_RATE):
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    return float((freqs * spec).sum() / (spec.sum() + 1e-7))


@torch.no_grad()
def sample_candidates_adaptive(model, mel, midi_note, k=4, steps=25, seed=42):
    """Adaptive candidate sampler supporting both 99-d (v3) and 108-d (v4) flow models."""
    device = next(model.parameters()).device
    model.eval()
    mel = mel.to(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    
    note_low, note_high = (12.0, 96.0)
    note_norm = np.clip((float(midi_note) - note_low) / (note_high - note_low), 0.0, 1.0)
    note_t = torch.full((k,), float(note_norm), device=device)
    
    mel_k = mel.repeat(k, 1, 1)
    emb = model.encoder(mel_k)
    x = torch.randn(k, model.param_dim, device=device, generator=gen)
    
    for i in range(steps):
        tv = i / steps
        t = torch.full((k,), tv, device=device)
        x[:, 0] = (1.0 - tv) * x[:, 0] + tv * note_t
        x = x + model(x, t, audio_emb=emb) * (1.0 / steps)
    
    x[:, 0] = note_t
    
    if model.param_dim == 126:
        p_ord = codec_v5.decode(x).cpu().numpy()
    elif model.param_dim == 108:
        p_ord = codec_v4.decode(x).cpu().numpy()
    else:
        p_ord = codec_v3.decode(x).cpu().numpy()
    
    return p_ord


def build_consensus_vector(param_vectors, param_dim=99):
    """Build consensus vector from a pool of parameter vectors.
    Uses median for continuous parameters and mode (majority vote) for categorical parameters.
    """
    if param_dim == 126:
        cat_indices = spec_v5.CAT_INDICES
        cont_indices = spec_v5.CONT_INDICES
        num_params = spec_v5.NUM_PARAMS
    elif param_dim == 108:
        cat_indices = spec_v4.CAT_INDICES
        cont_indices = spec_v4.CONT_INDICES
        num_params = spec_v4.NUM_PARAMS
    else:
        cat_indices = spec_v3.CAT_INDICES
        cont_indices = spec_v3.CONT_INDICES
        num_params = spec_v3.NUM_PARAMS
    
    consensus = np.zeros(num_params, dtype=np.float32)
    
    # Continuous: median
    for idx in cont_indices:
        consensus[idx] = float(np.median(param_vectors[:, idx]))
    
    # Categorical: mode (majority vote on discrete classes)
    for idx, k in cat_indices:
        classes = np.round(param_vectors[:, idx] * (k - 1)).astype(int)
        mode_cls = int(stats.mode(classes, keepdims=False)[0])
        consensus[idx] = float(mode_cls) / max(k - 1, 1)
    
    return consensus


def vector_to_patch_adaptive(vec, param_dim=99):
    if param_dim == 126:
        return spec_v5.vector_to_patch(vec)
    elif param_dim == 108:
        return spec_v4.vector_to_patch(vec)
    else:
        return spec_v3.vector_to_patch(vec)


def apply_patch_adaptive(plugin, patch, param_dim=99):
    if param_dim == 126:
        spec_v5.apply_patch(plugin, patch)
    else:
        spec_v4.apply_patch(plugin, patch)


FILTER_CONFIG_NAMES = ["Serial 1", "Serial 2", "Serial 3", "Dual 1", "Dual 2", "Stereo", "Ring", "Wide"]
FM_ROUTING_NAMES = ["Off", "2 > 1", "3 > 2 > 1", "2 > 1 < 3"]


def bpm_to_max_release_raw(bpm: float) -> float:
    """Calculate the maximum Surge XT raw release value corresponding to a 32nd note at the given BPM.
    Guarantees that notes decay before the next 16th note onset, eliminating hanging drones.
    """
    t_32_ms = (7500.0 / max(float(bpm), 60.0))
    raw = (np.log(max(t_32_ms, 5.0)) - np.log(3.9)) / 8.96
    return float(np.clip(raw, 0.08, 0.35))


def format_patch_details(patch, param_dim=99):
    """Format key synth parameters into human-readable summary string."""
    if param_dim in (126, 108):
        spec = spec_v5 if param_dim == 126 else spec_v4
        f1_type = spec.LP_FILTERS[patch["filter_idx"]][0]
        f2_type = spec.LP_FILTERS[patch.get("filter_2_type", 0)][0]
        ws_type = spec.WAVESHAPER_TYPES[patch["ws_idx"]][0]
        unison = patch.get("unison_voices_idx", 0) + 1
        oct1 = patch.get("osc_1_octave_offset", int(round((patch.get("osc_1_octave", 0.5) - 0.5) * 6.0)))
        cut1 = patch.get("cutoff", 0.5)
        res1 = patch.get("resonance", 0.0)
        cut2 = patch.get("filter_2_cutoff", 0.5)
        res2 = patch.get("filter_2_resonance", 0.0)
        drv = patch.get("drive_raw", 0.5)
        fb = patch.get("filter_feedback", 0.5)
        cfg_idx = patch.get("filter_configuration", 0)
        cfg_name = FILTER_CONFIG_NAMES[cfg_idx] if 0 <= cfg_idx < len(FILTER_CONFIG_NAMES) else f"Cfg{cfg_idx}"
        fm_idx = patch.get("fm_routing", 0)
        fm_name = FM_ROUTING_NAMES[fm_idx] if 0 <= fm_idx < len(FM_ROUTING_NAMES) else f"FM{fm_idx}"
        rel_ms = 3.9 * np.exp(8.96 * patch.get("aeg_release", 0.23))
        return (f"F1: {f1_type} (cut:{cut1:.2f}, res:{res1:.2f}) | "
                f"F2: {f2_type} (cut:{cut2:.2f}, res:{res2:.2f}) | "
                f"Cfg: {cfg_name} | FM: {fm_name} | "
                f"WS: {ws_type} (drv:{drv:.2f}) | Uni: {unison}v | "
                f"Oct: {oct1:+d} | Rel: {rel_ms:.0f}ms | FB: {fb:.2f}")
    else:
        f1_type = spec_v3.LP_FILTERS[patch["filter_idx"]][0]
        ws_type = spec_v3.WAVESHAPER_TYPES[patch["ws_idx"]][0]
        unison = "On" if patch.get("unison", False) else "Off"
        oct1 = int(round((patch.get("osc_1_octave", 0.5) - 0.5) * 6.0))
        cut = patch.get("cutoff", 0.5)
        res = patch.get("resonance", 0.0)
        drv = patch.get("drive_raw", 0.5)
        rel_ms = 3.9 * np.exp(8.96 * patch.get("aeg_release", 0.23))
        return f"F1: {f1_type} (cut:{cut:.2f}, res:{res:.2f}) | WS: {ws_type} (drv:{drv:.2f}) | Uni: {unison} | Oct: {oct1:+d} | Rel: {rel_ms:.0f}ms"


def compute_stereo_metrics(y, is_channels_first=False):
    if y.ndim == 1:
        return 0.0, 1.0
    if is_channels_first:
        L, R = y[0], y[1]
    else:
        L, R = y[:, 0], y[:, 1]
    mid = (L + R) / np.sqrt(2.0)
    side = (L - R) / np.sqrt(2.0)
    e_mid = float(np.sum(mid**2))
    e_side = float(np.sum(side**2))
    width = float(np.sqrt(e_side / (e_mid + 1e-9)))
    denom = np.sqrt(np.sum(L**2) * np.sum(R**2)) + 1e-9
    corr = float(np.sum(L * R) / denom)
    return width, corr


def evaluate_models_consensus(models_dict, out_dir, device="cpu", max_notes=6, k_per_note=2, steps=25, seed=42):
    E = resolve_mantu("/run/media/kim/Mantu/surge_200k_models/real_stems_eval")
    audio_ref_dir = os.path.join(E, "audio")
    midi_muscriptor_dir = os.path.join(E, "midi_muscriptor")

    audio_out_dir = os.path.join(out_dir, "audio")
    listen_dir = os.path.join(out_dir, "listen")
    presets_dir = os.path.join(out_dir, "vstpresets")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(listen_dir, exist_ok=True)
    os.makedirs(presets_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print("V5 REAL BASS STEM INVERSION: MULTI-NOTE CONSENSUS BENCHMARK (STEREO & RETRIG AWARE)", flush=True)
    print(f"Models: {list(models_dict.keys())}", flush=True)
    print(f"Max notes per stem: {max_notes} | Candidates per note: {k_per_note * 2}", flush=True)
    print(f"Output Directory: {out_dir}", flush=True)
    print("=" * 80, flush=True)

    loaded_models = {}
    for name, path in models_dict.items():
        print(f"Loading {name} ({path})...", flush=True)
        loaded_models[name] = load_inverter_v4(path, device=device)

    print("Initializing Surge XT plugin...", flush=True)
    synth = spec_v4.init_synth(DEFAULT_PLUGIN_PATH)
    loss_fn = MultiScaleSTFTLoss()

    results = []

    for idx, item in enumerate(STEM_CATALOG):
        stem_id = item["id"]
        name = item["name"]
        bpm = item["bpm"]
        style = item["style"]

        real_wav_path = os.path.join(audio_ref_dir, f"{stem_id}_real.wav")
        midi_path = os.path.join(midi_muscriptor_dir, f"{stem_id}_muscriptor.mid")

        if not os.path.exists(real_wav_path) or not os.path.exists(midi_path):
            print(f"[{idx+1:02d}/24] SKIPPING {name}: Missing files", flush=True)
            continue

        y_real, sr = sf.read(real_wav_path)
        if y_real.ndim > 1:
            real_L, real_R = y_real[:, 0], y_real[:, 1]
            y_real_mono = y_real.mean(axis=1)
        else:
            real_L, real_R = y_real, y_real
            y_real_mono = y_real
            y_real = np.stack([y_real, y_real], axis=-1)

        real_width, real_corr = compute_stereo_metrics(y_real, is_channels_first=False)
        real_stereo_class = "Pure Mono" if real_width < 0.02 else ("Wide Stereo" if real_width > 0.3 else "Narrow Stereo")
        real_mid = (real_L + real_R) / np.sqrt(2.0)
        real_side = (real_L - real_R) / np.sqrt(2.0)
        real_mid_norm = real_mid / (np.max(np.abs(real_mid)) + 1e-7)
        real_side_norm = real_side / (np.max(np.abs(real_mid)) + 1e-7)

        # Real binaural channel balance and spectral asymmetry
        real_rms_L = np.sqrt(np.mean(real_L**2))
        real_rms_R = np.sqrt(np.mean(real_R**2))
        real_bal = abs(real_rms_L - real_rms_R) / (real_rms_L + real_rms_R + 1e-6)

        phrase_dur = len(y_real_mono) / sr
        events = load_clean_pedalboard_events(midi_path, max_dur=phrase_dur)
        real_centroid = get_centroid(y_real_mono, sr)

        real_cent_L = get_centroid(real_L, sr)
        real_cent_R = get_centroid(real_R, sr)
        real_cent_asym = abs(real_cent_L - real_cent_R) / max(real_cent_L, real_cent_R, 1.0)

        # Extract notes and select clean representative subset
        all_notes = extract_notes_from_midi(midi_path, max_dur=phrase_dur)
        selected_notes = select_representative_notes(all_notes, max_notes=max_notes)

        pitches = [n["note"] for n in selected_notes]
        print(f"\n[{idx+1:02d}/24] Inverting '{name}' ({style}, {bpm} BPM, {real_stereo_class} W={real_width:.2f}, {len(all_notes)} notes -> selected {len(selected_notes)} notes: {pitches})...", flush=True)

        ref_phrase_dest = os.path.join(listen_dir, f"{stem_id}__real_phrase.wav")
        sf.write(ref_phrase_dest, y_real, sr, subtype="PCM_16")

        # Nominal root and sub-octave threshold
        root_midi = max(set(pitches), key=pitches.count) if pitches else 36
        nominal_f0 = 440.0 * (2.0 ** ((root_midi - 69.0) / 12.0))
        sub_thresh = 0.75 * nominal_f0
        fft_real = np.abs(np.fft.rfft(y_real_mono))
        freqs_real = np.fft.rfftfreq(len(y_real_mono), 1.0 / sr)
        t_sub = float(np.sum(fft_real[freqs_real < sub_thresh]**2) / (np.sum(fft_real**2) + 1e-9))

        stem_res = {
            "stem_id": stem_id,
            "name": name,
            "style": style,
            "bpm": bpm,
            "root_midi": root_midi,
            "real_centroid_hz": round(real_centroid, 1),
            "real_width": round(real_width, 3),
            "real_corr": round(real_corr, 3),
            "real_stereo_class": real_stereo_class,
            "models": {}
        }

        y_real_norm = y_real_mono / (np.max(np.abs(y_real_mono)) + 1e-7)

        for m_name, model in loaded_models.items():
            t0 = time.time()
            param_dim = model.param_dim

            # Collect candidates across all selected notes
            cand_vecs = []
            for n_idx, n in enumerate(selected_notes):
                s_idx = max(0, int(round(n["start"] * sr)))
                chunk_dur = min(0.24, max(0.08, n["dur"]))
                n_len = int(round(chunk_dur * sr))
                chunk = y_real_mono[s_idx : s_idx + n_len].copy()
                fade_len = int(sr * 0.005)
                if len(chunk) > fade_len:
                    chunk[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
                chunk /= (np.max(np.abs(chunk)) + 1e-7)
                sixteenth_s = chunk_dur

                t_phrase, t_rolling = build_conditioning_targets(chunk, sixteenth_s, sr=sr)
                mel_p = torch.from_numpy(make_mel_spec(t_phrase)).unsqueeze(0).float()
                mel_r = torch.from_numpy(make_mel_spec(t_rolling)).unsqueeze(0).float()

                vecs_p = sample_candidates_adaptive(model, mel_p, n["note"], k=k_per_note, steps=steps, seed=seed + idx * 100 + n_idx)
                vecs_r = sample_candidates_adaptive(model, mel_r, n["note"], k=k_per_note, steps=steps, seed=seed + idx * 100 + n_idx + 50)

                for c in range(k_per_note):
                    cand_vecs.append(vecs_p[c])
                    cand_vecs.append(vecs_r[c])

            cand_vecs = np.array(cand_vecs)

            # Build consensus vector
            consensus_vec = build_consensus_vector(cand_vecs, param_dim=param_dim)
            
            # Pool: consensus patch + all individual note candidate patches
            pool_vecs = [consensus_vec] + list(cand_vecs)
            pool_patches = [vector_to_patch_adaptive(v, param_dim=param_dim) for v in pool_vecs]

            # 32nd-note BPM-adaptive release ceiling: guarantees notes decay cleanly before next 16th note
            max_rel_raw = bpm_to_max_release_raw(bpm)
            for p in pool_patches:
                p["aeg_release"] = min(float(p.get("aeg_release", 0.23)), max_rel_raw)
                p["feg_release"] = min(float(p.get("feg_release", p["aeg_release"])), max_rel_raw)

            # Re-rank candidates on full phrase in Surge XT (with stereo & retrigger search)
            best_score = float("inf")
            best_patch = None
            best_audio = None
            best_type = "Consensus"
            best_retrig = True
            best_width = 0.0
            best_corr = 1.0

            for p_idx, p in enumerate(pool_patches):
                for retrig in [True, False]:
                    spec_v4.flush_synth(synth)
                    apply_patch_adaptive(synth, p, param_dim=param_dim)
                    synth.parameters["a_osc_1_retrigger"].raw_value = 1.0 if retrig else 0.0
                    synth.parameters["a_osc_2_retrigger"].raw_value = 1.0 if retrig else 0.0
                    synth.parameters["a_osc_3_retrigger"].raw_value = 1.0 if retrig else 0.0

                    pred_audio = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                    pred_L, pred_R = pred_audio[0], pred_audio[1]
                    pred_mid = (pred_L + pred_R) / np.sqrt(2.0)
                    pred_side = (pred_L - pred_R) / np.sqrt(2.0)
                    pred_width, pred_corr = compute_stereo_metrics(pred_audio, is_channels_first=True)

                    pred_mid_norm = pred_mid / (np.max(np.abs(pred_mid)) + 1e-7)
                    l = min(len(pred_mid_norm), len(real_mid_norm))
                    stft_mid = float(loss_fn(pred_mid_norm[:l], real_mid_norm[:l]))

                    # Sub-octave penalty
                    fft_c = np.abs(np.fft.rfft(pred_mid_norm))
                    freqs_c = np.fft.rfftfreq(len(pred_mid_norm), 1.0 / sr)
                    c_sub = float(np.sum(fft_c[freqs_c < sub_thresh]**2) / (np.sum(fft_c**2) + 1e-9))
                    sub_pen = max(0.0, c_sub - 0.10) * 2.0 if t_sub < 0.06 else 0.0

                    # Binaural balance and spectral asymmetry penalty (protects against single-sided dual filter muting)
                    pred_rms_L = np.sqrt(np.mean(pred_L**2))
                    pred_rms_R = np.sqrt(np.mean(pred_R**2))
                    pred_bal = abs(pred_rms_L - pred_rms_R) / (pred_rms_L + pred_rms_R + 1e-6)
                    bal_pen = max(0.0, pred_bal - real_bal - 0.08) * 3.0

                    pred_cent_L = get_centroid(pred_L, sr)
                    pred_cent_R = get_centroid(pred_R, sr)
                    cent_asym = abs(pred_cent_L - pred_cent_R) / max(pred_cent_L, pred_cent_R, 1.0)
                    cent_bal_pen = max(0.0, cent_asym - real_cent_asym - 0.12) * 2.0

                    # Stereo-aware score
                    if real_width > 0.03:
                        pred_side_norm = pred_side / (np.max(np.abs(pred_mid)) + 1e-7)
                        stft_side = float(loss_fn(pred_side_norm[:l], real_side_norm[:l]))
                        width_err = abs(pred_width - real_width)
                        score = stft_mid + 0.4 * stft_side + 0.6 * width_err + sub_pen + bal_pen + cent_bal_pen
                    else:
                        score = stft_mid + 1.2 * pred_width + sub_pen + bal_pen + cent_bal_pen

                    if score < best_score:
                        best_score = score
                        best_patch = p
                        best_audio = pred_audio
                        best_type = "Consensus" if p_idx == 0 else f"NoteCandidate_{p_idx}"
                        best_retrig = retrig
                        best_width = pred_width
                        best_corr = pred_corr

            el = time.time() - t0

            # Compute detailed audio metrics for winner
            pred_winner_mono = best_audio.mean(axis=0)
            pred_winner_norm = pred_winner_mono / (np.max(np.abs(pred_winner_mono)) + 1e-7)
            win_centroid = get_centroid(pred_winner_norm, sr)
            centroid_err = win_centroid - real_centroid
            l_eval = min(len(pred_winner_norm), len(y_real_norm))
            mel_ssim = float(am.mel_ssim_dist(y_real_norm[:l_eval], pred_winner_norm[:l_eval]))

            # Save winner audio
            cand_filename = f"{stem_id}__{m_name}.wav"
            cand_dest = os.path.join(listen_dir, cand_filename)
            pred_stereo_norm = (best_audio / (np.max(np.abs(best_audio)) + 1e-7)).astype(np.float32)
            sf.write(cand_dest, pred_stereo_norm.T, sr, subtype="PCM_16")

            patch_summary = format_patch_details(best_patch, param_dim=param_dim)
            patch_summary += f" | Retrig: {'On' if best_retrig else 'Off'}"

            stem_res["models"][m_name] = {
                "phrase_mss": round(best_score, 3),
                "mel_ssim": round(mel_ssim, 4),
                "centroid_hz": round(win_centroid, 1),
                "centroid_delta_hz": round(centroid_err, 1),
                "stereo_width": round(best_width, 3),
                "stereo_corr": round(best_corr, 3),
                "width_delta": round(best_width - real_width, 3),
                "retrigger": "On" if best_retrig else "Off",
                "winner_type": best_type,
                "elapsed_s": round(el, 2),
                "audio_file": cand_filename,
                "patch_summary": patch_summary,
                "best_patch": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_patch.items()}
            }

            print(f"  [{m_name}] MSS: {best_score:.3f} | SSIM: {mel_ssim:.4f} | Cent: {win_centroid:.0f}Hz ({centroid_err:+.0f}Hz) | W: {best_width:.2f} (dW: {best_width - real_width:+.2f}) | Retrig: {'On' if best_retrig else 'Off'} | {best_type} in {el:.1f}s", flush=True)

        results.append(stem_res)

    # Save summary JSON
    summary_path = os.path.join(out_dir, "consensus_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate HTML Listening Pack
    generate_listening_pack_html(results, listen_dir, list(models_dict.keys()))
    print(f"\nConsensus evaluation complete! Listening pack: file://{listen_dir}/index.html", flush=True)


def generate_listening_pack_html(results, listen_dir, model_names):
    """Generate modern, dark-themed responsive comparison page with audio players and metric meters."""
    rows = []
    for r in results:
        stem_id = r["stem_id"]
        ref_file = f"{stem_id}__real_phrase.wav"
        
        # Find best model for this stem
        model_scores = {m: r["models"][m]["phrase_mss"] for m in model_names if m in r["models"]}
        best_m = min(model_scores, key=model_scores.get) if model_scores else None

        mod_cells = []
        for m in model_names:
            if m not in r["models"]:
                mod_cells.append("<td>N/A</td>")
                continue
            d = r["models"][m]
            mss = d["phrase_mss"]
            ssim = d["mel_ssim"]
            cent = d["centroid_hz"]
            d_cent = d["centroid_delta_hz"]
            is_best = (m == best_m)
            badge_class = "badge-winner" if is_best else "badge"

            cell = f"""
            <td class="model-cell {'winner-cell' if is_best else ''}">
                <audio controls src="{d['audio_file']}"></audio>
                <div class="metrics">
                    <span class="{badge_class}">MSS: {mss:.3f}</span>
                    <span class="badge">SSIM: {ssim:.3f}</span>
                    <span class="badge">W: {d.get('stereo_width', 0.0):.2f} (dW: {d.get('width_delta', 0.0):+.2f})</span>
                </div>
                <div class="centroid-info">Centroid: {cent:.0f} Hz ({d_cent:+.0f} Hz) | Retrig: {d.get('retrigger', 'On')} | {d['winner_type']}</div>
                <div class="patch-info">{d['patch_summary']}</div>
            </td>
            """
            mod_cells.append(cell)

        row = f"""
        <tr>
            <td class="meta-cell">
                <strong>{r['name']}</strong><br>
                <span class="style-tag">{r['style']}</span><br>
                <small style="color: #8b949e;">{r['bpm']} BPM | Root: {r['root_midi']}</small>
            </td>
            <td class="ref-cell">
                <audio controls src="{ref_file}"></audio>
                <div class="centroid-info">Ground Truth Reference<br>Centroid: {r['real_centroid_hz']:.0f} Hz<br><span class="style-tag">Width: {r.get('real_width', 0.0):.2f} ({r.get('real_stereo_class', 'Mono')})</span></div>
            </td>
            {"".join(mod_cells)}
        </tr>
        """
        rows.append(row)

    html = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>V5 Multi-Note Consensus Benchmark on Real Stems (Stereo & Retrigger Fixes)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            margin: 24px;
        }}
        h1 {{
            color: #58a6ff;
            margin-bottom: 4px;
        }}
        p.subtitle {{
            color: #8b949e;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #21262d;
            color: #f0f6fc;
            padding: 12px 14px;
            text-align: left;
            border-bottom: 2px solid #30363d;
            font-size: 13px;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid #21262d;
            vertical-align: top;
        }}
        tr:hover td {{
            background: #1c2128;
        }}
        .meta-cell {{
            width: 170px;
        }}
        .style-tag {{
            display: inline-block;
            background: #388bfd26;
            color: #58a6ff;
            border: 1px solid #388bfd4d;
            border-radius: 12px;
            padding: 1px 8px;
            font-size: 11px;
            margin-top: 3px;
            margin-bottom: 3px;
        }}
        .ref-cell {{
            width: 240px;
            background: #13171f;
        }}
        .model-cell {{
            width: 270px;
        }}
        .winner-cell {{
            background: #23863610;
            border-left: 2px solid #3fb95044;
        }}
        audio {{
            width: 220px;
            height: 30px;
            margin-top: 2px;
            margin-bottom: 4px;
        }}
        .metrics {{
            margin-top: 3px;
            margin-bottom: 3px;
        }}
        .badge {{
            display: inline-block;
            background: #30363d;
            color: #c9d1d9;
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 11px;
            font-family: monospace;
            margin-right: 4px;
        }}
        .badge-winner {{
            display: inline-block;
            background: #23863633;
            color: #3fb950;
            border: 1px solid #23863666;
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 11px;
            font-family: monospace;
            font-weight: bold;
            margin-right: 4px;
        }}
        .centroid-info {{
            font-size: 11px;
            color: #8b949e;
            margin-top: 2px;
        }}
        .patch-info {{
            font-size: 10px;
            color: #7ee787;
            font-family: monospace;
            margin-top: 3px;
            line-height: 1.25;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>V5 Multi-Note Consensus Evaluation on Real Stems</h1>
    <p class="subtitle">Individual Note Segmentation via Muscriptor MIDI | Continuous Median &amp; Categorical Mode Consensus | Full Phrase Playback Re-ranking</p>
    <table>
        <thead>
            <tr>
                <th>Stem Information</th>
                <th>Ground Truth Reference</th>
                {"".join(f"<th>{m}</th>" for m in model_names)}
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</body>
</html>
"""
    with open(os.path.join(listen_dir, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Note Consensus Benchmark on Real Stems")
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "PATH"), required=True,
                        help="Add named model: --model NAME PATH")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_notes", type=int, default=6, help="Max clean notes to extract per stem")
    parser.add_argument("--k", type=int, default=2, help="Candidates per conditioning method per note")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = {name: path for name, path in args.model}
    evaluate_models_consensus(models, args.out_dir, device=args.device, max_notes=args.max_notes,
                              k_per_note=args.k, steps=args.steps, seed=args.seed)
