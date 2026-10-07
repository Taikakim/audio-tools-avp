#!/usr/bin/env python3
"""Sample and invert two dozen diverse real bass stems (multitrack + Demucs-separated).

Cleans HF residue / bleed with a 3.5 kHz filter, slices clean note onsets,
inverts via pitch-locked Flow matching with multi-domain reranking (STFT, F0, sub-octave,
bin energy resonance contour, stereo width), and renders A/B comparison phrases.
"""
import os
import sys
import time
import json
import subprocess
from collections import Counter
import numpy as np
import soundfile as sf
import torch
import librosa
import mido
from scipy.signal import butter, sosfilt

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from models import load_inverter
from surge_spec import init_synth, apply_patch, DEFAULT_PLUGIN_PATH, SAMPLE_RATE, AUDIO_LEN
from audio_utils import make_mel_spec, save_patch, extract_pitch_and_envelope
from inference import predict_and_rerank_candidates
from refine import refine_patch_phrase

OUT_DIR = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
CLIPS_DIR = os.path.join(OUT_DIR, "audio")
MIDI_DIR = os.path.join(OUT_DIR, "midi")
PRESETS_DIR = os.path.join(OUT_DIR, "vstpresets")
DAW_DIR = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")
MIR_PYTHON = "/home/kim/Projects/mir/mir/bin/python"
MIR_PIPELINE = "/home/kim/Projects/mir/src/bass_midi_pipeline.py"

os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)
os.makedirs(DAW_DIR, exist_ok=True)

STEM_CATALOG = [
    # 1. Classic Studio Multitrack Stems (Summamutikka & Aavepyörä)
    {
        "id": "stem_01_astral_prot_saw",
        "name": "Astral Protection Saw",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/summamutikka - astral protection 138  BPM 1996 style goa trance/bass/bassline rapid 16ths filtered saw with fast decay goa dry.flac",
        "bpm": 138.0,
        "style": "Goa 16th Filtered Saw"
    },
    {
        "id": "stem_02_acid_alien_ebm",
        "name": "Acid Alien Industrial Pluck",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance/bass/goa trance ebm industrial saw pluck bass sharp short metallic room reverb a phrygian 136 BPM.flac",
        "bpm": 136.0,
        "style": "Industrial Saw Pluck"
    },
    {
        "id": "stem_03_acid_alien_stab",
        "name": "Acid Alien Stabs",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance/acid/saw acid bass stabs clean delay with flanger a phrygian 136 BPM.flac",
        "bpm": 136.0,
        "style": "303 Flanged Acid Stabs"
    },
    {
        "id": "stem_04_goddess_square",
        "name": "Goddess Guerrilla Square",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance/bass/saturated bright square punchy midrange bassline doof style driving energetic dry F phrygian F minor 141 BPM.flac",
        "bpm": 141.0,
        "style": "Punchy Midrange Square"
    },
    {
        "id": "stem_05_goddess_unison",
        "name": "Goddess Guerrilla Unison",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance/bass/unison saw subosc resonant bass melody delay with high feedback short repetitions F phrygian F minor 141 BPM.flac",
        "bpm": 141.0,
        "style": "Resonant Unison Saw Sub"
    },
    {
        "id": "stem_06_increase_dose_disco",
        "name": "Increase Dose Disco Saw",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Increase The Dose or Drop The Poison driving groovy upbeat goa trance/bass/clean smooth saw octave offbeat disco bass f phrygian f minor 142 BPM.flac",
        "bpm": 142.0,
        "style": "Octave Disco Bass"
    },
    {
        "id": "stem_07_increase_dose_sub",
        "name": "Increase Dose Sub Phaser",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Increase The Dose or Drop The Poison driving groovy upbeat goa trance/bass/driving subosc bass melody saturated phaser f phrygian f minor 142 BPM.flac",
        "bpm": 142.0,
        "style": "Saturated Sub Phaser"
    },
    {
        "id": "stem_08_xenharmonic_rubber",
        "name": "Xenharmonic Rubbery",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/summamutikka - xenharmonic 145 BPM microtonal forest trance/bass/bassline rubbery.flac",
        "bpm": 145.0,
        "style": "Rubbery Forest Bass"
    },
    {
        "id": "stem_09_stay_pure_sync",
        "name": "Stay Pure Sync Disco",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/aavepyörä - stay pure 142 blissful suomisoundi/bass sync clipped rapid disco hinrg.flac",
        "bpm": 142.0,
        "style": "Clipped Sync Hi-NRG"
    },
    {
        "id": "stem_10_stay_pure_octave",
        "name": "Stay Pure Moroder Octave",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/aavepyörä - stay pure 142 blissful suomisoundi/bass rapid hinrg octave jumps avp moroder style clipped dry.flac",
        "bpm": 142.0,
        "style": "Rapid Octave Jumps"
    },
    {
        "id": "stem_11_stay_pure_damped",
        "name": "Stay Pure Damped Disco",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/aavepyörä - stay pure 142 blissful suomisoundi/bass smooth damped rapid hirg moroder disco dry.flac",
        "bpm": 142.0,
        "style": "Smooth Damped Disco"
    },
    {
        "id": "stem_12_unessa_lush_saw",
        "name": "Unessaelama Lush Stereo Saw",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Unessaelämä euphoric driving trance/fat lush stereo resonant saw bass filte decay medium dry C phrygian, G# major, F minor, D# mixolydian, A# dorian 135 BPM.flac",
        "bpm": 135.0,
        "style": "Lush Stereo Resonant Saw"
    },
    {
        "id": "stem_13_zarjaz_octave",
        "name": "Zarjaz Smooth Octave",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - zarjaz beats! upbeat energetic videogame scifi disco electro trance/saw smooth octave bass hinrg disco dry A Minor, D Dorian 136 BPM.flac",
        "bpm": 136.0,
        "style": "Electro Sci-Fi Octave"
    },
    {
        "id": "stem_14_vapausvoima_tb303",
        "name": "Vapausvoima TB-303 Acid",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Vapausvoima energetic epic  furious  freedom power trance/tb303 acid bass riff distorted classic delay chorus d sharp dorian 140 BPM.flac",
        "bpm": 140.0,
        "style": "TB-303 Distorted Acid"
    },
    {
        "id": "stem_15_vapausvoima_crisp",
        "name": "Vapausvoima Crisp Sub",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Vapausvoima energetic epic  furious  freedom power trance/crisp subosc saturated bass driving phaser delay d sharp dorian 140 BPM.flac",
        "bpm": 140.0,
        "style": "Crisp Subosc Phaser"
    },
    {
        "id": "stem_16_kaleidoscillator_sub",
        "name": "Kaleidoscillator Rounded Sub",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Kaleidoscillator colourful electronic eclectic dance music/smooth lowpass filter closed rounded subbass rhythmic steady dry sustained medium release c dorian 148 BPM.flac",
        "bpm": 148.0,
        "style": "Closed Rounded Subbass"
    },
    {
        "id": "stem_17_black_moon_acid",
        "name": "Black Moon Acid Riff",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Black Moon psychedelic goa trance/melody acid bass riff distorted resonant short release furious jumpy nervous d phrygian 144 BPM.flac",
        "bpm": 144.0,
        "style": "Distorted Resonant Acid"
    },
    {
        "id": "stem_18_astral_prot_rubber",
        "name": "Astral Protection Rubber",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/summamutikka - astral protection 138  BPM 1996 style goa trance/bass/bass rubberand.flac",
        "bpm": 138.0,
        "style": "Rubberband Pluck"
    },
    {
        "id": "stem_19_light_in_darkness_moog",
        "name": "Light in Darkness Moog Saw",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/aavepyörä - light in darkness a minor 116 BPM adventure trance/bass moog saw phat groovy upbeat dry.flac",
        "bpm": 116.0,
        "style": "Phat Moog Saw"
    },
    {
        "id": "stem_20_goddess_mid_saw",
        "name": "Goddess Midrange Melodic Saw",
        "path": "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance/bass/saturated saw midrange melodic avp bass phaser heavy feedback dry F phrygian F minor 141 BPM.flac",
        "bpm": 141.0,
        "style": "Midrange Melodic Saw"
    },

    # 2. Classic Separated Stems (Demucs separation with HF residue)
    {
        "id": "stem_21_astral_mahadeva",
        "name": "Astral Projection - Mahadeva",
        "path": "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/multi_artist_benchmarks/demucs_out/htdemucs/astral_mahadeva/bass.wav",
        "bpm": 140.0,
        "style": "Classic Goa Rolling Saw (Demucs)"
    },
    {
        "id": "stem_22_mfg_shape_future",
        "name": "MFG - Shape the Future",
        "path": "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/multi_artist_benchmarks/demucs_out/htdemucs/mfg_shape_future/bass.wav",
        "bpm": 143.55,
        "style": "Psy-Trance Punch (Demucs)"
    },
    {
        "id": "stem_23_cosmosis_alien_disco",
        "name": "Cosmosis - Alien Disco",
        "path": "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/multi_artist_benchmarks/demucs_out/htdemucs/cosmosis_alien_disco/bass.wav",
        "bpm": 140.0,
        "style": "Alien Disco Squelch (Demucs)"
    },
    {
        "id": "stem_24_final_mission",
        "name": "Final Mission",
        "path": "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/final_mission_test/real_stem_phrase.wav",
        "bpm": 140.0,
        "style": "Hypnotic Rolling Bass (Demucs)"
    }
]

def detect_stem_pitch(y, sr=SAMPLE_RATE):
    """Envelope-normalized autocorrelation with fundamental lag preference."""
    from scipy.signal import hilbert
    y_sub = y[:min(len(y), sr * 2)]
    env = np.abs(hilbert(y_sub)) + 1e-4
    y_norm = y_sub / env
    corr = np.correlate(y_norm, y_norm, mode="full")[len(y_norm)//2:]
    min_lag = int(sr / 200.0) # 220
    max_lag = int(sr / 25.0)  # 1764
    peaks = []
    for lag in range(min_lag, max_lag - 1):
        if corr[lag] > corr[lag-1] and corr[lag] > corr[lag+1]:
            f = sr / lag
            midi = int(round(69.0 + 12.0 * np.log2(f / 440.0)))
            peaks.append((lag, f, midi, corr[lag]))
    if not peaks:
        return 36, 65.4
    peaks.sort(key=lambda x: x[3], reverse=True)
    top_candidates = [p for p in peaks[:4] if p[3] > 0.4 * peaks[0][3]]
    top_candidates.sort(key=lambda x: x[0])
    best = top_candidates[0]
    midi_clamped = int(np.clip(best[2], 28, 50))
    return midi_clamped, best[1]


def find_active_bass_phrase(y, sr, bpm, bars=4):
    """Find a sustained, high-energy phrase in the stem and snap start to the nearest note onset."""
    four_bars_dur = bars * 4 * (60.0 / bpm)
    phrase_samples = int(four_bars_dur * sr)
    hop = int(sr * 0.1)
    frame_len = hop * 2
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    times = np.arange(len(rms)) * 0.1
    max_rms = np.max(rms) if len(rms) > 0 else 0
    if max_rms < 1e-4:
        return 0, min(len(y), phrase_samples)
    four_bars_hops = max(1, int(four_bars_dur / 0.1))
    max_search_hop = len(rms) - four_bars_hops
    best_score, best_hop = -1.0, 0
    if max_search_hop > 0:
        for i in range(max_search_hop):
            seg_rms = rms[i : i + four_bars_hops]
            score = np.mean(seg_rms) * (np.min(seg_rms) + 1e-4)
            if score > best_score:
                best_score = score
                best_hop = i
    rough_start_s = times[best_hop]
    window_start = max(0, int((rough_start_s - 0.1) * sr))
    window_end = min(len(y), int((rough_start_s + 0.4) * sr))
    local_y = y[window_start:window_end]
    snapped_start_sample = int(rough_start_s * sr)
    if len(local_y) > 512:
        onsets = librosa.onset.onset_detect(y=local_y, sr=sr, units='samples')
        if len(onsets) > 0:
            snapped_start_sample = window_start + onsets[0]
    snapped_start_sample = max(0, min(snapped_start_sample, max(0, len(y) - phrase_samples)))
    return snapped_start_sample, phrase_samples


def load_pedalboard_events_from_midi(midi_path, max_dur=None):
    """Parse MIDI file and convert to list of pedalboard-compatible mido Messages with absolute seconds."""
    mid = mido.MidiFile(midi_path)
    events = []
    t = 0.0
    for msg in mid:
        t += msg.time
        if max_dur is not None and t > max_dur:
            break
        if msg.type in ('note_on', 'note_off'):
            events.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=t))
    events.sort(key=lambda m: m.time)
    return events


def main():
    global OUT_DIR, CLIPS_DIR, MIDI_DIR, PRESETS_DIR
    import argparse
    ap = argparse.ArgumentParser(description="Invert the real-stem collection into Surge XT patches")
    ap.add_argument("--refine", choices=["phrase", "none"], default="phrase",
                    help="renderer-in-the-loop refinement of cutoff/envelopes against the whole phrase (refine.py)")
    ap.add_argument("--refine_midi", choices=["muscriptor", "mir"], default="muscriptor",
                    help="MIDI that plays the phrase during refinement (MuScriptor if its file exists)")
    ap.add_argument("--flow_ckpt", default="/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64/flow_latest.pt")
    ap.add_argument("--cond_tau", type=float, default=1.0,
                    help="condition on a partly noised reference (cond_noise flow models only; H5/H6). 1 = clean")
    ap.add_argument("--out_dir", default=OUT_DIR, help="eval output root (audio/, midi/, vstpresets/, summary)")
    ap.add_argument("--muscriptor_midi_dir", default=os.path.join(OUT_DIR, "midi_muscriptor"),
                    help="<id>_muscriptor.mid files (phrase slicing is deterministic, so they match any out_dir)")
    ap.add_argument("--playback_midi", choices=["mir", "muscriptor"], default="mir",
                    help="MIDI that plays the saved phrase clip (muscriptor if its file exists)")
    ap.add_argument("--no_user_copy", action="store_true", help="do not copy presets into the Surge user folder")
    ap.add_argument("--stems", default="", help="comma-separated stem ids to run (default: all)")
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    CLIPS_DIR, MIDI_DIR, PRESETS_DIR = (os.path.join(OUT_DIR, d) for d in ("audio", "midi", "vstpresets"))
    for d in (CLIPS_DIR, MIDI_DIR, PRESETS_DIR):
        os.makedirs(d, exist_ok=True)
    print("=" * 80)
    print("SURGE XT REAL-STEM INVERSION BENCHMARK (DYNAMIC SLICING & MIDI PLAYBACK)")
    print("=" * 80)
    print("Loading Flow model onto CPU for multi-stem inversion benchmark...")
    model = load_inverter(args.flow_ckpt, device="cpu")
    if args.cond_tau != 1.0:
        if not getattr(model, "cond_noise", False):
            raise SystemExit("--cond_tau < 1 needs a flow trained with --cond_noise")
        model.default_tau = args.cond_tau
    synth = init_synth(DEFAULT_PLUGIN_PATH)

    results = []
    print(f"\nProcessing {len(STEM_CATALOG)} Diverse Real Bass Stems...")
    print("-" * 80)

    only = {x for x in args.stems.split(",") if x}
    for idx, item in enumerate(STEM_CATALOG):
        if only and item["id"] not in only:
            continue
        stem_id = item["id"]
        name = item["name"]
        path = item["path"]
        bpm = item["bpm"]
        style = item["style"]
        is_demucs = item.get("is_demucs", False)

        if not os.path.exists(path):
            print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] Missing file: {path}")
            continue

        y, sr = sf.read(path)
        if y.ndim > 1:
            y_raw = y.T  # [2, N] for stereo phrase
            y_mono = np.mean(y, axis=1)
        else:
            y_raw = np.stack([y, y])
            y_mono = y

        if sr != SAMPLE_RATE:
            y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=SAMPLE_RATE)
            y_raw = librosa.resample(y_raw, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # 1. Conditioning: apply 3.5 kHz LPF for Demucs stems to remove separation hiss / cymbals
        if is_demucs:
            sos = butter(4, 3500.0, btype="lowpass", fs=sr, output="sos")
            y_clean = sosfilt(sos, y_mono)
        else:
            y_clean = y_mono

        # 2. Dynamic Phrase Slicing: scan track for active bass region (4 bars or 2 bars for short clips)
        bars = 4 if len(y_mono) / sr > 10.0 else 2
        phrase_start_sample, phrase_len_samples = find_active_bass_phrase(y_clean, sr, bpm, bars=bars)
        phrase_dur = phrase_len_samples / sr

        gt_phrase_chunk = y_raw[:, phrase_start_sample : phrase_start_sample + phrase_len_samples]
        if gt_phrase_chunk.shape[1] < phrase_len_samples:
            gt_phrase_chunk = np.pad(gt_phrase_chunk, ((0, 0), (0, phrase_len_samples - gt_phrase_chunk.shape[1])))
        gt_mono_chunk = y_clean[phrase_start_sample : phrase_start_sample + phrase_len_samples]
        if len(gt_mono_chunk) < phrase_len_samples:
            gt_mono_chunk = np.pad(gt_mono_chunk, (0, phrase_len_samples - len(gt_mono_chunk)))

        gt_norm = (gt_phrase_chunk / (np.max(np.abs(gt_phrase_chunk)) + 1e-7)).astype(np.float32)

        # 3. Transcribe Phrase to MIDI via Kim's MIR pipeline
        temp_slice_wav = f"/tmp/{stem_id}_slice.wav"
        midi_path = os.path.join(MIDI_DIR, f"{stem_id}_transcribed.mid")
        sf.write(temp_slice_wav, gt_mono_chunk, sr)

        mir_cmd = [
            MIR_PYTHON, MIR_PIPELINE,
            temp_slice_wav,
            "--bpm", str(bpm),
            "--output", midi_path,
            "--flux-thresh", "0.15",
            "--rms-thresh", "0.05"
        ]
        subprocess.run(mir_cmd, capture_output=True, text=True)
        if os.path.exists(temp_slice_wav):
            os.remove(temp_slice_wav)

        events = load_pedalboard_events_from_midi(midi_path, max_dur=phrase_dur)
        note_on_pitches = [m.note for m in events if m.type == 'note_on' and m.velocity > 0]
        num_notes = len(note_on_pitches)

        # 4. Extract single clean note onset for patch inversion
        sixteenth_s = (60.0 / bpm) / 4.0
        note_dur = sixteenth_s * 0.85

        # Find first sharp transient within the phrase
        slice_onsets = librosa.onset.onset_detect(y=gt_mono_chunk, sr=sr, units='samples')
        note_start = slice_onsets[0] if len(slice_onsets) > 0 else 0

        target_note = np.zeros(AUDIO_LEN, dtype=np.float32)
        chunk = gt_mono_chunk[note_start : note_start + int(sr * sixteenth_s)]
        fade_len = int(sr * 0.005)
        if len(chunk) > fade_len:
            chunk[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        target_note[:len(chunk)] = chunk
        target_note /= (np.max(np.abs(target_note)) + 1e-7)

        # Determine target MIDI note
        if note_on_pitches:
            midi_note = Counter(note_on_pitches).most_common(1)[0][0]
            f0_hz = librosa.midi_to_hz(midi_note)
        else:
            midi_note, f0_hz = detect_stem_pitch(gt_mono_chunk, sr)

        mel = torch.from_numpy(make_mel_spec(target_note)).unsqueeze(0)

        # 5. Invert via Flow matching with Pinned ODE and multi-metric candidate reranking
        t0 = time.time()
        best, t_desc, _ = predict_and_rerank_candidates(
            model, synth, target_note, mel, midi_note, note_dur=note_dur,
            n_candidates=20, steps=25, seed=42 + idx
        )
        t_invert = time.time() - t0
        p = best["patch"]

        # 5b. Renderer-in-the-loop refinement of the salient axes against the PHRASE (refine.py).
        # Held-out test on these 24 stems (refine_phrase/): paper-MSS 11.24 -> 7.78, better on 23/24;
        # single-note refinement does not generalise, so the phrase is always the target.
        refine_info = None
        if args.refine == "phrase":
            mu_midi = os.path.join(args.muscriptor_midi_dir, f"{stem_id}_muscriptor.mid")
            ref_events = (load_pedalboard_events_from_midi(mu_midi, max_dur=phrase_dur)
                          if args.refine_midi == "muscriptor" and os.path.exists(mu_midi) else events)
            if ref_events:
                def render_phrase(patch, _ev=ref_events):
                    apply_patch(synth, patch)
                    synth.reset()
                    a = synth.process(_ev, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2).mean(axis=0)
                    return (a / (np.max(np.abs(a)) + 1e-7)).astype(np.float32)
                t1 = time.time()
                p_flow = p
                p, j1, j0, n_r, _ = refine_patch_phrase(render_phrase, p_flow, gt_mono_chunk, fit_fraction=1.0)
                save_patch(synth, p_flow, os.path.join(PRESETS_DIR, f"Inverted_{stem_id}_flow_unrefined"))
                refine_info = {"objective": [round(j0, 3), round(j1, 3)], "renders": n_r,
                               "seconds": round(time.time() - t1, 1),
                               "midi": "muscriptor" if ref_events is not events else "mir"}

        # 6. Render note-matched playback with exact transcribed MIDI events
        mu_play = os.path.join(args.muscriptor_midi_dir, f"{stem_id}_muscriptor.mid")
        if args.playback_midi == "muscriptor" and os.path.exists(mu_play):
            events = load_pedalboard_events_from_midi(mu_play, max_dur=phrase_dur)
        apply_patch(synth, p)
        synth.reset()
        if events:
            pred_phrase = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
        else:
            pred_phrase = np.zeros_like(gt_norm)

        pred_norm = (pred_phrase / (np.max(np.abs(pred_phrase)) + 1e-7)).astype(np.float32)

        # 7. Save audio files and VST preset
        gt_wav = os.path.join(CLIPS_DIR, f"{stem_id}_real.wav")
        pred_wav = os.path.join(CLIPS_DIR, f"{stem_id}_midi_playback.wav")
        sf.write(gt_wav, gt_norm.T, SAMPLE_RATE)
        sf.write(pred_wav, pred_norm.T, SAMPLE_RATE)

        preset_name = f"Inverted_{stem_id}"
        save_patch(synth, p, os.path.join(PRESETS_DIR, preset_name), copy_to_user_dir=not args.no_user_copy)

        # 8. Compute comparative audio metrics
        desc_pred = extract_pitch_and_envelope(pred_norm, SAMPLE_RATE)
        min_len = min(gt_norm.shape[1], pred_norm.shape[1])
        sc_real = float(np.mean(librosa.feature.spectral_centroid(y=np.mean(gt_norm[:, :min_len], axis=0), sr=SAMPLE_RATE)))
        sc_pred = float(np.mean(librosa.feature.spectral_centroid(y=np.mean(pred_norm[:, :min_len], axis=0), sr=SAMPLE_RATE)))

        stft_loss = best["stft_loss"]
        cont_loss = best.get("contour_loss", 0.0)

        record = {
            "index": idx + 1,
            "id": stem_id,
            "name": name,
            "style": style,
            "bpm": bpm,
            "phrase_start_s": round(phrase_start_sample / sr, 2),
            "phrase_dur_s": round(phrase_dur, 2),
            "midi_note": midi_note,
            "f0_target_hz": round(f0_hz, 1),
            "f0_pred_hz": desc_pred["f0_hz"],
            "num_midi_notes": num_notes,
            "stft_loss": round(stft_loss, 3),
            "contour_loss": round(cont_loss, 3),
            "centroid_real_hz": round(sc_real, 1),
            "centroid_pred_hz": round(sc_pred, 1),
            "osc_shape": round(p["shape"], 2),
            "cutoff": round(p["cutoff"], 2),
            "resonance": round(p["resonance"], 2),
            "filter_idx": p["filter_idx"],
            "gt_wav": gt_wav,
            "pred_wav": pred_wav,
            "midi_file": midi_path,
            "latency_s": round(t_invert, 2),
            "refine": refine_info,
        }
        results.append(record)

        print(f"[{idx+1:02d}/{len(STEM_CATALOG)}] {name:30s} | @{record['phrase_start_s']:5.1f}s | Notes: {num_notes:2d} (Root: {midi_note}) | STFT: {stft_loss:5.3f} | Cont: {cont_loss:5.3f} | Shape: {p['shape']:.2f} | Cutoff: {p['cutoff']:.2f} | Res: {p['resonance']:.2f} | {t_invert:.1f}s")

    summary_file = os.path.join(OUT_DIR, "real_stems_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"MULTI-STEM EVALUATION COMPLETE: {len(results)} stems inverted and synthesized!")
    print(f"  Audio:   {CLIPS_DIR}/")
    print(f"  MIDI:    {MIDI_DIR}/")
    print(f"  Presets: ~/Documents/Surge XT/Patches/AI Inversions/")
    print(f"  Summary: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
