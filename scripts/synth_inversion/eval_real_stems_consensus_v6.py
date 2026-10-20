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

Fixes (KUANG 2026-10-19, V6_REVIEW.md):
- The note pinned along the ODE path was normalised with a 12..96 range; every layout trains on 28..50, so the
  model was told a note up to an octave below the one it heard (and raised the octave to compensate). The range now
  comes from the model's layout (layout_v6), and notes outside it are folded by whole octaves, with the sampled
  oscillators shifted back so the patch plays at the stem's real notes.
- The second phrase note (the training phrase is note, then note + 12) is the stem's own note an octave up when the
  transcription has one; the librosa pitch shift (phase-vocoder smearing the model never saw) is the fallback only.
  results record which source each note used.
- Per model and stem, results record the oscillator-1 octave of every candidate and of the winner.
- --occam_tol: after reranking, features of the winner (waveshaper, osc 2 / 3, sub, unison, ring, noise, feedback,
  detune, drift) are switched off one at a time while the phrase score stays within the tolerance (ablation_v6).
- The private scratch directory is no longer on sys.path (it could shadow repo modules).

KUANG 2026-10-20, from Kim's 24-stem listening test (HUMAN_EVALUATION_SYNTHESIS.md); all off by default:
- --max_eval_freq HZ: score (rerank, polish, Occam, reported metrics) on lowpassed real AND rendered audio; the saved
  audio stays full band. Separation leaves hiss above ~4 kHz that a full-band loss rewards filling.
- --cond_lowpass HZ: also lowpass the audio the inverter is conditioned on. The models were trained on full-band
  renders, so this is a distribution shift: an experiment, not a fix.
- --select lexi: the phrase score becomes am6.LexiScore: per band (sub < 120 Hz, body 120..2500 Hz, air
  2500..4000 Hz) dB distances on the mid channel, compared in that order with tolerances (--lexi_rel, --lexi_abs),
  the old composite score (stereo, penalties) last. Occam then removes a feature only if no band gets worse beyond
  the tolerance (am6.lexi_no_worse) and the composite stays within --occam_tol.
- --midi_scale STEM_ID=FACTOR: multiply the stem's MIDI times by FACTOR (a MIDI exported at the wrong project tempo:
  100 BPM for a 141 BPM song -> 100/141). Every stem's MIDI is also checked against the audio onsets
  (onset_scale_check) and a warning printed when another tempo scale fits clearly better.
- Results add real / winner resonance (am6.resonance_over_notes), spectral_emd_oct and band_dists for every model, so
  human_pref_agreement.py can test them against the listening ranks.
- --freq_decay GAMMA (--decay_knee HZ): the phrase score's multi-scale STFT loss weights each bin by
  (1 + f / knee)^-gamma (am6.DecayMultiScaleSTFTLoss; V6_1_PLAN.md 4.3). Scores are then not comparable with
  unweighted runs; results record the setting.
- --square_shape has no default now: 0.5 is a saw in Surge's classic oscillator (Antigravity's scan: parity 0.289),
  so the old default pushed square targets toward a saw. The parity penalty is off unless the value is given.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import param_codec_v3 as codec_v3
import param_codec_v4 as codec_v4
import param_codec_v5 as codec_v5
import surge_spec_v3 as spec_v3
import surge_spec_v4 as spec_v4
import surge_spec_v5 as spec_v5
from models_v4 import load_inverter_v4
from layout_v6 import fold_note, layout_for_dim, octave_class, shift_octaves
from audio_utils import MultiScaleSTFTLoss, make_mel_spec, extract_pitch_and_envelope
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu
import audio_metrics_v5 as am
import audio_metrics_v6 as am6

DEFAULT_PLUGIN_PATH = spec_v4.DEFAULT_PLUGIN_PATH
SAMPLE_RATE = 44100
AUDIO_LEN = 35280  # 0.8s


def scale_notes(notes, factor):
    """MIDI note times * factor (wrong export tempo; see --midi_scale)."""
    return [dict(n, start=n["start"] * factor, dur=n["dur"] * factor) for n in notes]


def onset_scale_check(notes, y_mono, sr=SAMPLE_RATE, scales=None, hop=512):
    """How well the transcribed onsets line up with the audio's onset strength when MIDI time is multiplied by each
    scale (0.6..1.5). Score = mean onset strength at the predicted onsets (+-1 frame) over the clip's mean.
    -> (best scale, its score, score at scale 1)."""
    if scales is None:
        scales = np.round(np.linspace(0.6, 1.5, 91), 3)
    if len(notes) < 4:
        return 1.0, float("nan"), float("nan")
    env = librosa.onset.onset_strength(y=np.asarray(y_mono, np.float32), sr=sr, hop_length=hop)
    env = np.maximum.reduce([env, np.roll(env, 1), np.roll(env, -1)])
    mean = float(env.mean()) + 1e-9
    starts = np.array([n["start"] for n in notes])

    def score(sc):
        fr = np.round(starts * sc * sr / hop).astype(int)
        fr = fr[(fr >= 0) & (fr < len(env))]
        return float(env[fr].mean() / mean) if len(fr) >= 4 else 0.0
    res = [(score(sc), float(sc)) for sc in scales]
    best = max(res)
    return best[1], best[0], score(1.0)


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


def filter_transcription(notes, max_midi=None, drop_polyphony=False):
    """V6_SPECIFICATION.md 3.5 transcription clean-up: drop notes above max_midi, and with drop_polyphony every note
    that overlaps another in time (a cluster is a transcription error on a monophonic bass)."""
    keep = [n for n in notes if max_midi is None or n["note"] <= max_midi]
    if drop_polyphony:
        def overlaps(a, b):
            return a is not b and a["start"] < b["start"] + b["dur"] and b["start"] < a["start"] + a["dur"]
        keep = [n for n in keep if not any(overlaps(n, m) for m in keep)]
    return keep


def events_from_notes(notes, max_dur):
    """Note list -> pedalboard MIDI events (the phrase the reranker renders), cut at max_dur."""
    ev = []
    for n in notes:
        if n["start"] >= max_dur:
            continue
        ev.append(mido.Message("note_on", note=n["note"], velocity=n.get("vel", 100), time=n["start"]))
        ev.append(mido.Message("note_off", note=n["note"], velocity=0, time=min(n["start"] + n["dur"], max_dur - 0.005)))
    ev.sort(key=lambda m: (m.time, m.type == "note_on"))
    return ev


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


def cut_note(y_mono, n, sr=SAMPLE_RATE, next_start=None):
    """One transcribed note -> peak-normalised chunk (125..240 ms, 5 ms fade out) and its length in seconds.
    Ensures at least 125 ms (the training note-1 gate). KUANG 2026-10-19: audio is taken only up to the next
    note's onset (next_start, seconds) and the rest of the 125 ms is silence. Reading past it put the start of the
    next note into the conditioning clip whenever notes are shorter than 125 ms (a 16th at 140 BPM is 107 ms)."""
    s_idx = max(0, int(round(n["start"] * sr)))
    chunk_dur = min(0.240, max(0.125, n["dur"]))
    n_len = int(round(chunk_dur * sr))
    end = s_idx + n_len
    if next_start is not None:
        end = min(end, max(s_idx + 1, int(round(next_start * sr))))
    audio = y_mono[s_idx:end].copy()
    fade_len = int(sr * 0.005)
    if len(audio) > fade_len:
        audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    chunk = np.zeros(n_len, dtype=np.float32)
    chunk[:len(audio)] = audio[:n_len]
    return chunk / (np.max(np.abs(chunk)) + 1e-7), chunk_dur


def next_onset(n, notes):
    """Start of the first transcribed note after n's start, or None."""
    later = [m["start"] for m in notes if m["start"] > n["start"] + 1e-4]
    return min(later) if later else None


POLISH_KEYTRACK_RAW = (0.50, 0.625, 0.75, 0.875)   # 0 / 50 / 100 / 150 %; -100 % (0.25) is not in the vector space


def polish_vector(vec, spec, score_fn, centroid_fn, real_centroid, pitch_span, start_score, start_out,
                  keytrack_span=7, centroid_tol_hz=120.0):
    """V6.1 local polish (V6_1_PLAN.md 2.4) in the stored-vector space, so Occam and the exported patch continue
    from the polished vector. score_fn(vec) -> (score, out); centroid_fn(out) -> Hz. Each stage tries its
    candidates from the stage's starting vector and keeps the best that lowers the score:
      1. cutoff (filter 1 and 2 together) +- 0.5 / 1.0 x clip(centroid error / 2500, +-0.15), if |error| > 120 Hz;
      2. filter envelope amount (both filters) +-0.08 / +-0.15;
      3. keytrack 0 / 50 / 100 / 150 % (filter 2 to the same raw value), only if the phrase spans >= 7 semitones.
    Values stay inside the spec's CONT_BOUNDS. -> (vec, score, out, log [(stage, value, score)])."""
    I = spec.PARAM_INDEX
    bounds = getattr(spec, "CONT_BOUNDS", {})

    def clip(name, x):
        lo, hi = bounds.get(name, (0.0, 1.0))
        return float(np.clip(x, lo, hi))
    best_v, best_s, best_out, log = np.array(vec, np.float32, copy=True), start_score, start_out, []

    def stage(name, cands):
        nonlocal best_v, best_s, best_out
        base_s = best_s
        found = None
        for tag, v in cands:
            s, out = score_fn(v)
            if s < best_s:
                best_v, best_s, best_out, found = v, s, out, (name, tag, round(float(s), 4))
        if found is not None and best_s < base_s:
            log.append(found)

    err = real_centroid - centroid_fn(best_out)
    if abs(err) > centroid_tol_hz:
        shift = float(np.clip(err / 2500.0, -0.15, 0.15))
        base = best_v.copy()
        cands = []
        for mult in (0.5, 1.0):
            v = base.copy()
            v[I["cutoff"]] = clip("cutoff", base[I["cutoff"]] + shift * mult)
            v[I["filter_2_cutoff"]] = clip("filter_2_cutoff", base[I["filter_2_cutoff"]] + shift * mult)
            cands.append((round(shift * mult, 4), v))
        stage("cutoff", cands)
    base = best_v.copy()
    cands = []
    for d in (-0.15, -0.08, 0.08, 0.15):
        v = base.copy()
        v[I["feg_amount"]] = clip("feg_amount", base[I["feg_amount"]] + d)
        v[I["filter_2_feg_amount"]] = clip("filter_2_feg_amount", base[I["filter_2_feg_amount"]] + d)
        cands.append((d, v))
    stage("feg_amount", cands)
    if pitch_span >= keytrack_span:
        base = best_v.copy()
        current_raw = 0.5 + 0.5 * float(base[I["keytrack"]])          # stored keytrack p -> raw 0.5 + 0.5 p
        cands = []
        for raw in POLISH_KEYTRACK_RAW:
            if abs(raw - current_raw) < 0.05:
                continue
            v = base.copy()
            v[I["keytrack"]] = clip("keytrack", (raw - 0.5) / 0.5)
            v[I["filter_2_keytrack"]] = clip("filter_2_keytrack", raw)  # filter 2 keytrack is stored raw
            cands.append((raw, v))
        stage("keytrack_raw", cands)
    return best_v, best_s, best_out, log


def octave_partner(n, notes):
    """The longest transcribed note exactly 12 semitones above n, or None."""
    up = [m for m in notes if m["note"] == n["note"] + 12]
    return max(up, key=lambda m: m["dur"]) if up else None


def build_conditioning_targets(chunk, sixteenth_s, sr=SAMPLE_RATE, chunk_oct=None):
    """Build matched 2-note octave phrase and rhythmic rolling pattern to fill 0.8s buffer.
    chunk_oct: the real note an octave up (preferred); None -> phase-vocoder pitch shift of chunk."""
    s2 = int(0.250 * sr)
    l1 = min(len(chunk), s2)
    t_phrase = np.zeros(AUDIO_LEN, dtype=np.float32)
    t_phrase[:l1] = chunk[:l1]

    if chunk_oct is None:
        chunk_oct = librosa.effects.pitch_shift(chunk[:l1], sr=sr, n_steps=12)
    chunk_oct = np.asarray(chunk_oct, dtype=np.float32)[:l1].copy()
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
    """Adaptive candidate sampler for 92-d (v3), 108-d (v4), 126-d (v5) and 114-d (v6) flow models. The note is
    folded into the layout's trained range by whole octaves and the oscillators are shifted back afterwards, so the
    returned vectors play right at midi_note (v6: the note is the model's input and osc 1 is fixed at octave 0, so
    nothing needs shifting). -> (vectors [k, P], octaves folded)."""
    device = next(model.parameters()).device
    model.eval()
    mel = mel.to(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))

    layout = layout_for_dim(model.param_dim)
    model_note, folded = fold_note(midi_note, layout)
    note_t = torch.full((k,), float(layout.note_unit(model_note)), device=device)
    
    mel_k = mel.repeat(k, 1, 1)
    emb = model.encoder(mel_k)
    x = torch.randn(k, model.param_dim, device=device, generator=gen)
    
    for i in range(steps):
        tv = i / steps
        t = torch.full((k,), tv, device=device)
        if layout.note_conditioned:
            x = x + model(x, t, audio_emb=emb, note=note_t) * (1.0 / steps)
        else:
            x[:, 0] = (1.0 - tv) * x[:, 0] + tv * note_t
            x = x + model(x, t, audio_emb=emb) * (1.0 / steps)

    if not layout.note_conditioned:
        x[:, 0] = note_t
    p_ord = layout.decode(torch.clamp(x, 0.0, 1.0) if layout.note_conditioned else x, note_t).cpu().numpy()
    if folded:
        p_ord, _ = shift_octaves(p_ord, -folded, layout)
    return p_ord, folded


def build_consensus_vector(param_vectors, param_dim=92):
    """Build consensus vector from a pool of parameter vectors.
    Uses median for continuous parameters and mode (majority vote) for categorical parameters.
    """
    spec = layout_for_dim(param_dim).spec
    cat_indices, cont_indices, num_params = spec.CAT_INDICES, spec.CONT_INDICES, spec.NUM_PARAMS
    
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


def vector_to_patch_adaptive(vec, param_dim=92):
    return layout_for_dim(param_dim).spec.vector_to_patch(vec)


def apply_patch_adaptive(plugin, patch, param_dim=92):
    if layout_for_dim(param_dim).name in ("v5", "v6"):           # v6 patches are surge_spec_v5 patch dicts
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


def format_patch_details(patch, param_dim=92):
    """Format key synth parameters into human-readable summary string."""
    if param_dim in (126, 114, 108):
        spec = spec_v4 if param_dim == 108 else spec_v5
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


def evaluate_models_consensus(models_dict, out_dir, device="cpu", max_notes=6, k_per_note=2, steps=25, seed=42,
                              second_note="auto", occam_tol=None, occam_abs=0.0, max_midi=None, drop_polyphony=False,
                              polish=True, square_shape=None, max_eval_freq=None, cond_lowpass=None, select="mss",
                              lexi_rel=0.05, lexi_abs=0.2, midi_scale=None, freq_decay=None, decay_knee=900.0):
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
    loss_fn = am6.DecayMultiScaleSTFTLoss(decay_knee, freq_decay) if freq_decay else MultiScaleSTFTLoss()
    midi_scale = dict(midi_scale or {})
    band = (lambda x: am6.lowpass(x, SAMPLE_RATE, max_eval_freq)) if max_eval_freq else (lambda x: x)
    lexi_bands = tuple(b for b in am6.BANDS_HZ if b[1] < (max_eval_freq or 4000.0))

    results = []

    for idx, item in enumerate(STEM_CATALOG):
        stem_id = item["id"]
        name = item["name"]
        bpm = item["bpm"]
        style = item["style"]

        real_wav_path = os.path.join(audio_ref_dir, f"{stem_id}_real.wav")
        midi_fix_path = os.path.join(midi_muscriptor_dir, f"{stem_id}_muscriptor_FIX.mid")
        if os.path.exists(midi_fix_path):
            midi_path = midi_fix_path
            print(f"[{idx+1:02d}/24] Using user-fixed MIDI: {os.path.basename(midi_fix_path)}", flush=True)
        else:
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
        y_real_full = y_real                                   # the saved reference stays full band
        y_cond_mono = am6.lowpass(y_real_mono, sr, cond_lowpass) if cond_lowpass else y_real_mono
        if max_eval_freq:                                      # everything scored below is band limited
            real_L, real_R = band(real_L), band(real_R)
            y_real = np.stack([real_L, real_R], axis=-1)
            y_real_mono = y_real.mean(axis=1)

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
        scale = float(midi_scale.get(stem_id, 1.0))
        real_centroid = get_centroid(y_real_mono, sr)

        real_cent_L = get_centroid(real_L, sr)
        real_cent_R = get_centroid(real_R, sr)
        real_cent_asym = abs(real_cent_L - real_cent_R) / max(real_cent_L, real_cent_R, 1.0)

        # Extract notes and select clean representative subset
        all_notes = extract_notes_from_midi(midi_path, max_dur=phrase_dur)
        if scale != 1.0:
            all_notes = [n for n in scale_notes(extract_notes_from_midi(midi_path), scale) if n["start"] < phrase_dur]
            events = events_from_notes(all_notes, phrase_dur)
            print(f"  MIDI times x {scale:.4f} (--midi_scale)", flush=True)
        best_sc, sc_best, sc_one = onset_scale_check(all_notes, y_real_mono, sr)
        if np.isfinite(sc_best) and abs(best_sc - 1.0) > 0.03 and sc_best > 1.2 * sc_one:
            print(f"  WARNING: MIDI onsets fit the audio better at time x {best_sc:.3f} (score {sc_best:.2f} vs "
                  f"{sc_one:.2f}); wrong export tempo? --midi_scale {stem_id}={best_sc * scale:.4f}", flush=True)
        if max_midi is not None or drop_polyphony:
            n_before = len(all_notes)
            all_notes = filter_transcription(all_notes, max_midi, drop_polyphony)
            events = events_from_notes(all_notes, phrase_dur)      # rerank the cleaned phrase, not the raw one
            print(f"  transcription filter: {n_before} -> {len(all_notes)} notes", flush=True)
        selected_notes = select_representative_notes(all_notes, max_notes=max_notes)

        pitches = [n["note"] for n in selected_notes]
        print(f"\n[{idx+1:02d}/24] Inverting '{name}' ({style}, {bpm} BPM, {real_stereo_class} W={real_width:.2f}, {len(all_notes)} notes -> selected {len(selected_notes)} notes: {pitches})...", flush=True)

        ref_phrase_dest = os.path.join(listen_dir, f"{stem_id}__real_phrase.wav")
        sf.write(ref_phrase_dest, y_real_full, sr, subtype="PCM_16")

        # Nominal root, sub-octave threshold, and harmonic parity
        root_midi = max(set(pitches), key=pitches.count) if pitches else 36
        nominal_f0 = 440.0 * (2.0 ** ((root_midi - 69.0) / 12.0))
        sub_thresh = 0.75 * nominal_f0
        fft_real = np.abs(np.fft.rfft(y_real_mono))
        freqs_real = np.fft.rfftfreq(len(y_real_mono), 1.0 / sr)
        t_sub = float(np.sum(fft_real[freqs_real < sub_thresh]**2) / (np.sum(fft_real**2) + 1e-9))
        real_parity = am6.parity_over_notes(y_real_mono, selected_notes, sr)    # each note at its own f0
        real_res_db, real_res_hz = am6.resonance_over_notes(y_real_mono, selected_notes, sr)

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
            "real_resonance_db": None if not np.isfinite(real_res_db) else round(float(real_res_db), 2),
            "real_resonance_hz": None if not np.isfinite(real_res_hz) else round(float(real_res_hz), 1),
            "midi_scale": scale,
            "midi_scale_check": {"best": best_sc, "score_best": round(sc_best, 3) if np.isfinite(sc_best) else None,
                                 "score_at_1": round(sc_one, 3) if np.isfinite(sc_one) else None},
            "select": select,
            "max_eval_freq": max_eval_freq,
            "cond_lowpass": cond_lowpass,
            "freq_decay": None if not freq_decay else {"gamma": freq_decay, "knee_hz": decay_knee},
            "models": {}
        }

        y_real_norm = y_real_mono / (np.max(np.abs(y_real_mono)) + 1e-7)

        for m_name, model in loaded_models.items():
            t0 = time.time()
            param_dim = model.param_dim

            # Collect candidates across all selected notes
            cand_vecs, note_info = [], []
            for n_idx, n in enumerate(selected_notes):
                chunk, chunk_dur = cut_note(y_cond_mono, n, sr, next_start=next_onset(n, all_notes))
                sixteenth_s = chunk_dur
                partner = octave_partner(n, all_notes) if second_note == "auto" else None
                chunk_oct = (cut_note(y_cond_mono, partner, sr, next_start=next_onset(partner, all_notes))[0]
                             if partner is not None else None)

                t_phrase, t_rolling = build_conditioning_targets(chunk, sixteenth_s, sr=sr, chunk_oct=chunk_oct)
                mel_p = torch.from_numpy(make_mel_spec(t_phrase)).unsqueeze(0).float()
                mel_r = torch.from_numpy(make_mel_spec(t_rolling)).unsqueeze(0).float()

                vecs_p, folded = sample_candidates_adaptive(model, mel_p, n["note"], k=k_per_note, steps=steps, seed=seed + idx * 100 + n_idx)
                vecs_r, _ = sample_candidates_adaptive(model, mel_r, n["note"], k=k_per_note, steps=steps, seed=seed + idx * 100 + n_idx + 50)
                note_info.append({"note": int(n["note"]), "second_note": "real" if partner is not None else "pitch_shift",
                                  "octaves_folded": int(folded)})

                for c in range(k_per_note):
                    cand_vecs.append(vecs_p[c])
                    cand_vecs.append(vecs_r[c])

            cand_vecs = np.array(cand_vecs)

            # Build consensus vector
            consensus_vec = build_consensus_vector(cand_vecs, param_dim=param_dim)
            
            # Pool: consensus patch + all individual note candidate patches
            pool_vecs = [consensus_vec] + list(cand_vecs)
            # 32nd-note BPM-adaptive release ceiling: guarantees notes decay cleanly before next 16th note
            max_rel_raw = bpm_to_max_release_raw(bpm)

            def capped_patch(v):
                p = vector_to_patch_adaptive(v, param_dim=param_dim)
                p["aeg_release"] = min(float(p.get("aeg_release", 0.23)), max_rel_raw)
                p["feg_release"] = min(float(p.get("feg_release", p["aeg_release"])), max_rel_raw)
                return p

            pool_patches = [capped_patch(v) for v in pool_vecs]

            def phrase_score(p, retrig):
                """Render the stem's MIDI phrase with patch p -> (score, stereo audio, width, corr)."""
                spec_v4.flush_synth(synth)
                apply_patch_adaptive(synth, p, param_dim=param_dim)
                synth.parameters["a_osc_1_retrigger"].raw_value = 1.0 if retrig else 0.0
                synth.parameters["a_osc_2_retrigger"].raw_value = 1.0 if retrig else 0.0
                synth.parameters["a_osc_3_retrigger"].raw_value = 1.0 if retrig else 0.0

                pred_audio = synth.process(events, duration=phrase_dur, sample_rate=SAMPLE_RATE, num_channels=2)
                pred_L, pred_R = band(pred_audio[0]), band(pred_audio[1])
                pred_scored = np.stack([pred_L, pred_R]) if max_eval_freq else pred_audio
                pred_mid = (pred_L + pred_R) / np.sqrt(2.0)
                pred_side = (pred_L - pred_R) / np.sqrt(2.0)
                pred_width, pred_corr = compute_stereo_metrics(pred_scored, is_channels_first=True)

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

                # Harmonic parity penalty for square wave disambiguation
                parity_pen = 0.0
                if square_shape is not None and np.isfinite(real_parity) and real_parity < 0.12:
                    cand_shape = float(p.get("shape", 0.0))
                    shape_pen = abs(cand_shape - square_shape) * 1.5
                    ring_pen = (float(p.get("ring_1x2_volume", 0.0)) + float(p.get("ring_2x3_volume", 0.0))) * 1.2
                    f2_res_pen = float(p.get("filter_2_resonance", 0.0)) * 0.8
                    parity_pen = shape_pen + ring_pen + f2_res_pen

                # Stereo-aware score
                if real_width > 0.03:
                    pred_side_norm = pred_side / (np.max(np.abs(pred_mid)) + 1e-7)
                    stft_side = float(loss_fn(pred_side_norm[:l], real_side_norm[:l]))
                    width_err = abs(pred_width - real_width)
                    score = stft_mid + 0.4 * stft_side + 0.6 * width_err + sub_pen + bal_pen + cent_bal_pen + parity_pen
                else:
                    score = stft_mid + 1.2 * pred_width + sub_pen + bal_pen + cent_bal_pen + parity_pen
                if select == "lexi":
                    bd = am6.band_dists(pred_mid_norm[:l], real_mid_norm[:l], sr, bands=lexi_bands)
                    score = am6.LexiScore([bd[b[0]] for b in lexi_bands], score, rel=lexi_rel, abs_db=lexi_abs)
                return score, pred_audio, pred_width, pred_corr

            # Re-rank candidates on full phrase in Surge XT (with stereo & retrigger search)
            best_score = float("inf")
            best_patch = None
            best_audio = None
            best_type = "Consensus"
            best_idx = 0
            best_retrig = True
            best_width = 0.0
            best_corr = 1.0

            for p_idx, p in enumerate(pool_patches):
                for retrig in [True, False]:
                    score, pred_audio, pred_width, pred_corr = phrase_score(p, retrig)
                    if score < best_score:
                        best_score = score
                        best_patch = p
                        best_audio = pred_audio
                        best_type = "Consensus" if p_idx == 0 else f"NoteCandidate_{p_idx}"
                        best_idx = p_idx
                        best_retrig = retrig
                        best_width = pred_width
                        best_corr = pred_corr

            lay = layout_for_dim(param_dim)
            win_vec = np.array(pool_vecs[best_idx], np.float32)
            polish_log, score_before_polish = [], best_score
            if polish:
                phrase_pitches = [n["note"] for n in all_notes]
                span = (max(phrase_pitches) - min(phrase_pitches)) if len(phrase_pitches) > 1 else 0

                def score_vec(v):
                    s_, a_, w_, c_ = phrase_score(capped_patch(v), best_retrig)
                    return s_, (a_, w_, c_)
                win_vec, best_score, (best_audio, best_width, best_corr), polish_log = polish_vector(
                    win_vec, lay.spec, score_vec, lambda out: get_centroid(band(out[0].mean(axis=0)), sr), real_centroid,
                    span, best_score, (best_audio, best_width, best_corr))
                best_patch = capped_patch(win_vec)
            occam_log, score_before_occam = [], best_score
            if occam_tol is not None:
                # Occam (ablation_v6): switch off features whose removal does not make the phrase score worse by
                # more than max(occam_abs, occam_tol * score); greedy, best removal first, up to 6 rounds.
                from ablation_v6 import DEFAULT_OCCAM, active, simplify
                for _ in range(6):
                    trials = []
                    for fname in DEFAULT_OCCAM:
                        if active(win_vec, fname, lay.spec):
                            v2 = simplify(win_vec, fname, lay.spec)
                            p2 = capped_patch(v2)
                            trials.append((*phrase_score(p2, best_retrig), fname, v2, p2))
                    if not trials:
                        break
                    s2, a2, w2, c2, fname, v2, p2 = min(trials, key=lambda tr: tr[0])
                    if float(s2) > float(best_score) + max(occam_abs, occam_tol * float(best_score)):
                        break
                    if select == "lexi" and not am6.lexi_no_worse(s2, best_score, lexi_rel, lexi_abs):
                        break
                    occam_log.append({"removed": fname, "score": round(float(s2), 4)})
                    best_score, best_audio, best_width, best_corr, win_vec, best_patch = s2, a2, w2, c2, v2, p2

            el = time.time() - t0

            # Compute detailed audio metrics for winner
            pred_winner_mono = band(best_audio.mean(axis=0))
            pred_winner_norm = pred_winner_mono / (np.max(np.abs(pred_winner_mono)) + 1e-7)
            win_centroid = get_centroid(pred_winner_norm, sr)
            centroid_err = win_centroid - real_centroid
            l_eval = min(len(pred_winner_norm), len(y_real_norm))
            # KUANG 2026-10-19: am.mel_ssim_dist is 1 - SSIM (lower = closer). Until now it was stored as "mel_ssim"
            # and read as a similarity (V6_SPECIFICATION.md 2.4 "Mel SSIM up"), which inverts the comparison.
            # Both are stored now: mel_ssim = SSIM (higher = closer), mel_ssim_dist = 1 - SSIM.
            mel_ssim_dist = float(am.mel_ssim_dist(y_real_norm[:l_eval], pred_winner_norm[:l_eval]))
            mel_ssim = 1.0 - mel_ssim_dist
            win_bands = am6.band_dists(pred_winner_norm[:l_eval], y_real_norm[:l_eval], sr, bands=lexi_bands)
            win_emd = am6.spectral_emd_oct(pred_winner_norm[:l_eval], y_real_norm[:l_eval], sr,
                                           fmax=min(4000.0, max_eval_freq or 4000.0))
            win_res_db, win_res_hz = am6.resonance_over_notes(pred_winner_mono, selected_notes, sr)

            # Save winner audio
            cand_filename = f"{stem_id}__{m_name}.wav"
            cand_dest = os.path.join(listen_dir, cand_filename)
            pred_stereo_norm = (best_audio / (np.max(np.abs(best_audio)) + 1e-7)).astype(np.float32)
            sf.write(cand_dest, pred_stereo_norm.T, sr, subtype="PCM_16")

            if lay.note_conditioned:                                 # v6: osc 1 is always at octave 0
                cand_oct, win_oct = [0] * len(cand_vecs), 0
            else:
                oct_i = lay.spec.PARAM_INDEX["osc_1_octave"]
                cand_oct = [int(c) - 3 for c in octave_class(cand_vecs[:, oct_i], lay)]
                win_oct = int(octave_class(win_vec[oct_i], lay)) - 3

            patch_summary = format_patch_details(best_patch, param_dim=param_dim)
            patch_summary += f" | Retrig: {'On' if best_retrig else 'Off'}"

            stem_res["models"][m_name] = {
                "phrase_mss": round(float(best_score), 3),
                "band_dists": {k: round(v, 3) for k, v in win_bands.items()},
                "spectral_emd_oct": round(win_emd, 4),
                "resonance_db": None if not np.isfinite(win_res_db) else round(float(win_res_db), 2),
                "resonance_hz": None if not np.isfinite(win_res_hz) else round(float(win_res_hz), 1),
                "mel_ssim": round(mel_ssim, 4),
                "mel_ssim_dist": round(mel_ssim_dist, 4),
                "centroid_hz": round(win_centroid, 1),
                "centroid_delta_hz": round(centroid_err, 1),
                "stereo_width": round(best_width, 3),
                "stereo_corr": round(best_corr, 3),
                "width_delta": round(best_width - real_width, 3),
                "retrigger": "On" if best_retrig else "Off",
                "winner_type": best_type,
                "osc1_octave_candidates": cand_oct,
                "osc1_octave_winner": win_oct,
                "occam": occam_log,
                "polish": polish_log,
                "phrase_score_before_polish": round(float(score_before_polish), 3),
                "real_parity": None if not np.isfinite(real_parity) else round(float(real_parity), 4),
                "phrase_score_before_occam": round(float(score_before_occam), 3),
                "notes": note_info,
                "elapsed_s": round(el, 2),
                "audio_file": cand_filename,
                "patch_summary": patch_summary,
                "best_patch": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best_patch.items()}
            }

            print(f"  [{m_name}] MSS: {best_score:.3f} | SSIM (higher = closer): {mel_ssim:.4f} | Cent: {win_centroid:.0f}Hz ({centroid_err:+.0f}Hz) | W: {best_width:.2f} (dW: {best_width - real_width:+.2f}) | Retrig: {'On' if best_retrig else 'Off'} | {best_type} in {el:.1f}s", flush=True)

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
                    <span class="badge">SSIM (1 = same): {ssim:.3f}</span>
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
    parser.add_argument("--no_polish", action="store_true", help="skip the V6.1 cutoff / FEG / keytrack polish")
    parser.add_argument("--square_shape", type=float, default=None,
                        help="osc shape value of a symmetric square in Surge (check_v6_spec.py --shape_scan); the "
                             "parity penalty is off without it")
    parser.add_argument("--max_eval_freq", type=float, default=None,
                        help="score on audio lowpassed at this frequency (e.g. 4000); saved audio stays full band")
    parser.add_argument("--cond_lowpass", type=float, default=None,
                        help="also lowpass the conditioning audio (distribution shift: experiment)")
    parser.add_argument("--select", choices=("mss", "lexi"), default="mss",
                        help="lexi: band-hierarchical score (sub > body > air), see the module docstring")
    parser.add_argument("--lexi_rel", type=float, default=0.05, help="relative band tolerance for --select lexi")
    parser.add_argument("--lexi_abs", type=float, default=0.2, help="absolute band tolerance (dB) for --select lexi")
    parser.add_argument("--freq_decay", type=float, default=None, metavar="GAMMA",
                        help="weight the phrase STFT loss by (1 + f / knee)^-GAMMA (V6_1_PLAN.md 4.3: 1.0..1.5)")
    parser.add_argument("--decay_knee", type=float, default=900.0, help="knee frequency for --freq_decay (Hz)")
    parser.add_argument("--midi_scale", action="append", default=[], metavar="STEM_ID=FACTOR",
                        help="multiply a stem's MIDI times (wrong export tempo); repeatable")
    parser.add_argument("--max_midi", type=int, default=None,
                        help="drop transcribed notes above this MIDI note (V6_SPECIFICATION.md 3.5: 60)")
    parser.add_argument("--drop_polyphony", action="store_true",
                        help="drop transcribed notes that overlap another note (clusters); the reranked phrase too")
    parser.add_argument("--second_note", choices=("auto", "pitch_shift"), default="auto",
                        help="auto: the stem's own note an octave up when transcribed, else pitch shift; "
                             "pitch_shift: always (the pre-2026-10-19 behaviour)")
    parser.add_argument("--occam_tol", type=float, default=None,
                        help="switch off features of the winner while the phrase score rises by at most this share "
                             "(e.g. 0.02); off by default")
    parser.add_argument("--occam_abs", type=float, default=0.0, help="absolute score tolerance for --occam_tol")
    args = parser.parse_args()

    models = {name: path for name, path in args.model}
    evaluate_models_consensus(models, args.out_dir, device=args.device, max_notes=args.max_notes,
                              k_per_note=args.k, steps=args.steps, seed=args.seed, second_note=args.second_note,
                              occam_tol=args.occam_tol, occam_abs=args.occam_abs, max_midi=args.max_midi,
                              drop_polyphony=args.drop_polyphony, polish=not args.no_polish,
                              square_shape=args.square_shape, max_eval_freq=args.max_eval_freq,
                              cond_lowpass=args.cond_lowpass, select=args.select, lexi_rel=args.lexi_rel,
                              lexi_abs=args.lexi_abs, freq_decay=args.freq_decay, decay_knee=args.decay_knee,
                              midi_scale={k: float(v) for k, v in (a.split("=", 1) for a in args.midi_scale)})
    os._exit(0)

