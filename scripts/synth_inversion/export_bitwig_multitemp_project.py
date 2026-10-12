#!/usr/bin/env python3
# Owner: ANTIGRAVITY (created 2026-10-12)
"""Export Surge XT Arm A2 & A3 Multi-Temperature Inversions into a native Bitwig DAWproject (.dawproject).

Features:
- Sequential arrangement inside Group Tracks per stem (eliminates vertical track clutter).
- Clean non-overlapping timeline playback: Target -> A2 EMA -> Target -> A2 Online -> Target -> A3 EMA -> Target -> A3 Online.
- Cue Markers for every single clip on the Bitwig timeline for instant jumping (Alt+Right / Alt+Left in Bitwig).
- Muted parallel track for raw (unsnapped) flow predictions for immediate A/B checking.
- Synchronized MIDI notes on a parallel track with Channel 9 percussion filtered out.
- Live Surge XT Instrument Track with the inverted VST3 preset loaded (muted by default).
"""

import os
import sys
import glob
import json
import shutil
import tempfile
import argparse
import mido
import soundfile as sf
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu


def parse_clean_midi_notes_in_beats(midi_path: str, max_dur_s: float, bpm: float):
    """Parse a MIDI file and extract note events in musical beats relative to clip start.
    Filters out Channel 9 (percussion).
    """
    if not os.path.exists(midi_path):
        return []
    mid = mido.MidiFile(midi_path)
    events = []
    t_s = 0.0
    active_notes = {}
    for msg in mid:
        t_s += msg.time
        if max_dur_s is not None and t_s >= max_dur_s:
            break
        if getattr(msg, 'channel', 0) == 9:
            continue
        beat = t_s * (bpm / 60.0)
        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = (beat, msg.velocity / 127.0)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                start_beat, vel = active_notes.pop(msg.note)
                dur_beats = max(0.05, beat - start_beat)
                events.append((start_beat, dur_beats, msg.note, vel))

    # Close any still-open notes
    if active_notes and max_dur_s is not None:
        cutoff_beat = (max_dur_s - 0.005) * (bpm / 60.0)
        for note, (start_beat, vel) in active_notes.items():
            dur_beats = max(0.05, cutoff_beat - start_beat)
            events.append((start_beat, dur_beats, note, vel))

    events.sort(key=lambda x: x[0])
    return events


def build_multitemp_dawproject(
    a2_audio_dir: str,
    a3_audio_dir: str,
    midi_dir: str,
    out_path: str,
    title: str = "Surge XT A2 & A3 Multi-Temp Evaluation",
    temps=("t03", "t05", "t07", "t085", "t10"),
    stems_to_include=None,
    a2_preset_dir=None,
    a3_preset_dir=None
):
    a2_audio_dir = resolve_mantu(a2_audio_dir)
    a3_audio_dir = resolve_mantu(a3_audio_dir)
    midi_dir = resolve_mantu(midi_dir)
    if a2_preset_dir: a2_preset_dir = resolve_mantu(a2_preset_dir)
    if a3_preset_dir: a3_preset_dir = resolve_mantu(a3_preset_dir)

    temp_labels = {
        "t03": "tau=0.30",
        "t05": "tau=0.50",
        "t07": "tau=0.70",
        "t085": "tau=0.85",
        "t10": "tau=1.00",
    }

    stems = STEM_CATALOG
    if stems_to_include:
        stems = [s for s in stems if any(k in s["id"] for k in stems_to_include)]

    # Filter to stems that actually have real audio
    available_stems = []
    for s in stems:
        sid = s["id"]
        rf = os.path.join(a2_audio_dir, f"{sid}_real.wav")
        if not os.path.exists(rf):
            rf = os.path.join(a3_audio_dir, f"{sid}_real.wav")
        if os.path.exists(rf):
            s_copy = dict(s)
            s_copy["real_path"] = rf
            available_stems.append(s_copy)

    if not available_stems:
        raise RuntimeError(f"No rendered stems found in {a2_audio_dir} or {a3_audio_dir}")

    print(f"Building DAWproject for {len(available_stems)} stems across Arms A2 & A3...")

    project = ET.Element("Project", version="1.0")
    ET.SubElement(project, "Application", name="Bitwig Studio", version="5.0")

    ref_bpm = available_stems[0].get("bpm", 140.0)
    transport = ET.SubElement(project, "Transport")
    ET.SubElement(transport, "Tempo", max="666.0", min="20.0", unit="bpm", value=f"{ref_bpm:.2f}", id="id_tempo", name="Tempo")
    ET.SubElement(transport, "TimeSignature", denominator="4", numerator="4", id="id_ts")

    structure = ET.SubElement(project, "Structure")
    arrangement = ET.SubElement(project, "Arrangement", id="id_arr")
    arr_lanes = ET.SubElement(arrangement, "Lanes", timeUnit="beats", id="id_lanes")
    markers_el = ET.SubElement(arrangement, "Markers", timeUnit="beats", id="id_markers")

    with tempfile.TemporaryDirectory() as tmp_build:
        proj_audio_dir = os.path.join(tmp_build, "audio")
        proj_plugins_dir = os.path.join(tmp_build, "plugins")
        os.makedirs(proj_audio_dir, exist_ok=True)
        os.makedirs(proj_plugins_dir, exist_ok=True)

        current_beat = 0.0
        id_counter = 100

        for s_idx, stem in enumerate(available_stems):
            stem_id = stem["id"]
            stem_name = stem["name"]
            bpm = stem.get("bpm", 140.0)
            real_wav = stem["real_path"]

            info = sf.info(real_wav)
            dur_s = info.duration
            phrase_dur_beats = round((dur_s * (bpm / 60.0)) / 4.0) * 4.0
            if phrase_dur_beats < 4.0:
                phrase_dur_beats = 4.0

            # Copy real audio to package
            rel_real_audio = f"audio/{stem_id}_real.wav"
            shutil.copy2(real_wav, os.path.join(tmp_build, rel_real_audio))

            # Transcribed MIDI
            mid_file = os.path.join(midi_dir, f"{stem_id}_muscriptor.mid")
            note_events = parse_clean_midi_notes_in_beats(mid_file, dur_s, bpm) if os.path.exists(mid_file) else []

            # Check for best VST3 preset (e.g. A3 EMA t07 harm or A2 EMA t07 harm)
            preset_file = None
            if a3_preset_dir:
                p_cand = os.path.join(a3_preset_dir, f"{stem_id}_ema_t07_harm.vstpreset")
                if os.path.exists(p_cand): preset_file = p_cand
            if not preset_file and a2_preset_dir:
                p_cand = os.path.join(a2_preset_dir, f"{stem_id}_ema_t07_harm.vstpreset")
                if os.path.exists(p_cand): preset_file = p_cand

            plugin_rel = None
            if preset_file and os.path.exists(preset_file):
                plugin_rel = f"plugins/{stem_id}.vstpreset"
                shutil.copy2(preset_file, os.path.join(tmp_build, plugin_rel))

            # -------------------------------------------------------------
            # Define Sequential Sequence of Clips for this Stem
            # -------------------------------------------------------------
            sequence_items = []

            # 1. Target Reference
            sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

            # 2. Arm A2 EMA
            for t in temps:
                fn_harm = f"{stem_id}_ema_{t}_harm.wav"
                fn_raw = f"{stem_id}_ema_{t}_raw.wav"
                p_harm = os.path.join(a2_audio_dir, fn_harm)
                p_raw = os.path.join(a2_audio_dir, fn_raw)
                rel_h, rel_r = None, None
                if os.path.exists(p_harm):
                    rel_h = f"audio/a2_{fn_harm}"
                    shutil.copy2(p_harm, os.path.join(tmp_build, rel_h))
                if os.path.exists(p_raw):
                    rel_r = f"audio/a2_{fn_raw}"
                    shutil.copy2(p_raw, os.path.join(tmp_build, rel_r))
                if rel_h:
                    sequence_items.append((f"A2 EMA {temp_labels.get(t, t)}", f"A2 EMA {temp_labels.get(t, t)} (Harm)", rel_h, False, rel_r))

            # 3. Target Reference Refresh
            sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

            # 4. Arm A2 Online
            for t in temps:
                fn_harm = f"{stem_id}_online_{t}_harm.wav"
                fn_raw = f"{stem_id}_online_{t}_raw.wav"
                p_harm = os.path.join(a2_audio_dir, fn_harm)
                p_raw = os.path.join(a2_audio_dir, fn_raw)
                rel_h, rel_r = None, None
                if os.path.exists(p_harm):
                    rel_h = f"audio/a2_{fn_harm}"
                    shutil.copy2(p_harm, os.path.join(tmp_build, rel_h))
                if os.path.exists(p_raw):
                    rel_r = f"audio/a2_{fn_raw}"
                    shutil.copy2(p_raw, os.path.join(tmp_build, rel_r))
                if rel_h:
                    sequence_items.append((f"A2 Online {temp_labels.get(t, t)}", f"A2 Online {temp_labels.get(t, t)} (Harm)", rel_h, False, rel_r))

            # 5. Target Reference Refresh
            sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

            # 6. Arm A3 EMA
            for t in temps:
                fn_harm = f"{stem_id}_ema_{t}_harm.wav"
                fn_raw = f"{stem_id}_ema_{t}_raw.wav"
                p_harm = os.path.join(a3_audio_dir, fn_harm)
                p_raw = os.path.join(a3_audio_dir, fn_raw)
                rel_h, rel_r = None, None
                if os.path.exists(p_harm):
                    rel_h = f"audio/a3_{fn_harm}"
                    shutil.copy2(p_harm, os.path.join(tmp_build, rel_h))
                if os.path.exists(p_raw):
                    rel_r = f"audio/a3_{fn_raw}"
                    shutil.copy2(p_raw, os.path.join(tmp_build, rel_r))
                if rel_h:
                    sequence_items.append((f"A3 EMA {temp_labels.get(t, t)}", f"A3 EMA {temp_labels.get(t, t)} (Harm)", rel_h, False, rel_r))

            # 7. Target Reference Refresh
            sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

            # 8. Arm A3 Online
            for t in temps:
                fn_harm = f"{stem_id}_online_{t}_harm.wav"
                fn_raw = f"{stem_id}_online_{t}_raw.wav"
                p_harm = os.path.join(a3_audio_dir, fn_harm)
                p_raw = os.path.join(a3_audio_dir, fn_raw)
                rel_h, rel_r = None, None
                if os.path.exists(p_harm):
                    rel_h = f"audio/a3_{fn_harm}"
                    shutil.copy2(p_harm, os.path.join(tmp_build, rel_h))
                if os.path.exists(p_raw):
                    rel_r = f"audio/a3_{fn_raw}"
                    shutil.copy2(p_raw, os.path.join(tmp_build, rel_r))
                if rel_h:
                    sequence_items.append((f"A3 Online {temp_labels.get(t, t)}", f"A3 Online {temp_labels.get(t, t)} (Harm)", rel_h, False, rel_r))

            # -------------------------------------------------------------
            # Build Group Track and Structure
            # -------------------------------------------------------------
            group_id = f"g_{id_counter}"
            id_counter += 1
            group_track = ET.SubElement(structure, "Track", contentType="tracks", loaded="true", id=group_id,
                                        name=f"{s_idx+1:02d} - {stem_name} ({bpm:.0f} BPM)", color="#00e5ff")
            g_chan_id = f"c_{id_counter}"
            id_counter += 1
            g_chan = ET.SubElement(group_track, "Channel", audioChannels="2", destination="id_mchan",
                                   role="master", solo="false", id=g_chan_id)
            ET.SubElement(g_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # Track 1: Target Audio Track (Reference)
            ref_track_id = f"t_ref_{id_counter}"
            id_counter += 1
            ref_track = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                                      id=ref_track_id, name="Target (Real Bass)", color="#b53bba")
            c_ref_id = f"c_{id_counter}"
            id_counter += 1
            c_ref = ET.SubElement(ref_track, "Channel", audioChannels="2", destination=g_chan_id,
                                  role="regular", solo="false", id=c_ref_id)
            ET.SubElement(c_ref, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # Track 2: Inversion Audio Track (Harmonic Snapped)
            harm_track_id = f"t_harm_{id_counter}"
            id_counter += 1
            harm_track = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                                       id=harm_track_id, name="Harmonic Snapped (Musical)", color="#00e676")
            c_harm_id = f"c_{id_counter}"
            id_counter += 1
            c_harm = ET.SubElement(harm_track, "Channel", audioChannels="2", destination=g_chan_id,
                                   role="regular", solo="false", id=c_harm_id)
            ET.SubElement(c_harm, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # Track 3: Raw Predictions Track (Muted by default)
            raw_track_id = f"t_raw_{id_counter}"
            id_counter += 1
            raw_track = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                                      id=raw_track_id, name="Raw Predictions (Muted)", color="#78909c")
            c_raw_id = f"c_{id_counter}"
            id_counter += 1
            c_raw = ET.SubElement(raw_track, "Channel", audioChannels="2", destination=g_chan_id,
                                  role="regular", solo="false", mute="true", id=c_raw_id)
            ET.SubElement(c_raw, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # Track 4: Transcribed MIDI
            midi_track_id = f"t_midi_{id_counter}"
            id_counter += 1
            midi_track = ET.SubElement(group_track, "Track", contentType="notes", loaded="true",
                                       id=midi_track_id, name="Transcribed MIDI", color="#ab47bc")
            c_midi_id = f"c_{id_counter}"
            id_counter += 1
            c_midi = ET.SubElement(midi_track, "Channel", audioChannels="2", destination=g_chan_id,
                                   role="regular", solo="false", id=c_midi_id)
            ET.SubElement(c_midi, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # Track 5: Live Surge XT Instrument Track (Muted by default)
            synth_track_id = None
            if plugin_rel:
                synth_track_id = f"t_synth_{id_counter}"
                id_counter += 1
                synth_track = ET.SubElement(group_track, "Track", contentType="notes", loaded="true",
                                            id=synth_track_id, name="Surge XT Live Patch (Muted)", color="#ff9100")
                s_chan_id = f"c_{id_counter}"
                id_counter += 1
                s_chan = ET.SubElement(synth_track, "Channel", audioChannels="2", destination=g_chan_id,
                                       role="regular", solo="false", mute="true", id=s_chan_id)
                devices = ET.SubElement(s_chan, "Devices")
                vst_dev = ET.SubElement(devices, "Vst3Plugin",
                                        deviceID="ABCDEF019182FAEB566D624153675854",
                                        deviceVendor="Surge Synth Team",
                                        deviceName="Surge XT",
                                        deviceRole="instrument",
                                        loaded="true",
                                        id=f"dev_{id_counter}",
                                        name="Surge XT")
                id_counter += 1
                ET.SubElement(vst_dev, "State", path=plugin_rel)
                ET.SubElement(s_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
                id_counter += 1

            # -------------------------------------------------------------
            # Build Arrangement Lanes & Sequential Clips
            # -------------------------------------------------------------
            # Group Lane
            group_lane = ET.SubElement(arr_lanes, "Lanes", track=group_id, id=f"lane_{id_counter}")
            id_counter += 1
            ET.SubElement(group_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # Reference Track Lane
            ref_lane = ET.SubElement(arr_lanes, "Lanes", track=ref_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            ref_clips_el = ET.SubElement(ref_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # Harm Track Lane
            harm_lane = ET.SubElement(arr_lanes, "Lanes", track=harm_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            harm_clips_el = ET.SubElement(harm_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # Raw Track Lane
            raw_lane = ET.SubElement(arr_lanes, "Lanes", track=raw_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            raw_clips_el = ET.SubElement(raw_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # MIDI Track Lane
            midi_lane = ET.SubElement(arr_lanes, "Lanes", track=midi_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            midi_clips_el = ET.SubElement(midi_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # Synth Track Lane
            synth_clips_el = None
            if synth_track_id:
                synth_lane = ET.SubElement(arr_lanes, "Lanes", track=synth_track_id, id=f"lane_{id_counter}")
                id_counter += 1
                synth_clips_el = ET.SubElement(synth_lane, "Clips", id=f"clips_{id_counter}")
                id_counter += 1

            stem_start_beat = current_beat

            for item_idx, (sec_title, clip_title, rel_path, is_ref, rel_raw_path) in enumerate(sequence_items):
                slot_beat = current_beat

                # Timeline Marker for this clip slot
                marker_label = f"{s_idx+1:02d}. {stem_name} - {sec_title}"
                ET.SubElement(markers_el, "Marker", time=f"{slot_beat:.3f}", name=marker_label)

                if is_ref:
                    # Place on Reference Track
                    clip_el = ET.SubElement(ref_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                            duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds",
                                            name=clip_title)
                    aud_el = ET.SubElement(clip_el, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                          sampleRate="44100", id=f"aud_{id_counter}")
                    id_counter += 1
                    ET.SubElement(aud_el, "File", path=rel_path)
                else:
                    # Place on Harmonic Snapped Track
                    clip_el = ET.SubElement(harm_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                            duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds",
                                            name=clip_title)
                    aud_el = ET.SubElement(clip_el, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                          sampleRate="44100", id=f"aud_{id_counter}")
                    id_counter += 1
                    ET.SubElement(aud_el, "File", path=rel_path)

                    # Also place on Raw Predictions Track if available
                    if rel_raw_path:
                        r_clip = ET.SubElement(raw_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                               duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds",
                                               name=clip_title.replace("(Harm)", "(Raw)"))
                        r_aud = ET.SubElement(r_clip, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                              sampleRate="44100", id=f"aud_{id_counter}")
                        id_counter += 1
                        ET.SubElement(r_aud, "File", path=rel_raw_path)

                # Place matching MIDI Clip in sync with this audio slot
                if note_events:
                    m_clip = ET.SubElement(midi_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                           duration=f"{phrase_dur_beats:.3f}", playStart="0.0",
                                           name=f"{stem_name} MIDI")
                    notes_el = ET.SubElement(m_clip, "Notes", id=f"notes_{id_counter}")
                    id_counter += 1
                    for n_time, n_dur, n_key, n_vel in note_events:
                        ET.SubElement(notes_el, "Note", time=f"{n_time:.3f}", duration=f"{n_dur:.3f}",
                                      channel="0", key=str(int(n_key)), vel=f"{n_vel:.3f}", rel=f"{n_vel:.3f}")

                    if synth_clips_el is not None:
                        # Copy MIDI clip to Surge live track
                        s_clip = ET.SubElement(synth_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                               duration=f"{phrase_dur_beats:.3f}", playStart="0.0",
                                               name=f"{stem_name} Surge Patch")
                        s_notes = ET.SubElement(s_clip, "Notes", id=f"notes_{id_counter}")
                        id_counter += 1
                        for n_time, n_dur, n_key, n_vel in note_events:
                            ET.SubElement(s_notes, "Note", time=f"{n_time:.3f}", duration=f"{n_dur:.3f}",
                                          channel="0", key=str(int(n_key)), vel=f"{n_vel:.3f}", rel=f"{n_vel:.3f}")

                # Advance to next sequential clip slot (flush, no overlap)
                current_beat += phrase_dur_beats

            # Spacing of 4 bars (16 beats) before next stem begins
            current_beat += 16.0

        # Master Track & Lane (at end of structure)
        master_track = ET.SubElement(structure, "Track", contentType="audio notes", loaded="true", id="id_master", name="Master")
        m_chan = ET.SubElement(master_track, "Channel", audioChannels="2", role="master", solo="false", id="id_mchan")
        ET.SubElement(m_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id="id_mvol", name="Volume")

        master_lane = ET.SubElement(arr_lanes, "Lanes", track="id_master", id=f"lane_{id_counter}")
        id_counter += 1
        ET.SubElement(master_lane, "Clips", id=f"clips_{id_counter}")
        id_counter += 1

        # Write XML files
        project_xml_path = os.path.join(tmp_build, "project.xml")
        ET.ElementTree(project).write(project_xml_path, encoding="utf-8", xml_declaration=True)

        meta_root = ET.Element("MetaData")
        ET.SubElement(meta_root, "Title").text = title
        ET.SubElement(meta_root, "Artist").text = "Antigravity AI Audio Lab"
        ET.SubElement(meta_root, "Comment").text = "Surge XT A2 (ASP) & A3 (ASP+Inharm) Sequential Evaluation Project"
        metadata_xml_path = os.path.join(tmp_build, "metadata.xml")
        ET.ElementTree(meta_root).write(metadata_xml_path, encoding="utf-8", xml_declaration=True)

        # Zip package as .dawproject
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        print(f"Compressing DAWproject package into {out_path}...")
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(project_xml_path, "project.xml")
            zf.write(metadata_xml_path, "metadata.xml")
            for root, _, files in os.walk(proj_audio_dir):
                for f in files:
                    p = os.path.join(root, f)
                    zf.write(p, os.path.relpath(p, tmp_build))
            for root, _, files in os.walk(proj_plugins_dir):
                for f in files:
                    p = os.path.join(root, f)
                    zf.write(p, os.path.relpath(p, tmp_build))

    print(f"Done! Successfully generated DAWproject: {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export Surge XT A2 & A3 Multi-Temp Evaluation to .dawproject format")
    parser.add_argument("--a2_audio", default="/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a2/eval_cpu_muscriptor_temps/audio")
    parser.add_argument("--a3_audio", default="/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a3/eval_cpu_muscriptor_temps/audio")
    parser.add_argument("--a2_presets", default="/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a2/eval_cpu_muscriptor_temps/vstpresets")
    parser.add_argument("--a3_presets", default="/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a3/eval_cpu_muscriptor_temps/vstpresets")
    parser.add_argument("--midi_dir", default="/run/media/kim/Mantu2/surge_200k_models/real_stems_eval/midi_muscriptor")
    parser.add_argument("--out", default="/home/kim/Documents/Surge_A2_A3_Sequential_Evaluation.dawproject")
    parser.add_argument("--title", default="Surge XT A2 & A3 Multi-Temp Evaluation")
    parser.add_argument("--stems", nargs="+", default=None)
    args = parser.parse_args()

    out_p = os.path.expanduser(args.out)
    build_multitemp_dawproject(
        args.a2_audio, args.a3_audio, args.midi_dir, out_p,
        title=args.title, stems_to_include=args.stems,
        a2_preset_dir=args.a2_presets, a3_preset_dir=args.a3_presets
    )


if __name__ == "__main__":
    main()
