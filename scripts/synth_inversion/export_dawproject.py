#!/usr/bin/env python3
"""Export Surge XT neural inversions into a native DAWproject (.dawproject) evaluation project.

Creates an open-standard .dawproject (supported by Bitwig Studio, Studio One, Reaper via converter, etc.)
allowing instant A/B/C diagnosis of inverted patches against real stem targets and offline renders:
  1. Live Surge XT Instrument Track: Loaded with the inverted preset (.vstpreset / .fxp) + transcribed MIDI notes.
  2. Isolated Real Bass Audio Track: Ground truth stem slice.
  3. Offline Rendered Playback Audio Track: Multi-stem script render.
  4. Cue Markers on the timeline with stem titles and BPM.

Usage:
  python export_dawproject.py --eval_dir /path/to/eval_root --out /path/to/project.dawproject
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import mido
import soundfile as sf


def parse_midi_notes_in_beats(midi_path: str, max_dur_s: float, bpm: float):
    """Parse a MIDI file and extract note events in musical beats relative to clip start."""
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


def build_dawproject(eval_dir: str, out_path: str, title: str = "Surge XT Stem Inversions Eval",
                     include_offline_render: bool = True):
    """Build a complete .dawproject zip container with XML, audio, MIDI, and VST3 states."""
    audio_dir = os.path.join(eval_dir, "audio")
    midi_dir = os.path.join(eval_dir, "midi")
    preset_dir = os.path.join(eval_dir, "vstpresets")
    summary_path = os.path.join(eval_dir, "real_stems_summary.json")

    # Discover stems
    records = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            records = json.load(f)
    else:
        # Fallback to scanning audio dir
        real_files = sorted(glob.glob(os.path.join(audio_dir, "*_real.wav")))
        for rf in real_files:
            sid = os.path.basename(rf).replace("_real.wav", "")
            records.append({"id": sid, "name": sid.replace("_", " ").title(), "bpm": 140.0})

    if not records:
        raise RuntimeError(f"No stem records found in {eval_dir}")

    # Create root Project XML
    project = ET.Element("Project", version="1.0")
    ET.SubElement(project, "Application", name="Bitwig Studio", version="5.0")

    ref_bpm = records[0].get("bpm", 140.0)
    transport = ET.SubElement(project, "Transport")
    ET.SubElement(transport, "Tempo", max="666.0", min="20.0", unit="bpm", value=f"{ref_bpm:.2f}", id="id_tempo", name="Tempo")
    ET.SubElement(transport, "TimeSignature", denominator="4", numerator="4", id="id_ts")

    structure = ET.SubElement(project, "Structure")

    arrangement = ET.SubElement(project, "Arrangement", id="id_arr")
    arr_lanes = ET.SubElement(arrangement, "Lanes", timeUnit="beats", id="id_lanes")
    markers_el = ET.SubElement(arrangement, "Markers", timeUnit="beats", id="id_markers")

    # Temp workspace for gathering files to zip
    with tempfile.TemporaryDirectory() as tmp_build:
        proj_audio_dir = os.path.join(tmp_build, "audio")
        proj_plugins_dir = os.path.join(tmp_build, "plugins")
        os.makedirs(proj_audio_dir, exist_ok=True)
        os.makedirs(proj_plugins_dir, exist_ok=True)

        current_beat = 0.0
        id_counter = 100

        for idx, rec in enumerate(records):
            stem_id = rec["id"]
            stem_name = rec.get("name", stem_id)
            bpm = rec.get("bpm", 140.0)

            real_wav = os.path.join(audio_dir, f"{stem_id}_real.wav")
            playback_wav = os.path.join(audio_dir, f"{stem_id}_midi_playback.wav")
            mid_file = os.path.join(midi_dir, f"{stem_id}_transcribed.mid")
            if not os.path.exists(mid_file):
                mid_file = os.path.join(midi_dir, f"{stem_id}_muscriptor.mid")
            preset_vst = os.path.join(preset_dir, f"Inverted_{stem_id}.vstpreset")

            if not os.path.exists(real_wav):
                continue

            info = sf.info(real_wav)
            dur_s = info.duration
            phrase_dur_beats = round((dur_s * (bpm / 60.0)) / 4.0) * 4.0
            if phrase_dur_beats < 4.0:
                phrase_dur_beats = 4.0

            # Copy media files into project
            shutil.copy2(real_wav, os.path.join(proj_audio_dir, f"{stem_id}_real.wav"))
            if os.path.exists(playback_wav):
                shutil.copy2(playback_wav, os.path.join(proj_audio_dir, f"{stem_id}_midi_playback.wav"))

            plugin_state_rel = None
            if os.path.exists(preset_vst):
                shutil.copy2(preset_vst, os.path.join(proj_plugins_dir, f"{stem_id}.vstpreset"))
                plugin_state_rel = f"plugins/{stem_id}.vstpreset"

            # Timeline Marker for this stem (Marker extends nameable, no id attribute)
            ET.SubElement(markers_el, "Marker", time=f"{current_beat:.3f}",
                          name=f"{idx+1:02d}. {stem_name} ({bpm:.1f} BPM)")

            # Group Track for this stem (must have contentType="tracks" for Bitwig to parse child tracks)
            group_id = f"g_{id_counter}"
            id_counter += 1
            group_track = ET.SubElement(structure, "Track", contentType="tracks", loaded="true", id=group_id,
                                        name=f"{idx+1:02d} - {stem_name} ({bpm:.0f} BPM)", color="#00e5ff")
            g_chan_id = f"c_{id_counter}"
            id_counter += 1
            g_chan = ET.SubElement(group_track, "Channel", audioChannels="2", destination="id_mchan",
                                   role="master", solo="false", id=g_chan_id)
            ET.SubElement(g_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # 1. Live Surge XT Synth Track
            synth_track_id = f"t_synth_{id_counter}"
            id_counter += 1
            synth_track = ET.SubElement(group_track, "Track", contentType="notes", loaded="true",
                                        id=synth_track_id, name="Surge XT (Neural Preset)", color="#00e5ff")
            s_chan_id = f"c_{id_counter}"
            id_counter += 1
            s_chan = ET.SubElement(synth_track, "Channel", audioChannels="2", destination=g_chan_id,
                                   role="regular", solo="false", id=s_chan_id)

            # In Channel sequence, Devices MUST precede Volume
            if plugin_state_rel:
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
                ET.SubElement(vst_dev, "State", path=plugin_state_rel)

            ET.SubElement(s_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # 2. Real Stem Target Track (Audio)
            real_track_id = f"t_real_{id_counter}"
            id_counter += 1
            real_track = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                                       id=real_track_id, name="Real Bass Target", color="#b53bba")
            r_chan_id = f"c_{id_counter}"
            id_counter += 1
            r_chan = ET.SubElement(real_track, "Channel", audioChannels="2", destination=g_chan_id,
                                   role="regular", solo="false", id=r_chan_id)
            ET.SubElement(r_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
            id_counter += 1

            # 3. Offline Rendered Playback Track (Audio)
            play_track_id = None
            if include_offline_render and os.path.exists(playback_wav):
                play_track_id = f"t_play_{id_counter}"
                id_counter += 1
                play_track = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                                           id=play_track_id, name="Offline Neural Playback", color="#ffab00")
                p_chan_id = f"c_{id_counter}"
                id_counter += 1
                p_chan = ET.SubElement(play_track, "Channel", audioChannels="2", destination=g_chan_id,
                                       role="regular", solo="false", id=p_chan_id)
                ET.SubElement(p_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id=f"v_{id_counter}", name="Volume")
                id_counter += 1

            # Arrangement Lanes (Group Track, Synth Track, Real Audio Track, Playback Audio Track)
            group_lane = ET.SubElement(arr_lanes, "Lanes", track=group_id, id=f"lane_{id_counter}")
            id_counter += 1
            ET.SubElement(group_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1

            # Synth Lane (Notes)
            synth_lane = ET.SubElement(arr_lanes, "Lanes", track=synth_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            synth_clips = ET.SubElement(synth_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1
            synth_clip = ET.SubElement(synth_clips, "Clip", time=f"{current_beat:.3f}",
                                       duration=f"{phrase_dur_beats:.3f}", playStart="0.0")
            notes_el = ET.SubElement(synth_clip, "Notes", id=f"notes_{id_counter}")
            id_counter += 1

            # Populate MIDI notes
            note_events = parse_midi_notes_in_beats(mid_file, dur_s, bpm)
            for n_time, n_dur, n_key, n_vel in note_events:
                ET.SubElement(notes_el, "Note", time=f"{n_time:.3f}", duration=f"{n_dur:.3f}",
                              channel="0", key=str(int(n_key)), vel=f"{n_vel:.3f}", rel=f"{n_vel:.3f}")

            # Real Audio Lane
            real_lane = ET.SubElement(arr_lanes, "Lanes", track=real_track_id, id=f"lane_{id_counter}")
            id_counter += 1
            real_clips = ET.SubElement(real_lane, "Clips", id=f"clips_{id_counter}")
            id_counter += 1
            real_clip = ET.SubElement(real_clips, "Clip", time=f"{current_beat:.3f}",
                                      duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds")
            real_aud = ET.SubElement(real_clip, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                     sampleRate=str(info.samplerate), id=f"aud_{id_counter}")
            id_counter += 1
            ET.SubElement(real_aud, "File", path=f"audio/{stem_id}_real.wav")

            # Playback Audio Lane
            if play_track_id:
                play_lane = ET.SubElement(arr_lanes, "Lanes", track=play_track_id, id=f"lane_{id_counter}")
                id_counter += 1
                play_clips = ET.SubElement(play_lane, "Clips", id=f"clips_{id_counter}")
                id_counter += 1
                play_clip = ET.SubElement(play_clips, "Clip", time=f"{current_beat:.3f}",
                                          duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds")
                play_aud = ET.SubElement(play_clip, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                         sampleRate=str(info.samplerate), id=f"aud_{id_counter}")
                id_counter += 1
                ET.SubElement(play_aud, "File", path=f"audio/{stem_id}_midi_playback.wav")

            # Advance current beat: phrase + 4 beats (1 bar) spacing
            current_beat += phrase_dur_beats + 4.0

        # Master Track in Structure (placed at end)
        master_track = ET.SubElement(structure, "Track", contentType="audio notes", loaded="true", id="id_master", name="Master")
        m_chan = ET.SubElement(master_track, "Channel", audioChannels="2", role="master", solo="false", id="id_mchan")
        ET.SubElement(m_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0", id="id_mvol", name="Volume")

        # Master Lane in Arrangement/Lanes (placed at end)
        master_lane = ET.SubElement(arr_lanes, "Lanes", track="id_master", id=f"lane_{id_counter}")
        id_counter += 1
        ET.SubElement(master_lane, "Clips", id=f"clips_{id_counter}")
        id_counter += 1

        # Save project.xml
        project_xml_path = os.path.join(tmp_build, "project.xml")
        ET.ElementTree(project).write(project_xml_path, encoding="utf-8", xml_declaration=True)

        # Save metadata.xml
        meta_root = ET.Element("MetaData")
        ET.SubElement(meta_root, "Title").text = title
        ET.SubElement(meta_root, "Artist").text = "Antigravity AI Audio Lab"
        ET.SubElement(meta_root, "Comment").text = "Surge XT neural synth inversion benchmark project"
        metadata_xml_path = os.path.join(tmp_build, "metadata.xml")
        ET.ElementTree(meta_root).write(metadata_xml_path, encoding="utf-8", xml_declaration=True)

        # Zip package as .dawproject
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
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

    print(f"Successfully generated DAWproject evaluation project at: {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Surge XT neural inversions to .dawproject format")
    parser.add_argument("--eval_dir", required=True, help="Directory containing audio, midi, vstpresets")
    parser.add_argument("--out", default="~/Documents/Surge_Inversions_Evaluation.dawproject", help="Output .dawproject path")
    parser.add_argument("--title", default="Surge XT Stem Inversions Evaluation", help="Project title")
    args = parser.parse_args()

    out = os.path.expanduser(args.out)
    build_dawproject(args.eval_dir, out, title=args.title)
