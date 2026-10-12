#!/usr/bin/env python3
# Owner: ANTIGRAVITY (created 2026-10-13)
"""Export Any Combination of Synth-JEPA Inversion Arms (A0, A1, A2, A3, A3') into a native Bitwig DAWproject (.dawproject).

Features:
- Multi-Arm Support: Seamlessly imports eval clips from any combination of arms (A0, A1, A2, A3, A3').
- Sequential Timeline Arrangement: Per-stem Group Tracks with flush, non-overlapping clips.
- Target Audio Refreshes: Re-anchors the listener's ear with the target stem before each arm.
- Cue Markers: Direct Bitwig timeline navigation markers on every clip boundary (Alt+Right / Alt+Left).
- Muted Parallel Tracks: Instant A/B comparison between Musical Snapped and Raw Flow predictions.
- Transcribed Clean MIDI: Synchronized notes with Channel 9 percussion strictly excluded.
- Live Surge XT Instrument Tracks: Pre-loaded with inverted VST3 presets when available.
"""
import os
import sys
import shutil
import zipfile
import tempfile
import argparse
import xml.etree.ElementTree as ET
import soundfile as sf
import mido

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu


TEMP_LABELS = {
    "t015": "tau=0.15",
    "t03": "tau=0.30",
    "t05": "tau=0.50",
    "t07": "tau=0.70",
    "t085": "tau=0.85",
    "t10": "tau=1.00",
}


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
        # Ignore MIDI Channel 9 (drums/percussion)
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


def resolve_arm_directory(base_dir: str, arm_name: str):
    """Resolve an arm name (e.g. 'a0', 'a1', 'a2', 'a3', 'a3_prime') to an existing directory."""
    raw = arm_name.upper()
    if raw.startswith("ARM"):
        raw = raw[3:].lstrip("_")
    arm_num = raw[1:] if raw.startswith("A") else raw
    label = f"A{arm_num}"
    clean = f"a{arm_num.lower().rstrip('\'').rstrip('_prime')}"

    candidates = [
        (label, os.path.join(base_dir, f"real_bass_v3_{clean}_seeded")),
        (f"{label}'", os.path.join(base_dir, f"real_bass_v3_{clean}_prime")),
        (label, os.path.join(base_dir, f"real_bass_v3_{clean}")),
        (label, os.path.join(base_dir, arm_name)),
    ]
    for label, path in candidates:
        audio_dir = os.path.join(path, "eval_cpu_muscriptor_temps", "audio")
        if os.path.isdir(audio_dir):
            preset_dir = os.path.join(path, "eval_cpu_muscriptor_temps", "vstpresets")
            return label, audio_dir, preset_dir

    if os.path.isdir(arm_name):
        audio_dir = os.path.join(arm_name, "eval_cpu_muscriptor_temps", "audio")
        if not os.path.isdir(audio_dir):
            audio_dir = arm_name
        preset_dir = os.path.join(arm_name, "eval_cpu_muscriptor_temps", "vstpresets")
        return os.path.basename(arm_name.rstrip("/")), audio_dir, preset_dir

    return None, None, None


def build_all_arms_dawproject(
    arms_to_include: list,
    base_dir: str,
    midi_dir: str,
    out_path: str,
    title: str = "Surge XT Multi-Arm Inversion Evaluation",
    temps=("t03", "t05", "t07", "t085", "t10"),
    kinds=("ema", "online"),
    stems_to_include=None,
    refresh_target_each_arm: bool = True
):
    base_dir = resolve_mantu(base_dir)
    midi_dir = resolve_mantu(midi_dir)

    # 1. Resolve all arms
    resolved_arms = []
    for a in arms_to_include:
        label, aud_dir, pre_dir = resolve_arm_directory(base_dir, a)
        if aud_dir and os.path.isdir(aud_dir):
            resolved_arms.append({
                "key": a,
                "label": label,
                "audio_dir": aud_dir,
                "preset_dir": pre_dir
            })
        else:
            print(f"[WARNING] Could not find eval audio for arm '{a}' in {base_dir}, skipping.")

    if not resolved_arms:
        raise RuntimeError(f"No valid arm eval directories found for requested arms: {arms_to_include}")

    print("=" * 80)
    print(f"BUILDING DAWPROJECT: {title}")
    print(f"Arms included ({len(resolved_arms)}): {[a['label'] for a in resolved_arms]}")
    print(f"Temperatures: {temps}")
    print(f"Model kinds: {kinds}")
    print(f"Output: {out_path}")
    print("=" * 80)

    # 2. Filter stems
    stems = STEM_CATALOG
    if stems_to_include:
        stems = [s for s in stems if any(k in s["id"] for k in stems_to_include)]

    # Locate ground-truth reference audio
    available_stems = []
    for s in stems:
        sid = s["id"]
        ref_path = None
        for ra in resolved_arms:
            cand = os.path.join(ra["audio_dir"], f"{sid}_real.wav")
            if os.path.exists(cand):
                ref_path = cand
                break
        if not ref_path:
            cand = os.path.join(base_dir, "real_stems_eval", "audio", f"{sid}_real.wav")
            if os.path.exists(cand):
                ref_path = cand

        if ref_path and os.path.exists(ref_path):
            s_copy = dict(s)
            s_copy["real_path"] = ref_path
            available_stems.append(s_copy)

    if not available_stems:
        raise RuntimeError("No rendered stems found across the selected arms.")

    # 3. Build DAWproject XML Structure
    project = ET.Element("Project", version="1.0")
    ET.SubElement(project, "Application", name="Bitwig Studio", version="5.0")

    ref_bpm = available_stems[0].get("bpm", 140.0)
    transport = ET.SubElement(project, "Transport")
    tempo_el = ET.SubElement(transport, "Tempo", unit="bpm")
    tempo_el.text = f"{ref_bpm:.2f}"
    time_sig = ET.SubElement(transport, "TimeSignature")
    time_sig.text = "4/4"

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

            # Find best preset to attach to live track (prefer later arms e.g. A3, A2)
            preset_file = None
            for ra in reversed(resolved_arms):
                if ra["preset_dir"] and os.path.isdir(ra["preset_dir"]):
                    for t_pref in ("t07", "t05", "t085", "t03", "t10"):
                        p_cand = os.path.join(ra["preset_dir"], f"{stem_id}_ema_{t_pref}_harm.vstpreset")
                        if os.path.exists(p_cand):
                            preset_file = p_cand
                            break
                    if preset_file:
                        break

            plugin_rel = None
            if preset_file and os.path.exists(preset_file):
                plugin_rel = f"plugins/{stem_id}.vstpreset"
                shutil.copy2(preset_file, os.path.join(tmp_build, plugin_rel))

            # -------------------------------------------------------------
            # Assemble Sequential Clips for this Stem
            # -------------------------------------------------------------
            sequence_items = []

            # Initial Target Reference
            sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

            for a_idx, ra in enumerate(resolved_arms):
                arm_label = ra["label"]
                arm_key = ra["key"]
                arm_audio = ra["audio_dir"]

                # Optional reference refresh before each subsequent arm
                if refresh_target_each_arm and a_idx > 0:
                    sequence_items.append(("REF Target", "Target (Real Bass)", rel_real_audio, True, None))

                for kind in kinds:
                    for t in temps:
                        fn_harm = f"{stem_id}_{kind}_{t}_harm.wav"
                        fn_raw = f"{stem_id}_{kind}_{t}_raw.wav"
                        p_harm = os.path.join(arm_audio, fn_harm)
                        p_raw = os.path.join(arm_audio, fn_raw)
                        rel_h, rel_r = None, None
                        if os.path.exists(p_harm):
                            rel_h = f"audio/{arm_key}_{fn_harm}"
                            shutil.copy2(p_harm, os.path.join(tmp_build, rel_h))
                        if os.path.exists(p_raw):
                            rel_r = f"audio/{arm_key}_{fn_raw}"
                            shutil.copy2(p_raw, os.path.join(tmp_build, rel_r))

                        if rel_h:
                            t_display = TEMP_LABELS.get(t, t)
                            marker_title = f"{arm_label} {kind.upper()} {t_display}"
                            clip_title = f"{arm_label} {kind.upper()} {t_display} (Harm)"
                            sequence_items.append((marker_title, clip_title, rel_h, False, rel_r))

            # -------------------------------------------------------------
            # Build Group Track and Structure
            # -------------------------------------------------------------
            group_id = f"group_{s_idx+1}"
            track_name_prefix = f"[{s_idx+1:02d}] {stem_name}"

            group_track = ET.SubElement(structure, "Track", contentType="audio notes", loaded="true",
                                        id=group_id, name=track_name_prefix)
            g_chan = ET.SubElement(group_track, "Channel", audioChannels="2", role="regular", solo="false",
                                   id=f"chan_{group_id}")
            ET.SubElement(g_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0",
                          id=f"vol_{group_id}", name="Volume")

            # Group Lanes
            group_lanes = ET.SubElement(arr_lanes, "Lanes", track=group_id, id=f"glanes_{group_id}")

            # Sub-track 1: Harmonic Snapped Audio (Active)
            t1_id = f"trk_harm_{s_idx+1}"
            t1 = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                               id=t1_id, name="Harmonic Snapped Audio")
            t1_chan = ET.SubElement(t1, "Channel", audioChannels="2", role="regular", solo="false",
                                    id=f"chan_{t1_id}")
            ET.SubElement(t1_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0",
                          id=f"vol_{t1_id}", name="Volume")
            t1_lane = ET.SubElement(group_lanes, "Lanes", track=t1_id, id=f"lane_{t1_id}")
            t1_clips_el = ET.SubElement(t1_lane, "Clips", id=f"clips_{t1_id}")

            # Sub-track 2: Raw Flow Predictions (Muted)
            t2_id = f"trk_raw_{s_idx+1}"
            t2 = ET.SubElement(group_track, "Track", contentType="audio", loaded="true",
                               id=t2_id, name="Raw Flow Predictions (Muted A/B)")
            t2_chan = ET.SubElement(t2, "Channel", audioChannels="2", role="regular", solo="false",
                                    id=f"chan_{t2_id}")
            ET.SubElement(t2_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0",
                          id=f"vol_{t2_id}", name="Volume")
            ET.SubElement(t2_chan, "Mute", value="true", id=f"mute_{t2_id}", name="Mute")
            t2_lane = ET.SubElement(group_lanes, "Lanes", track=t2_id, id=f"lane_{t2_id}")
            t2_clips_el = ET.SubElement(t2_lane, "Clips", id=f"clips_{t2_id}")

            # Sub-track 3: Transcribed MIDI
            t3_id = f"trk_midi_{s_idx+1}"
            t3 = ET.SubElement(group_track, "Track", contentType="notes", loaded="true",
                               id=t3_id, name="Transcribed MIDI Notes")
            t3_chan = ET.SubElement(t3, "Channel", role="regular", solo="false", id=f"chan_{t3_id}")
            ET.SubElement(t3_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0",
                          id=f"vol_{t3_id}", name="Volume")
            t3_lane = ET.SubElement(group_lanes, "Lanes", track=t3_id, id=f"lane_{t3_id}")
            t3_clips_el = ET.SubElement(t3_lane, "Clips", id=f"clips_{t3_id}")

            # Sub-track 4: Live Surge XT Instrument Track (Muted by default)
            t4_id = f"trk_synth_{s_idx+1}"
            t4 = ET.SubElement(group_track, "Track", contentType="notes", loaded="true",
                               id=t4_id, name="Surge XT Live Patch")
            t4_chan = ET.SubElement(t4, "Channel", audioChannels="2", role="regular", solo="false",
                                    id=f"chan_{t4_id}")
            ET.SubElement(t4_chan, "Volume", max="2.0", min="0.0", unit="linear", value="0.85",
                          id=f"vol_{t4_id}", name="Volume")
            ET.SubElement(t4_chan, "Mute", value="true", id=f"mute_{t4_id}", name="Mute")

            # Attach VST3 Plugin Device if preset exists
            if plugin_rel:
                devices = ET.SubElement(t4, "Devices")
                dev_el = ET.SubElement(devices, "Device", deviceRole="instrument", loaded="true",
                                       id=f"dev_{id_counter}", name="Surge XT")
                id_counter += 1
                vst3_el = ET.SubElement(dev_el, "Vst3Plugin", pluginId="53757267652D78740000000000000000",
                                        pluginVersion="1.3.4", id=f"vst_{id_counter}", name="Surge XT")
                id_counter += 1
                ET.SubElement(vst3_el, "State", path=plugin_rel)

            t4_lane = ET.SubElement(group_lanes, "Lanes", track=t4_id, id=f"lane_{t4_id}")
            t4_clips_el = ET.SubElement(t4_lane, "Clips", id=f"clips_{t4_id}")

            # -------------------------------------------------------------
            # Write Clips and Cue Markers Sequentially along Timeline
            # -------------------------------------------------------------
            stem_start_beat = current_beat
            for marker_title, clip_title, rel_path, is_target, rel_raw_path in sequence_items:
                slot_beat = current_beat

                # Add Cue Marker on Bitwig Timeline
                marker_color = "#FF4444" if is_target else "#22BBEE"
                marker_name = f"[{s_idx+1:02d}] {marker_title}"
                ET.SubElement(markers_el, "Marker", time=f"{slot_beat:.3f}",
                              name=marker_name, color=marker_color)

                # Primary Audio Clip (Harmonic Snapped or Target)
                clip_el = ET.SubElement(t1_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                        duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds",
                                        name=clip_title)
                aud_el = ET.SubElement(clip_el, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                       sampleRate="44100", id=f"aud_{id_counter}")
                id_counter += 1
                ET.SubElement(aud_el, "File", path=rel_path)

                # Parallel Raw Clip (if available)
                if rel_raw_path:
                    r_clip = ET.SubElement(t2_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                           duration=f"{phrase_dur_beats:.3f}", contentTimeUnit="seconds",
                                           name=clip_title.replace("(Harm)", "(Raw)"))
                    r_aud = ET.SubElement(r_clip, "Audio", channels="2", duration=f"{dur_s:.4f}",
                                          sampleRate="44100", id=f"aud_{id_counter}")
                    id_counter += 1
                    ET.SubElement(r_aud, "File", path=rel_raw_path)

                # Parallel MIDI Note Clips
                if note_events:
                    m_clip = ET.SubElement(t3_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                           duration=f"{phrase_dur_beats:.3f}", playStart="0.0",
                                           name=f"{stem_name} MIDI")
                    notes_el = ET.SubElement(m_clip, "Notes", id=f"notes_{id_counter}")
                    id_counter += 1
                    for n_time, n_dur, n_key, n_vel in note_events:
                        ET.SubElement(notes_el, "Note", time=f"{n_time:.3f}", duration=f"{n_dur:.3f}",
                                      channel="0", key=str(int(n_key)), vel=f"{n_vel:.3f}", rel=f"{n_vel:.3f}")

                    # Duplicate onto Surge XT live track
                    s_clip = ET.SubElement(t4_clips_el, "Clip", time=f"{slot_beat:.3f}",
                                           duration=f"{phrase_dur_beats:.3f}", playStart="0.0",
                                           name=f"{stem_name} Surge Patch")
                    s_notes = ET.SubElement(s_clip, "Notes", id=f"notes_{id_counter}")
                    id_counter += 1
                    for n_time, n_dur, n_key, n_vel in note_events:
                        ET.SubElement(s_notes, "Note", time=f"{n_time:.3f}", duration=f"{n_dur:.3f}",
                                      channel="0", key=str(int(n_key)), vel=f"{n_vel:.3f}", rel=f"{n_vel:.3f}")

                # Advance to next sequential clip slot
                current_beat += phrase_dur_beats

            # 4 bars spacing between stems
            current_beat += 16.0
            print(f"[{s_idx+1:02d}/{len(available_stems)}] {stem_name}: {len(sequence_items)} clips arranged ({stem_start_beat:.0f}b -> {current_beat:.0f}b)")

        # Master Track
        master_track = ET.SubElement(structure, "Track", contentType="audio notes", loaded="true",
                                     id="id_master", name="Master")
        m_chan = ET.SubElement(master_track, "Channel", audioChannels="2", role="master", solo="false",
                               id="id_mchan")
        ET.SubElement(m_chan, "Volume", max="2.0", min="0.0", unit="linear", value="1.0",
                      id="id_mvol", name="Volume")
        master_lane = ET.SubElement(arr_lanes, "Lanes", track="id_master", id=f"lane_{id_counter}")
        id_counter += 1
        ET.SubElement(master_lane, "Clips", id=f"clips_{id_counter}")
        id_counter += 1

        # Write XMLs
        project_xml_path = os.path.join(tmp_build, "project.xml")
        ET.ElementTree(project).write(project_xml_path, encoding="utf-8", xml_declaration=True)

        meta_root = ET.Element("MetaData")
        ET.SubElement(meta_root, "Title").text = title
        ET.SubElement(meta_root, "Artist").text = "Antigravity AI Audio Lab"
        ET.SubElement(meta_root, "Comment").text = f"Synth-JEPA Multi-Arm Sequential Project ({', '.join([a['label'] for a in resolved_arms])})"
        metadata_xml_path = os.path.join(tmp_build, "metadata.xml")
        ET.ElementTree(meta_root).write(metadata_xml_path, encoding="utf-8", xml_declaration=True)

        # Package as .dawproject
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
    parser = argparse.ArgumentParser(description="Export Any Combination of Synth-JEPA Arms to .dawproject format")
    parser.add_argument("--arms", nargs="+", default=["a0", "a1", "a2", "a3"],
                        help="Arms to include (e.g. a0 a1 a2 a3)")
    parser.add_argument("--base_dir", default="/run/media/kim/Mantu2/surge_200k_models")
    parser.add_argument("--midi_dir", default="/run/media/kim/Mantu2/surge_200k_models/real_stems_eval/midi_muscriptor")
    parser.add_argument("--temps", nargs="+", default=["t03", "t05", "t07", "t085", "t10"])
    parser.add_argument("--kinds", nargs="+", default=["ema", "online"])
    parser.add_argument("--out", default="/home/kim/Documents/Surge_All_Arms_Evaluation.dawproject")
    parser.add_argument("--title", default="Surge XT All Arms Inversion Evaluation")
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--no_refresh", action="store_true", help="Do not refresh target reference between arms")
    args = parser.parse_args()

    out_p = os.path.expanduser(args.out)
    build_all_arms_dawproject(
        arms_to_include=args.arms,
        base_dir=args.base_dir,
        midi_dir=args.midi_dir,
        out_path=out_p,
        title=args.title,
        temps=args.temps,
        kinds=args.kinds,
        stems_to_include=args.stems,
        refresh_target_each_arm=not args.no_refresh
    )


if __name__ == "__main__":
    main()
