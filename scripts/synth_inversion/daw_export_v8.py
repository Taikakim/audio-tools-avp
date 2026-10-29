#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 DAW Export Bundle: Native Surge XT Presets (.fxp / .vstpreset) + MIDI CC Automation (.mid).

Pillar 4 of the V8 Architecture:
Enables producers to drop the inverted sound directly into Bitwig, Ableton, FL Studio,
or Reaper with full parametric control and 10 Hz automation lanes:
- CC 74: Filter 1 Cutoff
- CC 71: Filter 1 Resonance
- CC 1:  Waveshaper Drive (Mod Wheel)
- CC 11: FM Depth (Expression)

KUANG 2026-10-29: lanes arrive in v6 vector units (dynamic_dataset_v8) and are converted to Surge raw values
(lane_to_raw: drive raw = 0.5 + 0.5 * drive) for the plugin and for the CC curves, so a CC that is MIDI-learned
to the parameter reproduces the lane. Surge does not map these CCs by itself; the manifest says so.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import save_patch
import surge_spec_v5 as S5
import surge_spec_v6 as S6
from dynamic_dataset_v8 import LANE_PLUGIN_PARAMS, lane_to_raw


CC_MAPPING = {
    0: {"cc": 74, "name": "Filter 1 Cutoff", "lane": "cutoff"},
    1: {"cc": 71, "name": "Filter 1 Resonance", "lane": "resonance"},
    2: {"cc": 1,  "name": "Waveshaper Drive", "lane": "drive"},
    3: {"cc": 11, "name": "FM Depth", "lane": "fm_depth"},
}


def create_daw_automation_midi(
    dynamic_lanes: np.ndarray,
    notes: Optional[List[Tuple[int, int, float, float]]] = None,
    bpm: float = 140.0,
    duration_s: float = 4.0,
    ticks_per_beat: int = 480,
) -> MidiFile:
    """Creates a MIDI file containing notes and 10.76 Hz CC automation curves.

    Args:
        dynamic_lanes: [4, 43] float array of Surge raw values in [0.0, 1.0] for the 4 automation lanes.
        notes: list of (midi_note, velocity, start_time_s, duration_s).
               If None, defaults to 16th-note phrase across 4.0s.
        bpm: Project tempo.
        duration_s: Clip duration in seconds.
        ticks_per_beat: Standard MIDI PPQ (480).

    Returns:
        mido.MidiFile
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Tempo meta-message
    tempo = mido.bpm2tempo(bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(MetaMessage("track_name", name="Surge XT V8 Inversion", time=0))

    seconds_per_tick = (60.0 / bpm) / ticks_per_beat

    # Collect all timed events: (time_s, event_type, msg_kwargs)
    events = []

    # 1. Notes
    if notes is None:
        # Default 16-note pattern (1 note every 0.25s)
        notes = []
        for t in np.arange(0.0, duration_s, 0.25):
            notes.append((36, 100, float(t), 0.18))  # C2 bassline

    for note, vel, start_s, dur_s in notes:
        events.append((start_s, "note_on", {"note": note, "velocity": vel, "channel": 0}))
        events.append((start_s + dur_s, "note_off", {"note": note, "velocity": 0, "channel": 0}))

    # 2. Dynamic CC Automation at 10.76 Hz
    num_lanes, num_frames = dynamic_lanes.shape
    frame_dur = duration_s / num_frames

    for f_idx in range(num_frames):
        t_s = f_idx * frame_dur
        for lane_idx in range(min(num_lanes, 4)):
            cc_num = CC_MAPPING[lane_idx]["cc"]
            norm_val = float(np.clip(dynamic_lanes[lane_idx, f_idx], 0.0, 1.0))
            cc_val = int(round(norm_val * 127))
            events.append((t_s, "control_change", {"control": cc_num, "value": cc_val, "channel": 0}))

    # Sort events chronologically. For same timestamp, CC comes before note_on
    def sort_key(item):
        t_s, etype, kwargs = item
        type_priority = 0 if etype == "control_change" else (1 if etype == "note_on" else 2)
        return (t_s, type_priority)

    events.sort(key=sort_key)

    # Convert absolute times to delta ticks
    curr_tick = 0
    for t_s, etype, kwargs in events:
        event_tick = int(round(t_s / seconds_per_tick))
        delta_ticks = max(0, event_tick - curr_tick)
        track.append(Message(etype, time=delta_ticks, **kwargs))
        curr_tick = event_tick

    return mid


def export_v8_daw_bundle(
    plugin,
    static_params: Union[np.ndarray, torch.Tensor],
    dynamic_lanes: Union[np.ndarray, torch.Tensor],
    output_stem: Union[str, Path],
    notes: Optional[List[Tuple[int, int, float, float]]] = None,
    bpm: float = 140.0,
    copy_to_user_dir: bool = False,
) -> Dict[str, str]:
    """Exports full V8 DAW Bundle: .fxp preset + .vstpreset + .mid automation + manifest.

    Args:
        plugin: Pedalboard Surge XT plugin instance.
        static_params: 59-dim static parameter vector.
        dynamic_lanes: [4, 43] dynamic parameter trajectory in v6 vector units.
        output_stem: Base path for generated files (without extension).
        notes: Optional note list.
        bpm: Tempo for MIDI export.
        copy_to_user_dir: Whether to copy to ~/Documents/Surge XT/Patches/AI Inversions/.

    Returns:
        Dict mapping artifact types to their created filepaths.
    """
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(static_params):
        static_params = static_params.detach().cpu().float().numpy()
    if torch.is_tensor(dynamic_lanes):
        dynamic_lanes = dynamic_lanes.detach().cpu().float().numpy()

    if static_params.ndim > 1:
        static_params = static_params.squeeze(0)
    if dynamic_lanes.ndim > 2:
        dynamic_lanes = dynamic_lanes.squeeze(0)

    # 1. Decode static patch and apply to plugin
    patch = S6.vector_to_patch(static_params)
    S5.apply_patch(plugin, patch)

    # Initial frame sets the default starting position for automated lanes (Surge raw values)
    raw_lanes = np.stack([lane_to_raw(k, dynamic_lanes[k]) for k in range(len(LANE_PLUGIN_PARAMS))])
    for k, name in enumerate(LANE_PLUGIN_PARAMS):
        plugin.parameters[name].raw_value = float(raw_lanes[k, 0])

    # 2. Export Presets (.fxp, .vstpreset, .pedalboard_state, .json)
    extra_meta = {
        "v8_architecture": "Dynamic Decoupled Parameter Inverter",
        "dynamic_lanes": {
            CC_MAPPING[i]["lane"]: {
                "cc": CC_MAPPING[i]["cc"],
                "name": CC_MAPPING[i]["name"],
                "min": float(dynamic_lanes[i].min()),
                "max": float(dynamic_lanes[i].max()),
                "mean": float(dynamic_lanes[i].mean()),
            }
            for i in range(4)
        },
    }
    save_patch(plugin, patch, str(output_stem), extra=extra_meta, copy_to_user_dir=copy_to_user_dir)

    # 3. Export MIDI CC Automation (.mid)
    midi_file = create_daw_automation_midi(raw_lanes, notes=notes, bpm=bpm)
    midi_path = f"{output_stem}_automation.mid"
    midi_file.save(midi_path)

    # 4. Manifest
    manifest_path = f"{output_stem}_manifest.json"
    manifest = {
        "status": "ready",
        "fxp_file": f"{output_stem}.fxp",
        "vstpreset_file": f"{output_stem}.vstpreset",
        "midi_file": midi_path,
        "bpm": bpm,
        "lanes": CC_MAPPING,
        "instructions": (
            "1. In your DAW (Bitwig/Ableton/FL/Reaper), load Surge XT onto an instrument track.\n"
            "2. Load the exported preset (.fxp or .vstpreset).\n"
            "3. Surge does not map these CCs by itself. MIDI-learn each parameter first (right-click the "
            "control > MIDI Learn, then send the CC): CC 74 -> Filter 1 Cutoff, CC 71 -> Filter 1 Resonance, "
            "CC 1 -> Waveshaper Drive, CC 11 -> FM Depth. CC 1 is also Surge's mod wheel source.\n"
            "4. Import the automation MIDI track."
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    bundle = {
        "vstpreset": f"{output_stem}.vstpreset",
        "pedalboard_state": f"{output_stem}.pedalboard_state",
        "json": f"{output_stem}.json",
        "midi": midi_path,
        "manifest": manifest_path,
    }
    if os.path.exists(f"{output_stem}.fxp"):
        bundle["fxp"] = f"{output_stem}.fxp"

    return bundle


if __name__ == "__main__":
    print("V8 DAW Export Bundle Module Loaded Successfully.")
