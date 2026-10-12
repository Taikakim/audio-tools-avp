#!/usr/bin/env python3
"""Musical snapping and harmonic conditioning for Surge XT neural inversion.

Implements synthesis rules:
1. RM Ratios: Ring modulation (1x2, 2x3) locked strictly to integer octaves {-24, -12, 0, +12, +24} st.
2. Filter FM Ratios: Modulating oscillator locked to whole number (octave), perfect fifth (3:2 / +7 st),
   or perfect fourth (4:3 / +5 st) intervals.
3. Sync & FEG Dynamics:
   - If sync is an exact octave (+12 or +24 st), preserve static harmonic sync.
   - If sync is non-octave, mandate Filter EG modulation into sync with filter sustain = 0%.
   - For sync > 7 semitones, scale FEG decay length proportionally with sync strength.
4. Harmonic Osc Tuning: Unmodulated multi-oscillator setups locked to harmonic partial intervals.
"""
import copy
import numpy as np

# Octave mapping in Surge XT:
# raw 0.0 = -3, raw 0.1667 = -2, raw 0.3333 = -1, raw 0.5 = 0, raw 0.6667 = +1, raw 0.8333 = +2, raw 1.0 = +3
OCTAVE_MAP = {
    -3: 0.0,
    -2: 0.16667,
    -1: 0.33333,
     0: 0.5,
     1: 0.66667,
     2: 0.83333,
     3: 1.0,
}

def semitones_to_pitch_raw(semi: float) -> float:
    """Converts semitones offset in [-7, 7] to Surge raw pitch value [0, 1]."""
    return float(np.clip(0.5 + (semi / 14.0), 0.0, 1.0))

def pitch_raw_to_semitones(raw: float) -> float:
    """Converts Surge raw pitch value [0, 1] to semitones offset in [-7, 7]."""
    return float((raw - 0.5) * 14.0)

def raw_to_octave_int(raw: float) -> int:
    """Converts Surge raw octave value [0, 1] to integer octave [-3, 3]."""
    return int(round((raw - 0.5) * 6.0))

def get_osc_total_semitones(patch: dict, osc_idx: int = 2) -> float:
    """Calculates total relative semitones of an oscillator from octave + fine pitch."""
    oct_raw = float(patch.get(f"osc_{osc_idx}_octave", 0.5))
    pitch_raw = float(patch.get(f"osc_{osc_idx}_pitch_raw", 0.5))
    oct_val = raw_to_octave_int(oct_raw)
    fine_semi = pitch_raw_to_semitones(pitch_raw)
    return oct_val * 12.0 + fine_semi

def set_osc_total_semitones(patch: dict, total_semi: float, osc_idx: int = 2) -> None:
    """Sets an oscillator to match the target semitone offset using octave and fine pitch."""
    if -7.0 <= total_semi <= 7.0:
        oct_int = 0
        rem_semi = total_semi
    else:
        oct_int = int(np.clip(round(total_semi / 12.0), -3, 3))
        rem_semi = total_semi - (oct_int * 12.0)
        if rem_semi > 7.0 and oct_int < 3:
            oct_int += 1
            rem_semi -= 12.0
        elif rem_semi < -7.0 and oct_int > -3:
            oct_int -= 1
            rem_semi += 12.0

    patch[f"osc_{osc_idx}_octave"] = OCTAVE_MAP.get(oct_int, 0.5)
    patch[f"osc_{osc_idx}_pitch_raw"] = semitones_to_pitch_raw(rem_semi)

# Musical interval tables (semitones)
OCTAVE_INTERVALS = [-24.0, -12.0, 0.0, 12.0, 24.0]

FM_MUSICAL_INTERVALS = [
    -12.0,       # 1:2 sub-octave
    -7.0,        # 2:3 fourth below (or fifth down)
    -5.0,        # 3:4 fifth below (or fourth down)
     0.0,        # 1:1 unison
     5.0,        # 4:3 perfect fourth
     7.0,        # 3:2 perfect fifth
    12.0,        # 2:1 octave
    17.0,        # 8:3 fourth above octave
    19.0,        # 3:1 fifth above octave
    24.0,        # 4:1 two octaves
]

GENERAL_HARMONIC_INTERVALS = [
    -12.0, 0.0, 5.0, 7.0, 12.0, 19.0, 24.0
]


def find_closest_interval(target_semi: float, allowed_intervals: list) -> float:
    """Returns the interval from allowed_intervals closest to target_semi."""
    return min(allowed_intervals, key=lambda x: abs(x - target_semi))


def apply_musical_snapping(patch: dict) -> tuple:
    """Applies musical snapping to a patch dictionary according to Kim's specification.
    
    Returns:
        snapped_patch (dict): Updated patch with musical harmonic rules applied.
        report (dict): Metadata explaining the adjustments made.
    """
    p = copy.deepcopy(patch)
    report = {}

    # 1. Osc 1 is root / fundamental: lock to 0 semitones
    p["osc_1_octave"] = 0.5

    # Check active modulations
    ring_active = (float(p.get("ring_1x2_volume", 0.0)) > 0.05 or
                   float(p.get("ring_2x3_volume", 0.0)) > 0.05)
    fm_active = float(p.get("fm_depth", 0.0)) > 0.05
    raw_sync = float(p.get("sync", 0.0))
    sync_active = raw_sync > 0.02
    sync_semitones = raw_sync * 60.0

    # 2. Oscillator Tuning based on modulation mode
    for osc_idx in [2, 3]:
        osc_vol = float(p.get(f"osc_{osc_idx}_volume", 0.5))
        if osc_vol < 0.02 and not ring_active and not fm_active:
            continue

        curr_semi = get_osc_total_semitones(p, osc_idx=osc_idx)

        if ring_active:
            target_semi = find_closest_interval(curr_semi, OCTAVE_INTERVALS)
            set_osc_total_semitones(p, target_semi, osc_idx=osc_idx)
            report[f"osc_{osc_idx}_mode"] = f"RM_octave ({target_semi:+.0f} st)"
        elif fm_active:
            target_semi = find_closest_interval(curr_semi, FM_MUSICAL_INTERVALS)
            set_osc_total_semitones(p, target_semi, osc_idx=osc_idx)
            report[f"osc_{osc_idx}_mode"] = f"FM_ratio ({target_semi:+.0f} st)"
        else:
            target_semi = find_closest_interval(curr_semi, GENERAL_HARMONIC_INTERVALS)
            set_osc_total_semitones(p, target_semi, osc_idx=osc_idx)
            report[f"osc_{osc_idx}_mode"] = f"Harmonic ({target_semi:+.0f} st)"

    # 3. Hard Sync & FEG Coupling
    if sync_active:
        near_oct12 = abs(sync_semitones - 12.0) <= 2.0
        near_oct24 = abs(sync_semitones - 24.0) <= 2.5

        if near_oct12:
            p["sync"] = 12.0 / 60.0  # Lock exact +12 st
            report["sync_mode"] = "static_octave_12st"
        elif near_oct24:
            p["sync"] = 24.0 / 60.0  # Lock exact +24 st
            report["sync_mode"] = "static_octave_24st"
        else:
            report["sync_mode"] = f"feg_modulated_sync ({sync_semitones:.1f} st)"
            p["feg_sustain"] = 0.0  # Zero sustain prevents continuous overpowering drone
            p["feg_to_sync_depth"] = float(np.clip(raw_sync * 1.5, 0.1, 1.0))
            p["sync_static_floor"] = 0.0

            if sync_semitones > 7.0:
                current_decay = float(p.get("feg_decay", 0.5))
                decay_boost = ((sync_semitones - 7.0) / 40.0) * 0.30
                p["feg_decay"] = float(np.clip(current_decay + decay_boost, 0.1, 0.95))
                report["feg_decay_scaled"] = f"{current_decay:.2f} -> {p['feg_decay']:.2f}"

    return p, report
