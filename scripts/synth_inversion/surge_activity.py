"""Activity masking and identifiability rules for Surge XT synthesizer parameters (v4).

Per Synth Inversion v4 Spec (Section 3):
A parameter that cannot be heard must not be a training target: the flow model would
learn to guess the prior for it, wasting capacity and injecting loss noise.

This module provides:
1. Declarative ACTIVE_IF rules mapping each parameter to an activity predicate lambda(patch).
2. canonicalize(patch): Clamps inactive parameters to fixed resting/neutral values.
3. get_activity_mask(patch, param_names): Returns boolean mask [P] indicating active parameters.
"""
from typing import Dict, Any, List, Callable, Set
import numpy as np

# Oscillator types supporting specific parameters in Surge XT
# Type 0: Classic, 1: Sine, 2: Wavetable, 7: Window, 8: Modern, 10: Twist
TYPES_WITH_SHAPE: Set[str] = {"classic", "wavetable", "window", "modern", "twist", 0, 2, 7, 8, 10}
TYPES_WITH_WIDTH: Set[str] = {"classic", "wavetable", "window", "modern", "twist", 0, 2, 7, 8, 10}
TYPES_WITH_SUB: Set[str] = {"classic", 0}
TYPES_WITH_SYNC: Set[str] = {"classic", "modern", "twist", 0, 8, 10}

def _to_type_str(val: Any) -> Any:
    if isinstance(val, str):
        return val.lower()
    return val

def is_osc_active(p: Dict[str, Any], n: int) -> bool:
    """Returns True if oscillator n is active (not muted, or feeds active ring modulation)."""
    # Direct path
    direct_on = not bool(p.get(f"osc{n}_mute", False if n == 1 else True))
    if direct_on:
        return True
    # Pre-mixer tap feeds ring mod 1x2 or 2x3 even if direct mixer mute is on
    if n in (1, 2) and not bool(p.get("ringmod_12_mute", True)):
        return True
    if n in (2, 3) and not bool(p.get("ringmod_23_mute", True)):
        return True
    return False

def osc_has_pw(p: Dict[str, Any], n: int) -> bool:
    """Returns True if oscillator n is active and its type supports pulse width."""
    if not is_osc_active(p, n):
        return False
    t = _to_type_str(p.get(f"osc{n}_type", 0))
    return t in TYPES_WITH_WIDTH

def is_dual_filter(p: Dict[str, Any]) -> bool:
    """Returns True if the patch routes audio through both filters or uses filter 2."""
    cfg = str(p.get("filter_config", "serial_1")).lower()
    # Serial configurations always pass through Filter 2 if Filter 2 is on
    if "serial" in cfg:
        f2_type = _to_type_str(p.get("filter_2_type", "off"))
        return f2_type not in ("off", 0, 0.0)
    # Dual/Wide configurations use Filter 2 if balance is not 100% Filter 1 (-1.0)
    bal = float(p.get("filter_balance", 0.0))
    f2_type = _to_type_str(p.get("filter_2_type", "off"))
    return (bal > -0.99) and (f2_type not in ("off", 0, 0.0))

def is_filter_active(p: Dict[str, Any], n: int) -> bool:
    """Returns True if Filter n is in the active audio signal path."""
    t = _to_type_str(p.get(f"filter_{n}_type", "off" if n == 2 else "lp_24db"))
    if t in ("off", 0, 0.0):
        return False
    if n == 1:
        # Filter 1 is active unless balance is 100% Filter 2 (+1.0) in Dual mode
        cfg = str(p.get("filter_config", "serial_1")).lower()
        if "serial" in cfg:
            return True
        bal = float(p.get("filter_balance", 0.0))
        return bal < 0.99
    else:
        return is_dual_filter(p)

def is_feg_active(p: Dict[str, Any]) -> bool:
    """Returns True if the Filter Envelope modulates any audible target."""
    # FEG -> Filter 1 cutoff
    if is_filter_active(p, 1) and abs(float(p.get("filter_1_feg_amount", 0.0))) > 0.001:
        return True
    # FEG -> Filter 2 cutoff
    if is_filter_active(p, 2) and abs(float(p.get("filter_2_feg_amount", 0.0))) > 0.001:
        return True
    # FEG -> Oscillator Pulse Width
    if abs(float(p.get("feg_to_pw", 0.0))) > 0.001 and any(osc_has_pw(p, n) for n in (1, 2, 3)):
        return True
    return False

# Declarative table mapping parameter patterns to activity predicates
ACTIVE_IF_RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    # Pitch & Global
    "midi_note": lambda p: True,
    "osc_drift": lambda p: any(is_osc_active(p, n) for n in (1, 2, 3)),

    # Oscillator 1
    "osc1_mute": lambda p: True,
    "osc1_type": lambda p: is_osc_active(p, 1),
    "osc1_volume": lambda p: not bool(p.get("osc1_mute", False)),
    "osc1_octave": lambda p: is_osc_active(p, 1),
    "osc1_pitch": lambda p: is_osc_active(p, 1),
    "osc1_shape": lambda p: is_osc_active(p, 1) and (_to_type_str(p.get("osc1_type", 0)) in TYPES_WITH_SHAPE),
    "osc1_width": lambda p: is_osc_active(p, 1) and (_to_type_str(p.get("osc1_type", 0)) in TYPES_WITH_WIDTH),
    "osc1_sub_mix": lambda p: is_osc_active(p, 1) and (_to_type_str(p.get("osc1_type", 0)) in TYPES_WITH_SUB),
    "osc1_sync": lambda p: is_osc_active(p, 1) and (_to_type_str(p.get("osc1_type", 0)) in TYPES_WITH_SYNC),
    "osc1_unison_voices": lambda p: is_osc_active(p, 1),
    "osc1_unison_detune": lambda p: is_osc_active(p, 1) and int(p.get("osc1_unison_voices", 1)) > 1,
    "osc1_retrigger": lambda p: is_osc_active(p, 1),
    "osc1_route": lambda p: (not bool(p.get("osc1_mute", False))) and is_dual_filter(p),

    # Oscillator 2
    "osc2_mute": lambda p: True,
    "osc2_type": lambda p: is_osc_active(p, 2),
    "osc2_volume": lambda p: not bool(p.get("osc2_mute", True)),
    "osc2_octave": lambda p: is_osc_active(p, 2),
    "osc2_pitch": lambda p: is_osc_active(p, 2),
    "osc2_shape": lambda p: is_osc_active(p, 2) and (_to_type_str(p.get("osc2_type", 0)) in TYPES_WITH_SHAPE),
    "osc2_width": lambda p: is_osc_active(p, 2) and (_to_type_str(p.get("osc2_type", 0)) in TYPES_WITH_WIDTH),
    "osc2_sub_mix": lambda p: is_osc_active(p, 2) and (_to_type_str(p.get("osc2_type", 0)) in TYPES_WITH_SUB),
    "osc2_sync": lambda p: is_osc_active(p, 2) and (_to_type_str(p.get("osc2_type", 0)) in TYPES_WITH_SYNC),
    "osc2_unison_voices": lambda p: is_osc_active(p, 2),
    "osc2_unison_detune": lambda p: is_osc_active(p, 2) and int(p.get("osc2_unison_voices", 1)) > 1,
    "osc2_retrigger": lambda p: is_osc_active(p, 2),
    "osc2_route": lambda p: (not bool(p.get("osc2_mute", True))) and is_dual_filter(p),

    # Oscillator 3
    "osc3_mute": lambda p: True,
    "osc3_type": lambda p: is_osc_active(p, 3),
    "osc3_volume": lambda p: not bool(p.get("osc3_mute", True)),
    "osc3_octave": lambda p: is_osc_active(p, 3),
    "osc3_pitch": lambda p: is_osc_active(p, 3),
    "osc3_shape": lambda p: is_osc_active(p, 3) and (_to_type_str(p.get("osc3_type", 0)) in TYPES_WITH_SHAPE),
    "osc3_width": lambda p: is_osc_active(p, 3) and (_to_type_str(p.get("osc3_type", 0)) in TYPES_WITH_WIDTH),
    "osc3_sub_mix": lambda p: is_osc_active(p, 3) and (_to_type_str(p.get("osc3_type", 0)) in TYPES_WITH_SUB),
    "osc3_sync": lambda p: is_osc_active(p, 3) and (_to_type_str(p.get("osc3_type", 0)) in TYPES_WITH_SYNC),
    "osc3_unison_voices": lambda p: is_osc_active(p, 3),
    "osc3_unison_detune": lambda p: is_osc_active(p, 3) and int(p.get("osc3_unison_voices", 1)) > 1,
    "osc3_retrigger": lambda p: is_osc_active(p, 3),
    "osc3_route": lambda p: (not bool(p.get("osc3_mute", True))) and is_dual_filter(p),

    # Mixer & Ring Modulation
    "ringmod_12_mute": lambda p: True,
    "ringmod_12_volume": lambda p: not bool(p.get("ringmod_12_mute", True)),
    "ringmod_23_mute": lambda p: True,
    "ringmod_23_volume": lambda p: not bool(p.get("ringmod_23_mute", True)),

    # Filter Global
    "filter_config": lambda p: is_filter_active(p, 2),
    "filter_balance": lambda p: is_filter_active(p, 2),
    "feedback": lambda p: is_filter_active(p, 1) or is_filter_active(p, 2),

    # Filter 1
    "filter_1_type": lambda p: True,
    "filter_1_cutoff": lambda p: is_filter_active(p, 1),
    "filter_1_resonance": lambda p: is_filter_active(p, 1),
    "filter_1_keytrack": lambda p: is_filter_active(p, 1),
    "filter_1_feg_amount": lambda p: is_filter_active(p, 1),

    # Filter 2
    "filter_2_type": lambda p: is_dual_filter(p),
    "filter_2_cutoff": lambda p: is_filter_active(p, 2),
    "filter_2_resonance": lambda p: is_filter_active(p, 2),
    "filter_2_keytrack": lambda p: is_filter_active(p, 2),
    "filter_2_feg_amount": lambda p: is_filter_active(p, 2),

    # Waveshaper
    "waveshaper_type": lambda p: True,
    "waveshaper_drive": lambda p: _to_type_str(p.get("waveshaper_type", "off")) not in ("off", 0, 0.0),

    # Amplitude Envelope (always active for sound generation)
    "aeg_attack": lambda p: True,
    "aeg_decay": lambda p: True,
    "aeg_sustain": lambda p: True,
    "aeg_release": lambda p: True,
    "aeg_mode": lambda p: True,

    # Filter Envelope
    "feg_attack": lambda p: is_feg_active(p),
    "feg_decay": lambda p: is_feg_active(p),
    "feg_sustain": lambda p: is_feg_active(p),
    "feg_release": lambda p: is_feg_active(p),
    "feg_mode": lambda p: is_feg_active(p),

    # Modulation
    "feg_to_pw": lambda p: any(osc_has_pw(p, n) for n in (1, 2, 3)),
    "chorus_mix": lambda p: bool(p.get("fx_a1_on", False)),
}

# Neutral resting values for inactive parameters
RESTING_VALUES: Dict[str, Any] = {
    "osc1_unison_detune": 0.0,
    "osc2_unison_detune": 0.0,
    "osc3_unison_detune": 0.0,
    "osc1_shape": 0.0,
    "osc2_shape": 0.0,
    "osc3_shape": 0.0,
    "osc1_width": 0.5,
    "osc2_width": 0.5,
    "osc3_width": 0.5,
    "osc1_sub_mix": 0.0,
    "osc2_sub_mix": 0.0,
    "osc3_sub_mix": 0.0,
    "osc1_sync": 0.0,
    "osc2_sync": 0.0,
    "osc3_sync": 0.0,
    "osc2_volume": 0.0,
    "osc3_volume": 0.0,
    "ringmod_12_volume": 0.0,
    "ringmod_23_volume": 0.0,
    "filter_balance": -1.0,  # 100% Filter 1 in single filter patches
    "filter_2_type": "off",
    "filter_2_cutoff": 0.0,
    "filter_2_resonance": 0.0,
    "filter_2_keytrack": 0.0,
    "filter_2_feg_amount": 0.0,
    "feedback": 0.0,
    "waveshaper_drive": 0.0,
    "feg_to_pw": 0.0,
    "chorus_mix": 0.0,
}

def is_parameter_active(name: str, patch: Dict[str, Any]) -> bool:
    """Evaluates whether parameter `name` is active given `patch`."""
    rule = ACTIVE_IF_RULES.get(name)
    if rule is None:
        return True  # By default active if not explicitly gated
    return bool(rule(patch))

def get_activity_mask(patch: Dict[str, Any], param_names: List[str]) -> np.ndarray:
    """Returns a boolean numpy array of shape [len(param_names)]."""
    mask = np.zeros(len(param_names), dtype=bool)
    for i, name in enumerate(param_names):
        mask[i] = is_parameter_active(name, patch)
    return mask

def canonicalize(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Sets inactive parameters to their canonical resting values."""
    p_copy = dict(patch)
    for name, resting in RESTING_VALUES.items():
        if not is_parameter_active(name, p_copy):
            p_copy[name] = resting
    return p_copy
