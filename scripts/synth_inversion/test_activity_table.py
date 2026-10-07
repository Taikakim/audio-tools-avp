"""Automated agreement test between declarative activity table and CPU identifiability probe.

Fails whenever surge_activity.py and activity_probe.json disagree on any condition.
Per Synth Inversion v4 Spec Section 3c.
"""
import os
import json
import pytest
from surge_activity import is_parameter_active

def test_probe_vs_table_agreement():
    probe_path = os.path.join(os.path.dirname(__file__), "activity_probe.json")
    assert os.path.exists(probe_path), f"activity_probe.json not found at {probe_path}"
    
    with open(probe_path, "r") as f:
        data = json.load(f)
        
    results = data["results"]
    assert len(results) > 0, "No probe results found in activity_probe.json"
    
    mismatches = []
    for entry in results:
        param = entry["param_name"]
        cond_patch = entry["condition_patch"]
        cond_name = entry["condition_name"]
        emp_active = entry["empirically_active"]
        
        # Test against current table logic
        table_active = is_parameter_active(param, cond_patch)
        if table_active != emp_active:
            mismatches.append(f"Param '{param}' under '{cond_name}': table={table_active} vs empirical={emp_active}")
            
    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches between table and probe:\n" + "\n".join(mismatches)

def test_canonicalize_resting_values():
    """Verify that canonicalize() sets inactive parameters to neutral resting values."""
    from surge_activity import canonicalize
    
    # Inactive Osc 2 & Inactive Filter 2 patch
    p = {
        "osc1_mute": False,
        "osc1_type": "Classic",
        "osc1_unison_voices": 1,
        "osc1_unison_detune": 0.85, # Inactive because voices == 1
        "osc2_mute": True,
        "osc2_volume": 0.9,         # Inactive because muted
        "osc2_unison_detune": 0.5,  # Inactive
        "filter_2_type": "off",
        "filter_2_cutoff": 25.0,    # Inactive
        "waveshaper_type": "off",
        "waveshaper_drive": 12.0,   # Inactive
        "ringmod_12_mute": True,
        "ringmod_12_volume": 0.8,   # Inactive
    }
    
    c = canonicalize(p)
    assert c["osc1_unison_detune"] == 0.0
    assert c["osc2_volume"] == 0.0
    assert c["osc2_unison_detune"] == 0.0
    assert c["filter_2_cutoff"] == 0.0
    assert c["waveshaper_drive"] == 0.0
    assert c["ringmod_12_volume"] == 0.0

if __name__ == "__main__":
    test_probe_vs_table_agreement()
    test_canonicalize_resting_values()
    print("ALL ACTIVITY TABLE TESTS PASSED!")

def test_v4_registry_coverage():
    """Verify that all parameters in V4_PARAM_NAMES have active-if rules and bounds."""
    from surge_v4_registry import V4_PARAM_NAMES, DEFAULT_BOUNDS
    from surge_activity import ACTIVE_IF_RULES
    
    missing_rules = [p for p in V4_PARAM_NAMES if p not in ACTIVE_IF_RULES]
    assert len(missing_rules) == 0, f"Parameters missing active-if rules: {missing_rules}"
    
    # Check bounds
    for p in V4_PARAM_NAMES:
        if p in DEFAULT_BOUNDS:
            lo, hi = DEFAULT_BOUNDS[p]
            assert lo < hi, f"Invalid bounds for {p}: ({lo}, {hi})"

