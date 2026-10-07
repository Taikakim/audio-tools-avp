"""CPU Identifiability Probe for Surge XT Synth Parameters (v4 Spec Section 3c).

Referees the declarative activity rules in surge_activity.py:
For each parameter under each gating condition, renders ~200 random patches
with the parameter at min vs max, computing the mel-spectrogram L1 difference
relative to the repeat-noise floor.
Outputs activity_probe.json.
"""
import os
import json
import random
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
from surge_synth import SurgeRenderer
from surge_activity import is_parameter_active, ACTIVE_IF_RULES

def mel_spec(y: np.ndarray, sr: int = 44100) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    return librosa.power_to_db(m, ref=np.max)

def mel_l1_distance(y1: np.ndarray, y2: np.ndarray, sr: int = 44100) -> float:
    m1 = mel_spec(y1, sr=sr)
    m2 = mel_spec(y2, sr=sr)
    return float(np.mean(np.abs(m1 - m2)))

def generate_base_random_patch(rng: random.Random) -> Dict[str, Any]:
    """Generates a random valid base patch."""
    return {
        "midi_note": rng.randint(28, 50),
        "note_dur": rng.uniform(0.2, 0.45),
        "osc1_mute": False,
        "osc1_type": "Classic",
        "osc1_shape": rng.uniform(-1.0, 1.0),
        "osc1_width": rng.uniform(0.1, 0.9),
        "osc1_octave": rng.choice([-1, 0, 1]),
        "osc1_pitch": rng.uniform(-1.0, 1.0),
        "osc1_unison_voices": 1,
        "osc1_unison_detune": 0.0,
        "osc2_mute": True,
        "osc2_type": "Classic",
        "osc2_shape": rng.uniform(-1.0, 1.0),
        "osc2_width": rng.uniform(0.1, 0.9),
        "osc2_octave": rng.choice([-1, 0, 1]),
        "osc2_pitch": rng.uniform(-1.0, 1.0),
        "osc2_unison_voices": 1,
        "osc2_unison_detune": 0.0,
        "osc3_mute": True,
        "osc3_type": "Classic",
        "osc3_shape": 0.0,
        "osc3_width": 0.5,
        "filter_1_type": "LP 24 dB",
        "filter_1_cutoff": rng.uniform(-20.0, 40.0),
        "filter_1_resonance": rng.uniform(0.0, 0.7),
        "filter_1_feg_amount": rng.uniform(0.0, 48.0),
        "filter_config": "Serial 1",
        "filter_balance": 0.0,
        "filter_2_type": "Off",
        "filter_2_cutoff": 0.0,
        "filter_2_resonance": 0.0,
        "waveshaper_type": "Off",
        "waveshaper_drive": 0.0,
        "aeg_attack": -8.0,
        "aeg_decay": rng.uniform(-4.0, 1.0),
        "aeg_sustain": rng.uniform(0.0, 0.8),
        "aeg_release": rng.uniform(-6.0, -1.0),
        "feg_attack": -8.0,
        "feg_decay": rng.uniform(-4.0, 1.0),
        "feg_sustain": rng.uniform(0.0, 0.8),
        "feg_release": rng.uniform(-6.0, -1.0),
        "ringmod_12_mute": True,
        "ringmod_12_volume": 1.0,
        "feedback": 0.0,
        "osc_drift": 0.0,
    }

def apply_patch_to_renderer(renderer: SurgeRenderer, p: Dict[str, Any]) -> None:
    """Applies patch dictionary parameters to the renderer."""
    renderer.ensure_fx_off()
    
    # Osc 1
    renderer.set_param("Osc 1 Mute", 1.0 if p.get("osc1_mute") else 0.0)
    renderer.set_param("Osc 1 Type", 0.0 if p.get("osc1_type") == "Classic" else 1.0)
    renderer.set_param("Osc 1 Shape", p.get("osc1_shape", 0.0))
    renderer.set_param("Osc 1 Width 1", p.get("osc1_width", 0.5))
    renderer.set_param("Osc 1 Octave", float(p.get("osc1_octave", 0)))
    renderer.set_param("Osc 1 Pitch", p.get("osc1_pitch", 0.0))
    renderer.set_param("Osc 1 Unison Voices", float(p.get("osc1_unison_voices", 1)))
    renderer.set_param("Osc 1 Unison Detune", p.get("osc1_unison_detune", 0.0))
    renderer.set_param("Osc 1 Retrigger", 1.0)

    # Osc 2
    renderer.set_param("Osc 2 Mute", 1.0 if p.get("osc2_mute", True) else 0.0)
    renderer.set_param("Osc 2 Type", 0.0 if p.get("osc2_type") == "Classic" else 1.0)
    renderer.set_param("Osc 2 Shape", p.get("osc2_shape", 0.0))
    renderer.set_param("Osc 2 Width 1", p.get("osc2_width", 0.5))
    renderer.set_param("Osc 2 Octave", float(p.get("osc2_octave", 0)))
    renderer.set_param("Osc 2 Pitch", p.get("osc2_pitch", 0.0))
    renderer.set_param("Osc 2 Unison Voices", float(p.get("osc2_unison_voices", 1)))
    renderer.set_param("Osc 2 Unison Detune", p.get("osc2_unison_detune", 0.0))
    renderer.set_param("Osc 2 Retrigger", 1.0)

    # Osc 3
    renderer.set_param("Osc 3 Mute", 1.0 if p.get("osc3_mute", True) else 0.0)

    # Filter 1
    renderer.set_param("Filter 1 Type", 2.0 if p.get("filter_1_type") != "Off" else 0.0)
    renderer.set_param("Filter 1 Cutoff", p.get("filter_1_cutoff", 10.0))
    renderer.set_param("Filter 1 Resonance", p.get("filter_1_resonance", 0.0))
    renderer.set_param("Filter 1 FEG Mod Amount", p.get("filter_1_feg_amount", 0.0))

    # Filter 2 & Config
    renderer.set_param("Filter 2 Type", 13.0 if p.get("filter_2_type") not in ("Off", "off", 0) else 0.0)
    renderer.set_param("Filter 2 Cutoff", p.get("filter_2_cutoff", 0.0))
    renderer.set_param("Filter 2 Resonance", p.get("filter_2_resonance", 0.0))
    renderer.set_param("Filter Configuration", 0.0 if "Serial" in str(p.get("filter_config")) else 4.0)
    renderer.set_param("Filter Balance", p.get("filter_balance", 0.0))

    # Waveshaper
    renderer.set_param("Waveshaper Type", 2.0 if p.get("waveshaper_type") not in ("Off", "off", 0) else 0.0)
    renderer.set_param("Waveshaper Drive", p.get("waveshaper_drive", 0.0))

    # Envelopes
    renderer.set_param("Amp EG Decay", p.get("aeg_decay", -2.0))
    renderer.set_param("Amp EG Sustain", p.get("aeg_sustain", 0.5))
    renderer.set_param("Filter EG Decay", p.get("feg_decay", -2.0))
    renderer.set_param("Filter EG Sustain", p.get("feg_sustain", 0.5))

    # Ring mod
    renderer.set_param("Ring Modulation 1x2 Mute", 1.0 if p.get("ringmod_12_mute", True) else 0.0)
    renderer.set_param("Ring Modulation 1x2 Volume", p.get("ringmod_12_volume", 1.0))

def run_probe_condition(
    renderer: SurgeRenderer,
    param_name: str,
    surge_param_name: str,
    val_min: float,
    val_max: float,
    condition_name: str,
    condition_setter: Dict[str, Any],
    n_patches: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tests a single condition state across n_patches random patches."""
    rng = random.Random(seed)
    param_diffs = []
    repeat_diffs = []
    silent_count = 0

    for _ in range(n_patches):
        p = generate_base_random_patch(rng)
        p.update(condition_setter)
        apply_patch_to_renderer(renderer, p)

        # 1. Render baseline at min
        renderer.set_param(surge_param_name, val_min)
        a_min1 = renderer.render_note(p["midi_note"], note_dur=p["note_dur"], total_dur=0.7)

        # 2. Render repeat baseline (measure noise floor)
        a_min2 = renderer.render_note(p["midi_note"], note_dur=p["note_dur"], total_dur=0.7)
        d_rep = mel_l1_distance(a_min1, a_min2, sr=renderer.sample_rate)

        # 3. Render at max
        renderer.set_param(surge_param_name, val_max)
        a_max = renderer.render_note(p["midi_note"], note_dur=p["note_dur"], total_dur=0.7)
        d_param = mel_l1_distance(a_min1, a_max, sr=renderer.sample_rate)

        param_diffs.append(d_param)
        repeat_diffs.append(d_rep)

        # Inactive condition check: param diff does not exceed repeat noise floor
        if d_param <= (d_rep * 1.2) or (d_param - d_rep) < 0.1:
            silent_count += 1

    arr_param = np.array(param_diffs, dtype=np.float32)
    arr_rep = np.array(repeat_diffs, dtype=np.float32)

    frac_silent = float(silent_count / n_patches)
    # Empirically active if fewer than 95% of patches are below the noise floor
    empirically_active = (frac_silent < 0.95)

    sample_patch = generate_base_random_patch(random.Random(0))
    sample_patch.update(condition_setter)
    table_active = is_parameter_active(param_name, sample_patch)

    agrees = (empirically_active == table_active)

    return {
        "param_name": param_name,
        "surge_param_name": surge_param_name,
        "condition_name": condition_name,
        "condition_patch": condition_setter,
        "n_patches": n_patches,
        "mean_param_diff": float(np.mean(arr_param)),
        "mean_repeat_noise": float(np.mean(arr_rep)),
        "frac_silent": frac_silent,
        "empirically_active": empirically_active,
        "table_predicted_active": table_active,
        "agrees": agrees,
    }

def main():
    parser = argparse.ArgumentParser(description="Surge XT Activity Masking Identifiability Probe")
    parser.add_argument("--n_patches", type=int, default=200, help="Number of random patches per probe condition")
    parser.add_argument("--out", type=str, default="activity_probe.json", help="Output JSON path")
    args = parser.parse_args()

    renderer = SurgeRenderer()

    probe_specs = [
        # 1. Osc 2 Octave
        ("osc2_octave", "Osc 2 Octave", -3.0, 3.0, "osc2_muted", {"osc2_mute": True}),
        ("osc2_octave", "Osc 2 Octave", -3.0, 3.0, "osc2_unmuted", {"osc2_mute": False}),

        # 2. Osc 2 Shape
        ("osc2_shape", "Osc 2 Shape", -1.0, 1.0, "osc2_muted", {"osc2_mute": True}),
        ("osc2_shape", "Osc 2 Shape", -1.0, 1.0, "osc2_unmuted", {"osc2_mute": False, "osc2_type": "Classic"}),

        # 3. Osc 1 Unison Detune
        ("osc1_unison_detune", "Osc 1 Unison Detune", 0.0, 1.0, "voices_1", {"osc1_unison_voices": 1}),
        ("osc1_unison_detune", "Osc 1 Unison Detune", 0.0, 1.0, "voices_2", {"osc1_unison_voices": 2}),

        # 4. Osc 2 Unison Detune
        ("osc2_unison_detune", "Osc 2 Unison Detune", 0.0, 1.0, "osc2_muted_voices_2", {"osc2_mute": True, "osc2_unison_voices": 2}),
        ("osc2_unison_detune", "Osc 2 Unison Detune", 0.0, 1.0, "osc2_unmuted_voices_1", {"osc2_mute": False, "osc2_unison_voices": 1}),
        ("osc2_unison_detune", "Osc 2 Unison Detune", 0.0, 1.0, "osc2_unmuted_voices_2", {"osc2_mute": False, "osc2_unison_voices": 2}),

        # 5. Waveshaper Drive
        ("waveshaper_drive", "Waveshaper Drive", -24.0, 24.0, "ws_off", {"waveshaper_type": "Off"}),
        ("waveshaper_drive", "Waveshaper Drive", -24.0, 24.0, "ws_hard", {"waveshaper_type": "Hard"}),

        # 6. Filter 2 Cutoff
        ("filter_2_cutoff", "Filter 2 Cutoff", -20.0, 40.0, "filter2_off", {"filter_2_type": "Off"}),
        ("filter_2_cutoff", "Filter 2 Cutoff", -20.0, 40.0, "filter2_on_serial", {"filter_2_type": "LP K35", "filter_config": "Serial 1"}),
        ("filter_2_cutoff", "Filter 2 Cutoff", -20.0, 40.0, "filter2_on_dual_mix", {"filter_2_type": "LP K35", "filter_config": "Dual 2", "filter_balance": 0.0}),
        ("filter_2_cutoff", "Filter 2 Cutoff", -20.0, 40.0, "filter2_on_dual_single_f1", {"filter_2_type": "LP K35", "filter_config": "Dual 2", "filter_balance": -1.0}),

        # 7. Ring Modulation 1x2 Volume
        ("ringmod_12_volume", "Ring Modulation 1x2 Volume", 0.0, 1.0, "ringmod_muted", {"ringmod_12_mute": True, "osc1_mute": False, "osc2_mute": False}),
        ("ringmod_12_volume", "Ring Modulation 1x2 Volume", 0.0, 1.0, "ringmod_unmuted_osc2_muted", {"ringmod_12_mute": False, "osc1_mute": False, "osc2_mute": True}),
        ("ringmod_12_volume", "Ring Modulation 1x2 Volume", 0.0, 1.0, "ringmod_unmuted_both_active", {"ringmod_12_mute": False, "osc1_mute": False, "osc2_mute": False}),
    ]

    print(f"Running identifiability probe on {len(probe_specs)} conditions (N={args.n_patches} patches each)...")
    results = []
    mismatches = 0

    for param, surge_param, val_min, val_max, cond_name, cond_dict in probe_specs:
        res = run_probe_condition(
            renderer=renderer,
            param_name=param,
            surge_param_name=surge_param,
            val_min=val_min,
            val_max=val_max,
            condition_name=cond_name,
            condition_setter=cond_dict,
            n_patches=args.n_patches,
        )
        status = "AGREED" if res["agrees"] else "MISMATCH"
        if not res["agrees"]:
            mismatches += 1
        print(f"[{status}] {param} under {cond_name}: Empirical={res['empirically_active']}, Table={res['table_predicted_active']} (param diff: {res['mean_param_diff']:.3f}, noise: {res['mean_repeat_noise']:.3f}, silent: {res['frac_silent']:.1%})")
        results.append(res)

    out_data = {
        "total_probed": len(results),
        "mismatches": mismatches,
        "n_patches_per_condition": args.n_patches,
        "results": results
    }

    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nProbe complete! Written to {args.out}. Total mismatches: {mismatches}/{len(results)}")
    if mismatches > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
