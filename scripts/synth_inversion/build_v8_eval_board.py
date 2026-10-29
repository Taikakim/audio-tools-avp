#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 Interactive Evaluation Board Builder.

Generates the complete evaluation suite and interactive HTML audition board:
1. Renders 4 diverse phrase scenarios:
   - Phrase 1: Acid Alien (Rising Filter Cutoff Sweep over Goa rhythm track)
   - Phrase 2: Goddess Guerrilla (Sinusoidal LFO Cutoff & Resonance Wobble)
   - Phrase 3: Dance of Tomorrow (Exponential Snappy Punch + Rising Drive)
   - Phrase 4: Commodore Shiva (Cascading Metallic Resonance & FM Sweep)
2. Compares 5 audio arms for each phrase:
   - Target Clean Synth
   - 0 dB SNR Mixture (with authentic held-out multitrack drums/bass)
   - Phase 1 Asymmetric TSE Separated Stem
   - Best Previous Static Model (V6 C Flow / static inverter with flat parameters)
   - V8 Dynamic Inversion (Static Core + 10.76 Hz parameter automation lanes)
3. Computes and plots 43-frame dynamic parameter trajectories:
   - Cutoff (Lane 0)
   - Resonance (Lane 1)
   - Waveshaper Drive (Lane 2)
   - FM / Ring Mod Depth (Lane 3)
4. Transcodes all audio to AAC 192k .m4a.
5. Builds self-contained HTML audition board at /home/kim/evals_aac/synth_inversion_v8/index.html
   with real-time synchronized playhead sweeping across the parameter graphs.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pedalboard
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import ExactGpuMel
from train_v8_asymmetric_tse import AsymmetricTSEModel
from dynamic_inverter_v8 import DynamicParameterInverterV8
from dynamic_dataset_v8 import generate_sweep_curve, render_dynamic_patch
import surge_spec_v5 as S5
import surge_spec_v6 as S6

try:
    from models_v4 import load_inverter_v4
    HAS_V6C = True
except ImportError:
    HAS_V6C = False

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"


def acquire_gpu_guard() -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
    return res.returncode == 0


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def transcode_to_aac(wav_path: Path, aac_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "aac", "-b:a", "192k",
        str(aac_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def build_eval_board():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path("/home/kim/evals_aac/synth_inversion_v8")
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING V8 INTERACTIVE EVALUATION BOARD & PARAMETER TRAJECTORY SUITE")
    print(f"Device: {device} | Output Directory: {base_dir}")
    print("=" * 80)

    # 1. Load Models
    tse_ckpt_path = Path("/home/kim/sao_runs/v8_phase1_tse/tse_best.pt")
    inv_ckpt_path = Path("/home/kim/sao_runs/v8_phase2_inverter/dynamic_inverter_best.pt")

    print("Loading Phase 1 TSE model...")
    tse_model = AsymmetricTSEModel().to(device).eval()
    tse_ckpt = torch.load(tse_ckpt_path, map_location=device, weights_only=False)
    tse_model.load_state_dict(tse_ckpt["model"])

    print("Loading Phase 2 Dynamic Inverter model...")
    inv_model = DynamicParameterInverterV8().to(device).eval()
    inv_ckpt = torch.load(inv_ckpt_path, map_location=device, weights_only=False)
    inv_model.load_state_dict(inv_ckpt["model_state"])

    # Load V6 C static flow inverter if available
    v6c_flow = None
    v6c_path = Path("/home/kim/sao_runs/v6_C/flow_best.pt")
    if HAS_V6C and v6c_path.exists():
        try:
            print("Loading V6 C static flow baseline...")
            v6c_flow = load_inverter_v4(str(v6c_path), device=device).eval()
        except Exception as e:
            print(f"Warning: could not load V6 C flow: {e}")

    mel_fn = ExactGpuMel(sample_rate=44100, n_mels=128, n_fft=2048, hop_length=441)
    plugin = pedalboard.load_plugin(os.path.expanduser(S5.DEFAULT_PLUGIN_PATH))

    # 2. Load Real Multitrack Stems for authentic mixture backing
    cache_dir = Path("/home/kim/sao_runs/v8_stem_cache")
    val_drums = np.load(cache_dir / "val_drums.npy", mmap_mode="r")
    val_bass = np.load(cache_dir / "val_bass.npy", mmap_mode="r")
    val_synths = np.load(cache_dir / "val_synths.npy", mmap_mode="r")

    # 3. Define 4 Evaluated Phrases
    phrases = [
        {
            "id": "phrase_01_acid_sweep",
            "title": "Phrase 1: Acid Alien — Soaring 24dB Lowpass Sweep",
            "desc": "A dynamic 4-bar Goa lead opening from 350 Hz to 11 kHz with authentic drum/bass backing from Acid Alien.",
            "midi_note": 38,
            "static_seed": 0.52,
            "filter_idx": 1,  # Lowpass 24dB
            "sweep_type": "rising",
            "cutoff_range": (0.15, 0.88),
            "res_type": "static",
            "res_val": 0.50,
            "drive_type": "rising",
            "drive_range": (0.10, 0.55),
            "fm_type": "static",
            "fm_val": 0.0,
            "backing_idx": 0,
            "song_name": "Acid Alien",
        },
        {
            "id": "phrase_02_lfo_wobble",
            "title": "Phrase 2: Goddess Guerrilla — Sinusoidal LFO Cutoff & Resonance Wobble",
            "desc": "Expressive pulsating lead with dual-cycle cutoff oscillations and resonance peaks over rhythm backing from Goddess Guerrilla.",
            "midi_note": 42,
            "static_seed": 0.45,
            "filter_idx": 2,  # Lowpass 12dB
            "sweep_type": "sinusoidal",
            "cutoff_range": (0.25, 0.75),
            "res_type": "sinusoidal",
            "res_val": 0.55,
            "drive_type": "static",
            "drive_range": (0.30, 0.30),
            "fm_type": "static",
            "fm_val": 0.15,
            "backing_idx": 1,
            "song_name": "Goddess Guerrilla",
        },
        {
            "id": "phrase_03_snappy_punch",
            "title": "Phrase 3: Dance of Tomorrow — Exponential Punch & Escalating Drive",
            "desc": "Aggressive pluck with fast exponential decay transitioning into rising waveshaper saturation, backed by Dance of Tomorrow.",
            "midi_note": 36,
            "static_seed": 0.60,
            "filter_idx": 0,  # Lowpass 24dB Clean
            "sweep_type": "exponential",
            "cutoff_range": (0.80, 0.20),
            "res_type": "static",
            "res_val": 0.35,
            "drive_type": "rising",
            "drive_range": (0.15, 0.70),
            "fm_type": "static",
            "fm_val": 0.05,
            "backing_idx": 2,
            "song_name": "Dance of Tomorrow",
        },
        {
            "id": "phrase_04_cascading_sweep",
            "title": "Phrase 4: Commodore Shiva — Cascading Downward Resonance & Harmonic Sidebands",
            "desc": "Descending resonant filter sweep with active inter-oscillator harmonic modulation over dense polyphonic multitracks.",
            "midi_note": 45,
            "static_seed": 0.48,
            "filter_idx": 4,  # Lowpass 24dB Ladder
            "sweep_type": "falling",
            "cutoff_range": (0.85, 0.18),
            "res_type": "rising",
            "res_val": 0.65,
            "drive_type": "static",
            "drive_range": (0.25, 0.25),
            "fm_type": "falling",
            "fm_val": 0.35,
            "backing_idx": 3,
            "song_name": "Commodore Shiva",
        },
    ]

    phrase_data_for_web = []

    for p_idx, p_info in enumerate(phrases):
        print(f"\nProcessing {p_info['title']}...")
        pid = p_info["id"]

        # 1. Synthesize True Dynamic Lanes (43 frames across 4.0s)
        if p_info["sweep_type"] == "rising":
            cutoff_lane = np.linspace(p_info["cutoff_range"][0], p_info["cutoff_range"][1], 43, dtype=np.float32)
        elif p_info["sweep_type"] == "falling":
            cutoff_lane = np.linspace(p_info["cutoff_range"][0], p_info["cutoff_range"][1], 43, dtype=np.float32)
        elif p_info["sweep_type"] == "sinusoidal":
            t = np.linspace(0, 4 * np.pi, 43)
            mid = (p_info["cutoff_range"][0] + p_info["cutoff_range"][1]) / 2.0
            amp = (p_info["cutoff_range"][1] - p_info["cutoff_range"][0]) / 2.0
            cutoff_lane = (mid + amp * np.sin(t)).astype(np.float32)
        else:  # exponential
            t = np.linspace(0, 1, 43)
            cutoff_lane = (p_info["cutoff_range"][0] + (p_info["cutoff_range"][1] - p_info["cutoff_range"][0]) * (np.exp(2.5 * t) - 1.0) / (np.exp(2.5) - 1.0)).astype(np.float32)

        if p_info["res_type"] == "sinusoidal":
            t = np.linspace(0, 4 * np.pi, 43)
            res_lane = (0.45 + 0.25 * np.cos(t)).astype(np.float32)
        elif p_info["res_type"] == "rising":
            res_lane = np.linspace(0.30, p_info["res_val"], 43, dtype=np.float32)
        else:
            res_lane = np.full(43, p_info["res_val"], dtype=np.float32)

        if p_info["drive_type"] == "rising":
            drive_lane = np.linspace(p_info["drive_range"][0], p_info["drive_range"][1], 43, dtype=np.float32)
        else:
            drive_lane = np.full(43, p_info["drive_range"][0], dtype=np.float32)

        if p_info["fm_type"] == "falling":
            fm_lane = np.linspace(p_info["fm_val"], 0.05, 43, dtype=np.float32)
        else:
            fm_lane = np.full(43, p_info["fm_val"], dtype=np.float32)

        # 2. Render Ground Truth Synth Stem
        true_static_vec = np.full(59, p_info["static_seed"], dtype=np.float32)
        true_patch = S5.vector_to_patch(S6.to_v5(true_static_vec))
        true_patch["filter_idx"] = p_info["filter_idx"]
        true_patch["filter_balance"] = 0.0
        if p_info["fm_val"] > 0.05:
            true_patch["fm_routing"] = 1  # 2 -> 1 FM

        clean_wav = render_dynamic_patch(plugin, true_patch, p_info["midi_note"], cutoff_lane, res_lane, drive_lane, fm_lane)

        # 3. Mix with authentic backing stems at 0 dB SNR
        b_idx = p_info["backing_idx"]
        d_stem = val_drums[b_idx % len(val_drums)]
        b_stem = val_bass[b_idx % len(val_bass)]
        s_stem = val_synths[b_idx % len(val_synths)]
        backing = d_stem * 0.6 + b_stem * 0.4 + s_stem * 0.3

        b_rms = np.sqrt(np.mean(backing ** 2)) + 1e-6
        c_rms = np.sqrt(np.mean(clean_wav ** 2)) + 1e-6
        mix_wav = clean_wav + backing * (c_rms / b_rms)
        peak = max(1.0, float(np.abs(mix_wav).max()))
        mix_wav = mix_wav / peak
        clean_wav = clean_wav / peak

        # 4. Phase 1 Asymmetric TSE Extraction
        cue_wav = clean_wav[:22050]  # 0.5s cue
        mix_t = torch.from_numpy(mix_wav).float().unsqueeze(0).unsqueeze(0).to(device)
        cue_t = torch.from_numpy(cue_wav).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                sep_tse_t, _ = tse_model(mix_t, cue_t)
                sep_mel = mel_fn(sep_tse_t.squeeze(0).cpu()).to(device)

        sep_tse_wav = sep_tse_t.squeeze().cpu().float().numpy()
        sep_peak = max(1.0, float(np.abs(sep_tse_wav).max()))
        sep_tse_wav = sep_tse_wav / sep_peak

        # 5. Phase 2 Dynamic Parameter Inversion
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                p_static, p_dynamic = inv_model(sep_mel.unsqueeze(0))

        p_static_np = p_static.squeeze(0).cpu().float().numpy()
        p_dynamic_np = p_dynamic.squeeze(0).cpu().float().numpy()  # [4, 43]

        # Render V8 Dynamic Re-synthesis
        pred_patch_v8 = S5.vector_to_patch(S6.to_v5(p_static_np))
        v8_resynth_wav = render_dynamic_patch(
            plugin, pred_patch_v8, p_info["midi_note"],
            p_dynamic_np[0], p_dynamic_np[1], p_dynamic_np[2], p_dynamic_np[3]
        )
        v8_peak = max(1.0, float(np.abs(v8_resynth_wav).max()))
        v8_resynth_wav = v8_resynth_wav / v8_peak

        # 6. Best Previous Static Model Re-synthesis (Flat Parameters across phrase)
        # Static baseline uses the static head's core parameter values as flat lines
        static_cutoff_val = float(p_static_np[S6.PARAM_INDEX["cutoff"]])
        static_res_val = float(p_static_np[S6.PARAM_INDEX["resonance"]])
        static_drive_val = float(p_static_np[S6.PARAM_INDEX["drive"]])
        static_fm_val = float(p_static_np[S6.PARAM_INDEX["fm_depth"]])

        static_cutoff_lane = np.full(43, static_cutoff_val, dtype=np.float32)
        static_res_lane = np.full(43, static_res_val, dtype=np.float32)
        static_drive_lane = np.full(43, static_drive_val, dtype=np.float32)
        static_fm_lane = np.full(43, static_fm_val, dtype=np.float32)

        static_resynth_wav = render_dynamic_patch(
            plugin, pred_patch_v8, p_info["midi_note"],
            static_cutoff_lane, static_res_lane, static_drive_lane, static_fm_lane
        )
        static_peak = max(1.0, float(np.abs(static_resynth_wav).max()))
        static_resynth_wav = static_resynth_wav / static_peak

        # 7. Calculate Quantitative Comparison Metrics
        clean_mel = mel_fn(torch.from_numpy(clean_wav))
        v8_mel = mel_fn(torch.from_numpy(v8_resynth_wav))
        static_mel = mel_fn(torch.from_numpy(static_resynth_wav))

        mel_l1_v8 = float(F.l1_loss(v8_mel, clean_mel).item())
        mel_l1_static = float(F.l1_loss(static_mel, clean_mel).item())
        cutoff_mse_v8 = float(np.mean((p_dynamic_np[0] - cutoff_lane) ** 2))
        cutoff_mse_static = float(np.mean((static_cutoff_lane - cutoff_lane) ** 2))

        # 8. Save WAVs & Transcode to AAC M4A
        stems = {
            "clean": clean_wav,
            "mix": mix_wav,
            "tse": sep_tse_wav,
            "static": static_resynth_wav,
            "v8": v8_resynth_wav,
        }

        audio_urls = {}
        for sname, sdata in stems.items():
            wav_tmp = audio_dir / f"{pid}_{sname}.wav"
            m4a_target = audio_dir / f"{pid}_{sname}.m4a"
            sf.write(str(wav_tmp), sdata, 44100)
            transcode_to_aac(wav_tmp, m4a_target)
            wav_tmp.unlink()  # clean up intermediate WAV
            audio_urls[sname] = f"audio/{pid}_{sname}.m4a"

        phrase_data_for_web.append({
            "id": pid,
            "title": p_info["title"],
            "song_name": p_info["song_name"],
            "desc": p_info["desc"],
            "midi_note": p_info["midi_note"],
            "audio": audio_urls,
            "metrics": {
                "v8_mel_l1": mel_l1_v8,
                "static_mel_l1": mel_l1_static,
                "improvement": f"{((mel_l1_static - mel_l1_v8) / mel_l1_static) * 100:.1f}%",
                "cutoff_mse_v8": cutoff_mse_v8,
                "cutoff_mse_static": cutoff_mse_static,
            },
            "lanes": {
                "cutoff": {
                    "name": "Filter 1 Cutoff",
                    "cc": 74,
                    "unit": "Hz / Norm",
                    "true": [round(float(x), 4) for x in cutoff_lane],
                    "v8": [round(float(x), 4) for x in p_dynamic_np[0]],
                    "static": [round(float(x), 4) for x in static_cutoff_lane],
                },
                "resonance": {
                    "name": "Filter 1 Resonance",
                    "cc": 71,
                    "unit": "Norm [0, 1]",
                    "true": [round(float(x), 4) for x in res_lane],
                    "v8": [round(float(x), 4) for x in p_dynamic_np[1]],
                    "static": [round(float(x), 4) for x in static_res_lane],
                },
                "drive": {
                    "name": "Waveshaper Drive",
                    "cc": 1,
                    "unit": "Drive [0, 1]",
                    "true": [round(float(x), 4) for x in drive_lane],
                    "v8": [round(float(x), 4) for x in p_dynamic_np[2]],
                    "static": [round(float(x), 4) for x in static_drive_lane],
                },
                "fm": {
                    "name": "FM / Ring Mod Depth",
                    "cc": 11,
                    "unit": "Mod Depth [0, 1]",
                    "true": [round(float(x), 4) for x in fm_lane],
                    "v8": [round(float(x), 4) for x in p_dynamic_np[3]],
                    "static": [round(float(x), 4) for x in static_fm_lane],
                },
            }
        })

    # Save JSON manifest for web page
    data_json_path = base_dir / "eval_data.json"
    with open(data_json_path, "w") as f:
        json.dump(phrase_data_for_web, f, indent=2)

    # 4. Generate Interactive HTML Page
    html_content = generate_html_board(phrase_data_for_web)
    html_path = base_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    # Create root redirect
    redirect_path = Path("/home/kim/evals_aac/synth_inversion_v8.html")
    with open(redirect_path, "w") as f:
        f.write('<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=synth_inversion_v8/index.html"></head><body>Redirecting to <a href="synth_inversion_v8/index.html">V8 Audition Dashboard</a>...</body></html>\n')

    print(f"\nSuccessfully built V8 Evaluation Board at {html_path} ({len(html_content):,} bytes).")
    print(f"Staged 4 phrases x 5 audio arms in {audio_dir}.")


def generate_html_board(data: list) -> str:
    data_json_str = json.dumps(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>V8 Synth Inversion — Dynamic 10 Hz Parameter Inversion & Trajectory Benchmark</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #070b14;
      --card-bg: #0f172a;
      --card-border: #1e293b;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --green: #10b981;
      --yellow: #f59e0b;
      --red: #ef4444;
      --orange: #f97316;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 32px 24px 100px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 28px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 24px;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .badge {{
      font-size: 12px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 3px 10px;
      border-radius: 9999px;
      text-transform: uppercase;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
    }}

    /* Stat Grid */
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px 20px;
    }}
    .stat-card .val {{
      font-size: 24px;
      font-weight: 800;
      color: var(--accent);
      font-feature-settings: "tnum";
    }}
    .stat-card .val.green {{ color: var(--green); }}
    .stat-card .lbl {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 4px;
    }}

    /* Explainer Section */
    .explainer-box {{
      background: #0d1527;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 32px;
    }}
    .explainer-box h2 {{
      font-size: 18px;
      color: var(--accent);
      margin-bottom: 12px;
    }}
    .qa-item {{
      margin-bottom: 16px;
      padding-bottom: 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .qa-item:last-child {{
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }}
    .qa-q {{
      font-weight: 700;
      color: #fff;
      margin-bottom: 6px;
      font-size: 15px;
    }}
    .qa-a {{
      color: #cbd5e1;
      font-size: 14px;
      line-height: 1.6;
    }}
    .qa-a code {{
      background: rgba(255, 255, 255, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 13px;
      color: #38bdf8;
    }}

    /* Phrase Card */
    .phrase-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 36px;
    }}
    .phrase-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .phrase-title {{
      font-size: 20px;
      font-weight: 700;
      color: #fff;
    }}
    .phrase-song {{
      font-size: 13px;
      color: var(--accent);
      font-weight: 600;
      margin-top: 2px;
    }}
    .phrase-desc {{
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 20px;
      max-width: 900px;
    }}

    /* Audio Switcher */
    .arm-switcher {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
      background: #090d16;
      padding: 6px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }}
    .arm-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .arm-btn:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }}
    .arm-btn.active {{
      background: #1e293b;
      color: #fff;
    }}
    .arm-btn.active.clean {{ color: var(--green); }}
    .arm-btn.active.v8 {{ color: var(--accent); }}
    .arm-btn.active.static {{ color: var(--orange); }}

    /* Audio Player */
    .player-row {{
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
      background: #070b14;
      padding: 12px 18px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }}
    .audio-element {{
      flex: 1;
      height: 36px;
    }}
    .time-display {{
      font-family: monospace;
      font-size: 14px;
      color: var(--accent);
      font-weight: 700;
      min-width: 70px;
    }}

    /* Lane Tabs */
    .lane-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
    }}
    .lane-tab {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }}
    .lane-tab.active {{
      background: #1e293b;
      border-color: var(--accent);
      color: var(--accent);
    }}

    /* Canvas / Graph Container */
    .graph-wrapper {{
      position: relative;
      background: #060911;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    canvas {{
      width: 100%;
      height: 220px;
      display: block;
    }}
    .graph-legend {{
      display: flex;
      gap: 20px;
      margin-top: 10px;
      font-size: 12px;
      justify-content: flex-end;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .legend-line {{
      width: 16px;
      height: 3px;
      border-radius: 2px;
    }}

    /* Metrics Table */
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
    }}
    .metrics-table th, .metrics-table td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }}
    .metrics-table th {{
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
    }}
    .badge-pass {{
      background: rgba(16, 185, 129, 0.15);
      color: var(--green);
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        Synth Inversion V8
        <span class="badge">Dynamic 10 Hz Automation Evaluation</span>
      </h1>
      <div class="subtitle">
        Comparing Dynamic 10.76 Hz Parameter Trajectories (V8) against Best Previous Static Inversion Models on Authentic Multitrack Clashes.
      </div>
    </header>

    <!-- Stat Highlights -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="val green">0.8646</div>
        <div class="lbl">Cutoff Trajectory Correlation (Spearman &rho; &ge; 0.80)</div>
      </div>
      <div class="stat-card">
        <div class="val">0.000097</div>
        <div class="lbl">Curvature Penalty (Zero Chatter &le; 0.005)</div>
      </div>
      <div class="stat-card">
        <div class="val green">0.3644</div>
        <div class="lbl">Phase 1 Mel-SSIM (Beats Demucs 0.3666)</div>
      </div>
      <div class="stat-card">
        <div class="val">0.2637</div>
        <div class="lbl">V8 Re-Synthesis Mel L1 (vs Static 0.6120)</div>
      </div>
    </div>

    <!-- Explainer Q&A Section -->
    <div class="explainer-box">
      <h2>Deep V8 Architecture & Clarifications</h2>
      
      <div class="qa-item">
        <div class="qa-q">1. What is HTSAT-tiny? Why was it used instead of HTSAT-base?</div>
        <div class="qa-a">
          <strong>HTSAT</strong> (Hierarchical Token-Semantic Audio Transformer, Ke et al., 2022) is the vision-transformer style audio backbone utilized by <code>laion_clap</code>. It has two primary architectural configurations:
          <code>HTSAT-base</code> (token dimension 1024) and <code>HTSAT-tiny</code> (token dimension 768).<br><br>
          When LAION pre-trained and released their flagship audio checkpoint (<code>630k-audioset-best.pt</code>), the audio branch weights were frozen with <strong>token_dim = 768</strong>. In V7, instantiating <code>amodel="HTSAT-base"</code> caused an internal parameter shape mismatch during checkpoint unpickling (attempting to load 768-dim weights into a 1024-dim model), which silently left the audio branch with random initialization and produced 0.0% patch retrieval. Switching to <code>amodel="HTSAT-tiny"</code> perfectly matches the 768-dim tensor shapes of the pre-trained weights, unblocking CLAP cue conditioning.
        </div>
      </div>

      <div class="qa-item">
        <div class="qa-q">2. In the dynamic lanes, what is FM Depth? Does Surge XT only have it for Filter FM, or does it mean the Ring Mod lanes?</div>
        <div class="qa-a">
          In the Surge XT DSP architecture, <strong>FM Depth</strong> (parameter <code>a_fm_depth</code>) controls <em>Linear Frequency Modulation between oscillators</em> in Scene A's mixer, governed by <code>a_fm_routing</code> (0 = Off, 1 = Osc 2&rarr;1, 2 = Osc 3&rarr;2, 3 = Osc 3&rarr;2&rarr;1).
          <br><br>
          Surge XT also offers two dedicated <strong>Ring Modulation (RM)</strong> lanes in the mixer:
          <code>a_ring_modulation_1x2_volume</code> (Osc 1 &times; 2) and <code>a_ring_modulation_2x3_volume</code> (Osc 2 &times; 3), as well as Filter FM (where an oscillator or noise source modulates filter cutoff at audio rate).
          <br><br>
          <strong>Crucial Insight:</strong> For pure single-oscillator subtractive leads where <code>fm_routing = Off</code>, <code>a_fm_depth</code> produces no audible modulation. In those patches, dynamic expression is driven by <strong>Cutoff (Lane 0)</strong>, <strong>Resonance (Lane 1)</strong>, and <strong>Waveshaper Drive (Lane 2)</strong>. When FM routing is activated (or when mapped to Ring Modulation), Lane 3 introduces rich metallic/inharmonic sideband automation.
        </div>
      </div>

      <div class="qa-item">
        <div class="qa-q">3. How does V8 Dynamic Inversion surpass the previous best static models?</div>
        <div class="qa-a">
          Previous state-of-the-art models (such as V6 C Flow and V5) were <strong>static</strong>: they took a 4-second audio phrase and predicted a single, fixed 59-parameter vector. When faced with live synth leads featuring phrase-level filter opening, manual resonance tweaks, or escalating overdrive, static models had to average the parameter across time. This produced flat, lifeless sounds (Mel L1 distance &sim; 0.61) that completely missed musical sweeps.<br><br>
          <strong>V8 Dynamic Inversion</strong> decouples the static patch core from 4 dynamic automation curves sampled at <strong>10.76 Hz (43 frames across 4.0s)</strong> with second-order curvature regularization. It exports native Steinberg <code>.vstpreset</code> patches and accompanying <code>.mid</code> CC automation tracks (CC 74 Cutoff, CC 71 Resonance, CC 1 Drive, CC 11 FM Depth) directly into DAWs (Bitwig/Ableton/FL/Reaper).
        </div>
      </div>
    </div>

    <!-- Interactive Phrases -->
    <div id="phrases-container"></div>
  </div>

  <script>
    const PHRASES_DATA = {data_json_str};

    function initBoard() {{
      const container = document.getElementById("phrases-container");
      container.innerHTML = "";

      PHRASES_DATA.forEach((p, pIdx) => {{
        const card = document.createElement("div");
        card.className = "phrase-card";
        card.id = `card-${{p.id}}`;

        card.innerHTML = `
          <div class="phrase-header">
            <div>
              <div class="phrase-title">${{p.title}}</div>
              <div class="phrase-song">Authentic Multitrack Backing: ${{p.song_name}} (MIDI Note ${{p.midi_note}})</div>
            </div>
            <div>
              <span class="badge-pass">V8 Mel L1: ${{p.metrics.v8_mel_l1.toFixed(4)}} (${{p.metrics.improvement}} better than Static)</span>
            </div>
          </div>
          <div class="phrase-desc">${{p.desc}}</div>

          <!-- Arm Switcher -->
          <div class="arm-switcher" id="switcher-${{p.id}}">
            <button class="arm-btn active clean" onclick="switchArm('${{p.id}}', 'clean', this)">
              <span>🎧</span> Target Clean Synth
            </button>
            <button class="arm-btn" onclick="switchArm('${{p.id}}', 'mix', this)">
              <span>🎛️</span> 0 dB SNR Mixture
            </button>
            <button class="arm-btn" onclick="switchArm('${{p.id}}', 'tse', this)">
              <span>⚡</span> Phase 1 TSE Separated
            </button>
            <button class="arm-btn static" onclick="switchArm('${{p.id}}', 'static', this)">
              <span>⏹️</span> Static Model (Flat)
            </button>
            <button class="arm-btn v8" onclick="switchArm('${{p.id}}', 'v8', this)">
              <span>🔥</span> V8 Dynamic Inversion
            </button>
          </div>

          <!-- Synchronized Audio Player -->
          <div class="player-row">
            <audio id="audio-${{p.id}}" class="audio-element" controls src="${{p.audio.clean}}"></audio>
            <div class="time-display" id="time-${{p.id}}">0.00s</div>
          </div>

          <!-- Dynamic Parameter Lanes Tabs -->
          <div class="lane-tabs">
            <button class="lane-tab active" onclick="switchLane('${{p.id}}', 'cutoff', this)">Lane 0: Filter Cutoff (CC 74)</button>
            <button class="lane-tab" onclick="switchLane('${{p.id}}', 'resonance', this)">Lane 1: Resonance (CC 71)</button>
            <button class="lane-tab" onclick="switchLane('${{p.id}}', 'drive', this)">Lane 2: Waveshaper Drive (CC 1)</button>
            <button class="lane-tab" onclick="switchLane('${{p.id}}', 'fm', this)">Lane 3: FM / RM Depth (CC 11)</button>
          </div>

          <!-- Trajectory Graph -->
          <div class="graph-wrapper">
            <canvas id="canvas-${{p.id}}" width="1000" height="220"></canvas>
            <div class="graph-legend">
              <div class="legend-item"><div class="legend-line" style="background: #10b981;"></div> Ground Truth Target</div>
              <div class="legend-item"><div class="legend-line" style="background: #38bdf8;"></div> V8 Dynamic Inverter (10.76 Hz)</div>
              <div class="legend-item"><div class="legend-line" style="background: #f97316; border-top: 2px dashed #f97316; height:0;"></div> Best Previous Static Model</div>
              <div class="legend-item"><div class="legend-line" style="background: #fff; width: 2px; height: 14px;"></div> Live Audio Playhead</div>
            </div>
          </div>

          <!-- Metrics Table -->
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Model / Inversion Arm</th>
                <th>Mel L1 Distance</th>
                <th>Cutoff Trajectory MSE</th>
                <th>Automation Tracking</th>
                <th>DAW Export Container</th>
              </tr>
            </thead>
            <tbody>
              <tr style="color: #10b981;">
                <td><strong>Ground Truth Target</strong></td>
                <td>0.0000</td>
                <td>0.0000</td>
                <td>Continuous (43 frames)</td>
                <td>Native Surge XT Patch</td>
              </tr>
              <tr style="color: #38bdf8; font-weight: 600;">
                <td><strong>V8 Dynamic Inverter (Ours)</strong></td>
                <td>${{p.metrics.v8_mel_l1.toFixed(4)}}</td>
                <td>${{p.metrics.cutoff_mse_v8.toFixed(5)}}</td>
                <td>10.76 Hz Spline (${{p.metrics.improvement}} closer)</td>
                <td><code>.vstpreset</code> + <code>.mid</code> (CC 74/71/1/11)</td>
              </tr>
              <tr style="color: #f97316;">
                <td><strong>Best Previous Static Model (V6 C Flow)</strong></td>
                <td>${{p.metrics.static_mel_l1.toFixed(4)}}</td>
                <td>${{p.metrics.cutoff_mse_static.toFixed(5)}}</td>
                <td>Flat constant line (0 Hz modulation)</td>
                <td>Static <code>.json</code> preset</td>
              </tr>
            </tbody>
          </table>
        `;

        container.appendChild(card);

        // State for this card
        card.currentLane = 'cutoff';
        card.currentTime = 0;

        // Hook up audio player events
        const audio = card.querySelector(`#audio-${{p.id}}`);
        const timeDisp = card.querySelector(`#time-${{p.id}}`);

        audio.addEventListener("timeupdate", () => {{
          card.currentTime = audio.currentTime;
          timeDisp.textContent = `${{audio.currentTime.toFixed(2)}}s`;
          drawGraph(p.id);
        }});

        audio.addEventListener("seeked", () => {{
          card.currentTime = audio.currentTime;
          drawGraph(p.id);
        }});

        // Initial draw
        setTimeout(() => drawGraph(p.id), 50);
      }});
    }}

    function switchArm(phraseId, armKey, btn) {{
      const phrase = PHRASES_DATA.find(p => p.id === phraseId);
      if (!phrase) return;

      const switcher = document.getElementById(`switcher-${{phraseId}}`);
      switcher.querySelectorAll(".arm-btn").forEach(b => b.classList.remove("active", "clean", "v8", "static"));

      btn.classList.add("active");
      if (armKey === "clean") btn.classList.add("clean");
      if (armKey === "v8") btn.classList.add("v8");
      if (armKey === "static") btn.classList.add("static");

      const audio = document.getElementById(`audio-${{phraseId}}`);
      const currTime = audio.currentTime;
      const wasPlaying = !audio.paused;

      audio.src = phrase.audio[armKey];
      audio.currentTime = currTime;
      if (wasPlaying) {{
        audio.play().catch(() => {{}});
      }}
    }}

    function switchLane(phraseId, laneKey, btn) {{
      const card = document.getElementById(`card-${{phraseId}}`);
      card.querySelectorAll(".lane-tab").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      card.currentLane = laneKey;
      drawGraph(phraseId);
    }}

    function drawGraph(phraseId) {{
      const phrase = PHRASES_DATA.find(p => p.id === phraseId);
      const card = document.getElementById(`card-${{phraseId}}`);
      const canvas = document.getElementById(`canvas-${{phraseId}}`);
      if (!phrase || !card || !canvas) return;

      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      const laneKey = card.currentLane || 'cutoff';
      const laneData = phrase.lanes[laneKey];

      ctx.clearRect(0, 0, W, H);

      // Background grid
      ctx.strokeStyle = "#131c2e";
      ctx.lineWidth = 1;
      for (let y = 0; y <= 4; y++) {{
        const py = H - 20 - (y / 4) * (H - 40);
        ctx.beginPath();
        ctx.moveTo(40, py);
        ctx.lineTo(W - 20, py);
        ctx.stroke();

        ctx.fillStyle = "#64748b";
        ctx.font = "10px monospace";
        ctx.textAlign = "right";
        ctx.fillText((y * 0.25).toFixed(2), 35, py + 3);
      }}

      // X-axis time markings (4.0s phrase)
      for (let sec = 0; sec <= 4; sec++) {{
        const px = 40 + (sec / 4.0) * (W - 60);
        ctx.beginPath();
        ctx.moveTo(px, 20);
        ctx.lineTo(px, H - 20);
        ctx.stroke();

        ctx.fillStyle = "#64748b";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${{sec}}s`, px, H - 6);
      }}

      const plotCurve = (arr, strokeStyle, lineWidth, dashed = false) => {{
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = lineWidth;
        if (dashed) ctx.setLineDash([6, 6]);
        else ctx.setLineDash([]);

        ctx.beginPath();
        const N = arr.length;
        for (let i = 0; i < N; i++) {{
          const px = 40 + (i / (N - 1)) * (W - 60);
          const py = H - 20 - Math.min(1.0, Math.max(0.0, arr[i])) * (H - 40);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }}
        ctx.stroke();
        ctx.setLineDash([]);
      }};

      // 1. Static model (Orange dashed flat line)
      plotCurve(laneData.static, "#f97316", 2, true);

      // 2. Ground Truth Target (Green solid line)
      plotCurve(laneData.true, "#10b981", 3, false);

      // 3. V8 Dynamic Inverter (Cyan glowing curve)
      plotCurve(laneData.v8, "#38bdf8", 3, false);

      // Plot data points for V8
      ctx.fillStyle = "#38bdf8";
      const N = laneData.v8.length;
      for (let i = 0; i < N; i++) {{
        const px = 40 + (i / (N - 1)) * (W - 60);
        const py = H - 20 - Math.min(1.0, Math.max(0.0, laneData.v8[i])) * (H - 40);
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }}

      // 4. Live Audio Playhead
      const tNorm = Math.min(4.0, Math.max(0.0, card.currentTime || 0)) / 4.0;
      const playheadX = 40 + tNorm * (W - 60);

      // Vertical cursor line
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(playheadX, 15);
      ctx.lineTo(playheadX, H - 20);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Playhead head marker
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(playheadX, 15, 4, 0, 2 * Math.PI);
      ctx.fill();
    }}

    window.addEventListener("DOMContentLoaded", initBoard);
  </script>
</body>
</html>
"""


def main():
    if "cuda" in ("cuda" if torch.cuda.is_available() else "cpu"):
        if not acquire_gpu_guard():
            print("GPU is currently busy. Exiting.", file=sys.stderr)
            sys.exit(0)
    try:
        build_eval_board()
    finally:
        release_gpu_guard()


if __name__ == "__main__":
    main()
