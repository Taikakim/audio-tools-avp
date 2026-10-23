#!/usr/bin/env python3
# Owner: KUANG / KIM / ANTIGRAVITY (created 2026-10-21)
"""Batch IDM 44.1 kHz 9-Class Drum Separation & Transcription Evaluation.

Decomposes both:
  1. All 24 JEPA bass evaluation stems (detecting percussive transient bleed & false positives)
  2. All 23 matched track drum mixes (real goa/psytrance drum kits in identical phrase alignment)
into 9 kit pieces on CPU, extracts symbolic onsets/velocities, and generates an interactive
HTML evaluation and audition page with solo/mute web audio players and onset timelines.

Usage:
  /home/kim/Projects/SAO/.venv/bin/python batch_idm_stems_eval.py --out_dir ~/sao_runs/idm_stems_eval
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Thread limit to preserve system stability
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

import numpy as np
import soundfile as sf
import librosa

# Include current directory and IDM repo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

IDM_ROOT = "/home/kim/Projects/inverse-drum-machine"
if IDM_ROOT not in sys.path:
    sys.path.insert(0, IDM_ROOT)

from idm.inference import load_model, separate
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu

DRUM_CLASSES = [
    "KD",       # Kick Drum
    "SD",       # Snare Drum
    "HH_CHH",   # Closed Hi-Hat
    "HH_OHH",   # Open Hi-Hat
    "CY_CR",    # Crash Cymbal
    "CY_RD",    # Ride Cymbal
    "TT_HFT",   # High Floor Tom
    "TT_HMT",   # High Mid Tom
    "TT_LMT",   # Low Mid Tom
]

DRUM_LABELS = {
    "KD": "Kick Drum",
    "SD": "Snare Drum",
    "HH_CHH": "Closed Hi-Hat",
    "HH_OHH": "Open Hi-Hat",
    "CY_CR": "Crash Cymbal",
    "CY_RD": "Ride Cymbal",
    "TT_HFT": "High Floor Tom",
    "TT_HMT": "High Mid Tom",
    "TT_LMT": "Low Mid Tom",
}

CLASS_COLORS = {
    "KD": "#ef4444",      # Red
    "SD": "#3b82f6",      # Blue
    "HH_CHH": "#10b981",  # Emerald green
    "HH_OHH": "#06b6d4",  # Cyan
    "CY_CR": "#f59e0b",   # Amber
    "CY_RD": "#8b5cf6",   # Purple
    "TT_HFT": "#ec4899",  # Pink
    "TT_HMT": "#d946ef",  # Fuchsia
    "TT_LMT": "#f97316",  # Orange
}


def pick_peaks(x, threshold=0.5):
    """Local maxima peak-picking across time dimension."""
    padded = torch.nn.functional.pad(x, (1, 1), mode='constant', value=0)
    return (x > padded[..., :-2]) & (x >= padded[..., 2:]) & (x > threshold)


def extract_symbolic_transcription(model, audio_tensor, threshold=0.5):
    """Extracts timestamped onsets and velocities from IDM encoder activations."""
    with torch.no_grad():
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        encoder_outs = model.encoder(audio_tensor.unsqueeze(0))
        activations = encoder_outs["activations"]
        onset_logits = activations["onset"]
        onsets = torch.sigmoid(onset_logits)
        peaks = pick_peaks(onsets, threshold=threshold)
        velocities = activations["velocity"]

        activation_rate = getattr(model.encoder, "frame_rate", 172.26)
        train_classes = getattr(model, "train_classes", DRUM_CLASSES)

        events = []
        B, K, T = peaks.shape
        for k_idx in range(K):
            class_name = train_classes[k_idx]
            peak_indices = torch.nonzero(peaks[0, k_idx] > 0.5).squeeze(-1).cpu().numpy()
            for idx in peak_indices:
                t_sec = float(idx / activation_rate)
                vel = float(velocities[0, k_idx, idx].cpu().numpy())
                conf = float(onsets[0, k_idx, idx].cpu().numpy())
                events.append({
                    "timestamp": round(t_sec, 4),
                    "class": class_name,
                    "label": DRUM_LABELS.get(class_name, class_name),
                    "velocity": round(vel, 3),
                    "confidence": round(conf, 3),
                })

        events.sort(key=lambda e: e["timestamp"])
        return events


def compute_stem_metrics(orig_audio, stems_dict, sr=44100):
    """Computes RMS energies and kit distribution metrics."""
    orig_rms = float(np.sqrt(np.mean(orig_audio**2))) + 1e-9
    stem_rms = {}
    stem_energy = {}
    total_stem_sq = 0.0

    for cls in DRUM_CLASSES:
        audio = stems_dict.get(cls, np.zeros_like(orig_audio))
        rms = float(np.sqrt(np.mean(audio**2)))
        stem_rms[cls] = round(rms, 5)
        sq = float(np.sum(audio**2))
        stem_energy[cls] = sq
        total_stem_sq += sq

    # Percentage energy
    pct_energy = {}
    for cls in DRUM_CLASSES:
        pct = (stem_energy[cls] / (total_stem_sq + 1e-9)) * 100.0
        pct_energy[cls] = round(pct, 2)

    return {
        "orig_rms": round(orig_rms, 5),
        "stem_rms": stem_rms,
        "pct_energy": pct_energy,
        "total_stem_sq": total_stem_sq,
    }


def process_audio_item(model, input_audio, sr, out_dir, threshold=0.5, alpha=1.0, masking="wiener"):
    """Separates audio into 9 stems, extracts onsets, and saves everything."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save original audio
    sf.write(out_path / "original.wav", input_audio.T if input_audio.ndim > 1 else input_audio, sr)

    # 1. Onsets
    audio_t = torch.from_numpy(input_audio).float()
    events = extract_symbolic_transcription(model, audio_t, threshold=threshold)

    # Save transcription files
    with open(out_path / "transcription.json", "w") as f:
        json.dump(events, f, indent=2)

    with open(out_path / "transcription.txt", "w") as f:
        f.write("# Timestamp(s)\tClass\tVelocity\tConfidence\tLabel\n")
        for e in events:
            f.write(f"{e['timestamp']:.4f}\t{e['class']}\t{e['velocity']:.3f}\t{e['confidence']:.3f}\t{e['label']}\n")

    # 2. Separation
    t0 = time.time()
    stems_tensor = separate(audio_t, model, output_dir=str(out_path), masking=masking, alpha=alpha, save_to_disk=True)
    sep_time = time.time() - t0

    # 3. Read back stems into dict and create recombined mix
    train_classes = getattr(model, "train_classes", DRUM_CLASSES)
    stems_dict = {}
    recombined = np.zeros_like(input_audio)
    for cls in train_classes:
        p = out_path / f"{cls}.wav"
        if p.exists():
            st, _ = sf.read(p, always_2d=True)
            st_t = st.T
            stems_dict[cls] = st_t
            recombined += st_t

    # Save recombined mix
    sf.write(out_path / "recombined.wav", recombined.T, sr)

    # 4. Metrics
    metrics = compute_stem_metrics(input_audio, stems_dict, sr=sr)
    metrics["separation_latency_s"] = round(sep_time, 2)
    metrics["onset_count"] = len(events)
    counts_by_class = {}
    for e in events:
        counts_by_class[e["class"]] = counts_by_class.get(e["class"], 0) + 1
    metrics["onsets_by_class"] = counts_by_class

    return events, metrics


def load_stem_drum_mix(item, summary_info):
    """Loads and cuts the matched drum mix for an evaluation track."""
    path = item["path"]
    bpm = item["bpm"]
    p_start = summary_info.get("phrase_start_s", 0.0)
    p_dur = summary_info.get("phrase_dur_s", 6.8)
    sr = 44100

    start_sample = int(round(p_start * sr))
    len_samples = int(round(p_dur * sr))

    if "avp-stems-original-classified" in path:
        parent = os.path.dirname(path)
        if os.path.basename(parent) in ["bass", "acid", "leads", "arp_keys", "pads", "sfx"]:
            track_dir = os.path.dirname(parent)
        else:
            track_dir = parent
        d_path = os.path.join(track_dir, "drums", "DRUMS_downmix.flac")
        k_path = os.path.join(track_dir, "kick", "KICK_downmix.flac")

        y_d = None
        y_k = None
        if os.path.exists(d_path):
            try:
                y_d, _ = sf.read(d_path, start=start_sample, stop=start_sample + len_samples, always_2d=True)
            except Exception:
                pass
        if os.path.exists(k_path):
            try:
                y_k, _ = sf.read(k_path, start=start_sample, stop=start_sample + len_samples, always_2d=True)
            except Exception:
                pass

        if y_d is not None and y_k is not None:
            min_l = min(len(y_d), len(y_k))
            y_mix = (y_d[:min_l] + y_k[:min_l]).T
        elif y_d is not None:
            y_mix = y_d.T
        elif y_k is not None:
            y_mix = y_k.T
        else:
            return None

        # Normalize headroom
        max_v = np.max(np.abs(y_mix))
        if max_v > 0.98:
            y_mix = (y_mix / max_v) * 0.95
        return y_mix.astype(np.float32)

    elif "multi_artist_benchmarks" in path:
        d_cand = path.replace("bass.wav", "drums.wav")
        if os.path.exists(d_cand):
            y_d, _ = sf.read(d_cand, start=start_sample, stop=start_sample + len_samples, always_2d=True)
            y_mix = y_d.T
            max_v = np.max(np.abs(y_mix))
            if max_v > 0.98:
                y_mix = (y_mix / max_v) * 0.95
            return y_mix.astype(np.float32)

    return None


def generate_html_eval_page(summary_data, out_html_path):
    """Generates a modern, responsive HTML page with interactive multi-stem audio players and onset visualizer."""
    jepa_entries = [d for d in summary_data if d["type"] == "jepa_stem"]
    drum_entries = [d for d in summary_data if d["type"] == "track_drum"]

    total_clips = len(summary_data)
    total_onsets = sum(d["metrics"]["onset_count"] for d in summary_data)
    avg_latency = float(np.mean([d["metrics"]["separation_latency_s"] for d in summary_data])) if summary_data else 0.0

    def render_track_card(d, rel_audio_base):
        sid = d["id"]
        name = d["name"]
        style = d["style"]
        bpm = d["bpm"]
        dur = d["duration_s"]
        metrics = d["metrics"]
        onsets = d["onsets"]
        t_type = d["type"]
        card_id = f"card_{t_type}_{sid}"

        folder = f"{rel_audio_base}/{sid}"
        orig_rel = f"{folder}/original.wav"
        recomb_rel = f"{folder}/recombined.wav"

        # Energy bars
        pct_bars = []
        for cls in DRUM_CLASSES:
            pct = metrics["pct_energy"].get(cls, 0.0)
            if pct > 0.5:
                color = CLASS_COLORS.get(cls, "#64748b")
                pct_bars.append(f"""
                  <div style="background:{color}; width:{pct}%; height:12px; border-radius:2px;" 
                       title="{cls} ({DRUM_LABELS.get(cls, cls)}): {pct:.1f}%"></div>
                """)

        # Stem rows with Solo / Mute buttons
        stem_rows = []
        for cls in DRUM_CLASSES:
            lbl = DRUM_LABELS.get(cls, cls)
            color = CLASS_COLORS.get(cls, "#64748b")
            pct = metrics["pct_energy"].get(cls, 0.0)
            rms = metrics["stem_rms"].get(cls, 0.0)
            hits = metrics["onsets_by_class"].get(cls, 0)
            stem_rel = f"{folder}/{cls}.wav"

            stem_rows.append(f"""
            <tr class="stem-row" data-cls="{cls}" style="border-bottom: 1px solid #1e293b;">
              <td style="padding: 6px 10px; font-weight: 600; color: {color}; width: 140px;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{color}; margin-right:6px;"></span>
                {cls} <span style="font-size:11px; font-weight:normal; color:#94a3b8;">({lbl})</span>
              </td>
              <td style="padding: 6px 8px; width: 75px;">
                <button class="sm-btn solo-btn" onclick="toggleSolo(this, '{card_id}', '{cls}')" title="Solo {cls}">S</button>
                <button class="sm-btn mute-btn" onclick="toggleMute(this, '{card_id}', '{cls}')" title="Mute {cls}">M</button>
              </td>
              <td style="padding: 6px 10px; width: 85px;">
                <span style="font-size: 11px; background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #cbd5e1;">
                  {hits} hits
                </span>
              </td>
              <td style="padding: 6px 10px; width: 120px;">
                <div style="background: #0f172a; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;">
                  <div style="background: {color}; height: 100%; width: {min(100.0, pct)}%;"></div>
                </div>
                <div style="font-size: 10px; color: #64748b; margin-top: 2px;">{pct:.1f}% ({rms:.4f} RMS)</div>
              </td>
              <td style="padding: 6px 10px;">
                <audio id="{card_id}_{cls}" class="stem-audio" controls preload="none" src="{stem_rel}" style="height: 28px; width: 100%; max-width: 440px;"></audio>
              </td>
            </tr>
            """)

        # Onset visualizer markers
        timeline_markers = []
        for e in onsets:
            pos_pct = (e["timestamp"] / (dur + 1e-7)) * 100.0
            pos_pct = min(100.0, max(0.0, pos_pct))
            cls = e["class"]
            col = CLASS_COLORS.get(cls, "#38bdf8")
            opacity = max(0.3, min(1.0, e["velocity"]))
            timeline_markers.append(f"""
              <div class="onset-pip" style="left: {pos_pct:.2f}%; background: {col}; opacity: {opacity:.2f};"
                   data-tip="{cls} @ {e['timestamp']:.2f}s (vel: {e['velocity']:.2f}, conf: {e['confidence']:.2f})"></div>
            """)

        # Onset table rows (first 25 rows)
        onset_table_rows = []
        for e in onsets[:25]:
            col = CLASS_COLORS.get(e["class"], "#cbd5e1")
            onset_table_rows.append(f"""
              <tr style="border-bottom: 1px solid #1e293b; font-size: 11px;">
                <td style="padding: 3px 8px; font-family: monospace;">{e['timestamp']:.3f}s</td>
                <td style="padding: 3px 8px; color: {col}; font-weight: bold;">{e['class']}</td>
                <td style="padding: 3px 8px;">{e['velocity']:.2f}</td>
                <td style="padding: 3px 8px;">{e['confidence']:.2f}</td>
              </tr>
            """)
        if len(onsets) > 25:
            onset_table_rows.append(f"""
              <tr style="font-size: 11px; color: #64748b;">
                <td colspan="4" style="padding: 4px 8px; text-align: center;">... and {len(onsets) - 25} more events ...</td>
              </tr>
            """)

        type_badge = (
            '<span style="background: #1e3a8a; color: #93c5fd; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">JEPA BASS STEM</span>'
            if t_type == "jepa_stem" else
            '<span style="background: #065f46; color: #6ee7b7; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">FULL DRUM MIX</span>'
        )

        return f"""
        <div id="{card_id}" class="track-card" style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 18px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
          
          <!-- Card Header -->
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; border-bottom: 1px solid #1e293b; padding-bottom: 12px; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {type_badge}
                <span style="font-size: 17px; font-weight: 700; color: #f8fafc;">{name}</span>
                <span style="font-size: 12px; color: #94a3b8;">({style})</span>
              </div>
              <div style="font-size: 12px; color: #64748b; font-family: monospace; margin-bottom: 6px;">
                ID: {sid} | BPM: {bpm:.1f} | Duration: {dur:.2f}s | Latency: {metrics['separation_latency_s']:.2f}s | Hits: {metrics['onset_count']}
              </div>
              <!-- Synced Multi-track Controls -->
              <div style="display: flex; gap: 8px; align-items: center;">
                <button class="action-btn" onclick="playCardSync('{card_id}')">&#9654; Play 9 Stems</button>
                <button class="action-btn" onclick="pauseCardSync('{card_id}')">&#10074;&#10074; Pause</button>
                <button class="action-btn" onclick="resetCardStems('{card_id}')">&#8634; Reset Solo/Mute</button>
              </div>
            </div>

            <!-- Master Players: Original vs Recombined -->
            <div style="display: flex; gap: 16px; align-items: center;">
              <div style="text-align: right;">
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 2px;">Original Audio:</div>
                <audio controls preload="none" src="{orig_rel}" style="height: 30px; width: 220px;"></audio>
              </div>
              <div style="text-align: right;">
                <div style="font-size: 11px; color: #38bdf8; margin-bottom: 2px;">Recombined Stems:</div>
                <audio controls preload="none" src="{recomb_rel}" style="height: 30px; width: 220px;"></audio>
              </div>
            </div>
          </div>

          <!-- Energy Distribution Bar -->
          <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px;">
              <span>Kit Energy Breakdown:</span>
              <span>{metrics['onset_count']} Detected Onsets</span>
            </div>
            <div style="display: flex; width: 100%; height: 12px; background: #020617; border-radius: 3px; overflow: hidden; gap: 1px;">
              {''.join(pct_bars)}
            </div>
          </div>

          <!-- Onset Timeline Visualizer -->
          <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-bottom: 4px;">
              <span>Symbolic Onset Timeline (click to seek):</span>
              <span>0.0s &rarr; {dur:.2f}s</span>
            </div>
            <div class="timeline-container" onclick="seekCard(event, '{card_id}', {dur:.2f})" 
                 style="position: relative; height: 26px; background: #020617; border-radius: 4px; border: 1px solid #1e293b; overflow: hidden; cursor: crosshair;">
              {''.join(timeline_markers)}
            </div>
          </div>

          <!-- 9-Stem Multitrack Table -->
          <div style="background: #090d16; border-radius: 6px; overflow: hidden; margin-bottom: 12px; border: 1px solid #1e293b;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
              <thead>
                <tr style="background: #020617; color: #64748b; border-bottom: 1px solid #1e293b; font-size: 11px;">
                  <th style="padding: 6px 10px;">Stem / Kit Piece</th>
                  <th style="padding: 6px 8px;">Solo / Mute</th>
                  <th style="padding: 6px 10px;">Hits</th>
                  <th style="padding: 6px 10px;">Energy Share</th>
                  <th style="padding: 6px 10px;">Audio Player</th>
                </tr>
              </thead>
              <tbody>
                {''.join(stem_rows)}
              </tbody>
            </table>
          </div>

          <!-- Collapsible Transcription Drawer -->
          <details style="margin-top: 8px;">
            <summary style="cursor: pointer; font-size: 12px; color: #38bdf8; user-select: none;">
              &blacktriangleright; View Symbolic Transcription Table ({len(onsets)} events)
            </summary>
            <div style="margin-top: 10px; max-height: 220px; overflow-y: auto; background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 8px;">
              <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                  <tr style="color: #64748b; font-size: 11px; border-bottom: 1px solid #1e293b;">
                    <th style="padding: 4px 8px;">Timestamp</th>
                    <th style="padding: 4px 8px;">Class</th>
                    <th style="padding: 4px 8px;">Velocity</th>
                    <th style="padding: 4px 8px;">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(onset_table_rows)}
                </tbody>
              </table>
            </div>
          </details>

        </div>
        """

    jepa_cards = [render_track_card(d, "../audio/jepa_stems") for d in jepa_entries]
    drum_cards = [render_track_card(d, "../audio/track_drums") for d in drum_entries]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IDM 44.1 kHz 9-Class Drum Separation & Transcription Benchmark</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: #020617;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
    }}
    h1 {{
      color: #f8fafc;
      font-size: 26px;
      margin-bottom: 6px;
      letter-spacing: -0.5px;
    }}
    p.lead {{
      color: #94a3b8;
      font-size: 14px;
      margin-top: 0;
      margin-bottom: 20px;
      line-height: 1.5;
    }}
    .stats-bar {{
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .stat-pill {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 13px;
    }}
    .stat-val {{
      font-weight: 700;
      color: #38bdf8;
      font-size: 16px;
    }}
    .tab-nav {{
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 10px;
      align-items: center;
    }}
    .tab-btn {{
      background: #0f172a;
      border: 1px solid #1e293b;
      color: #94a3b8;
      padding: 8px 18px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      transition: all 0.2s;
    }}
    .tab-btn:hover {{
      background: #1e293b;
      color: #f8fafc;
    }}
    .tab-btn.active {{
      background: #2563eb;
      color: #ffffff;
      border-color: #3b82f6;
    }}
    .timeline-container {{
      position: relative;
    }}
    .onset-pip {{
      position: absolute;
      top: 3px;
      bottom: 3px;
      width: 4px;
      border-radius: 2px;
      transform: translateX(-50%);
      transition: transform 0.1s;
      cursor: pointer;
    }}
    .onset-pip:hover {{
      transform: translateX(-50%) scaleY(1.3);
      z-index: 10;
    }}
    .onset-pip:hover::after {{
      content: attr(data-tip);
      position: absolute;
      bottom: 26px;
      left: 50%;
      transform: translateX(-50%);
      background: #000000;
      color: #ffffff;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 10px;
      white-space: nowrap;
      z-index: 20;
      border: 1px solid #334155;
    }}
    .search-box {{
      background: #0f172a;
      border: 1px solid #1e293b;
      color: #e2e8f0;
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      width: 260px;
      margin-left: auto;
    }}
    .sm-btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      font-weight: bold;
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 2px;
      transition: all 0.15s;
    }}
    .sm-btn:hover {{
      background: #334155;
      color: #f8fafc;
    }}
    .solo-btn.active-solo {{
      background: #eab308 !important;
      color: #000000 !important;
      border-color: #facc15 !important;
    }}
    .mute-btn.active-mute {{
      background: #ef4444 !important;
      color: #ffffff !important;
      border-color: #f87171 !important;
    }}
    .action-btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #e2e8f0;
      padding: 4px 10px;
      border-radius: 5px;
      font-size: 11px;
      cursor: pointer;
      font-weight: 600;
      transition: background 0.15s;
    }}
    .action-btn:hover {{
      background: #334155;
      color: #38bdf8;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>IDM 44.1 kHz 9-Class Drum Separation & Transcription Benchmark</h1>
    <p class="lead">
      Evaluation of Inverse Drum Machine (Torres et al., IEEE TASLP 2025) across real studio stems.
      Separates full 44.1 kHz audio into 9 drum stems (Kick, Snare, Closed Hat, Open Hat, Crash, Ride, Toms)
      and extracts frame-accurate symbolic onsets and velocities on CPU.
    </p>

    <div class="stats-bar">
      <div class="stat-pill">Total Audio Sets: <span class="stat-val">{total_clips}</span></div>
      <div class="stat-pill">Total Onsets Transcribed: <span class="stat-val">{total_onsets}</span></div>
      <div class="stat-pill">Avg Separation Latency: <span class="stat-val">{avg_latency:.2f}s</span></div>
      <div class="stat-pill">Device: <span class="stat-val">CPU (4 threads)</span></div>
      <div class="stat-pill">Sample Rate: <span class="stat-val">44.1 kHz</span></div>
    </div>

    <!-- Tab navigation -->
    <div class="tab-nav">
      <button class="tab-btn active" onclick="switchTab('drums')">Real Drum Kits ({len(drum_entries)} Tracks)</button>
      <button class="tab-btn" onclick="switchTab('jepa')">JEPA Bass Stems & Bleed ({len(jepa_entries)} Tracks)</button>
      <input type="text" id="filterInput" class="search-box" placeholder="Filter by track or style..." onkeyup="filterCards()">
    </div>

    <!-- Tab 1: Real Drum Kits -->
    <div id="tab-drums" class="tab-pane">
      {''.join(drum_cards)}
    </div>

    <!-- Tab 2: JEPA Bass Stems -->
    <div id="tab-jepa" class="tab-pane" style="display: none;">
      {''.join(jepa_cards)}
    </div>

  </div>

  <script>
    function switchTab(tabId) {{
      document.getElementById('tab-drums').style.display = tabId === 'drums' ? 'block' : 'none';
      document.getElementById('tab-jepa').style.display = tabId === 'jepa' ? 'block' : 'none';
      const btns = document.querySelectorAll('.tab-btn');
      btns[0].classList.toggle('active', tabId === 'drums');
      btns[1].classList.toggle('active', tabId === 'jepa');
      filterCards();
    }}

    function filterCards() {{
      const query = document.getElementById('filterInput').value.toLowerCase();
      const activePane = document.querySelector('.tab-pane[style*="block"]') || document.getElementById('tab-drums');
      const cards = activePane.querySelectorAll('.track-card');
      cards.forEach(card => {{
        const text = card.innerText.toLowerCase();
        card.style.display = text.includes(query) ? '' : 'none';
      }});
    }}

    function playCardSync(cardId) {{
      const card = document.getElementById(cardId);
      const audios = card.querySelectorAll('audio.stem-audio');
      audios.forEach(a => {{
        a.currentTime = 0;
        a.play().catch(e => {{}});
      }});
    }}

    function pauseCardSync(cardId) {{
      const card = document.getElementById(cardId);
      card.querySelectorAll('audio.stem-audio').forEach(a => a.pause());
    }}

    function toggleMute(btn, cardId, cls) {{
      const audio = document.getElementById(cardId + '_' + cls);
      audio.muted = !audio.muted;
      btn.classList.toggle('active-mute', audio.muted);
    }}

    function toggleSolo(btn, cardId, cls) {{
      const card = document.getElementById(cardId);
      const audios = card.querySelectorAll('audio.stem-audio');
      const targetAudio = document.getElementById(cardId + '_' + cls);
      const isSolo = btn.classList.contains('active-solo');
      if (isSolo) {{
        btn.classList.remove('active-solo');
        audios.forEach(a => a.muted = false);
        card.querySelectorAll('.mute-btn').forEach(b => b.classList.remove('active-mute'));
      }} else {{
        card.querySelectorAll('.solo-btn').forEach(b => b.classList.remove('active-solo'));
        btn.classList.add('active-solo');
        audios.forEach(a => {{
          a.muted = (a !== targetAudio);
        }});
        card.querySelectorAll('.stem-row').forEach(row => {{
          const rowCls = row.getAttribute('data-cls');
          const muteBtn = row.querySelector('.mute-btn');
          if (muteBtn) muteBtn.classList.toggle('active-mute', rowCls !== cls);
        }});
      }}
    }}

    function resetCardStems(cardId) {{
      const card = document.getElementById(cardId);
      card.querySelectorAll('audio.stem-audio').forEach(a => {{
        a.muted = false;
      }});
      card.querySelectorAll('.solo-btn').forEach(b => b.classList.remove('active-solo'));
      card.querySelectorAll('.mute-btn').forEach(b => b.classList.remove('active-mute'));
    }}

    function seekCard(e, cardId, totalDur) {{
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / rect.width));
      const targetTime = pct * totalDur;
      const card = document.getElementById(cardId);
      card.querySelectorAll('audio').forEach(a => {{
        a.currentTime = targetTime;
      }});
    }}
  </script>
</body>
</html>
"""
    with open(out_html_path, "w") as f:
        f.write(html_content)
    print(f"\n[OK] Interactive evaluation page written to -> {out_html_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Batch IDM Drum Separation & Transcription across Real Stems")
    ap.add_argument("--out_dir", default="/home/kim/sao_runs/idm_stems_eval", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5, help="Onset detection threshold")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    ap.add_argument("--masking", default="wiener", help="Separation masking (wiener or none)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Wiener exponent")
    ap.add_argument("--skip_existing", action="store_true", help="Skip stems already separated")
    ap.add_argument("--max_stems", type=int, default=None, help="Optional limit on stems to process")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    audio_base = out_base / "audio"
    listen_base = out_base / "listen"
    jepa_audio_dir = audio_base / "jepa_stems"
    drums_audio_dir = audio_base / "track_drums"

    out_base.mkdir(parents=True, exist_ok=True)
    jepa_audio_dir.mkdir(parents=True, exist_ok=True)
    drums_audio_dir.mkdir(parents=True, exist_ok=True)
    listen_base.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("BATCH IDM 44.1 kHz 9-CLASS DRUM SEPARATION & TRANSCRIPTION BENCHMARK")
    print(f"Output Directory: {args.out_dir}")
    print(f"Device:           {args.device}")
    print(f"Masking / Alpha:  {args.masking} / {args.alpha}")
    print(f"Threshold:        {args.threshold}")
    print("================================================================================")

    # Load summary json to get exact phrase windows
    summary_path = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval/real_stems_summary.json"
    stem_summary_map = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for item in json.load(f):
                stem_summary_map[item["id"]] = item

    # Load IDM model once
    t0 = time.time()
    print("Loading pretrained IDM model ('idm-44-train-kits')...", flush=True)
    model, model_name = load_model("idm-44-train-kits", device=args.device, log_dir=os.path.join(IDM_ROOT, "pretrained"))
    print(f"Model loaded in {time.time() - t0:.2f}s.\n", flush=True)

    catalog = STEM_CATALOG
    if args.max_stems:
        catalog = catalog[:args.max_stems]

    all_records = []
    sr = 44100

    # -------------------------------------------------------------------------
    # PART 1: Process Matched Track Drum Mixes (Real Drums)
    # -------------------------------------------------------------------------
    print("--- [PART 1/2] Processing Matched Track Drum Mixes ---", flush=True)
    for idx, item in enumerate(catalog):
        sid = item["id"]
        name = item["name"]
        style = item["style"]
        bpm = item["bpm"]
        s_info = stem_summary_map.get(sid, {})
        p_dur = s_info.get("phrase_dur_s", 6.8)

        track_out_dir = drums_audio_dir / sid
        summary_rec_path = track_out_dir / "meta.json"

        if args.skip_existing and summary_rec_path.exists():
            print(f"[{idx+1:02d}/{len(catalog)}] Skipping existing drum mix for {sid}...")
            with open(summary_rec_path) as f:
                all_records.append(json.load(f))
            continue

        drum_mix = load_stem_drum_mix(item, s_info)
        if drum_mix is None:
            print(f"[{idx+1:02d}/{len(catalog)}] No drum mix available for {sid}, skipping.")
            continue

        print(f"[{idx+1:02d}/{len(catalog)}] Separating Drum Mix: {sid} ({name}) | {drum_mix.shape[1]/sr:.2f}s...", flush=True)
        t_start = time.time()
        events, metrics = process_audio_item(
            model=model,
            input_audio=drum_mix,
            sr=sr,
            out_dir=track_out_dir,
            threshold=args.threshold,
            alpha=args.alpha,
            masking=args.masking,
        )
        dur_s = float(drum_mix.shape[1] / sr)
        print(f"    Done in {time.time() - t_start:.2f}s | {len(events)} onsets | KD: {metrics['onsets_by_class'].get('KD',0)}, SD: {metrics['onsets_by_class'].get('SD',0)}, HH: {metrics['onsets_by_class'].get('HH_CHH',0)}")

        record = {
            "id": sid,
            "name": name,
            "style": style,
            "bpm": bpm,
            "duration_s": round(dur_s, 2),
            "type": "track_drum",
            "metrics": metrics,
            "onsets": events,
        }
        with open(summary_rec_path, "w") as f:
            json.dump(record, f, indent=2)
        all_records.append(record)

    # -------------------------------------------------------------------------
    # PART 2: Process JEPA Bass Stems (Percussive Transient & Bleed Isolation)
    # -------------------------------------------------------------------------
    print("\n--- [PART 2/2] Processing JEPA Bass Stems ---", flush=True)
    real_audio_dir = resolve_mantu("/run/media/kim/Mantu/surge_200k_models/real_stems_eval/audio")
    for idx, item in enumerate(catalog):
        sid = item["id"]
        name = item["name"]
        style = item["style"]
        bpm = item["bpm"]
        real_wav = os.path.join(real_audio_dir, f"{sid}_real.wav")

        if not os.path.exists(real_wav):
            print(f"[{idx+1:02d}/{len(catalog)}] Missing real_wav: {real_wav}, skipping.")
            continue

        track_out_dir = jepa_audio_dir / sid
        summary_rec_path = track_out_dir / "meta.json"

        if args.skip_existing and summary_rec_path.exists():
            print(f"[{idx+1:02d}/{len(catalog)}] Skipping existing JEPA stem for {sid}...")
            with open(summary_rec_path) as f:
                all_records.append(json.load(f))
            continue

        y_real, real_sr = sf.read(real_wav, always_2d=True)
        if real_sr != sr:
            y_real = librosa.resample(y_real.T, orig_sr=real_sr, target_sr=sr).T
        y_real_t = y_real.T.astype(np.float32)

        dur_s = float(y_real_t.shape[1] / sr)
        print(f"[{idx+1:02d}/{len(catalog)}] Separating JEPA Stem: {sid} ({name}) | {dur_s:.2f}s...", flush=True)
        t_start = time.time()
        events, metrics = process_audio_item(
            model=model,
            input_audio=y_real_t,
            sr=sr,
            out_dir=track_out_dir,
            threshold=args.threshold,
            alpha=args.alpha,
            masking=args.masking,
        )
        print(f"    Done in {time.time() - t_start:.2f}s | {len(events)} onsets | KD: {metrics['onsets_by_class'].get('KD',0)}, SD: {metrics['onsets_by_class'].get('SD',0)}, HH: {metrics['onsets_by_class'].get('HH_CHH',0)}")

        record = {
            "id": sid,
            "name": name,
            "style": style,
            "bpm": bpm,
            "duration_s": round(dur_s, 2),
            "type": "jepa_stem",
            "metrics": metrics,
            "onsets": events,
        }
        with open(summary_rec_path, "w") as f:
            json.dump(record, f, indent=2)
        all_records.append(record)

    # -------------------------------------------------------------------------
    # PART 3: Summary JSON and Interactive HTML Page
    # -------------------------------------------------------------------------
    summary_file = out_base / "idm_eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n[OK] Complete evaluation summary saved to -> {summary_file}")

    html_file = listen_base / "index.html"
    generate_html_eval_page(all_records, html_file)


if __name__ == "__main__":
    main()
