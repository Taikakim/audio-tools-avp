#!/usr/bin/env python3
# Owner: KIM / KUANG / ANTIGRAVITY (created 2026-10-20)
"""Renderer-in-the-loop CPU refinement on the 24 real bass stems.

Applies coordinate ladder bracketing (refine.py) on the winning patches from the V6 foundation
benchmark (consensus_eval_summary.json), tuning salient continuous axes (cutoff, resonance,
feg_amount, feg_decay, aeg_decay, aeg_release) against lowpassed (<=4kHz) real stems.

Runs entirely on CPU in ~1-2 minutes, without touching the GPU or interfering with training.
"""
import argparse
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_real_stems_consensus_v6 as E
import surge_spec_v5 as spec_v5
import audio_metrics_v6 as am6

DEFAULT_AXES = [
    ("cutoff", 0.12),
    ("resonance", 0.15),
    ("feg_amount", 0.15),
    ("feg_decay", 0.15),
    ("aeg_decay", 0.15),
    ("aeg_release", 0.15),
]


def score_patch(synth, patch, events, dur, sr, real_lp, max_eval_freq=4000.0):
    spec_v5.apply_patch(synth, patch)
    out = synth.process(events, dur, sample_rate=sr, num_channels=2)
    mono = out[0]
    peak = np.max(np.abs(mono))
    if peak > 1e-7:
        mono = mono / peak
    mono_lp = am6.lowpass(mono, sr, max_eval_freq)
    dists = am6.band_dists(mono_lp, real_lp, sr)
    # Primary objective: body band distance + 0.5 * sub + 0.1 * air
    obj = float(dists["body"] + 0.5 * dists["sub"] + 0.1 * dists["air"])
    return obj, out, dists


def refine_patch_bracket(synth, initial_patch, events, dur, sr, real_lp, axes=DEFAULT_AXES,
                         n_coarse=7, n_fine=4, max_eval_freq=4000.0):
    cur_patch = dict(initial_patch)
    cur_score, cur_out, cur_dists = score_patch(synth, cur_patch, events, dur, sr, real_lp, max_eval_freq)
    trace = []
    total_renders = 1

    for key, half in axes:
        x = float(cur_patch.get(key, 0.5))
        lo = max(0.0, x - half)
        hi = min(1.0, x + half)
        grid = np.linspace(lo, hi, n_coarse)
        best_v = x
        best_s = cur_score

        # Coarse pass
        for v in grid:
            cand = dict(cur_patch, **{key: float(v)})
            s, _, _ = score_patch(synth, cand, events, dur, sr, real_lp, max_eval_freq)
            total_renders += 1
            if s < best_s:
                best_s = s
                best_v = float(v)

        # Fine bracket pass around the coarse winner
        idx = int(np.argmin(np.abs(grid - best_v)))
        flo = grid[max(0, idx - 1)]
        fhi = grid[min(len(grid) - 1, idx + 1)]
        if fhi > flo:
            for v in np.linspace(flo, fhi, n_fine + 2)[1:-1]:
                cand = dict(cur_patch, **{key: float(v)})
                s, _, _ = score_patch(synth, cand, events, dur, sr, real_lp, max_eval_freq)
                total_renders += 1
                if s < best_s:
                    best_s = s
                    best_v = float(v)

        if best_s < cur_score - 1e-4:
            trace.append((key, x, best_v, cur_score, best_s))
            cur_patch[key] = best_v
            cur_score = best_s

    # Final render with updated patch
    final_score, final_out, final_dists = score_patch(synth, cur_patch, events, dur, sr, real_lp, max_eval_freq)
    return cur_patch, final_score, final_out, final_dists, total_renders, trace


def build_refine_player_page(results, out_dir):
    """Build interactive HTML player comparing Real vs Raw vs Refined."""
    listen_dir = os.path.join(out_dir, "listen")
    os.makedirs(listen_dir, exist_ok=True)
    html_path = os.path.join(listen_dir, "index.html")

    cards = []
    for r in results:
        stem_id = r["stem_id"]
        name = r["name"]
        bpm = r["bpm"]
        style = r["style"]

        model_rows = []
        for m in r["models"]:
            m_name = m["name"]
            raw_s = m["raw_score"]
            ref_s = m["refined_score"]
            delta = ref_s - raw_s
            pct = (delta / (raw_s + 1e-6)) * 100
            col = "#10b981" if delta < -0.01 else "#64748b"

            raw_wav = f"{stem_id}__{m_name}_raw.wav"
            ref_wav = f"{stem_id}__{m_name}_refined.wav"

            trace_str = ", ".join([f"{t[0]}: {t[1]:.2f}→{t[2]:.2f}" for t in m["trace"]]) or "no change"

            model_rows.append(f"""
            <tr style="border-bottom: 1px solid #1e293b;">
              <td style="padding: 10px; font-weight: 600; color: #f8fafc;">{m_name}</td>
              <td style="padding: 10px;">
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px;">Raw ({raw_s:.2f})</div>
                <audio controls src="{raw_wav}" style="height: 32px; width: 220px;"></audio>
              </td>
              <td style="padding: 10px;">
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px;">Refined ({ref_s:.2f})</div>
                <audio controls src="{ref_wav}" style="height: 32px; width: 220px;"></audio>
              </td>
              <td style="padding: 10px; color: {col}; font-weight: 700;">{delta:+.2f} ({pct:+.1f}%)</td>
              <td style="padding: 10px; font-size: 12px; color: #cbd5e1; font-family: monospace;">{trace_str}</td>
            </tr>
            """)

        real_wav = f"{stem_id}__real.wav"
        cards.append(f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #334155; padding-bottom: 8px;">
            <div>
              <span style="font-size: 16px; font-weight: bold; color: #38bdf8;">{name}</span>
              <span style="font-size: 12px; color: #94a3b8; margin-left: 8px;">({style} | {bpm:.1f} BPM)</span>
            </div>
            <div>
              <span style="font-size: 12px; color: #94a3b8; margin-right: 8px;">Real Stem:</span>
              <audio controls src="{real_wav}" style="height: 32px; vertical-align: middle;"></audio>
            </div>
          </div>
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
            <thead>
              <tr style="color: #64748b; border-bottom: 1px solid #334155;">
                <th style="padding: 8px;">Model</th>
                <th style="padding: 8px;">Initial / Raw</th>
                <th style="padding: 8px;">Refined (Coordinate Ladder)</th>
                <th style="padding: 8px;">Delta</th>
                <th style="padding: 8px;">Tuned Parameters</th>
              </tr>
            </thead>
            <tbody>
              {''.join(model_rows)}
            </tbody>
          </table>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>V6 CPU Refinement Benchmark (24 Real Stems)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #f8fafc; margin-bottom: 4px; }}
    p.lead {{ color: #94a3b8; margin-top: 0; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>V6 CPU Coordinate-Search Refinement</h1>
    <p class="lead">Renderer-in-the-loop ladder search on 6 salient axes (cutoff, resonance, envelopes) scored against &le;4 kHz band distance.</p>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Wrote interactive player to {html_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run renderer-in-the-loop CPU refinement on 24 real stems")
    ap.add_argument("--summary", default="/home/kim/sao_runs/stems_v6_foundation_occam/consensus_eval_summary.json")
    ap.add_argument("--out_dir", default="/home/kim/sao_runs/stems_v6_foundation_refined")
    ap.add_argument("--models", nargs="+", default=["V6_C", "DPO_b300"])
    ap.add_argument("--max_eval_freq", type=float, default=4000.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    listen_dir = os.path.join(args.out_dir, "listen")
    os.makedirs(listen_dir, exist_ok=True)

    print(f"Loading summary from {args.summary}...")
    summary = json.load(open(args.summary))
    E_dir = E.resolve_mantu("/run/media/kim/Mantu/surge_200k_models/real_stems_eval")
    synth = spec_v5.init_synth(spec_v5.DEFAULT_PLUGIN_PATH)

    results = []
    t_start = time.time()

    for idx, item in enumerate(summary):
        stem_id = item["stem_id"]
        name = item["name"]
        bpm = item["bpm"]
        style = item["style"]

        print(f"\n[{idx+1:02d}/24] Refining {name} ({stem_id})...", flush=True)
        real_wav_path = os.path.join(E_dir, "audio", f"{stem_id}_real.wav")
        midi_path = os.path.join(E_dir, "midi_muscriptor", f"{stem_id}_muscriptor.mid")
        midi_fix = os.path.join(E_dir, "midi_muscriptor", f"{stem_id}_muscriptor_FIX.mid")
        if os.path.exists(midi_fix):
            midi_path = midi_fix

        if not os.path.exists(real_wav_path) or not os.path.exists(midi_path):
            print(f"  Missing files, skipping {stem_id}")
            continue

        real_audio, sr = sf.read(real_wav_path)
        if real_audio.ndim == 2:
            real_audio_mono = real_audio[:, 0]
        else:
            real_audio_mono = real_audio

        notes = E.extract_notes_from_midi(midi_path)
        dur = min(3.0, len(real_audio_mono) / sr)
        events = E.events_from_notes(notes, dur)
        real_crop = real_audio_mono[:int(round(dur * sr))].astype(np.float32)
        real_peak = np.max(np.abs(real_crop))
        if real_peak > 1e-7:
            real_crop = real_crop / real_peak
        real_lp = am6.lowpass(real_crop, sr, args.max_eval_freq)

        # Save reference real audio
        sf.write(os.path.join(listen_dir, f"{stem_id}__real.wav"), real_audio[:int(round(dur * sr))], sr)

        stem_res = {
            "stem_id": stem_id,
            "name": name,
            "bpm": bpm,
            "style": style,
            "models": []
        }

        for m_name in args.models:
            if m_name not in item["models"]:
                continue
            m_data = item["models"][m_name]
            init_patch = dict(m_data["best_patch"])

            # 1. Render initial raw patch
            spec_v5.apply_patch(synth, init_patch)
            raw_audio = synth.process(events, dur, sample_rate=sr, num_channels=2)
            raw_score, _, raw_dists = score_patch(synth, init_patch, events, dur, sr, real_lp, args.max_eval_freq)
            sf.write(os.path.join(listen_dir, f"{stem_id}__{m_name}_raw.wav"), raw_audio.T, sr)

            # 2. Refine patch
            ref_patch, ref_score, ref_audio, ref_dists, n_renders, trace = refine_patch_bracket(
                synth, init_patch, events, dur, sr, real_lp, max_eval_freq=args.max_eval_freq
            )
            sf.write(os.path.join(listen_dir, f"{stem_id}__{m_name}_refined.wav"), ref_audio.T, sr)

            delta = ref_score - raw_score
            delta_str = f"{delta:+.3f}" if delta < -0.001 else "0.000"
            print(f"  {m_name:10s}: {raw_score:6.2f} -> {ref_score:6.2f} (Δ {delta_str}) in {n_renders} renders", flush=True)

            stem_res["models"].append({
                "name": m_name,
                "raw_score": raw_score,
                "refined_score": ref_score,
                "raw_dists": raw_dists,
                "refined_dists": ref_dists,
                "refined_patch": ref_patch,
                "trace": trace
            })

        results.append(stem_res)

    total_time = time.time() - t_start
    print(f"\nRefinement completed in {total_time:.1f}s ({total_time/60:.2f}m)!", flush=True)

    # Save summary json
    summary_out = os.path.join(args.out_dir, "refine_summary.json")
    with open(summary_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_out}")

    # Build interactive player
    build_refine_player_page(results, args.out_dir)


if __name__ == "__main__":
    main()
