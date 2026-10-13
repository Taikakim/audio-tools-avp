# Owner: ANTIGRAVITY (created 2026-10-14)
# Last edited: ANTIGRAVITY 2026-10-14 -- URL-safe note filenames (Fs instead of F#) + --kind random/prior filter
"""Export random training clips from the V5 render bank for listening and sensibility review.

Samples clips from bank_v5_s0, writes 16-bit PCM WAV files with URL-safe filenames,
computes audio metrics (peak, RMS dB), decodes synth parameters to human-readable labels,
and generates an interactive HTML player page.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from render_bank_v5 import BankMainDataset  # noqa: E402
import surge_spec_v3 as spec  # noqa: E402

NOTE_NAMES_DISPLAY = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FILENAME = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def midi_to_names(midi: int) -> tuple[str, str]:
    octave = (midi // 12) - 1
    disp = f"{NOTE_NAMES_DISPLAY[midi % 12]}{octave}"
    fname = f"{NOTE_NAMES_FILENAME[midi % 12]}{octave}"
    return disp, fname


def decode_parameters(vec: np.ndarray) -> dict:
    f_idx = spec.ordinal_to_class(vec[spec.PARAM_INDEX["filter_type"]], len(spec.LP_FILTERS))
    ws_idx = spec.ordinal_to_class(vec[spec.PARAM_INDEX["waveshaper_type"]], len(spec.WAVESHAPER_TYPES))
    unison = bool(vec[spec.PARAM_INDEX["unison"]] > 0.5)

    return {
        "filter_type": spec.LP_FILTERS[f_idx][0],
        "cutoff": float(vec[spec.PARAM_INDEX["cutoff"]]),
        "resonance": float(vec[spec.PARAM_INDEX["resonance"]]),
        "keytrack": float(vec[spec.PARAM_INDEX["keytrack"]]),
        "waveshaper_type": spec.WAVESHAPER_TYPES[ws_idx][0],
        "drive": float(vec[spec.PARAM_INDEX["drive"]]),
        "shape": float(vec[spec.PARAM_INDEX["shape"]]),
        "width": float(vec[spec.PARAM_INDEX["width"]]),
        "sub_mix": float(vec[spec.PARAM_INDEX["sub_mix"]]),
        "sync": float(vec[spec.PARAM_INDEX["sync"]]),
        "fm_depth": float(vec[spec.PARAM_INDEX["fm_depth"]]),
        "unison": unison,
        "unison_voices": 2 if unison else 1,
        "unison_detune": float(vec[spec.PARAM_INDEX["unison_detune"]]) if unison else 0.0,
        "feg_amount": float(vec[spec.PARAM_INDEX["feg_amount"]]),
        "feg_decay": float(vec[spec.PARAM_INDEX["feg_decay"]]),
        "feg_sustain": float(vec[spec.PARAM_INDEX["feg_sustain"]]),
        "aeg_decay": float(vec[spec.PARAM_INDEX["aeg_decay"]]),
        "aeg_sustain": float(vec[spec.PARAM_INDEX["aeg_sustain"]]),
        "aeg_release": float(vec[spec.PARAM_INDEX["aeg_release"]]),
    }


def generate_html(clips: list, out_path: str, title: str):
    prior_count = sum(1 for c in clips if not c["is_random_patch"])
    random_count = len(clips) - prior_count
    avg_peak = np.mean([c["peak"] for c in clips]) if clips else 0.0
    avg_rms_db = np.mean([c["rms_db"] for c in clips]) if clips else 0.0

    rows = []
    for c in clips:
        p = c["params"]
        tag_badge = (
            '<span class="badge badge-random">Random Patch</span>'
            if c["is_random_patch"]
            else '<span class="badge badge-prior">Prior Manifold</span>'
        )
        ws_info = p["waveshaper_type"]
        if ws_info != "Off":
            ws_info += f" (+{p['drive']*15:.1f} dB)"
        unison_str = "2v" if p["unison"] else "1v"

        row = f"""
        <tr data-type="{'random' if c['is_random_patch'] else 'prior'}">
            <td class="num">#{c['index']:02d}</td>
            <td class="item-id">idx {c['global_id']}</td>
            <td><strong>{c['note_display']}</strong> ({c['midi']})</td>
            <td>{tag_badge}</td>
            <td class="player">
                <audio controls preload="none" src="wavs/{c['filename']}"></audio>
            </td>
            <td>{c['peak']:.3f} / {c['rms_db']:.1f} dB</td>
            <td>{p['filter_type']}</td>
            <td>cut {p['cutoff']:.2f} · res {p['resonance']:.2f}</td>
            <td>{ws_info}</td>
            <td>shape {p['shape']:.2f} · sub {p['sub_mix']:.2f}</td>
            <td>{unison_str}</td>
        </tr>
        """
        rows.append(row)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #121316;
        color: #e2e4e9;
        margin: 0;
        padding: 24px 32px;
        line-height: 1.5;
    }}
    h1 {{
        color: #f8fafc;
        margin-top: 0;
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.5px;
    }}
    .subtitle {{
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 24px;
    }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .stat-card {{
        background: #1c1e24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
    }}
    .stat-label {{
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stat-val {{
        font-size: 20px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }}
    .filter-bar {{
        margin-bottom: 16px;
        display: flex;
        gap: 10px;
        align-items: center;
    }}
    button {{
        background: #252830;
        border: 1px solid #3b404d;
        color: #e2e4e9;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        cursor: pointer;
    }}
    button.active {{
        background: #0284c7;
        border-color: #38bdf8;
        color: #fff;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: #181a20;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2a2e37;
        font-size: 13px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #242730;
    }}
    th {{
        background: #1f222a;
        color: #94a3b8;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
    }}
    tr:hover {{
        background: #20232c;
    }}
    .num {{
        color: #64748b;
        font-family: monospace;
    }}
    .item-id {{
        color: #8b949e;
        font-family: monospace;
        font-size: 12px;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }}
    .badge-prior {{
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }}
    .badge-random {{
        background: #4c1d95;
        color: #c4b5fd;
        border: 1px solid #7c3aed;
    }}
    audio {{
        height: 32px;
        width: 220px;
        outline: none;
    }}
</style>
</head>
<body>
    <h1>{title}</h1>
    <div class="subtitle">Exported from <code>bank_v5_s0</code> (786,432 items on OS NVMe, 44.1 kHz mono, Surge XT 1.3.4, FX off).</div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Reviewed</div>
            <div class="stat-val">{len(clips)} clips</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Prior Manifold</div>
            <div class="stat-val">{prior_count} ({prior_count / max(len(clips), 1):.0%})</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Random Patches</div>
            <div class="stat-val">{random_count} ({random_count / max(len(clips), 1):.0%})</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Mean Peak / RMS</div>
            <div class="stat-val">{avg_peak:.2f} / {avg_rms_db:.1f} dB</div>
        </div>
    </div>

    <div class="filter-bar">
        <span style="font-size: 13px; color: #8b949e;">Filter:</span>
        <button id="btn-all" class="active" onclick="filterClips('all')">All ({len(clips)})</button>
        <button id="btn-prior" onclick="filterClips('prior')">Prior Manifold ({prior_count})</button>
        <button id="btn-random" onclick="filterClips('random')">Random Patches ({random_count})</button>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Bank ID</th>
                <th>Note</th>
                <th>Type</th>
                <th>Audio Player</th>
                <th>Peak / RMS</th>
                <th>Filter Circuit</th>
                <th>Cutoff / Res</th>
                <th>Waveshaper</th>
                <th>Osc & Sub</th>
                <th>Voice</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>

    <script>
    function filterClips(type) {{
        document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
        if (type === 'all') document.getElementById('btn-all').classList.add('active');
        if (type === 'prior') document.getElementById('btn-prior').classList.add('active');
        if (type === 'random') document.getElementById('btn-random').classList.add('active');

        document.querySelectorAll('tbody tr').forEach(row => {{
            if (type === 'all' || row.getAttribute('data-type') === type) {{
                row.style.display = '';
            }} else {{
                row.style.display = 'none';
            }}
        }});
    }}
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="/home/kim/sao_runs/bank_v5_s0", help="path to bank directory")
    ap.add_argument("--out", default="/home/kim/sao_runs/bank_v5_s0_review_100", help="output directory")
    ap.add_argument("--n", type=int, default=100, help="number of clips to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling indices")
    ap.add_argument("--kind", choices=["all", "random", "prior"], default="all",
                    help="filter sample to only random patches, only prior presets, or all")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)

    print(f"Loading bank from {args.bank}...")
    ds = BankMainDataset(args.bank)
    total_items = len(ds)
    print(f"Bank has {total_items} items.")

    rng = np.random.default_rng(args.seed)

    if args.kind == "random":
        print("Scanning bank shards for unconstrained random patch items...")
        candidates = []
        n_shards = math.ceil(total_items / ds.shard)
        for k in range(n_shards):
            _, meta_shard = ds.sh.get(k)
            rp = meta_shard["random_patch"]
            shard_rp_indices = np.where(rp)[0]
            for j in shard_rp_indices:
                candidates.append(k * ds.shard + j)
            if len(candidates) >= args.n * 10:
                break
        print(f"Found {len(candidates)} candidate random patches.")
        chosen_indices = sorted(rng.choice(candidates, size=min(args.n, len(candidates)), replace=False))
        title = "V5 Training Bank — 100 Random Patches Sensibility Review"
    elif args.kind == "prior":
        print("Scanning bank shards for prior manifold items...")
        candidates = []
        n_shards = math.ceil(total_items / ds.shard)
        for k in range(n_shards):
            _, meta_shard = ds.sh.get(k)
            rp = meta_shard["random_patch"]
            shard_prior_indices = np.where(~rp)[0]
            for j in shard_prior_indices:
                candidates.append(k * ds.shard + j)
            if len(candidates) >= args.n * 10:
                break
        chosen_indices = sorted(rng.choice(candidates, size=min(args.n, len(candidates)), replace=False))
        title = "V5 Training Bank — 100 Prior Manifold Clips Review"
    else:
        chosen_indices = sorted(rng.choice(total_items, size=min(args.n, total_items), replace=False))
        title = "V5 Training Bank (bank_v5_s0) — 100 Random Clip Review"

    print(f"Exporting {len(chosen_indices)} clips to {args.out}...")
    clips_data = []

    for i, global_idx in enumerate(chosen_indices):
        k, j = divmod(int(global_idx), ds.shard)
        audio_shard, meta_shard = ds.sh.get(k)

        audio_i16 = audio_shard[j]
        audio_f32 = audio_i16.astype(np.float32) / 32767.0
        vec = meta_shard["vec"][j]
        midi = int(meta_shard["midi"][j])
        is_random = bool(meta_shard["random_patch"][j])
        note_disp, note_fname = midi_to_names(midi)

        # Audio statistics
        peak = float(np.max(np.abs(audio_f32)))
        rms = float(np.sqrt(np.mean(audio_f32**2)))
        rms_db = float(20 * np.log10(max(rms, 1e-9)))

        tag = "random" if is_random else "prior"
        fname = f"clip_{i:03d}_item{global_idx:06d}_{note_fname}_{tag}.wav"
        wav_path = os.path.join(args.out, "wavs", fname)

        sf.write(wav_path, audio_f32, spec.SAMPLE_RATE, subtype="PCM_16")

        params = decode_parameters(vec)

        clips_data.append(
            {
                "index": i,
                "global_id": int(global_idx),
                "filename": fname,
                "midi": midi,
                "note_display": note_disp,
                "note_filename": note_fname,
                "is_random_patch": is_random,
                "peak": peak,
                "rms_db": rms_db,
                "params": params,
                "vec": [round(float(x), 4) for x in vec],
            }
        )

    # Save manifest.json
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(clips_data, f, indent=2)

    # Save manifest.tsv
    tsv_path = os.path.join(args.out, "manifest.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("index\tglobal_id\tfilename\tmidi\tnote\ttype\tpeak\trms_db\tfilter\tcutoff\twaveshaper\tdrive\tunison\n")
        for c in clips_data:
            p = c["params"]
            t = "random" if c["is_random_patch"] else "prior"
            f.write(
                f"{c['index']}\t{c['global_id']}\t{c['filename']}\t{c['midi']}\t{c['note_display']}\t{t}\t"
                f"{c['peak']:.3f}\t{c['rms_db']:.1f}\t{p['filter_type']}\t{p['cutoff']:.2f}\t"
                f"{p['waveshaper_type']}\t{p['drive']:.2f}\t{p['unison_voices']}v\n"
            )

    # Save index.html
    html_path = os.path.join(args.out, "index.html")
    generate_html(clips_data, html_path, title=title)

    print(f"Successfully exported {len(clips_data)} clips:")
    print(f"  - WAV directory: {os.path.join(args.out, 'wavs')}")
    print(f"  - HTML review page: file://{html_path}")
    print(f"  - Manifest JSON: {manifest_path}")
    print(f"  - Manifest TSV: {tsv_path}")


if __name__ == "__main__":
    main()
