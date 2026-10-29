#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 1: Asymmetric Target Sound Extractor Evaluation & Separation Tax Gate.

Evaluates trained Asymmetric TSE (tse_best.pt) vs Demucs baseline on held-out songs:
1. Draws held-out test patches (default: fresh renders from the val split of the V6 prior).
2. Mixes each tiled 0.8 s phrase with authentic held-out stems (kicks, drums, synths) at 0 dB SNR.
3. Extracts with the V8 Asymmetric TSE, cued by 1.0 s of the same patch rendered at another note.
4. Separates with Demucs (htdemucs).
5. Inverts continuous parameters with the V6 C flow inverter (several draws per item) across:
   - (a) Clean Ground Truth (Ceiling)
   - (b) V8 TSE Extracted
   - (c) Demucs Separated (Baseline)
   - (d) Raw Mixture (Floor)
6. Computes Mel-SSIM distance, parameter MSE, cutoff error, and the Separation Tax, with a paired bootstrap interval
   for TSE minus Demucs.
7. Gate as implemented: Separation Tax (TSE parameter MSE / clean parameter MSE) <= 1.25. The spec words it as
   "Mel-SSIM error <= 1.25x of clean stem"; the owner should settle which one is registered.

KUANG 2026-10-29 (docs/synth-v8-code-review-2026-10-29.md, C4 / C5):
- The cue used to be the first 1.0 s of the clean target, which contains the scored 0.8 s: the extractor was given
  the answer. It is now the same patch at another note (as at use time, where the cue comes from elsewhere).
- The 20 test patches were bank_v6 shard 0 items 0-19, which the TSE trained on (and V6-C too if it trained on
  bank_v6). Default now: fresh val-prior renders (--patch-source prior with the two manifolds); --patch-source bank
  uses another shard (--bank-shard, default 1), which the TSE never sampled.
- One flow draw per item became --draws (default 4); per-item rows are saved.
- Still missing from the registered design: BS-RoFormer and the matched two-stem model as baselines.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from demucs.apply import apply_model
from demucs.pretrained import get_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics_v5 import mel_ssim_dist, octave_db_l1
from audio_utils import ExactGpuMel
from models_v4 import load_inverter_v4
import param_codec_v6 as codec
import surge_spec_v5 as S5
import surge_spec_v6 as S6
from surge_spec_v6 import FLOW_CONT_INDICES, PARAM_INDEX
from train_v8_asymmetric_tse import AsymmetricTSEModel

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"
CONDITIONS = ("clean", "tse", "demucs", "mix")
CLIP_SAMPLES = 176400  # 4.0 s
CUE_SAMPLES = 44100    # 1.0 s
SLICE = 35280          # the 0.8 s V6-C reads


def acquire_gpu_guard() -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
    return res.returncode == 0


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def cue_note(midi: int) -> int:
    """Another note for the cue: a fourth up, or down when that leaves the range."""
    return midi + 5 if midi + 5 <= S6.NOTE_HIGH else midi - 5


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of a - b with a 95 % percentile bootstrap interval over items."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.RandomState(seed)
    boots = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_items(args, plugin) -> List[Dict]:
    """Held-out test patches: [{vec [59], midi, target [0.8 s], cue [1.0 s], source}]."""
    items = []
    if args.patch_source == "prior":
        if not (args.bass_manifold and args.all_manifold):
            sys.exit("--patch-source prior needs --bass-manifold and --all-manifold (the V6 prior's npz files)")
        from soundset_prior_v6 import SoundsetPriorV6
        prior = SoundsetPriorV6(args.bass_manifold, args.all_manifold, split="val")
        rng = np.random.RandomState(args.seed)
        for _ in range(args.n_items):
            vec, midi, source = prior.sample_vector(rng)
            patch = S6.vector_to_patch(vec)
            target = S6.render_patch(plugin, patch, midi)
            items.append({"vec": vec, "midi": int(midi), "target": target, "patch": patch, "source": source})
    else:
        shard = f"main_{args.bank_shard:05d}"
        if args.bank_shard == 0:
            print("Warning: shard 0 is the shard the TSE trained on.", file=sys.stderr)
        print("Warning: if V6-C trained on bank_v6, these patches are V6-C training items (check its run_meta).",
              file=sys.stderr)
        audio = np.load(os.path.join(args.bank_dir, f"{shard}_audio.npy"), mmap_mode="r")
        meta = np.load(os.path.join(args.bank_dir, f"{shard}_meta.npz"))
        for i in range(args.n_items):
            vec = meta["vec"][i].astype(np.float32)
            items.append({"vec": vec, "midi": int(meta["midi"][i]), "target": audio[i].astype(np.float32) / 32768.0,
                          "patch": S6.vector_to_patch(vec), "source": f"{shard}[{i}]"})
    for it in items:
        it["cue"] = S6.render_patch(plugin, it["patch"], cue_note(it["midi"]), duration=CUE_SAMPLES / 44100)[:CUE_SAMPLES]
    return items


def run_eval(device: str, args):
    # 1. Load V8 Asymmetric TSE Model
    if not os.path.exists(args.tse_ckpt):
        sys.exit(f"Error: TSE checkpoint not found at {args.tse_ckpt}")
    tse_model = AsymmetricTSEModel().to(device).eval()
    ckpt = torch.load(args.tse_ckpt, map_location=device, weights_only=False)
    tse_model.load_state_dict(ckpt["model"])
    print(f"Loaded TSE model (Epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0.0):.4f}).")

    # 2. Load V6 C Inverter
    flow = load_inverter_v4(args.v6c, device=device).eval()
    mel_fn = ExactGpuMel(device=device)

    # 3. Load Demucs Baseline
    demucs_model = get_model("htdemucs").to(device).eval()
    bass_idx = demucs_model.sources.index("bass")

    # 4. Held-out backing from the val stem cache
    cache_dir = Path(args.cache_dir)
    val_kicks = np.load(cache_dir / "val_kicks.npy", mmap_mode="r")
    val_drums = np.load(cache_dir / "val_drums.npy", mmap_mode="r")
    val_synths = np.load(cache_dir / "val_synths.npy", mmap_mode="r")

    # 5. Held-out test patches and their cues
    plugin = S5.init_synth()
    items = load_items(args, plugin)
    print(f"Evaluating {len(items)} patches ({args.patch_source}) x {args.draws} flow draws...")

    cutoff_idx = PARAM_INDEX["cutoff"]
    rows = []
    for i, it in enumerate(items):
        note = torch.tensor([(it["midi"] - S6.NOTE_LOW) / (S6.NOTE_HIGH - S6.NOTE_LOW)], dtype=torch.float32,
                            device=device)
        target_vec = torch.from_numpy(np.asarray(it["vec"], dtype=np.float32))

        clean_4s = np.tile(it["target"], CLIP_SAMPLES // len(it["target"]) + 1)[:CLIP_SAMPLES]
        backing = (val_drums[i % len(val_drums)] * 0.6 + val_kicks[i % len(val_kicks)] * 0.4
                   + val_synths[i % len(val_synths)] * 0.3)
        scaled_backing = backing * ((np.sqrt(np.mean(clean_4s ** 2)) + 1e-6) / (np.sqrt(np.mean(backing ** 2)) + 1e-6))
        mix_4s = clean_4s + scaled_backing
        max_val = max(1.0, float(np.abs(mix_4s).max()))
        mix_4s, clean_4s, cue = mix_4s / max_val, clean_4s / max_val, it["cue"] / max_val

        # (a) V8 TSE Extraction
        t_mix = torch.from_numpy(mix_4s).float()[None, None].to(device)
        t_cue = torch.from_numpy(np.asarray(cue)).float()[None, None].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            pred_tse, _ = tse_model(t_mix, t_cue)
        tse_4s = pred_tse.squeeze().float().cpu().numpy()

        # (b) Demucs Separation
        mix_stereo = torch.from_numpy(mix_4s).float()[None].repeat(2, 1)[None].to(device)
        with torch.no_grad():
            sources = apply_model(demucs_model, mix_stereo, shifts=0, split=True, overlap=0.25)
        demucs_4s = sources[0, bass_idx].mean(dim=0).cpu().numpy()

        slices = {"clean": clean_4s[:SLICE], "tse": tse_4s[:SLICE], "demucs": demucs_4s[:SLICE], "mix": mix_4s[:SLICE]}
        row = {"item": i, "source": it["source"], "midi": it["midi"]}
        for cond in ("tse", "demucs", "mix"):
            row[f"ssim_{cond}"] = float(mel_ssim_dist(slices["clean"], slices[cond], sr=44100))
            row[f"oct_{cond}"] = float(octave_db_l1(slices["clean"], slices[cond], sr=44100))

        cont_true = target_vec[FLOW_CONT_INDICES]
        for cond, wav in slices.items():
            mel = mel_fn(torch.from_numpy(np.ascontiguousarray(wav)).float().to(device)).unsqueeze(0)
            mses, cuts = [], []
            for d in range(args.draws):
                gen = torch.Generator(device=device).manual_seed(1337 + 1000 * d + i)  # same seeds for every condition
                with torch.no_grad():
                    pred = codec.decode(flow.sample(mel, note=note, num_steps=25, generator=gen), note)[0].cpu()
                mses.append(float((pred[FLOW_CONT_INDICES] - cont_true).pow(2).mean()))
                cuts.append(abs(float(pred[cutoff_idx]) - float(target_vec[cutoff_idx])))
            row[f"mse_{cond}"] = float(np.mean(mses))
            row[f"cut_{cond}"] = float(np.mean(cuts))
        rows.append(row)
        print(f"Item {i+1:02d} | MSE clean {row['mse_clean']:.4f} TSE {row['mse_tse']:.4f} "
              f"Demucs {row['mse_demucs']:.4f} mix {row['mse_mix']:.4f}")

    col = lambda k: np.array([r[k] for r in rows])
    summary = {f"mean_mse_{c}": float(col(f"mse_{c}").mean()) for c in CONDITIONS}
    summary.update({f"mean_cut_{c}": float(col(f"cut_{c}").mean()) for c in CONDITIONS})
    summary.update({f"mean_ssim_{c}": float(col(f"ssim_{c}").mean()) for c in ("tse", "demucs", "mix")})
    summary.update({f"mean_oct_{c}": float(col(f"oct_{c}").mean()) for c in ("tse", "demucs", "mix")})
    summary["tax_ratio_mse"] = summary["mean_mse_tse"] / summary["mean_mse_clean"]
    summary["tax_demucs_mse"] = summary["mean_mse_demucs"] / summary["mean_mse_clean"]
    for k in ("mse", "cut", "ssim"):
        m, lo, hi = paired_bootstrap(col(f"{k}_tse"), col(f"{k}_demucs"))
        summary[f"tse_minus_demucs_{k}"] = {"mean": m, "ci95": [lo, hi]}
    summary.update({"n_items": len(rows), "draws": args.draws, "patch_source": args.patch_source,
                    "cue": "same patch at another note (cue_note), 1.0 s",
                    "gate": "tax_ratio_mse <= 1.25", "gate_pass": bool(summary["tax_ratio_mse"] <= 1.25),
                    "missing_baselines": ["BS-RoFormer", "matched two-stem model"]})

    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"Saved results to {out}")


def main():
    ap = argparse.ArgumentParser(description="V8 Phase 1: TSE vs Demucs separation tax on held-out patches")
    ap.add_argument("--patch-source", choices=["prior", "bank"], default="prior")
    ap.add_argument("--bass-manifold", default=None, help="V6 prior bass manifold npz (prior source)")
    ap.add_argument("--all-manifold", default=None, help="V6 prior all-presets manifold npz (prior source)")
    ap.add_argument("--bank-dir", default="/home/kim/sao_runs/bank_v6")
    ap.add_argument("--bank-shard", type=int, default=1)
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tse-ckpt", default="/home/kim/sao_runs/v8_phase1_tse/tse_best.pt")
    ap.add_argument("--v6c", default="/home/kim/sao_runs/v6_C/flow_best.pt")
    ap.add_argument("--cache-dir", default="/home/kim/sao_runs/v8_stem_cache")
    ap.add_argument("--out", default="/home/kim/Projects/SAO/eval/v8_phase1_tse_results_v2.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"V8 Phase 1: Asymmetric TSE Evaluation & Separation Tax Benchmark (device {device})")
    print("=" * 80)
    if "cuda" in device and not acquire_gpu_guard():
        print("GPU is currently busy. Refusing to run.")
        sys.exit(0)
    try:
        run_eval(device, args)
    finally:
        if "cuda" in device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
