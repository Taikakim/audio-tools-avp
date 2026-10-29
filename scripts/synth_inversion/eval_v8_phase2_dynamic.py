#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 2: Dynamic 10 Hz Parameter Inverter Evaluation & Centroid Trajectory Gate.

Evaluates the trained DynamicParameterInverterV8 on held-out dynamic sweeps:
1. Loads dynamic_inverter_best.pt.
2. Evaluates on the test split (dynamic_test.npz; the val split picked the checkpoint):
   - Static parameter MSE (lane parameters excluded) and its continuous-parameter subset.
   - Per-lane trajectory MSE over items where the lane is audible, with R^2 against a constant guess (the training
     mean of that lane): a lane whose R^2 is near 0 has not been learned.
   - Spearman rho between predicted and true cutoff curves, overall and per sweep family. Rising, falling and
     exponential curves are monotone, so any prediction moving the right way scores rho = 1 there; the sinusoidal
     family is the informative one.
3. Phase 2 gate as first implemented: mean cutoff rho >= 0.80 over non-static items. Note: the spec registers a
   centroid-trajectory correlation on re-rendered audio, which this script does not measure.
4. Outputs summary metrics to the results JSON.

KUANG 2026-10-29 (docs/synth-v8-code-review-2026-10-29.md, M6): test split, lane masks, per-family rho, R^2.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamic_dataset_v8 import LANE_NAMES, SWEEP_TYPES, DynamicSweepDataset
from dynamic_inverter_v8 import STATIC_LOSS_MASK, DynamicParameterInverterV8, curvature_smoothness_loss
import surge_spec_v6 as S6

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"


def acquire_gpu_guard() -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
    return res.returncode == 0


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def summarize(pred_static, true_static, pred_dyn, true_dyn, active, family, lane_means) -> Dict:
    """Metrics from stacked arrays: pred/true_static [N, 59], pred/true_dyn [N, 4, T], active [N, 4] bool,
    family [N] (index into SWEEP_TYPES, -1 unknown), lane_means [4] (the constant-guess baseline)."""
    mask = STATIC_LOSS_MASK.numpy().astype(bool)
    cont = np.zeros_like(mask)
    cont[S6.FLOW_CONT_INDICES] = True
    err_s = (pred_static - true_static) ** 2
    out = {
        "n_items": int(len(pred_static)),
        "static_mse": float(err_s[:, mask].mean()),
        "static_mse_cont": float(err_s[:, mask & cont].mean()),
        "lanes": {},
        "smoothness": float(curvature_smoothness_loss(torch.from_numpy(pred_dyn)).item()),
    }
    for k, name in enumerate(LANE_NAMES):
        a = active[:, k]
        if not a.any():
            out["lanes"][name] = {"n_active": 0}
            continue
        mse = float(((pred_dyn[a, k] - true_dyn[a, k]) ** 2).mean())
        base = float(((lane_means[k] - true_dyn[a, k]) ** 2).mean())
        out["lanes"][name] = {"n_active": int(a.sum()), "mse": mse, "baseline_mse": base,
                              "r2": float(1.0 - mse / base) if base > 0 else float("nan")}

    rhos: Dict[str, List[float]] = {}
    for i in range(len(pred_dyn)):
        t = true_dyn[i, 0]
        if np.std(t) <= 0.02:  # static cutoff: rank correlation undefined
            continue
        rho = spearmanr(pred_dyn[i, 0], t)[0]
        if np.isnan(rho):
            continue
        fam = SWEEP_TYPES[family[i]] if 0 <= family[i] < len(SWEEP_TYPES) else "unknown"
        rhos.setdefault(fam, []).append(float(rho))
    all_rhos = [r for v in rhos.values() for r in v]
    out["cutoff_rho_mean"] = float(np.mean(all_rhos)) if all_rhos else 0.0
    out["cutoff_rho_n"] = len(all_rhos)
    out["cutoff_rho_by_family"] = {f: {"n": len(v), "mean": float(np.mean(v))} for f, v in sorted(rhos.items())}
    return out


def run_eval(device: str, args):
    print("=" * 80)
    print("V8 Phase 2: Dynamic Inverter Evaluation & Trajectory Correlation Gate")
    print(f"Device: {device}")
    print("=" * 80)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DynamicParameterInverterV8().to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0.0):.4f}).")

    cache_dir = Path(args.cache_dir)
    test_cache = cache_dir / "dynamic_test.npz"
    if not test_cache.exists():
        print(f"Error: {test_cache} not found. The val split picked the checkpoint and cannot also gate it; "
              f"rebuild the caches with dynamic_dataset_v8.py.", file=sys.stderr)
        sys.exit(1)
    test_ds = DynamicSweepDataset(test_cache)
    if not test_ds.fixed_renderer:
        print("Warning: test cache rendered before the renderer fix; results are not valid.", file=sys.stderr)

    train_cache = cache_dir / "dynamic_train.npz"
    if train_cache.exists():
        tr = DynamicSweepDataset(train_cache)
        act = tr.lane_active.numpy()
        dyn = tr.dynamics.numpy()
        lane_means = np.array([dyn[act[:, k], k].mean() if act[:, k].any() else 0.5 for k in range(len(LANE_NAMES))])
    else:
        print("Warning: no train cache; the constant-guess baseline uses the test means (optimistic).")
        lane_means = test_ds.dynamics.numpy().mean(axis=(0, 2))

    ps, pd = [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), 32):
            mel = test_ds.mels[i:i + 32].to(device)
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                p_static, p_dynamic = model(mel)
            ps.append(p_static.float().cpu().numpy())
            pd.append(p_dynamic.float().cpu().numpy())

    res = summarize(np.concatenate(ps), test_ds.statics.numpy(), np.concatenate(pd), test_ds.dynamics.numpy(),
                    test_ds.lane_active.numpy(), test_ds.cutoff_family.numpy(), lane_means)
    res["gate_pass"] = bool(res["cutoff_rho_mean"] >= 0.80)
    res["gate_note"] = ("mean Spearman rho of the cutoff lane >= 0.80 over non-static items; the spec's registered "
                        "gate (centroid-trajectory correlation on re-rendered audio) is not measured here")
    res["checkpoint"] = str(ckpt_path)
    res["cache"] = str(test_cache)

    print(json.dumps(res, indent=2))
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"Saved results to {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the V8 Phase 2 dynamic inverter on the test split")
    ap.add_argument("--ckpt", default="/home/kim/sao_runs/v8_phase2_inverter_v2/dynamic_inverter_best.pt")
    ap.add_argument("--cache-dir", default="/home/kim/sao_runs/v8_dynamic_cache_v2")
    ap.add_argument("--out", default="/home/kim/Projects/SAO/eval/v8_phase2_dynamic_results_v2.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        if not acquire_gpu_guard():
            print("GPU is currently busy. Exiting.", file=sys.stderr)
            sys.exit(0)
    try:
        run_eval(device, args)
    finally:
        if "cuda" in device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
