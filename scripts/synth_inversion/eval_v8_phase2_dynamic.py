#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 2: Dynamic 10 Hz Parameter Inverter Evaluation & Centroid Trajectory Gate.

Evaluates the trained DynamicParameterInverterV8 on held-out dynamic sweeps:
1. Loads dynamic_inverter_best.pt from /home/kim/sao_runs/v8_phase2_inverter/.
2. Evaluates across 50 held-out dynamic sweep test items:
   - Static parameter MSE (59 parameters).
   - Per-lane dynamic trajectory MSE: Cutoff, Resonance, Drive, FM Depth.
   - Dynamic Trajectory Correlation (Spearman rho) between predicted and true sweep curves.
3. Evaluates the Phase 2 Gate: Spearman rho >= 0.80 on sweep lanes.
4. Outputs summary metrics to /home/kim/Projects/SAO/eval/v8_phase2_dynamic_results.json.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamic_dataset_v8 import DynamicSweepDataset
from dynamic_inverter_v8 import DynamicParameterInverterV8, curvature_smoothness_loss
import surge_spec_v6 as S6


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("V8 Phase 2: Dynamic Inverter Evaluation & Trajectory Correlation Gate")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Checkpoint
    ckpt_path = Path("/home/kim/sao_runs/v8_phase2_inverter/dynamic_inverter_best.pt")
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Dynamic Inverter from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DynamicParameterInverterV8().to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from Epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', 0.0):.4f}).")

    # 2. Load Validation Set
    val_cache = Path("/home/kim/sao_runs/v8_dynamic_cache/dynamic_val.npz")
    if not val_cache.exists():
        print(f"Error: validation cache not found at {val_cache}", file=sys.stderr)
        sys.exit(1)

    val_ds = DynamicSweepDataset(val_cache)
    print(f"Loaded {len(val_ds)} validation sweep items.")

    # 3. Evaluation Loop
    mse_statics = []
    mse_cutoffs = []
    mse_res = []
    mse_drive = []
    mse_fm = []
    spearman_rhos = []
    smoothness_scores = []

    lane_names = ["Cutoff", "Resonance", "Drive", "FM Depth"]

    for i in range(len(val_ds)):
        item = val_ds[i]
        mel = item["mel"].unsqueeze(0).to(device)
        t_static = item["target_static"].unsqueeze(0).to(device)
        t_dynamic = item["target_dynamic"].unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                p_static, p_dynamic = model(mel)

        # MSE
        e_stat = F.mse_loss(p_static, t_static).item()
        e_cut = F.mse_loss(p_dynamic[:, 0], t_dynamic[:, 0]).item()
        e_res = F.mse_loss(p_dynamic[:, 1], t_dynamic[:, 1]).item()
        e_drv = F.mse_loss(p_dynamic[:, 2], t_dynamic[:, 2]).item()
        e_fm = F.mse_loss(p_dynamic[:, 3], t_dynamic[:, 3]).item()
        smooth = curvature_smoothness_loss(p_dynamic).item()

        mse_statics.append(e_stat)
        mse_cutoffs.append(e_cut)
        mse_res.append(e_res)
        mse_drive.append(e_drv)
        mse_fm.append(e_fm)
        smoothness_scores.append(smooth)

        # Spearman rank correlation on cutoff trajectory
        p_cut_np = p_dynamic[0, 0].cpu().float().numpy()
        t_cut_np = t_dynamic[0, 0].cpu().float().numpy()

        # Check if the ground truth curve has variation (not a flat static line)
        if np.std(t_cut_np) > 0.02:
            rho, _ = spearmanr(p_cut_np, t_cut_np)
            if not np.isnan(rho):
                spearman_rhos.append(float(rho))

    # Aggregates
    mean_static_mse = float(np.mean(mse_statics))
    mean_cutoff_mse = float(np.mean(mse_cutoffs))
    mean_res_mse = float(np.mean(mse_res))
    mean_drive_mse = float(np.mean(mse_drive))
    mean_fm_mse = float(np.mean(mse_fm))
    mean_smoothness = float(np.mean(smoothness_scores))
    mean_rho = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0

    print("\n" + "=" * 80)
    print("PHASE 2 DYNAMIC PARAMETER INVERTER BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Static 59-Parameter Core MSE:       {mean_static_mse:.5f}")
    print(f"Dynamic Lane 0 (Cutoff) MSE:        {mean_cutoff_mse:.5f}")
    print(f"Dynamic Lane 1 (Resonance) MSE:     {mean_res_mse:.5f}")
    print(f"Dynamic Lane 2 (Drive) MSE:         {mean_drive_mse:.5f}")
    print(f"Dynamic Lane 3 (FM Depth) MSE:      {mean_fm_mse:.5f}")
    print(f"Curvature Smoothness Penalty:       {mean_smoothness:.6f}")
    print("-" * 80)
    print(f"Cutoff Trajectory Correlation (rho): {mean_rho:.4f} (across {len(spearman_rhos)} dynamic sweep items)")
    print("=" * 80)

    # Gate verification
    gate_pass = mean_rho >= 0.80
    print(f"\nPhase 2 Gate Status (Spearman rho >= 0.80): {'PASSED' if gate_pass else 'FAILED (rho=' + f'{mean_rho:.4f})'}")

    out_file = Path("/home/kim/Projects/SAO/eval/v8_phase2_dynamic_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(
            {
                "mean_static_mse": mean_static_mse,
                "mean_cutoff_mse": mean_cutoff_mse,
                "mean_res_mse": mean_res_mse,
                "mean_drive_mse": mean_drive_mse,
                "mean_fm_mse": mean_fm_mse,
                "mean_smoothness": mean_smoothness,
                "mean_spearman_rho": mean_rho,
                "num_sweep_items": len(spearman_rhos),
                "gate_pass": bool(gate_pass),
            },
            f,
            indent=2,
        )
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
