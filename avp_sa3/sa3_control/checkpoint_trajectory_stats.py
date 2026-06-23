"""Weight-space trajectory statistics across a run's checkpoints.

Characterizes HOW a control head moves through weight space as training proceeds —
independent of what any layer "means". Standing tool: run on every finished run so
we accumulate a comparable library (see SAO master-level guideline).

Per checkpoint (sorted by step) it computes, over the concatenated ck["state"] vector:
  global_norm        L2 norm of all weights
  d_from_init        ||W_t - W_0||
  d_from_prev        ||W_t - W_{t-1}||   (learning velocity)
  d_from_final       ||W_t - W_last||    (convergence)
  cos_to_final       cosine(W_t, W_last)
  step_cos           cosine of the step direction vs the previous step (path curvature;
                     ~1 = moving straight, ~0 = turning, <0 = doubling back)
and per-layer ||ΔW|| velocity (which tensors keep moving late).

Summary: path_length (Σ velocity), net_displacement (||W_last-W_0||),
path_efficiency = net/path (low = wandering/oscillating in a basin → averaging helps),
centroid checkpoint (closest to the mean of all checkpoints = the natural "soup center").

    python checkpoint_trajectory_stats.py --ckpt-dir <dir> --label lr2e5 \
        --out-dir /home/kim/Projects/SAO/checkpoint-stats
Writes <label>_trajectory.json + <label>_trajectory.md (+ PNGs if matplotlib present).
CPU only.
"""
import argparse
import glob
import json
import os
import re

import torch


def flat(state):
    return torch.cat([v.reshape(-1).to(torch.float64) for v in state.values() if torch.is_tensor(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--epoch-steps", type=int, default=5400)
    ap.add_argument("--out-dir", default="/home/kim/Projects/SAO/checkpoint-stats")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = glob.glob(os.path.join(args.ckpt_dir, "riffer_step*.pt"))
    paths = sorted(paths, key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    if not paths:
        print(f"[traj] no checkpoints in {args.ckpt_dir}")
        return
    steps = [int(re.search(r"step(\d+)", p).group(1)) for p in paths]
    print(f"[traj] {args.label}: {len(paths)} checkpoints, steps {steps[0]}..{steps[-1]}", flush=True)

    # anchors
    W0 = flat(torch.load(paths[0], map_location="cpu", weights_only=False)["state"])
    Wlast = flat(torch.load(paths[-1], map_location="cpu", weights_only=False)["state"])
    keys = [k for k, v in torch.load(paths[0], map_location="cpu", weights_only=False)["state"].items()
            if torch.is_tensor(v)]
    centroid = torch.zeros_like(W0)

    rows, prev, prev_dir, layer_prev = [], None, None, None
    flats = []
    for p, step in zip(paths, steps):
        st = torch.load(p, map_location="cpu", weights_only=False)["state"]
        fw = flat(st)
        flats.append(fw)
        centroid += fw / len(paths)
        d_prev = float((fw - prev).norm()) if prev is not None else 0.0
        cur_dir = (fw - prev) if prev is not None else None
        step_cos = (float(torch.dot(cur_dir, prev_dir) / (cur_dir.norm() * prev_dir.norm() + 1e-12))
                    if (cur_dir is not None and prev_dir is not None) else 0.0)
        layer_vel = {}
        for k in keys:
            t = st[k].reshape(-1).to(torch.float64)
            layer_vel[k] = (float((t - layer_prev[k]).norm()) if layer_prev else 0.0)
        rows.append({
            "step": step, "epoch": round(step / args.epoch_steps, 2),
            "global_norm": round(float(fw.norm()), 4),
            "d_from_init": round(float((fw - W0).norm()), 4),
            "d_from_prev": round(d_prev, 4),
            "d_from_final": round(float((fw - Wlast).norm()), 4),
            "cos_to_final": round(float(torch.dot(fw, Wlast) / (fw.norm() * Wlast.norm() + 1e-12)), 5),
            "step_cos": round(step_cos, 4),
            "layer_vel": {k: round(v, 5) for k, v in layer_vel.items()},
        })
        prev, prev_dir = fw, cur_dir
        layer_prev = {k: st[k].reshape(-1).to(torch.float64) for k in keys}
        del st

    path_len = sum(r["d_from_prev"] for r in rows)
    net = float((Wlast - W0).norm())
    cent_d = [float((fw - centroid).norm()) for fw in flats]
    cent_i = min(range(len(cent_d)), key=lambda i: cent_d[i])
    # which layers moved the most total
    tot_layer = {k: sum(r["layer_vel"][k] for r in rows) for k in keys}
    top_layers = sorted(tot_layer.items(), key=lambda kv: -kv[1])[:8]

    summary = {
        "label": args.label, "n_checkpoints": len(paths),
        "path_length": round(path_len, 4), "net_displacement": round(net, 4),
        "path_efficiency": round(net / (path_len + 1e-12), 4),
        "centroid_checkpoint": {"step": steps[cent_i], "epoch": round(steps[cent_i] / args.epoch_steps, 2),
                                "dist_to_centroid": round(cent_d[cent_i], 4)},
        "peak_velocity": max(rows[1:], key=lambda r: r["d_from_prev"]) if len(rows) > 1 else None,
        "top_moving_layers": [{"layer": k, "total_velocity": round(v, 4)} for k, v in top_layers],
    }
    out = {"summary": summary, "rows": [{k: v for k, v in r.items() if k != "layer_vel"} for r in rows],
           "ckpt_dir": args.ckpt_dir}
    jp = os.path.join(args.out_dir, f"{args.label}_trajectory.json")
    json.dump({**out, "layer_vel_by_step": {r["step"]: r["layer_vel"] for r in rows}},
              open(jp, "w"), indent=1)

    # markdown
    md = [f"# Checkpoint trajectory — {args.label}", "",
          f"- checkpoints: **{len(paths)}** (steps {steps[0]}–{steps[-1]})",
          f"- net displacement ||W_last−W_0||: **{net:.3f}**",
          f"- total path length Σ‖ΔW‖: **{path_len:.3f}**",
          f"- **path efficiency** (net/path): **{summary['path_efficiency']:.3f}**  "
          f"(low ⇒ wandering/oscillating in a basin → averaging should help)",
          f"- centroid (soup-center) checkpoint: **epoch {summary['centroid_checkpoint']['epoch']}** "
          f"(step {summary['centroid_checkpoint']['step']})",
          f"- peak learning velocity: epoch {summary['peak_velocity']['epoch']} "
          f"(‖ΔW‖={summary['peak_velocity']['d_from_prev']:.3f})", "",
          "## Per-checkpoint", "",
          "| epoch | global_norm | d_from_init | velocity ‖ΔW‖ | d_from_final | cos_to_final | step_cos |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['epoch']} | {r['global_norm']:.2f} | {r['d_from_init']:.3f} | "
                  f"{r['d_from_prev']:.3f} | {r['d_from_final']:.3f} | {r['cos_to_final']:.4f} | {r['step_cos']:.3f} |")
    md += ["", "## Top moving layers (total velocity)", ""]
    for k, v in top_layers:
        md.append(f"- `{k}`: {v:.3f}")
    open(os.path.join(args.out_dir, f"{args.label}_trajectory.md"), "w").write("\n".join(md))
    print(f"[traj] wrote {jp} (+ .md)", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in rows]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(ep, [r["d_from_prev"] for r in rows]); ax[0, 0].set_title("learning velocity ‖ΔW‖"); ax[0, 0].set_xlabel("epoch")
        ax[0, 1].plot(ep, [r["d_from_init"] for r in rows]); ax[0, 1].set_title("distance from init"); ax[0, 1].set_xlabel("epoch")
        ax[1, 0].plot(ep, [r["d_from_final"] for r in rows]); ax[1, 0].set_title("distance from final"); ax[1, 0].set_xlabel("epoch")
        ax[1, 1].plot(ep, [r["step_cos"] for r in rows]); ax[1, 1].axhline(0, color="r", ls=":"); ax[1, 1].set_title("step direction cos (path straightness)"); ax[1, 1].set_xlabel("epoch")
        fig.suptitle(f"weight trajectory — {args.label}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{args.label}_trajectory.png"), dpi=110)
        print(f"[traj] wrote {args.label}_trajectory.png", flush=True)
    except Exception as e:
        print(f"[traj] (no plot: {e})", flush=True)


if __name__ == "__main__":
    main()
