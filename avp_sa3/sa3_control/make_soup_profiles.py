"""Build a family of weight-averaged (model-soup) checkpoints with different
weighting PROFILES over a checkpoint trajectory, to probe whether different
temporal emphases surface anything (e.g. early+overtrained combos).

Profiles over N checkpoints (epoch start..end), index i=0..N-1, center c=(N-1)/2:
  desc     linear, high at start  (N-i)
  asc      linear, high at end    (i+1)
  tri      triangular, peak middle
  valley   inverse-triangular, high at both ends
  cosdesc  raised cosine 1->0
  cosasc   raised cosine 0->1
  sine     sine hump 0->1->0

Loads each checkpoint ONCE, distributing into all profile accumulators. CPU only.

    python make_soup_profiles.py --ckpt-dir <dir> --start-ep 10 --end-ep 40 \
        --epoch-steps 5400 --out-dir <soups_dir>
"""
import argparse
import math
import os

import torch


def profiles(N, start_ep=10, tau=4.0):
    c = (N - 1) / 2.0
    pk20, pk25 = 20 - start_ep, 25 - start_ep   # index of ep20 / ep25 within the window
    P = {
        "desc":      [N - i for i in range(N)],
        "asc":       [i + 1 for i in range(N)],
        "tri":       [(c - abs(i - c)) + 1e-6 for i in range(N)],
        "valley":    [abs(i - c) + 1e-6 for i in range(N)],
        "cosdesc":   [0.5 * (1 + math.cos(math.pi * i / (N - 1))) + 1e-6 for i in range(N)],
        "cosasc":    [0.5 * (1 - math.cos(math.pi * i / (N - 1))) + 1e-6 for i in range(N)],
        "sine":      [math.sin(math.pi * i / (N - 1)) + 1e-6 for i in range(N)],
        "expasc":    [math.exp(4.0 * i / (N - 1)) for i in range(N)],          # ~55x ratio, smooth ascending
        "exppeak20": [math.exp(-abs(i - pk20) / tau) for i in range(N)],        # Laplacian peak at ep20
        "exppeak25": [math.exp(-abs(i - pk25) / tau) for i in range(N)],        # Laplacian peak at ep25
    }
    return {k: [w / sum(v) for w in v] for k, v in P.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--start-ep", type=int, default=10)
    ap.add_argument("--end-ep", type=int, default=40)
    ap.add_argument("--epoch-steps", type=int, default=5400)
    ap.add_argument("--out-dir", default="/run/media/kim/Mantu/sa3_control_runs/soups")
    ap.add_argument("--tag", default="")  # appended to filenames, e.g. _ep10-40
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eps = list(range(args.start_ep, args.end_ep + 1))
    paths = [os.path.join(args.ckpt_dir, f"riffer_step{e * args.epoch_steps}.pt") for e in eps]
    paths = [p for p in paths if os.path.exists(p)]
    N = len(paths)
    tag = args.tag or f"_ep{args.start_ep}-{args.end_ep}"
    print(f"[profiles] {N} checkpoints (ep{args.start_ep}-{args.end_ep})", flush=True)
    W = profiles(N, args.start_ep)

    ref = torch.load(paths[0], map_location="cpu", weights_only=False)
    keys = [k for k, v in ref["state"].items() if torch.is_tensor(v)]
    acc = {name: {k: torch.zeros_like(ref["state"][k], dtype=torch.float64) for k in keys}
           for name in W}

    for i, p in enumerate(paths):
        st = torch.load(p, map_location="cpu", weights_only=False)["state"]
        for name in W:
            w = W[name][i]
            for k in keys:
                acc[name][k] += st[k].to(torch.float64) * w
        del st
        print(f"[profiles]  loaded {i+1}/{N}  {os.path.basename(p)}", flush=True)

    for name in W:
        out = dict(ref)  # shallow copy: keep args/scalar_norm/etc
        out["state"] = {k: (acc[name][k].to(ref["state"][k].dtype) if k in keys else ref["state"][k])
                        for k in ref["state"]}
        fp = os.path.join(args.out_dir, f"soup_{name}{tag}.pt")
        torch.save(out, fp)
        print(f"[profiles] saved {fp}", flush=True)


if __name__ == "__main__":
    main()
