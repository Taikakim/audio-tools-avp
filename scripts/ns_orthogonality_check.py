#!/usr/bin/env python
# Owner: KUANG (created 2026-10-17)
"""Newton-Schulz orthogonality check for test plan section T (C7, ns_dtype=bf16).

For each input matrix X, runs the ModularOptimizer quintic NS in fp32 and in bf16, plus FusionOpt's
newton_schulz_5 in bf16 (the path the shipped FiLM used), and reports
    err = ||U U^T - I||_F / sqrt(k)   on the smaller side k   (0 = exactly semi-orthogonal; quintic NS5
          deliberately stops at singular values ~0.7-1.2, so fp32 itself is NOT 0)
    cos = cosine(U_bf16, U_fp32)      (direction agreement)
Inputs: --ckpt PATH reads every 2-D tensor stored under an optimizer-state key named 'momentum' or 'm'
(what NS actually sees), up to --max-mats; without --ckpt, random Gaussian and low-rank matrices of
--shapes. No training, no model load. Runs on CPU or GPU (--device).

  python scripts/ns_orthogonality_check.py --shapes 3072x768 768x768 768x3072
  python scripts/ns_orthogonality_check.py --ckpt run/riffer_final.pt --device cuda --max-mats 20
"""
import argparse
import math

import torch

from stable_audio_tools.training.modular_opt.lmo import newton_schulz_quintic
from stable_audio_tools.training.fusion_opt import newton_schulz_5


def _err(U):
    U = U.float()
    if U.shape[0] > U.shape[1]:
        U = U.T
    k = U.shape[0]
    return float((U @ U.T - torch.eye(k, device=U.device)).norm() / math.sqrt(k))


def _from_ckpt(path, max_mats):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def walk(o, key=""):
        if len(found) >= max_mats:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key)
        elif torch.is_tensor(o) and o.ndim == 2 and key in ("momentum", "m") and min(o.shape) >= 8:
            found.append((f"{key}{tuple(o.shape)}", o.float()))
    walk(obj)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-mats", type=int, default=20)
    ap.add_argument("--shapes", nargs="*", default=["768x768", "3072x768", "768x3072", "256x256"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.ckpt:
        mats = _from_ckpt(a.ckpt, a.max_mats)
        if not mats:
            raise SystemExit(f"no 2-D 'momentum'/'m' tensors found in {a.ckpt}")
    else:
        mats = []
        for s in a.shapes:
            m, n = (int(x) for x in s.lower().split("x"))
            mats.append((f"gauss{m}x{n}", torch.randn(m, n)))
            r = max(1, min(m, n) // 16)
            mats.append((f"lowrank{r}_{m}x{n}", torch.randn(m, r) @ torch.randn(r, n) + 0.01 * torch.randn(m, n)))
    print(f"{'matrix':28s} {'err fp32':>9s} {'err bf16':>9s} {'err FO bf16':>11s} {'cos bf16':>9s} {'cos FO':>8s}")
    worst = 0.0
    for name, X in mats:
        X = X.to(a.device)
        u32 = newton_schulz_quintic(X, compute_dtype=torch.float32)
        u16 = newton_schulz_quintic(X, compute_dtype=torch.bfloat16)
        ufo = newton_schulz_5(X.to(torch.bfloat16)).float()
        c16 = float(torch.nn.functional.cosine_similarity(u16.float().flatten(), u32.flatten(), dim=0))
        cfo = float(torch.nn.functional.cosine_similarity(ufo.flatten(), u32.flatten(), dim=0))
        e16, efo = _err(u16), _err(ufo)
        worst = max(worst, e16 - efo)
        print(f"{name:28s} {_err(u32):9.4f} {e16:9.4f} {efo:11.4f} {c16:9.5f} {cfo:8.5f}")
    print(f"max (err bf16 - err FusionOpt bf16) = {worst:.4f}  "
          f"(plan T kill rule: bf16 'clearly worse than FusionOpt's')")


if __name__ == "__main__":
    main()
