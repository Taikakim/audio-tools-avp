"""Regression test for the full-FT latent-scale runaway (CONTINUITY, 2026-08-10).

Incident: the LUMI full fine-tunes (#68 fullft_bigset, precision_ladder — all
FusionOpt) decoded to spectral drone. Root cause, measured from the sampled
latents: the model's output scale RUNS AWAY during training (global latent std
0.7 good -> 1.3 @ep3 -> 5.6 @ep7; #channels with std>2.0: 0 -> 4 -> 166 of 256).
The mechanism is that FusionOpt's `weight_decay` DEFAULTS TO 0.0 (fusion_opt.py:196)
and train_lora's full-FT path never set it, so nothing bounds weight-norm growth:
a Muon/Schedule-Free optimiser with no decay, no grad clip, no LR decay, on all
1.4B params. The AdamW runs stayed healthy because AdamW ran with weight_decay=0.01.

This test pins the mechanism at the optimiser level, no GPU, in seconds:
under a SUSTAINED gradient (the drift condition), wd=0 lets the weight norm grow
without bound, while wd>0 reaches a bounded equilibrium. If a future change makes
FusionOpt ignore weight_decay (or a refactor drops the per-group wd), the drone
returns and this test fails first.

Runnable directly (pytest is not installed in sat-venv):
    sat-venv/bin/python tests/test_fusion_weight_decay.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.training.fusion_opt import FusionOpt

# #68's component set (full Fusion): MONA + NS5 + NorMuon + Schedule-Free.
FULLFT_COMPONENTS = {"mona", "ns5", "normuon", "sf"}


def _run_sustained(weight_decay, steps=500, lr=1e-2, seed=0):
    """Drive one 2D weight with a FIXED gradient every step (sustained drift),
    return the trajectory of the weight's Frobenius norm. lr is scaled up vs
    #68's 8e-5 so the linear-vs-bounded character shows in a few hundred CPU
    steps; the mechanism (unbounded accumulation without decay) is lr-invariant."""
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
    g = torch.randn(16, 16)                      # fixed drift direction
    g = g / g.norm()
    groups = [{"params": [w], "group_type": "spectral", "weight_decay": weight_decay}]
    opt = FusionOpt(groups, lr=lr, components=set(FULLFT_COMPONENTS), hot_dtype="fp32")
    norms = []
    for _ in range(steps):
        w.grad = g.clone()
        opt.set_loss(torch.tensor(1.0))
        opt.step()
        norms.append(float(w.detach().norm()))
    return norms


def test_no_weight_decay_runs_away():
    """wd=0 (the buggy default): weight norm keeps growing — the drone condition.

    Reproduces the incident at the optimiser level: still rising at the end and
    far above where it started."""
    n = _run_sustained(weight_decay=0.0)
    start, end = n[0], n[-1]
    late_slope = n[-1] - n[-50]
    assert end > 5 * start, f"expected runaway, ||w|| {start:.3f} -> {end:.3f}"
    assert late_slope > 0, f"expected still-rising at end, late delta={late_slope:.3f}"


def test_weight_decay_bounds_growth():
    """wd>0: weight norm reaches a bounded equilibrium (decay pull == grad push).

    This is the fix. The late-window growth is near-flat and the final norm is a
    small fraction of the wd=0 runaway."""
    n = _run_sustained(weight_decay=0.1)
    end = n[-1]
    plateau_slope = abs(n[-1] - n[-50]) / max(n[-50], 1e-9)
    assert plateau_slope < 0.02, f"expected plateau, late relative growth={plateau_slope:.3f}"
    runaway_end = _run_sustained(weight_decay=0.0)[-1]
    assert end < 0.5 * runaway_end, f"wd=0.1 end {end:.2f} not << wd=0 end {runaway_end:.2f}"


def test_weight_decay_is_monotone():
    """More weight decay -> smaller final weight norm (dose-response).

    Confirms it's the decay term doing the bounding, not an artefact."""
    ends = {wd: _run_sustained(weight_decay=wd)[-1] for wd in (0.0, 0.01, 0.1)}
    assert ends[0.0] > ends[0.01] > ends[0.1], f"expected monotone decrease, got {ends}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    # Diagnostic trajectory dump (the numbers that matter for the incident writeup)
    for wd in (0.0, 0.01, 0.1):
        n = _run_sustained(weight_decay=wd)
        print(f"  wd={wd:<5}: ||w||  start={n[0]:.3f}  ep-mid={n[len(n)//2]:.3f}  end={n[-1]:.3f}")
    sys.exit(1 if failed else 0)
