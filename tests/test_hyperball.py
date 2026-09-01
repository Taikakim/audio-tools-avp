"""Tests for HYPERBALL in FusionOpt (arXiv 2606.16899, Algorithm 1).

Hyperball constrains each spectral 2D weight matrix W to the hypersphere of radius
R = ‖W0‖_F, where W0 is the LOADED weight (R is captured ONCE, on the first step for
that param). The DIRECT-to-p spectral step is replaced by the Hyperball retraction:

    u_hat   = U / ‖U‖                    (U = FusionOpt's finalized spectral update)
    W_tilde = W - gamma_t * R * u_hat
    W       = R * W_tilde / ‖W_tilde‖

Consequences pinned here:
  (a) under a SUSTAINED fixed unit gradient, ‖W‖ stays glued to R every step (the
      retraction re-projects onto the sphere → constant Frobenius norm);
  (b) R == the INITIAL ‖W‖ (retrofit: first-step norm of the loaded weight is preserved);
  (c) hyperball=True together with 'sf' in components raises ValueError (Hyperball is its
      own iterate, incompatible with Schedule-Free);
  (d) hyperball=False is byte-identical to a normal FusionOpt step (off => unchanged).

Weight decay is IGNORED under hyperball (the norm constraint replaces it), and only the
spectral DIRECT-to-p path is touched (scalar path untouched).

Runnable directly (pytest is not installed in sat-venv):
    FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE sat-venv/bin/python tests/test_hyperball.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.training.fusion_opt import FusionOpt

# Hyperball's component set (Schedule-Free dropped): MONA + NS5 + NorMuon. This exercises
# the DIRECT-to-p branch (the only place Hyperball applies).
HYPERBALL_COMPONENTS = {"mona", "ns5", "normuon"}


def _run(hyperball, weight_decay=0.0, steps=300, lr=1e-2, seed=0, shape=(16, 16),
         components=HYPERBALL_COMPONENTS, capture=None):
    """Drive one 2D spectral weight with a FIXED unit gradient every step.
    Returns (weight, optimizer, per-step ‖W‖ trajectory). If `capture` is a dict it
    receives the initial weight norm under key 'init_norm' (before any step)."""
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(*shape) * 0.1)
    if capture is not None:
        capture["init_norm"] = float(w.detach().norm())
    g = torch.randn(*shape)
    g = g / g.norm()                                  # fixed unit-Frobenius drift direction
    groups = [{"params": [w], "group_type": "spectral", "weight_decay": weight_decay}]
    opt = FusionOpt(groups, lr=lr, components=set(components), hot_dtype="fp32",
                    hyperball=hyperball)
    norms = []
    for _ in range(steps):
        w.grad = g.clone()
        opt.set_loss(torch.tensor(1.0))
        opt.step()
        norms.append(float(w.detach().norm()))
    return w, opt, norms


def test_norm_stays_on_the_ball():
    """(a) Sustained fixed unit gradient over 300 steps: ‖W‖ stays within 1% of R
    at EVERY step (the retraction re-projects onto the sphere → constant norm)."""
    cap = {}
    w, opt, norms = _run(hyperball=True, capture=cap)
    R = cap["init_norm"]
    for i, n in enumerate(norms):
        rel = abs(n - R) / R
        assert rel < 0.01, f"step {i}: ||W||={n:.5f} drifted {rel*100:.3f}% off R={R:.5f}"


def test_R_is_the_initial_norm():
    """(b) R (state['hyperball_R']) equals the INITIAL ‖W‖ — the loaded-weight norm,
    captured on the first step before applying (retrofit preservation)."""
    cap = {}
    w, opt, norms = _run(hyperball=True, capture=cap, steps=1)
    R = float(opt.state[w]["hyperball_R"])
    assert abs(R - cap["init_norm"]) < 1e-6, (
        f"R={R:.6f} != initial ||W0||={cap['init_norm']:.6f}")


def test_sf_plus_hyperball_raises():
    """(c) Constructing with hyperball=True AND 'sf' in components raises ValueError."""
    w = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
    groups = [{"params": [w], "group_type": "spectral", "weight_decay": 0.0}]
    raised = False
    try:
        FusionOpt(groups, components={"mona", "ns5", "normuon", "sf"}, hyperball=True)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for hyperball=True with 'sf' in components"


def test_hyperball_off_is_byte_identical():
    """(d) hyperball=False is byte-identical to a normal FusionOpt step. Same seed/grad,
    explicit hyperball=False vs the default (arg omitted) must match bit-for-bit,
    exercising the same DIRECT-to-p (non-sf) branch Hyperball modifies."""
    torch.manual_seed(0)
    w_ref = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
    torch.manual_seed(0)
    w_hb = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
    assert torch.equal(w_ref, w_hb)  # identical init

    g = torch.randn(16, 16)
    g = g / g.norm()

    def _mk(w, **kw):
        groups = [{"params": [w], "group_type": "spectral", "weight_decay": 0.01}]
        return FusionOpt(groups, lr=1e-2, components=set(HYPERBALL_COMPONENTS),
                         hot_dtype="fp32", **kw)

    opt_ref = _mk(w_ref)                    # default: hyperball absent (=> False)
    opt_hb = _mk(w_hb, hyperball=False)     # explicit False

    for _ in range(50):
        w_ref.grad = g.clone()
        w_hb.grad = g.clone()
        opt_ref.set_loss(torch.tensor(1.0))
        opt_hb.set_loss(torch.tensor(1.0))
        opt_ref.step()
        opt_hb.step()

    assert torch.equal(w_ref, w_hb), (
        f"hyperball=False diverged from default: max|Δ|="
        f"{(w_ref - w_hb).abs().max().item():.3e}")


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
    # Diagnostic trajectory dump
    cap = {}
    _, _, norms = _run(hyperball=True, capture=cap)
    R = cap["init_norm"]
    print(f"  hyperball ON : R={R:.5f}  ||W|| start={norms[0]:.5f}  "
          f"mid={norms[len(norms)//2]:.5f}  end={norms[-1]:.5f}  "
          f"max_rel_dev={max(abs(n-R)/R for n in norms)*100:.4f}%")
    _, _, norms_off = _run(hyperball=False, capture={})
    print(f"  hyperball OFF: ||W|| start={norms_off[0]:.5f}  "
          f"mid={norms_off[len(norms_off)//2]:.5f}  end={norms_off[-1]:.5f}  (unconstrained)")
    sys.exit(1 if failed else 0)
