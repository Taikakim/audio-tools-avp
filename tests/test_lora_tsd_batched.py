"""Tests for the LoRA-TSD batched optimizer port (see
stable_audio_tools/training/lora_tsd/CONTRACT.md and
SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md).

Two independent things are validated:
  1. LoRATSDReference matches the UPSTREAM LoRA-TSD optimizer exactly (the oracle
     itself has to be trustworthy before it can be used to check the batched port).
  2. BatchedLoRATSD matches LoRATSDReference (the actual equivalence gate for the
     batched port). These tests are collected even if `batched.py` does not exist
     yet (parallel-worker development) -- they just skip.

CPU only, small shapes (r=8, several (n, m) including unequal-shape groups so
shape-based batching in BatchedLoRATSD is exercised).
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_LORA_TSD_DIR = Path(__file__).resolve().parents[1] / "stable_audio_tools" / "training" / "lora_tsd"
_UPSTREAM_SRC = Path("/home/kim/Projects/LoRA-TSD/src")


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_reference():
    """Import LoRATSDReference, tolerating a missing sibling batched.py.

    The package __init__.py does `from .batched import BatchedLoRATSD`, so a
    normal `from stable_audio_tools.training.lora_tsd import LoRATSDReference`
    fails to even import `reference` while batched.py doesn't exist yet (worker B
    writes it in parallel). Fall back to loading reference.py directly by path.
    """
    try:
        from stable_audio_tools.training.lora_tsd.reference import LoRATSDReference
        return LoRATSDReference
    except Exception:
        mod = _load_module_from_file("_lora_tsd_reference_standalone", _LORA_TSD_DIR / "reference.py")
        return mod.LoRATSDReference


def _load_batched():
    try:
        from stable_audio_tools.training.lora_tsd.batched import BatchedLoRATSD
        return BatchedLoRATSD
    except Exception:
        return None


def _load_upstream():
    if not _UPSTREAM_SRC.is_dir():
        return None
    sys.path.insert(0, str(_UPSTREAM_SRC))
    try:
        from optimizers.lora_tsd import LoRATSD
        return LoRATSD
    except Exception:
        return None


LoRATSDReference = _load_reference()
BatchedLoRATSD = _load_batched()
UpstreamLoRATSD = _load_upstream()

requires_batched = pytest.mark.skipif(
    BatchedLoRATSD is None, reason="stable_audio_tools/training/lora_tsd/batched.py not present yet"
)
requires_upstream = pytest.mark.skipif(
    UpstreamLoRATSD is None, reason="upstream LoRA-TSD not importable from /home/kim/Projects/LoRA-TSD/src"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# (r, n, m) shapes for pairs. Includes a duplicate shape (indices 0,1) so any
# shape-based batching groups >1 pair, plus several distinct shapes/unequal groups.
SHAPES = [
    (8, 20, 14),
    (8, 20, 14),
    (8, 12, 30),
    (8, 40, 8),
]


def build_named_params(shapes=SHAPES, seed=0, dtype=torch.float64, scale_a=0.5, scale_b=0.3,
                        n_extra_magnitude=0):
    """Build a real-style named-parameter list: blk{i}...lora_A/lora_B/magnitude.

    Returns (named_params, meta) where meta records shapes/prefixes for building
    matching gradients later.
    """
    g = torch.Generator().manual_seed(seed)
    named = []
    meta = []
    for i, (r, n, m) in enumerate(shapes):
        prefix = f"blk{i}.parametrizations.weight.0"
        A = (torch.randn(r, n, generator=g, dtype=torch.float64) * scale_a).to(dtype).requires_grad_(True)
        B = (torch.randn(m, r, generator=g, dtype=torch.float64) * scale_b).to(dtype).requires_grad_(True)
        named.append((f"{prefix}.lora_A", A))
        named.append((f"{prefix}.lora_B", B))
        mag = (torch.rand(m, generator=g, dtype=torch.float64) + 0.5).to(dtype).requires_grad_(True)
        named.append((f"{prefix}.magnitude", mag))
        meta.append({"prefix": prefix, "r": r, "n": n, "m": m})
    for j in range(n_extra_magnitude):
        m_len = 6 + j
        mag = (torch.rand(m_len, generator=g, dtype=torch.float64) + 0.5).to(dtype).requires_grad_(True)
        named.append((f"extra{j}.parametrizations.weight.0.magnitude", mag))
    return named, meta


def clone_named_params(named, dtype=None):
    out = []
    for name, p in named:
        t = p.detach().clone()
        if dtype is not None:
            t = t.to(dtype)
        out.append((name, t.requires_grad_(True)))
    return out


def make_grad_sequence(named, n_steps, seed=1234, dtype=torch.float64):
    """A fixed sequence of per-param gradients, reusable across two optimizer instances."""
    g = torch.Generator().manual_seed(seed)
    seq = []
    for _ in range(n_steps):
        step_grads = []
        for name, p in named:
            grad = torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype)
            step_grads.append(grad)
        seq.append(step_grads)
    return seq


def apply_grads(named, step_grads):
    for (name, p), grad in zip(named, step_grads):
        p.grad = grad.clone().to(p.dtype)


def as_state_dict(named):
    return {name: p.detach().clone() for name, p in named}


def _resolve_cls(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        return BatchedLoRATSD
    return LoRATSDReference


def _assert_updates_relatively_close(named_before, named_after_ref, named_after_bat, rtol, atol,
                                      msg_prefix=""):
    """Compare the UPDATE (Δ = after - before), not the raw parameter, relative-norm.

    Opus-critic finding (2026-09-24 fix round): comparing raw parameters after a few
    steps of lr=1e-3 dilutes rtol by ~300x, because the parameters are O(0.3) but each
    step's update is O(1e-3) -- a moderately wrong update (over-clip, squared-norm clip,
    a swapped AAt_i/BtB_i) barely moves the *parameter*-relative error, but is glaring on
    the *update*-relative error. This is the comparison that actually gates correctness;
    the raw-parameter check is kept alongside it as a cheap sanity check, not the gate.
    """
    before = {name: p.detach().clone() for name, p in named_before}
    after_ref = {name: p.detach().clone() for name, p in named_after_ref}
    after_bat = {name: p.detach().clone() for name, p in named_after_bat}
    for name in before:
        d_ref = after_ref[name] - before[name]
        d_bat = after_bat[name] - before[name]
        num = (d_ref - d_bat).norm()
        den = d_ref.norm()
        if den > 0:
            rel = (num / den).item()
            assert rel < rtol, (
                f"{msg_prefix}update mismatch on {name}: rel={rel} (||Δref||={den.item()}, "
                f"||Δref-Δbat||={num.item()})"
            )
        else:
            assert num.item() < atol, f"{msg_prefix}update mismatch on {name} (Δref≈0): abs={num.item()}"


# ---------------------------------------------------------------------------
# 1. reference vs upstream oracle
# ---------------------------------------------------------------------------

@requires_upstream
@pytest.mark.parametrize("ball_iters", [1, 3])
@pytest.mark.parametrize("balance", ["off", "norm"])
@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_reference_matches_upstream(ball_iters, balance, momentum):
    torch.manual_seed(0)
    r, n, m = 8, 17, 11
    A0 = torch.randn(r, n, dtype=torch.float64) * 0.5
    B0 = torch.randn(m, r, dtype=torch.float64) * 0.2

    A_up = A0.clone().requires_grad_(True)
    B_up = B0.clone().requires_grad_(True)
    up = UpstreamLoRATSD(
        [{"params": [A_up, B_up]}],
        use_momentum=momentum > 0, momentum_cf=momentum,
        lr_A=1e-3, lr_B=1e-3, ball_iters=ball_iters, ridge_eps=1e-8,
        max_delta_norm=0.1, ns_steps=5,
        balance=(balance == "norm"), balance_mode="norm", rebalance_every=1,
        log_optimizer_stats=False,
    )

    A_ref = A0.clone().requires_grad_(True)
    B_ref = B0.clone().requires_grad_(True)
    ref = LoRATSDReference(
        [("x.lora_A", A_ref), ("x.lora_B", B_ref)],
        lr=1e-3, momentum=momentum, ball_iters=ball_iters, ns_steps=5,
        max_delta_norm=0.1, balance=balance, ridge_eps=1e-8,
    )

    g = torch.Generator().manual_seed(99)
    for _ in range(5):
        gA = torch.randn(r, n, generator=g, dtype=torch.float64)
        gB = torch.randn(m, r, generator=g, dtype=torch.float64)
        A_up.grad = gA.clone(); B_up.grad = gB.clone()
        A_ref.grad = gA.clone(); B_ref.grad = gB.clone()
        up.step()
        ref.step()
        torch.testing.assert_close(A_up, A_ref, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(B_up, B_ref, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. batched vs reference equivalence (THE key test)
# ---------------------------------------------------------------------------

@requires_batched
@pytest.mark.parametrize("dtype,rtol,atol,update_rtol,update_atol", [
    (torch.float64, 1e-9, 1e-11, 1e-7, 1e-12),
    (torch.float32, 1e-4, 1e-6, 1e-3, 1e-8),
])
@pytest.mark.parametrize("ball_iters", [1, 3])
@pytest.mark.parametrize("balance", ["off", "norm"])
@pytest.mark.parametrize("momentum", [0.0, 0.95])
@pytest.mark.parametrize("warmup_steps", [0, 3])
def test_batched_matches_reference(dtype, rtol, atol, update_rtol, update_atol,
                                    ball_iters, balance, momentum, warmup_steps):
    named0, _ = build_named_params(dtype=dtype, n_extra_magnitude=2)
    named_ref = clone_named_params(named0)
    named_bat = clone_named_params(named0)

    kwargs = dict(lr=1e-3, momentum=momentum, ball_iters=ball_iters, ns_steps=5,
                  max_delta_norm=0.1, balance=balance, ridge_eps=1e-8, lr_magnitude=5e-4,
                  warmup_steps=warmup_steps)

    ref = LoRATSDReference(named_ref, **kwargs)
    bat = BatchedLoRATSD(named_bat, **kwargs)

    grad_seq = make_grad_sequence(named0, n_steps=4, dtype=dtype)
    for step_grads in grad_seq:
        apply_grads(named_ref, step_grads)
        apply_grads(named_bat, step_grads)
        ref.step()
        bat.step()

    # THE key comparison (see _assert_updates_relatively_close docstring): the update,
    # not the raw parameter -- catches a wrong clip/warmup/ridge that a raw-parameter
    # comparison dilutes into noise.
    _assert_updates_relatively_close(named0, named_ref, named_bat, update_rtol, update_atol,
                                      msg_prefix=f"(dtype={dtype}, warmup_steps={warmup_steps}) ")

    for (name_r, p_r), (name_b, p_b) in zip(named_ref, named_bat):
        assert name_r == name_b
        torch.testing.assert_close(p_r, p_b, rtol=rtol, atol=atol,
                                    msg=f"mismatch on {name_r} (dtype={dtype})")


# ---------------------------------------------------------------------------
# 3. zero-B first step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_zero_b_first_step(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    torch.manual_seed(0)
    r, n, m = 8, 16, 12
    A = (torch.randn(r, n, dtype=torch.float64) * 0.5).requires_grad_(True)
    B = torch.zeros(m, r, dtype=torch.float64, requires_grad=True)
    named = [("blk0.parametrizations.weight.0.lora_A", A),
             ("blk0.parametrizations.weight.0.lora_B", B)]
    opt = cls(named, lr=1e-3, momentum=0.95, ball_iters=1, ns_steps=5,
              max_delta_norm=0.1, balance="norm", ridge_eps=1e-8)

    A.grad = torch.randn(r, n, dtype=torch.float64)
    B.grad = torch.randn(m, r, dtype=torch.float64)
    opt.step()

    assert torch.isfinite(A).all(), f"{cls_name}: A not finite after zero-B first step"
    assert torch.isfinite(B).all(), f"{cls_name}: B not finite after zero-B first step"
    assert B.abs().sum() > 0, f"{cls_name}: B stayed exactly zero after the step"

    # second step should also stay finite
    A.grad = torch.randn(r, n, dtype=torch.float64)
    B.grad = torch.randn(m, r, dtype=torch.float64)
    opt.step()
    assert torch.isfinite(A).all() and torch.isfinite(B).all()


# ---------------------------------------------------------------------------
# 4. gauge equivariance (reference algorithm property, not a batched-vs-ref check)
# ---------------------------------------------------------------------------

def test_gauge_equivariance_reference():
    """A pure-gauge reparam of the inputs (A->XA, B->BX^-1, X near I) should leave
    the linearized weight update B'A'-BA agreeing with the ungauged BA-delta, since
    LoRA-TSD projects the weight-space gradient onto the tangent space of B@A
    before taking a spectral step. Measured to hold to ~1e-9 relative on the
    reference implementation (small gauge perturbation, no clipping) -- see the
    worker-A report. If a future change to reference.py breaks this, that is a
    real regression, not a flaky test; xfail was NOT needed here.
    """
    torch.manual_seed(7)
    r, n, m = 6, 10, 8
    A0 = torch.randn(r, n, dtype=torch.float64) * 0.5
    B0 = torch.randn(m, r, dtype=torch.float64) * 0.3
    dLdW = torch.randn(m, n, dtype=torch.float64) * 1e-2

    eps = 1e-3
    X = torch.eye(r, dtype=torch.float64) + torch.randn(r, r, dtype=torch.float64) * eps
    Xinv = torch.linalg.inv(X)

    def run_step(A, B, G_A, G_B):
        A = A.clone().requires_grad_(True)
        B = B.clone().requires_grad_(True)
        opt = LoRATSDReference(
            [("x.lora_A", A), ("x.lora_B", B)],
            lr=1e-3, momentum=0.0, ball_iters=1, ns_steps=5,
            max_delta_norm=0.1, balance="off", ridge_eps=1e-8,
        )
        A.grad = G_A.clone()
        B.grad = G_B.clone()
        opt.step()
        return A.detach(), B.detach()

    # Chain rule for factor grads of L(W), W = B@A: G_A = B^T dL/dW, G_B = dL/dW A^T.
    G_A = B0.t() @ dLdW
    G_B = dLdW @ A0.t()

    A0g = X @ A0
    B0g = B0 @ Xinv
    G_Ag = Xinv.t() @ G_A
    G_Bg = G_B @ X.t()

    # Sanity: the gauge transform preserves W0 exactly.
    torch.testing.assert_close(B0 @ A0, B0g @ A0g, rtol=0, atol=1e-9)

    A1, B1 = run_step(A0, B0, G_A, G_B)
    A1g, B1g = run_step(A0g, B0g, G_Ag, G_Bg)

    dW = B1 @ A1 - B0 @ A0
    dWg = B1g @ A1g - B0g @ A0g

    rel = (dW - dWg).norm() / dW.norm()
    assert rel < 1e-4, f"gauge-perturbed update diverged from ungauged update: rel={rel.item()}"


# ---------------------------------------------------------------------------
# 5. clip: ||dW_lin||_F <= max_delta_norm + eps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_clip_bound(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    max_delta_norm = 0.05
    named, meta = build_named_params(shapes=[(8, 20, 14), (8, 12, 30)], seed=3, dtype=torch.float64)
    before = as_state_dict(named)
    opt = cls(named, lr=1.0, momentum=0.0, ball_iters=1, ns_steps=5,
              max_delta_norm=max_delta_norm, balance="off", ridge_eps=1e-8)

    g = torch.Generator().manual_seed(5)
    for name, p in named:
        if p.grad is None and not name.endswith("magnitude"):
            pass
        p.grad = torch.randn(p.shape, generator=g, dtype=torch.float64) * 10.0  # large, to force clip
    opt.step()

    after = {name: p.detach() for name, p in named}
    for m_info in meta:
        prefix = m_info["prefix"]
        A_name, B_name = f"{prefix}.lora_A", f"{prefix}.lora_B"
        A0, B0 = before[A_name], before[B_name]
        A1, B1 = after[A_name], after[B_name]
        dA = A1 - A0
        dB = B1 - B0
        dW_lin = dB @ A0 + B0 @ dA
        norm = dW_lin.norm()
        assert norm <= max_delta_norm + 1e-3, f"{cls_name}: pair {prefix} clip violated: {norm.item()}"


# ---------------------------------------------------------------------------
# 6. magnitudes: stay positive; relative change identical across start values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_magnitude_positive_and_scale_invariant(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    torch.manual_seed(0)
    shape = (5,)
    m_a0 = 0.13
    m_b0 = 2.4
    m_a = torch.full(shape, m_a0, dtype=torch.float64, requires_grad=True)
    m_b = torch.full(shape, m_b0, dtype=torch.float64, requires_grad=True)
    named = [("blkA.parametrizations.weight.0.magnitude", m_a),
             ("blkB.parametrizations.weight.0.magnitude", m_b)]
    opt = cls(named, lr=1e-3, momentum=0.9, ball_iters=1, ns_steps=5,
              max_delta_norm=0.1, balance="off", ridge_eps=1e-8, lr_magnitude=1e-3)

    g = torch.Generator().manual_seed(42)
    n_steps = 1000
    for _ in range(n_steps):
        grad = torch.randn(shape, generator=g, dtype=torch.float64) + 1.0  # biased positive: consistent push toward zero
        m_a.grad = grad.clone()
        m_b.grad = grad.clone()
        opt.step()
        assert (m_a > 0).all(), f"{cls_name}: magnitude A went non-positive"
        assert (m_b > 0).all(), f"{cls_name}: magnitude B went non-positive"

    ratio_a = m_a.detach() / m_a0
    ratio_b = m_b.detach() / m_b0
    torch.testing.assert_close(ratio_a, ratio_b, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. state_dict round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_state_dict_round_trip(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    named0, _ = build_named_params(dtype=torch.float64)
    named_live = clone_named_params(named0)
    kwargs = dict(lr=1e-3, momentum=0.9, ball_iters=1, ns_steps=5,
                  max_delta_norm=0.1, balance="norm", ridge_eps=1e-8, lr_magnitude=5e-4)
    opt_live = cls(named_live, **kwargs)

    grad_seq = make_grad_sequence(named0, n_steps=6, dtype=torch.float64)
    k = 3
    for step_grads in grad_seq[:k]:
        apply_grads(named_live, step_grads)
        opt_live.step()

    # Snapshot params + optimizer state at step k.
    saved_state = opt_live.state_dict()
    saved_params = as_state_dict(named_live)

    # Take one more step on the live optimizer as the ground truth for "next step".
    apply_grads(named_live, grad_seq[k])
    opt_live.step()
    expected = {name: p.detach().clone() for name, p in named_live}

    # Rebuild fresh params at the saved values, a fresh optimizer, load state.
    named_resumed = [(name, saved_params[name].clone().requires_grad_(True)) for name, _ in named0]
    opt_resumed = cls(named_resumed, **kwargs)
    opt_resumed.load_state_dict(saved_state)

    apply_grads(named_resumed, grad_seq[k])
    opt_resumed.step()

    for name, p in named_resumed:
        torch.testing.assert_close(p.detach(), expected[name], rtol=1e-10, atol=1e-12,
                                    msg=f"{cls_name}: resumed step diverged on {name}")


# ---------------------------------------------------------------------------
# 8. unpaired non-magnitude param raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_unpaired_non_magnitude_raises(cls_name):
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    A = torch.randn(8, 10, dtype=torch.float64, requires_grad=True)
    stray = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    named = [("blk0.parametrizations.weight.0.lora_A", A),
             ("blk0.parametrizations.weight.0.some_other_tensor", stray)]
    with pytest.raises(ValueError):
        cls(named, lr=1e-3)


@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_half_pair_raises(cls_name):
    """A lora_A with no matching lora_B (or vice versa) is also an unpaired param."""
    if cls_name == "batched":
        if BatchedLoRATSD is None:
            pytest.skip("batched.py not present yet")
        cls = BatchedLoRATSD
    else:
        cls = LoRATSDReference

    A = torch.randn(8, 10, dtype=torch.float64, requires_grad=True)
    named = [("blk0.parametrizations.weight.0.lora_A", A)]
    with pytest.raises(ValueError):
        cls(named, lr=1e-3)


# ---------------------------------------------------------------------------
# Opus-critic fix round (2026-09-24): mutation testing on the ORIGINAL 8 tests
# above found 6 surviving mutants -- over-clip to 0.5x target, clip on the
# squared norm (missing sqrt), warmup ignored, warmup off-by-one, the ridge
# clamp applied in the wrong order (clamps the product instead of diag_mean
# alone -- differs once ||B||_F is small, i.e. our real B~=0 DoRA init), and
# LoRATSDReference's lr_magnitude=None default silently halved. Root causes
# (see docstrings below): the equivalence tests used lr=1e-3, so the clip path
# never activated; nothing set warmup_steps>0; nothing exercised B near zero
# with a class-vs-class comparison (test_zero_b_first_step only checks
# finiteness); nothing exercised the lr_magnitude=None default path (every
# existing test passes it explicitly); and raw-parameter comparisons (O(0.3))
# dilute a wrong per-step update (O(1e-3)) into a small relative error. Tests
# below close each gap. See NOTES in the worker-A report for the mutation
# table this was verified against.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (b) clip ACTIVE: the clipped update must land EXACTLY at max_delta_norm,
#     not merely <= it -- catches over-clip and clip-on-squared-norm, which
#     test_clip_bound's "<=" check cannot (both mutants still satisfy <=).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_clip_active_matches_target(cls_name):
    """dW_lin = dB@A0 + B0@dA is the LINEARIZED update the clip scales -- by
    construction (see reference.py's comment: "Reconstruct factor updates whose
    linearized product is -lr * L @ R", and X_fro is exactly ||L@R||_F via the
    trace identity ||L@R||_F^2 = sum((L^T L) o (R R^T))). So for a genuinely
    clipped pair, ||dW_lin||_F must equal max_delta_norm almost exactly (~1e-8
    relative, verified against the reference implementation), not merely be
    bounded by it. A 0.5x over-clip or a squared-norm clip both still satisfy
    "<=" (test_clip_bound) but fail this equality.
    """
    cls = _resolve_cls(cls_name)

    torch.manual_seed(4)
    r, n, m = 8, 20, 14
    max_delta_norm = 0.05
    A0 = torch.randn(r, n, dtype=torch.float64) * 0.5
    B0 = torch.randn(m, r, dtype=torch.float64) * 0.3
    A = A0.clone().requires_grad_(True)
    B = B0.clone().requires_grad_(True)
    named = [("blk0.parametrizations.weight.0.lora_A", A),
             ("blk0.parametrizations.weight.0.lora_B", B)]
    # lr=0.1 (not the equivalence tests' lr=1e-3): large enough that the
    # unclipped step reliably exceeds max_delta_norm=0.05, so the clip path
    # actually executes (this is exactly the gap the critic flagged).
    opt = cls(named, lr=0.1, momentum=0.0, ball_iters=1, ns_steps=5,
              max_delta_norm=max_delta_norm, balance="off", ridge_eps=1e-8)
    A.grad = torch.randn(r, n, dtype=torch.float64) * 5.0
    B.grad = torch.randn(m, r, dtype=torch.float64) * 5.0
    opt.step()

    assert opt.last_stats["clip_frac"] > 0.99, (
        f"{cls_name}: test setup didn't actually trigger the clip "
        f"(clip_frac={opt.last_stats['clip_frac']}) -- lr/grad scale needs raising"
    )

    dA = A.detach() - A0
    dB = B.detach() - B0
    dW_lin = dB @ A0 + B0 @ dA
    measured = dW_lin.norm().item()
    rel = abs(measured - max_delta_norm) / max_delta_norm
    assert rel < 1e-6, (
        f"{cls_name}: clipped update norm {measured} != max_delta_norm {max_delta_norm} "
        f"(rel={rel}) -- over-clip or clip-on-squared-norm bug"
    )


# ---------------------------------------------------------------------------
# (c) warmup_steps > 0: hand-computed expected trajectory at steps 1, 2, N.
#     Uses ONLY the magnitude path (deterministic: momentum=0 => buf=grad
#     exactly, constant +1 grad => sign is always +1), so the schedule itself
#     is isolated from any spectral-step numerics. lr and lr_magnitude share
#     the same warmup fraction (CONTRACT), so this also exercises the
#     fraction the spectral-pair path uses.
# ---------------------------------------------------------------------------

def _expected_warmup_frac(step_1indexed: int, warmup_steps: int) -> float:
    if not warmup_steps or warmup_steps <= 0:
        return 1.0
    return min(1.0, step_1indexed / warmup_steps)


@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_warmup_schedule_hand_computed(cls_name):
    cls = _resolve_cls(cls_name)

    lr = 2e-3
    warmup_steps = 5
    m0 = 1.0
    shape = (4,)
    m = torch.full(shape, m0, dtype=torch.float64, requires_grad=True)
    named = [("blk0.parametrizations.weight.0.magnitude", m)]
    opt = cls(named, lr=lr, momentum=0.0, ball_iters=1, ns_steps=5,
              max_delta_norm=0.1, balance="off", ridge_eps=1e-8,
              lr_magnitude=lr, warmup_steps=warmup_steps)

    N = 8
    log_ratio = 0.0
    for k in range(1, N + 1):
        m.grad = torch.ones(shape, dtype=torch.float64)  # buf==grad (momentum=0); sign is always +1
        opt.step()
        frac_k = _expected_warmup_frac(k, warmup_steps)
        # m_k = m_{k-1} * exp(-lr_magnitude_t * sign(buf)); sign==+1 every step here.
        log_ratio += -lr * frac_k
        if k in (1, 2, N):
            expected_m = m0 * float(torch.exp(torch.tensor(log_ratio, dtype=torch.float64)))
            torch.testing.assert_close(
                m.detach(), torch.full(shape, expected_m, dtype=torch.float64),
                rtol=1e-9, atol=1e-11,
                msg=(f"{cls_name}: warmup schedule diverged from hand-computed expectation "
                     f"at step {k} (frac should be {frac_k})"),
            )


# ---------------------------------------------------------------------------
# (d) B starts at (or very near) zero, balance="off", SEVERAL steps, compared
#     numerically ref-vs-batched (not just finiteness like test_zero_b_first_step)
#     -- catches a ridge-clamp order bug, which only differs from the
#     reference once the BtB/AAt diagonal mean is small (our real DoRA init).
# ---------------------------------------------------------------------------

@requires_batched
@pytest.mark.parametrize("b0_scale", [0.0, 1e-4])
@pytest.mark.parametrize("dtype,update_rtol,update_atol", [
    (torch.float64, 1e-6, 1e-9),
    (torch.float32, 1e-2, 1e-6),
])
def test_batched_matches_reference_near_zero_b(b0_scale, dtype, update_rtol, update_atol):
    torch.manual_seed(21)
    r, n, m = 8, 14, 10
    A0 = (torch.randn(r, n, dtype=torch.float64) * 0.4).to(dtype)
    B0 = (torch.zeros(m, r, dtype=torch.float64) if b0_scale == 0.0
          else torch.randn(m, r, dtype=torch.float64) * b0_scale).to(dtype)
    named0 = [("blk0.parametrizations.weight.0.lora_A", A0.clone().requires_grad_(True)),
              ("blk0.parametrizations.weight.0.lora_B", B0.clone().requires_grad_(True))]
    named_ref = clone_named_params(named0)
    named_bat = clone_named_params(named0)

    kwargs = dict(lr=1e-3, momentum=0.9, ball_iters=2, ns_steps=5, max_delta_norm=0.1,
                  balance="off", ridge_eps=1e-8)
    ref = LoRATSDReference(named_ref, **kwargs)
    bat = BatchedLoRATSD(named_bat, **kwargs)

    grad_seq = make_grad_sequence(named0, n_steps=6, seed=77, dtype=dtype)
    for step_grads in grad_seq:
        apply_grads(named_ref, step_grads)
        apply_grads(named_bat, step_grads)
        ref.step()
        bat.step()

    if dtype == torch.float32 and b0_scale > 0:
        # Near B ~ 0 the algorithm is ill-conditioned in fp32 (BtB^-1 ~ 1e8 amplifies
        # rounding in dA): the fp32 REFERENCE is itself ~10% off the fp64 answer here
        # (measured 2026-09-25), so ref-vs-batched at 1e-2 is not a meaningful bar.
        # Instead: both fp32 paths vs fp64 truth, batched no worse than 1.5x the oracle.
        named_t = clone_named_params([(n_, p.detach().double()) for n_, p in named0])
        tru = LoRATSDReference(named_t, **kwargs)
        for step_grads in make_grad_sequence(named_t, n_steps=6, seed=77, dtype=torch.float64):
            apply_grads(named_t, step_grads)
            tru.step()
        for (nm, p0), (_, pt), (_, pr), (_, pb) in zip(named0, named_t, named_ref, named_bat):
            d_t = pt.detach() - p0.detach().double()
            e_ref = float((pr.detach().double() - p0.detach().double() - d_t).norm() / d_t.norm())
            e_bat = float((pb.detach().double() - p0.detach().double() - d_t).norm() / d_t.norm())
            assert e_bat <= 1.5 * e_ref + 1e-4, f"{nm}: batched err {e_bat:.3e} vs reference err {e_ref:.3e}"
        return

    _assert_updates_relatively_close(named0, named_ref, named_bat, update_rtol, update_atol,
                                      msg_prefix=f"(b0_scale={b0_scale}, dtype={dtype}) ")


# ---------------------------------------------------------------------------
# (e) lr_magnitude=None must resolve to EXACTLY lr, not lr*0.5 or anything else.
#     Value check via a deterministic closed-form single step (momentum=0 =>
#     buf=grad exactly; grad=+1 => sign=+1 exactly), not just "it runs".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_lr_magnitude_default_equals_lr(cls_name):
    cls = _resolve_cls(cls_name)

    lr = 3e-3
    m0 = 1.0
    shape = (6,)
    m = torch.full(shape, m0, dtype=torch.float64, requires_grad=True)
    named = [("blk0.parametrizations.weight.0.magnitude", m)]
    # lr_magnitude intentionally OMITTED -> must default to lr exactly (CONTRACT).
    opt = cls(named, lr=lr, momentum=0.0, ball_iters=1, ns_steps=5,
              max_delta_norm=0.1, balance="off", ridge_eps=1e-8)

    m.grad = torch.ones(shape, dtype=torch.float64)
    opt.step()

    # m_1 = m_0 * exp(-lr_magnitude_used * sign(+1)) = m_0 * exp(-lr) IFF lr_magnitude
    # correctly resolved to lr. A lr*0.5 bug gives exp(-lr/2) instead.
    expected = m0 * float(torch.exp(torch.tensor(-lr, dtype=torch.float64)))
    torch.testing.assert_close(
        m.detach(), torch.full(shape, expected, dtype=torch.float64),
        rtol=1e-10, atol=1e-12,
        msg=f"{cls_name}: lr_magnitude=None did not resolve to lr exactly",
    )


# ---------------------------------------------------------------------------
# (f) construction must raise when rank r exceeds A's columns (n) or B's rows
#     (m) -- without the guard, torch.linalg.qr(mode="reduced") on the wide
#     A^T/B silently returns a Q with FEWER than r columns and every
#     downstream shape assumption desyncs, producing wrong numbers with no
#     error rather than a clean failure (verified: reference.py ran to
#     completion with n<r before this guard was added, see worker-A report).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
@pytest.mark.parametrize("bad_dim", ["n", "m"])
def test_rank_exceeds_n_or_m_raises(cls_name, bad_dim):
    cls = _resolve_cls(cls_name)

    r = 8
    n = 5 if bad_dim == "n" else 20
    m = 20 if bad_dim == "n" else 5
    A = (torch.randn(r, n, dtype=torch.float64) * 0.4).requires_grad_(True)
    B = (torch.randn(m, r, dtype=torch.float64) * 0.3).requires_grad_(True)
    named = [("blk0.parametrizations.weight.0.lora_A", A),
             ("blk0.parametrizations.weight.0.lora_B", B)]
    with pytest.raises(ValueError):
        cls(named, lr=1e-3)


# ---------------------------------------------------------------------------
# (g) state_dict round trip THROUGH REAL SERIALIZATION (torch.save/torch.load
#     over bytes, then explicit .cpu() on every loaded tensor) -- a stronger
#     version of test_state_dict_round_trip's in-memory check. This is the
#     realistic "saved to disk/CPU and loaded back" path a training resume
#     actually takes, and it forces every state tensor through an independent
#     copy (serialization cannot alias), so it cannot pass by accident the way
#     an in-memory dict COULD if a future change reintroduced aliasing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_state_dict_round_trip_via_serialization(cls_name):
    import io

    cls = _resolve_cls(cls_name)

    named0, _ = build_named_params(dtype=torch.float64)
    named_live = clone_named_params(named0)
    kwargs = dict(lr=1e-3, momentum=0.9, ball_iters=1, ns_steps=5,
                  max_delta_norm=0.1, balance="norm", ridge_eps=1e-8, lr_magnitude=5e-4)
    opt_live = cls(named_live, **kwargs)

    grad_seq = make_grad_sequence(named0, n_steps=6, dtype=torch.float64)
    k = 3
    for step_grads in grad_seq[:k]:
        apply_grads(named_live, step_grads)
        opt_live.step()

    # Serialize through actual bytes -- torch.save/load, not a Python reference.
    buf = io.BytesIO()
    torch.save(opt_live.state_dict(), buf)
    saved_params = as_state_dict(named_live)

    # Keep training the live optimizer -- if the serialized snapshot aliased
    # anything, this would corrupt it (exactly the bug worker-A found and
    # fixed in reference.py's state_dict()).
    apply_grads(named_live, grad_seq[k])
    opt_live.step()
    expected = {name: p.detach().clone() for name, p in named_live}

    buf.seek(0)
    loaded_state = torch.load(buf, weights_only=False)
    # Explicitly move every tensor in the loaded state through an independent
    # .cpu() clone, simulating a real cross-device (or just cross-process)
    # resume rather than reusing whatever object torch.load happened to hand back.
    for v in loaded_state.get("state", {}).values():
        if isinstance(v, dict):
            for kk, vv in list(v.items()):
                if torch.is_tensor(vv):
                    v[kk] = vv.cpu().clone()

    named_resumed = [(name, saved_params[name].clone().requires_grad_(True)) for name, _ in named0]
    opt_resumed = cls(named_resumed, **kwargs)
    opt_resumed.load_state_dict(loaded_state)

    apply_grads(named_resumed, grad_seq[k])
    opt_resumed.step()

    for name, p in named_resumed:
        torch.testing.assert_close(p.detach(), expected[name], rtol=1e-10, atol=1e-12,
                                    msg=f"{cls_name}: serialized resume diverged on {name}")


# ---------------------------------------------------------------------------
# (f) CholeskyQR2 must not fall back on an fp32 input of cond ~3e4 -- the regime the
#     critic measured on real 2r-wide L/R factors (20-80 fallbacks/step with an fp32
#     Gram, which squares cond to ~1e9). The Gram runs in fp64; Q must still be
#     orthonormal and reproduce Y.
# ---------------------------------------------------------------------------

@requires_batched
def test_batched_qr_fp32_moderately_ill_conditioned_no_fallback():
    from stable_audio_tools.training.lora_tsd.batched import _batched_qr
    torch.manual_seed(5)
    N, m, r = 4, 96, 16
    U, _ = torch.linalg.qr(torch.randn(N, m, r, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(N, r, r, dtype=torch.float64))
    S = torch.logspace(0, -4.5, r, dtype=torch.float64)
    Y = (U * S) @ V.transpose(-1, -2)
    Q, R, n_fb = _batched_qr(Y.float())
    assert int(n_fb) == 0, f"{int(n_fb)} fallbacks on a cond~3e4 fp32 input"
    eye = torch.eye(r)
    assert float((Q.transpose(-1, -2) @ Q - eye).abs().max()) < 1e-4
    assert float((Q @ R - Y.float()).norm() / Y.float().norm()) < 1e-5


# ---------------------------------------------------------------------------
# (g) load_state_dict must not adopt the caller's state tensors by reference: a later
#     in-place momentum update would otherwise rewrite the dict the caller still holds
#     (final critic 2026-09-25; torch only deep-copies param_groups).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_load_state_dict_does_not_alias_callers_tensors(cls_name):
    cls = _resolve_cls(cls_name)
    named0, _ = build_named_params(dtype=torch.float64)
    kwargs = dict(lr=1e-3, momentum=0.9, ball_iters=1, ns_steps=5,
                  max_delta_norm=0.1, balance="norm", ridge_eps=1e-8)
    named_a = clone_named_params(named0)
    opt_a = cls(named_a, **kwargs)
    grad_seq = make_grad_sequence(named0, n_steps=3, dtype=torch.float64)
    for g in grad_seq[:2]:
        apply_grads(named_a, g)
        opt_a.step()
    sd = opt_a.state_dict()
    before = {k: {kk: vv.clone() for kk, vv in v.items() if torch.is_tensor(vv)} for k, v in sd["state"].items()}
    named_b = clone_named_params(named_a)
    opt_b = cls(named_b, **kwargs)
    opt_b.load_state_dict(sd)
    apply_grads(named_b, grad_seq[2])
    opt_b.step()
    for k, v in before.items():
        for kk, vv in v.items():
            assert torch.equal(sd["state"][k][kk], vv), f"state {k}/{kk} changed under the caller"


# ---------------------------------------------------------------------------
# (h) Cloud-review fixes (2026-09-25): hyperparameters come from param_groups (so a torch
#     LR scheduler works), mixed ranks fail loudly at construction, mixed-dtype
#     magnitudes don't break the flat magnitude step.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", ["reference", "batched"])
def test_lr_scheduler_changes_the_step(cls_name):
    cls = _resolve_cls(cls_name)
    named0, _ = build_named_params(dtype=torch.float64)
    grad_seq = make_grad_sequence(named0, n_steps=2, dtype=torch.float64)
    kwargs = dict(lr=1e-3, momentum=0.0, ball_iters=1, ns_steps=5,
                  max_delta_norm=10.0, balance="off", ridge_eps=1e-8)

    def run(factor):
        named = clone_named_params(named0)
        opt = cls(named, **kwargs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e == 0 else factor)
        apply_grads(named, grad_seq[0]); opt.step(); sched.step()
        before = {n: p.detach().clone() for n, p in named}
        apply_grads(named, grad_seq[1]); opt.step()
        return {n: (p.detach() - before[n]) for n, p in named if n.endswith(("lora_A", "lora_B"))}

    full, half = run(1.0), run(0.5)
    for n in full:
        # momentum 0 and no clip: the step is linear in lr, so half the lr = half the step
        assert torch.allclose(half[n], 0.5 * full[n], rtol=1e-6, atol=1e-12), n
    zero = run(0.0)
    assert all(float(d.abs().max()) == 0.0 for d in zero.values())


@requires_batched
def test_batched_rejects_mixed_ranks():
    named = [("a.lora_A", torch.nn.Parameter(torch.randn(4, 16))),
             ("a.lora_B", torch.nn.Parameter(torch.randn(12, 4))),
             ("b.lora_A", torch.nn.Parameter(torch.randn(8, 16))),
             ("b.lora_B", torch.nn.Parameter(torch.randn(12, 8)))]
    with pytest.raises(ValueError, match="one rank"):
        BatchedLoRATSD(named)


@requires_batched
def test_batched_magnitude_buffer_follows_compute_dtype():
    """A buffer made while only the fp32 magnitude had a grad must be promoted once an
    fp64 one joins (compute goes fp64). torch's type promotion hid this -- no crash, just
    an fp32 buffer in an fp64 step -- so check the dtype, not merely that it runs."""
    import math
    m32 = torch.nn.Parameter(torch.ones(5, dtype=torch.float32))
    m64 = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    opt = BatchedLoRATSD([("x.magnitude", m32), ("y.magnitude", m64)], lr=1e-2, momentum=0.9)
    m32.grad = torch.ones_like(m32)
    opt.step()  # fp32-only step
    assert opt.state[m32]["momentum_buffer"].dtype == torch.float32
    m32.grad = torch.ones_like(m32)
    m64.grad = -torch.ones_like(m64)
    opt.step()  # mixed -> fp64 compute
    assert opt.state[m32]["momentum_buffer"].dtype == torch.float64
    assert torch.allclose(m32.detach().double(), torch.full((5,), math.exp(-2e-2), dtype=torch.float64), rtol=1e-5)
    assert torch.allclose(m64.detach(), torch.full((3,), math.exp(1e-2), dtype=torch.float64), rtol=1e-9)
