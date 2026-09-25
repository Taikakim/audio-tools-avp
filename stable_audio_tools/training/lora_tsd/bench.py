"""bench.py -- timing + correctness harness for the batched LoRA-TSD optimizer port.

PURPOSE
  Times LoRATSDReference (worker A), BatchedLoRATSD (worker B), and upstream LoRATSD
  (brain-lab-research/LoRA-TSD, unbatched -- the thing we are replacing) on our real
  229 DoRA-adapter (A, B) shape pairs (r=128, from an actual SA3 medium-base full-DoRA
  checkpoint -- see the 229-pair shape table in
  SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md, section 3) plus 229
  `*.magnitude` vectors. Also spot-checks batched-vs-reference numerical correctness:
  a NORMWISE relative error of the parameter UPDATE (not the raw parameter value), so a
  fast-but-wrong batched implementation shows up here rather than three hours into a
  training run. (Why normwise-of-the-update and not elementwise-of-the-value: see the
  correctness_check() docstring below -- the elementwise version reads 7e-4 on an
  implementation whose true normwise update error is 1.4e-8, because it's dominated by
  near-zero entries where a tiny absolute difference is a huge relative one.)

MEASURED BASELINE (upstream LoRATSD, unbatched, one param group per adapter, on our
RX 9070 XT / ROCm box -- see the handover, section 1):
    ball_iters=1:   8.5 s / optimizer step
    ball_iters=5:  31   s / optimizer step   (the paper's own rebalance-every-step setting)
  Cause: ~6 small linalg calls per adapter per step, each latency-bound on ROCm
  (torch.linalg.qr alone costs 12-17 ms/call regardless of matrix size on this stack).
  Batching target: well under 0.5 s/step for all 229 adapters together.

HOW TO RUN ON GPU
    cd /home/kim/Projects/SAO/stable-audio-tools
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
    /home/kim/Projects/SAO/.venv/bin/python -m stable_audio_tools.training.lora_tsd.bench --device cuda

  (equivalently, from any cwd:
    PYTHONPATH=/home/kim/Projects/SAO/stable-audio-tools \
        /home/kim/Projects/SAO/.venv/bin/python -m stable_audio_tools.training.lora_tsd.bench --device cuda)

  This box's GPU is a shared single-card mutex -- ACQUIRE THE LOCK FIRST, every time:
    /home/kim/Projects/SAO/Misc/gpu_guard.sh acquire <YOURHANDLE> $$
  and release it when done:
    /home/kim/Projects/SAO/Misc/gpu_guard.sh release <YOURHANDLE>
  Never run --device cuda without holding it -- see MASTER.md "GPU lock" for why.
  NOTE ON THE PEAK-MEMORY NUMBER ON GPU: it is this bench process ALONE on an otherwise
  empty card. In the real training job the optimizer state shares VRAM with a RESIDENT
  1.4B-param DiT + activations -- headroom next to THAT is the real deploy-time
  constraint, not this number. Treat it as an upper bound on the optimizer's own
  footprint, not as "will this fit".

HOW TO SMOKE-TEST ON CPU (fast, no GPU/lock needed)
    /home/kim/Projects/SAO/.venv/bin/python -m stable_audio_tools.training.lora_tsd.bench \\
        --device cpu --scale 0.05 --pairs-subset 20 --steps 1 --allow-rank-deficient

  (--allow-rank-deficient is required at --scale 0.05 because it shrinks n/m below
  r=128 -- see "RANK-DEFICIENT SHAPES" below. For a shape-safe fast smoke test, use
  --pairs-subset 13 at --scale 1.0 instead -- with the diversity-first pair ordering
  (below) that is exactly one pair of EVERY one of the 13 real shapes, full-size, so
  n/m >= r always and nothing is scaled down.)

CLI
  --device {cpu,cuda}                          default: cpu
  --ball-iters N [N ...]                       default: 1 5
  --steps N                                    timed steps (after 1 warmup step), default: 3
  --scale F                                    shrinks every shape dim EXCEPT r (fixed at 128)
  --impl {reference,batched,upstream} [...]    default: every impl that imports cleanly
  --pairs-subset N                             use only the first N pairs (diversity-first order,
                                                see expand_shapes())
  --grads {iid,structured}                     default: structured (see "GRADIENT MODES" below)
  --allow-rank-deficient                       bench pairs with n<r or m<r anyway (diagnostic only)

Each of --impl reference / --impl batched imports
`stable_audio_tools.training.lora_tsd.{reference,batched}`, which routes through this
package's `__init__.py` (`from .reference import LoRATSDReference; from .batched import
BatchedLoRATSD`) -- so if EITHER file is missing or broken, importing either name fails
and both are reported as skipped (the package failed to initialize, not necessarily the
specific file you asked for). Run with `--impl upstream` alone to still get the baseline
number while the other two are unfinished.

RANK-DEFICIENT SHAPES (Opus critic review, 2026-09-24). r=128 is FIXED by --scale (only
n and m shrink). Once --scale pushes n or m below r, A@A.T / B.T@B are singular or
numerically garbage by construction -- NOT a batching bug, a shape the algorithm was
never meant to run on. Measured: at --scale 0.05 (n,m both drop to ~13-77, well under
r=128) both reference and batched are "numerically chaotic" there (normwise update error
~1.28, i.e. the two updates are unrelated, not just imprecise). Worker B's
BatchedLoRATSD (and LoRATSDReference) construction is expected to raise ValueError for
any pair with n<r or m<r for exactly this reason -- bench.py checks for it BEFORE
building anything and refuses (or, with --allow-rank-deficient, warns loudly) rather
than reporting a meaningless comparison as if it were data.

GRADIENT MODES (Opus critic review, 2026-09-24). --grads iid (the original bench)
assigns i.i.d. N(0,1) grads directly to A, B and magnitude every step -- cheap, but it
HIDES the QR-fallback path: with iid grads and iid B, B^T B is essentially never
singular, so the reference/batched CholeskyQR2->real-QR fallback almost never fires
(0-2/step measured). --grads structured (the default) instead simulates what training
actually produces: B starts at ZERO (our real DoRA init) and A is kaiming-like
(uniform(-1,1)/sqrt(n)); each step draws a random low-rank-plus-noise WEIGHT-SPACE
gradient G_W = low_rank(rank 16) + 0.1*noise, and projects it onto the current factors,
G_A = B^T G_W, G_B = G_W A^T -- exactly the DoRA loss's actual gradient shape. This
exposes the fallback path for real (measured: 20-80 QR fallbacks/step on the small
256/257-dim pairs, vs 0-2 under iid). At step 1, B=0 makes G_A=0 -- a real, deliberately
kept edge case (the handover section 3 QR bullet: "the first step with B=0 must work,
because that's our init"); B starts moving on step 1 (G_B is nonzero, since A isn't
zero), so A only starts moving once B has grown -- run more than one step to see the
adapter actually training.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
import tracemalloc

import torch
import torch.nn as nn

# --- the 229 real (A, B) shape pairs, r=128 throughout (handover section 3) ---------------
# (count, A_shape=(r, n), B_shape=(m, r))
SHAPE_TABLE = [
    (76, (128, 1536), (1536, 128)),
    (24, (128, 1536), (7680, 128)),
    (24, (128, 1536), (3072, 128)),
    (24, (128, 1536), (4608, 128)),
    (24, (128, 1536), (12288, 128)),
    (24, (128, 6144), (1536, 128)),
    (24, (128, 257), (1536, 128)),
    (2, (128, 256), (1536, 128)),
    (2, (128, 768), (1536, 128)),
    (2, (128, 256), (256, 128)),
    (1, (128, 1536), (256, 128)),
    (1, (128, 1536), (9216, 128)),
    (1, (128, 256), (768, 128)),
]
N_PAIRS_FULL = sum(c for c, _, _ in SHAPE_TABLE)
assert N_PAIRS_FULL == 229, N_PAIRS_FULL
N_UNIQUE_SHAPES = len(SHAPE_TABLE)  # 13

UPSTREAM_SRC = "/home/kim/Projects/LoRA-TSD/src"

# Fixed bench hyperparameters (not exposed on the CLI -- same for every impl/ball_iters
# combo so the comparison is apples-to-apples; values are CONTRACT.md's own defaults,
# except lr which uses the handover's realistic modular-run scale, not the useless 5e-6
# trainer default).
BENCH_LR = 5e-4
BENCH_MOMENTUM = 0.95
BENCH_NS_STEPS = 5
BENCH_MAX_DELTA_NORM = 0.1
BENCH_BALANCE = "norm"
BENCH_RIDGE_EPS = 1e-8

# Structured-gradient shape (handover-realistic: low-rank-plus-noise weight gradient).
STRUCTURED_GRAD_RANK = 16
STRUCTURED_GRAD_SCALE = 1e-4
STRUCTURED_GRAD_NOISE = 0.1


# --- shape table -> concrete random pairs --------------------------------------------------

def _scale_shape_pair(a_shape, b_shape, scale: float):
    r, n = a_shape
    m, r2 = b_shape
    assert r == r2, (a_shape, b_shape)
    if scale == 1.0:
        return a_shape, b_shape
    n2 = max(1, round(n * scale))
    m2 = max(1, round(m * scale))
    return (r, n2), (m2, r)


def expand_shapes(scale: float = 1.0, subset: int | None = None):
    """Expand SHAPE_TABLE into the flat 229-entry [(A_shape, B_shape), ...] list, scaling
    every dim except r. DIVERSITY-FIRST ORDER: the first N_UNIQUE_SHAPES (13) entries are
    one-of-each real shape (table order), and every remaining repeat is appended after --
    so `--pairs-subset 13` is exactly "one of each real shape" (full multiplicities are
    preserved for larger/no subset: 76+24*6+24+2*3+1*3 = 229 total either way)."""
    unique = []
    repeats = []
    for count, a_shape, b_shape in SHAPE_TABLE:
        scaled = _scale_shape_pair(a_shape, b_shape, scale)
        unique.append(scaled)
        repeats.extend([scaled] * (count - 1))
    out = unique + repeats
    if subset is not None:
        out = out[:subset]
    return out


def find_rank_deficient(shapes, r: int = 128):
    """Pairs where n<r or m<r -- A@A.T / B.T@B are singular or numerically garbage there
    BY CONSTRUCTION, not because of a batching bug. Returns [(index, a_shape, b_shape), ...]."""
    bad = []
    for i, (a_shape, b_shape) in enumerate(shapes):
        n = a_shape[1]
        m = b_shape[0]
        if n < r or m < r:
            bad.append((i, a_shape, b_shape))
    return bad


def build_pairs(shapes, device: str, dtype=torch.float32, seed: int = 0, init: str = "iid"):
    """Build fresh (A, B, magnitude) triples for every shape pair. Deterministic for a given
    (shapes, device, dtype, seed, init).

    init="iid":        A ~ N(0, 0.05), B ~ N(0, 0.01) -- the original bench init.
    init="structured":  A = kaiming-like uniform(-1,1)/sqrt(n), B = 0 -- our real DoRA init
                        (r128 adapters start with B=0), for use with --grads structured.
    magnitude ~ U(0.1, 3) either way, length = B's first dim (m)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pairs = []
    for i, (a_shape, b_shape) in enumerate(shapes):
        n = a_shape[1]
        m = b_shape[0]
        if init == "iid":
            A = torch.randn(a_shape, generator=gen, device=device, dtype=dtype) * 0.05
            B = torch.randn(b_shape, generator=gen, device=device, dtype=dtype) * 0.01
        elif init == "structured":
            A = (torch.rand(a_shape, generator=gen, device=device, dtype=dtype) * 2 - 1) / math.sqrt(n)
            B = torch.zeros(b_shape, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        mag = torch.rand(m, generator=gen, device=device, dtype=dtype) * 2.9 + 0.1
        pairs.append({
            "name": f"layer{i}.parametrizations.weight.0",
            "A": nn.Parameter(A),
            "B": nn.Parameter(B),
            "mag": nn.Parameter(mag),
        })
    return pairs


def named_params_list(pairs):
    """(name, Parameter) pairs matching the CONTRACT pairing convention -- prefix before
    `.lora_A` / `.lora_B` / `.magnitude`."""
    out = []
    for p in pairs:
        out.append((p["name"] + ".lora_A", p["A"]))
        out.append((p["name"] + ".lora_B", p["B"]))
        out.append((p["name"] + ".magnitude", p["mag"]))
    return out


def assign_random_grads(pairs, generator: torch.Generator):
    """--grads iid: i.i.d. N(0,1) grads on every tensor. Cheap, but hides the QR-fallback
    path (see module docstring, GRADIENT MODES)."""
    for p in pairs:
        for key in ("A", "B", "mag"):
            param = p[key]
            param.grad = torch.randn(
                param.shape, generator=generator, device=param.device, dtype=param.dtype
            )


def assign_structured_grads(pairs, generator: torch.Generator, rank: int = STRUCTURED_GRAD_RANK,
                             weight_grad_scale: float = STRUCTURED_GRAD_SCALE,
                             noise_scale: float = STRUCTURED_GRAD_NOISE):
    """--grads structured (default): a random low-rank-plus-noise weight-space gradient G_W,
    projected onto the CURRENT A/B -- G_A = B^T G_W, G_B = G_W A^T. See module docstring,
    GRADIENT MODES, for why this (not iid) is the realistic test."""
    for p in pairs:
        m, r = p["B"].shape
        n = p["A"].shape[1]
        device, dtype = p["B"].device, p["B"].dtype
        GW = (
            torch.randn(m, rank, generator=generator, device=device, dtype=dtype)
            @ torch.randn(rank, n, generator=generator, device=device, dtype=dtype)
            + noise_scale * torch.randn(m, n, generator=generator, device=device, dtype=dtype)
        ) * weight_grad_scale
        p["A"].grad = p["B"].detach().t() @ GW
        p["B"].grad = GW @ p["A"].detach().t()
        p["mag"].grad = torch.randn(m, generator=generator, device=device, dtype=dtype)


def assign_grads(pairs, generator: torch.Generator, mode: str):
    if mode == "iid":
        assign_random_grads(pairs, generator)
    elif mode == "structured":
        assign_structured_grads(pairs, generator)
    else:
        raise ValueError(f"unknown grads mode: {mode!r}")


# --- lazy impl import -----------------------------------------------------------------------

def _import_attr(module_name: str, attr_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _import_upstream():
    if UPSTREAM_SRC not in sys.path:
        sys.path.insert(0, UPSTREAM_SRC)
    from optimizers.lora_tsd import LoRATSD  # noqa: E402  (path inserted just above)
    return LoRATSD


IMPL_IMPORTERS = {
    "reference": lambda: _import_attr("stable_audio_tools.training.lora_tsd.reference", "LoRATSDReference"),
    "batched": lambda: _import_attr("stable_audio_tools.training.lora_tsd.batched", "BatchedLoRATSD"),
    "upstream": _import_upstream,
}


def get_impl(name: str):
    """Returns (cls, None) on success or (None, reason) on failure. Never raises."""
    try:
        return IMPL_IMPORTERS[name](), None
    except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a skip-and-report path
        return None, f"{type(e).__name__}: {e}"


# --- optimizer construction per impl ---------------------------------------------------------

def build_ours(cls, pairs, ball_iters: int):
    """Construct LoRATSDReference / BatchedLoRATSD (identical constructor signature, CONTRACT.md)."""
    return cls(
        named_params_list(pairs),
        lr=BENCH_LR,
        momentum=BENCH_MOMENTUM,
        ball_iters=ball_iters,
        ns_steps=BENCH_NS_STEPS,
        max_delta_norm=BENCH_MAX_DELTA_NORM,
        balance=BENCH_BALANCE,
        ridge_eps=BENCH_RIDGE_EPS,
        lr_magnitude=None,
        warmup_steps=0,
    )


def build_upstream(cls, pairs, ball_iters: int):
    """One param group [A, B] per pair, upstream's own kwarg names (handover section 3 / 2)."""
    groups = [{"params": [p["A"], p["B"]]} for p in pairs]
    return cls(
        groups,
        lr_A=BENCH_LR,
        lr_B=BENCH_LR,
        use_momentum=BENCH_MOMENTUM > 0,
        momentum_cf=BENCH_MOMENTUM,
        ball_iters=ball_iters,
        ridge_eps=BENCH_RIDGE_EPS,
        max_delta_norm=BENCH_MAX_DELTA_NORM,
        ns_steps=BENCH_NS_STEPS,
        balance=(BENCH_BALANCE == "norm"),
        balance_mode="norm",
        rebalance_every=1,
        log_ns_gap=False,
        log_f_inner=False,
        log_grads=False,
        log_optimizer_stats=False,
    )


# --- timing ------------------------------------------------------------------------------

def _print_batched_stats(opt, ball_iters: int, step_label: str):
    stats = getattr(opt, "last_stats", None)
    if not stats:
        return
    qrf = stats.get("qr_fallbacks", "n/a")
    cf = stats.get("clip_frac", "n/a")
    dwn = stats.get("dW_norm_mean", "n/a")
    print(f"    [batched stats] ball_iters={ball_iters} step={step_label} "
          f"qr_fallbacks={qrf} clip_frac={cf} dW_norm_mean={dwn}")


def time_impl(name: str, cls, pairs, ball_iters: int, steps: int, device: str,
              grads: str = "structured", seed: int = 42):
    build_fn = build_upstream if name == "upstream" else build_ours
    opt = build_fn(cls, pairs, ball_iters)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if device == "cpu":
        tracemalloc.start()

    # one warm-up step (first-call overhead: kernel selection, lazy allocations, ...)
    assign_grads(pairs, gen, grads)
    opt.step()
    _print_batched_stats(opt, ball_iters, "warmup")
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elif device == "cpu":
        tracemalloc.reset_peak()

    times_ms = []
    for step_i in range(steps):
        assign_grads(pairs, gen, grads)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        _print_batched_stats(opt, ball_iters, str(step_i))

    mean_ms = sum(times_ms) / len(times_ms)
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif device == "cpu":
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 ** 2)
    else:
        peak_mb = None
    return mean_ms, peak_mb


# --- correctness spot-check ----------------------------------------------------------------

def correctness_check(ref_cls, batched_cls, shapes, device: str, ball_iters: int = 1,
                       grads: str = "structured", init_seed: int = 123, grad_seed: int = 999,
                       warm_steps: int = 3):
    """Run `warm_steps` steps from IDENTICAL initial params and IDENTICAL per-step grads for
    reference and batched (so B has grown away from its zero DoRA init under --grads
    structured -- see module docstring), then compare the FINAL step's parameter UPDATE
    (not the parameter value) tensor-by-tensor:

        error(tensor) = ||delta_ref - delta_batched||_2 / ||delta_ref||_2

    where delta = (post-step value) - (pre-step value), for each of A, B, magnitude, per
    pair. Returns (max_error, median_error) over all those per-tensor errors.

    WHY NORMWISE-OF-THE-UPDATE, NOT ELEMENTWISE-OF-THE-VALUE (Opus critic review,
    2026-09-24): the original version of this check computed max_i |a_i - b_i| /
    max(|a_i|, eps) over the raw parameter VALUES. That metric is dominated by
    near-zero entries -- at --scale 1.0 it read 7e-4 while the true normwise update
    error for the same run was 1.4e-8, a 5-order-of-magnitude overstatement driven
    entirely by entries where a tiny absolute difference divided by a tiny |a_i|
    looks huge. It also can't distinguish a real bug from rank-deficient-shape noise:
    at --scale 0.05 (n,m < r=128) BOTH implementations are legitimately chaotic there
    (normwise update error ~1.28) -- not comparable, see find_rank_deficient()."""
    init = "structured" if grads == "structured" else "iid"
    pairs_ref = build_pairs(shapes, device, torch.float32, seed=init_seed, init=init)
    pairs_batched = build_pairs(shapes, device, torch.float32, seed=init_seed, init=init)

    opt_ref = build_ours(ref_cls, pairs_ref, ball_iters)
    opt_batched = build_ours(batched_cls, pairs_batched, ball_iters)

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(grad_seed)
    gen_batched = torch.Generator(device=device)
    gen_batched.manual_seed(grad_seed)

    pre_ref = pre_bat = None
    for step_i in range(warm_steps):
        assign_grads(pairs_ref, gen_ref, grads)
        assign_grads(pairs_batched, gen_batched, grads)

        if step_i == warm_steps - 1:
            # snapshot immediately before the step we'll actually compare
            pre_ref = [{k: p[k].detach().clone() for k in ("A", "B", "mag")} for p in pairs_ref]
            pre_bat = [{k: p[k].detach().clone() for k in ("A", "B", "mag")} for p in pairs_batched]

        opt_ref.step()
        opt_batched.step()

    errs = []
    for i, (p_r, p_b) in enumerate(zip(pairs_ref, pairs_batched)):
        for key in ("A", "B", "mag"):
            d_ref = (p_r[key].detach() - pre_ref[i][key]).flatten()
            d_bat = (p_b[key].detach() - pre_bat[i][key]).flatten()
            ref_norm = d_ref.norm().item()
            diff_norm = (d_ref - d_bat).norm().item()
            errs.append(diff_norm / ref_norm if ref_norm > 1e-12 else diff_norm)

    errs.sort()
    max_err = errs[-1]
    median_err = errs[len(errs) // 2]
    return max_err, median_err


# --- CLI -------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bench LoRA-TSD (reference/batched/upstream) on our 229 DoRA adapter shapes.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ball-iters", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--impl", choices=list(IMPL_IMPORTERS), nargs="+", default=None)
    parser.add_argument("--pairs-subset", type=int, default=None)
    parser.add_argument("--grads", choices=["iid", "structured"], default="structured")
    parser.add_argument("--allow-rank-deficient", action="store_true",
                         help="Bench shapes where --scale pushed n or m below r=128 anyway. "
                              "Diagnostic only -- both implementations are numerically chaotic "
                              "there by construction, this is NOT a valid A-vs-B comparison.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False.")
        sys.exit(1)
    if args.device == "cuda":
        print("NOTE: this is a shared single-GPU box -- make sure you hold the lock "
              "(Misc/gpu_guard.sh acquire <HANDLE> $$) before this ran.")

    shapes = expand_shapes(scale=args.scale, subset=args.pairs_subset)
    print(f"[bench] {len(shapes)} (A,B) pairs, scale={args.scale}, device={args.device}, "
          f"grads={args.grads}")

    bad = find_rank_deficient(shapes)
    if bad:
        msg = (
            f"{len(bad)}/{len(shapes)} pairs have n<r or m<r at --scale {args.scale} "
            "(r=128 is fixed -- only n/m are scaled -- so A@A.T / B.T@B are singular or "
            "numerically garbage there BY CONSTRUCTION). Both reference and batched are "
            "expected to raise ValueError constructing these pairs for exactly this reason "
            "(worker B is adding that check) -- a batched-vs-reference comparison on them is "
            "not measuring a batching bug, it's measuring shared numerical chaos (normwise "
            "update error ~1.28, i.e. the two updates are unrelated). First offender: "
            f"pair {bad[0][0]} A={bad[0][1]} B={bad[0][2]}."
        )
        if not args.allow_rank_deficient:
            print("ERROR: " + msg + " Pass --allow-rank-deficient to bench them anyway.")
            sys.exit(1)
        else:
            print("WARNING: " + msg + " Continuing because --allow-rank-deficient was passed.")

    impl_names = args.impl if args.impl is not None else list(IMPL_IMPORTERS)
    loaded = {}
    for name in impl_names:
        cls, reason = get_impl(name)
        if cls is None:
            print(f"[skip] {name}: {reason}")
        else:
            loaded[name] = cls

    if not loaded:
        print("No requested impl imported successfully -- nothing to bench.")
        sys.exit(1)

    init = "structured" if args.grads == "structured" else "iid"
    rows = []
    for name, cls in loaded.items():
        for bi in args.ball_iters:
            pairs = build_pairs(shapes, args.device, torch.float32, seed=0, init=init)
            try:
                mean_ms, peak_mb = time_impl(name, cls, pairs, bi, args.steps, args.device,
                                              grads=args.grads)
            except Exception as e:  # noqa: BLE001 -- report and keep going with other combos
                print(f"[fail] {name} ball_iters={bi}: {type(e).__name__}: {e}")
                continue
            rows.append((name, bi, mean_ms, peak_mb))

    print()
    header = f"{'impl':<10} {'ball_iters':>10} {'mean ms/step':>14} {'peak MB':>10}"
    print(header)
    print("-" * len(header))
    for name, bi, mean_ms, peak_mb in rows:
        peak_str = f"{peak_mb:10.1f}" if peak_mb is not None else f"{'n/a':>10}"
        print(f"{name:<10} {bi:>10d} {mean_ms:14.2f} {peak_str}")

    if args.device == "cuda":
        print("\nNOTE: peak MB above is this bench process ALONE on an otherwise-empty card. "
              "In the real training job the optimizer state shares VRAM with a RESIDENT "
              "1.4B-param DiT + activations -- headroom next to THAT is the real constraint, "
              "not this number.")
    else:
        print("\nNOTE: peak MB above is tracemalloc's Python-level estimate -- an "
              "APPROXIMATION that does not fully account for torch's C-level tensor storage. "
              "On GPU the number that matters is headroom next to the resident 1.4B-param "
              "model, not an empty-card peak; see the GPU run instructions above.")

    if "reference" in loaded and "batched" in loaded:
        print()
        for bi in args.ball_iters:
            try:
                max_err, median_err = correctness_check(
                    loaded["reference"], loaded["batched"], shapes, args.device,
                    ball_iters=bi, grads=args.grads,
                )
                print(f"[correctness] ball_iters={bi}: normwise update error "
                      f"reference vs batched -- max={max_err:.3e} median={median_err:.3e}")
            except Exception as e:  # noqa: BLE001
                print(f"[correctness] ball_iters={bi}: FAILED -- {type(e).__name__}: {e}")
    else:
        missing = [n for n in ("reference", "batched") if n not in loaded]
        print(f"\n[correctness] skipped -- missing impl(s): {', '.join(missing)}")


if __name__ == "__main__":
    main()
