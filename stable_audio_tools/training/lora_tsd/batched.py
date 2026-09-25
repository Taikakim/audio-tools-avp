"""BatchedLoRATSD -- batched LoRA tangent-space spectral descent for our DoRA adapters.

Attribution: the algorithm this file reproduces is `LoRATSD` from
brain-lab-research/LoRA-TSD (arXiv 2609.02734), MIT license, commit e6ef0e1
(`/home/kim/Projects/LoRA-TSD/src/optimizers/lora_tsd.py`, read-only upstream).
`reference.py` in this package is the faithful per-pair port used as the correctness
oracle; this file is the *same* maths, restructured for throughput.

WHY THIS EXISTS (measured, see SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md
S1). The reference algorithm calls ~6 small linalg ops (two skinny QRs, two SPD
inverses, and — inside `ball_iters` — two more QRs plus a Newton-Schulz polar
iteration on a 2r x 2r core) *per adapter, per step*. On our 229 DoRA adapters
(rank 128 everywhere) this measured **8.5 s/step at ball_iters=1** on an RX 9070 XT /
ROCm, because `torch.linalg.qr` is latency-bound on ROCm at ~12-17 ms/call
*regardless of matrix size* (a 256x256 `inv` is ~2ms; the big matmuls are
0.07-0.43 ms) -- the job is almost entirely per-call dispatch overhead, and batching
removes nearly all of it.

WHAT IS BATCHED, AND WHY IT IS SAFE:

  1. Pairs are grouped by `(A.shape, B.shape)` (13 groups on our model). Every
     matmul that touches the "tall" dimension (n or m, which differ per group) is
     `torch.bmm`'d within its shape group.

  2. Because rank r = 128 for every one of our adapters, EVERY r x r matrix (the
     `B^T B` / `A A^T` Gram matrices feeding `_spd_inv`) and EVERY 2r x 2r matrix
     (the QR R-factors `R_L`/`R_R`, the Newton-Schulz core) has the *same shape
     across every shape group*, not just within one. So those ops are batched
     across ALL active pairs at once, concatenating across shape groups for the
     one call, then splitting back out. This is the main win beyond "just do bmm
     per group": it turns 229 x 6 small linalg calls into O(ball_iters) calls total.

  3. QR is replaced by batched CholeskyQR2 (`_batched_qr`): `Y -> L = chol(Y^T Y)`,
     `Q = Y @ L^{-T}`, applied twice for orthogonality (~1e-6 residual, matching the
     reference codebase's own CholeskyQR2 helper in `utils.py`). Falls back to
     `torch.linalg.qr` per-matrix for any batch entry where the Cholesky factor
     doesn't exist or is ill-conditioned -- in particular B = 0 on step 1 (our DoRA
     init), where `B^T B` is exactly singular. The fallback calls the *same*
     `torch.linalg.qr` the reference uses, on the *same* (unrefined) input, so a
     degenerate pair is bit-identical to the reference's own degenerate-input QR,
     not merely "close".

  QR SIGN CONVENTION -- proved sign-invariant below, not merely assumed, and then
  checked empirically by the equivalence tests (`tests/test_lora_tsd_batched.py`),
  which pass at ~1e-9 relative in fp64 against the reference despite the two QR
  implementations disagreeing on sign whenever a Cholesky path is taken.
  `torch.linalg.qr` and CholeskyQR2 can disagree on the sign
  of each returned column by an arbitrary diagonal D = diag(+-1). Working through
  the algorithm: (a) `_proj_T_factored`'s two outputs satisfy
  `proj_T_factored(L @ D, D @ R, U_B, V_A) = proj_T_factored(L, R, U_B, V_A) @ D'`
  for some diagonal D' built from D and the U_B/V_A sign choice (a direct algebra
  check on the two concatenations); (b) the Newton-Schulz/msign core is exactly
  equivariant under `X -> D_L X D_R` for diagonal +-1 D_L, D_R (X_0 = X/(norm) is
  equivariant, and each NS iteration `a*X + (b*X X^T + c*(X X^T)^2)@X` commutes with
  conjugation by a diagonal +-1 matrix). Composing these over the `ball_iters` loop
  shows the FINAL `L, R` differ from the reference's by `L=L_ref@D, R=D@R_ref` for a
  single fixed diagonal D (independent of which QR algorithm supplied which sign
  at which step) -- and every downstream quantity (`dA`, `dB`, `X_fro`) only ever
  uses `L` and `R` through the product `L @ R`, which equals `L_ref @ D @ D @ R_ref
  = L_ref @ R_ref` exactly (D^2 = I for a +-1 diagonal). So the batched output is
  provably independent of QR sign convention, PROVIDED each QR call returns an
  actually-orthonormal Q (this is why ill-conditioned Cholesky factors still need
  the `torch.linalg.qr` fallback: a lost-orthogonality Q, not merely a differently
  signed one, is a real numerical error, not a harmless relabeling).

  4. 1-D `*.magnitude` DoRA params (229 of them, all different lengths) are
     concatenated into one flat vector for the multiplicative sign step, since it
     is purely elementwise (see `_magnitude_step`) -- correctness doesn't depend on
     this, it's just cheap and avoids 229 tiny Python-level ops.

State: per-param `momentum_buffer`s live in `self.state[p]`, exactly like a normal
torch optimizer (state_dict round-trips through torch's own machinery). Shape
groups are recomputed from state only implicitly -- the grouping itself is fixed at
construction time from the pairing, and any large 3-D "stacked" tensors used inside
`step()` are transient work buffers, never the source of truth.

Compute is fp32 (fp64 if the params are fp64, for CPU tests), written back in the
param's own dtype -- mirrors the reference's `A.float()` / `.to(A.dtype)` pattern.

NOT verified here (GPU-side): actual wall-clock speedup on ROCm (see `bench.py`,
run there). NOT implemented: the "whiten" balance mode (out of scope per the spec).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["BatchedLoRATSD"]

_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315

# Below this ratio of (min diag L) / (max diag L), treat a Cholesky factor as too
# ill-conditioned to trust for orthogonality and fall back to torch.linalg.qr.
_CHOLESKY_COND_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Batched linear-algebra primitives
# ---------------------------------------------------------------------------


@torch.no_grad()
def _batched_qr(Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Batched CholeskyQR2: return (Q, R, num_fallbacks) with Q (N,m,r) orthonormal
    columns, R (N,r,r) upper-triangular, and Q @ R == Y (N,m,r), m >= r. Falls back
    to torch.linalg.qr per-matrix wherever the Cholesky factorization fails or is
    ill-conditioned (e.g. Y == 0, as at DoRA init: B^T B is then exactly singular).
    `num_fallbacks` is a 0-d device tensor (no host sync here; the caller sums them).

    PRECISION: the Gram matrix Y^T Y squares Y's condition number, so an fp32 Gram
    loses Cholesky at cond(Y) ~ 3e3 -- on real gradients the 2r-wide L/R factors hit
    that 20-80x per step (critic finding, 2026-09-24). The Gram, the Cholesky and the
    triangular solve therefore run in fp64 whatever the input dtype; Q is cast back.

    Two CholeskyQR passes compose as R_total = R2 @ R1 (Y = Q1 R1, Q1 = Q2 R2 =>
    Y = Q2 (R2 R1)). Fallback entries use torch.linalg.qr directly on the
    ORIGINAL Y (not any partially-refined intermediate), so they are bit-identical
    to the reference calling torch.linalg.qr on the same degenerate input.

    WIDE INPUT (m < r): CholeskyQR needs Y tall -- Y^T Y is then rank <= m < r,
    singular by construction. Pairs with r > min(m, n) are refused at construction,
    but the 2r-wide L/R factors can still be wide on small shapes, so the whole call
    goes to torch.linalg.qr (counted as fallbacks).
    """
    N, m, r = Y.shape
    if m < r:
        Q, R = torch.linalg.qr(Y, mode="reduced")
        return Q, R, torch.tensor(N, device=Y.device)
    out_dtype = Y.dtype
    Q = Y.to(torch.float64)
    eye = torch.eye(r, device=Y.device, dtype=torch.float64).expand(N, r, r)
    R_total = eye.clone()
    bad = torch.zeros(N, dtype=torch.bool, device=Y.device)

    for _ in range(2):  # CholeskyQR2: apply CholeskyQR twice for orthogonality
        L, info = torch.linalg.cholesky_ex(Q.transpose(-1, -2) @ Q)  # (N, r, r)
        diag = L.diagonal(dim1=-2, dim2=-1).abs()  # (N, r)
        # a "successful" but near-singular factor is a real orthogonality risk,
        # not merely a sign choice -- catch it too (see module docstring).
        ill_cond = (diag.amin(dim=-1) / diag.amax(dim=-1).clamp_min(1e-300)) < _CHOLESKY_COND_FLOOR
        bad = bad | (info != 0) | ill_cond | ~torch.isfinite(diag).all(dim=-1)
        # Bad entries get an identity factor (a harmless no-op) instead of being
        # masked out, so there is no data-dependent branch and no host sync here;
        # they are overwritten from the ORIGINAL Y below.
        L = torch.where(bad.view(-1, 1, 1), eye, L)
        Xt = torch.linalg.solve_triangular(L, Q.transpose(-1, -2), upper=False, left=True)
        Q = Xt.transpose(-1, -2)  # Q @ L^{-T}
        R_total = L.transpose(-1, -2) @ R_total

    Q = Q.to(out_dtype)
    R_total = R_total.to(out_dtype)
    n_bad = bad.sum()
    if int(n_bad) > 0:  # the one host sync per call
        idx = bad.nonzero(as_tuple=True)[0]
        Q_fb, R_fb = torch.linalg.qr(Y[idx], mode="reduced")
        Q[idx] = Q_fb
        R_total[idx] = R_fb
    return Q, R_total, n_bad


@torch.no_grad()
def _ns5_batched(T: Tensor, steps: int) -> Tensor:
    """Batched Newton-Schulz matrix-sign (polar factor) iteration on square cores
    T: (N, k, k). Mirrors upstream `_ns_sq` exactly, batched over dim 0."""
    norm = T.norm(dim=(-2, -1), keepdim=True)
    X = T / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        X = _NS_A * X + (_NS_B * A + _NS_C * (A @ A)) @ X
    return X


@torch.no_grad()
def _spd_inv_batched(M: Tensor, eps: float) -> Tensor:
    """Batched version of upstream `_spd_inv`: inv(M + ridge*I),
    ridge = eps * mean(diag(M)).clamp_min(1e-12), per batch entry."""
    N, r, _ = M.shape
    diag_mean = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (N,)
    # clamp the MEAN, then scale -- upstream's order. (eps*mean).clamp_min would put a
    # 1e-12 floor under the ridge itself, which is ~1e4x upstream's at B ~ 0.
    ridge = eps * diag_mean.clamp_min(1e-12)  # (N,)
    eye = torch.eye(r, device=M.device, dtype=M.dtype).expand(N, r, r)
    return torch.linalg.inv(M + ridge.view(-1, 1, 1) * eye)


@torch.no_grad()
def _proj_T_factored_batched(L: Tensor, R: Tensor, U_B: Tensor, V_A: Tensor) -> tuple[Tensor, Tensor]:
    """Batched version of upstream `_proj_T_factored`. L: (N,m,2r), R: (N,2r,n),
    U_B: (N,m,r), V_A: (N,n,r)."""
    C = U_B.transpose(-1, -2) @ L  # (N, r, 2r)
    D = R @ V_A  # (N, 2r, r)
    L_new = torch.cat([U_B, L @ D], dim=-1)  # (N, m, 2r)
    R_new = torch.cat(
        [C @ R - (C @ D) @ V_A.transpose(-1, -2), V_A.transpose(-1, -2)], dim=-2
    )  # (N, 2r, n)
    return L_new, R_new


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class BatchedLoRATSD(Optimizer):
    """Batched LoRA tangent-space spectral descent. See module docstring and
    CONTRACT.md for the interface. Numerically equivalent (per pair) to upstream
    `LoRATSD` / this package's `LoRATSDReference`, batched across adapters that
    share an (A.shape, B.shape) key, plus a global batch across ALL pairs for the
    r x r / 2r x 2r work (see docstring point 2).
    """

    def __init__(
        self,
        named_params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        ball_iters: int = 1,
        ns_steps: int = 5,
        max_delta_norm: float = 0.1,
        balance: str = "norm",
        ridge_eps: float = 1e-8,
        lr_magnitude: float | None = None,
        warmup_steps: int = 0,
    ):
        if balance not in ("off", "norm"):
            raise ValueError(f"BatchedLoRATSD: balance must be 'off' or 'norm', got {balance!r}")

        named_params = list(named_params)
        lr_magnitude = lr if lr_magnitude is None else lr_magnitude

        pair_map: dict[str, dict[str, tuple[str, Any]]] = {}
        magnitude_params: list[tuple[str, Any]] = []
        unsupported: list[str] = []

        for name, p in named_params:
            if not p.requires_grad:
                continue
            if name.endswith(".lora_A"):
                prefix = name[: -len(".lora_A")]
                pair_map.setdefault(prefix, {})["A"] = (name, p)
            elif name.endswith(".lora_B"):
                prefix = name[: -len(".lora_B")]
                pair_map.setdefault(prefix, {})["B"] = (name, p)
            elif name.endswith("magnitude"):
                magnitude_params.append((name, p))
            else:
                unsupported.append(name)

        if unsupported:
            raise ValueError(
                "BatchedLoRATSD: param(s) not matching .lora_A / .lora_B / "
                f"*magnitude and not handled: {unsupported}"
            )

        pairs: list[dict[str, Any]] = []
        incomplete: list[str] = []
        for prefix, d in pair_map.items():
            if "A" not in d or "B" not in d:
                incomplete.append(prefix)
                continue
            name_A, A = d["A"]
            name_B, B = d["B"]
            r, n_ = A.shape
            if r > n_ or r > B.shape[0]:
                # r > min(m, n) makes the reduced QRs return < r columns and every
                # downstream shape assumption silently wrong (see reference.py).
                raise ValueError(
                    f"BatchedLoRATSD: pair {prefix!r} has rank r={r} but A has only n={n_} "
                    f"columns and B has only m={B.shape[0]} rows -- LoRA-TSD requires r <= min(n, m)."
                )
            pairs.append({"prefix": prefix, "A": A, "B": B, "name_A": name_A, "name_B": name_B})
        if incomplete:
            raise ValueError(f"BatchedLoRATSD: lora_A/lora_B without a matching pair for: {incomplete}")

        ranks = sorted({pr["A"].shape[0] for pr in pairs})
        if len(ranks) > 1:
            # The r x r / 2r x 2r work is concatenated across ALL pairs (module docstring
            # point 2), which needs one rank. Mixed ranks would otherwise die inside step()
            # with an opaque torch.cat shape error. (Use LoRATSDReference for mixed ranks.)
            raise ValueError(f"BatchedLoRATSD: all LoRA pairs must share one rank, got ranks {ranks}")

        self.pairs = pairs
        self.magnitude_params = magnitude_params

        shape_groups: dict[tuple, list[dict[str, Any]]] = {}
        for pr in pairs:
            key = (tuple(pr["A"].shape), tuple(pr["B"].shape))
            shape_groups.setdefault(key, []).append(pr)
        self.shape_groups = shape_groups

        self.lr = lr
        self.momentum = momentum
        self.ball_iters = ball_iters
        self.ns_steps = ns_steps
        self.max_delta_norm = max_delta_norm
        self.balance = balance
        self.ridge_eps = ridge_eps
        self.lr_magnitude = lr_magnitude
        self.warmup_steps = warmup_steps
        self.balance_eps = 1e-12  # matches upstream default

        self._step = 0
        self.last_stats: dict[str, float] = {}

        all_params = []
        for pr in pairs:
            all_params.append(pr["A"])
            all_params.append(pr["B"])
        for _, p in magnitude_params:
            all_params.append(p)

        if not all_params:
            raise ValueError("BatchedLoRATSD: no trainable lora_A/lora_B pairs or *magnitude params found")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ball_iters=ball_iters,
            ns_steps=ns_steps,
            max_delta_norm=max_delta_norm,
            balance=balance,
            ridge_eps=ridge_eps,
            lr_magnitude=lr_magnitude,
            warmup_steps=warmup_steps,
        )
        super().__init__(all_params, defaults)

    # -- hyperparameters ---------------------------------------------------

    def _sync_hparams(self) -> None:
        """Take this step's hyperparameters from param_groups[0], the PyTorch convention.
        A torch LR scheduler writes group["lr"] and load_state_dict restores the groups;
        reading the constructor copies instead made both silently ineffective (cloud
        review, 2026-09-25). The attributes stay as mirrors, because the trainer's
        train/lr helper reads opt.lr. A scheduler moves only `lr`: `lr_magnitude` is its
        own group key, the same as LoRATSDReference's magnitude groups."""
        g = self.param_groups[0]
        if g["balance"] not in ("off", "norm"):
            raise ValueError(f"BatchedLoRATSD: balance must be 'off' or 'norm', got {g['balance']!r}")
        self.lr = g["lr"]
        self.momentum = g["momentum"]
        self.ball_iters = g["ball_iters"]
        self.ns_steps = g["ns_steps"]
        self.max_delta_norm = g["max_delta_norm"]
        self.balance = g["balance"]
        self.ridge_eps = g["ridge_eps"]
        self.lr_magnitude = g["lr_magnitude"]
        self.warmup_steps = g["warmup_steps"]

    # -- lr schedule -----------------------------------------------------

    def _warmup_scale(self) -> float:
        if self.warmup_steps and self.warmup_steps > 0:
            return min(1.0, self._step / self.warmup_steps)
        return 1.0

    # -- magnitude (unpaired 1-D DoRA params) -----------------------------

    @torch.no_grad()
    def _magnitude_step(self, lr_mag_t: float) -> None:
        """Multiplicative sign step, same rule as
        `modular_opt/optimizer.py::_magnitude_step` with magnitude_update=multiplicative:
        buf <- momentum*buf + (1-momentum)*grad ; m <- m * exp(-lr_mag_t * sign(buf))
        (buf collapses to grad when momentum == 0, matching the pair momentum gate)."""
        active = [(name, p) for name, p in self.magnitude_params if p.grad is not None]
        if not active:
            return

        compute_dtype = torch.float64 if any(p.dtype == torch.float64 for _, p in active) else torch.float32
        grads = [p.grad.detach().to(compute_dtype) for _, p in active]
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        if self.momentum > 0:
            bufs = []
            for (_, p), g in zip(active, grads):
                st = self.state[p]
                buf = st.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                elif buf.dtype != compute_dtype:
                    # mixed fp32/fp64 magnitudes put every buffer in fp64 compute; a
                    # buffer from another dtype would otherwise break the flat cat/sign
                    buf = buf.to(compute_dtype)
                buf = buf.mul(self.momentum).add_(g, alpha=(1.0 - self.momentum))
                st["momentum_buffer"] = buf
                bufs.append(buf)
            flat_dir = torch.sign(torch.cat([b.reshape(-1) for b in bufs]))
        else:
            flat_dir = torch.sign(flat_grad)

        scale = torch.exp(-lr_mag_t * flat_dir)
        offset = 0
        for (_, p), g in zip(active, grads):
            n = g.numel()
            s = scale[offset : offset + n].reshape(p.shape)
            offset += n
            p.mul_(s.to(p.dtype))

    # -- main step ---------------------------------------------------------

    @torch.no_grad()
    def _momentum_stack(self, params: list, compute_dtype) -> Tensor:
        """buf <- momentum*buf + (1-momentum)*grad per param, in place, then one stack.
        Returns the raw grads stacked when momentum == 0 (upstream use_momentum=False)."""
        if self.momentum <= 0:
            return torch.stack([p.grad.detach().to(compute_dtype) for p in params])
        bufs = []
        for p in params:
            st = self.state[p]
            buf = st.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros(p.shape, dtype=compute_dtype, device=p.device)
            elif buf.dtype != compute_dtype:
                buf = buf.to(compute_dtype)  # keep the momentum, don't silently reset it
            buf.mul_(self.momentum).add_(p.grad.detach().to(compute_dtype), alpha=(1.0 - self.momentum))
            st["momentum_buffer"] = buf
            bufs.append(buf)
        return torch.stack(bufs)

    @torch.no_grad()
    def step(self, closure=None):
        """MEMORY: every big per-group buffer is freed the moment it is dead -- grads and
        momentum stacks after the L_g/R_g factorisation, Q factors after each NS pass,
        L/R/A/B after the group's write-back. The first draft held all of them for all
        13 groups at once (~5.7 GB of scratch on the real 229-adapter model; critic
        finding 2026-09-24). Only the r x r / 2r x 2r pieces are kept across groups,
        because those are what get batched across ALL pairs.
        SYNCS: none per pair; one per QR call (fallback check) and one at the end."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sync_hparams()
        self._step += 1
        scale = self._warmup_scale()
        lr_t = self.lr * scale
        lr_mag_t = self.lr_magnitude * scale

        fallbacks: list[Tensor] = []
        dW_norms: list[Tensor] = []
        clip_masks: list[Tensor] = []

        group_ctx: list[dict[str, Any]] = []
        for pair_list in self.shape_groups.values():
            active = [pr for pr in pair_list if pr["A"].grad is not None and pr["B"].grad is not None]
            if not active:
                continue
            any_fp64 = any(pr["A"].dtype == torch.float64 or pr["B"].dtype == torch.float64 for pr in active)
            cd = torch.float64 if any_fp64 else torch.float32

            A_ = torch.stack([pr["A"].detach().to(cd) for pr in active])  # (Ng,r,n)
            B_ = torch.stack([pr["B"].detach().to(cd) for pr in active])  # (Ng,m,r)
            GA = self._momentum_stack([pr["A"] for pr in active], cd)
            GB = self._momentum_stack([pr["B"] for pr in active], cd)

            U_B, _, fb1 = _batched_qr(B_)
            V_A, _, fb2 = _batched_qr(A_.transpose(-1, -2))
            fallbacks += [fb1, fb2]
            group_ctx.append(dict(active=active, cd=cd, A=A_, B=B_, GA=GA, GB=GB, U_B=U_B, V_A=V_A,
                                  BtB=B_.transpose(-1, -2) @ B_, AAt=A_ @ A_.transpose(-1, -2)))

        if group_ctx:
            # ---- r x r inverses, batched across ALL pairs at once ----
            BtB_i_all = _spd_inv_batched(torch.cat([g.pop("BtB") for g in group_ctx]), self.ridge_eps)
            AAt_i_all = _spd_inv_batched(torch.cat([g.pop("AAt") for g in group_ctx]), self.ridge_eps)
            offset = 0
            for g in group_ctx:
                n = g["A"].shape[0]
                g["BtB_i"] = BtB_i_all[offset : offset + n]
                g["AAt_i"] = AAt_i_all[offset : offset + n]
                offset += n
            del BtB_i_all, AAt_i_all  # the slices keep the storage; drop the names

            # ---- factor the tangent-projected gradient as L_g @ R_g; grads are dead after ----
            for g in group_ctx:
                A_, B_, BtB_i, AAt_i = g["A"], g["B"], g["BtB_i"], g["AAt_i"]
                GA, GB = g.pop("GA"), g.pop("GB")
                Mc = BtB_i @ (GA @ A_.transpose(-1, -2)) @ AAt_i
                g["L"] = torch.cat([B_, GB @ AAt_i], dim=-1)
                g["R"] = torch.cat([BtB_i @ GA - Mc @ A_, A_], dim=-2)
                del GA, GB, Mc

            # ---- ball_iters: msign on the 2r x 2r core, batched across ALL pairs ----
            for _ in range(self.ball_iters):
                for g in group_ctx:
                    Q_L, R_L, fb_l = _batched_qr(g.pop("L"))
                    Q_R, R_R, fb_r = _batched_qr(g.pop("R").transpose(-1, -2))
                    fallbacks += [fb_l, fb_r]
                    g["_Q_L"], g["_Q_R"] = Q_L, Q_R
                    g["_core"] = R_L @ R_R.transpose(-1, -2)  # (Ng, k_L, k_R)

                # Batch NS5 across every group whose core has the SAME (k_L, k_R) shape --
                # all 13 groups on our real model (every core is 2r x 2r), but a wide L or
                # R.T on small shapes can give a smaller core, so bucket, don't assume.
                buckets: dict[tuple[int, int], list[int]] = {}
                for gi, g in enumerate(group_ctx):
                    buckets.setdefault(tuple(g["_core"].shape[-2:]), []).append(gi)
                for gis in buckets.values():
                    T_cat = _ns5_batched(torch.cat([group_ctx[gi].pop("_core") for gi in gis]), self.ns_steps)
                    offset = 0
                    for gi in gis:
                        n = group_ctx[gi]["A"].shape[0]
                        group_ctx[gi]["_T"] = T_cat[offset : offset + n]
                        offset += n
                    del T_cat

                for g in group_ctx:
                    Lp = g.pop("_Q_L") @ g.pop("_T")
                    g["L"], g["R"] = _proj_T_factored_batched(Lp, g.pop("_Q_R").transpose(-1, -2), g["U_B"], g["V_A"])
                    del Lp

            # ---- final factor updates + clip + apply + rebalance, freeing each group ----
            while group_ctx:
                g = group_ctx.pop(0)
                L, R, A_, B_, BtB_i, AAt_i = g["L"], g["R"], g["A"], g["B"], g["BtB_i"], g["AAt_i"]
                active, cd = g["active"], g["cd"]
                del g

                X_fro = ((L.transpose(-1, -2) @ L) * (R @ R.transpose(-1, -2))).sum(dim=(-2, -1)).clamp_min(0).sqrt()
                dB = -lr_t * (L @ (R @ A_.transpose(-1, -2))) @ AAt_i
                dA = BtB_i @ (-lr_t * ((B_.transpose(-1, -2) @ L) @ R) - (B_.transpose(-1, -2) @ dB) @ A_)
                del L, R

                dW_norm = lr_t * X_fro  # (Ng,)
                clip_mask = dW_norm > self.max_delta_norm
                clip_scale = torch.where(clip_mask, self.max_delta_norm / dW_norm.clamp_min(1e-30),
                                         torch.ones_like(dW_norm)).view(-1, 1, 1)
                dA = dA * clip_scale
                dB = dB * clip_scale
                dW_norms.append(dW_norm)
                clip_masks.append(clip_mask)

                for i, pr in enumerate(active):
                    pr["A"].add_(dA[i].to(pr["A"].dtype))
                    pr["B"].add_(dB[i].to(pr["B"].dtype))
                del dA, dB, A_, B_

                if self.balance == "norm":
                    nA = torch.stack([pr["A"].detach().to(cd).norm() for pr in active])
                    nB = torch.stack([pr["B"].detach().to(cd).norm() for pr in active])
                    valid = (nA >= self.balance_eps) & (nB >= self.balance_eps)
                    # invalid pairs get c = 1, and x*1 / x/1 are exact -- so no per-pair
                    # branch (that was 229 host syncs per step).
                    c = torch.where(valid, (nB / nA.clamp_min(1e-30)).sqrt(), torch.ones_like(nA))
                    for i, pr in enumerate(active):
                        ci = c[i]
                        pr["A"].mul_(ci.to(pr["A"].dtype))
                        pr["B"].div_(ci.to(pr["B"].dtype))
                        if self.momentum > 0:
                            stA, stB = self.state[pr["A"]], self.state[pr["B"]]
                            if "momentum_buffer" in stA:
                                stA["momentum_buffer"].div_(ci)
                            if "momentum_buffer" in stB:
                                stB["momentum_buffer"].mul_(ci)

        self._magnitude_step(lr_mag_t)

        if dW_norms:  # the one end-of-step sync
            all_norms = torch.cat(dW_norms)
            n_clip = torch.cat(clip_masks).sum()
            stats = torch.stack([all_norms.mean().to(torch.float64), n_clip.to(torch.float64),
                                 torch.stack(fallbacks).sum().to(torch.float64)]).tolist()
            self.last_stats = {"dW_norm_mean": stats[0], "clip_frac": stats[1] / all_norms.numel(),
                               "qr_fallbacks": stats[2]}
        else:
            self.last_stats = {"dW_norm_mean": 0.0, "clip_frac": 0.0, "qr_fallbacks": 0.0}
        return loss

    # -- state dict: add the step counter torch's own state_dict doesn't track --

    def state_dict(self) -> dict:
        # torch's default Optimizer.state_dict() aliases (does not copy) state
        # tensors whose dtype/device already match the param -- see
        # torch/optim/optimizer.py's cast(), which is a no-op Tensor.to() in that
        # case. Left alone, a snapshot taken here keeps sharing storage/dict
        # objects with this optimizer's own momentum buffers, so a further
        # step() on `self` silently mutates an already-"saved" state_dict in
        # place (verified: reassigning self.state[p]["momentum_buffer"] to a
        # new tensor is STILL visible through an earlier state_dict() capture,
        # because state_dict()'s per-param dict is the same object as
        # self.state[p], not a copy of it). Deepcopy so a snapshot is a real,
        # independent copy -- matches reference.py's same defense.
        # Copy to CPU rather than deepcopy on device: a same-device deepcopy doubles the
        # optimizer state's VRAM for the duration of a checkpoint save. torch's
        # Optimizer.load_state_dict moves state back to each param's device and dtype.
        sd = super().state_dict()
        sd = {
            "state": {k: {kk: (vv.detach().to("cpu", copy=True) if torch.is_tensor(vv) else copy.deepcopy(vv))
                          for kk, vv in v.items()} for k, v in sd["state"].items()},
            "param_groups": copy.deepcopy(sd["param_groups"]),
        }
        sd["lora_tsd_step"] = self._step
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        step = state_dict.pop("lora_tsd_step", 0)
        # torch only deep-copies param_groups; a same-device/dtype state tensor would be
        # adopted by reference and our in-place momentum updates would then write into the
        # caller's dict (final critic, reproduced on CPU). Clone first.
        state_dict["state"] = {k: {kk: (vv.clone() if torch.is_tensor(vv) else copy.deepcopy(vv))
                                   for kk, vv in v.items()} for k, v in state_dict["state"].items()}
        super().load_state_dict(state_dict)
        self._step = step
