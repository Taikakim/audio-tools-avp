"""
MIT License

Copyright (c) brain-lab-research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

LoRATSDReference: a faithful, one-pair-at-a-time port of the upstream LoRA-TSD
optimizer (arXiv 2609.02734), ported from
https://github.com/brain-lab-research/LoRA-TSD commit e6ef0e1
(`src/optimizers/lora_tsd.py` + `src/optimizers/utils.py`).

This is the CORRECTNESS ORACLE for `..batched.BatchedLoRATSD`, not a training
optimizer to run with — it loops over adapters one at a time (upstream's
per-pair `torch.linalg.qr`, no batching) and is expected to be numerically
slow. See `../CONTRACT.md` for the shared constructor signature and
`SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md` §2 for the algorithm
this ports.

Deliberately dropped vs. upstream `LoRATSD`: the diagnostic/logging branches
(`f`, `log_ns_gap`, `log_f_inner`, `log_grads`, `log_optimizer_stats`), the
per-group `lr_A_scheduler`/`lr_B_scheduler` hooks, `norm_ord` (only used by the
dropped diagnostic `f()`), and the "whiten" balance mode (CONTRACT only
exposes `balance: "off"|"norm"`). Everything that affects the actual weight
update is preserved verbatim.
"""
import torch
from torch.optim import Optimizer


@torch.no_grad()
def _ns_sq(T: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz quintic orthogonalization of a square matrix (upstream `_ns_sq`)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = T / (T.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X


@torch.no_grad()
def _proj_T_factored(L: torch.Tensor, R: torch.Tensor,
                      U_B: torch.Tensor, V_A: torch.Tensor):
    """Tangent projector applied to L @ R in factored form (upstream `_proj_T_factored`)."""
    C = U_B.t() @ L
    D = R @ V_A
    L_new = torch.cat([U_B, L @ D], dim=1)
    R_new = torch.cat([C @ R - (C @ D) @ V_A.t(), V_A.t()], dim=0)
    return L_new, R_new


@torch.no_grad()
def _spd_inv(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Invert a small SPD matrix after adding a relative ridge (upstream `_spd_inv`)."""
    r = M.shape[0]
    ridge = eps * M.diagonal().mean().clamp_min(1e-12)
    return torch.linalg.inv(M + ridge * torch.eye(r, device=M.device, dtype=M.dtype))


class LoRATSDReference(Optimizer):
    """Per-pair, diagnostics-stripped port of upstream `LoRATSD.step`.

    See `../CONTRACT.md` for the full contract (shared with `BatchedLoRATSD`).
    Compute happens in fp32, or fp64 if the incoming param/grad tensors are
    fp64 (so CPU-fp64 equivalence tests can run at tight tolerance); results
    are always written back in the param's own dtype.
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
            raise ValueError(f"balance must be 'off' or 'norm', got {balance!r}")

        named_params = list(named_params)
        lora_A = {}
        lora_B = {}
        magnitude_entries = []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if name.endswith(".lora_A"):
                prefix = name[: -len(".lora_A")]
                lora_A[prefix] = (name, p)
            elif name.endswith(".lora_B"):
                prefix = name[: -len(".lora_B")]
                lora_B[prefix] = (name, p)
            elif name.endswith("magnitude"):
                magnitude_entries.append((name, p))
            else:
                raise ValueError(
                    f"LoRATSDReference: parameter {name!r} is neither a lora_A/lora_B "
                    f"pair member nor a *magnitude* param — refusing to train it "
                    f"silently. Exclude it from named_params or rename it."
                )

        prefixes = sorted(set(lora_A) | set(lora_B))
        param_groups = []
        self._pair_names = []
        for prefix in prefixes:
            if prefix not in lora_A:
                raise ValueError(
                    f"LoRATSDReference: {lora_B[prefix][0]!r} has no matching "
                    f"lora_A (prefix {prefix!r}) — cannot pair for a spectral step."
                )
            if prefix not in lora_B:
                raise ValueError(
                    f"LoRATSDReference: {lora_A[prefix][0]!r} has no matching "
                    f"lora_B (prefix {prefix!r}) — cannot pair for a spectral step."
                )
            _, A = lora_A[prefix]
            _, B = lora_B[prefix]
            r, n_ = A.shape
            m_ = B.shape[0]
            if r > n_ or r > m_:
                # Found via testing, not a hypothetical: torch.linalg.qr(mode="reduced")
                # on the wide A^T (n_ x r, n_ < r) or B (m_ x r, m_ < r) silently returns
                # a Q with FEWER than r columns instead of erroring, which desyncs every
                # downstream shape assumption (U_B/V_A are no longer m_ x r / n_ x r) and
                # runs to completion with silently wrong numbers rather than raising.
                # LoRA-TSD requires r <= min(n, m) by construction (A: r x n, B: m x r
                # must both be full column/row rank r); refuse it explicitly.
                raise ValueError(
                    f"LoRATSDReference: pair {prefix!r} has rank r={r} but A has only "
                    f"n={n_} columns and B has only m={m_} rows — LoRA-TSD requires "
                    f"r <= min(n, m)."
                )
            param_groups.append({"params": [A, B], "kind": "pair", "name": prefix})
            self._pair_names.append(prefix)

        self._magnitude_names = []
        for name, p in magnitude_entries:
            param_groups.append({"params": [p], "kind": "magnitude", "name": name})
            self._magnitude_names.append(name)

        if not param_groups:
            raise ValueError(
                "LoRATSDReference: no trainable lora_A/lora_B pairs or magnitude "
                "params found in named_params."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ball_iters=ball_iters,
            ns_steps=ns_steps,
            max_delta_norm=max_delta_norm,
            balance=balance,
            ridge_eps=ridge_eps,
            lr_magnitude=lr_magnitude if lr_magnitude is not None else lr,
            warmup_steps=warmup_steps,
        )
        super().__init__(param_groups, defaults)

        self._step_count = 0
        self.last_stats = {"dW_norm_mean": 0.0, "clip_frac": 0.0, "qr_fallbacks": 0}

    @staticmethod
    def _compute_dtype(t: torch.Tensor) -> torch.dtype:
        return torch.float64 if t.dtype == torch.float64 else torch.float32

    def _warmup_frac(self, warmup_steps: int) -> float:
        if not warmup_steps or warmup_steps <= 0:
            return 1.0
        return min(1.0, (self._step_count + 1) / float(warmup_steps))

    @torch.no_grad()
    def _rebalance_pair(self, A: torch.Tensor, B: torch.Tensor, momentum: float,
                         cdtype: torch.dtype) -> None:
        """Rescale A,B so ||A||=||B|| without changing A@B (upstream `_rebalance_group`, "norm" mode)."""
        A32 = A.data.to(cdtype)
        B32 = B.data.to(cdtype)
        nA = A32.norm()
        nB = B32.norm()
        eps = 1e-12
        if nA < eps or nB < eps:
            return
        c = (nB / nA).sqrt()
        A.data.mul_(c.to(A.dtype))
        B.data.div_(c.to(B.dtype))
        if momentum > 0:
            bufA = self.state[A].get("momentum_buffer")
            bufB = self.state[B].get("momentum_buffer")
            if bufA is not None:
                bufA.div_(c)
            if bufB is not None:
                bufB.mul_(c)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dW_norms = []
        n_clipped = 0

        for group in self.param_groups:
            kind = group["kind"]
            momentum = group["momentum"]
            warmup_steps = group["warmup_steps"]
            frac = self._warmup_frac(warmup_steps)

            if kind == "magnitude":
                (p,) = group["params"]
                if p.grad is None:
                    continue
                cdtype = self._compute_dtype(p)
                grad = p.grad.to(cdtype)
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad, alpha=(1.0 - momentum))
                state["momentum_buffer"] = buf

                lr_mag = group["lr_magnitude"] * frac
                update = torch.exp((-lr_mag * torch.sign(buf)).to(p.dtype))
                p.data.mul_(update)
                continue

            # kind == "pair"
            A, B = group["params"]
            if A.grad is None or B.grad is None:
                continue

            cdtype = self._compute_dtype(A)
            lr = group["lr"] * frac
            ridge_eps = group["ridge_eps"]
            ball_iters = group["ball_iters"]
            ns_steps = group["ns_steps"]
            max_delta_norm = group["max_delta_norm"]
            balance = group["balance"]

            A32 = A.data.to(cdtype)
            B32 = B.data.to(cdtype)
            G_A32 = A.grad.to(cdtype)
            G_B32 = B.grad.to(cdtype)

            if momentum > 0:
                stA = self.state[A]
                bufA = stA.get("momentum_buffer")
                if bufA is None:
                    bufA = torch.zeros_like(G_A32)
                bufA.mul_(momentum).add_(G_A32, alpha=(1.0 - momentum))
                stA["momentum_buffer"] = bufA
                G_A32 = bufA

                stB = self.state[B]
                bufB = stB.get("momentum_buffer")
                if bufB is None:
                    bufB = torch.zeros_like(G_B32)
                bufB.mul_(momentum).add_(G_B32, alpha=(1.0 - momentum))
                stB["momentum_buffer"] = bufB
                G_B32 = bufB

            U_B = torch.linalg.qr(B32, mode="reduced")[0]
            V_A = torch.linalg.qr(A32.t(), mode="reduced")[0]
            BtB_i = _spd_inv(B32.t() @ B32, ridge_eps)
            AAt_i = _spd_inv(A32 @ A32.t(), ridge_eps)

            # Factor P_T(G_W) as L_g @ R_g, with M = (BtB)^-1 (G_A A^T) (AAt)^-1.
            M = BtB_i @ (G_A32 @ A32.t()) @ AAt_i
            L_g = torch.cat([B32, G_B32 @ AAt_i], dim=1)
            R_g = torch.cat([BtB_i @ G_A32 - M @ A32, A32], dim=0)

            L, R = L_g, R_g
            for _ in range(ball_iters):
                Q_L, R_L = torch.linalg.qr(L, mode="reduced")
                Q_R, R_R = torch.linalg.qr(R.t(), mode="reduced")
                T_orth = _ns_sq(R_L @ R_R.t(), steps=ns_steps)
                L, R = _proj_T_factored(Q_L @ T_orth, Q_R.t(), U_B, V_A)

            GLL = L.t() @ L
            GRR = R @ R.t()
            X_fro = (GLL * GRR).sum().clamp_min(0).sqrt()

            # Reconstruct factor updates whose linearized product is -lr * L @ R.
            RA_t = R @ A32.t()
            dB32 = -lr * (L @ RA_t) @ AAt_i
            BtL = B32.t() @ L
            BtdB = B32.t() @ dB32
            dA32 = BtB_i @ (-lr * (BtL @ R) - BtdB @ A32)

            dW_norm = lr * X_fro
            dW_norms.append(float(dW_norm))
            if dW_norm > max_delta_norm:
                scale = max_delta_norm / dW_norm
                dA32.mul_(scale)
                dB32.mul_(scale)
                n_clipped += 1

            A.data.add_(dA32.to(A.dtype))
            B.data.add_(dB32.to(B.dtype))

            if balance == "norm":
                self._rebalance_pair(A, B, momentum, cdtype)

        self._step_count += 1
        n = len(dW_norms)
        self.last_stats = {
            "dW_norm_mean": (sum(dW_norms) / n) if n else 0.0,
            "clip_frac": (n_clipped / n) if n else 0.0,
            "qr_fallbacks": 0,  # reference always uses real QR, never a Cholesky fallback
        }
        return loss

    def state_dict(self):
        # torch's default Optimizer.load_state_dict() aliases (does not copy) any
        # state tensor whose dtype/device already match the target param -- see
        # torch/optim/optimizer.py's `cast()`, which relies on `Tensor.to()` being
        # a no-op when the dtype/device are unchanged. Left alone, that means a
        # snapshot taken here would keep sharing storage with this optimizer's own
        # momentum buffers, and further `step()` calls on `self` would silently
        # mutate an already-"saved" state_dict in place -- exactly the case the
        # round-trip test exercises (keep training the live optimizer after saving
        # a checkpoint of it). Deepcopy so a snapshot is a real, independent copy.
        import copy
        d = copy.deepcopy(super().state_dict())
        d["step_count"] = self._step_count
        return d

    def load_state_dict(self, state_dict):
        sd = dict(state_dict)
        step_count = sd.pop("step_count", 0)
        super().load_state_dict(sd)
        self._step_count = step_count
