"""Preconditioners for Forward Whitening and Reverse Pullback.

Implements:
- KLShampooPreconditioner: Two-sided Kronecker covariance preconditioning with exact P_L @ M @ P_R pullback.
- RotatedSOAPPreconditioner: Eigendecomposition rotation and eigenspace preconditioning.
- IdentityPreconditioner: Zero-overhead pass-through for unconditioned layers.
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor

from .protocols import Preconditioner


def inv_quarter_spd(M: Tensor, delta: float = 1e-4) -> Tensor:
    """Compute (M + delta * I)^(-1/4) via eigendecomposition in FP32."""
    M32 = M.float()
    n = M32.shape[-1]
    eye = torch.eye(n, device=M32.device, dtype=M32.dtype)
    sym = 0.5 * (M32 + M32.transpose(-1, -2)) + delta * eye
    eigvals, eigvecs = torch.linalg.eigh(sym)
    inv_q = eigvals.clamp_min(1e-12).pow(-0.25)
    res = eigvecs @ torch.diag_embed(inv_q) @ eigvecs.transpose(-1, -2)
    return res.to(M.dtype)


class IdentityPreconditioner(Preconditioner):
    """No-op preconditioner for layers that do not use forward whitening."""

    name = "identity"

    def update_accumulators(self, gradient: Tensor) -> None:
        pass

    def transform(self, gradient: Tensor) -> Tensor:
        return gradient

    def inverse_transform(self, update: Tensor) -> Tensor:
        return update


class ShampooPreconditioner(Preconditioner):
    """Kronecker-factored Shampoo whitening, as used by Mousse (arXiv:2603.09697).

    NAMING, DELIBERATE (CONTINUITY 2026-09-22): this used to be called
    KLShampooPreconditioner. It is NOT KL-Shampoo. KL-Shampoo (arXiv:2509.03378 Eq. 5)
    estimates the two factors with a COUPLED update, each using the other's inverse:

        S_a <- (1-b) S_a + (b/d_b) G S_b^-1 G^T
        S_b <- (1-b) S_b + (b/d_a) G^T S_a^-1 G

    That coupling is what lets KL-Shampoo drop Adam step-size grafting. What is
    implemented here is the plain EMA of G G^T / G^T G -- the baseline that paper
    improves on. Note also that KL-Shampoo is inherently TWO-sided, so it cannot be
    used under the one-sided bottleneck below; adopting it means giving that up.

    SIDE CHOICE, and an honest caveat. Mousse section 5.3 ablates single- vs double-sided
    and finds single-sided "achieves comparable performance to the Mousse baseline, yielding
    a negligible decline or even slight improvements" -- so one-sided is validated, not a
    cost dodge. BUT it also finds the LEFT factor consistently slightly better than the
    right, attributing this to the preceding LayerNorm standardising the activations whose
    statistics L captures. We choose by SIZE, not by that preference, so on a (12288, 128)
    factor we take the side the paper found slightly worse -- because the better one is the
    unaffordable one.

    KL-SHAMPOO, RESOLVED. arXiv:2509.03378 Claim 1: the KL-optimal one-sided preconditioner
    is exactly S_a* = E[G G^T] -- the plain covariance computed below. So in ONE-SIDED mode
    this estimator IS the KL-optimal one, and the distinction from KL-Shampoo disappears.
    It only reappears if someone turns the bottleneck off.

    BOTTLENECK, and why it is on by default. Mousse assumes full weight matrices. Our
    shapes are LoRA/DoRA factors: one side is the rank (128), the other is up to 12288.
    Forming the covariance on the WIDE side is a 12288x12288 matrix, and one eigh of
    that size measured 7.23 s on this card -- 173 s per step across the 24 tensors of
    that shape, before touching the other 434. Preconditioning only the SMALLER side
    takes the whole parameter set from 27.9 GiB to 28.6 MiB and the eigh to 128x128.

    Mousse Algorithm 1, implemented here:
      line 4  Trace Normalization   Lbar <- dim(L)/(Tr(L)+eps) * L
      line 6  Spectral Tempering    S <- Lambda^(-alpha), alpha tunable; DEFAULT 0.125,
              not the classic Shampoo 0.25 -- the paper's own ablation (Figure 7a) finds
              0.125 'consistently outperforms the aggressive curvature correction
              (alpha = 0.25), yielding the lowest final loss'.
      line 7  Whitening             P @ G   (or G @ P)
      line 9  Unwhitening           P @ M   (or M @ P)  -- the SAME negative powers,
              not their inverses. This reads like a bug and is not one: confirmed at
              Algorithm 1 line 9 and the closed form on p5,
              dW = -L^(-1/4) msign(L^(-1/4) G R^(-1/4)) R^(-1/4).

    The gamma renormalization (lines 8/10) is NOT here -- it spans the Newton-Schulz
    call, so it lives in the optimizer's Stage 4.
    """

    name = "shampoo"

    def __init__(
        self,
        shape: Tuple[int, int],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        delta: float = 1e-4,
        update_freq: int = 10,
        alpha: float = 0.125,
        bottleneck: bool = True,
        max_dim: int = 1024,
    ) -> None:
        self.d_out, self.d_in = shape
        self.beta = beta
        self.delta = delta
        self.update_freq = max(1, int(update_freq))
        self.alpha = alpha
        self.step_count = 0

        # Precondition the SMALLER side; "left" means the d_out x d_out factor.
        self.side = "left" if self.d_out <= self.d_in else "right"
        if not bottleneck:
            self.side = "both"

        self.dim = self.d_out if self.side == "left" else self.d_in
        # Refuse a side we measured to be unaffordable rather than OOM mid-run.
        self.enabled = (self.side != "both") and (self.dim <= max_dim)

        if self.enabled:
            self.C = torch.zeros((self.dim, self.dim), device=device, dtype=torch.float32)
            self.P = torch.eye(self.dim, device=device, dtype=dtype)
        else:
            self.C = None
            self.P = None

    def _factor(self, C: Tensor) -> Tensor:
        """Trace-normalised, spectrally-tempered inverse power (Algorithm 1 lines 4-6)."""
        C32 = C.float()
        n = C32.shape[-1]
        eye = torch.eye(n, device=C32.device, dtype=C32.dtype)
        # line 4: Trace Normalization -- NOT bias correction. Named by the paper as one
        # of its two critical stability techniques.
        trace = torch.diagonal(C32, dim1=-2, dim2=-1).sum(-1)
        C32 = C32 * (n / (trace + self.delta))
        sym = 0.5 * (C32 + C32.transpose(-1, -2)) + self.delta * eye
        eigvals, eigvecs = torch.linalg.eigh(sym)
        # line 6: Spectral Tempering -- alpha is a tunable curvature exponent.
        scaled = eigvals.clamp_min(1e-12).pow(-self.alpha)
        return (eigvecs @ torch.diag_embed(scaled) @ eigvecs.transpose(-1, -2))

    def update_accumulators(self, gradient: Tensor) -> None:
        if not self.enabled:
            return
        G = gradient.float()
        self.step_count += 1
        if self.side == "left":
            self.C.mul_(self.beta).addmm_(G, G.transpose(-1, -2), alpha=(1.0 - self.beta))
        else:
            self.C.mul_(self.beta).addmm_(G.transpose(-1, -2), G, alpha=(1.0 - self.beta))
        # line 3: recompute only every update_freq steps. An eigh per step is the
        # single most expensive thing this class can do; do not default it to 1.
        if self.step_count % self.update_freq == 0 or self.step_count == 1:
            self.P = self._factor(self.C).to(gradient.dtype)

    def transform(self, gradient: Tensor) -> Tensor:
        if not self.enabled:
            return gradient
        return self.P @ gradient if self.side == "left" else gradient @ self.P

    def inverse_transform(self, update: Tensor) -> Tensor:
        # Same factor again, per Mousse Algorithm 1 line 9 -- see the class docstring.
        if not self.enabled:
            return update
        return self.P @ update if self.side == "left" else update @ self.P


# Back-compat: the old name asserted a method it never implemented.
KLShampooPreconditioner = ShampooPreconditioner


class RotatedSOAPPreconditioner(Preconditioner):
    r"""NOT SOAP. A rotated two-sided whitening; see the warning below.

    CONTINUITY 2026-09-22: real SOAP projects into the eigenbasis with ROTATION ONLY --
    the reference implementation (Emerging-Optimizers shampoo/soap_v3.py) has
    project_in(x) = Q_L.mT @ x @ Q_R and project_out(x) = Q_L @ x @ Q_R.mT, an exact
    inverse pair. The eigenvalue scaling lives in the Adam second moment computed
    INSIDE that basis; that is the whole point of SOAP. This class folds the scaling
    into the projection and applies it on the way in AND again on the way out, so the
    round trip is not an identity and no Adam runs in the rotated frame.

    It also lacks the reference's two economies: an incremental eigenbasis via
    orthogonal_iteration(power_iter_steps=1) rather than a full eigh, and re-projection
    of the stored moment when the basis rotates.

    Left in place for reference. Prefer ShampooPreconditioner + Mousse: by the Mousse
    paper's own argument (p6) SOAP needs an extra second-moment state that Mousse does
    not, and our checkpoints are already 3.1 GiB on a 16 GB card.

    Original docstring follows.

    SOAP (Second-Order Optimization in All Directions) Preconditioner.

    Whitens gradient along the eigenbases of L and R:
        L = Q_L \Lambda_L Q_L^T
        R = Q_R \Lambda_R Q_R^T
        G_{rot} = Q_L^T @ G @ Q_R

    Reverse unwhitening pulls back:
        U = Q_L @ M @ Q_R^T
    """

    name = "soap"

    def __init__(
        self,
        shape: Tuple[int, int],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        delta: float = 1e-4,
        update_freq: int = 1,
    ) -> None:
        self.d_out, self.d_in = shape
        self.beta = beta
        self.delta = delta
        self.update_freq = update_freq
        self.step_count = 0

        self.L = torch.zeros((self.d_out, self.d_out), device=device, dtype=torch.float32)
        self.R = torch.zeros((self.d_in, self.d_in), device=device, dtype=torch.float32)

        self.Q_L = torch.eye(self.d_out, device=device, dtype=dtype)
        self.Q_R = torch.eye(self.d_in, device=device, dtype=dtype)
        self.scale_L = torch.ones(self.d_out, device=device, dtype=dtype)
        self.scale_R = torch.ones(self.d_in, device=device, dtype=dtype)

    def update_accumulators(self, gradient: Tensor) -> None:
        G = gradient.float()
        self.step_count += 1

        self.L.mul_(self.beta).addmm_(G, G.transpose(-1, -2), alpha=(1.0 - self.beta))
        self.R.mul_(self.beta).addmm_(G.transpose(-1, -2), G, alpha=(1.0 - self.beta))

        if self.step_count % self.update_freq == 0:
            bc = 1.0 - self.beta ** self.step_count
            L_hat = (self.L / bc) + self.delta * torch.eye(self.d_out, device=self.L.device)
            R_hat = (self.R / bc) + self.delta * torch.eye(self.d_in, device=self.R.device)

            eig_L, Q_L = torch.linalg.eigh(L_hat)
            eig_R, Q_R = torch.linalg.eigh(R_hat)

            self.Q_L = Q_L.to(gradient.dtype)
            self.Q_R = Q_R.to(gradient.dtype)
            self.scale_L = eig_L.clamp_min(1e-12).pow(-0.25).to(gradient.dtype)
            self.scale_R = eig_R.clamp_min(1e-12).pow(-0.25).to(gradient.dtype)

    def transform(self, gradient: Tensor) -> Tensor:
        """Rotate into eigenspace and scale."""
        rot = self.Q_L.transpose(-1, -2) @ gradient @ self.Q_R
        return rot * self.scale_L.unsqueeze(1) * self.scale_R.unsqueeze(0)

    def inverse_transform(self, update: Tensor) -> Tensor:
        """Rotate back from eigenspace."""
        scaled = update * self.scale_L.unsqueeze(1) * self.scale_R.unsqueeze(0)
        return self.Q_L @ scaled @ self.Q_R.transpose(-1, -2)
