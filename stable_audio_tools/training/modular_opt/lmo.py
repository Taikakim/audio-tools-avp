r"""Linear Minimization Oracles (LMOs) and Dual Norm Kernels.

Implements scale-invariant first-order steepest descent operators:
- SignLMO: \ell_\infty operator norm, \ell_1 dual norm
- SpectralLMO: S_\infty (operator 2-norm), S_1 (nuclear/trace) dual norm
- ColNormLMO: (2, \infty) column-operator norm, (2, 1) dual norm

All LMOs are scale-invariant: the magnitude of the incoming gradient does not dictate
the magnitude of the update; only its direction and the geometric radius \rho_\ell do.
"""

from __future__ import annotations

import math
from typing import Literal
import torch
from torch import Tensor

from .protocols import NormConstraint


def newton_schulz_cubic(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-12,
) -> Tensor:
    r"""Cubic Newton-Schulz orthogonalization: X_{t+1} = 1.5 * X_t - 0.5 * X_t @ X_t^T @ X_t.

    Guaranteed convergence when all singular values of X_0 lie in (0, \sqrt{3}).
    Initializing X_0 = G / (||G||_F + eps) ensures singular values lie in (0, 1] < \sqrt{3}.
    """
    X = G.float()
    norm = X.norm() + eps
    X = X / norm

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()

    for _ in range(steps):
        # A = X @ X^T (dim x dim, smaller dimension)
        A = X @ X.transpose(-1, -2)
        X = 1.5 * X - 0.5 * (A @ X)

    if transposed:
        X = X.transpose(-1, -2)

    return X.to(G.dtype)


def newton_schulz_quintic(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-12,
) -> Tensor:
    """Quintic Newton-Schulz orthogonalization (Muon NS5).

    Coefficients (3.4445, -4.7750, 2.0315) pull singular values rapidly into [0.7, 1.3].
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    norm = X.norm() + eps
    X = X / norm

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()

    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        AA = A @ A
        B = b * A + c * AA
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-1, -2)

    return X.to(G.dtype)


_CUBIC5_COEFFS = [
    (3.3656576, -3.3420992),
    (2.5744352, -1.4957376),
    (2.5368962, -1.4312570),
    (2.4418906, -1.2764040),
    (2.2230472, -0.9630650),
]


def newton_schulz_cubic5(
    G: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Relaxed cubic Newton-Schulz orthogonalization (arXiv:2606.00371).

    Five-step cubic schedule with c=0. Eliminates the Gram-square matrix
    multiplication (A @ A), reducing per-step matmuls from 3 to 2 (33% FLOP/memory
    traffic savings) and eliminating hipblasLt issues on ROCm.
    """
    X = G.float()
    norm = X.norm() + eps
    X = X / norm

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()

    for a, b in _CUBIC5_COEFFS:
        A = torch.mm(X, X.transpose(-1, -2).contiguous())
        X = torch.addmm(X, A, X, alpha=b, beta=a)

    if transposed:
        X = X.transpose(-1, -2)

    return X.to(G.dtype)


_POLAR_EXPRESS_COEFFS = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
]


def newton_schulz_polar_express(
    G: Tensor,
    steps: int = 8,
    eps: float = 1e-12,
) -> Tensor:
    """Polar Express Newton-Schulz orthogonalization (arXiv:2505.16932).

    Eight-step schedule achieving exact polar factor convergence (singular values
    within 1e-4 of 1.0). Used for exact spectral clipping and hardcapping.
    """
    X = G.float()
    norm = X.norm() + eps
    X = X / norm

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()

    for i in range(min(steps, len(_POLAR_EXPRESS_COEFFS))):
        a, b, c = _POLAR_EXPRESS_COEFFS[i]
        A = X @ X.transpose(-1, -2)
        B = torch.addmm(A, A, A, alpha=c, beta=b)
        X = torch.addmm(X, B, X, alpha=1.0, beta=a)

    if transposed:
        X = X.transpose(-1, -2)

    return X.to(G.dtype)


def spectral_hardcap(
    X: Tensor,
    beta: float = 1.0,
    algorithm: Literal["polar_express", "cubic5", "quintic"] = "polar_express",
) -> Tensor:
    r"""Spectral hardcap function clips singular values from above to be less than beta.

    Computed using matrix sign via Newton-Schulz without any SVD.
    Based on Leloykun (https://leloykun.github.io/ponder/spectral-clipping/).
    """
    orig_dtype = X.dtype
    needs_transpose = X.shape[0] > X.shape[1]
    work = X.float()
    if needs_transpose:
        work = work.transpose(-1, -2).contiguous()

    if algorithm == "polar_express":
        ns_fn = lambda t: newton_schulz_polar_express(t, steps=8)
    elif algorithm == "cubic5":
        ns_fn = newton_schulz_cubic5
    else:
        ns_fn = newton_schulz_quintic

    OX = ns_fn(work)
    aX = torch.add(beta * OX, work, alpha=-1.0)
    result = torch.add(beta * OX, work)
    result = torch.addmm(
        result, aX, torch.mm(ns_fn(aX).transpose(-1, -2), OX), alpha=-1.0
    )
    result = result * 0.5
    if needs_transpose:
        result = result.transpose(-1, -2)
    return result.to(orig_dtype)


def spectral_clip(
    X: Tensor,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    algorithm: Literal["polar_express", "cubic5", "quintic"] = "polar_express",
) -> Tensor:
    r"""Applies spectral clipping to the input tensor, bounding singular values in [sigma_min, sigma_max].

    Computed using matrix sign via Newton-Schulz without any SVD.
    """
    orig_dtype = X.dtype
    needs_transpose = X.shape[0] > X.shape[1]
    work = X.float()
    if needs_transpose:
        work = work.transpose(-1, -2).contiguous()

    if algorithm == "polar_express":
        ns_fn = lambda t: newton_schulz_polar_express(t, steps=8)
    elif algorithm == "cubic5":
        ns_fn = newton_schulz_cubic5
    else:
        ns_fn = newton_schulz_quintic

    OX = ns_fn(work)
    result = (sigma_min + sigma_max) * OX
    identity_matrix = torch.eye(work.shape[0], device=work.device, dtype=torch.float32)
    for s, sign in zip([sigma_min, sigma_max], [1.0, -1.0]):
        A = torch.addmm(s * identity_matrix, OX, work.transpose(-1, -2), beta=1.0, alpha=-1.0)
        B = torch.add(s * OX, work, alpha=-1.0)
        result = torch.addmm(result, ns_fn(A), B, alpha=sign)
    result = result * 0.5

    if needs_transpose:
        result = result.transpose(-1, -2)
    return result.to(orig_dtype)


class SignLMO(NormConstraint):
    r"""Linear Minimization Oracle for \ell_\infty norm ball.

    Primal norm: ||v||_\infty = \max_i |v_i| <= \rho
    LMO solution: v* = -\rho * \text{sign}(g)
    Dual norm: ||g||_1 = \sum_i |g_i|
    """

    name = "sign"

    def compute_lmo(self, gradient: Tensor, radius: float = 1.0) -> Tensor:
        """Projects to \rho * sign(gradient) maximizing <G, v>."""
        return radius * torch.sign(gradient)

    def compute_dual_norm(self, gradient: Tensor, M: Tensor | None = None) -> float:
        r"""Dual norm is \ell_1 norm."""
        return float(gradient.abs().sum().item())


class SpectralLMO(NormConstraint):
    r"""Linear Minimization Oracle for Spectral (S_\infty) norm ball.

    Primal norm: ||V||_2 = \sigma_{\max}(V) <= \rho
    LMO solution: V* = -\rho * U @ V^T (semi-orthogonal matrix)
    Dual norm: ||G||_* = \sum_i \sigma_i(G) (Nuclear / Trace norm)
    """

    name = "spectral"

    def __init__(
        self,
        algorithm: Literal["cubic", "quintic", "cubic5"] = "quintic",
        steps: int = 5,
        eps: float = 1e-12,
    ) -> None:
        self.algorithm = algorithm
        self.steps = steps
        self.eps = eps

    def compute_lmo(self, gradient: Tensor, radius: float = 1.0) -> Tensor:
        r"""Semi-orthogonal projection via iterative Newton-Schulz solver.

        Args:
            gradient: 2D weight matrix [dout, din].
            radius: Radius scale \rho_\ell = \max(1, \sqrt{d_{out}/d_{in}}).

        Returns:
            \rho_\ell * U @ V^T
        """
        if gradient.ndim != 2:
            raise ValueError(f"SpectralLMO requires a 2D matrix, got shape {gradient.shape}")

        if self.algorithm == "cubic5":
            ortho = newton_schulz_cubic5(gradient, eps=self.eps)
        elif self.algorithm == "cubic":
            ortho = newton_schulz_cubic(gradient, steps=self.steps, eps=self.eps)
        else:
            ortho = newton_schulz_quintic(gradient, steps=self.steps, eps=self.eps)

        return radius * ortho

    def compute_dual_norm(self, gradient: Tensor, M: Tensor | None = None) -> float:
        r"""Compute the dual norm: nuclear/trace norm \sum \sigma_i(G).

        Using the Fenchel-Young dual pairing identity:
            \sup_{||X||_{op} <= 1} <X, G> = <U V^T, G> = \sum \sigma_i(G) = ||G||_*
        where M is the polar factor computed via Newton-Schulz (scaled by 1/0.822).
        This avoids rocSOLVER's O(m n^2) SVD and its frequent convergence failures
        on rank-deficient / ill-conditioned deep learning gradient matrices.
        """
        if gradient.ndim != 2:
            return float(gradient.norm().item())
        if M is not None:
            return max(0.0, float(torch.sum(M.float() * gradient.float()).item() / 0.822))
        try:
            with torch.no_grad():
                s = torch.linalg.svdvals(gradient.float())
                return float(s.sum().item())
        except (torch._C._LinAlgError, RuntimeError):
            M_fast = newton_schulz_quintic(gradient)
            return max(0.0, float(torch.sum(M_fast.float() * gradient.float()).item() / 0.822))


class ColNormLMO(NormConstraint):
    r"""Linear Minimization Oracle for (2, \infty) column operator norm.

    Primal norm: ||V||_{2, \infty} = \max_j ||V_{:, j}||_2 <= \rho
    LMO solution: V_{:, j}* = \rho * \frac{G_{:, j}}{||G_{:, j}||_2}
    Dual norm: ||G||_{2, 1} = \sum_j ||G_{:, j}||_2

    Mathematically equivalent to Spectral for 1-hot embedding matrices,
    at vastly reduced computational complexity (O(dout * din) vs O(dout * din^2)).
    """

    name = "colnorm"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def compute_lmo(self, gradient: Tensor, radius: float = 1.0) -> Tensor:
        """Normalize each column to radius \rho."""
        if gradient.ndim != 2:
            # Flatten to 2D if needed
            orig_shape = gradient.shape
            g_2d = gradient.view(gradient.shape[0], -1)
            col_norms = torch.linalg.norm(g_2d, dim=0, keepdim=True).clamp_min(self.eps)
            res = radius * (g_2d / col_norms)
            return res.view(orig_shape)

        col_norms = torch.linalg.norm(gradient, dim=0, keepdim=True).clamp_min(self.eps)
        return radius * (gradient / col_norms)

    def compute_dual_norm(self, gradient: Tensor, M: Tensor | None = None) -> float:
        r"""Dual norm is \sum_j ||G_{:, j}||_2."""
        if gradient.ndim != 2:
            g_2d = gradient.view(gradient.shape[0], -1)
        else:
            g_2d = gradient
        with torch.no_grad():
            col_norms = torch.linalg.norm(g_2d.float(), dim=0)
            return float(col_norms.sum().item())
