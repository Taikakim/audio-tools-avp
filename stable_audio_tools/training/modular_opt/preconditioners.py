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


class KLShampooPreconditioner(Preconditioner):
    """Two-sided Kronecker (Left/Right) Shampoo Preconditioner.

    Maintains covariance matrices:
        L_t = \beta_s L_{t-1} + (1 - \beta_s) G G^T
        R_t = \beta_s R_{t-1} + (1 - \beta_s) G^T G

    Forward Whitening:
        P_L = (L_t + \delta I)^{-1/4}
        P_R = (R_t + \delta I)^{-1/4}
        G_{white} = P_L @ G @ P_R

    Reverse Unwhitening Pullback:
        U_{orig} = P_L @ M @ P_R
    """

    name = "shampoo"

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

        # Covariance accumulators (stored in float32 for numerical stability)
        self.L = torch.zeros((self.d_out, self.d_out), device=device, dtype=torch.float32)
        self.R = torch.zeros((self.d_in, self.d_in), device=device, dtype=torch.float32)

        # Preconditioner factors P_L = L^{-1/4}, P_R = R^{-1/4}
        self.P_L = torch.eye(self.d_out, device=device, dtype=dtype)
        self.P_R = torch.eye(self.d_in, device=device, dtype=dtype)

    def update_accumulators(self, gradient: Tensor) -> None:
        """Update L and R covariances from raw gradient."""
        G = gradient.float()
        self.step_count += 1

        # Exponential moving average of second moments
        self.L.mul_(self.beta).addmm_(G, G.transpose(-1, -2), alpha=(1.0 - self.beta))
        self.R.mul_(self.beta).addmm_(G.transpose(-1, -2), G, alpha=(1.0 - self.beta))

        # Recompute inverse fourth roots when at update frequency
        if self.step_count % self.update_freq == 0:
            # Bias correction
            bc = 1.0 - self.beta ** self.step_count
            L_hat = self.L / bc
            R_hat = self.R / bc
            self.P_L = inv_quarter_spd(L_hat, delta=self.delta).to(gradient.dtype)
            self.P_R = inv_quarter_spd(R_hat, delta=self.delta).to(gradient.dtype)

    def transform(self, gradient: Tensor) -> Tensor:
        """Forward whitening: G_white = P_L @ G @ P_R."""
        return self.P_L @ gradient @ self.P_R

    def inverse_transform(self, update: Tensor) -> Tensor:
        """Reverse unwhitening: pullback U = P_L @ M @ P_R."""
        return self.P_L @ update @ self.P_R


class RotatedSOAPPreconditioner(Preconditioner):
    """SOAP (Second-Order Optimization in All Directions) Preconditioner.

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
