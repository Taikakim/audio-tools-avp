"""Protocols and typing for Modular Stage-Based Optimizer Framework.

Defines the mathematical contracts for:
- NormConstraint: Scale-invariant Linear Minimization Oracles (LMOs) and dual norms.
- Preconditioner: Forward whitening and reverse unwhitening transforms with state updates.
- Tensor shape annotations for geometric routing.
"""

from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable
import torch
from torch import Tensor

# Semantic shape annotations
Weight2D = Annotated[Tensor, "dout din"]
Bias1D = Annotated[Tensor, "dout"]
Kernel4D = Annotated[Tensor, "cout cin h w"]


@runtime_checkable
class NormConstraint(Protocol):
    """Protocol for norm constraints and Linear Minimization Oracles (LMOs).

    In steepest descent with respect to an operator norm ||.||:
        v* = argmax_{||v|| <= \rho} <G, v>
    The LMO returns this scale-invariant unit/radius-scaled vector.
    """

    name: str

    def compute_lmo(self, gradient: Tensor, radius: float = 1.0) -> Tensor:
        """Solve for the norm-ball boundary vector maximizing alignment with gradient.

        Args:
            gradient: The incoming gradient (or whitened gradient).
            radius: The target radius \rho_\ell (e.g. \max(1, \sqrt{d_{out}/d_{in}})).

        Returns:
            The update direction living on the boundary of the norm ball of radius `radius`.
        """
        ...

    def compute_dual_norm(self, gradient: Tensor) -> float:
        """Compute the dual norm ||G||_* of the gradient.

        Required for directional derivative monitoring, scale invariance, and step-size logic:
        - \ell_1 dual for \ell_\infty (Sign)
        - Trace/Nuclear norm dual for Spectral (S_\infty)
        - (2, 1) norm dual for ColNorm (2, \infty)
        """
        ...


@runtime_checkable
class Preconditioner(Protocol):
    """Protocol for curvature / covariance preconditioners operating on 2D matrices."""

    name: str

    def update_accumulators(self, gradient: Weight2D) -> None:
        """Update internal Left (L) and Right (R) covariance statistics from raw gradient."""
        ...

    def transform(self, gradient: Weight2D) -> Weight2D:
        """Forward whitening: map gradient from original space to preconditioned space.

        Typically G_white = P_L @ G @ P_R.
        """
        ...

    def inverse_transform(self, update: Weight2D) -> Weight2D:
        """Reverse unwhitening: pullback from preconditioned space to original space.

        Preserves the mathematical duality of the LMO update in the original metric.
        Typically U_orig = P_L @ M @ P_R (or P_L^{-1} @ M @ P_R^{-1}).
        """
        ...
