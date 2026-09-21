"""LIFO Transformation Stack for Preconditioner Whitening and Pullback Inversion.

Mathematical symmetry is non-negotiable. A Last-In-First-Out (LIFO) stack ensures that
multi-phase whitening transformations (e.g. Left/Right Kronecker preconditioning,
eigenbasis rotation) are inverted in the exact reverse order of their application:

Forward Whitening:
    G_0 = G
    G_1 = T_1(G_0)   -> stack.push(T_1, T_1_inv)
    G_2 = T_2(G_1)   -> stack.push(T_2, T_2_inv)
    M = LMO(G_2)

Reverse Unwhitening:
    M_1 = T_2_inv(M)   <- stack.pop()
    M_0 = T_1_inv(M_1) <- stack.pop()
    Update = M_0
"""

from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class StackEntry:
    name: str
    inverse_fn: Callable[[Tensor], Tensor]
    metadata: dict[str, Any] | None = None


class LIFOTransformationStack:
    """A LIFO stack tracking forward whitening transforms to guarantee exact reverse pullback."""

    def __init__(self) -> None:
        self._stack: list[StackEntry] = []

    def push(
        self,
        name: str,
        inverse_fn: Callable[[Tensor], Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Push an inverse transformation onto the stack."""
        self._stack.append(StackEntry(name=name, inverse_fn=inverse_fn, metadata=metadata))

    def pop(self) -> StackEntry:
        """Pop the most recent inverse transformation from the stack."""
        if not self._stack:
            raise IndexError("pop from empty LIFOTransformationStack")
        return self._stack.pop()

    def unwhiten(self, update: Tensor) -> Tensor:
        """Apply all stored inverse transforms in reverse order and clear the stack.

        Args:
            update: The update matrix M from the LMO solver in the whitened coordinate space.

        Returns:
            The pulled-back update in the original parameter space.
        """
        curr = update
        while self._stack:
            entry = self._stack.pop()
            curr = entry.inverse_fn(curr)
        return curr

    def clear(self) -> None:
        """Clear any residual stack entries."""
        self._stack.clear()

    def __len__(self) -> int:
        return len(self._stack)

    def is_empty(self) -> bool:
        return len(self._stack) == 0
