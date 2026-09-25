"""LoRA-TSD (tangent-space spectral descent for LoRA, arXiv 2609.02734) for our DoRA adapters.

reference.py -- faithful per-pair port of the upstream optimizer (correctness oracle)
batched.py   -- BatchedLoRATSD: the same maths batched across adapters (the one to train with)
See CONTRACT.md and SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md.
"""
from .reference import LoRATSDReference
from .batched import BatchedLoRATSD

__all__ = ["LoRATSDReference", "BatchedLoRATSD"]
