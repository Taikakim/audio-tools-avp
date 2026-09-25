# lora_tsd — interface contract (all parallel workers code against THIS)

Spec: `SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md` (read it first).
Upstream: `/home/kim/Projects/LoRA-TSD/src/optimizers/lora_tsd.py` (MIT, commit e6ef0e1). Read-only.

## Package layout (stable-audio-tools/stable_audio_tools/training/lora_tsd/)
- `__init__.py`: `from .reference import LoRATSDReference`; `from .batched import BatchedLoRATSD`
  (already written; keep the names).
- `reference.py`: worker A. `batched.py`: worker B. `bench.py`: worker D.
- Tests: `stable-audio-tools/tests/test_lora_tsd_batched.py` (worker A).

## Shared constructor signature (BOTH classes)
```python
class BatchedLoRATSD(torch.optim.Optimizer):          # LoRATSDReference: identical signature
    def __init__(self, named_params,                   # iterable of (name, Parameter), e.g. model.named_parameters()
                 lr: float = 1e-3,                     # upstream lr_A (used for BOTH factors)
                 momentum: float = 0.95,               # upstream momentum_cf; 0 => no momentum
                 ball_iters: int = 1,
                 ns_steps: int = 5,
                 max_delta_norm: float = 0.1,          # per-pair clip on lr * ||dW||_F
                 balance: str = "norm",                # "off" | "norm"  (upstream rebalance_every=1)
                 ridge_eps: float = 1e-8,
                 lr_magnitude: float | None = None,    # None => lr
                 warmup_steps: int = 0):               # linear lr (and lr_magnitude) ramp over the first N steps
```
- Pairing: `X.lora_A` with `X.lora_B` by the name prefix before `.lora_A` / `.lora_B`
  (real key: `model.to_timestep_embed.0.parametrizations.weight.0.lora_A`). Only params with
  `requires_grad`. A: (r, n), B: (m, r).
- Unpaired params: every name ending in `magnitude` takes the multiplicative sign step
  `m <- m * exp(-lr_mag_t * sign(buf))`, `buf = momentum*buf + (1-momentum)*grad`, no weight decay
  (the same rule as `modular_opt/optimizer.py::_magnitude_step`). Any OTHER unpaired param: raise
  ValueError naming it (don't train it silently in a way nobody chose).
- `step(closure=None)` as for torch optimizers. Params with `.grad is None` are skipped (the pair is
  skipped if either factor has no grad).
- `state_dict()` / `load_state_dict()`: per-param momentum buffers under `self.state[p]["momentum_buffer"]`
  (the standard torch layout, so torch's own state_dict round-trips) plus the step counter.
  Round-trip must give an identical next step.
- The reference MUST reproduce upstream `LoRATSD.step` numerically for the same settings (upstream groups
  one [A, B] per param group; the reference loops over pairs one at a time, calling the same maths).
  Mapping: upstream lr_A = lr; lr_B = lr (only affects the unused alpha default); momentum_cf = momentum;
  use_momentum = momentum > 0; balance=True/balance_mode="norm"/rebalance_every=1 iff balance == "norm".
- Do NOT expose `uses_sf_averaging`, `train()`/`eval()` swaps or `set_loss` (the trainer duck-types
  those for Schedule-Free; LoRA-TSD has none).
- Compute in fp32 (or the input dtype if fp64, for tests) regardless of the param dtype; write back in the
  param dtype.
- Telemetry (optional, cheap): `self.last_stats = {"dW_norm_mean":..., "clip_frac":..., "qr_fallbacks":...}`
  set after each step.
