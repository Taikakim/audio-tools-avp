# lora_tsd — implementation notes (the things the code comments point at)

Written 2026-09-25 (CONTINUITY) after the Opus critic pass on the first draft. Interface:
`CONTRACT.md`. Spec: `SAO/docs/handovers/2026-09-24-lora-tsd-batched-port.md`.

## state_dict aliasing
torch's `Optimizer.state_dict()` does not copy: its per-param state dicts are the same
objects as `self.state[p]`. A snapshot taken before a further `step()` would silently
change underneath the caller. Both classes therefore return a real copy. `BatchedLoRATSD`
copies to **CPU**, not a same-device deepcopy, so a checkpoint save doesn't briefly double
the optimizer state's VRAM. `Optimizer.load_state_dict` moves each state tensor back to
its param's device and dtype, and the round-trip test covers it. Because the snapshot is a
copy, the momentum buffers are updated **in place** during `step()`.

## Precision
- **CholeskyQR2 Gram in fp64.** `YᵀY` squares the condition number. With an fp32 Gram,
  Cholesky breaks at cond(Y) ≈ 3e3, and the 2r-wide L/R factors reach that on realistic
  gradients: the critic measured 20–80 `torch.linalg.qr` fallbacks per step, and each
  fallback costs the ~12–17 ms ROCm QR latency that batching exists to remove. The Gram,
  the Cholesky and the triangular solve now run in fp64 and Q is cast back. The test
  `test_batched_qr_fp32_moderately_ill_conditioned_no_fallback` runs at cond ≈ 3e4.
- **Near B ≈ 0 the algorithm itself is ill-conditioned in fp32.** `(BᵀB)⁻¹` is about 1e8
  there and amplifies rounding in dA. The upstream-faithful fp32 reference is ~10% off the
  fp64 answer after six steps at |B| ≈ 1e-4, and the batched version matches the
  reference's error (it doesn't beat it). An fp64 r×r inverse was tried and changed
  nothing, so it was reverted. In practice this doesn't matter: B = 0 exactly (our DoRA
  init) takes the exact QR-fallback path, and after one step |B| is ~1e-2, where fp32
  agrees to ~1e-5. The test compares both fp32 paths against fp64 truth instead of against
  each other.
- **Ridge.** `ridge = eps · mean(diag).clamp_min(1e-12)`, clamping the mean first, as
  upstream `_spd_inv` does. The first draft clamped the product, which puts the floor ~1e4×
  higher at B ≈ 0 (killed by the near-zero-B test).
- `ball_iters=5` in fp32: batched vs reference agree to ~5e-3 normwise, against ~2e-5 at
  `ball_iters=1`. The repeated NS iteration amplifies rounding differences (fp64 agrees to
  1e-9). Not a bug, but don't expect bit-parity at high ball_iters.

## Memory and syncs
- The first draft kept every per-group buffer (A, B, grads, momentum stacks, L, R, Q_L,
  Q_R) alive for all 13 shape groups at once: ~5.7 GB of scratch on the real model. Now
  grads die after the L_g/R_g factorisation, Q factors after each NS pass, and each group's
  L/R/A/B after its write-back. Only the r×r / 2r×2r pieces cross groups, because those
  are what get batched across all pairs.
- Host syncs: one per `_batched_qr` call (the fallback check) and one at the end of
  `step()` for `last_stats`. The per-pair `bool(valid[i])` in rebalancing is gone: an
  invalid pair gets c = 1, and ×1 / ÷1 are exact.

## Semantics worth knowing before tuning
- `max_delta_norm` clips **per pair** on `lr · ‖P_T-step‖_F`, the linearised ‖ΔW‖ of the
  product B·A (not of the factors, and not the DoRA-normalised weight). It's a trust region
  on how far one adapter's product moves in one step.
- The trainer's `--gradient_clip_val` is nearly inert under LoRA-TSD: the step is
  spectrally normalised (msign), so gradient scale barely reaches the update. It still
  scales the momentum input, which is the one place a gradient spike matters.
- Rank guard: `r > min(m, n)` raises ValueError in both classes. A reduced QR would
  otherwise silently return fewer than r columns.
- `qr_fallbacks` in `last_stats`: expect 3–4 per pair on step 1 (B = 0 makes U_B and L rank-deficient; measured 18 for 6 pairs at ball_iters=1, 24 at 5), then 0. A
  persistent non-zero count means conditioning is breaking down. Watch it as
  `train/tsd_qr_fallbacks`.

## After the cloud review (2026-09-25)
- **Hyperparameters come from `param_groups[0]` at every step**, the PyTorch convention. A torch LR
  scheduler, or a `load_state_dict`, now actually changes the step; before, both were silently
  ignored. A scheduler moves only `lr`. `lr_magnitude` is its own key, as in the reference's
  magnitude groups. The `opt.lr` etc. attributes are mirrors, kept because the trainer's
  `train/lr` reads them.
- **Mixed LoRA ranks are refused at construction** (the cross-group r×r batching needs one rank);
  use `LoRATSDReference` for a mixed-rank model.
- **Momentum buffers follow the compute dtype** (converted, not reset). The review's "crash" on mixed
  fp32/fp64 magnitudes didn't reproduce: torch's type promotion absorbed it. What really happened was
  an fp32 buffer in an fp64 step.
- `LoRATSDReference.load_state_dict` clones incoming state, like the batched class.
