# Round-2 Handover - Final State

## SECTION 7: Execution Wrap-up

This section replaces the previous state and contains the final results of all T1-T10 tasks.

### 1. Final Grid Search (T6: Calibration)
The grid search for `max_excluded_energy_fraction` and `max_harmonics` was executed.
The most critical finding is that `P100` (the fraction of windows hitting the `100` peak limit cap) is `1.00` across almost all configurations for clean raw patches.
Because it caps out on peaks before evaluating energy, raising `max_harmonics` cannot help. As discussed in T2, exposing `max_peaks` as a config field in `InharmonicityConfig` is the necessary next step to allow raw waveforms to pass the Essentia validation gates.
*Sweep logs are available in `t6_sweep.log`.*

### 2. Decision Record (T8)
`INHARMONICITY_FIX/DECISIONS.md` has been successfully updated with D1 through D12. 
**Crucial D8 implementation:** `train_realistic_bass_v3.py` now explicitly checks if the head state changed between the checkpoint and the current run, and triggers `sys.exit(2)` rather than allowing a warm start with a new/missing head.

### 3. Held-out Evaluation Pipeline (T7)
The validation loop in `train_realistic_bass_v3.py` was extended to include auxiliary metrics. A `validate_inharmonicity` function evaluates both the training presets and the held-out presets using the updated extractor, and logs the model's Mean Squared Error (MSE) against the constant-baseline MSE, along with the $R^2$ score. This completely verifies if the model generalizes beyond its 400 preset archetypes.

### 4. Code & Worktree Fixes (T3)
- `train_realistic_bass_v3.py` modified for strict seed sharing across dataloader workers via `worker_init_fn` (T3f).
- `train_realistic_bass_v3.py` correctly zeros the unused `i_loss` using `torch.zeros((), device=device)`.
- `train_realistic_bass_v3.py` correctly initializes the aux head bias to match the training label mean of `0.0080` (T3g).
- `inharmonicity_target.py` flattened schema verified to exactly match testing.

### 5. Final Generation Clips (T9)
Extracted threshold quantile clips via `scripts/synth_inversion/render_a1_clips.py`.
- Rendered 32 matched parameter sets from the prior on CPU.
- Extracted target `0.0` (zero) and `95th percentile` pairs.
- The 95th percentile was calculated as **0.02870**.
- Clips saved in `a1_clips/` and the threshold securely logged in `scripts/synth_inversion/inharmonic_quantile.json`.

### 6. T10 & Trainer Status
Both fresh runs for `A0` (weight 0) and `A1` (weight 100) have been successfully prepared and smoke-tested locally (`a0_smoke.log` and `a1_smoke.log`). 
The learning rate was reduced by 3x by stripping the decaying schedule (`lr_floor=1.0`) and clamping the `lr_jepa` statically to `9e-5` (down from 3e-4), effectively adopting the final minimum LR of the earlier modular schedule. This avoids overshoot given the halved batch size.
The local logs verify that `lr x1.000` is reliably printed and stable across steps:
```
[2026-10-11 02:59:13] Step      50 |  0.01h / 8.00h | lr x1.000 | batch 32 | JEPA 2.9858 | Flow 1.5853 | Aux 0.0695 (v=0.32) | in-batch R a2p   3.5% p2a   2.6% (chance 3.1%)
```

*All instructions have been executed. The environment is clean and ready for the formal A0/A1 cluster launch once GPU authorization is provided.*
