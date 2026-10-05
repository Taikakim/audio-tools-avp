#!/usr/bin/env bash
# Fresh training run from scratch using ModularOptimizer with advanced options from recent audio runs:
# - SCION role-based LMOs (SpectralLMO, SignLMO, ColNormLMO)
# - SCION Table 3 radius scaling rho_ell = max(1, sqrt(d_out / d_in))
# - KL-Shampoo Kronecker preconditioning (whitening=shampoo)
# - Cubic5 Newton-Schulz polynomial (ns_poly=cubic5)
# - Radial Brake 0.8 (soft limiting parameter norm inflation)
# - Variance-Aware Dynamic Dampening (var_dampening=1.15)
# - Schedule-Free averaging (sf) with sf_c_warmup=1000 burn-in
# - Everett & Qiu overtraining-aware weight decay (wd_jepa=0.02, wd_overtraining)
# - SNR gate OFF (disabled, avoiding artificial LR braking)
# - 5% unconstrained random patches across the full parameter space
# - 53M paper-sized model (ff_dim=1152)
# - Batch size 64, 2x LR (JEPA 1.8e-4, Flow 1.2e-4)
# Supervised by watchdog_supervisor.py on cuda:0
set -euo pipefail

export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=2
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0

OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64}"
PYTHON="${PYTHON:-/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== ModularOptimizer (SCION + Shampoo + cubic5 + VADD + Radial Brake + SF + 53M + 5% random) -> $OUT_DIR ($(date)) ==="
exec "$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" \
  --hours 14.0 \
  --batch_size 64 \
  --optimizer modular \
  --whitening shampoo \
  --ns_poly cubic5 \
  --radial_brake 0.8 \
  --var_dampening 1.15 \
  --sf_c_warmup 1000 \
  --sf_r 1.0 \
  --wd_jepa 0.02 \
  --wd_overtraining \
  --random_patch_frac 0.05 \
  --ff_dim 1152 \
  --num_workers 6 \
  --prefetch_factor 4 \
  --lr_jepa 1.8e-4 \
  --lr_flow 1.2e-4 \
  --purpose "Continuation 14h total (8h+6h) 53M model at batch 64 with ModularOptimizer SCION LMOs, Shampoo, cubic5, Radial Brake 0.8, VADD 1.15, SF c_warmup 1000, WD overtraining, SNR off, 5% random patches" \
  --device "cuda:0"
