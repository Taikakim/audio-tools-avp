#!/usr/bin/env bash
# Resume training from checkpoint_best.pt with dynamic batch decay over 6 hours (32 -> 2)
# Under crash-only watchdog supervisor on cuda:0
set -euo pipefail

OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2}"
PYTHON="${PYTHON:-/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== Realistic-bass dynamic batch decay training -> $OUT_DIR ($(date)) ==="
exec "$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" \
  --resume_from "$OUT_DIR/checkpoint_best.pt" \
  --hours 6.0 \
  --batch_size 32 \
  --min_batch_size 2 \
  --batch_schedule linear_decay \
  --num_workers 4 \
  --lr_jepa 1.127572e-06 \
  --lr_flow 7.517147e-07 \
  --warmup_steps 0 \
  --decay_frac 0.0 \
  --purpose "Continue 6h from checkpoint_best with dynamic batch size decaying 32 -> 2 at final converged LR" \
  --device "cuda:0"
