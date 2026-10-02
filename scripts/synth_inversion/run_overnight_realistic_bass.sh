#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass"
mkdir -p "$OUT_DIR"

PYTHON="/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python"
SCRIPT="/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion/train_realistic_bass_overnight.py"

echo "=== Launching 8-Hour Realistic Bass Overnight Training ==="
echo "Output Directory: $OUT_DIR"
echo "Start Time: $(date)"

"$PYTHON" "$SCRIPT" \
  --out_dir "$OUT_DIR" \
  --hours 8.0 \
  --batch_size 32 \
  --num_workers 4 \
  --lr_jepa 9e-5 \
  --lr_flow 6e-5 \
  --checkpoint_interval_steps 1000 \
  --log_interval_steps 50 \
  --device "cuda:0" 2>&1 | tee -a "$OUT_DIR/run.log"

echo "=== Overnight Training Finished at $(date) ==="
