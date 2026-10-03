#!/usr/bin/env bash
# Overnight joint Synth-JEPA + flow training on the real-preset bass prior, under the crash-only
# watchdog. Prerequisite: a v2 manifold (extract_real_bass_manifold.py; see README.md).
# Resumes automatically if OUT_DIR already holds a v2 checkpoint_latest.pt.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2}"
PYTHON="${PYTHON:-/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== Realistic-bass overnight training -> $OUT_DIR ($(date)) ==="
"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" \
  --hours 8.0 \
  --batch_size 32 \
  --num_workers 4 \
  --lr_jepa 9e-5 \
  --lr_flow 6e-5 \
  --purpose "${PURPOSE:-Synth-JEPA + flow on the v2 (plugin-calibrated) real-preset bass prior}" \
  --device "cuda:0"
echo "=== Finished $(date); see $OUT_DIR/training.log, val.jsonl, watchdog.log ==="
