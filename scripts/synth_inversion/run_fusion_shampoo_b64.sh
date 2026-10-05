#!/usr/bin/env bash
# Fresh training run from scratch matching paper model size (~53M, ff_dim=1152)
# with batch size 64, 2x LR, Shampoo, Schedule-Free, NorMuon, NS5, MONA, and SNR gating
# Supervised by watchdog_supervisor.py on cuda:0
set -euo pipefail

OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/fusion_shampoo_sf_b64}"
PYTHON="${PYTHON:-/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

echo "=== FusionOpt (Shampoo + Schedule-Free + 53M) training -> $OUT_DIR ($(date)) ==="
exec "$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" \
  --hours 8.0 \
  --batch_size 64 \
  --optimizer fusion \
  --fusion_components "shampoo,sf,normuon,ns5,mona,snr" \
  --ff_dim 1152 \
  --num_workers 4 \
  --lr_jepa 1.8e-4 \
  --lr_flow 1.2e-4 \
  --purpose "Scratch run 53M paper-sized model at batch 64 with FusionOpt Shampoo, Schedule-Free, NorMuon, MONA, SNR" \
  --device "cuda:0"
