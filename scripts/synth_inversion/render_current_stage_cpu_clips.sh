#!/usr/bin/env bash
set -uo pipefail

SAO=/home/kim/Projects/SAO
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a0"
FLOW_CKPT="$ROOT/flow_best.pt"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') Starting CPU audio clip generation for A0 current stage ==="
echo "Checkpoint: $FLOW_CKPT"

for TAU in 1.0 0.7 0.3; do
  TAG="t$(echo "$TAU" | tr -d '.')"
  if [ "$TAG" = "t10" ]; then TAG_LABEL="t1.0"; fi
  if [ "$TAG" = "t07" ]; then TAG_LABEL="t0.7"; fi
  if [ "$TAG" = "t03" ]; then TAG_LABEL="t0.3"; fi
  
  OUT_DIR="$ROOT/eval_cpu_$TAG"
  echo "--- Generating clips for tau = $TAU ($TAG_LABEL) -> $OUT_DIR ---"
  nice -n 10 "$PYTHON" "$HERE/invert_stem_collection_v3_cpu.py" \
    --flow_ckpt "$FLOW_CKPT" \
    --cond_tau "$TAU" \
    --out_dir "$OUT_DIR" \
    --no_user_copy
done

echo "--- Generating comparative A/B clips & metrics across tau levels ---"
COMP_DIR="$ROOT/eval_cpu_comparison"
nice -n 10 "$PYTHON" "$HERE/compare_clip_sets.py" "$COMP_DIR" \
  tau_1_0="$ROOT/eval_cpu_t10" \
  tau_0_7="$ROOT/eval_cpu_t07" \
  tau_0_3="$ROOT/eval_cpu_t03"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') CPU clip generation complete! ==="
echo "Comparison files written to: $COMP_DIR"
