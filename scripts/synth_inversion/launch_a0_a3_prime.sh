#!/usr/bin/env bash
# launch_a0_a3_prime.sh
# Automated 2x2 Factorial Runner for Synth Inversion Arms A0 and A3':
#   - Arm A0: Baseline Conv Flow, No Aux Head, Seed 42, 5.0h (12k steps)
#   - Arm A3': Temporal ASP Flow, Inharmonicity 100, Seed 42, 5.0h (12k steps)
# Followed by automatic Tier T0/T1 pre-registered evaluation.
#
# Usage:
#   ./launch_a0_a3_prime.sh                  # Run immediately
#   ./launch_a0_a3_prime.sh --start-at 22:00 # Sleep until 22:00 (cheap night electricity), then run

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="/home/kim/Projects/SAO/stable-audio-tools"
cd "$REPO_DIR"

START_AT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-at)
      START_AT="$2"
      shift 2
      ;;
    --now)
      START_AT=""
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--start-at HH:MM] [--now]"
      echo ""
      echo "Options:"
      echo "  --start-at HH:MM  Sleep until HH:MM (e.g. 22:00 for cheap off-peak electricity), then start"
      echo "  --now             Start immediately"
      echo "  --help, -h        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--start-at HH:MM] [--now]" >&2
      exit 1
      ;;
  esac
done

BASE_OUT="/run/media/kim/Mantu2/surge_200k_models"
SUITE_LOG="$BASE_OUT/suite_a0_a3_prime.log"
mkdir -p "$BASE_OUT"

exec > >(tee -a "$SUITE_LOG") 2>&1

# Delay if a specific start time was requested (for cheap off-peak electricity)
if [ -n "$START_AT" ]; then
  TARGET_SEC=$(date -d "$START_AT" +%s 2>/dev/null || date -d "tomorrow $START_AT" +%s)
  NOW_SEC=$(date +%s)
  if [ "$TARGET_SEC" -le "$NOW_SEC" ]; then
    TARGET_SEC=$(date -d "tomorrow $START_AT" +%s)
  fi
  SLEEP_SEC=$((TARGET_SEC - NOW_SEC))
  HOURS_FLOAT=$(awk -v s="$SLEEP_SEC" 'BEGIN { printf "%.2f", s / 3600 }')
  echo "=========================================================================="
  echo "SCHEDULED START AT: $START_AT ($(date -d "@$TARGET_SEC"))"
  echo "Sleeping for $SLEEP_SEC seconds (~$HOURS_FLOAT hours) until off-peak electricity..."
  echo "Log file: $SUITE_LOG"
  echo "=========================================================================="
  sleep "$SLEEP_SEC"
fi

echo "=========================================================================="
echo "LAUNCHING A0 + A3' TRAINING SUITE AT $(date)"
echo "=========================================================================="

export SAO="/home/kim/Projects/SAO"
export PYTHON="/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python"
export COND_NOISE=1
export FLOW_LOSS=mae
export MAX_STEPS=12000
export HOURS=5.0
export BATCH_SIZE=64
export NUM_WORKERS=8
export SEED=42
export DEVICE="cuda:0"

# -----------------------------------------------------------------------------
# 1. Arm A0: Baseline Conv Flow, No Aux Head, Seed 42 (5.0h / 12,000 steps)
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo ">>> STEP 1/2: Starting Arm A0 (Conv Flow, No Aux, Seed 42, 5.0h) <<<"
echo "Start time: $(date)"
echo "=========================================================================="

export OUT_DIR="$BASE_OUT/real_bass_v3_a0_seeded"
export FLOW_ENCODER=conv
export INHARM_WEIGHT=0
unset INHARM_CONFIG 2>/dev/null || true
export PURPOSE="A0 Control: True Recipe (batch 64, 5h, seed 42) Baseline Conv Flow + No Aux Head"

bash scripts/synth_inversion/run_v3_recipe.sh

echo "Arm A0 completed at $(date)!"
echo "Allowing GPU memory and thermals to settle..."
sleep 30

# Ensure all processes cleared
while pgrep -f "train_realistic_bass_v3.py" > /dev/null 2>&1; do
  echo "Waiting for previous trainer processes to terminate..."
  sleep 15
done

# -----------------------------------------------------------------------------
# 2. Arm A3': Temporal ASP Flow, Inharmonicity 100, Seed 42 (5.0h / 12,000 steps)
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo ">>> STEP 2/2: Starting Arm A3' (Temporal ASP, Aux 100, Seed 42, 5.0h) <<<"
echo "Start time: $(date)"
echo "=========================================================================="

export OUT_DIR="$BASE_OUT/real_bass_v3_a3_prime"
export FLOW_ENCODER=temporal_asp
export INHARM_WEIGHT=100.0
export INHARM_CONFIG=scripts/synth_inversion/inharm_config.json
export PURPOSE="A3' Replicate: True Recipe (batch 64, 5h, seed 42) Temporal ASP Flow + Inharmonicity 100"

bash scripts/synth_inversion/run_v3_recipe.sh

echo "Arm A3' completed at $(date)!"
echo "Allowing GPU memory and thermals to settle..."
sleep 30

# -----------------------------------------------------------------------------
# 3. Post-Run Automatic Tier T0/T1 Pre-Registered Evaluation
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo ">>> STEP 3/3: Running Automated 2x2 Factorial Evaluation Suite <<<"
echo "Start time: $(date)"
echo "=========================================================================="

EVAL_DIR="$BASE_OUT/eval_v4_2x2"
mkdir -p "$EVAL_DIR"

$PYTHON scripts/synth_inversion/heldout_eval_v4.py \
  --run A0="$BASE_OUT/real_bass_v3_a0_seeded" \
  --run A1="$BASE_OUT/real_bass_v3_a1" \
  --run A2="$BASE_OUT/real_bass_v3_a2" \
  --run A3="$BASE_OUT/real_bass_v3_a3" \
  --run A3_prime="$BASE_OUT/real_bass_v3_a3_prime" \
  --kinds sf online ema --tag latest --extra_n 2048 \
  --out "$EVAL_DIR" --device cuda:0

$PYTHON scripts/synth_inversion/compare_runs_v4.py \
  --dir "$EVAL_DIR" \
  --plan scripts/synth_inversion/eval_plan_v4_2x2.json \
  --out "$EVAL_DIR/report_v4_2x2.md"

echo "=========================================================================="
echo "ALL TRAINING AND EVALUATION TASKS COMPLETED SUCCESSFULLY AT $(date)!"
echo "Comparative Report: $EVAL_DIR/report_v4_2x2.md"
echo "=========================================================================="
