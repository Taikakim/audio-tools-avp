#!/usr/bin/env bash
# queue_a3_after_a2.sh — Automatically launches Arm A3 once A2 finishes.
set -euo pipefail

cd /home/kim/Projects/SAO/stable-audio-tools

echo "[queue_a3] Waiting for Arm A2 pipeline (PID 2293100) to complete..."
while ps -p 2293100 > /dev/null 2>&1; do
    sleep 30
done

echo "[queue_a3] Pipeline (A2) process exited at $(date)!"
echo "[queue_a3] Allowing GPU release to settle..."
sleep 15

while pgrep -f "train_realistic_bass_v3.py" > /dev/null 2>&1; do
    echo "[queue_a3] Waiting for trainer process to fully terminate..."
    sleep 15
done

echo "=========================================================="
echo "Starting Arm A3: Temporal ASP + Inharmonicity 100 (5.0h)..."
echo "=========================================================="

export SAO=/home/kim/Projects/SAO
export PYTHON=/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
export COND_NOISE=1
export FLOW_LOSS=mae
export MAX_STEPS=12000
export HOURS=5.0
export OUT_DIR=/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a3
export INHARM_WEIGHT=100.0
export INHARM_CONFIG=scripts/synth_inversion/inharm_config.json
export FLOW_ENCODER=temporal_asp
export PURPOSE="A3 Target: True Recipe (batch 64, 5h) + Temporal ASP Flow Encoder + Inharmonicity 100"

bash scripts/synth_inversion/run_v3_recipe.sh

echo "[queue_a3] Arm A3 finished successfully at $(date)!"
