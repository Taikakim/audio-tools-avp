#!/bin/bash
set -eo pipefail

export SAO=/home/kim/Projects/SAO
export PYTHON=/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
export COND_NOISE=1
export FLOW_LOSS=mae
export MAX_STEPS=26138
export HOURS=10.0

echo "Starting A0 (Weight 0) Training..."
export OUT_DIR=/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a0
export INHARM_WEIGHT=0
export PURPOSE="A0 Control: True Recipe (batch 64)"
bash scripts/synth_inversion/run_v3_recipe.sh

echo "A0 Finished. Starting A1 (Weight 100) Training..."
export OUT_DIR=/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a1
export INHARM_WEIGHT=100.0
export INHARM_CONFIG=scripts/synth_inversion/inharm_config.json
export PURPOSE="A1 Target: True Recipe (batch 64) + Inharmonicity 100"
bash scripts/synth_inversion/run_v3_recipe.sh

echo "All 10-hour runs finished!"
