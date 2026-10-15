#!/bin/bash
set -eo pipefail

export SAO=/home/kim/Projects/SAO
export PYTHON=/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
export COND_NOISE=1
export FLOW_LOSS=mae
export MAX_STEPS=12000
export HOURS=5.0

# -----------------------------------------------------------------------------
# Arm A1: Inharmonicity Aux Head Isolation (5.0h)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "Starting Arm A1: Inharmonicity 100 Isolation (5.0h)..."
echo "=========================================================="
export OUT_DIR=/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a1
export INHARM_WEIGHT=100.0
export INHARM_CONFIG=scripts/synth_inversion/inharm_config.json
export FLOW_ENCODER=conv
export PURPOSE="A1 Target: True Recipe (batch 64, 5h) + Inharmonicity 100"
bash scripts/synth_inversion/run_v3_recipe.sh

# -----------------------------------------------------------------------------
# Arm A2: Temporal Audio Encoder (ASP) Isolation (5.0h)
# -----------------------------------------------------------------------------
echo "=========================================================="
echo "Starting Arm A2: Temporal Audio Encoder (ASP) Isolation (5.0h)..."
echo "=========================================================="
export OUT_DIR=/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a2
export INHARM_WEIGHT=0
unset INHARM_CONFIG
export FLOW_ENCODER=temporal_asp
export PURPOSE="A2 Target: True Recipe (batch 64, 5h) + Temporal ASP Flow Encoder"
bash scripts/synth_inversion/run_v3_recipe.sh

echo "Both 5-hour runs (A1 and A2) finished successfully!"
