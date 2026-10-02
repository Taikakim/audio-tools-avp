#!/usr/bin/env bash
set -e

TRAIN_PY="/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python"
DATA_PY="/home/kim/Projects/synth_env/bin/python"
H5_PATH="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5"
OUT_DIR="/run/media/kim/Mantu/surge_200k_models/overtraining_suite"

mkdir -p "$OUT_DIR"
# Run the COMMITTED scripts next to this file. v1 cd'd into an agent scratch directory
# (~/.gemini/antigravity/.../scratch), so the code that produced the reported numbers was
# not necessarily the code in the repo.
cd "$(dirname "$(readlink -f "$0")")"
echo "Code: $(pwd) @ $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

export HIP_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo "=== Overnight Overtraining Suite: 200 Epochs SF + 100 Epochs AdamW ==="
echo "=========================================================================="
echo "Start Time: $(date)"
echo "Target Dataset: $H5_PATH"
echo "Output Directory: $OUT_DIR"

# -----------------------------------------------------------------------------
# Run 1: 200 Epochs Overtraining with NorMuon + Schedule-Free + Radial Brake
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 1/2] Launching 200-Epoch NorMuon + Schedule-Free Overtraining <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G03_resmlp_200ep_normuon_sf" \
    --model_type "resmlp" \
    --opt_family "normuon_sf" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --batch_size 64 \
    --epochs 200 \
    --device "cuda:0"

# -----------------------------------------------------------------------------
# Run 2: 100 Epochs Pure Decoupled AdamW Baseline (Cosine Annealing)
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 2/2] Launching 100-Epoch Pure AdamW Baseline <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G04_resmlp_100ep_adamw" \
    --model_type "resmlp" \
    --opt_family "adamw" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_adam 0.001 \
    --batch_size 64 \
    --epochs 100 \
    --device "cuda:0"

echo -e "\n=========================================================================="
echo "=== Training Complete! Output Leaderboard: ==="
cat "$OUT_DIR/leaderboard_v2.tsv"
echo "Finished at: $(date)"
