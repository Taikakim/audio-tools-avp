#!/usr/bin/env bash
set -e

TRAIN_PY="/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python"
H5_PATH="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5"
OUT_DIR="/run/media/kim/Mantu/surge_200k_models/v2_suite"

mkdir -p "$OUT_DIR"
cd "$(dirname "$(readlink -f "$0")")"
echo "Code: $(pwd) @ $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

export HIP_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo "=== Full Synth Inversion v2 Suite (Re-Runs + Overtraining + AdamW) ==="
echo "=========================================================================="
echo "Start Time: $(date)"
echo "Target Dataset: $H5_PATH"
echo "Output Directory: $OUT_DIR"

# -----------------------------------------------------------------------------
# Stage 1: Re-run 30-Epoch 200k ResMLP Baseline (v2 categorical heads)
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 1/4] Re-running 30-Epoch ResMLP Baseline (G01_v2) <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G01_resmlp_200k_v2_30ep" \
    --model_type "resmlp" \
    --opt_family "normuon_sf" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --batch_size 64 \
    --epochs 30 \
    --device "cuda:0"

# -----------------------------------------------------------------------------
# Stage 2: Re-run 30-Epoch 200k Flow Matching Baseline (v2 time-scale & codec)
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 2/4] Re-running 30-Epoch Flow Matching Baseline (G02_v2) <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G02_deepflow_200k_v2_30ep" \
    --model_type "flow" \
    --opt_family "normuon_sf" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --batch_size 64 \
    --epochs 30 \
    --device "cuda:0"

# -----------------------------------------------------------------------------
# Stage 3: 200 Epochs Extended Overtraining with NorMuon + Schedule-Free
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 3/4] 200-Epoch Overtraining Suite (NorMuon + SF + Radial Brake) <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G03_resmlp_200k_v2_200ep_sf" \
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
# Stage 4: 100 Epochs Pure Decoupled AdamW Baseline (Cosine Annealing)
# -----------------------------------------------------------------------------
echo -e "\n>>> [Stage 4/4] 100-Epoch Pure Decoupled AdamW Baseline <<<"
$TRAIN_PY train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "G04_resmlp_200k_v2_100ep_adamw" \
    --model_type "resmlp" \
    --opt_family "adamw" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_adam 0.001 \
    --batch_size 64 \
    --epochs 100 \
    --device "cuda:0"

echo -e "\n=========================================================================="
echo "=== All 4 Runs Complete! Leaderboard: ==="
cat "$OUT_DIR/leaderboard_v2.tsv"
echo "Finished at: $(date)"
