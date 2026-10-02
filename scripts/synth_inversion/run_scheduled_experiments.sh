#!/usr/bin/env bash
# run_scheduled_experiments.sh
# Runner for Phase 2 Synth Inversion Experiments:
# 1. B01: Deep & Wide Architecture (768d, 12 layers, 14.8M params) with constant batch B=64
# 2. B02: Baseline Architecture (512d, 6 layers) with Power-of-Two batch schedule (256 -> 1)
# 3. B03: Baseline Architecture (512d, 6 layers) with Linear batch schedule (128 -> 1)
# 4. B04: Deep & Wide Architecture (768d, 12 layers) with Power-of-Two batch schedule (256 -> 1)
#
# All runs standardized to 20 epochs with full ModularOptimizer (all brakes on except SNR gate).
set -e

PY="/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python"
H5_PATH="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5"
OUT_DIR="/run/media/kim/Mantu/surge_200k_models/v2_suite"
REPO_DIR="/home/kim/Projects/SAO/stable-audio-tools"

cd "$REPO_DIR"
export HIP_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo "=== Phase 2 Synth Inversion Suite: Scaling Architecture & Batch Sched ==="
echo "=========================================================================="
echo "Start: $(date)"

# Run 1: Deep & Wide (768-dim, 12 layers) with constant batch B=64
echo -e "\n>>> Launching B01_resmlp_768d_12L_20ep_modular <<<"
$PY scripts/synth_inversion/train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "B01_resmlp_768d_12L_20ep_modular" \
    --model_type "resmlp" \
    --opt_family "modular" \
    --hidden_dim 768 \
    --num_layers 12 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --radial_brake 0.85 \
    --adam_warmup_steps 100 \
    --batch_size 64 \
    --batch_schedule "none" \
    --epochs 20 \
    --device "cuda:0"

# Run 2: Baseline (512-dim, 6L) with Power-of-Two Batch Schedule (256 -> 1)
echo -e "\n>>> Launching B02_resmlp_512d_power2_batch_20ep <<<"
$PY scripts/synth_inversion/train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "B02_resmlp_512d_power2_batch_20ep" \
    --model_type "resmlp" \
    --opt_family "modular" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --radial_brake 0.85 \
    --adam_warmup_steps 100 \
    --batch_schedule "power_of_two" \
    --initial_batch_size 256 \
    --final_batch_size 1 \
    --epochs 20 \
    --device "cuda:0"

# Run 3: Baseline (512-dim, 6L) with Linear Batch Schedule (128 -> 1)
echo -e "\n>>> Launching B03_resmlp_512d_linear_batch_20ep <<<"
$PY scripts/synth_inversion/train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "B03_resmlp_512d_linear_batch_20ep" \
    --model_type "resmlp" \
    --opt_family "modular" \
    --hidden_dim 512 \
    --num_layers 6 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --radial_brake 0.85 \
    --adam_warmup_steps 100 \
    --batch_schedule "linear" \
    --initial_batch_size 128 \
    --final_batch_size 1 \
    --epochs 20 \
    --device "cuda:0"

# Run 4: Deep & Wide (768-dim, 12 layers) with Power-of-Two Batch Schedule (256 -> 1)
echo -e "\n>>> Launching B04_resmlp_768d_power2_batch_20ep <<<"
$PY scripts/synth_inversion/train_bracket.py \
    --h5_path "$H5_PATH" \
    --output_dir "$OUT_DIR" \
    --run_id "B04_resmlp_768d_power2_batch_20ep" \
    --model_type "resmlp" \
    --opt_family "modular" \
    --hidden_dim 768 \
    --num_layers 12 \
    --lr_muon 0.01 \
    --lr_adam 0.001 \
    --radial_brake 0.85 \
    --adam_warmup_steps 100 \
    --batch_schedule "power_of_two" \
    --initial_batch_size 256 \
    --final_batch_size 1 \
    --epochs 20 \
    --device "cuda:0"

echo -e "\n=========================================================================="
echo "=== Phase 2 Suite Complete at $(date) ==="
cat "$OUT_DIR/leaderboard_v2.tsv"
