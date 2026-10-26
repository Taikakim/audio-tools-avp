#!/bin/bash
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
# Automated Stage 1 & Stage 2 V7 Mixture-to-Patch Synth Inversion training pipeline

set -euo pipefail

SAO_DIR="/home/kim/Projects/SAO"
PYTHON="/home/kim/Projects/SAO/.venv-rocm10-amd/bin/python"
SCRIPT_DIR="/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion"
OUT_DIR="/home/kim/sao_runs/cgsp_extractor_v7"

mkdir -p "$OUT_DIR"

cleanup() {
    echo "[run_v7] Releasing GPU lock..."
    "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY || true
}
trap cleanup EXIT INT TERM

echo "[run_v7] Waiting for GPU lock..."
while ! "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$; do
    echo "[run_v7] GPU busy, sleeping 30s..."
    sleep 30
done

echo "[run_v7] GPU acquired by ANTIGRAVITY (pid $$)."
echo "[run_v7] Launching Stage 1 CGSP Extractor training (18,000 steps)..."

$PYTHON "$SCRIPT_DIR/train_cgsp_extractor.py" \
    --steps 18000 \
    --batch-size 16 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-steps 500 \
    --weight-decay 1e-2 \
    --temperature 0.1 \
    --sel-margin 0.0 \
    --lambda-inv 1.0 \
    --lambda-algn-a 1.0 \
    --lambda-algn-t 1.0 \
    --lambda-sel 0.25 \
    --lora-rank 8 \
    --lora-alpha 16.0 \
    --num-workers 4 \
    --log-interval 50 \
    --save-interval 1000 \
    --output-dir "$OUT_DIR" \
    --device cuda \
    2>&1 | tee -a "$OUT_DIR/train_stage1.log"

echo "[run_v7] Stage 1 training completed successfully!"
