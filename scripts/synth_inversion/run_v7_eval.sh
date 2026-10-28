#!/bin/bash
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
# Automated evaluation and local site generation for V7 CGSP Extractor

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAO_DIR="/home/kim/Projects/SAO"
PYTHON="/home/kim/Projects/SAO/.venv-rocm10-amd/bin/python"
SCRIPT_DIR="/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion"
OUT_DIR="/home/kim/evals_aac/synth_inversion_v7"

cleanup() {
    echo "[eval_v7] Releasing GPU lock..."
    "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY || true
}
trap cleanup EXIT INT TERM

echo "[eval_v7] Waiting for GPU lock..."
while ! "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$; do
    echo "[eval_v7] GPU busy, sleeping 15s..."
    sleep 15
done

echo "[eval_v7] GPU acquired by ANTIGRAVITY (pid $$)."
echo "[eval_v7] Running V7 CGSP evaluation and building local audition board..."

$PYTHON "$SCRIPT_DIR/eval_v7_cgsp_benchmark.py" \
    --extractor-ckpt "/home/kim/sao_runs/cgsp_extractor_v7/cgsp_extractor_best.pt" \
    --bank-dir "/home/kim/sao_runs/bank_v6" \
    --multitrack-dir "/run/media/kim/Mantu/avp-stems-original-classified" \
    --out-dir "$OUT_DIR" \
    --device cuda

echo "[eval_v7] Evaluation and local site generation completed successfully!"
