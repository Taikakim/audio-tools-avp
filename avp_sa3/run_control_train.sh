#!/usr/bin/env bash
# Launch wrapper for SA3 control-adapter training (onset-density / scalar heads).
#
# Bakes in the environment that lets a run start CLEANLY on this shared single-GPU box.
# These were found the hard way (2026-06-23) chasing first-step freezes: the *config* was
# never the problem -- it was the environment, three things stacking when training shares
# the GPU/box with another ROCm job (e.g. the ONNX-export instance). See SAO/MASTER.md §5.
#
#   1. NVMe latents, not the contended removable Lehto drive    (ENCODED_DIR default below)
#   2. CK flash-attn          FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE   (30-100% faster; precede torch)
#   3. TunableOp OFF          PYTORCH_TUNABLEOP_ENABLED=0   (7.14 cache hangs on torch-2.12/RDNA4;
#                             with CK FA, GEMM tuning is marginal anyway)
#   4. MIOpen FAST            MIOPEN_FIND_MODE=2            (no exhaustive search -> no find-mode freeze)
#   5. Thread caps            OMP/MKL/OPENBLAS/NUMEXPR=4    (no 24-thread x N-process oversubscription)
#
# Plus: the FIRST step does a one-time kernel compile (several minutes, CPU+GPU both busy, NO log
# line). This is NORMAL -- do NOT kill it; give it ~10 min of a reasonably quiet box.
#
# Usage:
#   ./run_control_train.sh <run-name> [lr] [optimizer] [extra train.py args...]
# Examples:
#   ./run_control_train.sh onset_AdamW_lr7.5e-5_continuous 7.5e-5 adamw
#   ./run_control_train.sh onset_Fusion_lr1e-4 1e-4 fusion --random-crop
# Overridable via env: SA3_VENV, ENCODED_DIR, SAVE_ROOT, STEPS, SAVE_EVERY, SEED, NUM_WORKERS.
set -eo pipefail

VENV=${SA3_VENV:-/home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python}
ENCODED_DIR=${ENCODED_DIR:-/home/kim/Projects/latents_sa3}    # NVMe mirror; Lehto copy is slow/contended
SAVE_ROOT=${SAVE_ROOT:-/run/media/kim/Mantu/sa3_control_runs}
STEPS=${STEPS:-54000}; SAVE_EVERY=${SAVE_EVERY:-5400}; SEED=${SEED:-42}; NUM_WORKERS=${NUM_WORKERS:-4}; BATCH=${BATCH:-1}

RUN_NAME=${1:?usage: run_control_train.sh <run-name> [lr] [optimizer] [extra train.py args...]}
LR=${2:-7.5e-5}
OPT=${3:-adamw}
shift 3 2>/dev/null || shift $#        # consumed positionals; rest passes through to train.py
EXTRA=("$@")

SAVE_DIR="$SAVE_ROOT/$RUN_NAME"
mkdir -p "$SAVE_DIR"
cd "$(dirname "$(readlink -f "$0")")"  # -> avp_sa3 (so sa3_control/train.py resolves)

# --- the standing config (must precede torch import; setsid inherits the exported env) ---
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export MIOPEN_FIND_MODE=2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

echo "[run] $RUN_NAME  opt=$OPT lr=$LR  encoded=$ENCODED_DIR"
echo "[run] -> $SAVE_DIR/train.log"
echo "[run] NOTE: the FIRST step compiles kernels (~minutes, CPU+GPU busy, no log line). Do NOT kill it early."

setsid nohup "$VENV" sa3_control/train.py \
  --encoded_dir "$ENCODED_DIR" --model medium-base --precision bf16 \
  --crop-frames 512 --batch "$BATCH" --lr "$LR" --steps "$STEPS" --optimizer "$OPT" \
  --wandb --wandb-project sa3-riffer --run-name "$RUN_NAME" \
  --no-checkpoint --warmup-steps 300 --no-preencode-text \
  --control-mode scalar --scalar-field onset_density --control-dim 768 --n-tokens 256 \
  --cfg-dropout 0.1 --save-dir "$SAVE_DIR" --save-every "$SAVE_EVERY" \
  --num-workers "$NUM_WORKERS" --seed "$SEED" \
  "${EXTRA[@]}" \
  > "$SAVE_DIR/train.log" 2>&1 < /dev/null &

echo "[run] launched PID $! (detached).  tail -f $SAVE_DIR/train.log"
