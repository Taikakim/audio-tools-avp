#!/usr/bin/env bash
# Train the SA3 audio-reference (riffer) control adapter on pre-encoded latents.
#
#   bash avp_sa3/launch_riffer.sh                 # default: bf16, crop 2048, 20k steps
#   CROP=4096 STEPS=40000 bash avp_sa3/launch_riffer.sh
#
# Perf notes (this box / ROCm):
#  - bf16 base + adapters (the supported ROCm path). TunableOp OFF — negligible on 7.14.
#  - Counting on native CK (Composable Kernel) flash-attn (~2x over Triton FA2) when run
#    on the ROCm 7.14 / CK stack (SAO/docs/flash-attn-ck-rdna4.md). On the prod 7.2.3
#    stack it falls back to Triton FA2 (FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE).
#  - Do NOT set MIOPEN_FIND_MODE=6 — it crashes SA3-medium's DiT (mode 2 is fine).
#  - DiT gradient checkpointing stays ON; our module-global control-token holder survives it.
#  - Only ~4.8% of params train (the adapters + ref conditioner); the 2.3B base is frozen.
set -u
cd /home/kim/Projects/SAO/stable-audio-tools/avp_sa3 || exit 1

# STACK=714 -> ROCm 7.14 / CK flash-attn (validated, ~27% faster end-to-end; needs the
#              backward grad-count patch, docs/flash-attn-ck-rdna4.md §5b).
# STACK=723 -> prod ROCm 7.2.3 / Triton FA2 (the safe default).
STACK="${STACK:-714}"
if [ "$STACK" = "714" ]; then
  PY=/home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE     # use CK kernels, not aiter-Triton
else
  PY=/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
fi
export PYTORCH_TUNABLEOP_ENABLED=0                    # negligible on 7.14

SAVE_DIR="${SAVE_DIR:-/run/media/kim/Lehto/sa3_control_runs/riffer}"
# CHECKPOINT=0 disables DiT gradient checkpointing (faster bwd; needs the VRAM headroom).
CK_ARG=""; [ "${CHECKPOINT:-1}" = "0" ] && CK_ARG="--no-checkpoint"
WB_ARGS=""                                            # wandb on by default; WANDB=0 disables
if [ "${WANDB:-1}" != "0" ]; then
  WB_ARGS="--wandb --wandb-project ${WANDB_PROJECT:-sa3-riffer} --run-name ${RUN_NAME:-$(basename "$SAVE_DIR")-$(date +%m%d-%H%M)}"
  mkdir -p "$SAVE_DIR"                                 # write wandb run data to the save-dir (Lehto),
  export WANDB_DIR="$SAVE_DIR"                         # not under the repo where it could shadow imports
fi

"$PY" sa3_control/train.py \
  --encoded_dir /run/media/kim/Lehto/latents_sa3 \
  --model medium-base --precision bf16 \
  --crop-frames "${CROP:-2048}" --batch "${BATCH:-1}" \
  --lr "${LR:-1e-4}" --steps "${STEPS:-20000}" \
  ${SUBSET:+--subset-tracks "$SUBSET"} $WB_ARGS $CK_ARG \
  ${MAX_HOURS:+--max-hours "$MAX_HOURS"} ${WARMUP:+--warmup-steps "$WARMUP"} ${OPTIMIZER:+--optimizer "$OPTIMIZER"} \
  ${NO_PREENCODE:+--no-preencode-text} ${TIMESTEP:+--timestep-sampler "$TIMESTEP"} ${RESUME:+--resume "$RESUME"} \
  ${CONTROL_MODE:+--control-mode "$CONTROL_MODE"} ${SCALAR_FIELD:+--scalar-field "$SCALAR_FIELD"} \
  --control-dim 768 --n-tokens 256 --cfg-dropout 0.1 \
  --save-dir "$SAVE_DIR" \
  --save-every "${SAVE_EVERY:-1000}" --num-workers 4 --seed 42
