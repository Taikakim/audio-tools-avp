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
SA3=/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python

PYTORCH_TUNABLEOP_ENABLED=0 "$SA3" sa3_control/train.py \
  --encoded_dir /run/media/kim/Lehto/latents_sa3 \
  --model medium-base --precision bf16 \
  --crop-frames "${CROP:-2048}" --batch "${BATCH:-1}" \
  --lr "${LR:-1e-4}" --steps "${STEPS:-20000}" \
  --control-dim 768 --n-tokens 256 --cfg-dropout 0.1 \
  --save-dir "${SAVE_DIR:-/run/media/kim/Lehto/sa3_control_runs/riffer}" \
  --save-every "${SAVE_EVERY:-1000}" --num-workers 4 --seed 42
