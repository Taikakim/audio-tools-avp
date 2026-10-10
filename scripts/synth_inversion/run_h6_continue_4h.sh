#!/usr/bin/env bash
# Continue H6 (cond-noise recipe) from its step-16000 checkpoint for ~4 h (2026-10-09, Kim).
# H6 died at step ~16.8k (12:43 07.10) when the Mantu volume dropped off the bus; the data is now at Mantu2.
# --resume_from resets the elapsed clock, so --hours 4 is the NEW budget and the final 30% of it
# (1.2 h) is the LR decay to --lr_floor. --max_steps 0 so the time budget, not the 78,416 cap, ends it.
# Recipe otherwise identical to run_h6_condnoise.sh (same manifold md5 148c4ccf..., so no --allow_prior_change).
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
ROOT="${ROOT:-/run/media/kim/Mantu2/surge_200k_models}"
SRC="${SRC:-$ROOT/h6_condnoise_b64}"
OUT_DIR="${OUT_DIR:-$ROOT/h6_cont4h_b64}"
HOURS="${HOURS:-4.0}"
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
echo "=== $(date) launcher pid $$ -> $OUT_DIR (resume $SRC/checkpoint_latest.pt)"

KIND=batch NOTE="synth-inversion H6 continuation, ${HOURS} h" \
  "$SAO/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$ || { echo "GPU busy - not starting"; exit 9; }

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --manifold "$ROOT/real_bass_manifold.npz" --resume_from "$SRC/checkpoint_latest.pt" \
  --hours "$HOURS" --max_steps 0 --batch_size 64 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 \
  --purpose "H6 continuation: resume the cond-noise run from step 16000 (killed by the Mantu drop) for ${HOURS} h with time-based end decay. Same recipe/manifold as H6." \
  --device cuda:0
rc=$?
echo "=== $(date) training exited rc=$rc"
"$SAO/Misc/gpu_guard.sh" release ANTIGRAVITY
