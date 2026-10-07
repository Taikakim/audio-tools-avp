#!/usr/bin/env bash
# EXPERIMENTS H6 (2026-10-07): H1 (run_fxfix_ladder_ema.sh) recipe with ONE change, --cond_noise: the flow
# trains on partly noised audio conditions (30% clean, otherwise tau ~ logit-normal(0,1)), tau is a model input.
# The Synth-JDF ablation that paper did not run: is the off-manifold gain from the joint objective, or just from
# training on noisy conditions? Same steps (78,416), seed handling, data and LRs as H1, so H1 is the control.
# After training: h6_post.sh (CPU) inverts the 24 real stems with the H6 online/EMA weights at several tau.
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/h6_condnoise_b64}"
EVAL_ROOT=/run/media/kim/Mantu/surge_200k_models
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
echo "=== $(date) launcher pid $$ -> $OUT_DIR"

KIND=batch NOTE="synth-inversion H6 cond-noise training (~24 h), then CPU clip renders" \
  "$SAO/Misc/gpu_guard.sh" acquire WINTERMUTE $$ || { echo "GPU busy - not starting"; exit 9; }

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --hours 30.0 --max_steps 78416 --batch_size 64 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 \
  --purpose "EXPERIMENTS H6: H1 recipe unchanged except the flow trains on partly noised audio conditions (30% clean, else tau ~ logit-normal), tau an input. Question: does noisy-conditioning training alone close the real-stem gap Synth-JDF attributes to the joint objective? Kill: real-stem MSS not better than the old Oct-5 model (7.31) at any tau." \
  --device cuda:0
rc=$?
echo "=== $(date) training exited rc=$rc"
"$SAO/Misc/gpu_guard.sh" release WINTERMUTE
# Post-step (CPU clips, tau sweep, scoring, page) lives in its own file so it can be finished while this runs.
[ -x "$HERE/h6_post.sh" ] && OUT_DIR="$OUT_DIR" "$HERE/h6_post.sh"
echo "=== $(date) all done"
