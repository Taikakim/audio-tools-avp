#!/usr/bin/env bash
# v3 clean retrain (2026-10-07). The first synth-inversion run whose renders have Surge FX truly OFF: setting a
# slot's output mix to 0 never silenced the init preset's delay, so H1/H6 and all earlier training sets carried a
# loud fixed delay (surge_spec.apply_patch, test_fx_off_single_note_has_no_delay_trail). Retrained directly with
# everything learned so far rather than as an ablation chain: H1 recipe + --cond_noise (H6) + --flow_loss mae.
# GPU: waits until the GPU lock has been FREE for 30 consecutive minutes (lets another instance that is waiting
# for the card go first), then holds it. After training: v3_post.sh (CPU clips, scoring, page).
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/v3_clean_b64}"
EVAL_ROOT=/run/media/kim/Mantu/surge_200k_models
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
echo "=== $(date) launcher pid $$ -> $OUT_DIR"

free_since=""
while true; do
  if [ -z "$(cat /tmp/gpu.lock 2>/dev/null)" ] && ! rocm-smi --showpids 2>/dev/null | grep -qE "^[0-9]+ "; then
    free_since=${free_since:-$(date +%s)}
    if [ $(( $(date +%s) - free_since )) -ge 1800 ] && \
       KIND=batch NOTE="synth-inversion v3 clean retrain (~19 h at 225 W), then CPU clip renders" \
         "$SAO/Misc/gpu_guard.sh" acquire WINTERMUTE $$; then
      break
    fi
  else
    free_since=""
  fi
  sleep 60
done
echo "=== $(date) GPU acquired after 30 min free" 

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --hours 30.0 --max_steps 78416 --batch_size 64 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 --flow_loss mae \
  --purpose "v3 clean retrain (2026-10-07): first run with Surge FX truly OFF (slot types Off; every earlier render carried a loud fixed delay), plus everything learned so far: H1 recipe (ladders + ordinal loss, OT coupling, EMA hl 2000) + noisy-conditioning flow (H6) + MAE velocity loss (Synth-JDF supp.). Same 78,416 steps. Kill: real-stem MSS not better than the old Oct-5 model re-rendered with FX off." \
  --device cuda:0
rc=$?
echo "=== $(date) training exited rc=$rc"
"$SAO/Misc/gpu_guard.sh" release WINTERMUTE
# Post-step (CPU clips, tau sweep, scoring, page) lives in its own file so it can be finished while this runs.
[ -x "$HERE/v3_post.sh" ] && OUT_DIR="$OUT_DIR" "$HERE/v3_post.sh"
echo "=== $(date) all done"
