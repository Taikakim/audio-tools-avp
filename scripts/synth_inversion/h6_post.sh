#!/usr/bin/env bash
# EXPERIMENTS H6 post-step, called by run_h6_condnoise.sh after training (GPU lock already released).
# 1. If the 30 h wall-clock cap stopped training short of --max_steps (the GPU is power-capped at 225 W this
#    week, ~20% slower than H1), re-take the GPU lock and resume to the full step count. The modular optimizer's
#    LR decay is step-based (frac = step / max_steps), so the resume does not change the schedule vs H1.
# 2. CPU: invert the 24 real stems with H6 online and EMA at tau=1 (clean, comparable to H1) and EMA at
#    tau 0.7 / 0.4 / 0.1 (Synth-JDF's noisy-reference conditioning).
# 3. Score all of them next to the old Oct-5 model and H1's EMA (same pipeline), then rebuild + upload the
#    synth inversion eval page.
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/h6_condnoise_b64}"
H1=/run/media/kim/Mantu/surge_200k_models/fxfix_ladder_ema_b64
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STEPS=78416

last_step() { grep -oE "Stopped at step [0-9]+" "$OUT_DIR/training.log" | tail -1 | grep -oE "[0-9]+$"; }

# --- 1. finish the step budget if the time cap cut it short ---
for attempt in 1 2; do
  s=$(last_step); s=${s:-0}
  [ "$s" -ge "$MAX_STEPS" ] && break
  echo "=== $(date) training stopped at step $s < $MAX_STEPS (time cap); resuming (attempt $attempt)"
  until KIND=batch NOTE="synth-inversion H6 resume to $MAX_STEPS steps" "$SAO/Misc/gpu_guard.sh" acquire WINTERMUTE $$; do
    echo "GPU busy; waiting 10 min"; sleep 600
  done
  "$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
    "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
    --out_dir "$OUT_DIR" --hours 30.0 --max_steps $MAX_STEPS --batch_size 64 \
    --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
    --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
    --num_workers 8 --prefetch_factor 4 \
    --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
    --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
    --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 \
    --resume_from "$OUT_DIR/checkpoint_latest.pt" \
    --purpose "EXPERIMENTS H6 resume after the 30 h time cap (GPU power-capped at 225 W)" \
    --device cuda:0
  echo "=== $(date) resume exited rc=$?"
  "$SAO/Misc/gpu_guard.sh" release WINTERMUTE
done

# --- 2. CPU clips ---
inv() {  # name ckpt tau
  echo "=== $(date) inverting 24 stems: $1 ($2, tau $3)"
  (cd "$HERE" && nice -n 10 "$PYTHON" invert_stem_collection.py --flow_ckpt "$2" --out_dir "$OUT_DIR/clips_$1" \
     --cond_tau "$3" --playback_midi muscriptor --no_user_copy)
  echo "=== $(date) $1 rc=$?"
}
inv h6_online "$OUT_DIR/flow_online_latest.pt" 1.0
inv h6_ema    "$OUT_DIR/flow_ema_latest.pt"    1.0
inv h6_ema_t07 "$OUT_DIR/flow_ema_latest.pt"   0.7
inv h6_ema_t04 "$OUT_DIR/flow_ema_latest.pt"   0.4
inv h6_ema_t01 "$OUT_DIR/flow_ema_latest.pt"   0.1

# --- 3. score next to the old model and H1, rebuild + upload the page ---
(cd "$HERE" && nice -n 10 "$PYTHON" compare_clip_sets.py "$OUT_DIR/clips_AB" old="$H1/clips_old" h1_ema="$H1/clips_ema" \
   h6_online="$OUT_DIR/clips_h6_online" h6_ema="$OUT_DIR/clips_h6_ema" h6_ema_t07="$OUT_DIR/clips_h6_ema_t07" \
   h6_ema_t04="$OUT_DIR/clips_h6_ema_t04" h6_ema_t01="$OUT_DIR/clips_h6_ema_t01")
echo "=== $(date) compare rc=$?"
(cd "$SAO" && python3 Misc/build_synth_inversion_page.py)
eval "$(grep -E '^(HOST|KEY|DEST)=' /home/kim/bin/transcode_evals.sh)"
(cd "$HOME/.cache/evals_aac" && rsync -az -e "ssh -i $KEY -o BatchMode=yes" synth_inversion.html synth_inversion "$HOST:$DEST/" >/dev/null 2>&1)
echo "=== $(date) page pushed rc=$?"
