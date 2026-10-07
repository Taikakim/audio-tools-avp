#!/usr/bin/env bash
# v3 clean-retrain post-step, called by run_v3_clean.sh after training (GPU lock already released).
# 1. If the 30 h wall-clock cap stopped training short of --max_steps, re-take the GPU lock and resume to the full
#    step count (step-based LR schedule, so the recipe is unchanged).
# 2. CPU: invert the 24 real stems with v3 online / EMA / SF average (clean condition) and EMA at tau 0.4 / 0.1, plus
#    the old Oct-5 model and H1 EMA re-rendered with FX truly off (their published clips carry the Surge delay).
# 3. Score them together, build + upload synth_inversion_v3.html.
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/v3_clean_b64}"
OLD=/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64
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
  until KIND=batch NOTE="synth-inversion v3 resume to $MAX_STEPS steps" "$SAO/Misc/gpu_guard.sh" acquire WINTERMUTE $$; do
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
    --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 --flow_loss mae \
    --resume_from "$OUT_DIR/checkpoint_latest.pt" \
    --purpose "v3 clean retrain resume after the 30 h time cap (GPU power-capped at 225 W)" \
    --device cuda:0
  echo "=== $(date) resume exited rc=$?"
  "$SAO/Misc/gpu_guard.sh" release WINTERMUTE
done

# --- 2. CPU clips ---
inv() {  # name ckpt tau  (skips a set that already has all 24 playbacks, e.g. the references rendered early)
  if [ "$(ls "$OUT_DIR/clips_$1/audio/"*_midi_playback.wav 2>/dev/null | wc -l)" -ge 24 ]; then
    echo "=== $(date) $1 already complete, skipping"; return 0
  fi
  echo "=== $(date) inverting 24 stems: $1 ($2, tau $3)"
  (cd "$HERE" && nice -n 10 "$PYTHON" invert_stem_collection.py --flow_ckpt "$2" --out_dir "$OUT_DIR/clips_$1" \
     --cond_tau "$3" --playback_midi muscriptor --no_user_copy)
  echo "=== $(date) $1 rc=$?"
}
# references re-rendered with FX truly off, so the comparison is fair (their old clips carry the delay)
inv ref_old    "$OLD/flow_latest.pt"              1.0
inv ref_h1_ema "$H1/flow_ema_latest.pt"           1.0
inv v3_online  "$OUT_DIR/flow_online_latest.pt"   1.0
inv v3_ema     "$OUT_DIR/flow_ema_latest.pt"      1.0
inv v3_sf      "$OUT_DIR/flow_latest.pt"          1.0
inv v3_ema_t04 "$OUT_DIR/flow_ema_latest.pt"      0.4
inv v3_ema_t01 "$OUT_DIR/flow_ema_latest.pt"      0.1

# --- 3. score next to the old model and H1, rebuild + upload the page ---
(cd "$HERE" && nice -n 10 "$PYTHON" compare_clip_sets.py "$OUT_DIR/clips_AB" ref_old="$OUT_DIR/clips_ref_old" \
   ref_h1_ema="$OUT_DIR/clips_ref_h1_ema" v3_online="$OUT_DIR/clips_v3_online" v3_ema="$OUT_DIR/clips_v3_ema" \
   v3_sf="$OUT_DIR/clips_v3_sf" v3_ema_t04="$OUT_DIR/clips_v3_ema_t04" v3_ema_t01="$OUT_DIR/clips_v3_ema_t01")
echo "=== $(date) compare rc=$?"
(cd "$SAO" && SYNTH_PAGE=v3 python3 Misc/build_synth_inversion_page.py)
eval "$(grep -E '^(HOST|KEY|DEST)=' /home/kim/bin/transcode_evals.sh)"
(cd "$HOME/.cache/evals_aac" && rsync -az -e "ssh -i $KEY -o BatchMode=yes" synth_inversion_v3.html synth_inversion_v3 "$HOST:$DEST/" >/dev/null 2>&1)
echo "=== $(date) page pushed rc=$?"
