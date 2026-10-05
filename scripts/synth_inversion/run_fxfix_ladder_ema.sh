#!/usr/bin/env bash
# Rerun of modular_shampoo_sf_b64 after the Surge FX state-leak fix (commit 'fix Surge FX state leak'):
# the previous run trained on poisoned renders (train JEPA 0.13 vs 2.8 on clean renders).
# Same recipe and step count (78,416), plus:
#   - 1.5x peak LR (JEPA 2.7e-4, flow 1.8e-4), linear decay to 1/3 over the last 30% of steps
#   - ladder batches every 4th step (bracketed minimal pairs on cutoff-low / AEG sustain / FEG amount /
#     AEG decay) with a JEPA ordinal loss; SIGReg skipped on ladders
#   - minibatch OT coupling for the flow
#   - EMA of the online weights, half-life 2000 steps; online, SF-averaged and EMA validated + exported
#   - held-out ladder validation (per-axis monotonicity of JEPA distance and flow estimates)
# Afterwards (CPU, GPU released): the 24 real stems are inverted with the OLD model, the new ONLINE and the
# new EMA weights through the same pipeline, scored and A/B'd.
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
OUT_DIR="${OUT_DIR:-/run/media/kim/Mantu/surge_200k_models/fxfix_ladder_ema_b64}"
EVAL_ROOT=/run/media/kim/Mantu/surge_200k_models
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
echo "=== $(date) launcher pid $$ -> $OUT_DIR"

KIND=batch NOTE="synth-inversion fxfix_ladder_ema_b64 training (~16 h), then CPU clip renders" \
  "$SAO/Misc/gpu_guard.sh" acquire WINTERMUTE $$ || { echo "GPU busy - not starting"; exit 9; }

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --hours 30.0 --max_steps 78416 --batch_size 64 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 \
  --purpose "FX-leak-fixed rerun of modular_shampoo_sf_b64 (same 78,416 steps), 1.5x LR with end decay, ladder batches + ordinal loss, OT coupling, EMA hl 2000; online vs EMA vs SF-x reported" \
  --device cuda:0
rc=$?
echo "=== $(date) training exited rc=$rc"
"$SAO/Misc/gpu_guard.sh" release WINTERMUTE

# --- clips: same pipeline for old model, new online, new EMA (CPU only) ---
for spec in "old=$EVAL_ROOT/modular_shampoo_sf_b64/flow_latest.pt" \
            "online=$OUT_DIR/flow_online_latest.pt" "ema=$OUT_DIR/flow_ema_latest.pt"; do
  name=${spec%%=*}; ckpt=${spec#*=}
  [ -f "$ckpt" ] || { echo "missing $ckpt"; continue; }
  echo "=== $(date) inverting 24 stems with $name ($ckpt)"
  "$PYTHON" "$HERE/invert_stem_collection.py" --flow_ckpt "$ckpt" --out_dir "$OUT_DIR/clips_$name" \
    --playback_midi muscriptor --no_user_copy
done
"$PYTHON" "$HERE/compare_clip_sets.py" "$OUT_DIR/clips_AB" \
  old="$OUT_DIR/clips_old" online="$OUT_DIR/clips_online" ema="$OUT_DIR/clips_ema"
echo "=== $(date) all done"
