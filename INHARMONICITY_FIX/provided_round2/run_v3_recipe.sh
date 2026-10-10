#!/usr/bin/env bash
# Reference-recipe launcher for the v3 trainer (train_realistic_bass_v3.py). Decision D9: a NEW launcher; run_v3.sh
# is left as it is (it calls the v1-stack trainer) and gets a "superseded" header only.
#
# Why it exists: the v3 trainer's parser DEFAULTS are far from the recipe (no ladders, no OT coupling, no EMA, no
# cond_noise, MSE flow loss, constant LR, batch 32, LRs about 3x lower). A command typed from memory silently runs
# that default recipe. Here every recipe flag is baked in and only the experimental switches are environment variables.
#
# Required (no defaults on purpose; decision D11):  OUT_DIR  MAX_STEPS  HOURS
# Recipe switches:  COND_NOISE=1|0 (default 1)  FLOW_LOSS=mae|mse (default mae)  INHARM_WEIGHT=0 (default 0)
#                   INHARM_CONFIG=<json>  (required when INHARM_WEIGHT > 0)
# Optional:         MANIFOLD  SEED (passed as --seed; the trainer rejects it until --seed exists)  PURPOSE
#                   EMBED_DIM FLOW_DIM FF_DIM (512 512 1152: every reference run)  LR_JEPA LR_FLOW (2.7e-4 1.8e-4)
#                   BATCH_SIZE NUM_WORKERS. The v3 scale-up of run_v3.sh is UNTESTED and 2.08x larger: to use it set
#                   EMBED_DIM=768 FLOW_DIM=768 FF_DIM=2048 LR_JEPA=3.0e-4 LR_FLOW=2.0e-4 explicitly.
#                   DEVICE PYTHON (default python3)  SAO (enables the GPU lock)  LOCK_HANDLE  DRY_RUN=1
# Example (reference recipe, aux off):   OUT_DIR=... MAX_STEPS=78416 HOURS=30 DRY_RUN=1 ./run_v3_recipe.sh
# To start a TWIN of a live run use twin_command.py instead: it copies the live run's own flags.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "run_v3_recipe: $*" >&2; exit 2; }

OUT_DIR="${OUT_DIR:-}";     [ -n "$OUT_DIR" ]   || die "set OUT_DIR (no default: the trainer's own default is a hardcoded drive path)"
MAX_STEPS="${MAX_STEPS:-}"; [ -n "$MAX_STEPS" ] || die "set MAX_STEPS (the modular optimizer's LR schedule is step-based; decision D11)"
HOURS="${HOURS:-}";         [ -n "$HOURS" ]     || die "set HOURS (wall-clock cap; decision D11)"
COND_NOISE="${COND_NOISE:-1}"
FLOW_LOSS="${FLOW_LOSS:-mae}"
INHARM_WEIGHT="${INHARM_WEIGHT:-0}"
EMBED_DIM="${EMBED_DIM:-512}"; FLOW_DIM="${FLOW_DIM:-512}"; FF_DIM="${FF_DIM:-1152}"
LR_JEPA="${LR_JEPA:-2.7e-4}";  LR_FLOW="${LR_FLOW:-1.8e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"; NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python3}"
case "$COND_NOISE" in 0|1) ;; *) die "COND_NOISE must be 0 or 1";; esac
case "$FLOW_LOSS" in mae|mse) ;; *) die "FLOW_LOSS must be mae or mse";; esac
awk -v w="$INHARM_WEIGHT" 'BEGIN{ exit !(w ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) }' || die "INHARM_WEIGHT must be a non-negative number"
USE_AUX=0; awk -v w="$INHARM_WEIGHT" 'BEGIN{ exit !(w+0 > 0) }' && USE_AUX=1
if [ "$USE_AUX" = 1 ]; then
  [ -n "${INHARM_CONFIG:-}" ] || die "INHARM_WEIGHT > 0 needs INHARM_CONFIG=<json>"
  { [ -f "$INHARM_CONFIG" ] || [ "${DRY_RUN:-0}" = 1 ]; } || die "INHARM_CONFIG not found: $INHARM_CONFIG"
fi
PURPOSE="${PURPOSE:-reference recipe (H1+H6+MAE): cond_noise=$COND_NOISE flow_loss=$FLOW_LOSS inharm_weight=$INHARM_WEIGHT}"

CMD=( "$PYTHON" "$HERE/train_realistic_bass_v3.py"
  --out_dir "$OUT_DIR" --hours "$HOURS" --max_steps "$MAX_STEPS" --batch_size "$BATCH_SIZE"
  --embed_dim "$EMBED_DIM" --flow_dim "$FLOW_DIM" --ff_dim "$FF_DIM"
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05
  --num_workers "$NUM_WORKERS" --prefetch_factor 4
  --lr_jepa "$LR_JEPA" --lr_flow "$LR_FLOW" --lr_floor 0.333 --decay_frac 0.3
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8
  --ot_coupling --ema_halflife_steps 2000
  --flow_loss "$FLOW_LOSS" --device "$DEVICE" )
[ "$COND_NOISE" = 1 ]       && CMD+=( --cond_noise --cond_noise_clean_frac 0.3 )
[ -n "${MANIFOLD:-}" ]      && CMD+=( --manifold "$MANIFOLD" )
[ -n "${SEED:-}" ]          && CMD+=( --seed "$SEED" )
[ "$USE_AUX" = 1 ]          && CMD+=( --inharm_weight "$INHARM_WEIGHT" --inharm_config "$INHARM_CONFIG" )
CMD+=( --purpose "$PURPOSE" )

print_cmd() { printf '%q' "${CMD[0]}"; local t; for t in "${CMD[@]:1}"; do case "$t" in --*) printf ' \\\n  %q' "$t";; *) printf ' %q' "$t";; esac; done; printf '\n'; }

if [ "${DRY_RUN:-0}" = 1 ]; then echo "# DRY RUN: nothing is started, nothing is written"; print_cmd; exit 0; fi

if [ -f "$OUT_DIR/checkpoint_latest.pt" ]; then
  echo "run_v3_recipe: $OUT_DIR already holds checkpoint_latest.pt: this RESUMES in place. Keep every flag identical;"
  echo "               the trainer refuses a change of the aux head on/off state (decision D8)." >&2
fi
mkdir -p "$OUT_DIR" || die "cannot create $OUT_DIR"
{ echo "# $(date '+%Y-%m-%d %H:%M:%S') launched by run_v3_recipe.sh"; print_cmd; echo; } >> "$OUT_DIR/launch_command.txt"

export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
GUARD=""; [ -n "${SAO:-}" ] && [ -x "$SAO/Misc/gpu_guard.sh" ] && GUARD="$SAO/Misc/gpu_guard.sh"
HANDLE="${LOCK_HANDLE:-ANTIGRAVITY}"
if [ -n "$GUARD" ]; then
  KIND=batch NOTE="$PURPOSE" "$GUARD" acquire "$HANDLE" $$ || { echo "GPU busy - not starting"; exit 9; }
else
  echo "run_v3_recipe: SAO not set or gpu_guard.sh missing: NOT taking the GPU lock; make sure the card is free" >&2
fi
"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- "${CMD[@]}"
rc=$?
[ -n "$GUARD" ] && "$GUARD" release "$HANDLE"
exit $rc
