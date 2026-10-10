#!/usr/bin/env bash
# Continue H6 (cond-noise recipe) from step 33230 for ~6 h (2026-10-09, Kim).
# Resumes from /run/media/kim/Mantu/surge_200k_models/h6_cont4h_b64/checkpoint_latest.pt
# Outputs to /run/media/kim/Mantu/surge_200k_models/h6_cont6h_b64
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
# Dynamic Mantu volume discovery
MANTU_MOUNT=""
for candidate in /run/media/kim/Mantu /run/media/kim/Mantu1 /run/media/kim/Mantu2 /run/media/kim/Mantu3; do
  if [ -d "$candidate/surge_200k_models" ] && touch "$candidate/.test_write" 2>/dev/null; then
    rm -f "$candidate/.test_write"
    MANTU_MOUNT="$candidate"
    break
  fi
done
if [ -z "$MANTU_MOUNT" ]; then
  udisksctl mount -b /dev/disk/by-label/Mantu 2>/dev/null || true
  for candidate in /run/media/kim/Mantu /run/media/kim/Mantu1 /run/media/kim/Mantu2 /run/media/kim/Mantu3; do
    if [ -d "$candidate/surge_200k_models" ] && touch "$candidate/.test_write" 2>/dev/null; then
      rm -f "$candidate/.test_write"
      MANTU_MOUNT="$candidate"
      break
    fi
  done
fi

if [ -z "$MANTU_MOUNT" ]; then
  echo "Error: Could not locate active Mantu volume" >&2
  exit 1
fi

ROOT="$MANTU_MOUNT/surge_200k_models"
SRC="$ROOT/h6_cont4h_b64"
OUT_DIR="$ROOT/h6_cont6h_b64"
HOURS="${HOURS:-6.0}"
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/checkpoint_latest.pt" ]; then
  RESUME_ARGS=()
  echo "=== $(date) launcher pid $$ -> $OUT_DIR (resuming in-place checkpoint_latest.pt)"
else
  RESUME_ARGS=(--resume_from "$SRC/checkpoint_latest.pt")
  echo "=== $(date) launcher pid $$ -> $OUT_DIR (initial resume $SRC/checkpoint_latest.pt)"
fi

# Yield resident server if holding GPU
pids=$(rocm-smi --showpids 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $4+0 > 500000000 { print $1 }')
for p in $pids; do
  cmd=$(ps -o args= -p "$p" 2>/dev/null)
  if [[ "$cmd" == *"explorer_render_server.py"* ]]; then
    echo "Yielding explorer_render_server (pid $p)..."
    kill "$p" || true
    sleep 2
    python3 "$SAO/Misc/filelock.py" release "$SAO/.gpu.lock" --handle WINTERMUTE 2>/dev/null || true
  fi
done

KIND=batch NOTE="synth-inversion H6 continuation, ${HOURS} h" \
  "$SAO/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$ || { echo "GPU busy - not starting"; exit 9; }

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --manifold "$ROOT/real_bass_manifold.npz" "${RESUME_ARGS[@]}" \
  --hours "$HOURS" --max_steps 0 --batch_size 64 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 --ff_dim 1152 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 \
  --purpose "H6 continuation: resume from step 33230 for ${HOURS} h with time-based end decay. Same recipe/manifold as H6." \
  --device cuda:0
rc=$?
echo "=== $(date) training exited rc=$rc"
"$SAO/Misc/gpu_guard.sh" release ANTIGRAVITY
