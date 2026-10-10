#!/usr/bin/env bash
set -uo pipefail
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0
export ROCR_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
SAO=/home/kim/Projects/SAO
MANTU_MOUNT=""
for candidate in /run/media/kim/Mantu /run/media/kim/Mantu1 /run/media/kim/Mantu2 /run/media/kim/Mantu3; do
  if [ -d "$candidate/surge_200k_models" ] && touch "$candidate/.test_write" 2>/dev/null; then
    rm -f "$candidate/.test_write"
    MANTU_MOUNT="$candidate"
    break
  fi
done
ROOT="$MANTU_MOUNT/surge_200k_models"
OUT_DIR="$ROOT/smoke_test_5k"
PYTHON=$SAO/stable-audio-tools/sat-venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

KIND=batch NOTE="synth-inversion smoke test" \
  "$SAO/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$ || { echo "GPU busy"; exit 9; }

"$PYTHON" "$HERE/watchdog_supervisor.py" --out_dir "$OUT_DIR" -- \
  "$PYTHON" "$HERE/train_realistic_bass_overnight.py" \
  --out_dir "$OUT_DIR" --manifold "$ROOT/real_bass_manifold.npz" \
  --max_steps 0 --hours 14.0 --batch_size 64 \
  --embed_dim 768 --flow_dim 768 --ff_dim 2048 \
  --optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 \
  --sf_c_warmup 1000 --sf_r 1.0 --wd_jepa 0.02 --wd_overtraining --random_patch_frac 0.05 \
  --num_workers 8 --prefetch_factor 4 \
  --lr_jepa 3.0e-4 --lr_flow 2.0e-4 --lr_floor 0.333 --decay_frac 0.3 \
  --ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8 \
  --ot_coupling --ema_halflife_steps 2000 --cond_noise --cond_noise_clean_frac 0.3 \
  --purpose "Smoke test with larger 768 dim and 3-note legato/break phrases" \
  --device cuda:0
rc=$?
"$SAO/Misc/gpu_guard.sh" release ANTIGRAVITY
exit $rc
