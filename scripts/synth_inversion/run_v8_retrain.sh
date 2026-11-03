#!/usr/bin/env bash
# Owner: WINTERMUTE (created 2026-11-03)
# run_v8_retrain.sh <stage> -- Synth Inversion V8 retrain after KUANG's review (docs/synth-v8-code-review-2026-10-29.md). One command per stage, each one ends in PASS or STOP.
# Handoff with the exact order: SAO docs/v8-retrain-handoff.md. Order: check, render-check, (a person listens), data, phase2, phase1, board, status.
# Nothing here publishes. SMOKE=1 runs every stage tiny into a throwaway directory, to test this script itself.
set -u
SAO=/home/kim/Projects/SAO
HERE=$SAO/stable-audio-tools/scripts/synth_inversion
PY=$SAO/.venv-rocm10-amd/bin/python
GUARD=$SAO/Misc/gpu_guard.sh
export ROCR_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF        # expandable_segments writes wrong audio on ROCm, MASTER section 5
RUNS=/home/kim/sao_runs
if [ "${SMOKE:-0}" = "1" ]; then
  ROOT=${SMOKE_DIR:-/tmp/v8_retrain_smoke}; N_TRAIN=24; N_VAL=8; N_TEST=8; P2_EPOCHS=1; P1_EPOCHS=1; P1_LEN=32; ITEMS=2; DRAWS=1
  MIN_FREE_GB=5
else
  ROOT=$RUNS/v8_retrain_2026-11; N_TRAIN=5000; N_VAL=200; N_TEST=300; P2_EPOCHS=15; P1_EPOCHS=20; P1_LEN=10000; ITEMS=40; DRAWS=4
  MIN_FREE_GB=12
fi
STEMS=$RUNS/v8_stem_cache
DYN=$ROOT/dynamic_cache
P2=$ROOT/phase2_inverter
P1=$ROOT/phase1_tse
J2=$ROOT/phase2_results.json
J1=$ROOT/phase1_results.json
BOARD=${BOARD_DIR:-$ROOT/board/synth_inversion_v8}
LOG=$ROOT/logs
LISTEN_OK=$ROOT/RENDER_CHECK_APPROVED
mkdir -p "$LOG"
cd "$HERE" || exit 1

stop() { echo; echo "STOP [$STAGE]: $*"; echo "Do not change code or settings. Report this message and the last 30 lines of $LOG/$STAGE.log."; exit 1; }
pass() { echo; echo "PASS [$STAGE]: $*"; echo "$(date '+%F %T') $STAGE PASS $*" >> "$ROOT/progress.txt"; exit 0; }
need() { [ -s "$1" ] || stop "expected file is missing or empty: $1"; }
run() { "$@" >> "$LOG/$STAGE.log" 2>&1; }
gpu_wait() { until KIND=batch NOTE="V8 retrain, $STAGE" "$GUARD" acquire ANTIGRAVITY $$ >/dev/null 2>&1; do echo "waiting for the GPU lock, $(date +%H:%M)"; sleep 120; done; }
gpu_free() { "$GUARD" release ANTIGRAVITY >/dev/null 2>&1; }
json_get() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));d=d.get('summary',d);print(d.get(sys.argv[2]))" "$1" "$2"; }

meta() {  # the sidecar every run must carry
  "$PY" - "$ROOT/run_meta.json" "$N_TRAIN" "$P2_EPOCHS" "$P1_EPOCHS" "$P1_LEN" <<'E'
import json,sys,subprocess,time
p,nt,e2,e1,l1=sys.argv[1:]
c=subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,cwd="/home/kim/Projects/SAO/stable-audio-tools").stdout.strip()
json.dump(dict(purpose="Synth Inversion V8 retrain on the fixed renderer, targets and cues.",
 hypothesis="With a whole-note renderer, true targets, cues that do not contain the answer, and 10x the data, do Phase 1 and Phase 2 pass their gates honestly?",
 spec="SAO docs/superpowers/specs/2026-10-28-synth-inversion-v8-dynamic-tse-spec.md",review="SAO docs/synth-v8-code-review-2026-10-29.md",handoff="SAO docs/v8-retrain-handoff.md",
 script="stable-audio-tools/scripts/synth_inversion/run_v8_retrain.sh",commit=c,created=time.strftime("%F %T"),
 recipe=dict(phase2=dict(train_renders=int(nt),epochs=int(e2)),phase1=dict(epochs=int(e1),epoch_length=int(l1))),
 dataset="bank_v6 patches through the fixed renderer; stems from the v8 stem cache, six songs held out",
 not_in_this_run=["R1 cleanup, R2 rerun, R3 as registered","BS-RoFormer and two-stem baselines","registered Phase 2 gate (centroid trajectory on re-rendered audio)","extractor resolution and reliability target (M4, M5)","groove-aligned mixtures (M3)"]),open(p,"w"),indent=1)
E
}

STAGE=${1:-}
case "$STAGE" in
check)
  : > "$LOG/check.log"
  [ -x "$PY" ] || stop "venv python not found: $PY"
  free=$(df --output=avail -BG "$RUNS" | tail -1 | tr -dc 0-9); [ "$free" -ge "$MIN_FREE_GB" ] || stop "only ${free} GB free on $RUNS, need $MIN_FREE_GB"
  for f in "$RUNS/v6_C/flow_best.pt" "$RUNS/real_manifold_v4_bass.npz" "$RUNS/real_manifold_v4_all.npz" "$RUNS/bank_v6" "$STEMS/synths_stem_ids.npy" "$STEMS/val_synths_stem_ids.npy"; do
    [ -e "$f" ] || stop "required input is missing: $f"; done
  (cd /tmp && run "$PY" -c "import torch,pedalboard,demucs,soundfile,skimage;print('imports ok, torch',torch.__version__)") || stop "a python import failed"
  (cd /tmp && run "$PY" -m pytest -q "$HERE/test_v8_fixes.py") || stop "the V8 fix tests failed"
  gpu=$("$GUARD" who ANTIGRAVITY 2>/dev/null | sed -n '/announced/,$p' | head -3 | tr '\n' ' ')
  pass "tests pass, inputs present, ${free} GB free. GPU now: $gpu" ;;
render-check)
  : > "$LOG/render-check.log"
  run "$PY" -c "
import numpy as np, soundfile as sf, surge_spec_v5 as S5, surge_spec_v6 as S6, dynamic_dataset_v8 as dd
p = S5.init_synth()
lanes = np.stack([np.linspace(.1, .9, 43)] + [np.full(43, .3)] * 3).astype(np.float32)
sf.write('$ROOT/v8_render_check.wav', dd.render_dynamic_patch(p, S6.vector_to_patch(np.full(59, .5, np.float32)), 40, *lanes), 44100)
" || stop "the render failed"
  need "$ROOT/v8_render_check.wav"
  run "$PY" -c "
import numpy as np, soundfile as sf
a,sr=sf.read('$ROOT/v8_render_check.wav'); a=a.mean(1) if a.ndim>1 else a
n=int(sr*4096/44100); c=[float(np.sqrt(np.mean(a[i:i+n]**2))) for i in range(0,len(a)-n,n)]
silent=sum(x<1e-4 for x in c); print('seconds',round(len(a)/sr,2),'chunks',len(c),'silent',silent,'peak',round(float(np.abs(a).max()),3))
raise SystemExit(0 if len(a)/sr>3.5 and silent<=len(c)//4 and np.isfinite(a).all() else 3)
" || stop "the render is too short, non-finite, or mostly silent chunks; see the log"
  tail -1 "$LOG/render-check.log"
  pass "written $ROOT/v8_render_check.wav. A PERSON must listen: notes should sound whole and the filter should open over 4 s. If it sounds right, that person runs: touch $LISTEN_OK" ;;
data)
  if [ "${SMOKE:-0}" != "1" ]; then [ -f "$LISTEN_OK" ] || stop "the render check has not been approved by a person. Do not create $LISTEN_OK yourself. Ask for the listen."; fi
  : > "$LOG/data.log"; meta
  run "$PY" dynamic_dataset_v8.py --out-dir "$DYN" --num-train "$N_TRAIN" --num-val "$N_VAL" --num-test "$N_TEST" || stop "building the dynamic cache failed"
  for s in train val test; do need "$DYN/dynamic_$s.npz"; done
  pass "dynamic cache built: $N_TRAIN train, $N_VAL val, $N_TEST test" ;;
phase2)
  for s in train val test; do need "$DYN/dynamic_$s.npz"; done
  : > "$LOG/phase2.log"
  gpu_wait; trap gpu_free EXIT
  run "$PY" train_v8_dynamic_inverter.py --cache-dir "$DYN" --output-dir "$P2" --epochs "$P2_EPOCHS" --wait-gpu || stop "Phase 2 training failed"
  need "$P2/dynamic_inverter_best.pt"
  gpu_wait
  run "$PY" eval_v8_phase2_dynamic.py --ckpt "$P2/dynamic_inverter_best.pt" --cache-dir "$DYN" --out "$J2" || stop "Phase 2 evaluation failed"
  need "$J2"
  g=$(json_get "$J2" gate_pass); r=$(json_get "$J2" cutoff_rho_mean)
  [ "$g" = "True" ] || stop "Phase 2 gate failed: cutoff rho $r, needs 0.80. This is a result, not a bug. Do not retrain. Results: $J2"
  pass "Phase 2 gate passed: cutoff rho $r. Results: $J2" ;;
phase1)
  need "$J2"
  [ "${SMOKE:-0}" = "1" ] || [ "$(json_get "$J2" gate_pass)" = "True" ] || stop "Phase 2 has not passed, so Phase 1 does not run"
  : > "$LOG/phase1.log"
  gpu_wait; trap gpu_free EXIT
  run "$PY" train_v8_asymmetric_tse.py --output-dir "$P1" --epochs "$P1_EPOCHS" --epoch-length "$P1_LEN" --wait-gpu || stop "Phase 1 training failed"
  need "$P1/tse_best.pt"
  gpu_wait
  run "$PY" eval_v8_phase1_tse.py --tse-ckpt "$P1/tse_best.pt" --bass-manifold "$RUNS/real_manifold_v4_bass.npz" --all-manifold "$RUNS/real_manifold_v4_all.npz" --n-items "$ITEMS" --draws "$DRAWS" --out "$J1" || stop "Phase 1 evaluation failed"
  need "$J1"
  g=$(json_get "$J1" gate_pass); r=$(json_get "$J1" tax_ratio_mse)
  [ "$g" = "True" ] || stop "Phase 1 gate failed: tax $r, needs 1.25 or less. This is a result, not a bug. Do not retrain. Results: $J1"
  pass "Phase 1 gate passed: tax $r. Results: $J1" ;;
board)
  need "$J1"; need "$J2"
  [ "${SMOKE:-0}" = "1" ] || { [ "$(json_get "$J1" gate_pass)" = "True" ] && [ "$(json_get "$J2" gate_pass)" = "True" ]; } || stop "both gates must pass before a board is built"
  : > "$LOG/board.log"
  gpu_wait; trap gpu_free EXIT
  run "$PY" build_v8_eval_board.py --tse-ckpt "$P1/tse_best.pt" --inv-ckpt "$P2/dynamic_inverter_best.pt" --phase1-json "$J1" --phase2-json "$J2" --base-dir "$BOARD" || stop "building the board failed"
  need "$BOARD/index.html"
  pass "board staged at $BOARD. Do not publish it. WINTERMUTE reviews and publishes." ;;
status)
  echo "run dir: $ROOT"; cat "$ROOT/progress.txt" 2>/dev/null || echo "no stage has passed yet"
  [ -f "$LISTEN_OK" ] && echo "render check approved" || echo "render check not approved"
  for j in "$J2" "$J1"; do [ -s "$j" ] && echo "$j gate_pass=$(json_get "$j" gate_pass)"; done ;;
*)
  echo "usage: $0 check|render-check|data|phase2|phase1|board|status"; exit 2 ;;
esac
