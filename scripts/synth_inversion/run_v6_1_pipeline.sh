#!/usr/bin/env bash
# Owner: KUANG (created 2026-10-20)
# V6.1 pipeline (V6_1_PLAN.md section 3, V6_REVIEW.md section 9), in order, each step a gate for the next.
# Replaces scratch/run_v6_dpo_pipeline.sh (not in the repository). Edit the paths below; run from this directory.
# Every step skips work that is already on disk (resumable), except the checks.
set -euo pipefail

PY=${PY:-python}
B=${B:-/home/kim/sao_runs/real_manifold_v4_bass.npz}
A=${A:-/home/kim/sao_runs/real_manifold_v4_all.npz}
PLUGIN=${PLUGIN:-"$HOME/.vst3/Surge XT.vst3"}
RUNS=${RUNS:-/home/kim/sao_runs}
BASE=${BASE:-$RUNS/v6_C/flow_ema_latest.pt}                 # the flow DPO starts from (V6_C 12k EMA)
S5=${S5:?set S5 to the Stage 5 flow checkpoint}
DPO5=${DPO5:-$RUNS/dpo_s5_b300/flow_dpo_latest.pt}
PAIRS=$RUNS/dpo_pairs_v6c_guided
DEV=${DEV:-cuda:0}

echo "== 0. tests"
$PY -m pytest test_v6.py test_v6_foundation.py test_v6_1.py -q

echo "== 1. Surge checks: detune / relative octaves, and which shape is the symmetric square"
$PY check_v6_spec.py --plugin "$PLUGIN" --shape_scan | tee "$RUNS/check_v6_spec.txt"
SQUARE=$(grep -o "use --square_shape [0-9.]*" "$RUNS/check_v6_spec.txt" | awk '{print $3}')
echo "square_shape = $SQUARE"

echo "== 2. guided preference pairs from the V6_C flow"
$PY render_pairs_dpo_v6.py --flow "$BASE" --bass_manifold "$B" --all_manifold "$A" --plugin "$PLUGIN" \
    --n 8192 --k_passive 4 --workers 14 --device "$DEV" --square_shape "$SQUARE" --out "$PAIRS"
# Gate: manifest.json headroom (best of 5 clearly below candidate 0) and guided_best_share. If the guided candidate
# almost never wins, the heuristics add nothing; if it wins almost always, check the pairs before training on them.

echo "== 3. Flow-DPO, beta 100 and 300"
for BETA in 100 300; do
  OUT=$RUNS/dpo_v6c_b$BETA
  [ -f "$OUT/flow_dpo_latest.pt" ] || $PY train_flow_dpo_v5.py --flow "$BASE" --pairs "$PAIRS" --out "$OUT" \
      --metric "mss=1,logmel_l1=0.5" --beta $BETA --lr 1e-5 --steps 2000 --batch 64 --device "$DEV"
done

echo "== 4. GATE: held-out synthetic audio evaluation (the step-4 gate the S5 DPO did not have)"
$PY audio_eval_v5.py --bass_manifold "$B" --all_manifold "$A" --plugin "$PLUGIN" --n 256 --k 8 --workers 14 \
    --device "$DEV" --flow V6C="$BASE" --flow V6C_b100="$RUNS/dpo_v6c_b100/flow_dpo_latest.pt" \
    --flow V6C_b300="$RUNS/dpo_v6c_b300/flow_dpo_latest.pt" --metrics logmel_l1 mss modspec_dist \
    --out "$RUNS/audio_eval_v6c_dpo"
# Read report.md: the paired deltas V6C_b* - V6C must be below 0 with the CI excluding 0 before step 5 means much.

echo "== 5. real stems: V6.1 eval (polish + Occam + transcription filter), all systems in one run (same MIDI)"
$PY eval_real_stems_consensus_v6.py --model S5 "$S5" --model S5_DPO_b300 "$DPO5" --model V6_C "$BASE" \
    --model V6_C_DPO_b100 "$RUNS/dpo_v6c_b100/flow_dpo_latest.pt" \
    --model V6_C_DPO_b300 "$RUNS/dpo_v6c_b300/flow_dpo_latest.pt" \
    --max_midi 60 --drop_polyphony --occam_tol 0.02 --square_shape "$SQUARE" --device "$DEV" \
    --out_dir "$RUNS/stems_v6_c_dpo_occam"

echo "== 6. paired statistics and the audition page"
$PY stem_stats_v6.py "$RUNS/stems_v6_c_dpo_occam/consensus_eval_summary.json" --ref V6_C \
    --out "$RUNS/stems_v6_c_dpo_occam/stats_phrase.md"
$PY stem_stats_v6.py "$RUNS/stems_v6_c_dpo_occam/consensus_eval_summary.json" --ref V6_C --metric mel_ssim \
    --out "$RUNS/stems_v6_c_dpo_occam/stats_ssim.md"
$PY build_interactive_model_eval_page.py "$RUNS/stems_v6_c_dpo_occam"
echo "done: listen at $RUNS/stems_v6_c_dpo_occam/listen/index.html"
