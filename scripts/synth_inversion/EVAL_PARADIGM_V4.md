# Evaluation paradigm v4: how we test inverter changes from now on

October 12, 2026. Reviewer: KUANG (Claude Opus 5.5) for Kim. Reviews `V4_EXPERIMENT_REPORT.md` and the code at
`091ea6b` (`fix/inharmonicity-aux`), proposes the testing paradigm and the next runs, and adds the tools that
implement it. Evidence tags: **[V]** verified in code or data here, **[I]** inferred, **[N]** from project notes,
**[U]** not run.

## 0. Summary

- **What A1–A3 support.** Temporal ASP in the flow's encoder lowered the held-out flow loss from 0.235 to 0.186
  and 0.181, about 10x the difference between two runs that share that flow configuration [V numbers from the
  report, I reading]. Nothing else in the report clears the run-to-run difference visible in its own table.
  The inharmonicity head's effect on retrieval is +1.5 points one way and −1.9 the other.
- **Why "A3 was not universally best" is expected.** JEPA and flow are separate models with separate optimizers,
  losses and gradient clipping, and the flow is not conditioned on the JEPA [V]. A3 is therefore A1's JEPA
  configuration plus A2's flow configuration. Its JEPA numbers should match A1's and its flow numbers A2's,
  up to run-to-run noise, and that is what the table shows.
- **Paradigm in one line.** Pre-register, score every run on the same held-out items, compare paired
  differences with presets as the resampling unit, require differences to exceed a replicate gauge, and settle
  on real stems with dev/test splits and blind listening.
- **Next run.** Train A0 (conv flow, no aux head) and a second A3, both seeded (§4). That completes the 2x2 and
  gives each contrast two runs per side, for 10 GPU-hours.
- **Still worth raising from the research outcomes** (§5): the noisy-reference `tau` sweep done properly (no
  training), validating the automatic metrics against blind preference, and the FX domain gap.

## 1. Review of what we have (`091ea6b`)

### 1.1 What the A-series establishes

Because the modules are independent, pairs of runs that share a module's configuration are free replicates
for that module's metrics [V code, I use]:

| Replicate pair (same config for this module) | Metric | Run-to-run difference |
|---|---|---|
| A1 vs A3 (JEPA with aux head) | held-out a→p / p→a online | +1.8 / −1.8 points |
| A1 vs A3 | ρ_jepa | 0.035 |
| A1 vs A3 | train-preset JEPA loss | 6.7 % |
| A2 vs A3 (temporal ASP flow) | held-out flow loss | 2.7 % |
| A2 vs A3 | ρ_flow | 0.035 |

| Contrast | Estimate from the report | Against the gauge |
|---|---|---|
| Temporal ASP vs conv flow: flow loss | 0.235 → 0.183 (mean of A2, A3), −22 % | 0.052 vs a 0.005 gauge (about 10x): likely real, on synthetic held-out data |
| Temporal ASP: ρ_flow | 0.463 → 0.543 | ~2x the gauge: suggestive |
| Aux head on JEPA retrieval (A1, A3 vs A2) | a→p +1.5, p→a −1.9 points (online) | inside the 1.8-point gauge, opposite signs: no effect shown |
| "v4 vs baseline" retrieval (14.5 % → ~20 %) | | A2 has no aux head and a flow-only change, yet its JEPA reaches 19.5 %: the gain is the recipe or noise, not a v4 component |

Single runs, one checkpoint each, and A1 stopped at 11 570 of 12 000 steps (in the WSD decay tail). These
are readings of the report's numbers, not a re-analysis of per-item data, which the old pipeline did not
keep. §3 does that re-analysis.

### 1.2 Problems found (report and code at `091ea6b`)

| # | Finding | Evidence | Fix |
|---|---|---|---|
| P1 | The arms were not seeded: `launch_experiments.sh` and `queue_a3_after_a2.sh` never set `SEED`, and `run_v3_recipe.sh` passes `--seed` only when it is set | [V] `run_v3_recipe.sh:56`; trainer seeds only `if args.seed is not None` (`train_realistic_bass_v3.py:379, 436`) | Always set `SEED`; record it in `run_meta.json` |
| P2 | The report's noise estimate is for one arm on 512 independent notes. The notes are clustered by held-out preset, the comparison is paired, and training variance is not evaluation noise. The quoted "8 notes" gap is 14 (22.3 vs 19.5 % of 512) | [V] | `eval_stats_v4.paired_delta` + replicate gauge |
| P3 | §3.2's "Pareto trade-off" has no ablation behind it, and its JEPA losses (1.485 vs 1.432) contradict the table (A3 lowest at 0.2557) | [V] report | Withdraw; a testable mechanism is P4 |
| P4 | JEPA gradient clipping is a global norm of 1.0 over all JEPA losses including the aux term ×100. If the aux term dominates the norm, clipping rescales the retrieval losses too, so the aux head would change the JEPA's effective learning rate | [V] `train_realistic_bass_v3.py:795-796`; magnitude [U] | Log the pre-clip norm per loss term and the clipped-step fraction |
| P5 | The aux head's own held-out readout is not reported, though `validate_inharmonicity` logs held-out MSE, R² and the constant-baseline MSE | [V] `train_realistic_bass_v3.py:151` | Report `aux_r2` and `aux_mse` vs baseline per run |
| P6 | The "temperatures" in the multi-temp evaluation are the condition-noise level `tau` (Synth-JDF's smoothed posterior), not a sampling temperature. At `tau < 1` one noise draw is shared by all 20 candidates (`model.noisy(mel, tau_t[0:1])`), so each low-`tau` result rests on a single noise realization | [V] `inference_v3.py:58-62`, `render_a1_multitemp_eval.py:214` | Rename; draw the noise per candidate; repeat over seeds (§2, T2) |
| P7 | Musical snapping writes `feg_to_sync_depth`, `sync_static_floor` and `osc_<n>_pitch_raw`, which are not parameters of the v3 spec. `apply_patch` sets only known keys, so the "FEG-coupled sync" rule only zeroes filter sustain and lengthens filter decay, and fine-pitch snapping does nothing | [V] `musical_snapping.py:65, 149-150`; `surge_spec_v3.py:436-480` | Assert unknown keys; the sync coupling needs a modulation route (surgepy), §5 |
| P8 | The channel-9 fix the report calls for is not applied. The root is still the most frequent note over all channels | [V] `invert_stem_collection_v3_cpu.py:279-293`; `render_a1_multitemp_eval.py` root line | Apply the fix; T0 root check |
| P9 | The multi-temp evaluation picks among 20 candidates using the phrase's first sixteenth note, then scores the whole phrase, which contains that note. It also reports the best of 20 systems on the same 10 stems with no dev/test split | [V] `render_a1_multitemp_eval.py:170-247` | Held-out-half protocol; dev/test (`stem_compare_v4.py`) |
| P10 | The v4 flow export carries no encoder type; `models_v3.load_inverter` cannot load A2/A3 | [V] trainer `export()`; `models_v4.load_inverter_v4` infers it from the state keys | Load with `load_inverter_v4` |
| P11 | The report lists musical snapping under "trained in neural weights" and the two-anchor objective under "not yet trained" | [V] | Snapping is post-processing; two-anchor can only be used at inference (Surge is not differentiable) |
| P12 | The DualMel plan cites "transient attack softening identified in Section 1"; no such finding is in the report | [V] | Gate A4 on §3 results |
| P13 | A2 and A3 were resumed after a suspend, with `elapsed_s` edited by hand | [N] report §5.2 | Check the logs for the resumed step count and the Schedule-Free state (the bug class in `training-findings.md` 20a) |

## 2. The paradigm

**Principles.**
1. **Pre-register.** Write the hypothesis, primary metric, minimal effect of interest and decision rule before
   scoring (`eval_plan_v4_aseries.json` is the template).
2. **Same items, paired.** Every run is scored on the same held-out items; compare per-item differences.
3. **Resample presets.** The bootstrap unit is the held-out preset, not the note.
4. **Replicates set the bar.** A difference counts only if it exceeds the difference between runs of an
   identical configuration (the gauge). Without a replicate the verdict is marked provisional.
5. **Integrity before comparison.** A pipeline that cannot reproduce the trainer's own logged number is not
   measuring the model.
6. **Real stems decide, and selection never sees the test stems.**

**Decision rule** (`eval_stats_v4.verdict`): *established* when the 95 % CI excludes 0 **and**
|Δ| ≥ max(minimal effect, gauge), with the Holm correction over the plan's primary metrics. An established call
must also hold for both EMA and online weights. Everything else is *not established*, which is a result, not a
failure.

**Tiers**

| Tier | What | Tool | Cost |
|---|---|---|---|
| T0 integrity | Parity with logged numbers on the trainer's own held-out set; render integrity (FX truly off, `test_fx_off_single_note_has_no_delay_trail`); MIDI root equals the audio's pitch per stem (channel 9 excluded); any inference-time transform (snapping) must change the render as claimed and reject unknown keys; mechanism audit for optimizer flags; seeds and `num_workers` recorded | `heldout_eval_v4.py` parity printout; existing tests | Minutes |
| T1 synthetic held-out, per item | Retrieval rank (a→p, p→a) in galleries of 512, reciprocal rank, flow loss per note (fixed noise and t), per-parameter error of the flow estimate, categorical accuracy, aux prediction, per-ladder ρ. Base set = the trainer's (parity), extra set (2 048 notes) and extra ladders (32 per axis) for power | `heldout_eval_v4.py`, `compare_runs_v4.py` | CPU render about 1-3 min per set (estimate); scoring minutes on GPU |
| T2 real stems | Held-out-half protocol (fit or select on the first half, judge on the second); pre-registered seeds (4 per stem) and selector, identical for every system; dev/test stem split for anything chosen by score (`tau`, kind, snapping); per-stem mean over seeds, exact sign test (24 stems need 18 wins), bootstrap over stems; FX and no-FX stems reported separately | `stem_compare_v4.py` (+ the existing `eval_refine_phrase.py` style renders) | CPU, about an hour |
| T3 blind listening | "Which is closer to the reference?", forced choice, sides and order randomized, loudness matched, attention checks, clips from the T2 protocol only (the listener never picks a seed) | `listening_ab_v4.py make / analyze` | ~30-45 min of listening per 24-stem pair |
| T4 metric validation | Agreement of each automatic metric (MSS, wMFCC, envelope cosine, two-anchor at 1:1:1, JEPA distance) with the T3 choices, with a stem bootstrap; the best-agreeing metric becomes T2's primary | analysis of T3 output | Free once T3 exists |

**Every results handover must contain:** the plan file; per-item files (`heldout_eval_v4.py` output); the
parity lines; seeds, `num_workers`, steps completed and resumes per run; which weight kinds were scored; the
`compare_runs_v4.py` report; and for real stems, the dev/test split and the seed protocol.

## 3. Applying it to A1–A3 now (no training)

```
# T0 + T1 (training machine; renders on CPU, scores on the GPU)
python scripts/synth_inversion/heldout_eval_v4.py \
    --run A1=/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a1 \
    --run A2=/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a2 \
    --run A3=/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a3 \
    --kinds sf online ema --tag latest --extra_n 2048 \
    --out /run/media/kim/Mantu/surge_200k_models/eval_v4 --device cuda:0
python scripts/synth_inversion/compare_runs_v4.py --dir /run/media/kim/Mantu/surge_200k_models/eval_v4 \
    --plan scripts/synth_inversion/eval_plan_v4_aseries.json

# T2 on the existing A1 multi-tau output (re-render after the P8 fix first)
python scripts/synth_inversion/stem_compare_v4.py --multitemp \
    /run/media/kim/Mantu/surge_200k_models/real_bass_v3_a1/eval_cpu_muscriptor_temps/results.json \
    --metric mss --baseline ema/tau1.00/raw --select "ema_tau=ema/tau" --select "online_tau=online/tau"
```

Check the parity lines first. If a recomputed number differs from the logged one on the same device type, stop
(T0). Expected outcome [I]: the flow-loss and parameter-error effect of temporal ASP survives the cluster
bootstrap; retrieval differences between A1, A2 and A3 do not.

## 4. New run suggestions

| Option | Benefit | Cost | Risk | Reversibility |
|---|---|---|---|---|
| **A. Seeded A0 (conv flow, no aux) + seeded second A3 (recommended)** | Completes the 2x2. Both contrasts get 2 runs per side: aux (A1, A3, A3′ vs A2, A0) and flow (A2, A3, A3′ vs A1, A0). Gives a JEPA no-aux replicate (A0, A2) and a conv-flow replicate (A0, A1) | 2 x 5 h GPU | Low | Full |
| B. Second seed of all four cells | 2 runs per cell, interaction testable | 4 x 5 h | Low | Full |
| C. Go straight to A4 (DualMel) | The next feature sooner | 5.5 h | High: one more single run on a design that cannot yet tell signal from noise | Full |

**Recommendation: A**, then decide A4 on §3 plus the A results. *Assumption that would change it:* if §3's per-item
re-analysis already shows the ASP flow effect far beyond the gauge **and** T2 shows a real-stem gain on the test
stems, run A4 seeded together with a seed-matched A3 instead.

**Run hygiene for every new run:**
- `SEED` set; the same seed list across arms of one comparison.
- `NUM_WORKERS` fixed and recorded (in the LatCH bracket, worker count alone moved results by 0.004 [N]).
- `MAX_STEPS` reached by every arm; no hand-edited `elapsed_s`. If a run resumes, check the step-count and
  Schedule-Free lines.
- Validation and checkpoint export on the same steps, so the parity check has a logged record.
- Log the per-term pre-clip gradient norms and the clipped-step fraction (P4). This is a small trainer change
  [U, not done here].
- Optional [I]: a `--train_only jepa|flow` switch would halve the cost of single-module contrasts, because the
  modules never interact.

## 5. Research outcomes still worth raising

From the Q1–Q10 report, my review of it (October 11, 2026), and `EXPERIMENTS.md`:

| Rank | Item | Why now | Cost | Fits tier |
|---|---|---|---|---|
| 1 | **Noisy-reference conditioning sweep** (Synth-JDF, H5): `tau ∈ {1, 0.9, 0.7, 0.5, 0.3, 0.1}` on real stems, `tau` picked on dev stems and reported on test, noise drawn per candidate, ≥ 4 seeds | A1–A3 were trained with `--cond_noise`, so no training is needed. The current sweep has P6 and P9 | CPU, ~1 h | T2 |
| 2 | **Validate the automatic metrics against blind preference** (Q1, Q10). Candidates include the CLAP "Huang style" statistics, best in the 21-dataset timbre benchmark (arXiv 2507.07764) | Every model choice so far rests on unvalidated metrics; MSS "is coarse" by our own note | ~45 min of listening | T3, T4 |
| 3 | **FX domain gap**: a differentiable FX stage fitted per stem after inversion (H9, DiffVox-style), and clean/wet FX consistency augmentation (Q8, rank 1) | H1 showed the real-vs-synth gap is the binding constraint, not the FX fix [N] | CPU per stem; augmentation needs a run | T2 |
| 4 | **wMFCC and a pitch-normalized harmonic-envelope distance** as rerank channels (Q1 #3) | Cheap; `resonance_v4.py` already computes the harmonic envelope | Small | T2 |
| 5 | **Renderer-selected best-of-N distillation** (Q7 #1) | Uses the renderer without differentiability; needs a validated metric (item 2) first | One run | T1, T2 |
| 6 | **Pedalboard vs surgepy equivalence test** before enveloped sync/FM (Q6) | Also the only route by which musical snapping's sync coupling (P7) can work | Small | T0 |
| 7 | **MLDR envelope loss in refinement** (H8, DiffVox) | Targets the attack and decay errors without retraining | CPU | T2 |
| 8 | **Leads not yet read**: DDSynth-RL code and checkpoints (CLAP and CREPE rewards), InverSynth II code, White Box Search, SynthRL, arXiv 2608.18226 | Read before any RL step | Reading | — |

MoSQITo roughness (Apache-2.0) is a cheap extra buzz meter for T2 next to `buzz_v4.py`. Joint audio-parameter flow
training (Synth-JDF) stays a later bet, after items 1 and 3 show how much of the real-stem gap is conditioning.

## 6. Tools added on this branch

| File | Role |
|---|---|
| `eval_stats_v4.py` | Paired cluster bootstrap, design effect, sign test and `min_wins`, McNemar, Holm, replicate gauge, decision rule, minimal detectable difference |
| `heldout_eval_v4.py` | Per-item re-scoring of `sf`, `online` and `ema` exports on the trainer's own held-out set (same RNG stream) plus larger sets, with each note's held-out preset; parity against `val_kinds.jsonl` |
| `compare_runs_v4.py` | Runs a pre-registered plan: group means, paired deltas, gauges, Holm, verdicts, Markdown report |
| `eval_plan_v4_aseries.json` | The plan for A1–A3 (and A0, A3′ once trained) |
| `stem_compare_v4.py` | Real-stem comparisons with a dev/test split, sign test and seed spread; reads the multi-temp `results.json` |
| `listening_ab_v4.py` | Builds a blind A/B page with attention checks and loudness matching; analyzes the answers |
| `test_eval_v4.py` | 27 CPU tests |

**Tested here (CPU, no Surge):** 27 tests pass, and the existing 40 v4 tests still pass. The draw functions
reproduce `render_fixed_set`'s and `render_ladder_set`'s RNG streams exactly. `score_items` and `score_ladders`
reproduce the trainer's own `validate()` and `ladder_validation()` (retrieval exactly, flow loss to 1e-5,
ladder means to 1e-9) for both v3 and v4 flows. Exports round-trip through the loaders. The command-line path runs
end to end with only the Surge render faked.

**Not tested:** rendering with real Surge, the Essentia inharmonicity labels, CUDA (parity holds per device
type), `pyloudnorm` (the RMS fallback was used), and anything on real checkpoints or stems.
