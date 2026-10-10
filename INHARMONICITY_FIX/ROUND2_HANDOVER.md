# Round 2 handover: the inharmonicity aux head, the live run, and the reference recipe

Author: KUANG, October 11, 2026. Audience: the Antigravity agent (and Kim). Base for round 2: commit `a08d184`
on `fix/inharmonicity-aux` (your round 1). Read `SPEC.md` first: its rules R3 to R13 still apply. Labels: **[V]**
read in code or output, **[I]** inferred, **[D]** a decision for Kim, never for you.

---

## 0. Where things stand

Kim's live run, reported October 11, 2026: step 19 950; JEPA loss about 0.40-0.44; flow loss 0.16; aux loss (as
logged) 0.01-0.05; valid fraction `v` = 0.32-0.34; in-batch retrieval a2p and p2a rose from 94% to 96%. Kim has also
asked you for test clips of the current state (section 5). Kim answered D1-D6 or authorized your judgment, so the
render-timeline change, the Softplus head and the config thresholds are accepted; record them (T8), do not revert them.

Round 1 review, status of each finding:

| Finding | Status |
|---|---|
| Default path (`--inharm_weight 0`) crashes: `i_loss = 0.0`, then `float(i_loss.detach())` at `train_realistic_bass_v3.py:741` | **Fix (T3a).** The live run is unaffected (head on) |
| Top-level `from inharmonicity_target import ...` on line 1 forces Essentia even when the feature is off | Fix (T3b) |
| Resume: `opt_jepa` / EMA loads unchanged, so resuming across a head on/off change probably raises | Verify, then fix (T3c) |
| Pooled result drops `analysis_configuration_id`, `coverage_diagnostics`, `window_results` | Fix (T3e); new test provided |
| `report_inharmonicity_validity.py` hardcodes a `/run/media/...` drive path as its default manifold, draws each patch twice | Rewrite (T2) |
| `run_v3.sh` launches `train_realistic_bass_overnight.py` (v1 imports), the v3 modules live in `train_realistic_bass_v3.py` | Verify (T3d) |
| `measure_phrase_timeline.py` and the baseline hashes never committed | Superseded by Kim's D6 answer; skip |

Verified in review (so you need not redo it): the loss is the spec's version and passes; the model flag and RNG
ordering pass; the extractor follows the API; your real-Essentia fixtures confirm SPEC section 1.1 (exception type
`RuntimeError`, empty input gives 0.0, all-zero magnitudes give 1.0).

---

## 0b. Decisions recorded, and what changed since the first draft (October 11, 2026)

**Decided by Kim:** D7 = a twin of the live run. D8 = refuse loudly. D9 = write the changes (below).
**Recommended by KUANG, awaiting Kim's confirmation:** D10 and D11 (end of this section).

**New finding that affects everything: the v3 parser's DEFAULTS are not the recipe** [V: parsed with the real parser at
`a08d184`]. A command that names only a few flags runs batch 32, LRs 9e-5 and 6e-5, a constant LR (`--lr_floor 1.0`),
no ladders (`--ladder_every 0`), no OT coupling, no EMA, no `--cond_noise`, an MSE flow loss and smaller networks
(512/512/1152). Of the 32 recipe flags, 16 differ for a minimal command. So "cond_noise is off" is probably not the only
difference in the live run. There is also NO `--seed` flag [V], so "same seed" in the first draft was not possible.

**D7: the twin.** Generate it from the live run's own record, never by retyping flags:

```
python twin_command.py <live out_dir> --out-dir <twin out_dir>
```

`provided_round2/twin_command.py` prints which recipe flags differ from the live run, then the twin command: every flag
of the live run except `--out_dir` (new), `--inharm_weight` (0), `--inharm_config` (dropped), `--resume_from` (dropped)
and `--purpose` (rewritten). `max_steps` and the LR settings are kept, so the schedule is identical. Do not launch it
(the GPU is held); save it next to the live run's `RUN_DEFINITION.md`. **Early read:** first copy the live run's
checkpoint files (T0). The trainer prunes rolling checkpoints to the last 3 (`--keep_last`), so the live run's weights
at the step where the twin stops are gone unless copied now. Run the twin to that step, compare at equal step
(retrieval, T4 probe), and continue only if the result is inconclusive.

**D8: refuse.** In `train()`, after the checkpoint is read and BEFORE any state is loaded: the checkpoint's head state
is "any key starting `inharmonicity_head` in `ckpt["jepa_state_dict"]`"; this run's is `args.inharm_weight > 0`. If they
differ, print a message naming both states and the two fixes (a new `--out_dir`, or the same `--inharm_weight`
setting) and exit non-zero. Delete the "will be randomly initialized" and "head weights ignored" branches. Test on CPU
with a tiny run: refusal in both directions, and a normal same-flag resume still works. No warm start.

**D9: the launcher.** (a) Copy `provided_round2/run_v3_recipe.sh` and `provided_round2/twin_command.py` byte for byte into
`scripts/synth_inversion/` (`chmod +x` the shell script). (b) In `run_v3.sh`, insert after the shebang line, and change
nothing else:

```
# SUPERSEDED (D9, October 11, 2026): this launcher calls train_realistic_bass_overnight.py, the v1 stack, with a v3
# manifold. Use run_v3_recipe.sh for the v3 trainer.
```

(c) Acceptance: run `OUT_DIR=<tmp> MAX_STEPS=78416 HOURS=30 DRY_RUN=1 bash run_v3_recipe.sh`, parse the printed command
with `build_parser().parse_args(...)`, and compare with `RECIPE` in `twin_command.py`: all 32 recipe flags equal (KUANG
checked this at `a08d184`). Paste the check. The launcher has no default for `OUT_DIR`, `MAX_STEPS` or `HOURS`
(decision D11) and writes `launch_command.txt` into the out dir, so every run records its exact command.

**D10, KUANG's recommendation (gates for the T6 sweeps; fix them BEFORE looking at which cell wins).** Per grid cell:
1. False-harmonic rate at most 5%.
2. Clean-class valid rate per window at least 90% (about 80% pooled over two windows, if independent).
3. Tracking: Spearman at least 0.7 between the descriptor and the swept parameter on valid windows, for `fm_depth` and
   `unison_detune`, with the same sign at every MIDI note.
4. Pitch neutrality on the clean class: the valid-rate spread across MIDI notes at most 20 percentage points, and
   |Spearman(descriptor, MIDI note)| at most 0.3.

Among cells passing 1 to 4, pick the highest pooled valid rate on the real prior (T2). If the best is below 50%, flag it
to Kim: the label would then describe a minority of the prior. These numbers are judgment, not derived from data; they
exist so the choice is made by rule, not by eye.

**D11, KUANG's recommendation.** The twin has weight 0, so `W` matters only for the live run (already set) and later
arms. Measure it, do not guess: on the copied live checkpoint, on CPU, compute the gradient norm with respect to `za`
of the JEPA loss and of the unweighted aux MSE on valid rows, and set `W = target_ratio * |g_jepa| / |g_aux|`. Ratios:
about 0.02 to 0.05 makes the head a probe in disguise (encoder barely moves); about 0.1 to 0.3 is the likely useful
range (watch a2p and p2a for a drop of more than 1 point); 1 or more lets aux dominate and is only an upper bracket.
Run length for the twin: stop at the step of the copied live checkpoint (the early read), then decide; the full planned
length is the final-quality comparison, after the LR decay. At the reference speed of about 4 100 steps/h, 20 k steps is
about 5 h and 78 416 steps about 19 h (estimates; use the real rate).

---

## 0c. EMA, cond_noise, schedule-free, the regulators and the LR: what the code says (October 11, 2026)

Kim's position: the recipe runs EMA, cond_noise and schedule-free, with the Scion oracles, the schedule-free damping,
Hyperball and the other regulators. What the v3 trainer does [V: `train_realistic_bass_v3.py` lines 360-417 and
`stable_audio_tools/training/modular_opt/` at `a08d184`]:

| Item | Status in the v3 trainer |
|---|---|
| Schedule-free | Hard-wired on in the modular path (`schedule_free=True`); `--sf_c_warmup 1000` and `--sf_r 1.0` are the defaults |
| Scion role-based oracles | On in the modular path via `build_modular_param_groups`: embeddings ColNorm, hidden 2D weights Spectral with radius scaling, outputs and AdaLN emitters Sign (`routing.py`). NorMuon is hard-wired on |
| Variance-aware dampening | On by default: `--var_dampening 1.15` (`var_dampening_power` stays at the optimizer's default 1.0) |
| Radial brake | On by default: `--radial_brake 0.8` |
| Overtraining-aware weight decay | On by default: `--wd_overtraining`, `--wd_jepa 0.02` |
| Whitening and NS polynomial | `--whitening shampoo` and `--ns_poly cubic5` by default |
| SNR gate | Off by default, and the recipe wants it off (`run_modular_shampoo_b64.sh` header: "avoiding artificial LR braking") |
| Escape velocity | In `ModularOptimizer` but NOT passed by the trainer: off, and there is no flag for it |
| **Hyperball** | **Not in `ModularOptimizer`.** It exists only in `FusionOpt` (`fusion_opt.py`), reachable with `--optimizer fusion --fusion_components ...` (default list `shampoo,sf,normuon,ns5,mona`, no hyperball). No reference-recipe run used it |
| EMA | OFF by default (`--ema_halflife_steps 0.0`); the recipe sets 2000. It only averages weights for export; it does not change the online training |
| cond_noise | OFF by default; it affects only the flow |

[V: with the real parser, a minimal command matches the recipe on every optimizer and regulator flag.] So a command
that names few flags already has the Scion oracles, schedule-free averaging, damping, radial brake and weight decay.
What it lacks is EMA, cond_noise, ladders, OT coupling, the MAE loss, batch 64, the recipe LRs and the LR schedule.

**Hyperball and escape velocity (proposed decision D12).** (a) Leave them out, matching every reference run; the radial
brake already limits norm growth. (b) A separate experiment on `--optimizer fusion` with hyperball added: a different
optimizer family, so not comparable with H1, H6 or v3-clean, and it needs its own control. (c) Port Hyperball into
`ModularOptimizer`, with tests, then ablate it. KUANG recommends (a) now and (b) or (c) as a later experiment. Exposing
`--escape_velocity` as a trainer flag is a small change if Kim wants it.

**The learning rate: what is recorded and what is not** [V: launcher headers and `HANDOVER_2026-10-02_realistic_prior.md`].
- Baseline: JEPA 9e-5, flow 6e-5, batch 32 (the parser defaults; `run_overnight_realistic_bass.sh`).
- `run_modular_shampoo_b64.sh`: batch 64 and "2x LR" (1.8e-4 and 1.2e-4).
- `run_fxfix_ladder_ema.sh` (H1): "1.5x peak LR" (2.7e-4 and 1.8e-4) with a linear decay to 1/3 over the last 30% of
  steps. Every later recipe run keeps these (H6, its continuations, v3 clean); `run_v3.sh` uses 3.0e-4 and 2.0e-4.
- The only deliberate lowering is the end-of-run decay (warmup, constant, linear decay). It replaced v1's watchdog,
  which halved the LR on noisy loss; that was removed ("LR decay belongs in the trainer's schedule").
- No LR sweep, range test or ablation for the synth-inversion JEPA or flow appears in the launcher headers, the
  benchmark summary, the walkthrough, or the main repo's worklog and dialogue [I: that does not prove none exists; logs
  on the training machine were not searched]. Both increases are stated as scaling heuristics, not test results.
- So the live run's LR is almost certainly a parser default (9e-5 and 6e-5, constant), not a decision. T0's table
  shows the real values.

**Revisit D7 after T0.** EMA only averages weights for export, and cond_noise, OT coupling and the MAE loss affect only
the flow, so a twin that shares them with the live run is still a valid test of the aux head. Ladders (an ordinal JEPA
loss on every 4th step), batch size, the LRs and the LR schedule DO change how the JEPA trains. If the T0 table shows
the live run differs from the recipe on those JEPA-side flags, the aux result may not transfer to the recipe: then
restart both arms on the recipe (the other D7 option, A0 and A1, with the early read keeping it to about 5 h per arm)
and keep the live run as a pilot. If the live run differs only on flow-side or export-side flags, the twin stands.

---

## 1. Rules for round 2

- **RR1 Never touch the tree the live run uses.** Do not edit, check out, pull, rebase or `git switch` there. The
  watchdog restarts the trainer after a crash [V: the launchers wrap it in `watchdog_supervisor.py`, and the trainer
  exits with code 3 after `--max_bad_steps` skips], `v3_post.sh` resumes it after the time cap, and any restart
  re-reads the files on disk; DataLoader workers import
  `inharmonicity_target` lazily on first use [V]. Work in a separate worktree:
  `git worktree add <path-outside-the-run-tree> -b fix/inharm-round2 a08d184`.
- **RR2 CPU only.** The run holds the GPU. Use `nice -n 10`. Do not take the GPU lock. Surge renders on CPU are fine.
- **RR3 No thresholds invented.** Calibration produces evidence; Kim picks the numbers. Constructor fields stay
  required, with no defaults.
- **RR4 Nothing is pushed unless Kim asks.** Branch name for this round: `fix/inharm-round2`.
- **RR5 Real output only (R4).** Paste the last 15 lines of every command that matters.

---

## 2. What the live run is, and what it cannot tell you

- **The JEPA and the flow are independent networks that see the same batch.** `jepa(...)` and `flow_loss(...)` both
  take `mel` and `params`, each has its own optimizer and EMA [V: training loop and the module docstring "Both models
  see the same batch each step"]. Consequences:
  - The aux head touches only the JEPA. It cannot change the flow, so it cannot change the clips the flow produces.
  - `--cond_noise` touches only the flow. It adds a `tau_embed` module and a tau input [V: `models_v3.py` lines
    133-134 and 175-178], so a flow trained without it has different weights. **This run cannot be resumed with
    `--cond_noise` turned on** (the keys differ) [I: strict load fails; confirm before relying on it].
  - `invert_stem_collection.py --cond_tau < 1` refuses a flow without `cond_noise` [V: lines 337-339]. For the live
    run use `--cond_tau 1.0` only.
- **So the clips Kim asked for judge the flow, not the aux head.** The head is judged in the JEPA: retrieval, the
  linear probe (T4) and the held-out aux error (T7).
- **Reading the log.** `Aux` = `inharm_weight * MSE`, averaged over the interval, with zeros for all-invalid batches
  [V: diff]. `v=` is the batch mean of the pooled validity [V]. Neither number is interpretable without the label
  variance (T2): predicting the mean gives MSE equal to the variance.

---

## 3. The reference recipe ("v3 clean": H1 + H6 + MAE) and what is comparable

Sources: `run_v3_clean_home.sh` (header: "H1 recipe + --cond_noise (H6) + --flow_loss mae", Surge FX truly off) for
the recipe, `run_v3.sh` for the v3 network sizes. All flags below appear in those launchers [V]; whether every one
exists in the `train_realistic_bass_v3.py` parser is now **[V]**: all 32 recipe flags exist (checked against the parser
at `a08d184`). The parser's DEFAULTS differ from the recipe on 16 of them (section 0b).

| Flag(s) | Value in the launchers | Purpose |
|---|---|---|
| `--manifold` | v3 prior: `real_bass_manifold_v3.npz` | realistic bass prior, v3 parameters |
| `--batch_size` | 64 | |
| `--embed_dim --flow_dim --ff_dim` | 768 768 2048 (`run_v3.sh`) | v3 scale-up (the v1 recipe used `--ff_dim 1152`) |
| `--optimizer modular --whitening shampoo --ns_poly cubic5 --radial_brake 0.8 --var_dampening 1.15 --sf_c_warmup 1000 --sf_r 1.0` | as shown | modular optimizer, schedule-free |
| `--wd_jepa 0.02 --wd_overtraining` | | JEPA weight decay |
| `--lr_jepa --lr_flow --lr_floor 0.333 --decay_frac 0.3` | 3.0e-4 / 2.0e-4 (`run_v3.sh`); 2.7e-4 / 1.8e-4 (v1 clean) | LR and end-of-run decay |
| `--random_patch_frac` | 0.05 | share of unconstrained random patches |
| `--ladder_every 4 --ladder_anchors 4 --ladder_steps 16 --ladder_workers 3 --lambda_ord 0.5 --ladder_val_anchors 8` | | ladder batches with an ordinal loss in place of SIGReg |
| `--ot_coupling` | on | minibatch OT pairing of noise and targets for the flow |
| `--ema_halflife_steps 2000` | | EMA weights exported next to the online ones |
| **`--cond_noise --cond_noise_clean_frac 0.3`** | **on** | H6: the flow trains on partly noised audio conditions (30% clean, else tau ~ logit-normal(0,1)); aimed at off-manifold real-stem audio |
| `--flow_loss mae` | `run_v3_clean_home.sh` only (not `run_v3.sh`) | MAE velocity loss (Synth-JDF supp. 4) |
| `--hours` `--max_steps` | 30 h and 78 416 steps (v1 clean); 8 h (`run_v3.sh`) | **[D]** for this run |

Aux arm adds: `--inharm_weight W --inharm_config scripts/synth_inversion/inharm_config.json` ([D] for W).

**Which comparisons are valid**

| Question | Valid comparison | Not valid |
|---|---|---|
| Does the aux head help the JEPA? | two runs identical in every flag and in step count except `--inharm_weight 0` vs `W` (no `--seed` flag exists: T3f) | the live run against any earlier run |
| Are the flow clips as good as H6 or v3-clean? | only if `--cond_noise`, `--flow_loss mae`, FX off and the step budget match | the live run (cond_noise off) |
| Does `cond_noise` matter? | two runs differing only in `--cond_noise` | anything else |

**Decided (D7): a twin of the live run** with `--inharm_weight 0` (needs T3a), generated by `twin_command.py`. If Kim wants the
flow on the reference recipe anyway, start the pair **A0** (reference recipe, weight 0) and **A1** (reference recipe,
weight W); the live run then serves as a pilot [D7].

---

## 4. Tasks (none needs the GPU; do T0 first)

**T0 Record the run definition (read-only, about 10 minutes).** Write `RUN_DEFINITION.md` into the run's out dir:
the exact command line (`ps -o args= -p <pid>`), `git rev-parse HEAD` and `git status --short` of the run tree, the
contents of `run_meta.json` [V: written at a fresh launch], a copy and sha256 of `inharm_config.json`, python, torch
and essentia versions, and the value of `--inharm_weight`. Then print `python train_realistic_bass_v3.py --help` and
mark each reference-table flag present or missing, and mark each as passed or not passed in the live command. The
checkpoint does not appear to store the config [I: nothing in the diff does]; this file is the only record. Also NOW,
read-only: copy the live run's `checkpoint_latest.pt` and its exported `*_latest.pt` files to `<run>/ref_step<N>/` (N =
that checkpoint's step). The trainer prunes rolling checkpoints to the last 3 (`--keep_last`), and the twin's early read
needs the live run's weights at the step where the twin stops. Then run `python twin_command.py <live out_dir> --out-dir
<twin out_dir>` (section 0b) and paste its WHOLE output: it prints which of the 32 recipe flags differ from the live run.

**T1 Test clips of the current checkpoint.** See section 5.

**T2 Label statistics and validity (CPU).** Rewrite `report_inharmonicity_validity.py` in the worktree. No
hardcoded drive paths: import `MANIFOLD_PATH` from `realistic_bass_prior_v3`, accept `--manifold`. One draw per
sample, fixed `--seed`, `--n 2000` or more, both prior splits reported separately. Print and write `label_stats.json`:
pooled valid rate; per-window valid rate for window 1 and window 2; rejection reason histogram per window (needs T3e);
quantiles (min, 5, 25, 50, 75, 95, max), mean and std of valid per-window values and of the pooled label; the
distribution of `excluded_energy_fraction` over ALL windows as the share above 0.05, 0.10, 0.15, 0.25 and 0.35;
valid rate by MIDI note; and **the constant-prediction MSE = variance of the pooled labels**, next to the live run's
logged aux value divided by its weight. Acceptance: reruns with the same seed give identical JSON; no `/run/media`
string in the file.

**T3 Fix round-1 defects, in the worktree.** (a) `i_loss = torch.zeros((), device=device)` instead of `0.0` on the
disabled path. (b) Delete the unused top-level `from inharmonicity_target import InharmonicityExtractor` (line 1);
the dataset already imports it lazily. (c) Resume across a head on/off change: first reproduce the failure on CPU
with a tiny run; then make it REFUSE loudly (decided D8; exact behavior in section 0b; no warm start).
(d) Compare the import blocks of `train_realistic_bass_overnight.py` and `train_realistic_bass_v3.py`; if
`run_v3.sh` points at the v1 trainer with a v3 manifold, confirm it. KUANG already checked this: it does. Implement D9 as
written in section 0b.
(e) Make `pool_note_results` return the full schema and `window_results` (SPEC section 4). Copy
`provided_round2/test_inharmonicity_pooled_schema.py` byte for byte into `scripts/synth_inversion/` and show it red
first. **Acceptance:** all six test files pass with `REQUIRE_ALL=1` and 0 skipped; a 5-step CPU smoke run
(`--device cpu --batch_size 4 --num_workers 0 --max_steps 5`) logs a step with `--inharm_weight 0` AND with a weight
above 0; paste both.
(f) PROPOSED, needs Kim's yes before you write it: a `--seed` flag that seeds torch, numpy, python `random` and the
DataLoader workers, so a twin can start from identical initial weights. The trainer has none today [V].

**T4 Linear probe: does the embedding retain inharmonicity?** New `probe_inharmonicity.py`. Render fixed train-split
and val-split sets (fixed seeds, same `render_patch`), keep rows with a valid pooled label, compute `za` from the
exported JEPA weights with the exact input path the trainer uses, fit a ridge regression on the train split, and
report held-out R-squared and MSE against the constant baseline, with a bootstrap 95% interval, overall and by MIDI
note. Run it on: the live run's EMA JEPA, and an untrained (randomly initialized) JEPA as the floor. When a weight-0
twin exists, run it on that too; the aux head earns its keep only if the interval separates from the twin.

**T5 The twin (decided D7; it needs the GPU, so write it but do not launch).** Generate the twin command with
`twin_command.py` (section 0b), save it next to the live run's `RUN_DEFINITION.md`, and state the step at which the
twin should stop (the step of the T0 checkpoint copy).

**T6 Threshold calibration sweeps (CPU Surge).** Using `surge_spec_v3` parameters with FX off: a neutral clean-saw
patch (define it, show its parameters); sweeps of ONE factor at a time over about nine steps and at least six MIDI
notes: `fm_depth`, `unison_detune`, each `osc_*_octave`, `sync`, `noise_color`, waveshaper drive. Per window record
the descriptor value, excluded fraction, selected partial count and validity. Evaluate a grid of
`tolerance` in {0.10, 0.15, 0.20, 0.25}, `max_harmonics` in {20, 40, 80} and `max_excluded_energy_fraction` in
{0.05, 0.10, 0.15, 0.25}. For each cell report: (i) valid rate on the clean class; (ii) valid rate on each inharmonic
class; (iii) Spearman correlation between descriptor value and the swept parameter on valid windows; (iv) the
false-harmonic rate: inharmonic windows reported valid with a value below the clean class's 95th percentile.
Deliver the tables and plots. Once Kim confirms the D10 gates (section 0b), mark each cell pass or fail against them;
do not pick a winner beyond that rule.

**T7 Held-out aux error (design, then implement in the worktree).** Add a separate `validate_inharmonicity` called at
the validation interval, leaving `validate()` and `render_fixed_set` untouched (your S7 proposal). Log held-out MSE,
the constant-baseline MSE and R-squared for the train-preset and held-out-preset sets.

**T8 Record decisions.** `DECISIONS.md`: D1 to D6 as answered (ask Kim for the wording if you do not have it), plus
D7 to D11 (section 6), each with date and source of the answer.

**T9 Label sanity by ear (CPU).** From the val-split prior render 24 notes: 6 each at pooled-label quantiles 10%,
50% and 90%, and 6 rejected for high excluded energy. File names carry the label value. Kim listens to see whether
the value tracks detuned, metallic or noisy character. This is the perceptual check the SPEC asked for.

**T10 LR check (PROPOSED; needs the GPU, so after the twin and only if Kim says go).** There is no recorded test behind
the LRs (section 0c). A 3-point check at constant LR: 0.5x, 1x and 2x the recipe LRs for about 5 k steps each, all other
flags identical (build each command with `twin_command.py` and change only `--lr_jepa` and `--lr_flow`). Compare
held-out-preset retrieval and flow validation loss at equal step, before the LR decay starts.

---

## 5. T1 in detail: test clips for the current checkpoint

Run on CPU, from `scripts/synth_inversion` in the run's tree (read-only use; it writes only to a new out dir). The
launchers use `PY=$SAO/stable-audio-tools/sat-venv/bin/python` [V].

```
OUT=<the run's out_dir>
ls "$OUT"/*.pt        # use the names that exist; v3_post.sh expects the three below
for v in online ema sf; do
  case $v in online) CK=$OUT/flow_online_latest.pt;; ema) CK=$OUT/flow_ema_latest.pt;; sf) CK=$OUT/flow_latest.pt;; esac
  nice -n 10 "$PY" invert_stem_collection.py --flow_ckpt "$CK" --out_dir "$OUT/clips_cur_$v" \
      --cond_tau 1.0 --playback_midi muscriptor --no_user_copy
done
```

- `online` is the weights being trained (schedule-free y), `ema` the EMA of those, `sf` the schedule-free average
  (`flow_latest.pt`) [V: trainer docstring for the export helper]. If the trainer writes different file names, use
  those and say so.
- `--cond_tau 1.0` only: the live flow has no tau input [V].
- Use `--stems <ids>` for a quick first batch if time is short; the full set is 24 real stems.
- Scoring: if reference sets already exist (`clips_ref_old`, `clips_ref_h1_ema` from the clean run), run
  `compare_clip_sets.py` exactly as `v3_post.sh` does, adding the three new sets. If they do not exist, skip scoring
  and say so; do not re-render references (hours of CPU).
- What Kim listens for, per stem: the real stem, the MIDI playback, then each model: pitch and timing, timbre match,
  stray delay tail, noise. State plainly that these clips judge the flow trained WITHOUT `--cond_noise`, so they are
  the control for H6 and not a comparison with the H6 or v3-clean results.
- To judge the aux head by ear you need JEPA retrieval or reranking with and without the head (two models); the clips
  above cannot show it. T9 is the available check until the twin exists.

---

## 6. Decisions needed from Kim

- **D7 DECIDED (October 11, 2026): a twin of the live run.** Options were a restart on the reference recipe (A0 and
  A1) or a twin; the recommendation was the twin, unless
  you want the flow clips on the H6 recipe, because the JEPA question needs only the aux flag to differ.
- **D8 DECIDED (October 11, 2026): refuse loudly** on a resume across a head on/off change (a warm start was the
  alternative).
- **D9 DECIDED (October 11, 2026): write the changes.** A new launcher plus the twin tool, and `run_v3.sh` marked
  superseded (section 0b).
- **D10** Acceptance numbers for the calibration tables (valid rate on clean patches, false-harmonic rate). KUANG's
  recommendation is in section 0b; awaiting Kim.
- **D11** The aux weight `W` and the run length (hours or steps) for the next runs. KUANG's recommendation is in
  section 0b; awaiting Kim.

---

## 7. Report template

```
T0  RUN_DEFINITION.md path; command line; weight; twin_command.py output (pasted whole); checkpoint copy path and step
T1  clip dirs written, count of stems, scoring done or skipped and why
T2  commit <sha>; label_stats.json summary (pasted): pooled valid rate, per-window rates, constant-baseline MSE
T3  commit <sha>; six test files summary line (pasted); both smoke runs (pasted); D8/D9 findings
T4  probe table (pasted): R-squared with 95% interval for live EMA, untrained floor, twin if it exists
T5  the twin command file path and the step it should stop at, not run
T6  tables and plots written, no winner picked
T7  commit <sha>; sample log lines
T8  DECISIONS.md path
T9  clip dir and the quantile table
DEVIATIONS  none, or each with the reason and who approved it
```
