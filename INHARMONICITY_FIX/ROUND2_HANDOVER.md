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
exists in the `train_realistic_bass_v3.py` parser is **[I]**: T0 checks it with `--help`.

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
| Does the aux head help the JEPA? | two runs identical in every flag, seed and step count except `--inharm_weight 0` vs `W` | the live run against any earlier run |
| Are the flow clips as good as H6 or v3-clean? | only if `--cond_noise`, `--flow_loss mae`, FX off and the step budget match | the live run (cond_noise off) |
| Does `cond_noise` matter? | two runs differing only in `--cond_noise` | anything else |

Cheapest valid test of the aux head: a **twin of the live run** with `--inharm_weight 0` (needs T3a). If Kim wants the
flow on the reference recipe anyway, start the pair **A0** (reference recipe, weight 0) and **A1** (reference recipe,
weight W); the live run then serves as a pilot [D7].

---

## 4. Tasks (none needs the GPU; do T0 first)

**T0 Record the run definition (read-only, about 10 minutes).** Write `RUN_DEFINITION.md` into the run's out dir:
the exact command line (`ps -o args= -p <pid>`), `git rev-parse HEAD` and `git status --short` of the run tree, the
contents of `run_meta.json` [V: written at a fresh launch], a copy and sha256 of `inharm_config.json`, python, torch
and essentia versions, and the value of `--inharm_weight`. Then print `python train_realistic_bass_v3.py --help` and
mark each reference-table flag present or missing, and mark each as passed or not passed in the live command. The
checkpoint does not appear to store the config [I: nothing in the diff does]; this file is the only record.

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
with a tiny run; then make it fail loudly with a clear message, or implement a parameter-group-aware load **[D8]**.
(d) Compare the import blocks of `train_realistic_bass_overnight.py` and `train_realistic_bass_v3.py`; if
`run_v3.sh` points at the v1 trainer with a v3 manifold, report it, do not "fix" it before Kim answers **[D9]**.
(e) Make `pool_note_results` return the full schema and `window_results` (SPEC section 4). Copy
`provided_round2/test_inharmonicity_pooled_schema.py` byte for byte into `scripts/synth_inversion/` and show it red
first. **Acceptance:** all six test files pass with `REQUIRE_ALL=1` and 0 skipped; a 5-step CPU smoke run
(`--device cpu --batch_size 4 --num_workers 0 --max_steps 5`) logs a step with `--inharm_weight 0` AND with a weight
above 0; paste both.

**T4 Linear probe: does the embedding retain inharmonicity?** New `probe_inharmonicity.py`. Render fixed train-split
and val-split sets (fixed seeds, same `render_patch`), keep rows with a valid pooled label, compute `za` from the
exported JEPA weights with the exact input path the trainer uses, fit a ridge regression on the train split, and
report held-out R-squared and MSE against the constant baseline, with a bootstrap 95% interval, overall and by MIDI
note. Run it on: the live run's EMA JEPA, and an untrained (randomly initialized) JEPA as the floor. When a weight-0
twin exists, run it on that too; the aux head earns its keep only if the interval separates from the twin.

**T5 Matched baseline (needs the GPU; plan only until Kim says go).** Write the exact two or four commands for
section 3's pair, same seed and step count, differing only in the intended flag; do not launch.

**T6 Threshold calibration sweeps (CPU Surge).** Using `surge_spec_v3` parameters with FX off: a neutral clean-saw
patch (define it, show its parameters); sweeps of ONE factor at a time over about nine steps and at least six MIDI
notes: `fm_depth`, `unison_detune`, each `osc_*_octave`, `sync`, `noise_color`, waveshaper drive. Per window record
the descriptor value, excluded fraction, selected partial count and validity. Evaluate a grid of
`tolerance` in {0.10, 0.15, 0.20, 0.25}, `max_harmonics` in {20, 40, 80} and `max_excluded_energy_fraction` in
{0.05, 0.10, 0.15, 0.25}. For each cell report: (i) valid rate on the clean class; (ii) valid rate on each inharmonic
class; (iii) Spearman correlation between descriptor value and the swept parameter on valid windows; (iv) the
false-harmonic rate: inharmonic windows reported valid with a value below the clean class's 95th percentile.
Deliver the tables and plots only; Kim sets the acceptance numbers **[D10]**. Do not pick a winner.

**T7 Held-out aux error (design, then implement in the worktree).** Add a separate `validate_inharmonicity` called at
the validation interval, leaving `validate()` and `render_fixed_set` untouched (your S7 proposal). Log held-out MSE,
the constant-baseline MSE and R-squared for the train-preset and held-out-preset sets.

**T8 Record decisions.** `DECISIONS.md`: D1 to D6 as answered (ask Kim for the wording if you do not have it), plus
D7 to D10 below, each with date and source of the answer.

**T9 Label sanity by ear (CPU).** From the val-split prior render 24 notes: 6 each at pooled-label quantiles 10%,
50% and 90%, and 6 rejected for high excluded energy. File names carry the label value. Kim listens to see whether
the value tracks detuned, metallic or noisy character. This is the perceptual check the SPEC asked for.

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

- **D7** Restart on the reference recipe (A0 and A1) or run a twin of the live run? Recommendation: the twin, unless
  you want the flow clips on the H6 recipe, because the JEPA question needs only the aux flag to differ.
- **D8** Resume across a head on/off change: refuse loudly (recommended, minimal) or implement a warm start.
- **D9** Should `run_v3.sh` be repointed at `train_realistic_bass_v3.py`, after T3d shows what each trainer imports?
- **D10** Acceptance numbers for the calibration tables (valid rate on clean patches, false-harmonic rate).
- **D11** The aux weight `W` and the run length (hours or steps) for the next runs.

---

## 7. Report template

```
T0  RUN_DEFINITION.md path; command line; weight; flags missing or not passed (pasted table)
T1  clip dirs written, count of stems, scoring done or skipped and why
T2  commit <sha>; label_stats.json summary (pasted): pooled valid rate, per-window rates, constant-baseline MSE
T3  commit <sha>; six test files summary line (pasted); both smoke runs (pasted); D8/D9 findings
T4  probe table (pasted): R-squared with 95% interval for live EMA, untrained floor, twin if it exists
T5  the written commands, not run
T6  tables and plots written, no winner picked
T7  commit <sha>; sample log lines
T8  DECISIONS.md path
T9  clip dir and the quantile table
DEVIATIONS  none, or each with the reason and who approved it
```
