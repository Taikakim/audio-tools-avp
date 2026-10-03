# Handover 2026-10-02: real-preset bass prior, overnight trainer, watchdog (v2)

Branch `feature/surge-xt-neural-inversion`. This fixes the review of `9a6ea25`. All commands run
from `/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion`.

## What was wrong (v1) and what changed

| File | v1 problem | v2 |
|---|---|---|
| `extract_real_bass_manifold.py` | Guessed every unit conversion. Sub mix read width 2 (`param2`) and sync read sub mix (`param3`). Shape was discontinuous at 0, so 0 % went to raw 0 instead of 0.5. That produced the "58.8 % pure saw" spike. Filter type was `type/9` and waveshaper `type/5`, which relabelled HP/BP/comb presets as LP circuits. Drive was dB/24 (should be dB/15.36). Keytrack was `(kt+1)/2` (should be kt). Detune above 0.35 became 0. Out-of-range values became 0, which inflated the zero rates. Oscillator params were read even when osc 1 wasn't Classic. | Native → raw conversion goes through Surge's parameter ranges. Each range is **checked against the plugin's display strings** before use, and a mismatch stops the run. Enums map through `surge_spec`'s verified raw values. Non-LP filters, unknown waveshapers and non-Classic osc 1 are **skipped and counted**. Clipping is counted per parameter (`clip_frac`). Duplicate presets are removed, and a held-out split (10 %) is keyed by a vector hash. |
| `realistic_bass_prior.py` | Re-rolled each on/off switch independently, discarding the preset's on/off pattern. Clipping piled values up at 0.05 / 0.08 / 0.75. Jitter also moved the categorical parameters. | Keeps the base preset's categoricals and on/off pattern. Jitter is shaped like the preset covariance and **reflected** into the presets' range, so nothing piles up. Supports `split` (train/val/all) and `support_bounds()`. Refuses a v1 manifold. |
| `train_realistic_bass_overnight.py` | Fresh start crashed (`normalizer` used before it was created). No validation, only in-batch retrieval at 1/32 chance. Resume lost the optimiser state. Constant LR. Checkpoints written in place. The flow model took normalised mel and was unreadable by `load_inverter`. Unbounded checkpoint pile-up. | Fixed validation sets from held-out presets and from training presets, scored every 1000 steps: JEPA, flow, and retrieval over 512 notes. The **gap** between the two sets is the memorisation check. Exact resume: both optimisers, the normaliser and the best score. Atomic saves. WSD schedule (warmup → constant → decay over the last 20 % of the time budget). Non-finite steps are skipped, and the trainer exits 3 after 20 in a row. Exports `jepa_{best,latest}.pt` and `flow_{best,latest}.pt` in the formats the eval scripts read. Writes `run_meta.json`. Keeps the last 3 rolling checkpoints. |
| `watchdog_supervisor.py` | Halved LR and doubled batch on noise (5 rising 50-step averages ≈ once per 120 log lines). Rolled back to a `checkpoint_best.pt` that was never written. Sent SIGTERM first, so the bad state was saved. | Crash-only. Restarts on non-zero exit or a stalled log (20 min). Gives up after 5 restarts per hour. Never touches hyperparameters. |
| `synth_jepa_search.py` + eval scripts | Search box was always the uniform prior's `CONT_BOUNDS`, so it searched chorus/delay that the realistic model never saw. | `SynthJEPASearcher(bounds=...)`; `checkpoint_bounds(path)` reads the checkpoint's `prior_bounds`. Wired into `evaluate_holdout_audio`, `evaluate_200k_inversion`, `evaluate_envelope_guided_search` and `render_jepa_clips`. |

Tests: `python -m pytest -q test_synth_inversion.py` should report 39 passed (after round 2 below). The new tests are a
calibration accept/reject, an XML → vector → `apply_patch` raw-value round trip, skip/zeroing rules,
prior spike rates / bounds / no edge pile-up / categoricals copied, trainer fresh → resume → exports
→ search, refusal of v1 checkpoints, and watchdog restart/give-up.
**None of this has run against real Surge or on the GPU.** Step 1 below is the first real check.

## Round 2 (2026-10-03): everything else on the branch

| File | Problem | Fix |
|---|---|---|
| `train_synth_jepa.py --online` | Seeds were `min_seed + idx`, so every epoch re-rendered the **same 160k patches**. "Online" was just an uncached fixed dataset. | Seed = `min_seed + epoch·length + idx`. The DataLoader now uses spawn instead of fork. |
| `train_realistic_bass_overnight.py` | `1ddaf61` frees the main-process synth before the fork. `del` doesn't guarantee the plugin unloads. | Also uses spawn workers, which never inherit the main process's plugin or audio state. |
| `train_synth_jepa_139.py` + `surge_139_spec.py` | (a) Enum sizes were guessed (osc 7, filter 12, voices 8, LFO 5) and raw = class/(n−1). The five "deterministic" LFO classes actually swept the whole LFO list, including Noise, S&H, Step Seq and MSEG. (b) Names the plugin lacks were silently skipped. (c) Retrieval was the cosine between z_a and z_p, two different spaces, so it measured nothing, and "best" was chosen by it. (d) Same per-epoch seed replay. (e) Mute switches were treated as continuous, and all-muted renders trained as normalised silence. | Enum classes are read from the plugin (sweeping raw values and collecting display strings). LFO shapes are limited to Sine/Triangle/Square/Sawtooth and Audio Input is dropped. Missing names now fail loudly. Retrieval goes through the predictors over the whole validation set, and best is chosen by validation loss. Seeds advance per epoch. Mutes are categorical and silent renders are redrawn. Added `--lambda_sig` (default still 0.1), gradient clipping and atomic checkpoints that include the enum tables. **Old `jepa_139_online_v1` checkpoints won't load into the new model** (class counts changed). |
| `envelope_extractor.py` | The envelope-guided search's suggestions used linear maps for log-scaled controls: decay half-life ÷ 150 ms, and cutoff (centroid − 150 Hz) ÷ 3000. | Uses Surge's scales: envelope times as log2 seconds over [−8, 5], cutoff as semitones relative to 440 Hz over [−60, 70]. Decay uses the 20 dB time and release the 60 dB time. Values are clipped to the training box. |
| `audio_utils.py` | `save_patch` copied every preset into `~/Documents/Surge XT/Patches/AI Inversions/` **by default**, overwriting same-named files. `ExactGpuMel.to()` raised `NameError` (torch wasn't imported). | The copy is opt-in (`copy_to_user_dir=True`) and never overwrites (it uses `_2`, `_3`, …). `.to()` is fixed. |
| `benchmark_ground_truth_reconstruction.py` | The inverter's input was 0.8 s cut from a rolling phrase (6+ overlapping notes), which is out of distribution, and best-of-8 was scored against that chunk. | The input is one note rendered alone, as in training. The out-of-distribution warning fires for gates under 180 ms, and best-of-8 is labelled as peeking at the target. |
| `benchmark_multi_artist.py`, `warm_mel_inversion.py`, `benchmark_ground_truth_reconstruction.py` | A "prime" call, `process(array, 1024/sr, sr, 2)`, was an *effect* call at sample_rate = 0.023 Hz. Directories on Mantu were created at import time. There was a duplicate DAW copy under a second name. | Prime call removed (`init_synth` already primes). Directories are created in `main()`. A single DAW copy is made via `save_patch`. |
| `match_rolling_phrase.py` | Read only MIDI track 0, which in a type-1 file is often the tempo track. That means no notes, and DE "matched" a silent synth. Filter-index comment was wrong. | Tracks are merged, and it errors if there are no note-ons. Comment corrected. |
| `render_eval_clips.py` | Flow results were best-of-8 chosen against the target, unlabelled. | Labelled `selection: best_of_8_vs_target`. |

The 139-parameter run has its own gaps (not fixed, design decisions):
- The LFOs aren't routed to anything in a default patch, so their 60 parameters are inert.
- The MIDI note (24–96) and note length aren't in the parameter vector, so the model can't
  attribute pitch to anything.

## Known, not fixed (needs your ears or a decision)
- **FM depth may be inert in every dataset.** `surge_spec.init_synth` never sets the FM routing.
  If Surge's default patch has FM routing Off, `fm_depth` has had no audible effect in the 200k
  set or in the realistic prior. Check it with the command in step 0. If it's Off, fixing it changes
  the dataset distribution, so the fix needs a new dataset version, not a silent edit.
- The projected presets don't sound like the original presets. Only osc 1, filter 1 and the two
  envelopes are read (no osc 2/3, LFOs, modulation, filter 2 or FX). It's a prior over our 23-d
  space, not a preset library.

## Runnable blocks

### 0. Stop the v1 run and check FM routing
- **WHAT / WHY:** the v1 run trains on the mis-converted prior, and its watchdog changes the LR at
  random. Its checkpoints are kept, not deleted.
- **RUN:**
  ```
  pkill -f watchdog_supervisor.py; pkill -TERM -f train_realistic_bass_overnight.py
  /home/kim/Projects/synth_env/bin/python -c "import sys; sys.path.insert(0,'.'); from surge_spec import init_synth; p=init_synth(); print('FM routing:', p.parameters['a_fm_routing'].string_value if 'a_fm_routing' in p.parameters else 'no a_fm_routing parameter; list: '+str([k for k in p.parameters if 'fm' in k]))"
  ```
- **TAKES:** seconds.
- **VERIFY:** `pgrep -f train_realistic_bass_overnight.py` prints nothing.
- **REPORT BACK:** the `FM routing:` line.

### 1. Rebuild the manifold (v2)
- **RUN:**
  ```
  cd /home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion
  /home/kim/Projects/synth_env/bin/python extract_real_bass_manifold.py 2>&1 | tee /run/media/kim/Mantu/surge_200k_models/real_bass_manifold_v2.log
  ```
- **TAKES:** about a minute.
- **IF IT STOPS with "Native ranges do not match this Surge build":** the message lists each
  parameter with the measured range. FM depth's maximum (16 vs 24 dB) and keytrack's range (±100 %
  vs ±200 %) are the likely ones. Look at the list. If the measured values are plausible, rerun with
  `--use_measured_ranges`. If a display string "cannot be read", paste the message back.
- **VERIFY:** `real_bass_manifold.npz` is rewritten, and the log ends with `Kept N presets (M held out). Skipped:`
  followed by reasons and a per-parameter table.
- **REPORT BACK:** `tail -40 /run/media/kim/Mantu/surge_200k_models/real_bass_manifold_v2.log`.
  Check the skip counts (how many presets used non-LP filters or non-Classic osc 1) and the
  `clip%` column.
- **ROLLBACK:** the v1 file is overwritten. It's wrong anyway; restore it from git history only if
  you need to reproduce a v1 result (`git show 9a6ea25:` holds the extractor, not the data).

### 2. Launch the v2 overnight run
- **RUN:**
  ```
  cd /home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion
  setsid nohup bash run_overnight_realistic_bass.sh > /run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2.launch.log 2>&1 &
  ```
  Environment overrides: `OUT_DIR=…`, `PYTHON=…` (default `sat-venv`; it needs torch **and**
  pedalboard), `PURPOSE="…"`.
- **TAKES:** 8 h. Startup (validation renders plus 250 normaliser batches) takes about 1–2 min
  before the first `Step` line.
- **VERIFY:** in `/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2/`, `training.log` shows
  `Step` lines every 50 steps and a `VAL step` line every 1000 steps. Also check `run_meta.json`,
  `val.jsonl`, `jepa_best.pt`, `flow_best.pt`, and at most 3 `checkpoint_step_*.pt` files.
  `watchdog.log` should show a single `Launching` line.
- **REPORT BACK:** `grep "VAL step" .../overnight_realistic_bass_v2/training.log | tail -5; tail -3 .../watchdog.log`.
  What to read: held-out JEPA loss falling, and `gap` (held-out minus training presets) staying
  small. A growing gap means it's memorising the ~400 archetypes. Retrieval is now over 512 notes
  (chance 0.2 %), so it shouldn't sit near 100 % the way in-batch retrieval did.

### 3. Score it on audio (after the run)
- **RUN:**
  ```
  $PY evaluate_holdout_audio.py --jepa rb=/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2/jepa_best.pt \
      --ckpt rbflow=/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2/flow_best.pt --n 200
  ```
  `$PY` is a venv with torch **and** pedalboard. This scores on the uniform-prior h5 hold-out, which
  is outside this model's training distribution. Read it against `mean_patch` and as a generalisation
  number, not as the model's best case.
- **VERIFY / REPORT BACK:** paste the summary table that `holdout_audio_summary.json` prints.
