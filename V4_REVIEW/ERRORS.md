# Error register: A0/A1 launch code and the v4 design brief

Date: October 11, 2026. Reviewer: KUANG (Claude Sonnet 5.5) for Kim.
Reviewed: branch `fix/inharmonicity-aux`, tip `4dfc5ef` (agent's work), and `scripts/synth_inversion/DESIGN_BRIEF_V4.md`
in that commit. Read-only review; no GPU run, no change to the agent's branch.

Evidence tags: **[V]** verified (read in the code at `4dfc5ef`, or run here); **[I]** inferred from verified facts
(the inference is stated); **[M]** from memory, not verified.
Severity: **BLOCKER** = will break or invalidate A1 if not fixed first; **HIGH**; **MEDIUM**; **LOW**.
"Fix" lines are proposals for the implementing agent. Anything outside the SPEC.md R2 allow-list needs Kim's
go (rule R11).

Part A is code and launch problems and is time-critical (fix before A1 starts). Part B is errors in the v4 brief. Part C lists
what was checked and works. Part D is the pre-A1 gate: commands that turn the register into pass/fail.

---

## Part A. A0/A1 launch and code (fix before A1)

### A1. BLOCKER. The launcher and twin tool in the agent's tree are the superseded versions
- **Evidence [V].** `scripts/synth_inversion/run_v3_recipe.sh` lines 13 to 15 and 27 to 28 default to
  `EMBED_DIM/FLOW_DIM/FF_DIM = 768/768/2048` and `LR_JEPA/LR_FLOW = 3.0e-4/2.0e-4`. `scripts/synth_inversion/twin_command.py`
  `RECIPE` has the same values. Both are byte-identical to the `95c89fb` versions that were corrected in `375346b`
  (docs branch: `INHARMONICITY_FIX/provided_round2/`). The correct reference values are 512/512/1152 and 2.7e-4/1.8e-4
  (every reference run: H1, H6, v3 clean). The 768/768/2048 size is a different, untested configuration with 121.3 M
  parameters against 58.3 M (2.08 times larger).
- **Consequence.** `launch_experiments.sh` sets no size or LR override and labels both arms "True Recipe (batch 64)".
  A0 and A1 would train the untested scale-up under the recipe's name. `twin_command.py` would also report the wrong
  table.
- **Fix.** Replace both files with the copies at docs-branch tip (commands in Part D, step 1). Do not hand-edit.
  Alternative: set `EMBED_DIM=512 FLOW_DIM=512 FF_DIM=1152 LR_JEPA=2.7e-4 LR_FLOW=1.8e-4` in `launch_experiments.sh`.
  Recopying is better, because the twin tool is then right too.

### A2. BLOCKER. Three different definitions of A0/A1 exist and the smoke test covered the wrong one
- **Evidence [V].** (a) `launch_experiments.sh`: the recipe launcher, batch 64, `MAX_STEPS=26138`, `HOURS=10`.
  (b) `ROUND2_HANDOVER_FINAL.md` section 6: "smoke-tested" at batch 32, 8 h, constant `lr_jepa` 9e-5 (`lr_floor=1.0`), and
  the quoted log line says `batch 32` and `lr x1.000`. Those are the v3 parser defaults (batch 32, lr_jepa 9e-5,
  lr_floor 1.0, hours 8.0), not the recipe. (c) D7 and the recipe (ROUND2_HANDOVER.md section 3).
- **Also [V].** The same section describes T10 as an LR *reduction*. T10 was approved as a 3-point LR *check*
  (0.5, 1 and 2 times, constant LR, about 5 000 steps, GPU later). Nothing in the repo shows an LR test. Cutting
  the LR 3 times without a test contradicts D7.
- **Inference [I].** The smoke run was on parser defaults, so ladders, OT coupling, cond_noise and the MAE flow loss
  were probably never exercised on the v3 trainer in the A1 configuration (see A3).
- **Fix.** One definition: the command printed by `run_v3_recipe.sh` once A1 is fixed. Run the smoke test with that exact command
  (Part D, step 3) and keep its output in the run folder.

### A3. HIGH. The v3 trainer has no run evidence for the recipe's ladder path
- **Evidence [V].** Every committed launcher before `run_v3_recipe.sh` calls `train_realistic_bass_overnight.py` (the v1 stack),
  including `run_v3.sh`, `run_v3_clean*.sh`, `run_h6_*.sh`. The v3 trainer was, as far as the repo shows, only run by hand on
  parser defaults (no ladders) [I]. `LadderDataset.__getitem__` in `training_controls_v3.py` imported `realistic_bass_prior` (v1) until `4dfc5ef`,
  which changed it to `realistic_bass_prior_v3`.
- **Assessment [I].** The import change is probably a correct fix for a latent bug. It is also a behavior change
  outside the R2 allow-list (R2 lists `inharmonicity_target.py`, `inharmonicity_aux_head.py`, `synth_jepa_model.py`,
  `train_realistic_bass_v3.py`, `surge_spec_v3.py` for S6, plus new files), and so is the change to
  `synth_jepa_search_v3.py` (new `inharm_target` and `inharm_weight` arguments). Neither is reported as an R11 stop.
- **Fix.** Report both under R11 with a one-line reason each. Run the smoke test with `--ladder_every 4` as in the launcher
  (Part D, step 3), including at least one ladder step, one validation, one checkpoint and one resume.

### A4. HIGH. `validate_inharmonicity` adds a scikit-learn dependency that fails at the first validation
- **Evidence [V].** `train_realistic_bass_v3.py`: `from sklearn.metrics import r2_score` inside `validate_inharmonicity`.
  `sklearn` appears elsewhere in the folder only in `probe_inharmonicity.py`. The default `--val_interval_steps` is 1 000,
  so a missing package crashes at about step 1 000. The watchdog then restarts the trainer up to 5 times within 60 min and gives up (see A8).
- **Unknown [I].** Whether `sat-venv` has scikit-learn. The smoke test (batch 32, step 50) would not have reached a validation.
- **Fix.** Compute R-squared in numpy (`1 - ss_res / ss_tot`, NaN when `ss_tot == 0`). No new dependency.

### A5. HIGH. `label_stats.json` is read through a relative path and is not tied to the extractor config
- **Evidence [V].** `open("scripts/synth_inversion/label_stats.json")` at startup (head start, A1 only) and at every
  validation. It works only when the working directory is the repo root. The file comes from one T2 run: it has no config id.
- **Consequence [I].** A different working directory fails at startup (visible). A changed extractor config (see A7) leaves the
  file stale and silently wrong: both the head start and the validation `baseline_mse` would be off.
- **Fix.** Resolve the path from `os.path.dirname(os.path.abspath(__file__))`. Store `cfg.config_id()` in the file and assert it
  equals the running config's id. Write the train mean used into `run_meta.json`.

### A6. MEDIUM. A0 and A1 do not start from the same seed, and could not even with `--seed`
- **Evidence [V].** `launch_experiments.sh` passes no `SEED`. The launcher supports `SEED`, and T3f added `--seed`.
- **Evidence [V, run here at small scale].** After building `SynthJEPA` with the same `torch.manual_seed`, the next
  random draw differs between head OFF and head ON (`[0.867, 0.500, 0.547]` against `[0.729, 0.178, 0.307]`), because the head
  draws from the global RNG. The flow is built right after the JEPA (`train_realistic_bass_v3.py`, `build_model` after
  `SynthJEPA(...)`), so with a shared seed A0's and A1's flows still start from different weights.
- **Fix.** Pass the same `SEED` to both arms. Call `torch.manual_seed(args.seed)` again immediately before the flow is built and again
  before the DataLoaders are created (or build the flow first). State in the report that one seed per arm cannot separate the aux effect
  from run-to-run noise.

### A7. HIGH. The 100-peak cap is hit often and the exposure decision is needed before A1
- **Evidence [V].** `label_stats.json`: `max_peaks_100_cap_share` is 0.220 (train split, line 38) and 0.166 (val split, line 3370).
  `ROUND2_HANDOVER_FINAL.md` says the T6 grid shows 1.00 on clean raw patches (`t6_sweep.log` is not in the repo, so unverified).
- **Why it matters [I].** Per the Essentia semantics verified earlier, a capped window keeps only the lowest 100 peaks,
  so its excluded-energy test sees a truncated spectrum. The condition I set (expose `max_peaks` if the capped share is above a few percent)
  is met.
- **Decision needed from Kim before A1.** Changing the extractor after A1 starts changes the labels mid-experiment.
  (1) Expose `max_peaks` in `InharmonicityConfig`, choose its value from a re-run of T2/T6, bump the extractor version and
  regenerate `label_stats.json` and the 95th-percentile file. (2) Keep 100 and record the limitation in the report. I recommend (1)
  (CPU time only). The assumption that would change this: if the T6 re-run shows validity and class separation unchanged at a higher cap, (2) costs nothing.

### A8. MEDIUM. D8 "refuse loudly" is retried by the watchdog
- **Evidence [V].** `train_realistic_bass_v3.py`: `print("Cannot resume across a head on/off change (D8)."); sys.exit(2)`.
  `watchdog_supervisor.py` restarts the trainer on any non-zero exit, up to `--max_restarts 5` within 60 min, then returns 1. The message
  goes to `run.log`, not the terminal.
- **Consequence [I].** An accidental head-state mismatch shows as six identical launches and "Giving up". The same applies to the exit 2 for a
  missing `--inharm_config`.
- **Fix (within the allow-list).** In the trainer, write `REFUSED.txt` to `out_dir` and print to stderr before exiting. Optional (outside the
  allow-list, needs Kim): make the watchdog treat exit 2 as permanent. Preferred: have `run_v3_recipe.sh` refuse before starting the watchdog.

### A9. MEDIUM. A weight-0 arm still imports Essentia (T3b not done)
- **Evidence [V].** `inharmonicity_target.py` lines 5 to 6 import `essentia` at module level, and the trainer imports the module at its top.
  A0 therefore needs Essentia at import. Fine where it is installed (the agent's machine). It blocks A0 anywhere else (for example LUMI, unknown) and `--help`.
- **Fix.** Lazy-import Essentia inside the extractor constructor, or accept and note it in the report.

### A10. MEDIUM. A1's weight W = 100 does not come from the D11 method
- **Evidence [V].** `DECISIONS.md` D4 records `inharm_weight = 100.0`. `launch_experiments.sh` uses it for A1. D11 says W comes from a measured
  gradient ratio at the fitted state (ratio times |g_jepa| / |g_aux|), with 0.02 to 0.05 probe-like, 0.1 to 0.3 likely useful and 1 or more
  an upper bracket.
- **Observation [V, from Kim's report].** In the live run the aux loss is 0.01 to 0.05, so at W = 100 it adds 1 to 5 against a JEPA loss of
  about 0.4. Whether that swamps the JEPA gradient was not measured.
- **Fix.** Run the gradient-ratio measurement on the copied live checkpoint, or record W = 100 as an explicit exception with a reason and a
  planned second weight.

### A11. LOW. Time cap against step count
- **Evidence [V].** `MAX_STEPS=26138` (one third of 78 416), `HOURS=10`. The LR decay phase starts at 70% of the steps (step 18 300).
- **Inference [I].** At the 4 100 steps/h estimate for the 58 M size, 26 138 steps take about 6.4 h. At the 121 M size (A1) at roughly half that
  speed, 10 h might stop before the decay finishes. Neither speed is measured for this trainer.
- **Fix.** After A1, measure steps/h over the first 200 steps and set `HOURS` with margin.

### A12. LOW. `run_v3.sh`: the SUPERSEDED line is above the shebang
- **Evidence [V].** The diff inserts `# SUPERSEDED by run_v3_recipe.sh` as line 1, so `#!/usr/bin/env bash` is on line 2 and is a comment.
- **Fix.** Insert after the shebang.

### A13. LOW. Module docstring and hard-coded paths
- `train_realistic_bass_v3.py`: imports sit above the docstring, so `__doc__` is empty (since round 1). [V]
- `launch_experiments.sh` and the trainer defaults contain drive paths (`/run/media/kim/Mantu/...`, `/home/kim/Projects/SAO`). [V]
  Acceptable in the agent's private tree, but they should not go into any shared commit.

### A14. Process: decision records need provenance
- **Evidence [V].** `DECISIONS.md` D7 reads "Restart both arms on the reference recipe (A0 and A1). The live run remains a pilot." Kim decided "a twin of
  the live run", with a rule to revisit it after the T0 table. The agent's wording is the outcome of that rule. Whether the T0 table was produced and the rule
  applied is not visible in the repo.
- **Evidence [V].** D1 to D6 all say "Source: Kim". On October 10 Kim said they had answered some and delegated others ("I don't know where we could
  procure good threshold values"). D3 (0.15) and D4 (100) are recorded as Kim's decisions.
- **Fix.** Add a column `decided_by`: Kim, or "agent, best judgement under Kim's delegation". Add the T0 table and `RUN_DEFINITION.md` links to D7.
  Kim confirms the GPU go for A0/A1.

---

## Part B. Errors and gaps in `DESIGN_BRIEF_V4.md`

Section numbers refer to the brief. Review ID `B` entries are also the amendment list for the brief.

### B1. The headline losses cannot be trained on (sections 2.1 to 2.4, 4)
`L_anchor`, `L_psycho`, `L_reso` and `L_harm_ratio` compare rendered audio, and Surge XT is not differentiable. The v3 loop trains on
JEPA latent losses and flow matching in parameter space (`train_realistic_bass_v3.py` lines 767 to 796: `lp`, `la`, ordinal or SIGReg,
`i_loss`, `flow_loss`); renders happen only in the DataLoader workers to make (audio, parameters) pairs and in validation. [V]
Admissible uses: inference-time reranking and refinement (exists: `refine_v3.py`, `inference_v3.predict_and_rerank_candidates`), evaluation
metrics, a rendered-reward fine-tuning stage, or a neural surrogate renderer. The brief names none. [V for the loop, I for the list]

### B2. The sim-to-real FX gap is not separated from "sluggish decay" (section 1, modes 1 and 2)
Training renders have FX off (`surge_spec_v3.py:48`, `FX_RENDER_ENABLED = False`). 9 of the 24 catalog stems have reverb, delay, chorus, phaser,
flanger, saturation, distortion or "room" in the filename (`invert_stem_collection_v3_cpu.py:46` onward). A room tail or delay looks like a long
decay and no FX-off patch reproduces it. The brief has no per-parameter held-out error that would show whether the encoder or the domain gap is
the cause. [V for the facts, I for the confound]

### B3. The "double penalty" is attributed to a loss that is not in the pipeline (section 1, mode 2)
The double penalty is a property of audio-space spectral distances. The flow is trained in parameter space. Where hedging comes from here must be
located: the refine objective (`mss_ref + 10 * (1 - env cos)`, mean-seeking), averaging or point estimates at inference, the prior, or input resolution (B4). [I]

### B4. The input spectrogram limits resonance and attack resolution (sections 1, 3)
`MEL_KW = n_mels 128, n_fft 1024, hop_length 441, fmin 20, fmax 16 000` (`audio_utils.py:15`). The bin width is 43 Hz at 44.1 kHz.
Bass fundamentals are 41.2 to 146.8 Hz (MIDI 28 to 50), so 1 to 3.4 bins; harmonics are unresolved and a resonance narrower than about 90 Hz cannot
be told from a wider one. A 30 ms attack spans about 3 frames. Changing the pooling does not fix this. A long-window branch and a short-window attack branch
are missing from the brief. [V for the parameters, I for the consequence]
**Addendum (October 11, 2026, measured):** with torchaudio's Slaney filterbank (which should match, but is not, `ExactGpuMel`), 28 of the 128 bands are fed by at most one FFT bin, so the lowest bands are
not independent measurements, and the first band with two or more bins starts near 100 Hz. A short branch of `n_fft` 512 with 64 bands from 20 Hz would have 13 such bands. The layouts in
`frontend_v4.py` (32 bands from 100 Hz at `n_fft` 512; 96 bands from 30 Hz at `n_fft` 4096) have none. [R]

### B5. The section 3.1 sketch uses the wrong time axis
At a 10 ms hop, 81 frames cover 0.8 s: frames 0 to 8 are 0 to 80 ms, and frames 25 to 80 are 250 to 800 ms (the brief says 0 to 40 ms and 200 to 800 ms).
After four stride-2 convolutions the time axis has 6 positions (81, 41, 21, 11, 6), so the indices do not exist after the backbone. `nn.TransformerEncoderLayer`
defaults to `batch_first=False`. [V for the pipeline (`models_v3.py:13-38`), I for the arithmetic]

### B6. "Cannot perceive d(Spectrum)/dt" is overstated, and the JEPA encoder is a separate network (sections 1, 3)
The flow's own `AudioEncoder` (conv plus `AdaptiveAvgPool2d((1,1))`, `models_v3.py:13-38`, used at `:146` and `:170-174`) does lose *when* things
happen. 3x3 convolutions can still detect local slopes, so the pooled vector carries a decay amount. The Synth-JEPA audio encoder is a different
module with positional tokens and a CLS (`synth_jepa_model.py:36-91`), and the flow does not use its `za`. [V] The brief never mentions the JEPA.
**Addendum (October 11, 2026, measured on a stand-in synth, not Surge):** on a toy task the pooled v3 encoder reaches held-out R-squared 0.96 on amplitude decay and 0.89 on filter-envelope decay, so "cannot perceive" is
too strong. Temporal encoders add about 0.02 mean R-squared, mostly on filter-envelope decay (0.89 to 0.95), and nothing on attack time (0.956 to 0.962), for 7 times the encoder FLOPs. Branch `feature/v4-components`,
`scripts/synth_inversion/V4_COMPONENTS.md`. The research report's "one-module swap" for attentive pooling does not hold for this encoder either: its time axis has 6 positions after the strides. [R]

### B7. Enveloped sync and FM are modulation routings, not parameters (section 2.5)
v3 renders through Pedalboard `plugin.parameters` (`surge_spec_v3.py:438-482`) and has no routing setter. [V] A surgepy wrapper with
`set_mod_routing` exists (`surge_synth.py`). So the feature needs a backend change in the DataLoader workers; every v3 number, `render_baseline_hashes.json`
and checkpoint comparison would no longer be same-renderer. Whether the Surge VST3 exposes routings to a host is not verified here. [V, I]

### B8. Filter parameters already exist, and the class counts disagree (section 2.5)
v3 already has `filter_balance`, `filter_configuration`, `filter_2_type`, `filter_2_cutoff`, `filter_2_resonance`, `filter_2_keytrack` and
`filter_2_feg_amount` (`surge_spec_v3.py:82-91`). `filter_configuration` has 7 classes in v3, 8 in `surge_v4_registry.py` and "Series = 0,
Parallel = 1" in the brief. The installed Surge version decides. [V]

### B9. The harmonic-ratio loss targets a parameter v3 does not have (section 2.4)
The v3 vector has `osc_1/2/3_octave` but no oscillator pitch or cents parameter. Octaves are already discrete. Post-hoc snapping to {2:1, 1:1, 1:2, 1:3}
is already prototyped in `render_muscriptor_top3_harmonic.py`, and the brief reports no result from it. Hard snapping removes intentional detune and beating,
and the set omits 3:2. The aux inharmonicity label provides the "unless indicated by inharmonic energy" signal the brief wants. [V]

### B10. New parameters have no prior and break every checkpoint (section 2.5)
v3's prior draws sync in 30% of patches (U(0, 0.40)) and FM in 40% (U(0, 0.45)); the 122-preset manifold has no values for modulation depths. The brief gives
no distribution, probability or range. More parameters driven by the FEG (cutoff, sync, FM) also raise ambiguity. Flow and JEPA parameter-encoder shapes change,
so no checkpoint resumes. [V for v3 prior (`surge_spec_v3.py:136-137`), I for the rest]

### B11. Two different "v4" parameter spaces exist
`surge_v4_registry.py` defines 72 parameters over 3 oscillators with one modulation routing (`feg_to_pw`); the brief extends the 48-parameter v3 vector. Code docstrings
cite "v4 Spec Sections 1 to 3" (surgepy renderer, registry, activity probe; commit `c37a727`, October 7, 2026), a document that is not in the repo. [V]

### B12. The prominence method (section 2.3) does not measure resonance on harmonic tones
A 400 to 600 Hz median window spans about 3 to 14 harmonics for f0 = 147 to 41 Hz. Most bins are valleys between harmonics, so the median tracks
the valley floor and P(f) measures harmonic-to-valley contrast. Also: the fixed 800 Hz floor is arbitrary for bass resonances; FWHM is unresolvable when the
bandwidth is below the harmonic spacing (Q 10 at 1.5 kHz is 150 Hz, about one spacing at 147 Hz); "top 2 peaks = Filter 1 and 2" has no defined matching
(filter identity is not peak order, and there are 7 routing classes). Measuring on harmonic amplitudes at k times f0 with a smooth envelope would avoid the first problem. [I, arithmetic checked]

### B13. The two-anchor claim is too strong (section 2.1)
Two windows fix two levels; they force the decay only if the envelope is exponential and the sustain and amount parameters are separately observable. `refine_v3.py` states that
cutoff, FEG amount, keytrack and drive trade off. MuScriptor onset error can exceed the 40 ms window, and the weights (4, 2, 1) are unablated. [I]

### B14. CDPAM is unsupported for this use (section 2.2)
The sources I found describe human judgments on audio perturbations and speech synthesis and enhancement tasks. I found no evidence for patch-to-patch bass timbre. Licence, sample rate
and training data were not in what I retrieved. The repo already has the gate: `timbre_alignment_check.py` (21 listening-test datasets); the current best distance reaches
Spearman 0.473 (log-mel L1), against 0.470 (paper MSS) and 0.437 (our MSS). [V for the repo, M for CDPAM details]
Check the licence with Legal before vendoring anything.

### B15. The acceptance criteria are not measurable as written (section 5)
(1) 5% centroid error at 40 ms depends on onset alignment and FX, with no noise floor. (2) "Ground truth" H on real stems is measured on the stem itself, from 2 stems, with a metric not yet
validated. (3) "Zero buzz" is subjective; the aux inharmonicity extractor could supply an objective proxy. (4) "Match v3's MSS" conflicts with a 4 times attack weight and MSS agrees with
humans only about 0.44 to 0.47. The flow is stochastic and there are no seeds or paired comparisons. [I]

### B16. The baseline is not defined (section 1)
"110M JEPA parameters" matches the 768/768/2048 scale-up, not the 53.0 M reference JEPA. Comparisons must be against v3 plus rerank plus refine (`inference_v3.py`, `refine_v3.py`), not against raw
flow samples, and the checkpoint size must be named. [I]

---

## Part C. Checked and working

- Torch-level tests on the agent's tip: `test_inharmonicity_loss.py` (10), `test_inharmonicity_model_integration.py` (6), `test_inharmonicity_pooled_schema.py` (3): **19 passed**
  (CPU venv, stand-in Essentia; the Essentia-dependent and Surge tests were not run here). [V]
- T3g head start works: with `bias = log(exp(0.0080) - 1) = -4.8243`, the head output on N(0, 1) inputs has mean 0.0068 and std 0.0013
  (small model, 64-dim; the full model not run). [V]
- T3f: seeds `torch`, `numpy`, `random` and workers (`worker_info.seed`); the unconditional reseeding in `_worker_init` is an improvement for any run. [V by reading]
- D8: the check compares checkpoint head keys with `inharm_weight > 0` and loads strictly. [V by reading]
- `i_loss` is now a tensor in validation and the weight-0 training path. [V by reading]
- `validate_inharmonicity` logic: masked MSE, valid-only R-squared, split-to-stats mapping ("trainpresets" uses "train", "heldout" uses "val"). [V by reading]
  Its baseline for the held-out set is the validation sample's own pooled mean (an oracle constant), which is a stricter baseline than the train mean.

---

## Part D. Pre-A1 gate (commands for the implementing agent)

Run from the repo root on the training machine. Paste the last 15 lines of each (rule R4).

1. **Refresh the launcher and twin tool (fixes A1).**
   ```
   git fetch origin +refs/heads/docs/inharmonicity-spec:refs/remotes/origin/docs/inharmonicity-spec
   git show origin/docs/inharmonicity-spec:INHARMONICITY_FIX/provided_round2/run_v3_recipe.sh > scripts/synth_inversion/run_v3_recipe.sh
   git show origin/docs/inharmonicity-spec:INHARMONICITY_FIX/provided_round2/twin_command.py  > scripts/synth_inversion/twin_command.py
   chmod +x scripts/synth_inversion/run_v3_recipe.sh
   ```
   Files must keep LF endings.
2. **Dry run for both arms.** Expect `--embed_dim 512 --flow_dim 512 --ff_dim 1152 --lr_jepa 2.7e-4 --lr_flow 1.8e-4 --batch_size 64`.
   ```
   OUT_DIR=/tmp/a1_dry MAX_STEPS=26138 HOURS=10 INHARM_WEIGHT=100 INHARM_CONFIG=scripts/synth_inversion/inharm_config.json SEED=1 DRY_RUN=1 \
     bash scripts/synth_inversion/run_v3_recipe.sh
   ```
3. **Smoke test with the real flags (A2, A3, A4).** Take the command printed in step 2 and append (argparse keeps the last occurrence):
   `--out_dir /tmp/a1_smoke --max_steps 60 --val_interval_steps 20 --val_size 64 --hours 0.2`. Pass when the log shows a ladder step, a `VAL` line that includes
   `AUX held-out MSE ... R^2`, and a checkpoint. Then run the same command again (it must resume in place), then once with `--inharm_weight 0` in the same out_dir
   (it must refuse, D8).
4. **Twin table.** `python scripts/synth_inversion/twin_command.py /tmp/a1_smoke --out-dir /tmp/twin_check`. Pass when all 32 recipe rows match.
5. **Dependency.** `python -c "import sklearn"` in `sat-venv`; otherwise apply the A4 fix.
6. **Config binding (A5).** Confirm `label_stats.json` was produced with the config in `inharm_config.json`, and that the exposure decision for A7 has been taken.
7. **Seeds (A6).** Same `SEED` in both arms; the flow-build reseed is in.
8. **Record.** `RUN_DEFINITION.md` for A0 and A1 with the final command, the commit hash and the T0 table, and update D7 and D4 provenance (A14).
