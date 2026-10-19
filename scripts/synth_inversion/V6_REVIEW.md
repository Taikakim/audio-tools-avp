# V6 review: the state of synth inversion, the V6 proposal, and how to implement it

**Owner:** KUANG (Claude Opus 5.5), for Kim. **Date:** October 19, 2026.
**Branch:** audio-tools-avp `feature/v6-ablation`, based on `origin/fix/inharmonicity-aux` at `a079a5f` (Antigravity's V4/V5
pipeline commit and `V6_DISCRIMINATORS_AND_ACOUSTIC_PROPOSAL.md`).
**Evidence tags:** **[V]** verified by reading code or running it here (CPU, laptop); **[T]** covered by a CPU test in
`test_v6.py`; **[I]** inferred, not measured; **[D]** needs the desktop (Surge, GPU, real stems).
Nothing here has run on Surge or a GPU. No result in this document is a listening or Surge measurement.

---

## 1. Summary

1. **The V4 components were a real step forward, but part of the V5 pipeline quietly turned them off.** The A-series
   (A1 inharmonicity aux head, A2 temporal attentive-statistics encoder, A3 both) improved held-out retrieval and ladder
   monotonicity beyond the evaluation's own noise (section 2.1). The V5 trainer reaches those components only when the launch command asks
   for them (`--flow_encoder conv`, `--inharm_weight 0`, `--ladder_every 0` are the defaults), and its bank holds no
   inharmonicity labels and no ladders [V]. Whether the Stage 4/5 runs used them is **unknown from the repository**:
   check each run's `run_meta.json` / training log first (section 6, step 0).
2. **A long-standing bug hid half the parameters from the JEPA in every V3-trainer run** (A-series included) [V][T]:
   the JEPA's parameter encoder took its sizes from the original `surge_spec` (20 continuous, 3 categorical) and the
   V3 trainer never passed its own (43 and 10). It silently ignored the oscillator octaves, oscillator 2 / 3 volumes,
   ring modulation, noise, filter 2, filter routing, FM routing and envelope modes. Antigravity's new shape check in
   `a079a5f` exposed it (6 tests failed on that commit). V4 / V5 trainers pass the right sizes. Fixed for new V3 runs;
   old checkpoints load as trained.
3. **The real-stem evaluation told the model the wrong note** [V][T]: `eval_real_stems_consensus_v5.py` normalized
   the pinned note with 12..96, every layout trains on 28..50. The model heard a note up to an octave above what it was
   told and could only reconcile that through the oscillator octave. This plausibly explains much of the "octave
   transposition drift" in section 1 of the V6 proposal [I]. Fixed.
4. **"Spurious" oscillators, filters and distortion are largely what the training prior teaches** [V]: 60 % of the V4 / V5
   prior draws are unconstrained random patches, where oscillator 2 is on 60 % of the time, oscillator 3 40 %,
   unison 35 %, feedback 35 %, filter routing uniform over 7 configurations, filter 2 always randomized. For an
   ambiguous sound the flow's posterior follows that prior. Discriminator vetoes treat the symptom.
5. **Distortion can be detected where it is audible and only there** (section 4). A label from the waveshaper setting
   is the wrong target. `ablation_v6.py` labels the measured effect of switching a feature off, measures whether
   audio can detect it, and simplifies inverted patches when a feature changes nothing (Occam test).
6. **Flow-DPO is built but has never run**; until this branch it only worked on V3 (92-d) flows. It now follows the
   flow's layout and can rank by a weighted metric sum.

---

## 2. Current state of the inversion code

### 2.1 V4 components (A-series), what the evidence supports

Reported in `V4_EXPERIMENT_REPORT.md` (one seed per arm, 12 000 steps, 512 held-out notes):

| | Baseline | A1 inharm aux | A2 temporal ASP | A3 both |
|---|---|---|---|---|
| held-out audio-to-preset retrieval (online) | 14.5 % | 20.1 % | 19.5 % | 21.9 % |
| flow ladder monotonicity (Spearman rho) | ~0.40 | 0.46 | 0.53 | 0.56 |

- The gain of each arm over the baseline (5 to 7 percentage points) is larger than the evaluation's own spread
  (about ±3.6 points at 95 %, as the report computes) [V: arithmetic]. The differences **between** A1, A2 and A3 are
  inside that spread, so "A3 is best" is not established; "each component helps" is plausible.
- One seed per arm, and the baseline is described as "pre-A1 v3" without a matched rerun: part of the gain may come
  from other changes made at the same time [I].
- All A-series JEPAs saw only 23 of 53 parameters (finding 2). Within the A-series the comparison is fair (same bug
  everywhere); absolute retrieval numbers would likely change with the fix [I].
- The temporal encoder and the inharmonicity head are the parts with evidence. DualMel, two-anchor objective and the
  resonance card are built and unit-tested on a stand-in, not trained or validated on Surge (`V4_COMPONENTS.md`).

### 2.2 V5 Stage 4 / 5 (`train_realistic_bass_v5.py`, 126-d codec)

- **Layout:** 60 stored parameters; osc 1 / 2 / 3 octaves became 7-class categoricals (126-d encoding). The codec and
  the spec digitize octaves with Surge's thresholds; rendering used plain rounding, which disagreed with the labels near
  a boundary (1.4 % of random-patch draws, computed) [V][T]. Fixed: the spec now uses the same thresholds.
- **Data:** live mode could not start: it used an unimported `RealisticBassPrior` (a V3-layout prior) [V]. Fixed
  (soundset prior). Bank mode works, but `render_bank_v4.py` stores zero inharmonicity labels (marked invalid), has
  an empty ladder dataset, and the trainer checked only the bank's format number [V]. Fixed: the bank can store real
  labels (`--inharm_config`) and records manifold md5s; the trainer refuses a bank that cannot serve the run.
- **Glitch redraws in the bank** reused one seed per MIDI note, so every glitching item on a note got the same
  replacement patch, and the stored MIDI note was the original, not the redrawn one [V]. Fixed (seed from the item's own
  vector; stored note = rendered note; source = "redraw").
- **Live and bank data differ** [V]: the trainer's own `sample_unconstrained_random_patch` (used for
  `--random_patch_frac` in live mode) turns the waveshaper on in 7 of 8 draws; the bank's prior in 1 of 10. Not changed;
  decide which one you want before comparing live and bank runs.
- **Validation** uses the soundset prior's val split, which is also 60 % fresh random patches [V]: "held-out presets"
  describes only 40 % of it.
- **Acoustic loss (Stage 5)** [V]: `1 - cos(JEPA predictor(params(x_hat_1)), z_a)`, applied at every flow time t with
  soft categoricals `softmax(x / 0.5)`. Two problems: at small t, `x_hat_1` is a blurred average, not a patch, so the
  loss pulls the velocity field toward acoustically "safe" averages; and soft categoricals are never the one-hots the
  JEPA was trained on. Added, off by default (Stage 5 reproduces exactly): `--acoustic_t_min`, `--acoustic_t_weight t`,
  `--acoustic_cat st` (straight-through one-hot) [T]. **The claim that Stage 5 "solved the muffled filter cutoff" has no
  measurement in the repository**; it needs the paired audio evaluation of section 6.

### 2.3 Real-stem evaluation (`eval_real_stems_consensus_v5.py`)

Fixed [V][T]: note range from the model's layout, out-of-range notes folded by octaves with the oscillators shifted
back; the second phrase note is the stem's own note an octave up when transcribed (pitch shifting only as fallback, it
adds phase-vocoder smearing the model never heard); per-candidate and winner oscillator-1 octaves recorded; Antigravity's
private scratch directory removed from `sys.path`; optional `--occam_tol`. Still true: the evaluation reranks by a
hand-weighted phrase score (MSS plus stereo and sub penalties) that has not been checked against listening.

### 2.4 Flow-DPO (`dpo_pairs_v5.py`, `train_flow_dpo_v5.py`, `flow_sampling_v5.py`, `audio_eval_v5.py`)

- Written October 13, 2026, CPU-tested, **never run on real pairs** [V].
- Was hard-wired to the V3 layout (codec, note range, render spec, prior) [V]. Now follows the flow's layout
  (`layout_v6.py`), so it runs on Stage 4 / 5 flows; `--metric "mss=1,modspec_dist=0.5"` ranks by a weighted sum and
  uses the same weighted render-noise floor as the margin [T].
- V6 step 1's composite `MSS + 0.5 D_contour + 1.0 D_harmonic`: `D_contour` exists only inside `inference_v3.py`
  (`compute_bin_energy_contour_loss`), `D_harmonic` does not exist in the code. Use metrics that passed the ladder gate
  (`mss`, `logmel_l1`) plus `modspec_dist` (the only one that sees detune) until the others are defined and gated.

---

## 3. The V6 proposal, section by section

| V6 item | Assessment | Recommendation |
|---|---|---|
| D1 distortion probe, label `ws > 0 or drive > 0.15` | Label from the setting; includes inaudible cases (soft shaper at low drive). Physics: on a steady periodic tone a waveshaped wave and another oscillator shape can give the same harmonics, so no detector can separate them there [I: standard nonlinear-systems result; verify per Surge config with the probe] | Label by audible effect; measure detectability per stratum; Occam test at inference (section 4) |
| "Immune to ring-modulation sidebands" | A hope, not a property: a learned probe uses whatever correlates | Measure with `ring_off` labels |
| D2 multi-oscillator, gate clamps osc 2 / 3 when P < 0.15 | Same label issue; threshold uncalibrated | Occam test does this exactly, no threshold to tune |
| D3 sub-oscillator, `osc_2_octave < 0` | Octaves are absolute per oscillator: osc 2 at -1 with osc 1 at -1 is not a sub | Label relative to osc 1, or by audible effect |
| D4 odd-only (square vs saw) | Ignores pulse width (only 50 % is odd-only) and asymmetric shapers (add even harmonics) | Drop as a gate; EOHR is fine as a descriptor |
| Lock osc 1 to ±1 octave | Right direction, half a fix: the flow predicts both the note (28..50) and osc 1's octave, so several pairs give the same pitch. That ambiguity, plus finding 3, is the octave drift [I] | Fix osc 1 at octave 0 and give the note to the flow as an input (section 5) |
| Detune ±0.2 semitones in 20 % of draws | Sensible. Log-mel distances are blind to it (October 13 finding); only `modspec_dist` sees it. `osc_2_pitch` already spans ±1 semitone in the spec; the ±0.2 mapping must be checked by render | Keep, score with `modspec_dist`, verify the mapping on Surge |
| Pitch head (128 classes + cents) | 22 of 128 classes ever occur; cent target is 0 except detune, so the fine head learns a constant; "frees z_a for timbre" is unsupported, it may as well make z_a pitch-dominated | Use the known note as a flow input instead |
| Flow-DPO first | Fine, after the evaluation fixes; check headroom before training | Section 6, step 4 |

---

## 4. Can a distortion discriminator be built?

Yes, for distortion that is audible; no, for the part that is not, and that part does not need detecting.

- **Ground truth by counterfactual render** (`ablation_v6.py labels`): render each patch twice with everything equal
  except the feature (waveshaper off, osc 2 off, ...), and once more unchanged for Surge's render-to-render noise.
  `delta = d(on, off)`, `audible = delta > k * noise` (default k = 2, plus an optional absolute floor). The summary
  says, per feature, how often the prior turns it on and how often it is then audible.
- **Detectability** (`ablation_v6.py probe`): an L2 logistic probe, 5-fold cross-validated, on clip-level log-mel
  statistics and, with `--jepa`, on the JEPA audio embedding V6 would put its heads on. AUROC (0.5 = chance) of the
  audible label, overall and per stratum (single oscillator vs multi / unison, static vs swept filter), next to the
  AUROC of the parameter label. Expected [I]: high AUROC where the waveshaper meets detuned or multiple oscillators
  (intermodulation off the harmonic comb) or a swept filter; near chance on a single static oscillator.
- **Use at inference: Occam test, not a learned veto** (`ablation_v6.py occam`, `eval_real_stems_consensus_v5.py
  --occam_tol`): for the chosen patch, try switching off each active feature (waveshaper, osc 2, osc 3, sub, unison,
  ring, noise, feedback, detune, drift); keep the removal that hurts the score least if the score rises by at most
  `max(tol_abs, tol_rel * score)`; repeat. Exact, needs no training, and directly addresses "distortion on clean plucks"
  and "extra oscillators on single-oscillator sounds". Cost: about 10 renders per round per item.
- **Only if the probe shows high AUROC in a stratum** is a learned head worth adding, and then as a training-time
  auxiliary loss on audible labels, not as an inference gate.

---

## 5. How to implement V6 correctly (design)

1. **Note as input, osc 1 octave fixed at 0.** Add the MIDI note (normalized) as a conditioning input to the flow
   (like `tau`), and remove `midi_note` and `osc_1_octave` from the predicted vector (osc 2 / 3 octaves become relative
   to osc 1, which they already are acoustically). At inference the note comes from the transcription (Muscriptor),
   as now. Out-of-range notes: fold by octaves (`layout_v6.fold_note`). This removes the note/octave ambiguity instead
   of narrowing it. Needs a new layout (v6 spec / codec), a new bank, a retrain.
2. **Simplicity-biased prior.** Replace the 60 % unconstrained random share by a sparse random patch: each optional
   feature on with probability about 0.2 to 0.3, filter routing Serial 1 in most draws, filter 2 off unless routed.
   Use `ablation_v6.py labels` on the old and new prior to check the share of features that are on but inaudible
   drops. Keep a small share (5 to 10 %) of fully random patches for coverage.
3. **Bring back the A-series components explicitly**: `--flow_encoder temporal_asp`, `--inharm_weight 100` with a
   labelled bank (`render_bank_v4.py build --inharm_config ...`), ladders in live mode or a ladder bank (none exists for
   V4 / V5 yet).
4. **Acoustic loss as an ablation arm, corrected:** `--lambda_acoustic L --acoustic_cat st --acoustic_t_min 0.5`
   against the same run without it, same seed and bank. Keep it only if the audio evaluation says so.
5. **Detune** as proposed, scored by `modspec_dist`.
6. **No discriminator gates.** Occam test at inference; probe-driven auxiliary heads only where detectable.
7. **Flow-DPO** on the chosen base, pairs ranked by a gated metric sum, beta sweep {30, 100, 300}.

---

## 6. Plan for the desktop (Antigravity runs, KUANG writes code)

Each step has a gate. Paths: `S=~/Projects/SAO/stable-audio-tools/scripts/synth_inversion`,
`B=/home/kim/sao_runs/real_manifold_v4_bass.npz`, `A=/home/kim/sao_runs/real_manifold_v4_all.npz`.

**Step 0: what did Stage 4 / 5 actually train with?** (5 minutes, no compute)
For each Stage 4 / 5 run directory: `run_meta.json` (`args`: `flow_encoder`, `inharm_weight`, `ladder_every`, `bank`,
`lambda_acoustic`, `acoustic_temp`) and the training log's `Aux ... (v=...)` field (v = 0.00 means the inharmonicity
head trained on nothing). Report the table to Kim. Gate: we know which A-series components each run had.

**Step 1: tests on the desktop.** `cd $S && python -m pytest test_v6.py test_v5.py test_eval_v4.py test_v4_components.py -q`
(laptop, October 19, 2026: 115 pass = 16 + 32 + 26 + 41, with the essentia / mido fakes test_v6 installs; run test_v6 first).

**Step 2: re-run the real-stem evaluation with the fixes.**
```
python eval_real_stems_consensus_v5.py --model A3 <A3 flow> --model S4 <stage 4 flow> --model S5 <stage 5 flow> \
    --out_dir ~/sao_runs/stems_v6_fixed --device cuda:0
python eval_real_stems_consensus_v5.py ... --occam_tol 0.02 --out_dir ~/sao_runs/stems_v6_occam
```
(A3 is a 48-parameter checkpoint: it needs the worktree at `aa734a8`, as noted on October 15; run it there or drop it.)
Read: `osc1_octave_candidates` / `osc1_octave_winner` per stem (does the drift persist with the right note?),
`notes[].second_note` (real vs pitch shift), `occam` (what was removed, score change). Kim listens to both packs.
Gate: if octave drift is gone, V6's octave lock is not urgent; if Occam removals are common and inaudible, the
discriminator gates are not needed.

**Step 3: ablation labels and detectability.**
```
python ablation_v6.py labels --layout v5 --bass_manifold $B --all_manifold $A --n 4096 --enrich waveshaper_off=0.5 \
    --metric mss --workers 14 --out ~/sao_runs/abl_v5
python ablation_v6.py probe --labels ~/sao_runs/abl_v5 --target waveshaper_off --jepa <stage 5 jepa_ema_latest.pt> --device cuda:0
python ablation_v6.py probe --labels ~/sao_runs/abl_v5 --target osc2_off
```
Read: `summary.json` (on share vs audible-when-on per feature; render-noise median) and `probe_*.json` (AUROC per
stratum). Gate: decide whether any learned head is worth it (AUROC well above 0.5 in a stratum that matters).

**Step 4: Flow-DPO headroom on the Stage 5 flow** (now layout-aware).
```
python dpo_pairs_v5.py --flow <S5 flow_ema_latest.pt> --bass_manifold $B --all_manifold $A --source synth \
    --n 4096 --k 8 --metrics mss logmel_l1 modspec_dist --workers 14 --out ~/sao_runs/dpo_pairs_s5 --device cuda:0
```
Gate: `best of 8` clearly below `candidate 0` and the spread above render noise in most conditions (the script prints
both). Then `train_flow_dpo_v5.py --metric "mss=1,modspec_dist=0.5" --beta 100` (and 30 / 300), evaluated with
`audio_eval_v5.py --bass_manifold $B --all_manifold $A --flow S5=... --flow S5DPO=...`.

**Step 5: V6 training (after KUANG writes the v6 layout of section 5).** Arms at small size, same bank and seed:
(a) V5 settings with A-series components on; (b) + note-as-input / osc 1 fixed; (c) + simplicity prior;
(d) + corrected acoustic loss. Two seeds for the arm that wins. Gate: paired audio metrics (`audio_eval_v5.py`) and
the real-stem pack, then a blind listening test (`listening_test_v5.py`).

---

## 7. What changed in this branch

| File | Change | Test |
|---|---|---|
| `layout_v6.py` (new) | flow width -> spec / codec (widths read from the codecs: 92 / 108 / 126); note range; octave folding and shift | [T] |
| `ablation_v6.py` (new) | counterfactual labels, Occam simplification, AUROC, cross-validated probe, strata; CLI labels / probe / occam | [T] fake renderer |
| `eval_real_stems_consensus_v5.py` | note range, octave folding, real second note, octave logging, `--occam_tol`, no scratch path | [T] helpers |
| `train_realistic_bass_v5.py`, `_v4.py` | live mode uses the soundset prior; real bank checks; V5: `--acoustic_cat`, `--acoustic_t_min`, `--acoustic_t_weight` (defaults = Stage 5) | [T] |
| `render_bank_v4.py`, `render_pool_v4.py` | `--inharm_config` labels, manifold md5s, `check_compatible` -> (hard, soft); redraw seed per item; stored note = rendered note | [T] checks |
| `surge_spec_v5.py` | octave class by Surge's thresholds (matches the codec labels) | [T] |
| `synth_jepa_model.py`, `synth_jepa_search_v3.py`, `train_realistic_bass_v3.py` | legacy truncation explicit; old checkpoints load as trained (warning); new V3 runs see all parameters; resume refuses to mix (`--legacy_jepa_params`) | [T] |
| `flow_sampling_v5.py`, `render_pool_v5.py`, `dpo_pairs_v5.py`, `train_flow_dpo_v5.py`, `audio_eval_v5.py` | layout-aware; DPO weighted metric; audio_eval soundset items with honest clusters | [T] + existing test_v5 |
| `test_eval_v4.py` | its tiny JEPA now uses the V3 sizes | |

Not done here: the v6 layout (note as input, osc 1 fixed), the simplicity prior, a ladder bank for V4 / V5. They change
the training data and need Kim's go after steps 0 to 3. (Done in section 8, after Kim's go.)

---

## 8. October 19, 2026, afternoon: the Steps 0 to 4 results, and V6 phase 1 code

Antigravity's `V6_SPECIFICATION.md` (commit `6cc2d74`) reports Steps 1 to 4 and specifies V6. Kim asked for the code
the specification needs. Reading of the results first, then the code.

### 8.1 What the results show, and what they do not

- **Step 0 is not reported.** Whether Stage 4 / 5 trained with the temporal encoder, the inharmonicity head
  (`v=` in the log) or ladders is still unknown. It is a five-minute check of `run_meta.json`; needed before saying
  what V5 "is".
- **Note-range fix (Step 1/2):** S5 real-stem wins rose from 3 to 11 of 24, which fits the diagnosis in section 1.
- **"Mel SSIM" in the results is a distance, so its direction is inverted** [V]. The evaluation stored
  `am.mel_ssim_dist` (= 1 - SSIM, lower is closer) as `mel_ssim`; the specification reads it as "higher is better".
  So in the DPO table β = 30 (0.719) has the *worst* structural match and S1 (0.649) the best; S5 0.682 -> β 100 0.669
  is a small improvement, β 300 (0.695) and β 30 (0.719) are worse. The landmark values (0.834, 0.867, 0.922) are
  distances too. Fixed: new summaries store `mel_ssim` = SSIM and `mel_ssim_dist`; the audition page converts old files.
  Pages built from old summaries show distances.
- **Occam (Step 2):** mean phrase score of S5 5.458 -> 5.394 with 128 features removed, but S5's wins fell from 11 to
  7. No interval is given; with 24 stems a 0.06 change can be one stem. `stem_stats_v6.py` gives the paired answer.
- **Ablation (Step 3)** agrees with section 4: the waveshaper is audible in 60 % of the draws where it is on and
  detectable from z_a (AUROC 0.84 to 0.86); a second oscillator is audible in only 20 % of the draws where it is on.
  The 0.587 is the AUROC for *audible* osc 2, so it says audible second oscillators are hard to detect from z_a; it is
  not itself a measurement of parameter-label training. Both points support Occam over learned gates.
- **Flow-DPO (Step 4):** the synthetic headroom is large (best of 8: MSS 3.43 vs 5.75 for the first draw; spread above
  render noise in 96 % of conditions), so the preference data carries signal. **On the real stems DPO is not shown to
  help:** all three arms have a higher mean phrase score than the base (5.545 / 5.705 / 5.748 vs 5.536) and fewer wins
  (3 / 2 / 1 vs 7). The per-stem gains (#04, #12, #14) are each picked from the best of three β values, so they are a
  selection, not an effect. The two numbers that decide it are missing: the held-out synthetic evaluation of the DPO
  checkpoints (`audio_eval_v5.py`, the step 4 gate) and the DPO validation pair accuracy (88.3 % is training).
- **`bass_energy_dist`** matches the low-band level (|log RMS below 600 Hz|); it does not reward "more body". It has
  not been through the ladder gate; run `ladder_metric_check_v5.py` with it before giving it weight.
- **Specification 3.4's training objective is not implementable as written:** MSS, D_bass and D_modspec need rendered
  audio, and Surge is not differentiable, so they cannot be training losses (as `V4_COMPONENTS.md` found for the V4
  brief). Only the JEPA term trains the flow; the audio metrics belong in DPO ranking and evaluation. The 0.2 weight on
  the inharmonicity loss also disagrees with section 3.3's `--inharm_weight 100` (the A-series value); use 100 unless
  an ablation says otherwise.
- **Smaller corrections:** the V6 width is 114, not 110 (126 - 1 - 7 - 4); "waveshapers 87 %" is the trainer's live
  random sampler, the bank prior used 10 % (Step 0 tells which a run saw).

### 8.2 Phase 1 code (this commit)

| File | What | Test |
|---|---|---|
| `surge_spec_v6.py` | 59 stored values: note at index 0 as the condition, osc 1 fixed at octave 0, osc 2 / 3 relative octaves (-2..+2), ±0.2 st fine pitch; `to_v5` / `from_v5`; renders through surge_spec_v5 | [T] |
| `param_codec_v6.py` | 114-d encoding without the note; `decode(x, note)`, `jepa_cont` (note back into the JEPA input) | [T] |
| `soundset_prior_v6.py` | simplicity prior: 70 % sparse (spec table), 20 % presets, 10 % random | [T] shares |
| `models_v6.py` | `NoteConditionedFlow` (note as sin / cos features into the parameter projection), `build_model_v6`, loader; `load_inverter_v4` hands v6 checkpoints over | [T] |
| `train_realistic_bass_v6.py` | v5 trainer on the v6 layout, prior, bank and note-conditioned flow; ladders refused; `--flow_encoder temporal_asp` default | [T] end to end on a fake bank |
| `render_bank_v6.py` | format-6 bank from the simplicity prior, inharmonicity labels, glitch redraws, `verify` | [T] |
| `check_v6_spec.py` | desktop check of the ±0.2 st fine pitch and the relative octaves on Surge | [T] measurement on synthetic tones |
| `stem_stats_v6.py` | paired real-stem statistics (bootstrap CI over stems, sign test) | [T] |
| `layout_v6.py`, `flow_sampling_v5.py`, `train_flow_dpo_v5.py`, `dpo_pairs_v5.py`, `audio_eval_v5.py`, `ablation_v6.py` | v6 support (note passed to the flow, v6 prior) | [T] |
| `eval_real_stems_consensus_v5.py` | v6 support; `--max_midi 60 --drop_polyphony` (specification 3.5 transcription filter, the reranked phrase too); SSIM fix | [T] |
| `build_interactive_model_eval_page.py` | SSIM column reads "higher = closer" for old and new files | |

### 8.3 Desktop order (each a gate for the next)

```
cd $S
python -m pytest test_v6.py test_v6_foundation.py test_v5.py test_eval_v4.py test_v4_components.py -q
python check_v6_spec.py --plugin "~/.vst3/Surge XT.vst3"                     # exit 0, or fix V5_PITCH_ST in surge_spec_v6
python stem_stats_v6.py ~/sao_runs/stems_v6_dpo_occam/consensus_eval_summary.json --ref S5_16.5h
python stem_stats_v6.py ~/sao_runs/stems_v6_dpo_occam/consensus_eval_summary.json --ref S5_16.5h --metric mel_ssim_dist
python audio_eval_v5.py --bass_manifold $B --all_manifold $A --n 256 --k 8 --workers 14 --device cuda:0 \
    --flow S5=<S5 flow_ema> --flow B100=~/sao_runs/dpo_s5_b100/flow_dpo_latest.pt --flow B300=~/sao_runs/dpo_s5_b300/flow_dpo_latest.pt \
    --metrics logmel_l1 mss modspec_dist bass_energy_dist --out ~/sao_runs/audio_eval_dpo     # the step 4 gate
python render_bank_v6.py build --out ~/sao_runs/bank_v6 --bass_manifold $B --all_manifold $A --n_items 100000 \
    --workers 14 --inharm_config inharm_config.json && python render_bank_v6.py verify --bank ~/sao_runs/bank_v6
python ablation_v6.py labels --layout v6 --bass_manifold $B --all_manifold $A --n 4096 --workers 14 --out ~/sao_runs/abl_v6
```
Gate for training: in `abl_v6/summary.json`, on_share x (1 - audible_share_of_on) per feature (on but inaudible) well
below the v5 values (osc 2: 0.75 x 0.80 = 0.60).

Arms (same bank, seed and step budget; add the S5 launch's model-size and `--cond_noise` flags):
```
python train_realistic_bass_v6.py --bank ~/sao_runs/bank_v6 --bass_manifold $B --all_manifold $A \
    --flow_encoder temporal_asp --ladder_every 0 --max_steps 12000 --batch_size 64 --seed 0 --device cuda:0 --out_dir ~/sao_runs/v6_A
  ... --inharm_weight 100 --inharm_config inharm_config.json --out_dir ~/sao_runs/v6_B
  ... (B) --lambda_acoustic <S5 value> --acoustic_cat st --acoustic_t_min 0.5 --out_dir ~/sao_runs/v6_C
```
Then `audio_eval_v5.py` on the three (held-out synthetic, paired), the real-stem evaluation with
`--max_midi 60 --drop_polyphony --occam_tol 0.02`, `stem_stats_v6.py`, and Kim's blind listening.

---

## 9. October 20, 2026: review of the V6.1 plan (`V6_1_PLAN.md`, commit `a29fefd`) and fixes

V6.1 adds a local polish of the reranked winner (cutoff, filter envelope, keytrack), a harmonic-parity penalty,
longer note crops ("gate normalization"), and guided Flow-DPO pairs (an extra, heuristically edited candidate per
condition). The plan says the pipeline has not run yet. Code-read and CPU checks:

**Bugs fixed** [V][T]
1. **Keytrack polish crashed:** it read `notes`, undefined in that function, so every stem whose phrase spans 7 or
   more semitones would have raised `NameError`. Now uses the stem's notes.
2. **Occam threw the polish away:** the polish edited the patch dict, Occam restarted from the unpolished vector, so
   any accepted removal discarded the polish. The polish now works on the stored vector (`polish_vector`), and Occam
   continues from it. Keytrack -100 % (raw 0.25) is not representable in the vector space and is dropped;
   values stay inside the spec's bounds.
3. **Parity measured the wrong audio:** `rfft(y, n=8192)` kept only the first 186 ms, so on a real stem it
   measured whatever came first (often silence or another note) at the root note's frequency. Over a whole phrase it
   is wrong anyway: a note an octave up puts its odd harmonics on the root's even ones (a square phrase with its
   octave note measures 137, the first note alone 0.006). Now: `parity_over_notes` (each transcribed note at its own
   f0, median) for stems and `phrase_parity` (first note) for training-phrase renders.
4. **"Gate normalization" read into the next note:** stretching every crop to 125 ms took the next note's onset into
   the conditioning clip whenever a note is shorter (a 16th at 140 BPM is 107 ms). The crop now stops at the next
   onset and the rest of the 125 ms is silence.
5. **Guided DPO candidate ignored the candidate:** it compared the target's centroid with fixed cutoff thresholds;
   the plan's rule is the centroid ERROR above 120 Hz. It now renders passive candidate 0 and uses its centroid
   (`guide_candidate_vector`, in the stored v6 vector instead of a patch round trip). The manifest is written before
   the first shard and reports headroom and how often the guided candidate wins (`guided_best_share`); each
   condition stores which rules fired (`guided_rules`).

**Assumptions now checked instead of assumed**
- "shape = 0.50 is a symmetric square": `check_v6_spec.py --shape_scan` renders osc 1 at nine shape values and
  reports parity; the pipeline passes the measured value as `--square_shape` to the pair generator and the evaluation.
- `compute_spectral_shoulder` does not measure the filter [V]: on a 55 Hz saw through a 2-pole low-pass, cutoffs of
  800 / 1 600 / 3 200 Hz give "shoulders" of 613 / 704 / 719 Hz. The -8 dB-below-passband rule mostly finds the saw's
  own 1/k roll-off. It is unused by the V6.1 scripts (the polish uses the centroid); do not use it until it passes a
  ladder check. It now averages the whole clip (Welch) instead of reading the first 93 ms.

**Method notes**
- The polish and Occam choose by the same phrase score that is reported, so the reported score of a polished system
  is optimistic against an unpolished one. Compare systems inside one run, all polished (the pipeline does), with
  `stem_stats_v6.py`, and let listening decide; `--no_polish` gives the unpolished reference.
- Guided pairs are honest as long as the pair is chosen by the rendered score (it is): a wrong heuristic just loses.
  If one rule wins most conditions, DPO learns that rule; `guided_rules` shows which.
- The plan's base numbers ("V6_C held-out MSS 6.07 / SSIM 0.284") come without the V6_A / B / C comparison or a
  reference; please add the `audio_eval_v5.py` report of A / B / C against S5 (and say whether 0.284 is SSIM or
  1 - SSIM).
- The `a_osc_N_mute = 0` lines added to `surge_spec_v5.apply_patch` are a sensible guard (an oscillator muted by an
  earlier state would stay muted).

**Run:** `run_v6_1_pipeline.sh` (replaces the unversioned `scratch/run_v6_dpo_pipeline.sh`): tests, Surge checks
with the shape scan, guided pairs, DPO β 100 / 300, **the held-out synthetic gate** (`audio_eval_v5.py`; the S5 DPO
skipped it), the real-stem evaluation of S5 / S5-DPO / V6_C / V6_C-DPO in one run, paired statistics, audition page.
