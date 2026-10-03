# Surge XT Neural Synth Inversion Toolchain

Neural inversion of synthesizer patches: from an isolated bass note (or stem) to a **Surge XT 1.3.4** patch
that re-creates it, plus a closed-loop Differential Evolution (DE) search as a reference.

> **v2 (2026-10-01 review).** The v1 reports compared models on numbers that could not show what they claimed;
> see `docs/synth_inversion_reports/WALKTHROUGH.md` (status banner) for which results are invalidated.
> What changed: categoricals learned as classes; validation comparable across model types; an audio-domain
> held-out comparison (`evaluate_holdout_audio.py`); a DE search that no longer inherits the ResMLP's settings;
> targets resampled/onset-aligned with out-of-distribution flags; one shared scoring metric; enum raw values
> verified against the plugin; patch export no longer mislabelled as `.vstpreset`. v1 checkpoints still load.

---

## 1. Architecture & Capabilities

- **23-dimensional sound design space** (single source: `surge_spec.py`):
  - **Oscillator**: Shape (Saw <-> Morph <-> Pulse), Pulse Width, Sub-oscillator mix, Hard Sync, FM depth, Unison (1 or 2 voices) + detune.
  - **Filter**: 10 lowpass circuits (`LP 12 dB`, `LP 24 dB`, `LP Legacy Ladder`, `LP Vintage Ladder`, `LP OB-Xd 12 dB`, `LP OB-Xd 24 dB`, `LP K35`, `LP Diode Ladder`, `LP Cutoff Warp`, `LP Res Warp`), cutoff, resonance, keytracking.
  - **Envelopes**: filter EG amount, decay, sustain; amp EG decay, sustain, release. (Attack and the filter-EG release are left at Surge's defaults; v1's README claimed a "FEG_release >= AMP_release" coupling that was never implemented.)
  - **Distortion**: 6 waveshaper types (`Off`, `Soft`, `Hard`, `Asymmetric`, `Sine`, `Fuzz`) with drive.
  - **FX**: chorus mix, delay mix and feedback.
  - Three parameters are **categorical** (filter circuit, unison, waveshaper). The h5 stores them as `class/(n-1)` for compatibility; the v2 models learn them as classes (`param_codec.py`).

- **Models** (`models.py`):
  - `ResMLPInverter`: CNN mel encoder + residual MLP, 512-dim, 6 blocks. Point estimate: sigmoid head for continuous parameters, one cross-entropy head per categorical. Inference < 2 ms.
  - `FlowMatchingResMLP`: conditional flow matching (OT-CFM path, Euler sampler), same backbone. Learns a distribution over patches: continuous params + one-hot slots per categorical. Time embedding scales t by 1000.
  - `load_inverter(path)` rebuilds whichever architecture/encoding a checkpoint was trained as (v1 or v2).

- **Optimizers** (`optimizer.py`, single source):
  - **Muon** for hidden 2D matrices (Newton-Schulz orthogonalisation, aspect-ratio scaling) with a **radial brake** (keeps 85 % of any Frobenius-norm growth per step). v1 called this "NorMuon"; it never had NorMuon's per-neuron normalisation, so it is now named for what it is (`NorMuon` remains an alias, and `--opt_family normuon_sf` keeps its name for old scripts).
  - **Schedule-Free AdamW** for the input projection, output heads, conv kernels, norms, biases and embeddings (100 warmup steps; train/eval iterate swap, checkpoints hold the averaged iterate).
  - **Decoupled AdamW baseline** with cosine annealing (`--opt_family adamw`).

---

## 2. Directory Layout & Scripts

```
scripts/synth_inversion/
|-- surge_spec.py              # Parameter layout, frozen patch sampler, Surge mapping + verification, render
|-- param_codec.py             # Ordinal storage <-> class/one-hot model space for categoricals
|-- audio_utils.py             # Mel front-end, target preparation (resample/trim/OOD flags), THE scoring metric, patch export
|-- models.py                  # ResMLP + flow-matching inverters, checkpoint loader (v1 + v2)
|-- optimizer.py               # Muon (+ radial brake) + Schedule-Free AdamW
|-- inference.py               # Model -> patch helpers shared by the evaluation scripts
|-- generate_bass_dataset.py   # Multi-process (pedalboard VST3) dataset generator
|-- train_bracket.py           # Training harness (domain-weighted losses, comparable validation, telemetry)
|-- evaluate_holdout_audio.py  # Audio-domain model comparison on held-out notes (+ oracle and trivial baselines)
|-- evaluate_200k_inversion.py # Real-stem inversion, render, score, export
|-- match_untitled_note.py     # Single-note match: neural inversion vs DE reference search
|-- synth_jepa_model.py        # Synth-JEPA (arXiv:2609.31024): audio/param encoders, predictors, SIGReg
|-- train_synth_jepa.py        # Synth-JEPA training (Welford pre-pass, WSD schedule, retrieval-accuracy validation)
|-- synth_jepa_search.py       # Renderer-free JADE + Adam search over the training support; checkpoint loader
|-- extract_real_bass_manifold.py  # Real Surge bass presets (.fxp) -> 23-d vectors, plugin-calibrated (v2)
|-- realistic_bass_prior.py    # Patch prior sampled around those presets (train / held-out split)
|-- train_realistic_bass_overnight.py  # Online joint Synth-JEPA + flow training on that prior
|-- watchdog_supervisor.py     # Crash-only restarter for the overnight trainer
|-- test_synth_inversion.py    # CPU tests with a fake Surge (no plugin or GPU needed)
`-- run_overnight_suite.sh     # 200-epoch Muon+SF run, then 100-epoch AdamW baseline
```

---

## 3. Running it

All commands run **from this directory** (`scripts/synth_inversion/`). The scripts import each other by
module name, so the working directory matters.

Venvs (from `run_overnight_suite.sh`): training uses `/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python`
(torch, ROCm); dataset generation uses `/home/kim/Projects/synth_env/bin/python` (pedalboard, librosa, h5py; no
torch needed). The **evaluation scripts need torch AND pedalboard in the same venv**; check one with
`$PY -c "import torch, pedalboard, librosa, h5py, soundfile, scipy"`.

All Surge-facing scripts first check the hard-coded enum raw values against the plugin's display strings and
stop with a list of mismatches if a Surge build has moved them. Set `SURGE_SKIP_MAPPING_CHECK=1` only after
confirming the differences are cosmetic.

**Tests (no Surge, no GPU):**
```
python -m pytest -q test_synth_inversion.py      # ~20 s; expect "33 passed"
```

**Generate a dataset** (v2 also stores `note_dur` and generator attributes; params are identical to v1 for the same seeds):
```
/home/kim/Projects/synth_env/bin/python generate_bass_dataset.py --samples 200000 \
    --out /run/media/kim/Mantu/surge_dataset/surge_bass_200k_v2.h5 --workers 12
# optional: --note_dur_min 0.06 to cover 16th-note stabs (changes only the note lengths, not the RNG stream)
```
Takes ~1 h for 200k on 12 workers. Verify: the h5 has datasets `audio mel params note_dur`, each with 200000 rows.

**Train:**
```
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python train_bracket.py --model_type flow --run_id G05_flow_v2 \
    --h5_path /run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5 --epochs 30
```
Verify: `<output_dir>/<run_id>_best.pt`, `<run_id>_val.jsonl` (one line per epoch) and a row in
`leaderboard_v2.tsv`. `val_score` is comparable between ResMLP and flow runs; the flow additionally logs
single-draw and best-of-K scores.

**Compare models on audio (the fair comparison):**
```
$PY evaluate_holdout_audio.py --ckpt resmlp=/path/G01_best.pt --ckpt flow=/path/G02_best.pt --n 200
```
Takes a few minutes (each note is rendered once per method, 8 times for a flow). Verify:
`holdout_audio_summary.json`. Read it against the two reference arms: `oracle` (the floor) and `mean_patch`
(the trivial baseline). A model that does not clearly beat `mean_patch` has learned nothing audible.

**Real stems / single note:**
```
$PY evaluate_200k_inversion.py --stem acid:/path/target_acid_pluck_a2.wav:45:0.22 --ckpt resmlp=/path/G01_best.pt
$PY match_untitled_note.py --target /path/untitled.wav --midi_note 36 --note_dur 0.08 --ckpt flow=/path/G02_best.pt
```
`match_untitled_note.py` defaults to DE over all 10 circuits at maxiter 15 / popsize 8 (about 10 min at roughly
40 ms per render). Watch for the printed `OOD flags`: a clip or note length outside the training distribution
makes the neural numbers a statement about generalisation, not about the method.

Patch exports are `<name>.json` (readable values) plus `<name>.pedalboard_state` (restore with
`plugin.raw_state = open(p, "rb").read()` in pedalboard). Neither is a Surge `.vstpreset` file.

**Synth-JEPA** (Hayes, Tian, Lattner, arXiv:2609.31024; the PDF is the reference for every "Sec." in the code):
```
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python train_synth_jepa.py --run_id synth_jepa_v2_20ep --epochs 20
$PY evaluate_holdout_audio.py --ckpt resmlp=/path/G01_best.pt --ckpt flow=/path/G02_best.pt \
    --jepa jepa=/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs/synth_jepa_v2_20ep/checkpoint_latest.pt --n 200
```
Training prints the model size (paper: 53M; ours 50.7M at the default `--ff_dim 1024`, a guess since the paper
gives no depth/FFN width) and, per epoch, cross-modal retrieval accuracy with its chance level (1/batch); a
retrieval figure near chance means the embeddings carry no pairing information whatever the losses say.
Known gaps from the paper that code cannot close: it trains 1M steps on audio rendered online (always-new
sounds) over 139 parameters at 3 s stereo; we have a fixed 200k set of 0.8 s mono notes over 23 parameters,
so long runs revisit the same data (the script prints how many passes) and the train/val gap needs watching.
The EMA-teacher ablation is not implemented. The search keeps to the training support (`surge_spec.CONT_BOUNDS`)
and pins the MIDI note; see `synth_jepa_search.py`'s docstring for why both matter on this dataset.

**Real-preset bass prior + overnight online training** (v2, 2026-10-02 review; handover:
`HANDOVER_2026-10-02_realistic_prior.md`):
```
/home/kim/Projects/synth_env/bin/python extract_real_bass_manifold.py      # needs the Surge plugin; checks every unit conversion
bash run_overnight_realistic_bass.sh                                        # trainer under the crash-only watchdog
```
The v1 manifold (any `real_bass_manifold.npz` without `format_version` 2) was built with guessed unit conversions
(wrong oscillator slots for sub/sync, a discontinuous shape mapping, filter/waveshaper enums divided by guessed
list lengths, wrong drive/keytrack scaling); the v2 prior refuses to load it. The trainer validates on notes from
HELD-OUT presets and on training presets every 1000 steps (`val.jsonl`); the gap between the two is the
memorisation check. Exports: `jepa_best.pt` (for `--jepa`, carries the prior's search box) and `flow_best.pt`
(for `--ckpt`).
