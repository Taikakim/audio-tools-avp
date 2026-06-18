# avp_sa3 — SA3 tooling (ARCHITECTURE)

Tooling for working with **Stable Audio 3 `medium-base`** on this system. Lives in
the `audio-tools-avp` repo; runs with the **SA3 `.venv` (3.13)** where
`stable_audio_3` is installed. Requires **no changes to the SA3 fork** — everything
wraps SA3 at runtime, so `Taikakim/stable-audio-3` stays upstream-syncable.

> **Reuse the plumbing.** Before building, read the SAO-level map
> `/home/kim/Projects/SAO/ARCHITECTURE.md` (the cross-repo reuse index) and this file.
> We keep finding things already built (e.g. the bungee binding + comparison GUI).

## Components

### `sa3_control/` — control-adapter training (MuseControlLite-style)
Trains **decoupled cross-attention adapters** on pre-encoded SAME-L latents +
grid-aligned `mir` control features. The adapter wraps the fork's `Attention` (its
own K/V, zero-init output, additive — base frozen). The **audio-reference branch is
the "similarity riffer."** Status: **dataset done + tested**; adapter / inject /
conditioner / train / generate in progress.
- `dataset.py` — `LatentControlDataset`: reads `Lehto/latents_sa3` (`.npy` + `.json`
  + `.TIMESERIES.npz`) → `latent (256,4096)` + controls (dynamics 4 / rhythm 3 /
  melody 12, **already T=4096, no resampling**) + prompt + `ref_latent` (a different
  crop of the same track = the riffer pairing) + padding_mask. **No audio I/O.**
- *(planned)* `adapters.py`, `inject.py`, `conditioner.py`, `train.py`,
  `generate.py`, `stretch.py` (pluggable: **bungee** binding in `mir/pitch_venv`,
  rubberband fallback).

### `scripts/` — generation-time CLI tools (training-free)
- `sa3_flowsep.py` — generative source separation via **FlowEdit/AUDEDIT**
  (inversion-free difference field) + `--anchor-eta` hybrid fidelity dial. Doubles as
  a riffer (swap the isolate-prompt for a variation prompt). Prefix-aware
  (`TrackType: Instrument` targets, `TrackType: Music, VocalType: Instrumental` source).
- `sa3_zerosep_rf.py` — **RF-Solver** flow-inversion separation + `--eta` z0-anchor
  fidelity controller; near-transparent inversion (round-trip rel-err ~0.23).
- `stem_score.py` — **ground-truth bracketing**: sum project stems into role submixes,
  rank param sweeps by perceptual closeness (logmel-L1 / LSD / env-corr). **Run with
  the mir venv** (librosa).

## Data
- `Lehto/latents_sa3` — 5400 crops, SAME-L 256-d, T=4096; `.json` (prompt + metadata)
  + `.TIMESERIES.npz` (21 grid-aligned control fields). Adapter training data.
- Project stems (per-generator, named by instrument, ~16 tracks) — ground truth for
  `stem_score` and future audio-reference pairs.

## Key design decisions
- **Trained adapter** (this) vs **training-free guidance** (LatCH / FlowEdit) — both
  steer SA3; see MASTER §4. Adapter = stronger/modular, needs training; guidance =
  instant, no training.
- **Runtime wrapping** of `Attention` → zero SA3-fork divergence.
- Rectified-flow objective (`v = ε − x0`), **adapter-only params**, per-modality
  cfg-dropout (enables decoupled CFG at inference).
- Stretch/pitch augmentation = **bungee** (binding already built in `mir/pitch_venv`),
  rubberband fallback. Not on the riffer MVP critical path (ref = different crop).

## Run
```bash
SA3=/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
MIR=/home/kim/Projects/mir/mir/bin/python
PYTORCH_TUNABLEOP_ENABLED=0 $SA3 avp_sa3/scripts/sa3_flowsep.py -i mix.wav ...
$MIR avp_sa3/scripts/stem_score.py --role drums --stems-dir <stems> ...
```
