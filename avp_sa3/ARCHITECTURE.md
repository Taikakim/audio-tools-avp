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
the "similarity riffer."** Status (2026-06-20): **riffer trained + validated** against
`medium-base` (wraps 24 cross-attn, 4.8% params). **Reference-specific only at lr≈1e-4**
(peaks ~step6000 then "elbow"-declines); **heavier LR mode-collapses** (lr1e-3 collapsed).
**Metric lesson: judge by cross-reference AUDIO diff + MERIT, NOT chroma** (blind to collapse)
and **NOT loss** (noise-dominated — flat for working *and* collapsed). Gain ~1–2 clean,
>4 artifacts. **Next milestone → attribute branches** (`sa3_control/ATTRIBUTE_BRANCHES.md`).
- `dataset.py` — `LatentControlDataset`: reads `Lehto/latents_sa3` (`.npy` + `.json`
  + `.TIMESERIES.npz`) → `latent (256,4096)` + controls (dynamics 4 / rhythm 3 /
  melody 12, **already T=4096, no resampling**) + prompt + `ref_latent` (a different
  crop of the same track = the riffer pairing) + padding_mask. **No audio I/O.**
- `adapters.py` ✓ — decoupled cross-attn adapter (own K/V, shared frozen `to_q`/
  `apply_attn`, zero-init out) + `ControlledCrossAttention`. Control tokens via a
  **module-global holder** (NOT a ContextVar — dropped by gradient-checkpoint recompute).
- `inject.py` ✓ — `install_adapters` (wraps every `…layers[i].cross_attn` in place at
  runtime, no fork edit), `freeze_base_train_adapters`, `adapter_state_dict`.
- `conditioner.py` ✓ — `AudioRefEncoder`: ref latent → control tokens (strided convs + pool).
- `train.py` ✓ — bf16/fp32 RF training, adapter-only optimizer, per-item control cfg-dropout.
  `--optimizer adamw|fusion|sfadamw|fusion_full` (FusionOpt SF-NorMuon / SF-AdamW / full —
  **OOMs at crop2048 no-ckpt, run crop≤1024**), `--warmup-steps`, `--timestep-sampler`
  (logit_normal|log_snr…, underfit borrow), `--resume` (warm-start adapter weights),
  `--max-hours`, `--no-checkpoint`, `--no-preencode-text`, `--subset-tracks`.
- `generate.py` ✓ — decoupled-CFG inference + `--gain`; saves via **soundfile PCM_16**
  (never torchaudio/torchcodec — clips fp16). `load_adapter_state` shared with the bracket.
- `merit_eval.py` ✓ — **MERIT** disentangled similarity (MERT-330M + 3 heads) →
  S_mel/S_rhy/S_tim, the eval metric (cross-ref-diff is blunt). Wired into the collapse
  bracket via `MERIT_EVAL=1`.
- *(next)* **attribute branches** (`sa3_control/ATTRIBUTE_BRANCHES.md` — time-aligned
  conditioner, the differentiator), `stretch.py` (**bungee** in `mir/pitch_venv`, rubberband fallback).

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
  + `.TIMESERIES.npz` (21 grid-aligned control fields). Adapter training data. **For
  training, point `--encoded_dir` at the NVMe mirror `/home/kim/Projects/latents_sa3`,
  not Lehto — cold random reads off the removable drive crawl (~2 MB/s) and stall the
  dataloader (first-step freeze, MASTER §5).** Lehto stays the canonical copy.
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

## [2026-06-28] CPU eval servers in stable-audio-3/scripts/ — cross-repo note (stable-audio-3 agent)

File-drop eval servers that complement the `sa3_control/` training pipeline. Scripts live in
`stable-audio-3/scripts/`, run with the SA3 `.venv`:

- **`control_eval_server.py`** + **`submit_control_job.py`** — long-lived all-CPU control-adapter
  ONNX eval server; queue `SAO/control_eval_queue`.
- **`latch_eval_server.py`** + **`submit_latch_job.py`** — CPU LatCH-guidance sibling: plain DiT
  ONNX forward-only (ORT CPU EP) + torch autograd through the ~5-7M-param guidance head only; queue
  `SAO/latch_eval_queue`. `--prompts` takes one verbatim prompt per flag occurrence (no comma-split —
  musical prompts contain commas). Commit 020b6c3, branch `latch-sa3-phase1`.
- **`latch_validate.py`** — CPU/GPU z0-cosine harness; GPU half deferred (`--run-gpu`).
- **`make_text_cond.load_conditioner`** — shared by both servers: loads with `device="cpu"` so the
  1.4 B T5-Gemma weights never hit VRAM. Keep the GPU visible — `HIP_VISIBLE_DEVICES=""` breaks the
  `flash_attn`/`aiter` driver probe at import time.

Details: `stable-audio-3/docs/onnx-amd-inference.md` and `SAO/MASTER.md §5`.
