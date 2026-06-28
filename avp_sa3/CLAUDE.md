# CLAUDE.md — avp_sa3 (SA3 tooling)

@/home/kim/Projects/SAO/MASTER.md

## Before building anything: check what already exists

Read the **SAO-level map `/home/kim/Projects/SAO/ARCHITECTURE.md`** (the cross-repo
"reuse the plumbing" index, kept current as a standing task) and this folder's
`ARCHITECTURE.md` *before writing new code*. We keep rediscovering things already
built — e.g. a working **bungee** time-stretch/pitch-shift Python binding **and** an
A/B comparison GUI were already in `mir/pitch_venv` / `mir/repos/bungee` /
`mir/pitch_shifter_gui.py`. Grep the three repos (`mir`, `stable-audio-tools`,
`stable-audio-3`) for a capability before reimplementing it.

## What this is
Tooling for SA3 `medium-base` — control-adapter training (`sa3_control/`) and
training-free generative separation / riffer + scoring (`scripts/`). Full map in
`ARCHITECTURE.md` (this folder).

## Venvs (invoke by absolute path)
- Anything importing `stable_audio_3`: **SA3 `.venv` (3.13)**
  `/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python`, with
  `PYTORCH_TUNABLEOP_ENABLED=0` and
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` (set **before** `import torch` to
  activate the CK flash-attn build — 30–100% faster; forgetting it →
  `No module named 'aiter'` + FA disabled). `run_control_train.sh` already bakes
  both in. See MASTER §5.
- `stem_score.py` / librosa analysis: **mir venv**
  `/home/kim/Projects/mir/mir/bin/python`.

## Placement rule
Tooling lives **here** (`audio-tools-avp`). The **SA3 fork stays upstream-syncable** —
only genuine SA3-side interface code (LatCH, ROCm env) belongs there. The control
adapter needs **no** SA3-fork changes (it wraps `Attention` at runtime).

## Gotchas
- **Saving audio — never write raw model output.** File writers (esp. the new
  **torchcodec** backend `torchaudio.save` routes through) clip **fp16** / **>1.0**
  float; SA3 output regularly peaks above 1.0. Always go through
  **`sa3_control.audio_io.save_audio()`** (float32 → peak-normalize/clamp → int16 PCM) —
  never `torchaudio.save(x.float().cpu(), …)` raw. This bit the riffer auditions. See MASTER §5.

## Keep docs current
When you add or change a component, update this folder's `ARCHITECTURE.md` **and**
add the new plumbing to the SAO-level `ARCHITECTURE.md` so the next instance can find
and reuse it.
