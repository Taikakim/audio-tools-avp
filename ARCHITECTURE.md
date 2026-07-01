# stable-audio-tools — architecture (thin fork)

**Read `/home/kim/Projects/SAO/MASTER.md` first** — cross-repo facts live there.
`SAO/ARCHITECTURE.md` is the full pipeline map. This file describes only what is
*local* to this fork, so other instances don't recreate tooling already in the master
repo.

## Role
Thin fork of Stability-AI **stable-audio-tools** (the "audio-tools-AVP" fork).
Model-agnostic control research: LatCH head architecture + training, FusionOpt,
recipes, eval. General tooling has moved to the `SAO/` master repo; only genuine
package deltas remain here.

## Remotes (important — get this right)
- **`avp`** = `git@github.com:Taikakim/audio-tools-avp.git` — **your fork; push here**
  (`git push avp main`).
- **`origin`** = `https://github.com/Stability-AI/stable-audio-tools.git` — read-only
  **upstream**; sync *from* it, never push to it. (The branch tracks `origin/main` for
  syncing, which is why a bare `git push` targets upstream and fails — use `avp`.)

## What lives HERE (package deltas — preserve on upstream rebase)
- `stable_audio_tools/models/latch.py` — LatCH head (SAO-Small family).
- `stable_audio_tools/rocm_env.py` + `rocm_env.yaml` — canonical ROCm profile.
- Flash-Attention backward patch (`FlashAttnFunc.backward` → 13 grads).
- `stable_audio_tools/training/fusion_opt.py` — FusionOpt optimizer.

## What does NOT live here — canonical home is the SAO master repo
LatCH training (`train_latch.py`, datasets, render / audition, probes), `avp_sa3`
control, and eval live in `SAO/{latch,control,eval}/`. **`LATCH_RESULTS.txt` is the
authoritative experiment log.** Some legacy copies remain under `scripts/` here
(pre-thinning) — prefer the SAO copy.

## Findings & work log
Cross-repo findings → `SAO/WORKLOG.md`. Local analyses under `docs/*_analysis.md`.
