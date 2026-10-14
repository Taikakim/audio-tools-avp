# CLAUDE.md

Guidance for Claude Code when working in the **audio-tools-AVP** fork of
`stable-audio-tools` (LatCH, FusionOpt, SAO-Small training, audition renders).

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
>
> ⚠️ **`WORKLOG.md` and the `AGENT_DIALOGUE.md` cross-instance channel are PUBLIC** (the
> dialogue log auto-mirrors to a public URL for remote review). **Never write secrets** —
> passwords, API keys/tokens, SSH creds, `.netrc` contents, or credential-revealing paths —
> into either; keep secrets in the shell/env. (See MASTER §4.)
@/home/kim/Projects/SAO/MASTER.md

## This repo at a glance

- **venv:** `sat-venv/bin/python` (Python 3.10, torch 2.10 ROCm). See MASTER §3.
- **ROCm env:** config-driven via `rocm_env.yaml` (repo root) + `stable_audio_tools/rocm_env.py`.
  Package `__init__` applies the `inference` profile on import; `scripts/train_latch.py`
  loads `rocm_env.py` standalone and applies `training`. Shell exports override the YAML
  (`setdefault`). Tunings: `~/pytorch-tunings-7.2.3` (torch 2.10).
- **Flash Attention (native CK) — always on.** `sat-venv` ships the CK-backend `flash_attn 2.8.4` for
  RDNA4, but it's inactive until you `export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` (before
  `import torch`) → **30–100% faster** than the `aiter`/SDPA fallback; forgetting it logs
  `No module named 'aiter'` + `flash_attn not installed`. Set it for every train/infer/eval run (the
  `avp_sa3` train + eval drivers already export it). FA training is safe (13-grad backward patch
  applied). Canonical note: `SAO/MASTER.md` §5 + `SAO/docs/flash-attn-ck-rdna4.md`.

## Key local docs (the durable record)

- **`LATCH_RESULTS.txt`** — the running experiment log (§1–23): smoothing/optimizer/
  architecture bake-offs, the validated ship recipe (§21: SF-NorMuon, d256/dp4, bf16,
  `--compile`, adaln_zero), energy/power studies, seed-variance findings. Read before
  proposing any LatCH training change.
- **`docs/FUSION_SHAREABLE.md`** — FusionOpt (Muon+MONA+KL-Shampoo+ScheduleFree) +
  TimeConditioningCache, what transfers to other models, what to skip.
- **`docs/fusion-optimiser.md`**, **`docs/specs/`** — design specs.

## LatCH essentials

- LatCH heads = small transformers that predict an MIR feature from a noisy latent and
  steer generation (training-free guidance). Targets come from mir (see MASTER §4).
- Training: `scripts/train_latch.py` (flags `--optimizer fusion --components ns5,normuon,sf
  --t-injection adaln_zero --compile --hot-dtype bf16`). Dataset: `scripts/latch_dataset.py`
  (`--target-source db|whole_track`, `--npz-root`).
- Audition: `scripts/render_audition*.py` → `renders/<set>/` (`manifest.json` + `index.html`).
  Score with mir's Audiobox (mir venv). See MASTER §4.

## Conventions

- Branch discipline: LatCH work has lived on `latch-rms-control`; merge to `main` and note
  it in `WORKLOG.md` so SA3 (which mirrors the LatCH model) stays in sync.
- When staging files for commit, stage only your own — leave unrelated working-tree changes.

## Synth inversion / Synth-JEPA (experimental, planned GUI component)

`scripts/synth_inversion/` — Surge XT 1.3.4 patch inversion (ResMLP, OT-CFM flow, Synth-JEPA arXiv:2609.31024) plus
the v4 components and eval paradigm. Added 2026-10-12 from Antigravity/KUANG's work; owner WINTERMUTE.
- **Start at** `scripts/synth_inversion/README.md`, then `V4_COMPONENTS.md`, `EVAL_PARADIGM_V4.md`, `V4_EXPERIMENT_REPORT.md`. Map: `SAO/ARCHITECTURE.md` §G.
- Run scripts **from that directory** (they import each other by module name). Training venv: `stable-audio-3/.venv`; dataset generation / Surge: `synth_env` (pedalboard). Eval scripts need torch AND pedalboard in one venv.
- Tests need no Surge or GPU: `python -m pytest -q test_synth_inversion.py test_v4_components.py`.
- **Evidence rule:** the v4 [R] results are on a stand-in synth, not Surge; the v1 reports were invalidated (see `docs/synth_inversion_reports/WALKTHROUGH.md`). Compare runs with the paired, replicate-gated tools, not single-run numbers.
- **V5 (October 2026):** state, rules and next steps are in the "Status" section at the top of
  `scripts/synth_inversion/V5_PROPOSAL_REVIEW.md`; tool list in the README's "V5 tools" paragraph. Tests:
  `python -m pytest -q test_v5.py test_flow_seeded_v5.py` (CPU, no Surge; the trainer import needs Essentia or the fakes).
  Render banks carry a code fingerprint and must be rebuilt after edits to the spec, prior, ladders or random-patch
  sampler. The layout is 53 values since `e6e07f4`; the 48-value A-series checkpoints load only at an older commit.
  Division of work: KUANG (laptop) writes and CPU-tests ML code; runs, timing and Surge work happen on the desktop.
- Branches: `fix/inharmonicity-aux` (current; v5 tools merged), `feature/v5-flow-seeded-search` (flow-seeded search,
  bank code guard), `feature/v4-components`, `feature/v4-eval-paradigm`. Interfaces are not frozen.
- Optimizer details for `ModularOptimizer` / `FusionOpt`: `fusion-optimiser/docs/MANUAL.md` (branch `docs/manual`, option-by-option).
