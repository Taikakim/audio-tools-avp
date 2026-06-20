# audio-tools-AVP

**A fork of Stability AI's [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools), turned into a workshop for *controlling* audio generation.**

The base library trains and runs Stable Audio models. This fork is about the next question: once a model can generate convincing audio from a prompt, **how do you steer it** — at training time and at inference time — toward the specific musical things you can hear in your head but cannot say in words? Bass that breathes with the downbeat, brightness that follows a phrase, a key, a density, a groove. The work here is a growing set of *intervention points* layered onto the base models: trained latent-control heads, control adapters, training-free guidance, a purpose-built optimizer, and the tooling to train, audition, and score it all.

> Personal research fork by **[aavepyora.online](https://aavepyora.online)** ("AVP"). It is one of three repos in a small MIR → control → generation pipeline (a feature-extraction repo feeds targets to the control heads trained here, which steer a Stable Audio 3 inference repo). Primary development is on **AMD (RDNA4 / ROCm)** — see [`install.sh`](install.sh) and [`CLAUDE.md`](CLAUDE.md).

---

## Attribution — LatCH

**LatCH (Latent-Control Heads)** is the research of **Zachary Novack, Zack Zukowski, CJ Carr, Julian Parker, Zach Evans, Josiah Taylor, Taylor Berg-Kirkpatrick, Julian McAuley, and Jordi Pons** (UC San Diego & Stability AI), from the paper *"Low-Resource Guidance for Controllable Latent Audio Diffusion."*

**Official project page: <https://zacharynovack.github.io/latch/latch.html>** — please refer to it for the paper, citation, and any official code release.

The trainer and guidance in this fork are an **independent, community re-implementation** of the method described in that paper (written with Claude, Anthropic). This fork is **not affiliated with, nor endorsed by, the LatCH authors**. Please cite the original work, and treat the authors' own release as the canonical source if/when it is published. See [`scripts/LATCH_README.md`](scripts/LATCH_README.md) for the implementation notes.

---

## What's here right now

Layered roughly from "how directly you steer the sound" outward:

### 🎛️ LatCH — Latent-Control Heads (training-free guidance)
Small networks that read a musical feature (bass energy, spectral texture, rhythmic density, key / tonic strength, HPCP…) out of a *noisy* latent, then run in reverse at inference to **nudge generation toward a target you set** — no retraining of the base model.
- Train: [`scripts/train_latch.py`](scripts) + [`scripts/latch_dataset.py`](scripts); configs `latch_train_*.yaml` (`density`, `rhythm`, `spectral`, `hpcp`, `tonic_strength`, `fusion`, `production`).
- The running experiment log + the validated ship recipe: **[`LATCH_RESULTS.txt`](LATCH_RESULTS.txt)** — read it before changing anything in the training path.
- Trained heads in `latch_weights/`; auditions via `scripts/render_audition*.py` → `renders/`.

### ⚙️ FusionOpt — the optimizer that trains them well
A composite optimizer (Muon + MONA + KL-Shampoo + ScheduleFree; the **SF-NorMuon** subset is the validated sweet spot) built to train the heads stably and fast, plus a time-conditioning cache. Docs: **[`docs/FUSION_SHAREABLE.md`](docs/FUSION_SHAREABLE.md)**, [`docs/fusion-optimiser.md`](docs/fusion-optimiser.md). The `run-fusion-*.sh` scripts record the bake-off.

### 🎚️ avp_sa3 — control tooling for Stable Audio 3
Tooling for SA3 `medium-base`, kept here so the SA3 model fork stays upstream-syncable (the adapter wraps attention at runtime — no base-model changes):
- **Control-adapter trainer ("riffer")** — a decoupled cross-attention adapter with reference-faithful control, trained without touching the base weights (`avp_sa3/sa3_control/`, `avp_sa3/launch_riffer.sh`, `avp_sa3/generate.py`).
- **Generative separation / editing** — training-free, text-prompted "separation" (inversion-free FlowEdit + true RF-Solver inversion) in `avp_sa3/scripts/`.
- Recipes, scoring, and the next-milestone design — **[`avp_sa3/ATTRIBUTE_BRANCHES.md`](avp_sa3/ATTRIBUTE_BRANCHES.md)** (explicit MIR-feature control + a disentangled MERIT evaluation).
- Map: **[`avp_sa3/ARCHITECTURE.md`](avp_sa3/ARCHITECTURE.md)**.

### 🏗️ SAO-Small training & finetuning
Train/finetune Stable Audio Open–Small and encode datasets to latents: `train.py`, `run-sa-finetune.sh`, `encode_*.py`, `pre_encode.py`, on top of the base library in `stable_audio_tools/`.

### 📖 *Hands Inside the Instrument* — the field guide
A book of **questions, not blueprints**, for musicians and instrument-builders. It explains Stable Audio 3 as an instrument you can put your hands inside, and **where control can be injected** at every layer — from the latent material, to the carving process, to LoRA/DoRA, to control heads, to a map of every method, to inventing entirely new controls and naming them precisely enough that an AI can build them. No code, no math. Preface + 16 chapters.
→ **Start at [`docs/book/`](docs/book/).**

### 📚 Reference docs
Architecture and usage notes for the underlying models in [`docs/`](docs/) (`autoencoders`, `conditioning`, `diffusion`, `datasets`, `pre_encoding`, `pretransforms`). Project orientation and conventions in [`CLAUDE.md`](CLAUDE.md).

---

## Getting started

```bash
./install.sh            # set up the dev stack (AMD/ROCm-aware);  or:  pip install .
```
- **Models** are the standard Stability AI checkpoints, gated on Hugging Face — accept the licence and set `HF_TOKEN`.
- Quick test via the Gradio interface (see the base-library section below):
  `python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0`
- **Hardware note.** Primary development is on an **AMD RX 9070 XT (RDNA4 / gfx1201), ROCm 7.2.x, torch 2.10, Python 3.10**; the ROCm environment is config-driven via `rocm_env.yaml`. The base library runs on PyTorch ≥ 2.5 (CUDA included), but the AVP-specific tooling (LatCH guidance, FusionOpt, `avp_sa3`) has only been exercised on the ROCm dev stack — treat CUDA support for those as untested.

---

## Using the base library (upstream `stable-audio-tools`)

The full upstream library is intact. The essentials:

### Install
```bash
pip install stable-audio-tools     # from PyPI
# or, from a clone of this repo:
pip install .
```
Requires PyTorch 2.5+ (for Flash / Flex Attention). Development is in Python 3.10.

### Interface
A basic Gradio interface tests trained models, e.g. (after accepting the model terms on Hugging Face):
```bash
python3 ./run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0
```
Key `run_gradio.py` flags: `--pretrained-name` (HF repo) or `--model-config` + `--ckpt-path` (local); `--pretransform-ckpt-path` (swap in a finetuned decoder); `--model-half`; `--share`; `--username`/`--password`.

### Training
You need a **model config** and a **dataset config** (see *Configurations* and [`docs/datasets.md`](docs/datasets.md)), and a Weights & Biases login (`wandb login`). Then:
```bash
python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name my_run
```
Training wraps the model in a PyTorch-Lightning training wrapper (with EMA, optimizer state, discriminators); checkpoints include it. Use `unwrap_model.py` to extract the bare model for inference, for use as a pretransform, or for partial-init finetuning:
```bash
python3 ./unwrap_model.py --model-config /path/to/model/config --ckpt-path /path/to/wrapped.ckpt --name model_unwrap
```
**Finetuning:** continue a wrapped run with `--ckpt-path`, or start fresh from an unwrapped pretrained model with `--pretrained-ckpt-path`. Useful flags: `--save-dir`, `--checkpoint-every` (10000), `--batch-size` (8), `--num-gpus`/`--num-nodes`, `--accum-batches`, `--strategy` (`deepspeed` for ZeRO-2), `--precision` (16), `--num-workers`, `--seed`, and `--config-file` (path to `defaults.ini` when running outside the repo root).

### Configurations
Model and dataset are defined by JSON configs. A **model config** sets `model_type` (`autoencoder`, `diffusion_uncond`, `diffusion_cond`, `diffusion_cond_inpaint`, `diffusion_autoencoder`, `lm`), `sample_size`, `sample_rate`, `audio_channels`, the `model` block, and the `training` block. A **dataset config** supports local audio directories or WebDataset on S3 — see [`docs/datasets.md`](docs/datasets.md).

---

## Heritage & license

A fork of **[Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)**. The base training/inference code and the Stable Audio models are Stability AI's, under the **[Stability AI Community License](https://stability.ai/license)**; the additions in this fork (the LatCH re-implementation, FusionOpt, `avp_sa3`, and the book) are community/research work under the same terms. Please cite the original authors for **LatCH** (above) and for the underlying Stable Audio models.

## Todo
- [ ] Troubleshooting section
- [ ] Contribution guidelines
