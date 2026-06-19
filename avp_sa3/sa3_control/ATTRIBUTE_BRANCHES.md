# NEXT MILESTONE — Attribute-branch control (the actual novel contribution)

**Decided 2026-06-19.** After the 200-track LR/optimizer bracket lands a non-collapsing
training recipe, the next milestone is **explicit, time-varying feature control of SA3** —
condition generation on **rhythm / melody / dynamics curves** from the mir timeseries.
The audio-reference riffer stays as the "style" mode; this becomes the headline.

## Why this, not more riffer tuning

- **It's the differentiator.** "ControlNet-for-music on SA3, driven by *real extracted MIR
  features*" is something nobody else can do — it requires our 97-feature, per-frame mir
  pipeline. underfit/DoRA bake a *fixed style* into weights and can't take a chroma curve as
  input; the SA3 paper is the base model; MuseControlLite is an older model. The recent
  LR/optimizer work is generic recipe-finding (underfit territory) — a means, not the goal.
- **It should also work *better* than the riffer.** The riffer conditions on an **opaque
  reference latent** — a weak, indirect signal (which is why it fought us: mode collapse,
  weak specificity). Attribute branches condition on **explicit features that ARE the target's
  structure** (chroma = key/melody, beat-activations = rhythm, band-RMS = dynamics). Direct,
  strong, learnable signal: "produce velocity whose decoded chroma matches *this* curve."
- **The eval is unambiguous** (unlike the riffer's fuzzy "style match"). The control *is* the
  target → success is directly measurable (see Eval below).

## Data — already in place

`LatentControlDataset(controls=("dynamics","rhythm","melody"))` already loads, **time-aligned
to the latent (T=4096, no resampling)**, from each crop's `.TIMESERIES.npz`:
- `dynamics` (4) — per-band RMS curves (bass/body/mid/air)
- `rhythm` (3) — beat / downbeat / onset activations
- `melody` (12) — chroma (HPCP)

≈19 feature channels × T. The riffer training just passed `controls=()`; flip it on.

## Design

**Reuse the validated decoupled cross-attn adapter** (`adapters.py`) — base frozen, own K/V,
zero-init out, the gain knob, the module-global control holder. Only the **conditioner**
changes.

**Key difference from the riffer: do NOT pool to global tokens — keep it TIME-ALIGNED.**
- Riffer `AudioRefEncoder`: 256×T ref latent → AdaptiveAvgPool → 256 *global* tokens (a style
  summary, no time structure). Correct for "be like this track."
- Attribute conditioner: feature curve (≈19×T) → light strided convs (e.g. /4 → T/4 tokens) →
  control tokens **that retain the time axis** + strong fractional position encoding
  (`add_fractional_positions` already exists). The cross-attn + position encoding then lets
  each output frame attend to its *aligned* control region → approximate local conditioning.

**Injection choice to settle first (one probe):** the SA3 DiT forward already exposes
`modular_local_cond` / `local_add_cond` (per-frame additive conditioning, `modular_local_cond_configs`).
If the `-base` checkpoint has usable local-cond plumbing, feeding attribute features there is the
truer ControlNet path (strictly local). If not, the time-aligned-cross-attn-adapter route above
works with zero fork changes. **Probe `sam.model.model.model` for non-empty
modular_local_cond support before committing.**

**Per-branch or fused:** start with one conditioner that concatenates the enabled channels →
control tokens (simplest). Later, separate branches per attribute with independent gains
(compose "this rhythm + that chroma" at inference) — the real payoff.

## Training

Same loop, with the recipe the bracket picks (best LR/optimizer, likely `--timestep-sampler
log_snr`, short crop per underfit). Keep per-item cfg-dropout on the control so CFG works at
inference. Train against the per-frame features sliced to the crop window.

## Eval — measurable, no ears required to gate it

Generate with a *held-out* control curve, decode, re-extract the feature with mir, compare to
the input curve:
- **chroma**: per-frame chroma-of-output vs input chroma curve → time-resolved correlation.
- **rhythm**: onset-detect the output, compare to the input beat grid → F-measure / alignment.
- **dynamics**: output band-RMS envelope vs input dynamics curve → correlation.
Success = the output *follows the curve* and follows a **different** curve differently
(the cross-ref-diff lesson: vary the control, the output must vary with it). Gain sweep as before.

**Primary metric: MERIT disentangled similarity** (`github.com/AMAAI-Lab/MERIT`, cloned to
`Projects/MERIT`). Frozen MERT-330M + 3 tiny pre-trained heads (~11 MB each) → three *independent*
cosine scores per audio pair: **`S_mel` (melody), `S_rhy` (rhythm), `S_tim` (timbre)**. This is the
fix for our whole metric crisis: chroma can't see collapse and only sees key; cross-ref-diff is a
blunt RMS that can't say *which* factor matched. MERIT decomposes it.
- **Attribute-branch eval:** the control curve came from a source track → decode the output, run
  MERIT against that source. Steering **melody** must raise **`S_mel`** *while `S_rhy`/`S_tim` stay
  put* — that's **control + disentanglement** in one shot. Our factors (mel/rhy/dynamics) map ~1:1
  to MERIT's (mel/rhy/tim). The hand-rolled per-frame correlations above stay as the *time-resolved*
  check; MERIT is the *factor-identity* check.
- **Riffer eval too:** output-vs-each-reference per factor tells us *what* the riffer transfers
  (e.g. high `S_tim`, low `S_rhy`) — and collapse = uniform similarity to all references.
- **Validation-during-training (solves "loss is a non-metric"):** every N steps, generate a few
  and log `S_*` to target — a **meaningful curve to watch** instead of the flat RF loss. Cheap-ish
  (decode + one MERT-330M forward; fits ~1.3 GB).
- **Heavier, later:** a MERIT *perceptual loss* on the 1-step `x0` decode (disentangled training
  signal). Caveats: per-step decode cost + noisy `x0` → defer past the validation use.
- **License:** heads are CC BY-NC-SA (MoisesDB-derived) — fine for research/eval, flag before any
  commercial use.

## Alignment with the SA3 Latent Explorer (`mir/plots/explorer_sa3/`)

Built 2026-06-19; it's the natural UI + eval harness for this milestone — **use it, don't
duplicate it.**
- **Same data**: reads `/home/kim/Projects/latents_sa3` (`.npy` + `.json` + `.TIMESERIES.npz`).
  The `.TIMESERIES.npz` **21 per-frame fields ARE our control features** — feature definitions
  must match the explorer's, so a curve seen in the viewer == the curve we condition on.
- **Two-process split to reuse**: Viewer (mir venv, Dash, :8051) reads sidecars + plots; Player
  (SA3 venv, :7892) owns the SAME-L VAE + LatCH heads + chunked decode over HTTP (`/decode`,
  `/mix`, `/steer?crop=&head=&gain=`). Run the **feature-following eval through the player's
  decode** (don't stand up a second decoder), and expose the adapter as a **new player endpoint**
  (`/generate?adapter=&control=`) so the dashboard auditions it like any other recipe.
- **LatCH = the training-free twin.** The explorer's `/steer` is LatCH (a head decodes a feature
  from the latent, gradient-nudge toward target; gain ≈ 48–96, judge by spread). Our attribute
  adapter is the **trained** version of the *same idea on the same features* — complementary
  (MASTER §4). A feature the LatCH head can already steer is a proven-latent-decodable, ideal
  **first attribute target** (chroma is steerable today → start there).

## Execution order (after the bracket)

1. Probe `modular_local_cond` support on `medium-base`; pick injection path.
2. `conditioner.py`: add `AttributeEncoder` (time-aligned, no pool, position-encoded).
3. `train.py`: `controls=("dynamics","rhythm","melody")`, feed the per-frame features.
4. Smoke (grads to adapter, base frozen) → short train with the bracket recipe.
5. Feature-following eval (above). Iterate gains.
6. If it follows: separate per-attribute branches + independent gains (compose controls).

## audioscope (activation steering) — borrow the probe + a cheap baseline

`github.com/guglielmocamporese/audioscope` (cloned to `Projects/audioscope`) does
mechanistic-interp **activation steering** on SA3's DiT: mean-difference a direction in the
**1536-d residual stream** (`v_l = mean(acts_pos) − mean(acts_neg)`, unit-norm), add `α·v_l` to
the block output at inference → mood shifts, no prompt change. A **third control paradigm** next
to our trained adapter and LatCH — and the lightest (zero training). Three concrete borrows:

1. **The per-layer probe answers our "where to inject" question.** audioscope fits a logistic
   probe on every DiT layer's activations → valence is most linear at **layers 4–11** (first half;
   "middle layers encode semantics"). **Run the same probe for rhythm/melody/dynamics** → pick the
   injection layer(s) empirically instead of guessing. This is **step 1** of the execution order
   (do it alongside the `modular_local_cond` probe).
2. **Free global-mood vectors from OUR labels.** audioscope uses contrastive *prompts*; we have
   thousands of crops with **essentia mood labels** (and 496 class labels total). Collect DiT
   activations on `happy` vs `sad` crops (real labeled audio, better-grounded than prompts) →
   mean-diff → an instant **mood dial**, and a **training-free baseline** to measure the trained
   adapter against. Cheap, GPU-gated, no training.
3. **`@torch.compile` gotcha (cross-check our adapter).** SA3's `TransformerBlock` is
   `@torch.compile`'d on CUDA → `register_forward_hook` *fires but doesn't modify the graph*; the
   fix is monkey-patching `block.forward` (which is what our adapter already does). We run
   `TORCH_COMPILE=0` (ROCm), so hooks likely work for us — but the wrap approach is robust either way.

**Scope boundary:** activation steering is a **global, constant** direction (one `α`) → great for
*categorical/global* attributes (mood, valence), **cannot** do a time-varying curve. So it
*complements*, not replaces, the attribute branches: global moods → steering (free); time-varying
rhythm/melody/dynamics → the trained adapter (this milestone).

## Open questions for Kim
- Which attributes first? (chroma is the most striking demo; dynamics the easiest to verify.)
- One fused control or per-attribute branches from the start?
- modular_local_cond (truer, maybe needs fork peek) vs time-aligned cross-attn (zero fork change)?
