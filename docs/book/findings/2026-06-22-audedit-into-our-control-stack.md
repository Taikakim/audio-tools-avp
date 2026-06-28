# AudEdit into our control stack

*Finding — 2026-06-22. Source: [AudEdit, arXiv 2606.15149](https://arxiv.org/abs/2606.15149) (Zhongyuan Fu, Nankai; single-author preprint, 2026-06-13) read in full, checked against `avp_sa3/scripts/sa3_flowsep.py` and the trained onset-density head. Advances the [Sourcebook](../CONTROL_METHODS_SOURCEBOOK.md) entry #3 "FlowEdit + AudEdit" (Ch9/10): its status was "core, de-risk first — no code, you reimplement it." That status is now **superseded** — we already have a near-faithful implementation.*

---

## What AudEdit is

A **training-free editor for our exact instrument** — SA3 medium + the SAME latent — that revises a *real recording* under a new prompt without the noising/inversion detour. The whole trick (Algorithm 1):

- Start the edited latent **at the source itself**, never noised: `z_edit ← x_src`.
- At each flow step, build a stochastic source state `ẑ_src = (1−t)·x_src + t·ε` and a *matched* target state `ẑ_tar = z_edit + (ẑ_src − x_src)` — the **same** noise displacement, transplanted onto the running edit.
- Step along the **velocity difference** `V_tar(ẑ_tar, c_tar) − V_src(ẑ_src, c_src)` (same frozen DiT, two prompts, two CFG scales). Because both branches carry the identical ε, the common denoising behaviour cancels and what's left is a pure **edit direction**.
- The path is **coarse-to-fine**: early (high-σ) steps move semantics — genre, instrument, texture — while late (low-σ) steps *preserve* transients, onset placement, microtiming, melodic contour. An optional `n_min > 0` tail finishes with plain target sampling for stronger genre/timbre rewrites.

Paper defaults: 28 steps, `n_max=24`, `n_min=0`, `n_avg=1`, `w_src=1.5`, `w_tar=3.5`, `η=1.0`. `n_max` is the primary edit-strength control. Results: it **shifts the prompt-adherence ↔ source-preservation frontier** (not just picks a different point on it) — CLAP-T, CLAP-A, LSD, MCD, LPAPS, structure (MelodySim/AudioBERTScore) and FAD all improve *together*, and it wins all three MOS axes against SDEdit, ODE-inversion and FireFlow.

## The core realisation

> AudEdit's velocity difference `V_tar − V_src` is the **same object** as our decoupled adapter's additive delta `base + gain·adapter(ctrl)` — both isolate an edit *direction* by subtracting away the shared style/structure component. The difference is what the direction is **anchored to**: AudEdit anchors it to a *real source latent* (the same track with everything-but-the-edit held fixed) — the one invariant the density head never had, because the corpus never showed one track at two densities.

This makes the relationship **complement, not substitute**, and bidirectional:

- **What AudEdit lacks** (its own §8 Limitations, verbatim): *no beat-synchronous constraints, no stem-level controls, no pitch-contour preservation, no time-varying local control.* Those are **exactly our trained heads** ([Ch13](../chapters/13-adding-a-tuning-peg-control-heads.md)).
- **What our heads lack**: they only steer *fresh* generation. AudEdit edits a *real recording*. Wire a head in as the `V_tar` branch and every head becomes an **editor of existing audio**.

## Where our code already is

`sa3_flowsep.py` is a **near-faithful** Algorithm 1: `z_edit=x0.clone()`, the Eq-17 stochastic source marginal, the Eq-18 shared-noise target proxy, the Eq-19 difference with independent src/tar CFG, the Eq-20 Euler step, and the `n_max` window all match. It reuses SAME encode/decode and the real schedule via the `make_flowedit_euler` monkey-patch — no architecture surgery. **Three gaps** remain vs the paper:

1. **`η` (step coefficient) is hard-wired to 1.0** — not exposed as a knob (functionally fine at the default, but the sweep dimension is missing).
2. **No `n_min > 0` tail** — the paper's style-edit mode (difference-ODE early, plain target sampling for the last `n_min` steps) is absent. This is the one genuinely missing *capability*.
3. **`anchor_eta` is a non-paper RF-Solver fidelity dial** (pulls `z_edit` toward `x0` for `t≥τ`) that **name-collides** with the paper's `η`. At `anchor_eta=0` the script is pure AudEdit; the recipe default `0.05` departs from it.

## Two caveats to bank

- **Do not trust the paper's operating point or absolute numbers.** Their SAME latent is **32-channel**; our SAME-L is **256-channel and ~10× less gain-sensitive** — the same fact that forces LatCH gain 48–96 vs the SAO-Small default of 8. Our `sa3_flowsep` already runs `cfg_tar=13.5` (vs paper 3.5), `n_max=33` (vs 24) for this reason. The operating point is **codec-dependent** — re-measure on our latent, never import.
- **`anchor_eta` is the wrong axis for disentanglement** — it co-varies density *and* style (a faithfulness dial, not a density dial). Use the `n_max`/`n_min` cut-point instead.

## The honest catch (load-bearing)

The tempting idea — *use AudEdit to manufacture "busy-techno / sparse-goa" counterfactuals and retrain the density head disentangled* — **might launder the disease rather than cure it.** Density and style may be entangled in the **model's text-conditioned prior**, not only in our training data (which is *why* we trained a head instead of using text at all). AudEdit edits density via a *text delta*, so it could drag style exactly as the head does. The paper proves preservation of *timing/identity*, but **never demonstrates decoupling one continuous attribute from genre** — that is outside what it was evaluated on, and App. J documents identity leakage.

So the leverage is in a **cheap probe, not the expensive engine** — and either outcome is a win: if AudEdit *can* move density while MERIT holds `S_tim`/`S_mel`, the data engine is a major result; if it *can't*, that null **retroactively justifies the trained head** and reframes style-entanglement as inherent to the only available lever, not a fixable data artifact.

## The plan, in dependency order

**Tier 0 — pure code, zero GPU contention (safe during the 40-epoch run):**
1. Close the three gaps in `sa3_flowsep.py`: expose `--step-eta`, add the `n_min>0` style-edit tail, add a `--paper-mode` preset, rename `anchor_eta` to end the collision. Validate `n_min=0` is bit-identical to today first.
2. Build `audedit_eval.py` (mir venv; CLAP confirmed importable): CLAP-T / CLAP-A / FAD + SDEdit and ODE-inversion baselines (the `rf_integrate` helper already builds those). First defensible metric for the open-vocab separation niche — replaces the env-corr proxy and by-ear folklore.

**Tier 1 — first GPU-day, decides everything downstream:**
3. **The entanglement probe** (highest leverage): 20 crops, AudEdit-edit to hi/lo density, run the trained head at matched densities, MERIT-score both vs source → a single `entanglement_index` (does `S_tim`/`S_mel` stay flat while `S_rhy` moves?). No training, no data written. Wire it as a reusable index for *every* future head — finally turns "trust the ear" into a logged number the flat RF loss can't give.

**Tier 2 — gated on the probe passing:**
4. **Head-as-`V_tar` fusion**: head-on as the target branch, head-off as source, inside the difference-ODE → source-preserving density editing of a real track **with no retraining**. Try *before* the data engine; if it works the retrain may be unnecessary.
5. **Counterfactual data-engine pilot**: 50 tracks only, capture `z_edit` pre-decode (no re-encode), label from *decoded audio* via librosa, MERIT-gate, **report kept-fraction before** any full-corpus run. Low yield ⇒ the engine is dead and you've spent almost nothing.

**Tier 3 — later:** re-establish the operating point on our 256-ch codec; curve/chroma-conditioned editing (needs the chroma head trained, [Ch5](../chapters/05-how-moments-listen-attention.md) cross-attention door); an AudEdit `/generate-edit` endpoint in the latent explorer alongside `/steer`.

## Discarded

- **AudEdit as a substitute for trained heads** — its own §8 sinks it: source-preserving *edits* only, weak on broad regeneration (App. J), no time-varying/beat-sync/pitch control, 4 NFE/step. Complement-only.
- **`anchor_eta`-graded counterfactual ladder** — built on the wrong axis (see caveat); only the `n_max`/`n_min` cut-point variant is sound, and only after the data-engine pilot proves clean pairs are obtainable.

---

*Open question:* if the probe shows density is **not** text-steerable on SA3 — that style rides along no matter how we phrase the edit — is that a property of *this corpus* (goa, where busy really does imply a sound), or of the SAME latent's geometry itself? The first is fixable with broader data; the second would mean the trained head isn't a workaround but the **only** lever there is. The probe measures the symptom; telling the two causes apart is the next question.

---

## Update 2026-06-23 — the weight-trajectory result reframes the probe as *the architecture fork*

A weight-space trajectory analysis of the onset-density head (lr 2e-5, 40 ckpts; data in `SAO/checkpoint-stats/`) connects directly to the **core realisation** above and makes the probe load-bearing for the whole control program.

**The finding.** The head's weights drift *directionally* (cos-to-final → 1.0, constant global norm — they rotate on a shell) toward an attractor the rectified-flow loss likes, at a velocity that decays but **never reaches zero**. Yet eval-measured control authority (the slope) peaks ~**epoch 24** then **declines** while the loss stays flat. The higher-LR run (8e-5) reaches the *same* net displacement in ¼ the steps → **the destination is LR-invariant; LR only sets the speed.** So this is *directed overshoot of a mid-trajectory optimum under a loss blind to control* — **not** overfitting, **not** a stuck minimum. Lower LR can't fix it (same destination, just slower); the only levers are EMA / weight-averaging / eval-based early-stop.

**Why it points back here.** The core realisation says `adapter(ctrl)` and `V_tar − V_src` are the same edit-direction object. The trajectory result adds: the **learned** version of that object is structurally unstable — a blind loss drifts it past the good region, forcing checkpoint-selection hacks. AudEdit **computes the same delta from the frozen model** — no optimisation trajectory, hence no drift, no "which checkpoint", no EMA. Wherever the frozen model can express an attribute, the training-free path is the more robust way to obtain the identical control direction.

**So the probe is now the fork that decides whether a trained head is even the right tool for density:**
- Probe **passes** (AudEdit moves density while MERIT holds `S_tim`/`S_mel`) ⇒ the frozen model *can* express density; the drift-prone trained head is a reinvention we can demote — prefer training-free editing; the soup/EMA work becomes moot *for density*.
- Probe **fails** ⇒ density isn't text-expressible; the trained head is the only lever *despite* its drift ⇒ commit to EMA / early-stop / soup to manage it (exactly what the current soup sweep is testing).

**Sharpened probe (supersedes the Tier-1 sketch).** Add a *checkpoint axis* to test whether the drift **is** entanglement deepening: MERIT-score density-move (`S_rhy`↑) vs style-hold (`S_tim`/`S_mel`≈) for **(a)** AudEdit text-edit, **(b)** head @ **ep24** (peak control), **(c)** head @ **ep40** (drifted). Falsifiable predictions: head entanglement *worsens* ep24→ep40 (drift = trading clean density control for joint goa-manifold memorisation); AudEdit is either cleaner (→ use it) or equally entangled (→ prior-level — the open question resolves toward *geometry, not corpus*). One probe answers both the drift mechanism *and* the AudEdit-vs-head question. **GPU-gated** — queued behind the clean-range sweep + soup evals now running.

**Empirical result (2026-06-23) — the drift is mostly benign; corrects the framing above.** The clean-range Audiobox sweep (16 ckpts ep15–30 + 8 soups, gains 0.8–2, densities 5–8.8) lands two facts: **(1) later checkpoints are *better* in the usable range** — CE quality and cleanliness improve ~monotonically ep15→ep27 (CE 5.70→6.31, disintegrating-cell fraction 0.89→0.22), control slope ~constant — so the "authority declines after ep24" was a **high-gain artifact**, not a clean-range one. **(2) No soup beats the best single late checkpoint** (best soup `cosasc` 6.17 < ep27 6.31; front-loaded soups *worst*, dragging in low-quality early ckpts). **⇒ the "EMA/average out the drift" recommendation is NOT supported** — when quality is monotonic in epoch, just take a late checkpoint (~ep27+, gain ~1.5). This *weakens* the "the head is structurally unstable, escape to AudEdit" pessimism: the trained head is a solid operating point. The probe is therefore reframed — not a **rescue** from a broken head, but an **upgrade** question: can AudEdit move density with *less* style-drag (lower entanglement) than the (decent) trained head? Audiobox can't answer that — quality ≠ disentanglement — so the MERIT probe is still the decisive next test. (Caveat persists: Audiobox ranks gain 2.0 best, but the ear caps clean at ~1.1 — trust the *epoch trend* for checkpoint choice, the ear for gain.)
