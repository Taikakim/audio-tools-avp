# Control-Methods Sourcebook — revisit pass, 2026-07-06

A second adversarial pass against 14 newly surfaced papers, reconciled against every existing entry in [`CONTROL_METHODS_SOURCEBOOK.md`](CONTROL_METHODS_SOURCEBOOK.md). Convention followed throughout: corrections are **appended**, never silently rewritten — every change in the sourcebook itself is marked inline with `**Revisit 2026-07-06:**` (annotation on an existing entry) or `*Added 2026-07-06.*` (new entry). This file is the accounting of what happened and why; the sourcebook is the source of truth for the actual content.

## Summary counts

- **33 existing bullet-level entries annotated** with a non-empty `Revisit 2026-07-06` note (confirmed or revised verdicts with real new information), plus **6 section-level annotations** (the shortlist table's above-table note, the Ch8 preamble, the Ch10 inversion-family cross-reference + new-threads paragraph, the Ch13 "genuinely open" callout, and the Ch15 closing paragraph) — 39 `Revisit 2026-07-06` markers total in the sourcebook. Where a delta's `final_note` was empty, the entry was either left unmarked or given a brief "no change" note only where doing so was cheap and useful for completeness.
- **14 new entries added** across 8 chapter sections, covering **11 distinct new papers** (3 of the 11 — TADA, DEMON, Diffusion Warm Initialization — each earned a second, differently-angled entry in a second relevant chapter, since the same paper mattered to two sections in genuinely different ways).
- **Shortlist (top-12) table: ranking unchanged, two rows' "first experiment" guidance sharpened**, with a dated note added above the table explaining why.
- **No chapter narrative prose was edited.** Every `chapter_note` in the delta packet described texture, footnote-worthy resonance, or an open question that stayed open — none of the 14 new papers closed or reframed a chapter's actual open question, so per the standing instruction ("only touch a chapter if a new paper genuinely closes or reframes one of its open questions... when in doubt, leave the chapter alone"), all 8 chapter files were left untouched. Each tension is logged below instead.

---

## Section-by-section

### Shortlist — what to build first
- **Annotated:** none of the 12 rows needed a footnote-style annotation (the shortlist doesn't carry prose per-row beyond the table), but the underlying facts from this section's delta packet were folded into the actual chapter entries they concern: TADA's bottleneck finding → Ch5 HeadRouter entry; InnerControl's LatCH-B angle → Ch13 OC-Flow entry; the DirectAudioEdit/SA3-t=0.075 facts → Ch9 FlowEdit/AudEdit entries.
- **Table change:** rows 7 (HeadRouter) and 10 (Diffusion Forcing/DFoT) had their "First experiment" column sharpened — HeadRouter now points at TADA's causally-verified 2-layer bottleneck as a cheaper starting probe; DFoT now flags that Live Music Diffusion Models offers a materially cheaper path for its *streaming* leg specifically (its other two legs — LatCH-B substrate, sigma-field substrate — are untouched). A dated note was added directly above the table explaining both changes and stating explicitly that the *rank order* did not change — nothing was promoted, demoted, or removed. DEMON was explicitly considered and rejected as a competing entrant for DFoT's streaming leg (see honesty-ledger addition).
- **No new shortlist rows added** — the delta packet's own `confirmed_new_entries` for this section was empty, meaning the verification pass itself did not judge any new paper shortlist-worthy on its own merits; this was treated as a signal, not an oversight.

### Ch4 — Sigma, and the per-moment field
- **Annotated:** Patch Forcing, Time-to-Move, Stochastic Generation for Flow Models (RF→VP-SDE kernel), Restart Sampling, Iterative Partial Refinement, Region-Adaptive vector-σ diffusion — all six got a `Revisit 2026-07-06` note; in every case the newest papers were checked against the entry's specific claim and found not to overlap or not to change the assessment (a few, like Patch Forcing and the RF→VP-SDE kernel, needed the note to explicitly rule out DEMON/Diffusion Warm Init as false confirmations).
- **New entries:** **DEMON** (2605.28657) and **Diffusion Warm Initialization** (2606.18968), both fringe/production-reference additions with explicit caution against over-reading them as confirming the chapter's literal per-moment-sigma claim.
- **Chapter file (`04-sigma-the-molten-ness-dial.md`):** not touched. The chapter_note flagged two loose ends for a maintainer (TADA's Ch5 HeadRouter propagation gap, now fixed in this pass; and a DEMON/AudEdit cross-reference opportunity, noted below) but explicitly said neither open question in the chapter itself is resolved.

### Ch5 — Attention as form
- **Annotated:** HeadRouter got the substantive TADA-bottleneck addendum (causally-verified layers {12,13}/24 on the sibling model, plus the Ch11 cross-cutting caution). ReFlex, Attend-and-Excite, EditGen, Dual-Channel Attention Guidance, P2P/PnP/MasaCtrl had empty-note confirmations — left unmarked.
- **New entry:** **TADA** (2602.11910) — a compact version focused on the Ch5 angle (causal cross-attention bottleneck + head-probing implications), cross-referencing the fuller Ch5/6 entry for the steering-method benchmarks.
- **Chapter file (`05-how-moments-listen-attention.md`):** not touched. The "Teaching the door new manners" paragraph's implicit claim (adapters at the cross-attention junction are generically fine) is now known to be more nuanced (steering-locus ≠ good-adapter-locus), but the packet itself called this "not urgent enough to rewrite prose over" — logged as a tension for a future real revision, not acted on now.

### Ch5/6 — Activation steering & representation engineering
- **Annotated:** Steering Diffusion Transformers with SAEs (revised — TADA reproduces its mechanism on Ace-Step, with a scope correction that SAO wasn't used for the SAE benchmark itself), TIDE (confirmed — TADA still doesn't test a timestep dimension, sharpening TIDE's rationale), FreeSliders (revised — TADA's head-to-head was run on Ace-Step, not SAO, correcting the entry's framing), Activation Patching/DiffMean for music (revised — TADA's steering-vector benchmark is mainly Ace-Step, not SAO). Concept Steerers and SAEdit got brief "no change" notes.
- **New entry:** **TADA**, the full version — all three models' bottleneck layers, the Ace-Step-only scope correction on the steering benchmarks, the weight-space-adapter backfire finding, the provenance note (initially mis-flagged as fabricated due to an acronym collision, now confirmed real), and the RF/diffusion-loss framing tension against this sourcebook's other SAO entries.
- **Chapter file (`06-voicing-the-body-adaln-and-norms.md`):** not touched. TADA's bottleneck is a resonant but non-identical echo of the chapter's "selective deafness" question (cross-attention-layer finding vs. an adaLN-sensitivity question) — the packet itself warned not to overstate the connection.

### Ch8 — The conditioner: a native time-varying channel
- **Annotated:** the "two injection styles" preamble claim was narrowed (now reads as the two dominant patterns for *numeric* control, given SegTune's NL-caption pattern). JASCO got an added corroboration note from SegTune's ablation. Sketch2Sound, Music ControlNet, top-k CQT, MusiConGen got brief "no change" notes.
- **New entries:** **SegTune** (2606.02638) and **UNISON** (2605.31530) — both with explicit corrections to how the source material undersold them (SegTune's code is public; UNISON's RF was mislabeled ✘ when the paper is a genuine linear-interpolant rectified flow, and its maturity was undersold).
- **Chapter file (`08-the-voice-prompt-and-conditioner.md`):** not touched. UNISON's depth-matched-injection ablation is a real, quantified counter-example to the chapter's "you do not command which layer hears which word" line, but it's a single ablation on a different backbone/VAE — the packet itself said this isn't strong enough to rewrite the chapter's claim about SA3 specifically, and no change was made.

### Ch9 — The dials: guidance and re-melting
- **Annotated:** FlowEdit and AudEdit both got DirectAudioEdit cross-reference notes (independent corroboration that RF's straight paths don't need DirectAudioEdit's heavier re-noising machinery); AudEdit additionally got the SA3 `t=0.075` truncation fact folded in, sharpening (not changing) the existing port plan referenced in the 2026-06-22 status note. CFG-Zero*, APG, Limited-interval guidance, CFG-MP, Autoguidance, CADS, RF-Solver-Edit, FireFlow, RF-Inversion, FlowAlign all had empty-note confirmations.
- **New entry:** **Audio-to-Audio via Diffusion Warm Initialization** (2606.18968) — the Ch9-specific angle (a systematic empirical sweep of SDEdit warm-start on the sibling model), distinct from the Ch4 entry on the same paper, which frames it as a sigma-field-adjacent inference mechanism instead.
- **Chapter file (`09-dials-guidance-and-re-melting.md`):** chapter_note was empty; not touched.

### Ch10 — Holding material still
- **Annotated:** expanded the previously terse Ch10 section with explicit `Revisit 2026-07-06` notes on the Time-to-Move cross-reference (unchanged), the inversion-family cross-reference (DirectAudioEdit corroboration), and three new named threads that didn't have prior textual homes in this section: seam-quality/crossfade-vs-reconciler, sliding-window drift & re-anchoring (Openings Q5), and the mask-in-sigma/clamping-cost questions (both explicitly unchanged).
- **New entry:** **Diffusion Domain Expansion (DDE)** (2605.23275) — a trained z0-hat patch-reconciler as an alternative to naive overlap-averaging, fails both hard filters but offers a transferable abstraction.
- **Chapter file (`10-holding-material-still-inpaint-and-schedule.md`):** not touched. Verified the actual "Openings" text: the relevant open question ("over a hundred windows, what re-anchoring would hold the piece's identity against drift?") is explicitly marked `(frontier)` and remains open — DDE, DEMON, and LMDM/ARC-Forcing all add scaffolding or analogous fixes on *other* models, but none measures or fixes drift on SA3/Longform itself, so per instructions this was logged as a tension rather than edited into the chapter.

### Ch11 — Changing the body: LoRA, DoRA, adapters
- **Annotated:** Concept Sliders (TADA's -21%-AUC weight-space-adapter-at-the-bottleneck finding, explicitly vindicating rather than undercutting Concept Sliders' own broad-training recommendation) and DoRA (a weak-evidence DoRA-on-SA3-Medium sighting via the new Entropy-as-Structural-Prior paper, flagged as an unverified lead). stable-audio-controlnet, ControlNet-XS, EasyControl had empty-note confirmations.
- **New entries:** none.
- **Chapter file (`11-changing-the-body-lora-dora-adapters.md`):** not touched. A concrete real-world illustration of the chapter's "where would you bond the laminate" question now exists (TADA's steering-locus-vs-adapter-locus finding), but the packet itself judged this "not urgent enough to rewrite prose over" — logged as a tension for a future revision.

### Ch12 — Two kinds of teaching: optimizers & training recipes
- **Annotated:** Muon-in-ViT recipe got a substantial addition (Cautious Weight Decay as a concrete regularization knob, plus corrections about prior art — CWD is not a "formalization" of the modded-nanogpt trick, and masking an orthogonalized Muon update already exists informally in two GitHub PRs). Immiscible Diffusion and Self-Transcendence got brief "no change" notes. NorMuon, DiffRhythm 2, SOAP had empty-note confirmations.
- **New entries:** **Cautious Weight Decay** (2510.12402), **Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics** (2602.23696, with a maturity-tag correction from paper-only to research-code), and **Entropy as a Structural Prior** (2606.07207, flagged explicitly as an unverified lead per house standard: single seed, single prompt, deferred ablations).
- **Chapter file (`12-two-kinds-of-teaching-body-vs-sense.md`):** chapter_note was empty; not touched.

### Ch13 — Control heads & training-free guidance
- **Annotated:** Diffusion Classifier Guidance (InnerControl's analogous clean-z0-meter failure mode, framed as an analogy across domains/papers, not a refutation), Readout Guidance (InnerControl cites it as direct inspiration — external confirmation the head-training blueprint generalizes to the noise-conditioned regime), the "Genuinely open LatCH-B callout" blockquote (InnerControl added as the most concrete scaffolding piece, with real shipped code). OC-Flow got the InnerControl LatCH-B-scaffolding note carried over from the shortlist packet. TFG, D-Flow, MPGD, ESS-Flow, RB-Modulation had empty-note or brief "no change" confirmations.
- **New entry:** **InnerControl** (2507.02321) — full entry with the FID/RMSE ablation numbers, the "quality held only at lower guidance scale" correction, and explicit note that it's `⏱train`-only (probes supply training-time pseudo-ground-truth, not an inference-time head).
- **Chapter file (`13-adding-a-tuning-peg-control-heads.md`):** not touched. The chapter_note explicitly said no correction was needed and enrichment was optional; the chapter stands fine without it.

### Ch15 — The frontier: streaming, sigma fields, live feedback
- **Annotated:** Diffusion Forcing/DFoT got the big revised note (LMDM as a cheaper streaming-leg alternative; DEMON explicitly rejected as an equal alternative, with the paper's own disclaimer quoted); Self Forcing got a corroboration note (LMDM's ARC-Forcing as an independent audio-domain instance of the self-rollout-distillation idea). Rolling Forcing, FreeAudio, FIFO-Diffusion, InfiniteAudio got brief "no change" notes.
- **New entries:** **Live Music Diffusion Models (LMDM) / ARC-Forcing** (2605.22717) and **DEMON** (2605.28657, the fuller Ch15-specific version covering the live-instrument systems angle, the Oobleck/SAME lineage distinction, and explicit non-claims). A closing paragraph ties both papers back to the chapter's "tuner that never stops listening" frontier, echoing the InnerControl cross-reference from Ch13.
- **Chapter file (`15-the-honest-frontier.md`):** not touched. Verified the actual prose ("nobody has tuned that schedule cleanly enough to hand you the curve" — the gain-scheduling frontier is explicitly still open): neither DEMON nor InnerControl resolves it, both only add texture, exactly as the chapter_note said.

### Ch16 — Dreaming up new controls
- No deltas, no new entries in the packet. Reconsidered anyway (per the instruction to touch every section, even to confirm nothing changed) — nothing in the 14-paper sweep bore on stereo/spatial, coupled-head constraints, or generative separation. Left untouched.

---

## Cross-cutting edits made outside the per-chapter entries

- **Header note** (top of file) now points to this changelog and flags that a second adversarial pass occurred.
- **"Where each thread plugs into your existing work"** section got three short additions (InnerControl → LatCH-F/LatCH-B line; SegTune + UNISON → native chroma/MIR conditioner line; Cautious Weight Decay → fusion-optimiser line) so the new entries are discoverable from the existing cross-reference map, not just buried in their chapters.
- **Honesty ledger** got five new bullets recording the corrections this pass made to the *newly added* entries themselves (TADA's fabrication scare, SegTune/UNISON's undersold maturity, Cautious Weight Decay's prior-art mischaracterization, and the DFoT/DEMON shortlist reasoning) — consistent with the ledger's existing purpose of keeping corrections visible.
- **Closing "Method inventory" line** updated from 84 to 95 candidates, itemizing the 11 new papers and noting 3 of them produced a second chapter entry.

## Tensions logged but not acted on (for the next real revision pass, not this one)

1. Ch5's "Teaching the door new manners" paragraph could use a one-sentence forward-reference to the Ch11 steering-locus-vs-adapter-locus distinction TADA surfaced.
2. Ch5/6's "selective deafness" Opening could gain a low-confidence footnote pointing to TADA's cross-attention bottleneck as an analogous (not identical) phenomenon.
3. Ch8's "you do not command which layer hears which word" line has a real, if narrow and off-backbone, counter-example now (UNISON) — worth a footnote if this chapter is ever revised with concrete ablation numbers.
4. Ch10's Opening Q5 (multi-window drift/re-anchoring) has three new scaffolding papers (DDE, DEMON, LMDM/ARC-Forcing) that sharpen but don't close the question — worth a footnote pointing to them as "nearest available scaffolding, still unverified on our stack."
5. Ch11's "where would you bond the laminate" Opening now has a concrete, numeric illustration (TADA's Concept-Sliders-at-the-bottleneck backfire) available if the chapter is ever revised with real citations instead of hypotheticals.
6. Ch13's "are you turning the peg, or just watching a number move?" Opening could cite InnerControl's Fig. 1 (RMSE-up-FID-down) as a real published instance of the same trap, via a different failure mechanism than the chapter's own example.
7. Ch15's gain-scheduling frontier could footnote InnerControl (meter needs a schedule, not a constant) and DEMON (the plumbing for an always-on per-frame signal is fast enough to be real) — neither resolves the frontier, both add texture.
8. A maintainer housekeeping item surfaced by the shortlist packet: TADA's specific "cross-attn layers 12-13 of 24" bottleneck finding is now in the Ch5 HeadRouter entry (fixed in this pass) — this item is resolved, listed here only for completeness since it was raised as a propagation gap.
