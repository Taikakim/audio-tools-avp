# Hands Inside the Instrument
### Understanding and tuning Stable Audio 3 — a book of questions, in the spirit of Bart Hopkin

*(working title)*

This is a book of **questions, not blueprints**. It will not hand you a settings file, a parameter table, or a step-by-step recipe. It tries instead to build your intuition about a particular instrument — Stable Audio 3 — thoroughly enough that you can ask your own questions and go discover the answers. Most sections end by *opening* a question rather than closing one. The closed answers will be obsolete before the ink dries; the open questions are yours to keep.

It is written for a **musician and maker** — someone who knows resonance, damping, envelopes, and filters under their hands, who follows the basics of digital sound, and who has begun to steer a generating machine (training small control heads, tuning guidance) and wants to understand the whole instrument well enough to invent new controls. It is *not* a manual for engineers; there is no code and no heavy math here, only the parts explained as roles and the places a builder can put their hands in.

**The one throughline.** Every part of this instrument is presented twice — as a thing that *works*, and as a *place where a new control could attach*. The recurring question is: **where, and how, can control be injected** — at training time and at inference time? By the end you should be able to dream a control you have always wanted and name it precisely enough — *where it hooks in, what it measures or biases, when it acts* — that someone, or something, could build it. A model's enormous capacity stays locked until the right words are uttered; this book is here to teach you to utter them.

→ **Start with the [Preface](00-preface.md).**

→ Once you know *which* control you want and need the real artifacts to build it, see the **[Control-Methods Sourcebook](CONTROL_METHODS_SOURCEBOOK.md)** — the nuts-and-bolts companion: every surveyed paper, repo, and experiment hung on these chapters, judged against SA3's exact construction (rectified flow, waveform-VAE latent, DiT), with a first experiment for each.

---

## How the book is built

> An apprenticeship, not a manual. It begins by asking what kind of instrument SA3 even is, descends into the raw material it works in (the latent), teaches the central act of carving order from noise and the single "molten-ness" dial that governs it, walks the body of the carver part by part, turns to the voice, hands you the dials already on the instrument, and then — the heart of the book — your own additions: what a LoRA really is, the deep difference between changing the body and bolting on a sense, your native territory of control heads, the master map that places every method and exposes the empty squares, the honest frontier, and finally the craft of dreaming a control and saying the right words to make it real.

---

## Contents

### Movement I — The Instrument and its Material
**[1. What Kind of Instrument Is This?](chapters/01-what-kind-of-instrument.md)**
Not a recording, synth, or sampler but a three-part instrument — a compressor-expander body, a carver that settles noise into a latent passage, and a translator that turns words into steering — whose real working material is the abstract latent stream, not the waveform. Establishes the throughline: every part and every seam hosts a handhold.
> *What would it mean to add a control that does not yet have a name — one that steers a feeling rather than a label?*

**[2. The Material: Inside the Latent](chapters/02-the-material-inside-the-latent.md)**
The SAME autoencoder as the prepared clay: ~4096 raw samples folded into one ~256-number slice every ~93 ms. What actually lives in those dimensions (oblique, unlabelled directions), the information floor, where encode/decode loss concentrates, and the decoder's voicing as its own intervention site.
> *Is there a way to feel a latent direction's grain before you build a control around it — to know whether you are carving with the wood or against it?*

### Movement II — The Act of Making
**[3. Carving Order from Noise](chapters/03-carving-order-from-noise.md)**
SA3 carves music from pure noise through a long walk of small directional nudges, each read from the half-formed state. Because it is many small steps, it is steerable: every pause is a doorway, and there are two pressure points — the noisy material mid-walk, and the model's running guess of the finished piece.
> *If every pause between steps is a doorway, are some doorways wider than others — does an early nudge reach further than a late one?*

**[4. Sigma: The One Dial That Says How Molten](chapters/04-sigma-the-molten-ness-dial.md)**
Sigma, the diffusion timestep, as a single "temperature" reading how molten vs set the material is — and why its high-to-low asymmetry (bones resolve before skin) makes the same intervention a different tool at different molten-ness. Teaches you to ask of every control: *at what molten-ness does this act?*
> *For any control you build, is there a right molten-ness at which to apply it — and how would you find that band by ear?*

### Movement III — The Body of the Carver
**[5. How Moments Listen: Attention as Form](chapters/05-how-moments-listen-attention.md)**
Self-attention as the instrument's ear for its own form (repetition, groove, return); cross-attention as the single locatable doorway where the prompt enters. The door consults *vectors, not words* — so anything you can shape into a sequence of vectors could walk through it.
> *What else could you walk through the cross-attention door — a brightness contour, a rhythmic-density curve, a control head's running report?*

**[6. Voicing the Whole Body: adaLN and the Stabilisers](chapters/06-voicing-the-body-adaln-and-norms.md)**
adaLN as a living act of voicing — sigma re-graduates every block's responsiveness — a third category of control distinct from LoRA and heads; and the quiet stabilisers (the norms) as the bracing that keeps expressiveness from tearing itself apart. Stability is the precondition of expressiveness, and sets the ceiling on how hard any nudge can drive.
> *Where exactly does steering turn into clipping — and does that ceiling move with sigma?*

**[7. A Sense of Time and a Scratch of Memory](chapters/07-sense-of-time-and-memory.md)**
Two small pryable seams: rotary positional embedding (the model's sense of *relative* time) and the free-floating memory tokens (an untethered channel through every layer). Each a candidate handle — with cheap diagnostics to tell a true handle from the mere appearance of one.
> *What makes a small mechanism worth prying open — and how would you decide, before sinking weeks in, whether a seam is a real handle?*

### Movement IV — The Voice
**[8. The Voice: Prompt and Conditioner](chapters/08-the-voice-prompt-and-conditioner.md)**
The text prompt and its T5-Gemma translator as the most direct but bluntest voice: where it sits, its corpus-shaped grain, how its grip shifts with molten-ness, and the duration conditioner as proof that a measured *number* can be first-class conditioning — a fork toward new native channels and toward control heads.
> *Which musical quantities are already measurable and stable enough to deserve their own numeric channel — and who decides what the instrument is trained to obey?*

### Movement V — The Dials Already On It
**[9. The Dials Already On the Instrument: Guidance and Re-Melting](chapters/09-dials-guidance-and-re-melting.md)**
Two free, reversible factory dials: classifier-free guidance (amplify the prompt's pull) and SDEdit re-melting (start the carve from a real sound dissolved to a chosen depth). Both coarse, open-loop — the outer wall of per-performance steering, and the open-vs-closed-loop divide the finer tools cross.
> *When you re-melt deeper, what dissolves first — rhythm, timbre, pitch, space — and what would a consistent order reveal about the latent's geometry?*

**[10. Holding Material Still: Inpainting, Clamping, and the Schedule](chapters/10-holding-material-still-inpaint-and-schedule.md)**
The mask (freeze chosen slices, grow coherent material around them) and the sigma schedule (where the finite step budget is spent). Separates *melt* (how much you dissolve) from *mask* (what you hold fixed) from *schedule* (where you spend steps); reframes sliding-window long-form as a mask reapplied with the frozen edge as seed. (This one *ships*.)
> *Is clamping free — or does freezing slices actually make a masked run cheaper than regenerating the whole clip?*

### Movement VI — The Builder's Own Additions
**[11. Changing the Body: What a LoRA Actually Is (and DoRA, and the Family)](chapters/11-changing-the-body-lora-dora-adapters.md)**
The most consequential site — the model's own weights. Full finetuning (rebuild the body, risk forgetting) vs LoRA (a thin removable laminate that biases a frozen body, with inference-time dials: strength, stacking, blending, peeling off). DoRA's separation of magnitude from direction; the adapter family by *where it attaches* and *what shape the correction takes*. A weight-change can teach a new bias but never a new *sense*.
> *If you wanted to teach not a style but a responsiveness — a predisposition to react to a signal the base model ignores — where would you bond the laminate, and how would you know you succeeded?*

**[12. Two Kinds of Teaching: Re-Voicing the Body vs Bolting On a Sense](chapters/12-two-kinds-of-teaching-body-vs-sense.md)**
The book's central distinction: re-voicing the body (finetune/LoRA — change the weights once, open-loop, unwatched) vs bolting a learned sense + feedback loop onto its outside (a control head that reads, compares to a target, and nudges *live*, closed-loop, detachable). The trade-offs, and how to choose, combine, or invent.
> *If both can press on the same dimension (say bass weight), how do you decide which to reach for — and are there dimensions only a live correction can hold?*

**[13. Adding a Tuning Peg: Control Heads and Training-Free Guidance](chapters/13-adding-a-tuning-peg-control-heads.md)**
Your native territory. A small separately-trained net that reads one musical quantity out of the latent and, run backwards at inference, becomes a live tuning peg. The two flavours of nudge, why gain must be matched to the latent's scale, what makes a feature learnable — and the load-bearing diagnostic: never trust the head's own climbing self-report; decode and measure with an independent yardstick.
> *Your honest test measures decoded audio against an independent yardstick — but that yardstick also trained the head. What third witness catches a shared blind spot?*

### Movement VII — The Map, the Frontier, and the Dream
**[14. The Map: Placing Every Control and Finding the Empty Squares](chapters/14-the-map-placing-every-control.md)**
Hang every control on a seven-axis coordinate system — *where, when, cost, breadth, which loop, how reversible, what signal*. Once every tool has an address, the crowded corners and vacant edges become legible, and an empty square turns into a precise specification for a control that does not yet exist.
> *Which currency on the what-signal axis is least spent — and which signal the world already hands you (a timeseries, a beat grid) has no control yet spending it?*

**[15. The Honest Frontier: Per-Moment Molten-ness and Live Feedback](chapters/15-the-honest-frontier.md)**
The two genuinely unfinished frontiers — a per-moment sigma *field* instead of one global value, and a control head as a continuous closed-loop tuner — plus the discipline of sorting any frontier claim into "blocked by engineering" vs "blocked by something fundamental." Honest about what is code-written-but-untested vs unproven at musical scale.
> *What other assumptions, quietly accepted as fixed since Chapter One, are conveniences waiting to be lifted — and which are genuinely load-bearing?*

**[16. Dreaming Up New Controls: Saying the Right Words](chapters/16-dreaming-up-new-controls.md)**
A genuinely new control is almost always an unvisited square on the map. The musician's real labor is translating a felt desire ("brighter without getting louder") into three coordinates — *where it hooks in, what it measures or biases, when it acts* — precise enough that an AI can build it. Specification as both an opening and a filter, grounded in the unsolved problems SA3 musicians face today.
> *Think of a control you have wanted and could never get. Can you name its three coordinates — where, what, when? If one resists naming, which one, and what does that tell you?*

---

*~44,000 words. A book of questions. Every chapter ends in **Openings** — questions left deliberately ajar.*
