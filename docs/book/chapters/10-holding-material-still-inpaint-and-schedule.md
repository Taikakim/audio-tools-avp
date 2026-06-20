# Chapter 10 - Holding Material Still: Inpainting, Clamping, and the Schedule

A skilled restorer of old instruments does not refinish the whole body. They mask the varnish they want to preserve and work only the damaged patch, letting the surrounding wood hold its voice while the repair settles to match its edges. The mask declares, before any work begins, *this stays, that changes.*

This chapter is about that kind of control — steering by what you refuse to change, and by where you spend your effort. Hold three words apart, because they are forever conflated: the **melt** (how much you dissolve before carving — the *re-melting* dial of Chapter 9), the **mask** (what you freeze while carving), and the **schedule** (where you spend a finite step budget). The melt is a preprocessing dose; mask and schedule are this chapter's pair, and neither asks anything of the words you typed — they work below language. And unlike a LoRA or a full finetune — the body-reshaping methods of Chapter 12 — neither touches the instrument: they steer by *process*, gone the moment the run ends.

> **SHIPPED vs FRONTIER.** Two ideas here reach past the model's single-pass horizon and are constantly confused. **Sliding-window long-form** *ships* — it runs today in the repository you have been working in. **Per-moment molten-ness** — where each slice carries its own temperature — is a genuine, unproven *frontier*, held until Chapter 15.

## Three tools that look alike and are not

They blur together because all three reach into the same latent. Separate them by *when* each acts on the carve.

The **melt** (Chapter 9) happens *before* carving: you dissolve a real sound back to a chosen molten-ness and hand it to the carver as a starting point. A question of *how much*, applied to the whole clip uniformly.

The **mask** (inpainting, this chapter) interferes *during* the carve — some slices held, others left free. A question of *what*, a border drawn in time.

The **schedule** (the second half of this chapter) also acts *during* the carve, governing its pacing. A question of *where you spend your steps* along the molten-to-set journey.

They compose freely — melt a seed, hold part of it with a mask, pace it with a skewed schedule, all in one run — but they remain three different hands doing three different jobs.

## The mask: a decision made slice by slice

Recall the latent stream: a sequence of slices in time, roughly ten and a half per second, each standing in for the resonant signature of that moment. A mask is a decision applied slice by slice. *Hold this one. Free that one.* Held slices are clamped — pinned to their encoded values and re-pinned at every step, so that however the diffusion churns the free region, the frozen region never drifts. The free region starts as noise and is carved toward coherence.

What makes this more than copy-paste is that the free region is not carved in isolation. Self-attention (Chapter 5) broadcasts the frozen material to the free slices, the way an improviser hears the rhythm section: not a suggestion to weigh but a groove already there to play *to*.

So the free region must do two things at once: become coherent music in its own right, and coherent music *with* the frozen material at its borders — matching its energy, pulse, and harmonic colour.

## What a seam is made of, and how it fails

The seam where frozen meets free is where inpainting earns its keep. Picture it failing. You freeze a four-bar loop and ask the model to continue it, but expose the new slices with almost no frozen context — one or two slices at the border, so the model has barely heard the groove. The continuation comes back *plausible on its own* but wrong at the join: the kick lands a hair early, a faint discontinuity sits at the boundary — the audible equivalent of a colour-mismatched patch catching the light.

Two repairs, mapping onto the two tools of this chapter. First, **widen the context**: give the free region more frozen material to listen to, so self-attention has a real groove to lock onto rather than a fragment. Second, **spend more schedule in the molten range** — let the carve linger where pulse and energy are still negotiable, so the free region finds the frozen tempo before it commits detail around it. A seam fails when the free material can neither *hear* the frozen edge nor *deliberate* long enough to agree with it.

## Everything long-form unfolds from this one idea

**Continuation** covers everything you have and leaves only the future open. Freeze the whole clip; expose empty slices at the end; carve. The model hears the existing piece as ground and extends it — the natural next gesture.

**Gap-filling** holds both sides and frees the middle. You have a sound with a wound in it — a dropout, a clumsy edit. Freeze both shores and ask the model to bridge them, honouring lead-in and lead-out at once.

**Region regeneration** inverts that: hold the whole track *except* one phrase, and re-carve only that phrase. The restorer's craft at its purest — re-cutting one bar that never quite sat right.

And **long-form sliding-window rendering**[^shipped] is not a new idea at all — it is *mask applied repeatedly, with the frozen edge itself as the seed for the next window's carving.* Generate a window, freeze what you have, expose a little more future, carve; slide and repeat. Each new stretch is born hearing the settled stretch behind it, so the groove carries and the key holds. Melt, mask, and schedule are one system here: the frozen edge is a seed (the melt's territory), the clamp is a mask, and how cleanly each stretch joins is the seam question above, asked over and over.

[^shipped]: SHIPPED — runs today. Not to be confused with per-moment molten-ness (Chapter 15, frontier).

What happens after ten windows? A hundred? Coherence *can* drift. The frozen edge carries only a short memory, so over many slides the key can wander, the energy sag, the identity of the piece slowly dissolve — the long-form analogue of a photocopy of a photocopy. Whether re-anchoring cures this — periodically freezing a longer span, or re-injecting the original as a reference — is, for the shipped tools, an open question.

## The schedule: where you spend your steps

The sigma schedule is a quieter, less intuitive control. It touches no material; it governs the *pacing* of the carve.

Remember from Chapter 4 that sigma measures how molten the material is: high sigma allows coarse gestures — tempo, the broad bones of form; low sigma permits only fine surface work — timbre, the crispness of a transient. And from Chapter 3, that this is an *ordering*: the model roughs out large structure while molten and commits detail only as it sets. The schedule does not change that order — only how much of your finite budget (eight passes for fast work, fifty for careful) lands in each band.

Think of a bell-founder dividing labour between two stages of one casting. The pour and the cooling decide the bell's strike note while the bronze is still moving toward solid; the lathe-tuning that follows only refines a profile already largely fixed. Lavish your care on the mould and you settle the fundamental character before it sets; spend it on the lathe and you are perfecting partials on a casting whose core voice was decided hours ago.

What does this *sound* like? Skew hard toward the molten end and the model commits its rhythm and large arc more deliberately, often arriving at a more confidently shaped *gesture* — but with texture that can feel roughed-in, because few passes were left for the set range. Skew toward the set end and you get the opposite: structure locks early, sometimes generically, and the carve pours attention into surface — sharper transients and finer grain on a hastily decided frame. The schedule controls *what kind of decision* the model spends its budget on.

## Constraint and pacing, together

A mask is a border drawn in *time* — which slices are held. The schedule is a budget spent across *molten-ness* — where the effort lands. Different axes entirely, which raises a question: what if you could mask not only in time but in *molten-ness* too? Hold a region fixed through the high-molten passes, while the coarse structure of its neighbours is decided, then *release* it for the low-molten passes to re-voice its fine detail in sympathy with what settled around it. Freeze the bones, free the skin. That mask-in-sigma is a genuine frontier proposal — different from Chapter 15's, where each *moment* gets its own temperature; here, each moment gets its own *mask, scheduled in sigma*.

Three practical interactions a builder tends to discover the hard way.

**Mask and classifier-free guidance.** Crank CFG high over a masked run and that pressure acts only on the *free* region — the frozen part is re-pinned every step regardless. The quiet danger is at the seam: push hard enough and the free region obeys the words so insistently that it pulls *away* from the frozen edge it is meant to match. High insistence and clean seams are in mild tension, a trade yours to tune.

**Masking in the spectral domain.** Every example here masks in *time*. But Chapter 2 taught that the latent has internal structure, not just length — so could you hold part of that structure while freeing the rest, pinning the dimensions that carry brightness while letting density move? That freezes a *quality* across the whole clip rather than a *stretch* of it. The latent's axes wear no tidy labels, so this is far less charted than temporal masking — yet it sits squarely on Chapter 2's geometry.

**Mask and a control head.** A mask holds a region fixed; a control head (Chapter 13) biases a region toward a measured target — complementary by construction. Freeze a span, apply a head to the free slices, and it works the smaller territory you left open. The subtle part is again the boundary: a head pushing toward more bass energy also pushes *away* from seam-agreement, so its authority is lowest right where it competes with the clamp. Whether that tug is a nuisance or a feature is another empty square.

Both tools steer without instructing. The prompt, CFG, and a head all name a target; the mask and the schedule say nothing about the music. The mask steers by *subtraction* — removing a region from change shapes everything that grows around it; the schedule steers by *emphasis* — deciding which kind of decision gets the most deliberation. You can author a great deal of an outcome by choosing what to hold still and where to spend your attention.

## Openings

- What determines the quality of the *seam* where frozen and free material meet? Wider context and more high-molten schedule fix the common failure — but is the model *always* able to make the join inaudible, or are there materials — a sharp transient, a sudden key change — where nothing will hide the patch?

- What is the *minimum frozen context* a free region needs to settle coherently — a single slice, or seconds of it? Does a steady drone need less than a passage thick with rhythmic incident? This is testable on your own bench: shrink the frozen span window by window until the seam audibly breaks, on a drone and on a busy groove.

- Is clamping *free*? Does the model pay for holding material fixed, or does freezing slices actually *save* effort by shrinking the degrees of freedom the carve must resolve — making a masked run cheaper than a full regeneration, not dearer? *(speculative)*

- Could you genuinely combine a mask in time with a mask in *sigma* — holding a region fixed only during the coarse passes, then freeing it for the fine ones? What repair would that make possible that neither tool alone can reach? *(frontier)*

- In sliding-window long-form, everything downstream is born listening to the frozen edge — so could you deliberately *voice* that edge, choosing where to cut your window so the seed leads the model where you want the next stretch to go? And over a hundred windows, what re-anchoring would hold the piece's identity against drift? *(frontier)*
