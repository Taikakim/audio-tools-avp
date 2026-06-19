# Chapter 10 - Holding Material Still: Inpainting, Clamping, and the Schedule

A skilled restorer of old instruments does not refinish the whole body. They mask the original varnish they want to preserve and work only the damaged patch, letting the surrounding wood hold its voice while the repair settles in to match its edges. The mask is what makes that possible: it declares, before any work begins, *this stays, that changes.*

This chapter is about that kind of control — steering by what you refuse to change, and by where you spend your effort. Hold three words apart from the start, because they are forever conflated: the **melt** (how much of a sound you dissolve before carving — the *re-melting* dial of Chapter 9), the **mask** (what you freeze while carving), and the **schedule** (where you spend a finite step budget along the molten-to-set journey). One is a preprocessing dose; the other two are interferences with the tool already in motion. This chapter owns the latter pair. Neither asks anything of the words you typed. They work below language, in the material and in the pacing of the carve. And unlike a LoRA or a full finetune — the body-reshaping methods we reach in Chapter 12 — neither one touches the instrument itself. They leave every weight exactly as it was and steer by *process*, not by re-voicing the model. They are inference-time tools to their bones: free at each performance, gone the moment the run ends.

> **SHIPPED vs FRONTIER.** Two ideas in this chapter reach past the model's single-pass horizon, and they are constantly confused. **Sliding-window long-form** — covered here — *ships*. It runs today in the repository you have been working in. **Per-moment molten-ness** — where each slice would carry its own temperature — is a genuine, unproven *frontier*, and we hold it until Chapter 15. Where the text reaches for one or the other, a footnote will remind you which you are looking at.

## Three tools that look alike and are not

They blur together because all three reach into the same latent. Get them clear once and a great deal of later confusion never arises. The cleanest way to separate them is by *when* each one acts on the carve.

The **melt** — re-melting, the subject of Chapter 9 — happens *before* carving begins. It is a preprocessing choice: you take a real sound, dissolve it back to a chosen molten-ness, and hand the result to the carver as a starting point. It is a question of *how much*, applied to the whole clip uniformly.

The **mask** — inpainting, this chapter — happens *during* the carve. It interferes with the process at every step: some slices are clamped to their encoded values and re-pinned each pass, others left entirely free. It is a question of *what*, a border drawn in time.

The **schedule** — the second half of this chapter — also happens *during* the carve, governing its pacing. It is a question of *where you spend your steps* along the molten-to-set journey.

Melt is a choice you make before you pick up the tool. Mask and schedule are interferences with the tool already in motion. They compose freely — melt a seed, hold part of it with a mask, pace the carve with a skewed schedule, all in one run — but they are three different hands doing three different jobs.

## The mask: a decision made slice by slice

Recall what the latent stream is: a sequence of slices in time, roughly ten and a half per second, each a thin column of numbers standing in for the resonant signature of that moment. A mask is nothing more exotic than a decision applied to that sequence, slice by slice. *Hold this one. Free that one.* Held slices are clamped — pinned to their encoded values and re-pinned at every step of the carving walk, so that no matter how the diffusion process churns the free region, the frozen region never drifts. The free region starts as noise, as always, and is carved toward coherence.

What makes this more than copy-paste is that the free region is not carved in isolation. Self-attention — the model's sense of how every moment stands relative to every other, which you met in Chapter 5 — broadcasts the frozen material's presence to the free slices. The fixed region is heard as a given. It enters the conversation as established fact, the way an improviser hears the rhythm section: not a suggestion to weigh but ground to stand on. You play *to* the groove that is already there.

So the free region must do two things at once. It must become coherent music in its own right, and it must become coherent music *with* the frozen material at its borders — matching its energy, its pulse, its harmonic colour, the way the restorer's patch must match the varnish at its edge.

## What a seam is made of, and how it fails

That edge — the seam where frozen meets free — is where inpainting either earns its keep or betrays itself, so it is worth standing at it for a moment.

Picture the failure. You freeze a four-bar loop and ask the model to continue it, but you expose the new slices with almost no frozen context — one or two slices at the border. The model has barely heard the groove. The continuation comes back *plausible on its own* but wrong at the join: the kick lands a hair early, the brightness jumps, a faint spectral discontinuity sits right at the boundary — the audible equivalent of a colour-mismatched patch catching the light. The free material could not hear enough of the frozen material to agree with it.

Two repairs, and they map onto the two tools of this chapter. First, **widen the context**: give the free region more frozen material to listen to, so self-attention has a real groove to lock onto rather than a fragment. Second, **spend more schedule in the molten range** — let the carve linger where pulse and energy are still negotiable, so the free region finds the frozen tempo before it commits detail around it. A seam fails when the free material can neither *hear* the frozen edge well enough nor *deliberate* long enough to agree with it. Context and schedule fix exactly that.

## Everything long-form unfolds from this one idea

The remarkable thing is how much falls out of a single binary decision made over time.

**Continuation** is a mask that covers everything you already have and leaves only the future open. Freeze the whole clip; expose empty slices at the end; carve. The model hears the existing piece as ground and extends it — the natural next gesture, in the established groove.

**Gap-filling** holds both sides and frees the middle. You have a sound with a wound in it — a dropout, a clumsy edit. Freeze both shores and ask the model to bridge them, honouring the lead-in and the lead-out at once.

**Region regeneration** inverts that: hold the whole track *except* one phrase, and re-carve only that phrase. This is the restorer's craft at its purest — re-cutting one bar that never quite sat right while everything around it stays exactly as it was.

And **long-form sliding-window rendering**[^shipped] is the deepest lesson here, because it is not a new idea at all — it is *mask applied repeatedly, with the frozen edge itself as the seed for the next window's carving.* Generate a window. Freeze what you have. Expose a little more future. Carve. Slide, freeze, expose, carve again. Each new stretch is born hearing the settled stretch behind it, so the groove carries, the key holds, the energy arc continues. Melt, mask, and schedule are not three separate tricks here but one system: the frozen edge is a seed (the melt's territory), the clamp is a mask, and how cleanly each new stretch joins is a matter of schedule and context width — exactly the seam question above, asked over and over.

[^shipped]: SHIPPED. Sliding-window long-form runs today. Do not confuse it with per-moment molten-ness (Chapter 15), which is frontier.

What happens after ten windows? A hundred? Here the honesty the book promises matters: coherence *can* drift. The frozen edge carries only a short memory, so over many slides the key can wander, the energy sag or run away, the identity of the piece slowly dissolve — the long-form analogue of a photocopy of a photocopy. Whether re-anchoring fully cures this — periodically freezing a longer span, or re-injecting the original as a reference — is, for the shipped tools, an open and interesting question rather than a settled one.

## The schedule: where you spend your steps

The sigma schedule is a quieter control, and a less intuitive one. It touches no material. It governs the *pacing* of the carve.

Remember from Chapter 4: sigma measures how molten the material is. High sigma allows coarse gestures — tempo, energy arc, the broad bones of form; low sigma permits only fine surface work — timbre, the exact crispness of a transient. And remember from Chapter 3 that this is an *ordering*: the model roughs out large structure while the material is molten and commits detail only as it sets, so *which* features get decided at *which* molten-ness is fixed by the model's own coarse-to-fine habit. The schedule does not change that order. It changes how much of your finite step budget — eight passes for fast work, fifty for careful work — lands in each band of it.

Think of a bell-founder dividing their labour between two stages of the same casting. The pour and the cooling decide the bell's voice — its strike note, the great bones of its sound — while the bronze is still moving toward solid; the tuning that follows, lathe-cutting metal from the rim and inner wall, only refines a profile already largely fixed. Spend your care lavishly on the mould and the pour and you settle the fundamental character before it sets. Spend it on the lathe instead and you are perfecting partials on a casting whose core voice was decided hours ago. Same total labour; profoundly different distribution of it, and a profoundly different bell at the end.

What does this *sound* like? Skew hard toward the molten end and the model dwells long where tempo, energy, and form are still negotiable: it commits its rhythm and large arc later and more deliberately, often arriving at a more resolved, confidently shaped *gesture* — but with detail and texture that can feel roughed-in, because few passes were left for the set range. Skew hard toward the set end and you get the opposite: structure locks early, sometimes a touch generically, and the carve pours attention into surface — sharper transients, finer timbral grain, more polished texture sitting on a hastily decided frame. Because which molten-ness the model dwells in determines what kind of decision it is making, the schedule is, in a real sense, a control over *what the model attends to* — without your having said a word about the music itself.

## Constraint and pacing, together

Hold both tools in view at once, because their most interesting use is in combination.

A mask is a border drawn in *time* — which slices are held. The schedule is a budget spent across *molten-ness* — where the effort lands. Different axes entirely, and the moment you see that, a question almost asks itself: what if you could mask not only in time but in *molten-ness* too? Hold a region fixed during the high-molten passes, while the coarse structure of its neighbours is decided, then *release* it for the low-molten passes so its fine detail can be re-voiced in sympathy with what settled around it. Freeze the bones, free the skin. That mask-in-sigma is a genuine frontier proposal, mostly empty squares on the map, waiting for someone to articulate it precisely enough to build. It is worth holding beside Chapter 15's frontier: there we ask what happens when each *moment* gets its own temperature; here, what happens when each moment gets its own *mask, scheduled in sigma*. Related, but different frontiers.

Three practical interactions are worth naming, because a builder tends to discover them the hard way.

**Mask and classifier-free guidance.** Crank CFG high over a masked run and that pressure acts only on the *free* region — the frozen part is re-pinned every step regardless, so guidance cannot move it. The quiet danger is at the seam: push hard enough and the free region obeys the words so insistently that it pulls *away* from the frozen edge it is meant to match, and the join suffers. High insistence and clean seams are in mild tension; that trade is yours to tune.

**Masking in the spectral domain.** Every example here masks in *time*. But Chapter 2 taught that the latent has internal structure, not just length — so could you hold part of that structure while freeing the rest, pinning the dimensions that carry brightness while letting density move? That freezes a *quality* across the whole clip rather than a *stretch* of it. The latent's axes wear no tidy labels, so this is far less charted than temporal masking, but it sits squarely on Chapter 2's geometry — one of the more inviting empty squares on the map.

**Mask and a control head.** A mask holds a region fixed; a control head (Chapter 13) biases a region toward a measured target — complementary by construction. Freeze a span, apply a head to the free slices, and the head simply works the smaller territory you left open. The subtle part is again the boundary: a head pushing toward more bass energy is also pushing *away* from seam-agreement with the frozen edge, so its authority is lower right at the border, where it competes with the clamp. Whether that tug is a nuisance or a feature is another empty square.

Both tools share a quality worth naming. They steer without instructing. The prompt tells the model what you want; CFG sets how hard to insist; a head measures and corrects toward a target. But the mask and the schedule say nothing about the music at all. The mask steers by *subtraction* — removing a region from change shapes everything that grows around it. The schedule steers by *emphasis* — deciding which kind of decision gets the most deliberation. You can author a great deal of an outcome by choosing what to hold still and where to spend your attention, never once naming the sound you are after.

## Openings

- What determines the quality of the *seam* where frozen and free material meet? We saw it fail from too little context and saw two fixes — wider context, more high-molten schedule at the boundary. But is the model *always* able to make the join inaudible, or are there materials — a sharp transient, a sudden key change — where no amount of context or schedule will hide the patch?

- What is the *minimum frozen context* a free region needs to settle coherently — a single slice at the border, or seconds of it? Does that minimum change with the material: does a steady drone need less context to extend than a passage thick with rhythmic incident? This is testable on your own bench — shrink the frozen span window by window until the seam audibly breaks, on a drone and on a busy groove, and you will have measured it for your own corpus.

- Is clamping *free*? Does the model pay for holding material fixed, or does freezing slices actually *save* effort by shrinking the degrees of freedom the carve must resolve — making a masked run, in principle, cheaper than regenerating the whole clip rather than dearer? *(speculative)*

- Could you genuinely combine a mask in time with a mask in *sigma* — holding a region fixed only during the coarse passes, then freeing it for the fine ones? What repair would that make possible that neither tool alone can reach? *(frontier)*

- In sliding-window long-form, everything downstream is born listening to the frozen edge — so how does the *character* of that edge shape what follows, and could you deliberately *voice* the edge, choosing where to cut your window so the seed you freeze leads the model exactly where you want the next stretch to go? And over a hundred windows, what re-anchoring would hold the piece's identity against drift? *(frontier)*
