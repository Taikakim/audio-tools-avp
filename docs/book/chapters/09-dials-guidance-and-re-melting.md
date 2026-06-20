# Chapter 9 - The Dials Already On the Instrument: Guidance and Re-Melting

Before you train anything — before you bolt on a single tuning peg — go back to the instrument as it shipped and notice the two dials fitted at the factory. They cost nothing to turn, and a player who never opens the case can still get an enormous range of voices out of them. One reaches into the *sampling process* — the carve itself — and changes how hard the prompt presses. The other reaches into the *initialisation* and changes how much of a real sound you let dissolve before carving begins.

They are the *coarse* dials, which is why this book does not stop here. CFG offers "push harder," SDEdit "melt deeper"; both act on the whole output at once, with a single number. The chapters ahead refine that toward *precision* — targeting one feature instead of all (the control heads), applying force only at a chosen moment (sigma-banded guidance), sculpting one region instead of the whole clip. But holding two dials of such different principle side by side teaches something about control that neither teaches alone.

## The First Dial: Classifier-Free Guidance

Every time the instrument generates with a prompt, it runs the carving process *twice* at every step: once with your words steering, once completely blind, working from nothing but its own sense of what music is.

Why twice? Because the difference between the two runs is precisely, and only, the prompt's *pull*. The blind run shows where the material drifts on its own; the prompted run shows where the words want it to go; subtract one from the other and what remains is the authority of language over this moment of sound. And it must be recomputed each step, since the blind drift depends on where the material currently is — itself a product of the seed and everything the prompt has already nudged into place.

Classifier-free guidance — CFG — is the dial that decides how loudly you amplify that gap. At a guidance of one, you take the prompted run and go. Turn it up and you push the material past the prompted prediction, leaning hard into the words. Turn it down toward zero and the prompt's pull goes slack, the carving following the blind run into the model's own instinct.

Recall from Chapter 5 that the prompt enters through one specific door, the cross-attention. CFG is the dial on *that* door: it does not add words or change their meaning, only amplify the difference the door already made — raising not the prompt's volume but its *consequence*.

The temptation is to treat this as a quality knob — more guidance, more obedience, therefore better. But the relationship is not monotonic. Press too hard and you over-correct, too literal — the audio equivalent of a mix where every track is slammed to the ceiling and nothing breathes. Relax too far and the music wanders toward whatever the blind run happened to want. And the low end breathes for a reason: high guidance follows the prompted prediction tightly, so the spread of outputs across seeds collapses inward; low guidance leans on the freer blind run, so it widens. The variety you lose at high CFG is not noise being cleaned up — it is genuine alternatives squeezed out.

So the sweet spot is not a number you look up — CFG is a *voicing* act, not a setting. It moves with the prompt (a dense, over-specified one wants a *gentler* hand; a sparse one a firmer hand to register at all) and with the *instrument*, because it is calibrated against the *scale* of the latent being pushed — a value that whispers on SA3's larger model can shout on a smaller sibling, so cookbook numbers borrowed across models mislead. (Chapter 13 takes up *calibrating force to the latent's scale* properly; for now, distrust borrowed settings.) Finding the right guidance is like a bell-founder tuning a finished casting: not by a chart that says "grind 0.4 millimetres from the inner rim," but by striking it, hearing where the partials sit, taking a little metal off, striking again.

And the dial stops being a single thing. Recall from Chapter 6 that sigma already re-voices every block of the network through the adaLN modulation; CFG re-voices it too, but by *how hard the prompt presses*. Whether the two cooperate or fight is a real open question, and it bites because the prompt's pull is not constant across the walk. From Chapter 3 you know the carve proceeds coarse-to-fine: the molten beginning decides large structure — form, tempo, the skeleton — while the set end leaves only surface free to move. A hard push early therefore shapes *structure*; the same push near the set end shapes only *texture*.

You can exploit that asymmetry. Want a generation locked to a tempo but with the timbre loose and alive? You would want a *firm* hand early, while structure is decided, and a *lighter* hand late, while texture settles. The instrument ships with one number for the whole carve, so you cannot do this with the factory dial alone — but knowing the dial *means different things at different molten-nesses* is exactly the intuition the later sigma-banded controls are built to act on.

## The Second Dial: Re-Melting a Real Sound

The second dial reaches into a different part of the instrument — not the carving but the block you start carving from.

Ordinarily, generation begins from pure noise: a latent of maximum molten-ness, sigma at one, and the Chapter 3 walk proceeds from there. Even that bare draw is a quiet control surface — *which* random block you pour in (the seed number) fixes which performance you get, the cheapest dial of all. But no law says the block must start formless. You can instead begin from a *real* piece of audio — encode it into the latent the instrument speaks (Chapter 2), partly re-melt it, and carve from *that*.

This is SDEdit — audio-to-audio re-synthesis. The seed sound is *not* a prompt: it does not enter through the cross-attention door but as a *physical initialisation*, the literal starting material of the carve. So seeding *and* prompting at once gives two concurrent steering channels on the same latent: the seed sets where the material begins, the prompt which direction the carving leans.

How far you re-melt the block is the second control, layered over the first. "Re-melting" is a sigma choice — you already own its meaning from Chapter 4. You pick a point on the molten-ness walk, drop your real sound in, and let the carve run to set. The depth of that drop is everything.

Melt *shallow* — near the set end, at low sigma — and you barely disturb it; the carve only re-settles the outermost skin. This is the zone for *touch-up*: cleaning a noisy recording, nudging a timbre, re-skinning a sound while its rhythm, phrasing, and identity stay intact.

Melt *deep* — near the molten beginning, at high sigma — and you offer the seed's very bones up for reconsideration. Only the coarsest structure survives; the model rebuilds everything finer from scratch, guided by the prompt and its own sense of music. This is the zone for *gesture theft*: keep the broad arc and rough rhythmic feel, let the instrument rebuild the actual voice, and you return a performance that merely *moves* like the one you brought. The middle band keeps the seed's larger phrasing while its surface is up for grabs — recognisably related to the source, audibly a new thing.

A deep melt has a quieter, diagnostic use too. Re-melt almost to the bone and listen to what comes back changed versus unchanged: *what does the latent preserve when everything fine is dissolved?* Whatever survives is what the encoder's compression (Chapter 2) and the diffusion's restructuring together consider *essential* — the instrument's own sense of what a sound fundamentally *is*.

So SDEdit is two independent controls in one: what you pour in (the seed) and how much you let dissolve (the melt-depth). The space is a grid, not a line, mostly unexplored but not uniformly inviting. The *safe* corner is shallow melt with a simple, clean seed: a light, predictable touch-up. The *experimental* corner is deep melt with a complex, busy seed: the instrument must rebuild a great deal while holding a tangled gesture, and may hold the wrong thing or lose coherence — thin or ambiguous seeds melted deep tend to fail outright, too little structure to anchor the rebuild. Start safe and scan toward the experimental corner one axis at a time, so that when something interesting happens you know which dial did it.

Do not confuse SDEdit with masking (Chapter 10): SDEdit re-melts the *entire* seed uniformly — every slice gets the same molten-ness, because sigma is still a single global number (Chapter 4) — whereas masking freezes part of the seed entirely and regenerates the rest from noise. One is a dial of *depth*; the other a decision of *where*. Chapter 10 owns that second tool; here it is enough to keep the two from blurring.

## Two Dials, One Boundary

These two dials are profoundly unlike each other. CFG lives in the *sampling process*: it changes how the carve behaves, needing no input but words and a number. SDEdit lives in the *initialisation*: it changes what the carve starts from, needs a seed sound and a melt-depth, and ties the output to a real piece of audio you brought. One is a pressure; the other a starting point.

Yet they share an address. *The Map* (Chapter 14) will lay out axes for placing any control — where it intervenes, when, what it costs, how far it reaches. On every axis that separates these factory dials from the heavier machinery *later* in the book, they agree: both act at *inference time*, cost nothing beyond the run, and are perfectly reversible — turn the dial back and the instrument is exactly as it was.

And both are *open-loop* — the deepest thing they share, and the line the rest of the book is written across. You set the guidance and melt-depth, let the carve run, and no tuner in the loop measures the result and corrects it mid-flight. That is not a limitation of inference-time controls in general: the control heads of Chapter 13 also run at inference time and reverse cleanly — yet they *close the loop*, measuring a musical quantity as the carve proceeds and correcting toward it. The real divide is not training-time versus inference-time but *open-loop versus closed-loop* — set-and-forget versus measure-and-correct — and these factory dials sit firmly on the open-loop side. Chapter 12 returns to this boundary to ask what kind of teaching each side calls for.

Together they mark the *outer wall* of free, per-performance, open-loop control: inside it you can obey or breathe, start from noise or from a remembered sound, preserve much or regenerate most — all without altering a single weight. The chapters that follow step outside the wall; some open the body and ask for data, compute, and permanence, some bolt on a new sense and close the loop. Knowing where this boundary lies is what lets you judge whether you need to cross it.

## Openings

- The blind, unprompted run is a real generation in its own right. Does it produce something coherent and musical, or closer to static? Whatever it produces tells you how much this instrument knows about music *without any words at all* — and it matters practically, because a low-guidance generation or a prompt-free SDEdit leans on that same blind run. If it makes music, the seed can carry a re-melt alone; if it makes mush, the prompt is doing more work than it looks.

- Seed and melt-depth are two separate controls. Hold one fixed and scan the other — one recording melted shallow to deep, or one melt-depth across a row of seeds. What does that grid sound like as a *space*, and where do the interesting failures live, between the safe and experimental corners?

- When you re-melt deeper, what dissolves *first* — rhythm, timbre, pitch, sense of space? (frontier) A consistent order would tell you something about the geometry of the latent itself: which musical properties live in the coarse structure that survives a deep melt, and which only in the fine surface a shallow melt preserves.

- CFG and SDEdit turn at the same time. Do they compound, interfere, or ignore each other? (speculative) Can you reason it from first principles — a deep melt wants the prompt to rebuild what was dissolved and may want firm guidance, a shallow melt wants it slack — or are you left auditioning combinations until your ear finds the ones that work?

- If a hard push means *structure* at the molten beginning and *texture* at the set end, what would you want the guidance to *do* across a single carve — rise, fall, hold, or move in some shape keyed to the music? The factory gives one number for the whole walk. Once you can imagine the curve you wish you could draw, you have already half-specified a control that does not yet ship — exactly the kind of wish the rest of this book is here to help you say out loud.
