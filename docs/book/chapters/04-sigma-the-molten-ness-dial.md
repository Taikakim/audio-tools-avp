# Chapter 4 - Sigma: The One Dial That Says How Molten

Picture a glassblower's gather spinning on the end of the pipe. Fresh from the furnace it is entirely fluid — any shape is still possible, every decision still open. As it cools, the outer skin firms first, fixing the broad form while the interior stays workable; later still, the surface detail sets; and at some point the whole piece is glass, committed, done. What governs all of this is a single quantity: temperature — not the shape, not intention, not tools — just how hot it is, right now, in this moment of the work.

The instrument in front of you has a quantity that behaves exactly like that temperature. It is called the timestep, or sigma, and learning to feel it is the difference between a builder who watches generation happen and one who reaches into it.

## One Number From Molten to Set

Chapter 3 walked you through the carving itself: how SA3 starts in pure noise and, step by step, settles that noise into a coherent latent, each step a nudge rather than a decision. That chapter showed you the *walk*. Sigma is the *map* of it — the number that tells you, at any moment, where you stand. It runs from one — pure molten noise, every possibility open, nothing committed — down to zero, a finished latent ready to be handed to the decoder and turned into sound. Generation is nothing more than walking sigma from one end of that range to the other, pausing at each position to ask the model which way to move.

It is worth being precise about what sigma is *not*. It is not the shape, the prompt, or any musical content — only readiness, how much the material can still become something other than what it is. Two completely different pieces, at the same point in their respective carvings, share the same sigma. It carries no content whatsoever. It reads one thing and one thing only: how workable is this, right now.

And like temperature, sigma is felt by everything in the workshop. The model consults it at every step to decide how gently or boldly to move — when the material is molten it is free to make sweeping commitments, and when it is nearly set it dares only the lightest touch. It reaches into the network through a pathway called adaLN, which Chapter 6 will unfold in full; for now, simply hold the idea that sigma reaches into every block and *re-voices* it, so that the carver behaves differently when the material is molten than when it is nearly set. The model that pushes hot glass is, in a real sense, a different model from the one that polishes cold glass, even though it is the same network. Sigma is what tells it which one to be.

## The Asymmetry That Changes Everything

Here is the fact that makes sigma the most consequential single number in the instrument, and it is an asymmetry.

Early in the walk, when sigma is high and the material is molten, almost nothing is committed. The broad architecture of the piece — its tempo, the arc of its energy, where the sections fall, whether it builds or recedes — is still unresolved, still up for grabs. A push applied here has wide reach. It is coarse, but it is powerful: it can bend the whole form. This is the glassblower opening or closing the entire shape of the vessel while the gather is still soft all the way through.

Late in the walk, when sigma is low and the material is nearly set, that broad architecture is already baked in. You cannot move it anymore; it has cooled past working. What remains workable is only the surface — fine texture, the exact grain of a timbre, the crispness of a transient, the last sheen of detail. A push applied here is a polish, not a gesture. You are no longer shaping the vessel; you are smoothing its skin.

The two ends are not a wall between molten and set; they are the extremes of a continuous gradient, and the most interesting work happens in the middle. At mid-sigma the model might be locking the harmonic movement of a progression into place while the brightness of the chords still breathes, free to be tinted lighter or darker before it too cools. The coarse-to-fine ordering runs across the whole range, not in two jumps but as a slow handing-off from structure to detail.

This ordering is not something the builders of SA3 imposed by hand. The latent's high-dimensional geometry settles large-scale relationships before fine detail can cohere on top of them — not arranged by anyone but by the order in which diffusion itself works. Large-scale structure must be settled first because everything else is built upon it; fine detail can only resolve once the structure beneath it is decided. The glassblower does not *choose* to firm the broad form before the surface — that is simply the order in which glass cools, from the large toward the small, from the load-bearing toward the decorative. It is like blocking out the large arc of tension in a phrase before you decide where the vibrato sits: the arc has to exist before the ornament has anything to ornament.

For a builder this asymmetry is the whole game, because it tells you that *the same intervention means different things at different sigma.* A nudge toward "more bass energy" applied at high sigma reaches into the bones — it may reshape where the low end sits across the entire form. The identical nudge applied at low sigma can only touch the surface — it tints the timbre of a low end whose structure is already decided. Same hand, same pressure, two completely different tools, depending only on how molten the material was when you touched it. This is why the choice of *re-melting depth*, which we come to next, is your first and most direct way to steer the carving. And Chapter 13 takes this exactly: it asks whether a control head's nudge ought to know what sigma it is acting at — whether the gain that is right for shaping bones is the gain that is right for polishing skin. Hold that thought; it is one of the live ones.

## Where You Start Is a Choice

If sigma describes how molten the material is, then *where on the walk you begin* is itself a control — and a deep one.

The default is to begin at the top, at sigma one, in pure noise, and carve the whole way down. But nothing forces you to start there. You can take a real recording, encode it into the latent, and drop it into the walk partway down — at some chosen sigma — by melting it only partway back toward noise and then re-carving from there. This is the re-melting technique, and Chapter 9 develops it fully as a method; here I want only to use it to make sigma's meaning concrete, because re-melting will be your first hands-on way to *feel* sigma's asymmetry rather than merely read about it.

Drop your recording in at a low sigma — melt it only a little — and you are touching only its surface. The bones of the original survive: the rhythm stays locked, the harmonic skeleton unmoved, the broad spectral shape committed, because you never heated them past working. Only the timbre breathes; the carving can re-finish the skin and nothing more. Drop the same recording in at a high sigma — melt it deep — and you are offering its very bones up for reconsideration. The coarse structure dissolves back into something workable, and the model is free to rebuild it into something new, keeping only the faintest memory of what you poured in.

How much of what you brought do you want to honour, and how much do you want to dissolve? That question — honour versus dissolve — *is* the entire re-melting problem, solved by one number. A musician who internalises that high sigma touches bones and low sigma touches skin understands re-melting already, before reading a word about how it is done.

The same logic reaches into the schedule — the question of *how many steps* you spend in the molten range versus the set range, and whether you can bend that pacing toward structure or toward detail. That is a genuine and powerful intervention, but it belongs to Chapter 10, which owns the schedule in full. For now, simply notice that it exists: the path sigma takes from one to zero is not fixed in stone, and where along that path you concentrate your effort is yet another place to put your hands. I leave you the question and pass the tool downstream.

## The Seam: One Furnace for the Whole Gather

There is one fact about sigma that deserves to be said plainly, on its own, because it quietly carries the weight of several chapters still to come.

Sigma is currently a single, global value. One number, shared by the entire clip, at every moment, at every step of the walk.

Let that land. Every slice of the latent — the one holding the downbeat of the first bar, the one holding the tail of the final chord, the one holding the silence between phrases — is *equally molten or equally set* at any given step. They all cool together, in lockstep, governed by the same single reading. When sigma is at one, the whole clip is molten. When sigma reaches zero, the whole clip is set. No part of the piece is further along than any other. The instrument, as it ships today, has only one furnace temperature for the entire gather of glass.

This is not a law of nature. Nothing in the underlying process of diffusion *requires* sigma to be one number. It is a choice baked into the current design — a convenience, but also a simplifying assumption that every moment should advance through the carving together. You can already feel the strangeness of it against your craftsman's intuition. A real glassblower does not keep the whole gather at one temperature. They let the foot of a vessel set firm while the rim stays soft enough to flare; they work heat into one region and let another hold its shape. The hands move between hot and cool parts of the same piece constantly. The idea of a single temperature for the entire object, enforced everywhere at once, would feel to them like a constraint imposed by the furnace, not chosen by the artist.

And it is exactly that. Lifting it — giving each slice its own molten-ness, sculpting a *field* of temperature across the clip instead of a single shared value — is the mechanical basis of streaming and endless generation, and it is the proper subject of Chapter 15. There we will weigh honestly how far it has actually been built and how far it remains a hope — and, just as honestly, we will ask the question that comes *before* breaking the lockstep: what does the lockstep quietly buy us? The fact that every moment is equally molten at once may be holding a kind of coherence together — letting each slice hear and answer the others as equals — that a temperature-field would have to find some other way to preserve. I will not develop any of that here. I plant only the seam, and I hand the lifting of it forward.

But I want you to leave this chapter feeling that seam under your fingers. Because here is the habit this chapter most wants to give you: from now on, ask of *every* intervention you meet in this book — the re-melting, the schedule, the guidance dial, the masks, the control heads, the adapters — the same quiet question. *At what molten-ness does this act?* Does it reach in while the material is still soft enough to reshape its bones, or only late, when all that is left is to polish? That single question will sharpen everything that follows, because almost every control in this instrument has a sigma at which it lives — and many of them have not yet been asked what they would do at a different one.

A seam is not a flaw to be ashamed of. A seam is a place to put your hand.

## Openings

- If the same nudge means "reshape the bones" at high sigma and "polish the skin" at low sigma, then for any control you build, is there a *right* molten-ness at which to apply it — and how would you find that band by ear rather than by guesswork?

- When you re-melt a real recording, the coarse structure dissolves first and the fine detail last — but can you locate *where in the latent* that coarse structure actually lives, and re-melt only the part you want to reconsider while leaving the rest untouched?

- Sigma reaches every block of the model through the voicing mechanism of Chapter 6. If you disturbed sigma's expected path — started it somewhere unusual, or moved it in an unusual way — what would the voicing system, trained to expect a smooth walk from one to zero, assume that you were now violating?

- The whole clip cools in lockstep under one shared temperature. Before Chapter 15 asks how to break that lockstep, ask the prior question: *what does the lockstep currently buy you?* The question of per-moment sigma is also a question about moment-to-moment listening — if two moments sat at different temperatures, could they still hear each other the way they do when equally molten, and what coherence might quietly depend on their being in step?

- Is a single control head, applied across all of sigma, secretly several different tools wearing one coat — and how would you even *know* whether a head is responding to sigma's current state or behaving the same way regardless? (Does the answer tell you something about whether it *should*?) Chapter 13 takes this up directly.
