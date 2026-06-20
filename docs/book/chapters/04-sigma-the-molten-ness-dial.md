# Chapter 4 - Sigma: The One Dial That Says How Molten

Picture a glassblower's gather spinning on the end of the pipe. Fresh from the furnace it is entirely fluid — any shape still possible. As it cools, the outer skin firms first, fixing the broad form while the interior stays workable; later the surface detail sets; and at some point the whole piece is glass, committed, done. What governs all of this is a single quantity: temperature — not the shape, not intention, not tools — just how hot it is right now.

The instrument in front of you has a quantity that behaves exactly like that temperature. It is called the timestep, or sigma, and learning to feel it is the difference between a builder who watches generation happen and one who reaches into it.

## One Number From Molten to Set

Chapter 3 walked you through the carving: how SA3 starts in pure noise and settles it, nudge by nudge, into a coherent latent. That chapter showed you the *walk*; sigma is the *map* of it, the number that tells you where you stand. It runs from one — pure molten noise, every possibility open — down to zero, a finished latent ready for the decoder. Generation is nothing more than walking sigma down that range, pausing at each step to ask the model which way to move.

Be precise about what sigma is *not*. It is not the shape, the prompt, or any musical content — only readiness, how much the material can still become something other than what it is. Two completely different pieces at the same point in their carvings share the same sigma.

Like temperature, sigma is felt by everything in the workshop. The model consults it at every step to decide how boldly to move, reaching into the network through a pathway called adaLN, which Chapter 6 unfolds in full; for now, hold the idea that it reaches into every block and *re-voices* it. The model that pushes hot glass is a different model from the one that polishes cold glass, even though it is the same network. Sigma tells it which one to be.

## The Asymmetry That Changes Everything

What makes sigma the most consequential single number in the instrument is an asymmetry between its two ends.

Early in the walk, when sigma is high and the material is molten, almost nothing is committed. The broad architecture — its tempo, where the sections fall, whether it builds or recedes — is still up for grabs. A push here has wide reach: coarse, but powerful enough to bend the whole form. This is the glassblower opening or closing the entire shape of the vessel while the gather is still soft.

Late in the walk, when sigma is low and the material is nearly set, that architecture is already baked in; it has cooled past working. What remains workable is only the surface — fine texture, the grain of a timbre, the crispness of a transient. A push here is a polish, not a gesture — no longer shaping the vessel, only smoothing its skin.

The two ends are not a wall between molten and set but the extremes of a continuous gradient, and the most interesting work happens in the middle. At mid-sigma the model might lock the harmonic movement of a progression into place while the brightness of the chords still breathes, free to be tinted before it too cools. This coarse-to-fine ordering runs across the whole range, a slow handing-off from structure to detail.

This ordering is not something the builders of SA3 imposed by hand. The latent's high-dimensional geometry settles large-scale relationships first because everything else is built upon them. Think of casting a bronze bell. The founder does not *choose* to fix the bell's great curve — the profile that sets its fundamental pitch — before the fine wall thickness that colours its overtones. That is simply the order in which the cooling bronze gives up its freedom: the timbre has nothing to sit on until the body exists.

For a builder this asymmetry is the whole game, because *the same intervention means different things at different sigma.* A nudge toward "more bass energy" at high sigma reaches into the bones, reshaping where the low end sits across the entire form. The identical nudge at low sigma can only tint the timbre of a low end whose structure is already decided — same hand, same pressure, two different tools. This is why the choice of *re-melting depth*, which we come to next, is your first and most direct way to steer the carving. Chapter 13 takes this exactly: it asks whether a control head's nudge ought to know what sigma it is acting at — whether the gain that shapes bones is the gain that polishes skin.

## Where You Start Is a Choice

If sigma describes how molten the material is, then *where on the walk you begin* is itself a control — a deep one.

The default is to begin at sigma one, in pure noise, and carve the whole way down. But nothing forces you to start there. You can take a real recording, encode it into the latent, and drop it into the walk partway down — at some chosen sigma — melting it only partway back toward noise and re-carving from there. This is the re-melting technique, which Chapter 9 develops fully; here it is your first hands-on way to *feel* sigma's asymmetry.

Drop your recording in at a low sigma — melt it only a little — and you touch only its surface. The bones of the original survive: rhythm locked, harmonic skeleton unmoved, broad spectral shape committed. Only the timbre breathes. Drop it in at a high sigma — melt it deep — and you offer its very bones up for reconsideration: the coarse structure dissolves back into something workable, and the model rebuilds it, keeping only the faintest memory of what you poured in.

How much of what you brought do you want to honour, and how much to dissolve? That question *is* the entire re-melting problem, solved by one number. A musician who has internalised that high sigma touches bones and low sigma touches skin understands re-melting already.

The same logic reaches into the schedule — *how many steps* you spend in the molten range versus the set range, bending that pacing toward structure or detail. Where you concentrate your effort along the path from one to zero is yet another place to put your hands. That intervention belongs to Chapter 10, which owns the schedule in full.

## The Seam: One Furnace for the Whole Gather

One fact about sigma carries the weight of several chapters to come. It is currently a single, global value — one number, shared by the entire clip at every step of the walk.

Every slice of the latent — the one holding the downbeat of the first bar, the tail of the final chord, the silence between phrases — is *equally molten or equally set* at any given step. They cool together, in lockstep. The instrument, as it ships today, has only one furnace temperature for the entire gather of glass.

This is not a law of nature. Nothing in diffusion *requires* sigma to be one number; it is a simplifying assumption, baked into the current design, that every moment should advance together. It runs against your craftsman's intuition: no one working a real piece keeps the whole at one temperature, letting one region set firm while another stays soft enough to move. Lifting that constraint — giving each slice its own molten-ness, a *field* of temperature across the clip instead of a single shared value — is the mechanical basis of streaming and endless generation, and the proper subject of Chapter 15. I plant only the seam here.

But leave this chapter with one habit: ask of *every* intervention in this book — the re-melting, the schedule, the guidance dial, the masks, the control heads, the adapters — the same quiet question. *At what molten-ness does this act?* Does it reach in while the material is still soft enough to reshape its bones, or only late, when all that is left is to polish? Almost every control in this instrument has a sigma at which it lives — and many have not yet been asked what they would do at a different one.

A seam is not a flaw to be ashamed of. A seam is a place to put your hand.

## Openings

- If the same nudge reshapes bones at high sigma and polishes skin at low sigma, then for any control you build, is there a *right* molten-ness at which to apply it — and how would you find that band by ear? (speculative)

- When you re-melt a recording, coarse structure dissolves first and fine detail last — but can you locate *where in the latent* that coarse structure lives, and re-melt only the part you want to reconsider? (frontier)

- Sigma reaches every block through the voicing mechanism of Chapter 6. If you disturbed its expected path — started it somewhere unusual, or moved it in an unusual way — what would the voicing system, trained to expect a smooth walk from one to zero, assume you were violating? (speculative)

- The whole clip cools in lockstep. Before Chapter 15 asks how to break that lockstep, ask the prior question: *what does it currently buy you?* If two moments sat at different temperatures, could they still hear each other the way they do when equally molten — or does some coherence quietly depend on their being in step? (frontier)

- Is a single control head, applied across all of sigma, secretly several different tools wearing one coat — and how would you even *know* whether a head responds to sigma's current state or behaves the same regardless? (Does the answer tell you whether it *should*?) Chapter 13 takes this up directly. (speculative)
