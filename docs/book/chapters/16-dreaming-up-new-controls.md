# Chapter 16 - Dreaming Up New Controls: Saying the Right Words

Every chapter has been preparation for this one: the three-part body of the instrument, the clay it sculpts and the dial that says how molten it is, the controls already fitted and the art of bolting on new ones. The map from Chapter 14 pointed forward to this — naming the coordinates of a control that is not there yet.

That is a luthier's act, and the history of instruments is mostly recombination — sympathetic strings, bolts laid across the wire, the slide that fretlessly rides a fretted neck, each an unvisited crossing of things already understood, imagined before it could be built. Almost every genuinely new control is an empty square on a map you already hold.

## New methods live in the empty squares

A new control method is almost always an unvisited combination — one axis crossed with another, a familiar technique applied one layer deeper, or one sigma-band earlier, than anyone has tried. Three examples, to show how the empty squares feel under the hand.

A head trained to chase a single number — a target bass energy, a target brightness — is well understood (Chapter 13). A head trained to chase a *shape through time* lives in the square beside it: not one brightness value held flat, but a brightness *envelope* — dark in the opening, opening through the middle, closing at the end. The instrument flattens time by default; its heads report one number per window and cannot tell a sustained low note from a transient of the same energy. An envelope-chasing head refuses that flattening.

Two heads running independently are familiar — one for bass, another for brightness, each pushing. Two heads *coupled*, so steering one holds the other constant, is the square beside that: steer brightness up while the loudness head pushes back to hold loudness where it was. No prompt can do this; words shape a room, as Chapter 8 had it, and "brighter but no louder" is too coarse for language to enforce. The coupling lives in the latent, mid-carve, closed-loop — the same *holding-while-steering* the attention surgery of Chapter 5 showed at the prompt doorway, applied now to two trained senses instead of two attention paths.

A LoRA that teaches a style is common — the whole adapter family of Chapter 11 was built for it. A LoRA that teaches the model to become newly *responsive* to a signal it currently ignores is rarer — not "play more like this" but "from now on, *listen* to this new channel I am going to feed you, and let it steer you." Chapter 12 set the dividing line as a hinge — bodies can be re-voiced, senses can be bolted on — and a LoRA that teaches a body to *grow* a new sensitivity straddles both at once. One of the least-charted squares on the map.

There are wider vacancies still. **Stereo and spatial placement** is one — Chapter 2 noted that the material carries a left and a right, and that almost nothing built so far reaches in to *steer* that braid. A control that widened or narrowed the image, that walked a sound from one side to the other as the piece set, would sit in one of the largest empty squares on the bench.

None of these are recipes — each is a finger laid on a vacancy, not so you build it, but so you can now *feel where the vacancies are*.

## From a feeling to its coordinates

You do not start with coordinates. You start with a *feeling*: brighter without getting louder; a track that breathes wider as it goes; an instrument that finally listens to the groove you keep humming at it. The feeling is precise in your body; translating it into coordinates is a separate skill, harder than the examples make it look.

Consider "brighter without getting louder." Sit with it and you find at least three different controls. It might be a **coupled-head** problem — two senses, one pushed, one pinned, at inference, in the latent. It might be a **LoRA sensitivity** — you teach the body, once, that there is a brightness knob it should respond to, and loudness stays where the training data left it. Or a **prompt-fusion** problem — language and guidance that lean the carver brightward without the side effect, never touching weights or latent. Three sites, two timings, three costs; the feeling does not tell you which. *You* decide, and deciding means asking where you would rather intervene, and what you would have to spend to do it there.

The real work, then, is not the having of the wish but the disciplined narrowing of a felt desire onto named coordinates. And specifying a control is not *realizing* it: you may grow fluent at saying where it hooks in, what it measures, and when, and still need a collaborator — human or machine — to build the mechanism and work through the failure after the first attempt. That second labor is real, mostly unspoken, and not yours to do alone.

## What a specification asks of you, and what it closes off

To articulate a control is to name three things, precisely.

**Where** it hooks in — a named site from the map Chapter 14 laid out. You do not re-derive the map; you point at a hook and say *there*.

**What** it measures or biases. A head measures, so name the quantity *and* whether it is a single number or a shape through time. A LoRA biases, so name the direction, or the signal you want the model to grow responsive to. A guidance dial amplifies, so name the gap it leans on.

**When** it acts. Once, at training time, settling in permanently? Or freshly at each run? And if at inference, across the whole sigma walk or confined to a band — while the material is still molten, or only late, when the form is set and just the surface is workable? Chapter 4 taught that the same nudge is a different gesture at high sigma than at low; the "when" is where that lesson is spent.

But precision is also a *filter*. The moment you fix the site, you close off every control that would have lived at another. Name "the latent, mid-carve" and you rule out everything the prompt could have done holistically — the surgeon's knife instead of the architect's pen. Name "training time" and you give up changing your mind per performance. A specification converts a cloud of vague possibility into one buildable thing by *killing the alternatives* — and feeling which ones you are killing, and whether you mind, is itself part of the craft.

Some feelings resist this narrowing, which is information. Take a problem SA3 musicians are genuinely stuck on: **phase and timing**. The timestep is one global value — the whole clip shares a single molten-ness — so a control that wants *different moments at different molten-ness*, or that acts on the *relative timing between two events*, has no clean coordinate to land on. Specify "make the kick and the bass lock tighter as the track sets" and you need four or five coordinates at once: a sense that reads relative phase (which the heads, flattening time, barely do), a site that can act per-moment (which the global timestep forbids), a band of sigma where timing is still negotiable, and probably prior testing just to learn whether such a sense is trainable. When a specification balloons like this and leans on untested ground, that *hardness is a signal* — usually that you are pressing against one of the instrument's real structural limits, not your own vocabulary. The frontier is where the coordinates stop being clean.

## When the words were right and the mechanism still failed

This matters more than any success. You will, sooner or later, specify a control perfectly — site, measurement, timing all named and sensible — build it, and watch it fail. The most common shape: a head that reports brightness beautifully but, run in reverse to steer, moves nothing.

How do you know? This is the diagnostic skill of Chapter 13, the prerequisite for everything here. A sense that *correlates* with a feature is not a sense that *controls* it. Before you specify a control, verify that nudging your candidate mechanism moves the music, not just the readout — the peg that merely *reports* string tension while the string stays slack is the failure waiting at the end of every clean specification.

So when it fails, the next move is not despair and not a new wish, but to ask *which coordinate lied*. Was the measurement only a correlate (a Chapter 13 problem)? Was the site wrong — a layer too late in the carve to still bend the form? Was the gain mismatched to the latent's scale, so the nudge was real but inaudible? Each failure narrows the next specification, and a control you got working on the second try, after the first taught you which coordinate was wrong, is the ordinary way these things are found.

## The literal situation

"Saying the right words" is not encouraging poetry; it is the literal situation. The capacity of this instrument stays *locked* until someone describes a control precisely enough to build the sense that reads it and the loop that steers it. The latent already holds directions for brightness, density, attack, a hundred unnamed properties — Chapter 2 told you they are there, tangled and oblique, but not reachable by wishing. Run your dream along every axis Chapter 14 hung on the wall — where it hooks, what it measures or biases, when it acts, what it costs, whether it stacks or bakes in, which currency it spends — and you have written its specification. The placing and the specifying are one act.

The musician's gift — your gift — is knowing what control you *want*: not a mood but a behaviour the instrument should have. This dimension should bend while that one holds; this feeling should rise across the piece and fall at the end; the instrument should start *listening* to a channel it currently discards as noise. You have always known how to want like this. This book's only gift is the grammar to say it out loud — and the honesty that saying it is the start of the work, not its end.

## Openings

- Think of a control you have wanted and could never get. Can you name its three coordinates — where, what, when? If one resists naming, which, and what does that resistance tell you about the control? (known — the three-coordinate grammar is settled; this is an exercise in using it.)

- Take a single felt desire — "warmer," "more urgent," "looser" — and decide whether it is a coupled-head problem, a LoRA sensitivity, or a prompt-fusion problem. What did you have to know about the feeling before you could choose the site?

- When you fix a control's site and timing, list what you have just closed off. Do you mind? Is there a version of the same wish that would have wanted a different site entirely?

- If you coupled two heads so steering one held the other constant, which pair of dimensions would you bind — and could you verify beforehand that each head truly *controls* its feature rather than merely correlating with it? (speculative — no one has built a coupled-head constraint here, so whether the pinning holds mid-carve is genuinely open.)

- Pick a control that would need four or five coordinates instead of three. What is the extra hardness pointing at — your vocabulary still groping for clean coordinates, or the instrument itself refusing one? How would you tell them apart before spending weeks building? (frontier — the phase-and-timing squares are unmapped, and which side the hardness lives on is what no one has settled.)

- Imagine a head you specified perfectly that still failed to steer. Walk the coordinates — measurement, site, timing, gain — and decide which you would suspect first, and how you would test it. (known — correlate-versus-control is a diagnostic the instrument already lets you run, as Chapter 13 showed.)
