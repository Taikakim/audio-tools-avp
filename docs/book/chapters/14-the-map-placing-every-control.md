# Chapter 14 - The Map: Placing Every Control and Finding the Empty Squares

You have a workshop full of tools. Each does something real. But a heap is harder to work from than a wall of labelled hooks — because the hooks tell you what is *missing*. This chapter hangs every control you own on a coordinate system precise enough that the gaps become visible, and visible gaps become buildable. We will not re-teach mechanisms; their workings live in the chapters behind you. We will only place them. The shift is small to state and large to absorb: stop thinking of your controls as a list, and start thinking of them as points in a space. A list you memorise. A space you navigate.

## Seven axes

A coordinate needs axes. Here are seven, and together they can locate any control method you have met or will ever invent. Read them not as a quiz but as seven questions you can ask of anything that claims to steer this instrument.

**Where does it intervene?** The most structural question. At the words going in, at the weights of the instrument itself, in the latent material mid-carve, in the pacing of the carve, or at the autoencoder's compressing ear and re-voicing mouth? Five sites; every method touches at least one, and some touch two.

**When does it act?** Training time — built once, before any performance, then simply *there* — or inference time, fresh at each performance, leaving no mark once the run is done. The difference between voicing a soundboard and choosing how hard to bow.

**What does it cost, and does it persist?** Braided with *when* but not identical: a training-time method spends data and compute up front and persists; an inference-time method spends almost nothing and lasts one run. The combinations are more interesting than the rule of thumb, as we will see.

**How broadly does it reach?** Holistic and sweeping — genre, mood, bearing — or surgical, pressing on one chosen dimension. A prompt reshapes the whole room; a bass-energy head leans on one wall. Neither is better; knowing which you need is half of choosing the tool.

**Open loop or closed?** Does the control set something and let go, or measure continuously and correct — a tuner that keeps listening? This is the axis Chapter 12 asked you to understand, because it divides how control *happens*. All training-time methods are open-loop by definition: built once, then forgotten. Most inference-time methods are open-loop by habit: set and released. Closed-loop — a measurement folded back into the carving, step after step — is rare, and worth hunting for.

**Can you stack it, dial it, peel it off?** Inference tricks layer freely and leave when the run ends. LoRAs are unusual among training-time methods in being stackable and reversible — blend two laminates, detach either; a full finetune is baked into the body. Reversibility is freedom, unevenly distributed.

**What currency does it spend?** Every control consumes some signal: words for the prompt, recordings for a LoRA, a measurable target for a head, a seed sound for SDEdit, a mask for inpainting. This axis is quietly the most generative of the seven, and it earns its own section below.

Seven questions. *Where, when, what cost, how broad, which loop, how reversible, what signal.* Hold them lightly — handles for thinking, not a form to fill out.

### Which axes are truly free, and which are tied

Treat the seven as independent and you will mis-read the map. *Where* is structural — it sets the others' range. *When* gates much of what follows: choose training time and you have already chosen open-loop, because a control built once cannot listen during a performance it will never attend. *Cost* rides along with *when*, and so, often, does *reversibility*: weight-baked finetunes resist removal, per-run tricks vanish on their own.

The genuinely free axes — the ones you can move without disturbing the others — are *where* and *what-signal*. You can change a head's currency from a number to a curve without touching anything else on the map. That independence is exactly why those two axes are where new controls are born. The braided axes describe the terrain; the free axes are where you dig.

## Every tool finds its address

Lay the axes down and the heap resolves into a constellation, with a pattern no single chapter stated.

The **prompt** sits at the input, inference-time, nearly free, holistic, open-loop, composable, spending words — the most accessible control, and by the same token the bluntest.

**Classifier-free guidance** sits one step over, at the sampling *process* rather than the input. That single difference is its whole character: CFG spends no new signal and does not change the prompt — it reframes how hard the prompt presses. Carry that forward — a control that reshapes the *force* of an existing signal rather than adding one will always feel like a knob, never like a new voice.

**SDEdit** — the re-melting of an earlier chapter — sits at the latent as initialisation, inference-time, free, holistic-leaning, open-loop, composable, spending a seed sound. **Inpainting** sits at the latent too, as a held mask, surgical in time, spending a mask. Two latent-site tricks, cleanly told apart by what they consume.

A **LoRA** sits at the weights, training-time, costly to make but reversible, stylistic, open-loop, stackable, spending recordings. **DoRA** sits at very nearly the same address — its split of magnitude from direction does not move it on this map. It is a *better-built* laminate, not a laminate in a new place: the map's way of saying that an existing square can be sharpened without being vacated. Not every advance is territorial, and telling refinement from exploration is part of the craft. Full **finetuning** shares the weights site and the training clock but flips two axes: not reversible, not cleanly stackable. Same neighbourhood, opposite permanence.

The **control head** sits at the latent mid-carve, inference-time, free per run, surgical, *closed-loop*, stackable, spending a measurable target. It is the only method in your current workshop that closes the loop — and that lone occupied closed-loop coordinate should make your eye drift toward every closed-loop address that nothing yet fills.

One method does not behave like the others: the **global conditioning vector** that rides alongside the timestep into the instrument. Like a control head, it sits at a latent site — but where a head touches one layer of the carver, the global vector reaches *every* block at once, re-voicing the whole network in a single gesture. "Holistic versus surgical" does not capture this; that axis is about the *output's* breadth, not the *reach into the body*. A whole-body hook and a single-layer hook can both be surgical in effect and radically different to build. Keep that distinction in your pocket; it marks a near-empty region of its own.

Step back and the constellation has a shape: nearly everything free and per-performance clusters at the latent or the sampling process; nearly everything costly and persistent clusters at the weights; the input holds almost only the prompt; and the autoencoder's ear holds almost nothing. Two crowded corners, a sparse middle, and a vacant edge.

### Why are some regions empty?

A blank region asks a question before it offers a tool: is it empty because something forbids building there, or because no one has gone? The autoencoder's ear is nearly vacant, and the honest answer is that we do not know which emptiness it is. The encoder and decoder may be *working as designed* — little reason to steer a part whose whole job is to translate without opinion. Or the design may be frozen by habit and waiting, a control reaching into how the latent grid is laid down perfectly possible and merely unattempted. Telling "blocked by a real limit" from "unvisited" is itself a skill, and the map lets you *ask the question precisely* even when the answer is still open.

### The cost-and-permanence landscape

"Training is costly, inference is free" is true on average and misleading in particulars. The corners that teach most are the off-diagonal ones:

- **Cheap to make, cheap per run** — the prompt.
- **Costly to make, free per run** — a LoRA or finetune; you pay once, then wield it for free. (Note the LoRA's asymmetry: expensive to *create*, instant to *remove* — cost and reversibility are not the same axis.)
- **Free to make, cheap per run** — a trained control head, and the tricks that need no training at all.
- **Cheap to make, permanent** — the interesting, under-explored corner: a method you could teach from a handful of examples that nonetheless *bakes in* and does not peel off. Most of this quadrant is empty. Ask why.

Holding cost and permanence apart, rather than collapsing them into "training versus inference," is what makes that fourth quadrant visible at all.

## The signal axis: a grammar of currencies

Of the seven, the currency a control spends is the one most likely to hand you a new tool. Five are already spent on your wall: **words** (the prompt), **recordings** (a LoRA), **a target number** (a head), **a seed sound** (SDEdit), and **a mask** (inpainting). Together they form something like a grammar — each a kind of thing the world can hand the instrument, each unlocking a different sentence.

Now ask: *what else is measurable that no control yet spends?* The book has already named several signals in passing and left them on the bench. A **reference timeseries** — a curve of some feature moving through time. A **beat grid** — a rhythmic skeleton. A **spectral template** — a target shape for the frequency content. A **MIDI line**, a **whole-track envelope**. None is invented territory; the instrument's surrounding world already produces them. They are *named but unconsumed*: signals with no hook to hang from.

This is the subtle move. A target number flattens all of time into one average; a target *curve* keeps time alive. What would it mean to spend a harmonic target — to hand a control a chord, not a brightness? A rhythmic grid — a pulse to lock to? A time-varying reference spectrum — to say not "brighter" but "*this* bright here, *that* bright there, breathing across the whole take"? Each unspent currency is an empty square with a musical meaning already attached, waiting for someone to build the hook.

## The empty square is a specification

Here is the turn the whole chapter exists to make. Sorting tools into bins is the boring half of a map. The living half is navigation, and the fact that matters is this: **an empty square is not a blank. It is a specification.**

Find a region no method occupies and you have not found nothing — you have found a precise description of a tool that does not yet exist. You already know where it hooks in, when it acts, what it spends, how broadly it reaches. The vague "I-wish-I-could" of wanting a new control has been replaced by coordinates.

**The pattern, in two examples.** Take the control head's address and change two coordinates: *confine it to a chosen molten-ness band*, and *let its target be a curve through time* rather than a single number.

- **Address:** latent, molten-ness-band-confined, inference-time, closed-loop, stackable, spending a measured shape through time.
- **Why nothing hangs there:** your current workshop has no such tool. But you know where it attaches, when it acts, and that what it measures is a curve, not a point. You have *specified* it.

Make one coordinate musical, so the gap stops being abstract. Today's brightness head presses one flat value across the whole take — average right, *shape* wrong. A time-varying target would let you ask for a dark, smouldering opening that blooms open at the drop and dims into the outro — an arc, not a setting. And why confine it to the molten high-sigma band? Because coarse structure — the *where* of the arc — is decided early, while the material is still fluid; fine detail is settled late. A control shaping a long gesture should speak while the gesture is being laid down, and hold its tongue once the instrument is merely polishing. The coordinate gap and the musical reason are the same fact, read twice.

Now a second empty square, built the same way — because the *method* matters more than either example. Pick an unspent currency: a **rhythmic grid**. Imagine a control that does not measure the latent but *biases* it toward landing its energy on a pulse you hand it.

- **Address:** latent during sampling, inference-time, open-loop (it imposes rather than corrects, unless you wire it to listen), surgical-in-rhythm, stackable, spending a beat grid.
- **Why nothing hangs there:** the beat grid is a signal the book has already named, and no control consumes it. You know what it takes in (a skeleton of pulses), where it presses (the molten latent), and when (early, where rhythmic placement is still negotiable). A second tool, specified without being built.

That is the method, and it is repeatable: name an address, name what is missing, then say what you would already know if it existed. Do this often enough and the wall stops being a record of what you own and becomes a map of what you could.

**The composition rule.** Specifying a tool is not the same as knowing it will play well with its neighbours, and the map gives you a rule of thumb. Two controls at the *same site in the same carving window* may interfere — two heads nudging the same molten latent at the same step are pulling one rope two ways. Two at *different sites* usually compose: CFG at the pacing and SDEdit at the initialisation are handled in sequence, not in parallel; a head at the latent and a LoRA at the weights never even meet. And two at the *same site but different molten-ness bands* usually compose — one shaping coarse structure early, another refining detail late, sharing the latent but never the moment. That said, a method touching *two* sites at once is worth watching: two hooks mean two handles, and they may pull in different directions. Extra precision and an extra way to surprise you arrive together; watch the coupling, not just the method.

One unspent signal sits in plain sight. A control head reading the latent and the carver's own attention are two readings of the same geometry: a head reads a latent slice and reports a number — a measurement *of* the material — while attention reads the *relationships between* slices and weights them, a score of mutual relevance, of how moments lean on one another. The first is a control you own; the second is a signal the instrument computes internally and spends only on itself. A buildable question we leave for the closing.

That is the map's real gift: it converts the ache of a missing control into an address you can name aloud. "This feels missing and I can't say why" becomes "this lives at *these coordinates*, and it feels missing because nothing occupies them." A frustration becomes an inventory.

What is known here is solid — the seven axes, the addresses of every method you own, the crowded corners and the sparse middle. What is frontier is most of what closed-loop, time-varying, molten-ness-banded, autoencoder-sited control could become. One square deserves a flag: the molten-ness-banded, time-varying head we specified sits in territory the next chapter walks straight into. The global-sigma constraint — one molten-ness shared by the whole gather — was shown earlier to be deliberate, a seam and not a flaw. Lifting it, making molten-ness a *field* that varies across the clip, is the first frontier this map reveals, and the next chapter will ask whether it has already been *partly built*. The map does not pretend these squares are filled; it does something better and shows you exactly where they are. Everything that follows — specifying a new control by its coordinates and having a collaborator realise it — is the next chapter's promise. You are learning the language in which to make that promise real.

## Openings

- Read down the *what-signal-it-spends* axis on your own wall. Which currency is least used — and which signal the world already hands you (a timeseries, a beat grid, a spectral template) has no control yet spending it?

- The autoencoder's ear and mouth host almost no controls. Is the part working as designed, with no reason to be steered, or has no one yet asked what a control at that site would even consume? How would you tell those two emptinesses apart?

- Is there a square structurally impossible to fill, where two axes contradict each other — and would discovering one teach you about the instrument's built-in limits rather than its unexplored ones?

- If a control spent a signal the instrument *already computes inside itself* — the attention weights that score how moments lean on one another, or the carver's running guess of the finished piece — what would it cost to expose and re-spend it as input to a new control? And since that signal is still being generated as the control reads it, would re-spending it close a loop onto a latent that is also being carved — a feedback circuit running on the instrument's own private thoughts?

- Of all the empty squares you can now see, which corresponds to a musical feeling you have wanted for years but could never name precisely enough to reach for? What are its seven coordinates?
