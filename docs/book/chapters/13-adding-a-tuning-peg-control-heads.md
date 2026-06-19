# Chapter 13 - Adding a Tuning Peg: Control Heads and Training-Free Guidance

Imagine you could bolt a contact microphone onto the body of your instrument — not to amplify it, but to measure one thing: a single resonance, the tension in one string, the strength of one overtone. And then imagine you wired that microphone into a tiny servo that watches the reading and turns a bridge pin until the measurement matches a target you dialled in before the performance. You would have built, in effect, a new tuning peg — one that governs a dimension the instrument shipped without a knob for. That is, very nearly, what a control head does inside SA3.

This is your home territory, so let us be precise about what the device actually is and how it earns its authority — because the difference between a peg that turns and a peg that only *appears* to turn is the sharpest thing you will learn in this book.

## A learned sense, trained once

A control head is a small, separate network. It does not generate anything. Its entire job is to *read* — to look at the latent stream the carver is shaping and report back a single musical quantity. How much energy lives in the low end. How bright this passage is. How densely events are packed across a window of time. One number, or a small handful, out of that whole column of unlabelled latent dimensions you met back in Chapter 2.

That reading is not free. The head has to be *trained* to hear its one thing, and this is the first place your hands go in: at training time, you choose which feature to make audible to the head. You take a body of latents, measure the true value of your chosen feature in the corresponding audio — the actual bass energy, the actual brightness — and teach the head to predict that value from the latent alone. When it can read accurately, you have a sense: an ear that hears one thing with precision and ignores everything else. It learns to project the messy geometry of the latent down onto the single axis you care about — the map from a direction nobody drew to a number you can read.

But notice already what the head throws away. It reads *one scalar per window* — one bass-energy value for a whole span of slices. Inside that window, time is flattened: a sustained low drone and a punchy transient that average to the same energy will read identically. The head cannot tell them apart, because you never asked it to. Hold onto this; it is quiet now but it shapes the ceiling of everything that follows, and the last of the Openings comes back to it.

## Running the sense backwards

Training the head is only the first act. The second — the one that makes it a control rather than a meter — is running it *backwards*.

At inference time, you set a target. Say a specific bass energy. Then, during the carving walk, at each step you stop and ask the head to read the current latent. It reports a value. You compare that report to your target, and the difference between them — *target minus report*, the gap — becomes a push. A small nudge applied directly to the latent, in the direction that would close the gap, that would make the head's next reading land closer to your mark.

This is the whole engine. Target minus report drives the nudge. It is a feedback loop in the truest sense: not set-and-forget, but measure-and-correct, alive during every step of every performance. This is exactly the contrast Chapter 12 drew and the reason it matters here: a LoRA retunes the instrument's body *once*, for every future performance; a head bolts on a tuner that measures-and-corrects *during* each performance, live, and is gone the moment you stop. The generative model itself is never touched. Its weights do not move. You have not retuned the body — you have bolted a tuning peg onto the outside of a finished instrument and routed your target through it.

The injection point is the *latent during sampling*. That is where the nudge lands. The live control input — the thing you turn in real time — is the *target*. Everything else about the head is fixed once trained; the target is the dial under your hand during the performance.

And here a mechanical detail matters more than it looks. After you nudge the latent, the model reads that *nudged* latent fresh on the very next step — self-attention, the part that lets every moment listen to every other moment, now listens to a latent you have already pushed. Head and attention are in conversation. Your nudge is not a private adjustment the model never sees; it becomes part of what the model attends to next, which means a nudge can either compound (the model picks up your direction and runs with it) or be damped (the model's own velocity pulls back toward where it was heading). Whether it compounds or dampens depends a great deal on *which* of the two flavours of nudge you choose.

## Two flavours of nudge

Recall the two pressure points from Chapter 3: the noisy material as it sits mid-carve, and the model's running guess at the finished piece. A head can press on either, and these are the two flavours of nudge. They are not interchangeable, and choosing between them is a tuning act.

The first flavour presses on the **molten material directly**. Mid-carve, the latent is still partly noise — still warm, still moving. You read it, compute the gap, and shove the noisy latent itself toward your target before the next step. It is immediate, but you are working against a vibrating thing: some of your push gets smeared by the motion already underway, and because the model re-reads the pushed latent next step, a too-eager nudge here can be amplified into incoherence.

The second flavour presses on the **projected clean signal** — the model's running estimate of where all this is heading, the finished latent it currently believes it is carving toward. You let the model form that estimate, then nudge *the estimate* before it is used to take the next step. You are speaking in the same currency the model uses to plan: you nudge the goal, and the carver finds its own graceful path there. This tends to be gentler and more coherent, because the model's own momentum carries the adjustment forward rather than fighting it.

Which one has more leverage at a given molten-ness is, honestly, still open. There is a reasonable intuition — that pressing the clean estimate earns more authority early, when the destination is still negotiable, and that pressing the molten material does little once everything is nearly set — but treat that as a hypothesis to test on your own features, not a law. The right answer may differ for bass energy and for brightness.

## Matching the gain to the latent's scale

Now the part that is *solid*, that you can take to the bank, and that is missed more often than any other single thing.

The nudge has a strength — a gain, a number that says how hard to push for a given gap. And that gain has to be matched to the scale of the latent it is pushing on. This is not optional polish; it is the difference between a peg that turns and a peg that does nothing or snaps the string.

Here is the concrete fact. The latent of the large SA3 model carries roughly **ten times** the numerical magnitude of the latents in earlier Stable Audio versions — bigger numbers in every slot. So a gain of, say, 0.01 that was perfectly voiced on SA1 becomes effectively inaudible at the same setting on SA3-large: your nudge whispers, the head reads, reports, pushes, and nothing audible moves. Flip it the other way and overcorrect — push the gain up to 0.1 to compensate — and you can blow the material clear out of coherence. That is worth defining precisely, because "breaking" is not vague. Two things break. First, the push can breach the **stabiliser ceiling**: it drives the latent harder than the norms can absorb, and downstream they clamp or overflow. Second, even short of that, an oversized nudge destroys **local coherence** — neighbouring slices that should agree (a bass note and its own continuation a moment later) get shoved in inconsistent directions, and the carver can no longer settle them into music. You get smear, noise, a ruined take.

So gain is not a setting to look up. It is a tuning act, and it is *instrument-specific*. When you carry a head — or any guidance technique — from one model to another, the first thing to re-voice is the gain, scaled to the new latent's magnitude. And recall the ceiling from Chapter 6: the stabilisers — QK-normalisation, the RMS-like norms, the dynamic-tanh damping that keeps attention and levels from blowing up — set the upper bound on how hard you can drive any nudge before the instrument loses coherence. Gain lives in the room between *too quiet to hear* and *hard against that ceiling*. Finding that room is the voicing.

Two threads worth carrying about where that room sits. First, from Chapter 6: the stabilisers' authority is not necessarily constant across the walk — the damping may bite harder at some molten-nesses than others, which means the headroom beneath the ceiling can tighten and loosen as the material sets. Second, from Chapter 4: the carving walk has a single global furnace temperature — *one* sigma shared by every moment of the clip — and the schedule decides how that temperature descends. If you skew the schedule toward the molten end or toward the set end, you change which molten-nesses get the most steps, and there is every reason to think the right gain rides that curve rather than sitting fixed. A nudge at high sigma is plausibly a different-sized gesture than the same nudge at low sigma. Whether you confine the nudge to a chosen *sigma-band* at all — pushing only while the material is still molten, then letting go — is itself one of your tuning choices, and it is the same single global temperature from Chapter 4 you are choosing to act within. You cannot, today, hold one moment molten while another sets; the band you pick applies to the whole clip at once.

## What is even learnable as a head

Not every musical feature you might want will submit to this treatment. Before you build a head, it pays to ask whether the feature is *tractable* — whether it can become a sense at all.

Three questions sort the learnable from the hopeless. First: does the feature leave a **clear, consistent signature in the latent**? Bass energy does — low-frequency weight survives the compression legibly, so a head can find it. Some subtle perceptual quality may be smeared across the lossy compression so thoroughly that there is no stable footprint to read. Second: is that signature **geometrically stable across prompts and seeds**? A head that reads brightness beautifully on techno but reads garbage on solo piano has learned the feature *as it appears in one corner* of the space, not the feature itself. Third: do the head's reported values **spread over a useful range**, or do they crowd into a corner?

That word *useful* deserves a number. If your feature is normalised to a [0,1] scale, a head whose outputs across the whole training set live between, say, 0.3 and 0.7 has only ever exercised the middle — it has never seen, and cannot confidently steer toward, the extremes you will most want (the truly bass-heavy, the truly thin). You want spread approaching the full 0.0 to 1.0, or at least wide enough that the targets you intend to dial fall comfortably inside it. As a rough threshold: if the ratio of the head's smallest to largest reading huddles above, say, three-quarters — most readings within a narrow band near one value — the head is reciting an average dressed up as a measurement, not reading a feature. That huddle is your first warning sign, and it leads straight to the most important diagnostic in this entire territory.

## Are you turning the peg, or just watching a number move?

Here is the trap, and it is a seductive one. You train a head. You run it backwards. You set a target, you generate, and you watch the head's own reading climb obediently toward your mark. The number moves. You declare victory.

You may have proven nothing at all.

The head's report agreeing with your target tells you only that *the head thinks the latent matches the target*. It does not tell you that the **audible output** changed. These are two different claims, and conflating them is the single most common way to fool yourself in this work.

Picture the **false positive** concretely. You sweep the target from low to high. The head's self-report climbs beautifully, tracking your dial almost perfectly — a clean diagonal line. You decode the audio at each setting, drop the renders into a meter, and the actual measured bass energy is *flat*: a near-horizontal line, barely twitching, while the head's report soared. What happened? Your nudge pushed the latent into a region the head *misreads* as "more bass" — exploiting a quirk of the head's own mapping — without moving the part of the latent the decoder turns into actual low end. The peg spun freely. The string never moved. You were watching a number, not turning a tuning peg.

Now the **true positive**. Same sweep. The head's self-report climbs — and when you decode and measure independently, the *real* bass energy of the rendered audio climbs too, spreading across the range you asked for. You ask for more, you get more, all the way out to the waveform. The two lines rise together. *That* is authority.

So how do you tell them apart? You close the loop *on the world, not on the head*. Take the latent you steered, decode it all the way to audio, and **measure the feature independently** — not with the head, but with the honest acoustic measurement you trained the head against in the first place. If your independent measurement spreads across your target sweep, the head has genuine authority. If the head's self-report soars while the independently-measured output sits flat, you have a mirage.

This is why correlation is such a treacherous proxy. A head can show a near-perfect correlation between its reading and the true feature and *still* have almost no control authority — because correlation measures whether the head can *describe* the feature, while authority measures whether nudging the head can *change* it. Those come apart constantly. Spread in the decoded measurement is authority. Movement in the head's self-report is just the meter agreeing with itself.

Build that test first, before you trust any head. It is slower than watching the number climb. It is also the only thing that distinguishes a control from a comfortable illusion — and it is load-bearing for everything downstream in this book.

A last placement, to carry you forward. A head sits at the *latent*, acts at *inference time*, costs nothing per run, is *surgical* (one dimension, not a whole style), runs *closed-loop* (measure-and-correct), and *stacks* with others. Those six coordinates — where, when, cost, breadth, loop, stackability — will be the language of the next chapter, where we lay every control method on the same map and go looking for the empty squares.

## Openings

- When you nudge toward your bass-energy target, the latent is a single tangled geometry — so what *else* moves along with it? Does your brightness drift when you pull on the bass? Could you ever fully account for the ride-along, and would a second head, watching while the first one pushes, help you catch it?

- The two flavours — pressing the molten material versus pressing the projected-clean estimate — must have a sigma-band where each truly belongs. How would you design an experiment to find that band for one of your own features, and would the answer differ feature to feature?

- If the right gain shifts as the material sets, what exactly is it tracking — the latent's own changing magnitude across the walk, the shrinking room beneath the stabiliser ceiling as the norms tighten, or something about how much the carver's velocity will amplify your push downstream?

- Your honest test measures the decoded audio against an *independent* yardstick. But that yardstick was also what trained the head — so is there a subtler mirage where head and yardstick share a blind spot, and what *third* witness could you bring in to catch it?

- A head reads one number per window, and in doing so it flattens time: it cannot tell a sustained low drone from a punchy transient that averages the same energy. What would it mean to give the peg a *shape* to chase through time — an envelope, a rising arc — rather than a single value? That is not merely a more ambitious head; it asks for a different training target and perhaps a different place for the nudge to land. What would you have to retool?
