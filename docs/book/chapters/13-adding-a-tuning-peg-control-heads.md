# Chapter 13 - Adding a Tuning Peg: Control Heads and Training-Free Guidance

Imagine bolting a contact microphone onto your instrument's body — not to amplify it, but to measure one thing: a single resonance, the tension in one string, the strength of one overtone. Now wire that microphone into a tiny servo that turns a bridge pin until the measurement matches a target you dialled in before the performance. You would have built a new tuning peg — one that governs a dimension the instrument shipped without a knob for. That is, very nearly, a control head inside SA3.

This is your home territory, so let us be precise about how the device earns its authority. The difference between a peg that turns and a peg that only *appears* to turn is the sharpest thing in this book.

## A learned sense, trained once

A control head is a small, separate network. It does not generate anything; its entire job is to *read* — to look at the latent stream the carver is shaping and report back a single musical quantity. How much energy lives in the low end. How bright this passage is. How densely events are packed across a window of time. One number, or a small handful, out of that whole column of unlabelled latent dimensions from Chapter 2.

That reading is not free. The head must be *trained* to hear its one thing, and this is where your hands first go in. You choose the feature, take a body of latents, measure the true value of that feature in the corresponding audio, and teach the head to predict it from the latent alone. When it reads accurately, you have a sense: an ear that projects the messy geometry of the latent onto the single axis you care about — a direction nobody drew, turned into a number you can read.

But notice what the head throws away. It reads *one scalar per window* — one bass-energy value for a whole span of slices. Inside that window, time is flattened: a sustained low drone and a punchy transient that average to the same energy read identically, because you never asked the head to tell them apart. Hold onto this; it shapes the ceiling of everything that follows, and the last of the Openings comes back to it.

## Running the sense backwards

Training the head is only the first act. What makes it a control rather than a meter is running it *backwards*.

At inference time you set a target — say, a specific bass energy. Then, during the carving walk, at each step you ask the head to read the current latent. The difference — *target minus report* — becomes a push: a small nudge applied directly to the latent, in the direction that would make the head's next reading land closer to your mark.

This is the whole engine: a feedback loop, alive during every step. Chapter 12 drew the contrast — a LoRA retunes the instrument's body *once*, for every future performance; a head measures-and-corrects *during* each performance and is gone the moment you stop. The generative model is never touched — you have bolted a tuning peg onto the outside of a finished instrument and routed your target through it.

So the nudge lands on the *latent during sampling*, and the live control input is the *target* — the dial under your hand. Everything else about the head is fixed once trained.

One mechanical detail matters more than it looks. The model reads the *nudged* latent fresh on the very next step — so the moments-listen-to-moments machinery of Chapter 5 now attends to a latent you have already pushed. Your nudge can therefore compound (the model picks up your direction and runs with it) or be damped (its velocity pulls back toward where it was heading), depending a great deal on *which* of the two flavours of nudge you choose.

## Two flavours of nudge

Recall the two pressure points from Chapter 3: the noisy material as it sits mid-carve, and the model's running guess at the finished piece. A head can press on either, and choosing is a tuning act.

The first flavour presses on the **molten material directly**. Mid-carve, the latent is still partly noise — warm, moving. You read it, compute the gap, and shove the noisy latent toward your target before the next step. It is immediate, but you are working against a vibrating thing: some of your push gets smeared by the motion underway, and because the model re-reads the pushed latent, a too-eager nudge can be amplified into incoherence.

The second flavour presses on the **projected clean signal** — the model's running estimate of where all this is heading, the finished latent it believes it is carving toward. You let the model form that estimate, then nudge *the estimate*, and the carver finds its own path to the moved goal. This tends to be gentler and more coherent, because the model's momentum carries the adjustment forward rather than fighting it.

Which one has more leverage at a given molten-ness is still open. The reasonable intuition — pressing the clean estimate earns more authority early, when the destination is still negotiable; pressing the molten material does little once everything is nearly set — is a hypothesis to test on your own features, not a law, and the right answer may differ for bass energy and for brightness.

## Matching the gain to the latent's scale

Here is the part that is solid, and missed more often than any other. The nudge has a strength — a gain, a number that says how hard to push for a given gap — and it has to be matched to the scale of the latent it pushes on. This is not optional polish; get it wrong and the peg either does nothing or snaps the string.

The latent of the large SA3 model carries roughly **ten times** the numerical magnitude of the latents in earlier Stable Audio versions. So a gain of 0.01 perfectly voiced on SA1 becomes inaudible on SA3-large: your nudge whispers, and nothing moves. Overcorrect to 0.1 and you blow the material out of coherence. Two things break. First, the push breaches the **stabiliser ceiling**, driving the latent harder than the norms can absorb until downstream they clamp or overflow. Second, even short of that, an oversized nudge destroys **local coherence** — neighbouring slices that should agree (a bass note and its own continuation a moment later) get shoved in inconsistent directions, and the carver can no longer settle them into music. Smear, noise, a ruined take.

So gain is a tuning act, not a setting to look up, and it is *instrument-specific*: when you carry a head — or any guidance technique — from one model to another, the first thing to re-voice is the gain, scaled to the new latent's magnitude. And recall the ceiling from Chapter 6, *Voicing the Whole Body*: the stabilisers that keep attention and levels from blowing up also bound how hard you can drive any nudge. Gain lives in the room between *too quiet to hear* and *hard against that ceiling* — and finding that room is the voicing.

Two threads about where that room sits. First, from Chapter 6: the stabilisers' authority is not constant across the walk — the damping may bite harder at some molten-nesses than others, so the headroom beneath the ceiling tightens and loosens as the material sets. Second, from Chapter 4, *Sigma: The One Dial That Says How Molten*: that single shared molten-ness governs the whole clip at once, and the schedule decides how it descends. Skew the schedule and you change which molten-nesses get the most steps, so the right gain plausibly rides that curve rather than sitting fixed — a nudge while the material is warm is a different-sized gesture than the same nudge once it has nearly set. Whether to confine the nudge to a *band* of molten-ness at all is itself a tuning choice. And recall the constraint Chapter 4 leaves you with: you cannot, today, hold one moment molten while another sets; the band applies to the whole clip.

## What is even learnable as a head

Not every feature will submit to this treatment. Before you build a head, ask whether the feature is *tractable* — whether it can become a sense at all. Three questions sort the learnable from the hopeless.

First: does the feature leave a **clear, consistent signature in the latent**? Bass energy does — low-frequency weight survives the compression legibly. Some subtle perceptual quality may be smeared so thoroughly that there is no stable footprint to read. Second: is that signature **geometrically stable across prompts and seeds**? A head that reads brightness beautifully on techno but garbage on solo piano has learned the feature *as it appears in one corner* of the space, not the feature itself. Third: do the head's reported values **spread over a useful range**, or crowd into a corner?

That word *useful* deserves a number. On a normalised [0,1] scale, a head whose outputs across the training set live between 0.3 and 0.7 has only exercised the middle — it has never seen, and cannot confidently steer toward, the extremes you will most want (the truly bass-heavy, the truly thin). As a rough threshold: if the ratio of the head's smallest to largest reading sits above three-quarters — most readings huddled in a narrow band — the head is reciting an average dressed up as a measurement, not reading a feature. That huddle is your first warning sign, and it leads straight to the most important diagnostic in this territory.

## Are you turning the peg, or just watching a number move?

Here is the trap, and it is seductive. You train a head, run it backwards, set a target, generate, and watch the head's own reading climb obediently toward your mark. The number moves; you declare victory — and you may have proven nothing. The report agreeing with your target tells you only that *the head thinks the latent matches the target*, not that the **audible output** changed. Conflating those two claims is the single most common way to fool yourself in this work.

Picture the **false positive**. You sweep the target from low to high; the head's self-report climbs in a clean diagonal. But decode the audio at each setting, meter the renders, and the measured bass energy is *flat* — barely twitching while the head's report soared. Your nudge pushed the latent into a region the head *misreads* as "more bass," exploiting a quirk of its own mapping, without moving the part the decoder turns into actual low end. The peg spun freely; the string never moved.

Now the **true positive**. Same sweep — but when you decode and measure independently, the *real* bass energy climbs too, spreading across the range you asked for. The two lines rise together. *That* is authority.

The difference is the test. You close the loop *on the world, not on the head*: decode the steered latent to audio and **measure the feature independently** — not with the head, but with the honest acoustic measurement you trained it against. Spread across your target sweep means genuine authority; a soaring self-report over a flat measured output is a mirage.

This is why correlation is a treacherous proxy. A head can correlate near-perfectly with the true feature and *still* have almost no control authority — because correlation measures whether the head can *describe* the feature, while authority measures whether nudging it can *change* it. Those come apart constantly.

Build that test first, before you trust any head. Slower than watching the number climb, it is the only thing that separates a control from a comfortable illusion — load-bearing for everything downstream in this book.

One distinction to carry forward: a head runs *closed-loop*, measuring and correcting live during the performance, where a LoRA simply sets the body and walks away. The next chapter takes that quality and several others and lays every control method out on one shared map, where you can finally see which squares are filled and which sit empty.

## Openings

- When you nudge toward your bass-energy target, the latent is a single tangled geometry — so what *else* moves along with it? Does your brightness drift when you pull on the bass, and would a second head, watching while the first one pushes, help you catch the ride-along? *(speculative)*

- The two flavours — pressing the molten material versus pressing the projected-clean estimate — may each have a band of molten-ness where they belong, but which one wins where is unsettled. How would you design an experiment to find that band for one of your own features, and would the answer differ feature to feature? *(frontier)*

- If the right gain shifts as the material sets, what is it tracking — the latent's own changing magnitude across the walk, the shrinking room beneath the stabiliser ceiling, or how much the carver's velocity will amplify your push downstream? *(speculative)*

- Your honest test measures decoded audio against an *independent* yardstick — but that yardstick also trained the head. Is there a subtler mirage where head and yardstick share a blind spot, and what *third* witness could catch it? *(speculative)*

- A head reads one number per window, flattening time: it cannot tell a sustained low drone from a punchy transient that averages the same energy. What would it mean to give the peg a *shape* to chase through time — an envelope, a rising arc — rather than a single value? That asks for a different training target and perhaps a different place for the nudge to land — the doorway into Chapter 15, where a head stops being a dial you preset and becomes a voice that follows the music as it plays. What would you have to retool to take that first step? *(frontier)*
