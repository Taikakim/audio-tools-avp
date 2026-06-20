# Chapter 6 - Voicing the Whole Body: adaLN and the Stabilisers

A luthier learns early that voicing is never finished. You graduate the top, shave the bracing, tap and listen — but those cuts do not fix the response. Press a string harder and the body answers differently; warm the room and the wood loosens, chill it and the top stiffens. The instrument has a body, but how that body *responds* shifts continuously with the conditions around it.

The carver at the heart of SA3 works the same way. Playing the room's temperature, re-voicing the body moment to moment, is sigma: the molten-ness number from Chapter 4. That chapter left you a question — whether it is wise that a single furnace heats the whole gather of glass, one molten-ness shared by every moment of the clip. This chapter shows the machinery that furnace drives.

## The Three Global Knobs

The mechanism is adaptive layer normalisation, adaLN. Wired into every block of the carver — and they are stacked deep — sits a set of three knobs: a *scale*, a *shift*, and a *gate*.

Borrow the mixing console. The scale is a channel gain — how loudly a layer's internal voice speaks. The shift is a bias, a DC offset sliding the signal up or down, changing what counts as centred. The gate is the fader at the output — how much of what this block computed passes to the next, from "let it all through" to "hold this block back entirely." None of these act on audio; they act on the internal state of each layer.

Now the crucial part. Those three knobs are not set once; they are recomputed *fresh at every step of the carve*, and their settings flow from a single object worth naming: the **global conditioning vector**. This is an *input*, assembled new each run, carrying the current sigma, the intended duration, and a summary of the prompt. Feed it in, and out come the scale, shift, and gate for every block.

Make it concrete. Probe one block early, at sigma near 0.8: the modulation might hand it a gate near 0.3 — *hold most of this back, too molten for fine work*. Run the same block late, at sigma near 0.2, and it might open that gate toward 0.9 and lift the scale — *speak fully now; there is real structure to refine.* Same block, same weights, two voicings, and all that changed was the molten-ness folded into the global vector. (Treat the numbers as a sketch; the point is the *direction* of the swing, not the digits.)

So as generation walks from molten toward set, the whole deep stack is continuously re-voiced — early attending to bones, late to surface. Think of a pipe organ. Every rank of pipes is fixed metal, cut once — yet raise or drop the wind pressure feeding the chest and every pipe speaks differently at once, re-voiced by one shared condition no single pipe controls. Sigma is that wind: the body does not change, its *responsiveness* does, everywhere at once.

## The Hook Hiding in Plain Sight

Recall the carver's other door from Chapter 5: cross-attention, where each block reaches for the translated words. Cross-attention is information a block *reaches for*; the global vector is a condition it is *immersed in*, applied to every layer whether it asks or not. Players reading their own charts, versus the temperature of the hall that every instrument obeys.

Here is what should make a builder sit up. The global conditioning vector is an *input* — a real, locatable signal, not an emergent property buried out of reach. And it already carries a *measured quantity*: duration (Chapter 8's to own — the vector is a place where such numbers get folded in), alongside sigma and the prompt.

Which raises the obvious question: what *else* could ride in it?

Anything you could express as a vector and fuse into that signal would re-voice not one layer, not one head, but the entire body at once. Mark this, because it is a *third category* of control. A LoRA retunes the instrument's body permanently, at training time (Chapter 11). A control head — the tuning peg Chapter 13 builds in full — presses instead on the clay, the latent, during one carve. The global conditioning vector is neither: **control-at-runtime that re-voices the whole body without any training-time cost**, a lever already wired to reach every block, touching no weights. It adjusts *the hands*, fresh each run.

Whether you fold a new signal in at training time — teaching the network to read it — or at inference time, reaching into the assembled vector, is where the design questions live. The two are not equivalent: a network never taught to read a channel may simply ignore it, or worse, mistake it for noise it has learned to suppress. We leave that as an opening, not a recipe.

## The Quiet Bracing

Beside this expressive voicing sits a second family of mechanisms, far less glamorous: QK-normalisation, RMS-like norms, dynamic-tanh, running statistics. You will never reach for them as a control, but they are the reason any control works at all.

Return to the workshop. A great soundboard is *braced* — graduated struts that let it flex where flexing makes tone and stiffen it where stiffness prevents collapse. A bell that rings too freely develops a wolf-tone: one note that howls and swamps its neighbours. The bracing is not there to make the instrument *expressive*; it keeps expressiveness from tearing it apart.

The stabilisers play this role inside the carver. Attention, left unbraced, is prone to its own wolf-tone: a few values grow enormous and the whole computation howls into one screaming note. QK-normalisation keeps attention's internal levels in range so no single comparison runs away with the room. The RMS-like norms and running statistics keep signal levels from exploding toward infinity or collapsing toward silence block after block — damping that keeps a long, resonant body from accumulating a rattle over its length. Dynamic-tanh and its kin are soft limiters that fold back the extremes before they turn destructive.

None of this is the voice. All of it is the bracing that lets the voice exist.

## Stability Is the Precondition of Expressiveness

**Stability is not the opposite of expressiveness. It is its precondition.**

It is tempting to imagine the stabilisers as killjoys — constraints you would rip out to let the instrument sing louder. But an unbraced soundboard does not sing louder; it caves in. The bracing is what makes loud, free playing *survivable*.

This matters the moment you push on the instrument with your own controls. Drive a control head's nudge hard and you are testing how much the body can take. **There is a ceiling, and it is the single most important fact to carry into your control work.** Push gently and the network absorbs your nudge and stays coherent. Push past the ceiling and the stabilisers can no longer hold the levels in range: your nudge stops *steering* and starts *clipping* — folding the material toward silence at one extreme or incoherent mush at the other. The latent no longer decodes to music.

And the ceiling is not fixed. It almost certainly *moves with sigma*. When the material is molten — high sigma, the network's internal signals already noisier and closer to their limiters — there is less headroom before something saturates, so the ceiling sits lower; as the material sets, the levels calm and headroom may open. (Whether it cleanly rises as the glass cools or dips somewhere mid-walk is exactly the kind of thing to go *measure*.) This is why a single fixed nudge strength rarely behaves the same across the whole carve — and it foreshadows the per-band gain work of Chapter 13.

Recognising the ceiling as a *principle* is this chapter's contribution; how high it sits, how to find it, how it shifts as the material sets are the calibration questions Chapter 13 cashes in when it tunes a head's gain to the latent's scale. The stabilisers *define the room* in which all control happens.

## Global Voicing, Local Detail

One last tension. AdaLN re-voices the whole body *at once*: a single sigma, a single global vector, every block modulated together. But the blocks are not all doing the same work — earlier ones tend toward coarse structure, later toward fine detail. Is one uniform whole-body re-voicing always what you want, or is there something to learn where one block is made *less* sensitive to the global signal?

Call it a selective deafness — like asking one player in a conducted ensemble to ignore the conductor's tempo while everyone else follows. If late-stage detail blocks were made partly deaf to sigma's "we're nearly done, refine the surface" instruction, might they keep doing structural work later than the global schedule intends? We do not know. That is exactly the kind of unvisited square the closing chapters are built to help you find.

There is a deeper, more dangerous assumption hiding here. The adaLN machinery has only ever been taught to read sigma as a single number that *descends monotonically* — molten to set, one value shared by the entire clip. Every voicing it has learned is keyed to that orderly walk; this is baked into the modulation, not a soft preference. So when later chapters reach for the frontier — a *per-moment* sigma, where different slices sit at different temperatures at once, the basis for endless and streaming generation — the break is not cosmetic. You would feed the global vector a situation it was never built to voice: a body asked to be molten and set in the same breath. Whether the existing bracing even holds under that is a genuinely open architectural question, not a tuning detail. Chapter 15 picks it up.

## Openings

- The global conditioning vector already reaches every block at once. What would it mean to fold a *new* signal into it — and how is doing so at training time (teaching the body to read it from the start) fundamentally different from reaching into the assembled vector at inference time, where the network may ignore a channel it was never taught to hear?

- The stabilisers set a ceiling on how hard any nudge can drive before the network loses coherence. Can you find where that ceiling sits for your own control heads — and does it stay put as the material sets, or move with sigma: lower when the glass is molten, higher when nearly cool? Where exactly does steering turn into clipping?

- AdaLN re-voices the whole body uniformly, but different blocks do different work. Is one global re-voicing always what you want — and what might you learn from a selective deafness, deliberately reducing a single block's sensitivity to that global hand?

- Sigma's walk from molten to set is not just *used* by the adaLN machinery but *assumed* by it, monotonic and single-valued. What happens to that assumption when you disturb the path — skewing the schedule, dropping in a re-melted seed, or imagining a per-moment molten-ness? What is the modulation quietly counting on?

- A control head presses on the clay; the global vector would press on the hands. If you had both, would they cooperate — or would re-voicing the hands change the very ceiling that bounds how hard the nudge can press?
