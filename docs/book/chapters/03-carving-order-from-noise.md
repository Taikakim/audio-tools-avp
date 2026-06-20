# Chapter 3 - Carving Order from Noise

Here is what is *not* happening when Stable Audio 3 makes music: a single, decisive pour — raw material slung into a mould, struck off, left to set. That image is wrong in every part that matters, and unlearning it is the first work of this chapter.

Here is what happens instead. Picture a stone carver before a rough block. They do not know what the finished form will be — only the direction of it. They take a pass with the chisel, step back, and read what is now there: this plane has opened, that mass still crowds, the grain runs unexpectedly here. They take another, informed by the first. And another. The form emerges not from one decision but from a long sequence of small readings and small cuts, each responding to what the last left behind. When it is done it feels as though it had been waiting in the stone — but nothing was; it was found by a hand that kept asking: given what is here now, where do I cut next?

That is the generative act in SA3. Hold onto the carver; almost everything in this book is painted on this chapter's canvas.

## Starting from nothing

The carver at least begins with stone — real structure, grain, flaws. SA3 begins with something far emptier: pure noise. A latent stream (the compact inner material of Chapter 2) with no pitch, no pulse, no timbre, no direction — random numbers, the audio equivalent of television static. That emptiness is not a defect but the honest starting condition: the bare cane blank before the reed-voicer's knife has touched it, still capable of becoming any voice.

From that nothing, order has to be *carved*. Not filtered out of an existing signal — there is none yet. Not selected from a menu of stored songs — nothing is stored. Built from static into structure: coherent music walked out of pure randomness.

And here is your first handhold, before any of the famous dials. *Which* noise you start from is itself a choice. Two draws of static, carved under the same prompt and settings, arrive at two different pieces — related in style, distinct in their particulars. The starting noise is a kind of prior, a faint set of leanings the carving amplifies into specifics. Builders call fixing it the *seed*: the zeroth control surface, the grain of the blank block. You will not steer with it precisely — far too tangled — but it is where your hand first rests on the instrument.

The model does not read this blank block alone; it also reads your text prompt (Chapter 8), and together prompt and current state determine which way the guide points.

And this seed-setting is the zero point of a continuum. Chapter 9 explores starting instead from a real recording, re-melted to a chosen degree of chaos, and carving from there — a technique called SDEdit. Pure noise is one end of that scale; a barely-disturbed recording the other; everything between is yours.

## A velocity is a direction, not a speed

At each step the model is shown the material in its half-formed state and asked one question: which way should this move to become more like real music? Its answer is called a *velocity*.

That word can mislead a musician. This velocity is not a tempo, a loudness, or a speed in any audible sense. It is a *direction* — an arrow through the 256-dimensional latent space of Chapter 2, a nudge that points which way lies *more like music* from where you currently are. Imagine a tuner who cannot name the right pitch but can always tell you "sharpen" or "flatten," and by how much, from wherever the string sits; follow that in enough small moves and you arrive in tune, the target note never named. So with the model: you cannot tell it "make it F#," but you can nudge and let it adapt.

Each step moves the material a little along the arrow just drawn; from the new position, the model reads again and draws a fresh one. Step, read, nudge — the whole generation, from maximum chaos toward coherent audible order, is the accumulation of these.

This is what the field means by *flow matching*, or *rectified flow* — forbidding names for a homely idea. There is a path from noise to signal, and the model has been trained as a trustworthy guide *at every point along it*, whether the material is nearly pure static or nearly a finished song. Training makes the guide reliable everywhere; generation is following it.

And the model does not read each moment in isolation; the arrow for any single slice is informed by the whole half-formed piece, which is why large-scale coherence can be carved out of noise at all. How the moments listen to one another is the whole of Chapter 5, "How Moments Listen."

## The walk from molten to set

A quantity tracks how raw the material still is versus how finished. Think of the glassblower's gather: fresh from the furnace, fully molten, every shape is possible; as it cools it firms, and the room for change narrows toward none. The model uses its sense of where it sits between molten and set to judge how to read the material and how boldly to nudge.

That quantity is called *sigma*, and it has a chapter of its own. Why the coarse architecture of a piece is settled early while fine surface detail is left for last, and how that asymmetry becomes a control surface in its own right, is the subject of Chapter 4. For now, one image: a single number walking from one (molten) toward zero (set), telling the carving how much latitude remains.

Sigma does not merely tell the model where it stands; as it cools, it re-voices the model itself. The answer "which way is music" is re-tuned through adaptive layer normalisation (Chapter 6), so the guidance grows more cautious and precise as the material firms — the arrow at the first molten step and the one near the set end coming from the same network in two voicings.

## Why a path is steerable and a pour is not

*Because generation is a walk of many small steps rather than one decisive pour, it is open to steering at all.*

A pour gives you no purchase: material in, casting out, no moment between where a hand could enter. A walk is nothing but in-between moments — between every step and the next, a pause where the material sits in a definite, readable, *modifiable* state, waiting for the next nudge. Dozens of such doorways in a single generation, strung along the path like frets along a neck. This iteration is not an implementation detail; it is the very thing that makes the instrument playable. A model that leapt from noise to finished music in one jump would be mute to your hands. The steps are the handholds.

So where can a hand go in? This chapter names two pressure points — the two everything downstream is painted on.

**The first is the noisy material itself, mid-walk.** At any pause, the half-carved latent sits there, partly molten, in a known state. You can lean on it directly — shift it in a direction the model did *not* propose — and let go. The model takes its next reading from the *moved* material, not the one it expected, and carries on. Like nudging the spinning clay between the potter's own passes: you alter what their next pass meets.

**The second is subtler, and you must own it fully now**, because a whole later technique rests on it. At every step, to decide which way to nudge, the model forms a *running guess at the finished piece* — a projection, recomputed fresh each step, of what the settled latent would be if it followed its own guidance to the end from here. Think of the woodworker who, with the joint half cut, holds the two pieces loosely together and *sees* the finished fit, adjusting the next paring to that imagined whole. Early this guess is vague; late, nearly the real thing — but *always present*, the model's best picture of where all this is heading.

That guess is a second surface you can press on. Rather than shoving the raw material around, you push the *imagined finished piece* toward what you want and let it inform the next nudge. Pressing the clay and adjusting the woodworker's preview are different gestures with different leverage, though both bend the same walk. When each bites hardest, and what each is good for, is the craft Chapter 13 teaches; this chapter only opens both doorways for you.

And some leans are not a single shove but a loop: measure how far the material is from a target, nudge accordingly, read again, measure again — the closed loop of Chapter 13.

## Many small steps, many small openings

Every step is a moment when the material rests in a definite state, ready to be read — and so a moment when *you* could read it too, and lean. Some leans go with the grain, strengthening the direction the model already favours. Some cut across it, asking for something the model would not have reached on its own, and the carving resists, or complies roughly, or surprises you. Some hold part of the material fixed throughout the walk — a technique called *inpainting*, where you freeze regions the model must carve around as if they were unmovable stone (Chapter 10). Learning the difference by ear is much of what it means to become a player of this instrument rather than an operator of it.

Either pressure point is open at any pass in the walk — the whole gift of an iterative process. Those handholds become more powerful still when paired with measurement (Chapter 13): a head that reads the latent and reports back a single musical feature, so you can nudge not blindly but toward a target.

## Openings

- Which features get roughed out in the earliest passes, while the material is nearly static — tempo, energy, the broad arc — and which are left for the final cuts? Does that coarse-to-fine order line up with the molten-to-set walk Chapter 4 takes up? (That *some* coarse-to-fine order exists is known; whether it maps onto the things a musician would name is open.)
- Does the *character* of a step change as the material settles? Is the model's sense of "which way is music" a confident, sweeping arrow early on and a cautious, local one near the end — and how would you tell from the outside? (speculative)
- The running guess of the finished piece sharpens as the walk proceeds. Is there a point where it becomes trustworthy enough to steer against — and is pressing on a vague early guess versus a near-final one the same act or two different tools? (frontier)
- What does it mean to steer *with* the grain versus *against* it? Lean the way the model already favoured and you strengthen what it wanted; cut across and it resists. How would you hear that resistance, and what does it tell you about the dimension you are trying to control? (speculative)
- If every pause is a doorway, are some wider than others — does a nudge applied early reach further into the final result than the same nudge applied late, and which pressure point is more sensitive to *when* you press it? (frontier)
- If you lean on both pressure points at once — nudging the raw material *and* the projected clean vision — what happens when they pull in different directions, and can you use that tension deliberately? (frontier)
