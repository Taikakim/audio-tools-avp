# Chapter 7 - A Sense of Time and a Scratch of Memory

An experienced musician knows their position in a piece not by consulting a clock but by feeling how far they are from things. Not "I am at measure thirty-two" but "I am six beats after the last downbeat, twelve after the chord change, and the turnaround is coming fast." This is proprioception — the sense by which a body knows where its limbs are without looking; the drummer who lands the one without counting has it. It is a sense of *relative distance* that keeps an improviser coherent across spans too long to hold in working memory.

SA3 has something quietly analogous. Leaving the large mechanisms behind, we turn to two small parts. A well-placed sound-post is a stick of spruce you could snap between two fingers, yet it governs the entire voice of a violin.

## How the model feels where it is

The model's sense of time is rotary positional embedding, or RoPE. You *could* give each slice of the latent stream an absolute address — slice one, slice two, up to the end. RoPE refuses to: instead of stamping each slice with where it *is*, it encodes how far apart any two slices *are*. Absolute addressing ties a lesson to its location; relative distance frees it.

A model trained on absolute addresses learns phrase structure *at those addresses* — bars one through eight, say — and when the same shape recurs later, must learn it again from scratch. It is the helplessness of a guitarist who practised a scale only from the third fret and freezes when you ask for the eighth. RoPE removes that brittleness by construction.

How it does so connects back to "How Moments Listen" (Chapter 5), the faculty by which every moment can listen to every other. Attention proposes which moments talk; RoPE colours the *weight* by how much their separation should pull them together or hold them apart — so the model knows "these two are close" or "these two are far" without knowing what bar number either falls on. RoPE is the proprioceptive nerve; attention the muscle — sever or reroute the nerve and everything the muscle does changes, though it never moves the limb itself.

The consequence is portability — both feature and cost. Because the model's knowledge rides on *relative* distances, a structural lesson learned anywhere transfers everywhere: the feel of a downbeat four beats after the last is the same at the opening or in the coda. But the ear was schooled only on the *range* of distances it met in training; portability is guaranteed within that range and merely hoped-for beyond it — a distinction the second half of this chapter turns on.

A musician will ask: does this sense govern *harmony* — voice-leading, tension and resolution — or only rhythm and large form? RoPE conditions every conversation alike; it does not know harmony from rhythm. Whatever the model learned about how a suspension wants to resolve, it learned *through* these distance-coloured conversations, the only sense of time it has. So relative distance almost certainly touches harmonic movement too — but as a substrate the model built its harmony *on*, not a knob labelled "resolution." Bear that in mind before you bend distances expecting the rhythm to warp while the harmony stays put.

## A small notepad carried alongside

Beside this sense of time, the model carries a handful of learned memory tokens: extra slots that travel through every attention layer beside the audio material, tied to no particular moment. They do not represent "the sound at second four"; they float free of the timeline.

Before we say what they might be *for*, be honest about what we do not know: their contents are the model's private shorthand, unreadable to us, and we do not yet know what it costs — in coherence, in stability — to write to them by hand.

Think of the tokens as a small notepad the model keeps at the edge of the workbench, jotting global impressions that belong to no single slice. Imagine a long crescendo: that *energy is accumulating across the whole passage* has nowhere natural to live in the moment-by-moment material, where each slice knows its own loudness but not the trend. A free-floating slot is exactly where "the build is two-thirds done, keep climbing" could be carried forward — less a track on the tape than the engineer's margin notes, never crowding the music.

The *existence* of such a channel is what should quicken a builder's pulse: a wire already running the full length of the instrument, carrying a signal that is neither audio nor prompt. The duration conditioner offers the same proof-of-concept in Chapter 8: a single number — intended length in seconds — threaded into the right place, steering the whole generation. The memory tokens are a second, independent witness that such an untethered signal *can* touch everything — that the instrument is already wired for a builder to inject new information.

## Two confirmed seams

Each mechanism is a genuine seam — a joint in the instrument, a place a builder might pry.

The first is the relative-time sense. Those distances are in principle manipulable — and here a distinction enters that Chapter 10 leans on. One sense of "manipulating RoPE" is to change *which positions the model sees* — telling it a fresh slice sits at position N rather than zero, without touching the machinery that turns position into distance. The other is to reach into that machinery and *rescale what distance means* — making far-apart moments read as neighbours, or the reverse. Sliding-window long-form rendering, the subject of Chapter 10, lives almost entirely in the *first* kind: managing positions, not rewriting the distance function. Keep the two apart; they cost different things and break different ways.

Suppose you took the harder, second path and compressed the perceived distances, so a long passage's moments read as close together — the way a film editor speeds a sequence by cutting frames — or expanded them, so neighbours read as far apart. A hypothesis to listen *against*, not a finding: compressing may degrade long-form coherence, the model letting the piece wander or repeat; expanding may do the opposite, fragmenting fine detail and local groove. Audition it; do not assume.

There is a cheaper test than weeks of listening. Self-attention, at each layer, records which moment-pairs it weighted heavily — Chapter 5's conversation, made visible — and RoPE imposes its own distance-weighting on top. Lay one over the other: does attention's concentration track the distances RoPE feeds it, or ignore them? Nudge a distance — if the attention pattern shifts in step, the model is *leaning* on relative time and your lever has purchase; if nothing moves, this seam is, for this material, decorative rather than load-bearing. That turns "is RoPE a real handle?" from faith into measurement.

This seam connects forward to Chapter 10's long-form and looping work, and to the frontier of *per-moment* time, where position becomes load-bearing in a new way.

The second seam is the notepad. What if, at inference time, you could *seed* the memory tokens with a signal of your choosing, or *clamp* them to a steady value? You would be writing in that margin yourself, to see whether the band downstream reads it. Chapter 12 will sharpen this move: seeding-and-clamping is *open-loop*. You write once and do not correct based on what the music does — like choosing a prompt, the opposite of the measure-and-correct feedback Chapter 12 builds its control heads around.

What would you write there — intended density, a marker of "this is the climax," a value some control head computed? And, harder, how would you *read back* whether your scribble had any effect? The first seam's diagnostic applies: change what you wrote, hold everything else fixed, measure whether anything downstream moved. A margin the model never learned to consult will swallow your handwriting without a trace; a number that drifts on its own will fool you into thinking you steered when you merely watched.

These are small parts at real seams, tantalising — but "looks like a seam" and "is a usable handle" are different claims. The memory tokens might be a rich channel or a vestigial one; the relative-time sense might bend gracefully or shatter the moment you push it past the distances it met in training. A quieter worry spans both and reaches back into "Voicing the Whole Body" (Chapter 6): the body-wide re-voicing that chapter describes runs the full melt sigma governs (Chapter 4's one dial), and RoPE's distance-sense is supposed to stay *stable* across that melt — two moments no farther apart just because the material got more set. Do the two levers stay independent, or does pulling on RoPE during re-voicing produce coupling neither of us can predict — a wolf-tone where two well-behaved adjustments meet? We do not know. It is the interaction a builder must audition for, not assume away.

## Openings

- If the model feels time as relative distance rather than absolute address, what does a "distance" actually structure — only rhythm and large form, or harmonic movement and resolution too? *(frontier)* Settle this in your own ear first; every question below depends on it.

- What happens when you present the model with positional relationships it never saw — a very long repeating cycle, a metre from outside its training? *(speculative)* Does its portable ear stretch to reach them, or is "portable" only true within the range of distances it has met?

- *Then,* and only then, the manipulations: if you compressed or expanded the perceived distances, what do you *expect* to hear, and what would that reveal about which musical structures you believe live in relative time? *(speculative)*

- The memory tokens are a notepad in the model's own shorthand. If you seeded or clamped them, what would you write — and, harder, how would you design the read-back test that tells you the model actually consulted it, rather than fooling yourself with a number that moved on its own?

- A free-floating channel that touches every layer and carries something other than audio or words fits the memory tokens — but it is also a *specification* for a control seam in the abstract. Where else in the instrument might such a channel already exist, unnoticed, waiting to be repurposed?

- These are the smallest parts the book examines closely, and the most frontier-flavoured. What makes a small mechanism worth prying open — and how would you decide, before sinking weeks into it, whether a seam is a true handle or only the appearance of one?
