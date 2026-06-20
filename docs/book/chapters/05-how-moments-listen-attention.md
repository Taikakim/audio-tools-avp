# Chapter 5 - How Moments Listen: Attention as Form

Imagine you are the drummer in a band that has played together for years. Without turning your head, you know where the bassist's root note landed two bars ago, you feel the guitarist leaning toward the coming turnaround, and underneath it all you keep faith with the very first downbeat — the one that told everyone what kind of night this would be. You are not consulting a memory. You are *hearing* these things at once, as a single field of relationship — and that is what lets you play form rather than merely play notes.

A cheap machine cannot do this. A simple processor hears the present sample, acts, and forgets; with no faculty for relationship across time, it cannot build a chorus, lock a groove, or recognise that bar thirty-three is a return of bar one. What sets SA3's diffusion transformer apart is exactly this listening, called **attention** — the closest thing the instrument has to an internal ear for its own form.

This chapter is about two ears. One points inward, at the music's own material; one points outward, at your words. They are different faculties, they live at different junctions inside the instrument, and — the part that should make a builder's hands itch — both are places you can reach in and put your hands.

## The Inward Ear: Self-Attention

Recall from Chapter 2 that the carver works not on raw audio but on the latent — a stream of slices, roughly ten and a half per second, each a column of numbers standing in for the resonant signature of that moment. And recall from Chapter 3 that generation is a walk: many passes, each re-reading the whole block and deciding where to cut next.

At every step of that re-reading, every slice is allowed to *consult every other slice* — not just its neighbours but the whole length of the clip, past and future alike, since all the slices exist at once during carving. A slice near the end can reach back and ask the opening slice, "what kind of thing are you?" — and shape itself by the answer. This is **self-attention**: the music's moments listening to one another.

What is exchanged: each moment produces from its own contents a small set of vectors. One acts like a *question* (a learned pattern it is searching for), another like an *answer* (a learned pattern it broadcasts about itself). The mechanism compares every question against every answer; where they point the same way, that is a match, and the questioning moment draws material from the moment it matched. A slice in the second chorus carries a question the first chorus answers loudly, so the second leans into the first, inheriting its shape. A backbeat carries a question the other backbeats answer, so the groove coheres. None of this is stored as a "chorus memory" or "groove rule"; it is re-derived every step, by moments interrogating moments.

That is why self-attention is the **acoustic body of form itself**. A note is not "a chorus." A chorus is a *relationship* — a kinship between stretches of time that recognise each other. Repetition, call-and-response, the arc of a build, the satisfaction of a return: every one is a pattern of which-moments-lean-on-which. Self-attention holds those leanings — not a stored memory but a live, mutual awareness.

One thing this faculty cannot do alone: tell *how far apart* two moments sit in time. Matching questions to answers is timeless — the opening and the coda meet as equals. The sense of *distance* arrives through a separate mechanism, the relative time-sense you will meet in Chapter 7. This chapter is where moments meet; that one is how they know the span between.

## The Outward Ear: Cross-Attention

Now the second ear. Everything so far has been the music listening to itself. But the carver also has to obey you — to attend to the prompt you typed. That is a fundamentally different act, because the thing listened to is not part of the music at all; it is outside instruction.

This is where mystified thinking creeps in. People imagine the prompt's influence as a perfume — diffused everywhere, soaked into the model, impossible to point at. It is not. The words enter the instrument at a **specific, locatable junction**, and nowhere else.

Here is the path. Your phrase is handed to the conditioner — the text encoder, which Chapter 8 examines in full — and comes back not as words but as a sequence of dense vectors, the translated shadows of what you said. At certain layers of the carver, interleaved with the self-attention layers, sits a second kind of attention: **cross-attention**. Here the carver's moments pose their questions not to each other but *across*, to the waiting prompt-vectors. Each asks, in effect, "of everything the words offer, what is relevant to me?" — and draws accordingly.

That word *across* is the whole point. Self-attention is the music consulting itself; cross-attention is the music consulting the words — a *door with a frame*, a named place where outside instruction is admitted into the carving. Call it **the door**: the single spot where language becomes music, and where later chapters will send things through.

So prompting is not a vague spell cast over the whole system. The *effect* of a prompt may be holistic — a mood colours everything — but the *point of entry* is sharp and singular. A luthier who can name the junction can work the junction.

## Both Ears Are Control Surfaces

Attention is not only how the instrument works; it is a surface with grain a builder can shape. Three handholds, roughly in order of how directly they touch what we have just described — each sitting in a different place on the larger map of control that Chapter 14 lays out in full.

**Widening or narrowing the door.** The prompt's authority is not fixed. The door can be opened wider, so the music leans harder on the words, or narrowed, so it consults them more loosely. Be concrete about "wider," because Chapter 9 leans on it: opening the door sharpens contrast rather than amplifying every prompt feature evenly — features the words clearly call for get pushed harder, features they are silent or ambivalent about get pushed *away*. Narrow the door and the music consults the words gently, conflicting instructions blur rather than fight, and the base instrument's own tendencies have more room. You have already met this dial without knowing where it lived: classifier-free guidance, the subject of Chapter 9, amplifies the gap between music-that-heard-the-words and music-that-did-not — and now you know the doorway it amplifies *through*. It is a volume knob on this ear, set at carving time and colouring the whole piece at once — the broad-stroke kind, about as far as you can get from the surgical, one-thing-at-a-time tuning peg to come, a contrast Chapter 14 places properly.

**Teaching the door new manners.** Because cross-attention is a definite location, it is somewhere you can *attach* a trained correction — a small adapter, of the kind whose taxonomy Chapter 11 lays out, bonded right at this junction. It does not rewrite the whole instrument; it biases, at the door, *which* prompt features the music leans toward — a thin shim fitted at the threshold so that certain instructions are heard more sympathetically, a learned predisposition to listen for, say, rhythmic words over textural ones. You would teach it as Chapter 11 describes: with a modest pile of examples that *already* embody the bias you want, and a training signal that nudges the door's listening toward what they contain. What belongs *here* is that the junction is a mounting point, and that an adapter bonded here is a near relative of the control head Chapter 13 builds, only attached at an attention surface rather than read off the bare latent.

**Masking who hears whom.** The inward ear, too, is workable. Self-attention's rule is "every moment may consult every other moment" — but that *may* is a permission, and permissions can be edited. A mask is a decision, slice by slice, about which consultations are allowed. Mask the second verse from hearing the first, and the kinship that would have made it a *return* is severed — the model can no longer recognise the verse as a homecoming. What arrives in its place? A passage that no longer knows it was supposed to come home, that re-states without re-meaning. You sculpt structure by selective deafening. This is a cousin of the inpainting and clamping in Chapter 10, but aimed at the faculty of relationship rather than the material itself — and like an adapter at the door, a place where a control head could in principle attach, deciding mid-carve which kinships to permit (again, Chapter 13).

Hold the honest line through all three. The first two are literal and workable: guidance on the door is dependable, and adapters at the door are established practice whose *musical* targeting is still an open craft. The third — masking the inward ear — is mechanically real but largely untested as a deliberate compositional tool, because form is subtle and the consequences of cutting a connection are not always where you expect them. The surface is real; what nobody has yet mapped is what music it yields. Knowing which is which is part of knowing the instrument.

## What Else Could Walk Through the Door

The cross-attention doorway opens onto the largest country in this book.

We have described it as the place where *words* enter. But notice what it actually requires: not language, but vectors — a sequence of numerical shapes the carver's moments can pose their questions to. The conditioner happens to produce those shapes from text; the door does not know that. It knows only how to let the music consult *something* laid before it.

So the real question is not "how do I word the prompt better?" — that belongs to Chapter 8. This chapter's question is stranger: *what else could you lay before that door?* If a sequence of vectors can pass through, and the carver will dutifully consult it at every layer, then text is merely the first thing to send. A measured musical contour over time. The output of a control head. Some shape you have not named yet. The doorway is established; what walks through it is, increasingly, up to the builder. Asking "what else could walk through the door?" is exactly the move Chapter 16 will ask you to make again and again, at every junction in the instrument — and now you have done it once, here, where instruction becomes sound.

## Openings

- If form lives in moments listening to each other, what happens when you deliberately prevent certain moments from hearing others? Could you compose large-scale structure by *deafening* — severing the kinships that would produce a return — and what would that sound like as it failed?
- The cross-attention door requires vectors, not words. What else could you walk through it — a brightness contour, a rhythmic-density curve, the running report of a control head — and what would it cost in fidelity to send something the door was never trained to expect?
- An adapter at the cross-attention junction can bias which prompt-features the music leans toward. What musical qualities are unreachable by words alone — things you can hear in your head but cannot make the prompt deliver — and is the door the right place to teach them?
- The self-attention weights are a live score of mutual relevance: a map of which moments the model thinks belong together. What would it mean to *read that map back out* as a signal — to see, mid-carve, what structure the instrument already believes it is building?
- We have two ears pointed at two things: the music's own material and your words. Is there a third thing worth pointing an ear at — and would you build it as a new doorway, or teach one of the existing ears to listen differently?
