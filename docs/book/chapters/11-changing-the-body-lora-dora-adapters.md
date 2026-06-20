# Chapter 11 - Changing the Body: What a LoRA Actually Is (and DoRA, and the Family)

Every instrument can be restrung, refretted, re-bodied. The question is never whether you *can* change the wood — it is what the change costs, how long it lasts, and how much of the old character walks out with the sawdust. A luthier who replaces a top plate gains a new voice and loses the one the player had grown into; one who slips a thin brace under the existing top changes the response without unmaking the instrument. Both are real interventions, but not the same kind of act — and confusing them is how you ruin a good guitar.

This chapter reaches the most consequential intervention site of all: not the words going in, not the latent mid-carve, but **the weights themselves** — the billions of learned numbers that encode everything the model knows about sound. Every dial so far (prompt, guidance, re-melting, masks) left the body untouched and worked at performance time. Here we change the substance, *before the performance begins* — before the descent from molten to set that Chapter 4 calls the sigma walk has even started. A mask freezes material mid-walk, a control head reaches in while it is still cooling, but a LoRA does neither: whatever it teaches, it teaches the carving tool, not the cut.

Two philosophies do this. One rebuilds the body; the other bonds a removable laminate to it. They differ in kind, not degree.

## Rebuilding the Body: Full Finetuning

The most direct way to change what the instrument does is to keep teaching it: gather a corpus in the sound world you want and let the model keep learning as it first learned, except that now every weight is free to drift. Feed it enough Bavarian brass band recordings and it reaches, by default, for those timbres and cadences. This is **full finetuning**, and it genuinely moves the body.

It is also the most expensive and least reversible thing in this book. You are re-graduating the entire soundboard at once, and two costs follow. The first is sheer effort: finetuning a model this size means tens to hundreds of hours of curated audio and serious accelerator time, because no corner is held still.

The second is subtler and more dangerous: **the instrument forgets.** Push the weights hard toward your brass band and its grip on what it used to do well erodes — the capabilities *least represented in your new corpus* going first. Finetune on hours of dark ambient drones, then ask for a crisp hi-hat, and the model reaches, sheepishly, for a soft wash. The technical name is catastrophic forgetting, but the luthier already knows it: you cannot aggressively re-voice one register without risking the others, because it is all one piece of wood. And the change is more or less permanent — you can finetune again, but you cannot peel it back off.

Finetuning has its place — when you truly want a different instrument and have the data and patience for it. But for most of what a musician-builder wants to try, it is far more surgery than the goal requires.

## The Laminate: What a LoRA Actually Is

LoRA — Low-Rank Adaptation — begins from a refusal: it does not touch the existing weights at all. Freeze them, and the instrument keeps its body as built. Then, *alongside* certain chosen internal matrices, slip in a small, separate correction that rides on top of what the frozen weight already does.

Picture a thin laminate bonded to the soundboard. The plate is untouched beneath it; the laminate biases its response in a direction you have chosen — warmer, brighter, quicker to speak — and because it is a separate piece, you can shape it, dial how firmly it presses, or lift it off entirely. The rest of the chapter leans on that image.

The defining trick is in the word "low-rank." The correction is forced through a **bottleneck** — a narrow channel and out again — so it has only a handful of independent degrees of freedom rather than the sprawling freedom of the matrix beside it. A correction with few knobs can be learned from little data — minutes to a couple of hours of audio, where the full finetune wanted tens or hundreds — stored in a tiny file, and it tends to learn a *direction of adjustment* rather than a thousand unrelated details. You bond a laminate with a definite grain, not re-carve the plate freehand.

When does the laminate stop being enough? The shallow answer is data: below a few hours, LoRA is the only option; above tens, either is viable, and the choice is permanence versus reversibility. The deeper answer is *depth of effect*: a thin shim can lean the existing world, not supply a new one. When the change must touch every corner — a different instrument, not a biased one — you rebuild the body.

And where does it bond? Most often at the same internal junctions we met in Chapter 5 — the attention layers, including the cross-attention door where the prompt's authority enters. *Where* you attach the correction shapes *what* it can teach.

## The Training-Time Change That Behaves Like a Runtime Control

The learning happens once, at training time, but because the correction is a *separate, additive* piece never baked into the frozen body, it behaves at performance time like a set of live controls — all four turning during generation, no further training:

- **Dial its strength.** A style learned at full intensity can be applied at a quarter for a subtler inflection. The training was fixed; the dose is not.
- **Stack laminates.** Bond a rhythmic-feel correction, a timbral one, and a production aesthetic all at once. They combine *at the moment you press play* — you assemble the voice live, from a shelf, not in some prior run that fused them.
- **Blend.** Mix two corrections by weight, so the space *between* two laminates becomes a continuous dial you sweep in performance. Finetuning, having baked everything into one body, cannot offer this.
- **Peel it off.** Detach the laminate and the instrument is, to the number, exactly what it was before.

So a LoRA is a *training-time* change to the weights that hands you, in return, a fistful of *inference-time* dials. This is your first taste of the map this book is building toward — the one that gives every method a formal address (Chapter 14 draws it in full). For now, feel only the distinction it captures: finetuning and LoRA touch the *same wall* at the *same moment* (the weights, before the carving), yet sit at opposite ends of permanence — the finetune welded on, the laminate peeling off.

## When the Laminates Fight

Stacking is the most seductive promise, and two biases on the same patch of wood do not politely average. Two LoRAs that both lean on the low end add their pressures where they overlap, and the stack can boom or smear where neither did alone; two that colour the *same* attention junction in different directions compete for the same degrees of freedom, and the louder wins. Corrections that stack most gracefully tend to touch *different* things — a rhythmic-feel laminate and a timbral one are likelier to coexist than two timbral ones — but even that is a tendency, not a law.

This is the first wall you hit when composing interventions, and not the last: the same caution returns in Chapter 13, the moment you stack a LoRA *and* a control head *and* guidance in one run. Three pressures on one plate, two set-and-forget and one correcting in real time — do they cooperate or interfere? We do not know in general; it is a genuine frontier. Stack deliberately, listen hard, add one thing at a time.

## How You Know It Worked

How do you *know* a laminate did what you wanted, rather than something you talked yourself into hearing? A LoRA almost always changes *something*, and the ear is generous to its own effort — the danger is mistaking *any* change for the *intended* one. Three habits guard against it. Generate the same prompt *and the same seed* with the laminate off and on; fixing both isolates its contribution from the ordinary variation between takes. Sweep the strength knob up from zero and check that the quality you care about moves *monotonically and in the direction you trained*: if "warmer" arrives at a quarter, vanishes at half, and returns inverted at full, you taught noise that sounds warm at one setting, not warmth. And test on prompts *outside* the training material; a laminate that only steers the phrases it was trained on has memorised, not generalised. The discipline rhymes with the one Chapter 13 builds for control heads, but the laminate needs its own version, because it has no readout — only your ears.

## DoRA: Separating Strike from Partials

DoRA refines the laminate idea by noticing something plain LoRA blurs together. Any correction to a weight has two independent qualities: *how much* the response shifts and *which way* it shifts. Plain LoRA learns both tangled into one bundle; DoRA pulls them apart, a separate handle for magnitude and one for direction.

The bell-founder already lives inside this distinction. A cast bell has a **strike weight** — how forcefully it speaks, set by its mass — and a **partial structure** — the tuned chord of overtones that gives it its colour, set by the curve of its inner profile. Coupled in feel, separable in fact: the founder retunes the partials by shaving metal from the inside of the wall, re-colouring the voice without making the bell heavier or louder. Adjusting the two *independently* is finer command than only pouring a bigger bell and accepting whatever shift in timbre comes cast into it.

That independence is what DoRA buys, and it matters because real finetuning, watched closely, mostly changes *direction* while keeping *magnitude* on a shorter leash — so DoRA mirrors the natural motion and learns more steadily from the same small dataset. Which musical jobs benefit most? Those about *re-colouring without re-forcing*: a new timbral tint, a new vowel at the same loudness, a shift in *character* that must not become a shift in *intensity*. These live almost entirely in direction, and a correction that cannot hold magnitude still while it turns will overshoot them — getting louder when you only wanted it darker.

## The Adapter Family: One Idea Wearing Many Shapes

LoRA and DoRA are the two you will meet most, but they sit inside a wider family — LoHa, LoKr, IA3, prefix-tuning, bias-tuning, more arriving as you read. Do not memorise it as a catalogue of gadgets: they are variations on a single idea, distinguished by two questions: *where do you hook in,* and *what shape does the correction take?*

The first is the more musical, because *where* is usually treated as a technical default when it is really a choice. Recall the two attentions of Chapter 5 — the cross-attention door where the prompt is heard, and the self-attention where moments relate across time. Bond the laminate at the prompt door and you bias *how the words are interpreted*; bond it among the moments and you bias *structure* — how a phrase answers an earlier one, how repetition and form cohere. (Others touch only the tiny bias terms, or prepend a few learned slots without altering any weight at all.) So the attachment point decides whether you teach a new *style*, a new *timbre*, or — most intriguingly — a new *responsiveness* to a signal it did not used to weigh.

The second is geometry: a narrow bottleneck (LoRA), a magnitude-and-direction split (DoRA), two patterns multiplied, a per-channel scaling (IA3) — each a different shape for the small change you are allowed.

Hold onto those two questions. They are not only how you place an adapter — they are the two you end up asking of *every* control method in this book.

One wrinkle for the builder who wants precision. You met **adaLN** in Chapter 6 — the stabiliser that re-voices every block according to how molten the material currently is. Where does a laminate sit relative to it? Not in conflict: adaLN changes step to step as the carving proceeds, while the LoRA is a *fixed, local* bias welded into one matrix. The laminate changes *what the junction computes*; adaLN changes *how loudly that computation may speak*. They layer — which is why a LoRA's effect can feel stronger or weaker at different stages though the laminate itself never changed.

## The Hinge

Everything in this chapter teaches the body a new **direction of adjustment**. A finetune moves the whole body that way permanently; a LoRA leans it reversibly; a DoRA leans it more faithfully; the wider family leans it from different attachment points. All answer the same question: *which way should the instrument tend to move, by default, before the first note sounds?*

There is one thing none of them can do, and naming it is the most important sentence in the chapter. A weight-change can teach a new *bias* — a lean, a preference, a default direction. It cannot teach a new *sense*: it cannot make the body **responsive to a signal it has never been given**, cannot bolt on a fresh ear that listens, during the performance, for something you care about and corrects toward it in real time.

That distinction — a bias baked in versus a feedback loop added on — is the hinge the next chapter turns on. Cross it and things become possible no laminate can reach: an instrument that *tracks* a target you set while it plays, holding a brightness, following an envelope, correcting toward a loudness. We have re-voiced the body; we have not yet given it a sense it lacked. Hold the difference — it is about to do a great deal of work.

## Openings

- If two laminates trained on different material can be blended mid-performance, can the space *between* them hold a voice neither could reach alone — and is there a difference between a sound the model *learned* and one that lives only in the interpolation between two corrections it never saw together?
- *Where* an adapter attaches changes what it can teach. If you wanted to teach not a style but a *responsiveness* — a predisposition to react to a signal the base model currently ignores — where would you bond the laminate, and how would you even know you had succeeded?
- DoRA separates magnitude from direction. Are there musical qualities that live almost entirely in *direction* — a re-colouring with no change in force — such that a correction blind to the split would chase them and never catch them? And is there a converse, qualities of *pure magnitude*?
- When you stack a set-and-forget laminate with a measure-and-correct control head in one run — a frozen lean and a live feedback loop pressing on the same latent — does the baked-in bias help the head reach its target faster, or fight its corrections at every step?
- Every method here teaches a direction the body will *tend* toward by default. Which musical intentions are fundamentally about a *tendency* — a habit, a default reach — and which are really about *listening and correcting in the moment*, and therefore cannot be a laminate at all, no matter how cleverly shaped?
