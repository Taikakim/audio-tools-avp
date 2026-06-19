# Chapter 12 - Two Kinds of Teaching: Re-Voicing the Body vs Bolting On a Sense

There is a moment in lutherie when you set down your tools and accept that the instrument is, now, what it is. The wood has been voiced, the bracing placed, the neck set, the finish cured. From here forward, every performance draws on that body — the same body, unchanged, for every player who picks it up, every piece they play, every night. You did your shaping months ago. The instrument no longer negotiates; it simply responds, with the tendencies you built into it.

Something almost exactly analogous happens when you retrain Stable Audio 3's weights. Whether you run a full finetune or bond on a LoRA laminate — the body-changing operations covered in Chapter 11 — you are working on the substance of the instrument itself: the billions of numbers that encode what it knows about sound. You are adjusting what it will *prefer* to do, which sounds it reaches toward when left to its own devices. Because those weights are the body, the change reaches everywhere the body acts — into how attention gathers a moment's neighbours (Chapter 5), even into the per-layer voicing knobs the timestep sets through adaLN (Chapter 6), which a LoRA can quietly reshape so the instrument re-voices itself differently as the material sets. That teaching happens once, at training time, and then it persists. Every subsequent generation inherits it, without asking and without being asked. The instrument was changed, and now it is that way.

This chapter is about a second kind of teaching that is not like the first at all — not a bigger version of it, not a smaller version, but a different act entirely. And the whole project of this chapter is to make that difference vivid enough that you feel it in your hands, because it is the hinge on which the rest of the book turns.

## The first teaching: changing what the instrument is

Hold the body-change idea still and look at its character.

When you finetune or apply a LoRA, you change the instrument *before* the first note of any future performance sounds. After that, you let go: you type a prompt, press the button, and the instrument plays out its newly-set tendencies on its own. There is nothing left to intervene in — you already did it, at training time. This is the oven you set to a temperature and trust: you commit to a setting and live with whatever it produces; you do not stand over it with a thermometer correcting it. A LoRA is exactly this. It biases the instrument's response toward a style, a sound-world, a behaviour — and then steps back. Every run thereafter carries that bias, baked in, unwatched.

There is enormous power in this, and also a particular blindness. The power is that the change is *deep*: it reaches into how the instrument moves, not just what you ask of it — a coherence no inference trick can match, because the preference is now part of the body's reflexes. The blindness is that you cannot see, mid-performance, whether any single generation actually landed where you wanted, nor correct a run that goes wrong while it is going wrong. You set the tendency; you do not correct the outcome.

And a tendency is not a guarantee. Suppose you teach the body to lean bass-heavy. Some runs will come out gloriously deep, and some will drift thin — not because the lean failed, but because a LoRA only adds one bias among the thousands already in the body, and every run still starts from its own particular seed of noise (Chapter 3) and is pulled by everything else the model knows. A preference is a thumb on the scale, not a clamp. The thin runs happen, and nothing is watching to catch them. The body does what the body now does, and then you take what you get.

## The second teaching: adding a sense and a loop

Now set the body down — untouched — and reach for something completely different.

You are going to bolt a small device onto the *outside* of the instrument. Not into its body; onto it. This device is a control head: a separate little network, trained apart from the generative model, whose entire job is to *listen* to the latent material as it forms and report one number back to you — how much energy sits in the low frequencies right now, how bright this moment is, how dense the events are across this stretch of time. It is a learned ear, tuned to hear one thing with precision — and it is not part of the instrument. The generative model does not know it is there. Exactly how that ear is built and trained is Chapter 13's business; here we care only about what it *is* relative to the body.

By itself, that device is just a meter — a needle that twitches as the material forms. The second move is what makes it a *teaching*. During generation — during the actual carving walk from noise to music of Chapter 3 — the head reads the material, compares what it reports against a target you set when you sat down to play, and nudges the material to narrow the gap between the two. Then the model reads the nudged material and carries on. Read, compare, nudge. Read, compare, nudge. The head sits *outside* the walk, not inside it: the model takes its step, and then, between steps, the head measures and corrects what the step produced. A loop closed around the outside of an instrument whose insides were never opened. (The shape of that nudge — and how hard it should push — is the craft Chapter 13 unpacks; what matters for the contrast here is only that a loop has been closed at all.)

Here the closest analogy is the pipe organ's swell box: the head is a *real-time set of shutters driven by its own ear*. The ear hears the low ranks sagging beneath the line you drew, and the shutters open to push them back toward it — continuously, while the sound is still being made, not after. That is the character of a control head: a learned listener bonded to a corrective move. This is closed-loop control, and the difference from the first teaching is total. You are no longer setting a tendency and trusting it; you are measuring the actual material, in this actual run, and correcting it toward the target *as it forms*.

And a correction does not accomplish the same thing all the way down the walk, because the material it pushes on is not the same stuff throughout — it sets as the run proceeds (high sigma to low, Chapter 4). Early, a nudge bends the whole trajectory of where the run is heading; late, the same nudge adjusts a near-finished surface rather than redirecting the journey. The loop runs across the entire walk; *what* each correction buys you shifts as the material hardens under it. How tightly to close that loop, and where along the walk to act, are real choices — Chapter 13 is where you learn to make them. The point that belongs *here* is only that such choices exist at all, and that they exist for the head and not for the body: once a LoRA is baked, there is no walk left to intervene in.

## Why this is a difference of kind, not degree

It would be easy — and wrong — to file these two as "a big change" and "a small change." They differ on more fundamental axes than size. Lay them side by side.

*Where they intervene.* The body-change reaches into the weights — the instrument itself, a site you can only touch at training time, behind a door that locks again before the performance begins. The head reaches into the latent material during sampling, a site open the entire time the music is being carved, on every step, with the weights left exactly as they were.

*When they act.* The body-change happens once, in the past, and never again until you retrain. The head acts continuously, freshly, every performance — and could behave differently on the very next run if you turn its target dial.

*What kind of authority they carry.* The body-change carries the authority of a *tendency* — a lean applied uniformly to everything that follows, never checked against any particular outcome. The head carries the authority of a *correction* — it measures and answers, responding to the specific material in front of it. A tendency is open-loop; a correction is closed-loop. Not two strengths of the same thing — two different relationships between the control and the result.

It is worth setting the head beside one neighbour it is easily confused with: the mask of Chapter 10. Both leave the weights untouched and act on the latent at inference time, yet they answer different problems. A mask is a *constraint* — it pins part of the material and forbids the model to change it. A head is a *nudge* — it pushes free material *toward* a target without freezing anything. One says *this shall not move*; the other says *this should lean that way*. Different verbs entirely.

So the trade-off names itself. The LoRA buys *deep stylistic coherence* — a voice woven into the body's reflexes — at the price that it cannot correct a single failing run while it fails. The head buys *surgical, watched targeting of one dimension*, live-correctable, at the price that the instrument itself stays neutral and prefers nothing on its own. One makes the instrument *want* the thing; the other makes the process *check for* the thing. Wanting and checking are not the same faculty — that is the whole chapter in a sentence.

## Choosing, stacking, and what each costs

Two practical things follow when you sit down to actually build a control.

**The costs sit on opposite ends.** A LoRA is *expensive to make and free to run*: you pay in data and training compute up front, but once it is baked in, it adds nothing to the cost of any future generation — the body is just the body. A head is the mirror image: *cheap to make and a standing cost to run*. Training one is light, but it is a live network that fires on every diffusion step it is wired into, so it adds a small overhead to each run, as long as it stays attached. Which side of that ledger you want depends on scale: a voice you will summon ten thousand times wants the cost paid once, in the body; a target that varies run to run, or matters only sometimes, wants the cheap, detachable sense.

**They stack — but differently, and this is the deeper lesson.** Multiple LoRAs blend into *one* body: stack a few and their corrections sum into a single altered instrument you can no longer cleanly separate — you cannot reach back in and silence laminate number two mid-performance, because there is no longer a "number two," only the blended whole. Heads do not blend that way. Each stays its own bolted-on device; stack three and you have three independent meters-and-correctors, each adjustable, silenceable, retargetable mid-session, without touching the others. And you can run *both kinds at once*: a LoRA leaning the body's natural voice toward a sound-world, *and* a head enforcing a specific feature live on top of it. Whether they cooperate — the body predisposed the right way, the head merely tightening it — or reinforce each other redundantly, or quietly fight, is exactly the sort of thing this book wants you to go and *test* rather than be told.

A rough way to choose, then — not a rule, a starting lean:

- Reach for a **LoRA** when you want a *consistent stylistic voice* or a deep preference woven through everything the instrument makes, and you are content to commit to it before the run begins.
- Reach for a **head** when you want to *target one feature precisely across varying material*, to change that target from run to run, or to *catch and correct a failure live* while it is still soft enough to move.

Underneath both choices is the one concrete fact that anchors everything: in the second teaching, *not a single weight moves*. The head's authority is entirely procedural — it lives in the process you build around the instrument, not in the instrument. Detach it between one run and the next and the model is precisely what it was before you ever attached it: no trace, no residue. Hold the final image: the luthier shapes the soundboard once, for every performance to come; the tuner stands at the peg through *this* performance, turning, listening, turning again, while the music plays. Both are teaching. They are not the same act.

Once you truly see that a LoRA and a head sit at different *places*, act in different *times*, and carry different *characters* of authority, you can start to ask the question that unlocks everything later: what about the methods that are *neither*? The feedback loop itself, you will come to see, is not just a property of the head — it is a *new place to intervene* in its own right, and the choice between open and closed loop becomes a design decision you get to make freshly for any control you ever dream up. For now, it is enough that you can feel the two poles, and the open country between them.

## Openings

- If both the body-change and the feedback loop can press on the *same* musical dimension — say, bass weight — how would you decide which to reach for? Are there dimensions that genuinely demand a tendency baked into the body, and others that only a live correction can hold?

- The closed loop corrects toward a target. But who sets that target, at what resolution, and must it be a single fixed number — or could the thing you correct toward be a shape that changes through time, a contour rather than a point? *(speculative)*

- What happens if you do both at once: a LoRA that has leaned the body's natural tendencies one way, and a head correcting live toward a target the other way? Do they cooperate, fight, or open some third character of control that is neither purely set-and-forget nor purely measure-and-correct? *(frontier)*

- The body-change is unwatched by definition — nothing checks any single run. Is that always a weakness, or are there musical situations where you *want* the instrument to commit to a tendency and not second-guess itself mid-performance?

- The head reads the latent and nudges it; but the model is reading that same latent on its very next step. When your correction and the model's own momentum meet on the same material, do they add, cancel, or interact in ways you would need to watch for — and is that interaction itself a place a future control could live? *(frontier)*

- If a head's nudge does something different early in the walk than late — bending the whole trajectory while the material is still soft, adjusting a near-set surface once it has hardened — could you place a control deliberately in just one band of molten-ness, and what new kind of steering would that buy you? *(speculative)*
