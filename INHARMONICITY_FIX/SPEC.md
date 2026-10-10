# Inharmonicity auxiliary head: fix spec for the Antigravity agent

Author: KUANG, October 10, 2026. Base: repo `audio-tools-avp` (`https://github.com/Taikakim/audio-tools-avp.git`),
commit `51c603b8c84ded947537b93b53af1ced16f44ae8` (short form `51c603b`), the tip of branch
`feature/surge-xt-neural-inversion`; checked on the remote on October 10, 2026. This folder: `INHARMONICITY_FIX/`
(SPEC.md, `provided/`, and two offline copies of the base: `inharm-base.bundle`, `synth_inversion_at_51c603b.zip`).

**Status of this spec.** The two torch-only test files WERE run, on a Windows venv (torch 2.7.1), against the spec's
reference changes in a throwaway copy of the base: `test_inharmonicity_loss.py` gave 3 failed / 7 passed on the old
code and 10 passed with the section 4 loss; `test_inharmonicity_model_integration.py` gave 6 failed on the old code
and 6 passed with the S5 change (16 of 16 together). Everything that needs Essentia (`test_inharmonicity_peaks.py`,
`test_inharmonicity_waveform.py`, `test_inharmonicity_phrase.py`, the fixture generator) or Surge
(`measure_phrase_timeline.py`) is UNRUN: Essentia has no Windows wheel and the author had no Linux machine. Those
files were checked only for syntax and against a plain-Python model of Essentia's two algorithms (from the upstream
source), which agrees with every hand-computed constant in the tests. Expect test-side errors there and follow R3.
Labels: **[V]** read in the code or the upstream source; **[I]** inferred, not confirmed; **[D]** a decision that
belongs to Kim, not to you.

Goal of the head (unchanged from the original brief): make the audio embedding RETAIN measured inharmonicity.
It is detached supervision. It is NOT a waveform loss and must never push sounds toward being more harmonic.

---

## 0. How to work (read before touching anything)

You are prone to subtle errors. These rules exist to make those errors visible. They override your own judgment.

- **R1 Branch.** Do step S0a first: it gets you the base commit and tags it `inharm-base`. Then
  `git switch -c fix/inharmonicity-aux inharm-base`. Never commit to `main` or to the feature branch. Never push, never
  rebase, never force anything. One commit per step, message `step Sx: <what>`.
- **R2 Allow-list.** You may edit ONLY: `scripts/synth_inversion/inharmonicity_target.py`,
  `inharmonicity_aux_head.py`, `synth_jepa_model.py`, `train_realistic_bass_v3.py`, `surge_spec_v3.py`
  (steps S6 only: add `phrase_events` / `phrase_note_spans`, make `render_patch` use them, fix the comment), and
  create the new files named in section 5. Any other file: STOP and ask (R11).
- **R3 Provided files are read-only.** Copy `provided/*` byte for byte into `scripts/synth_inversion/`. Never edit
  an assertion, constant or parametrize list; never add `skip`, `xfail`, `-k` filters or `--deselect`; never loosen
  a tolerance. If a provided test fails and you believe the TEST is wrong: STOP, paste the traceback, ask.
- **R4 Real output only.** After every command that matters, paste the last 15 lines. "Tests pass" requires the
  pytest summary line INCLUDING the skipped count. On the training machine always run with `REQUIRE_ALL=1`: any skip
  is a failure.
- **R5 No invented numbers.** Every value marked [D] in section 3 comes from Kim through a config JSON. The code has
  NO defaults for them (constructor fields without defaults). If you need a value to proceed, STOP and ask.
- **R6 Do not reconstruct Essentia.** Section 1.1 describes the UPSTREAM source. The installed version is the
  authority: step S1 generates fixtures from it; the parity tests compare against those fixtures.
- **R7 Minimal diffs.** No reformatting, renaming, import re-ordering or docstring rewrites outside lines you must
  change. After each step print `git diff --stat HEAD~1`; only allow-listed files may appear.
- **R8 No broad `except`.** Catch only the exception type that S1 shows Essentia raising. Never `except Exception`
  around anything that produces a label. Never convert an error into a valid label.
- **R9 Invalid is NaN, not zero.** The extractor returns `value = float("nan")` for every invalid result. The ONE
  place invalid becomes `(0.0, mask 0.0)` is `to_training_pair`. Do not write `0.0` for an invalid value anywhere else.
- **R10 Do not change what Surge renders.** The audio from `render_patch` must be identical before and after. Do not
  "fix" the note timeline (decision D6). Fixing the wrong comment is allowed and required.
- **R11 Stop and ask** if: S0a cannot give you the base commit by any of its five routes; you need a [D] value; a provided test seems wrong; the fixtures (S1) disagree with
  section 1.1; a checkpoint loader rejects the new keys; a file outside the allow-list needs a change; the S0
  timeline measurement contradicts section 1.2.
- **R12 Finish with the report** (section 7), listing every deviation. An empty "deviations" list must be true.
- **R13 Line endings.** At the base commit the files you edit are stored with CRLF line endings, and
  `inharmonicity_aux_head.py` mixes CRLF and LF [V: byte counts of the extracted files]. Preserve every file's existing
  endings. A tool that rewrites a whole file with LF produces a whole-file diff: if `git diff --stat` shows hundreds of
  changed lines for a small edit, you changed the endings, so revert that file and redo the edit. Run
  `git ls-files --eol <file>` before and after each edit. An exact-string edit that spans lines must use the file's own
  line ending (`\r\n`) or it will silently not match. New files you create (tests, scripts) are LF.

Known traps (each has bitten before): a tensor compared in `if` forces a host sync, so avoid it; `0.0 * NaN` is NaN,
so masking by multiplication does not remove NaN; broadcasting a `[B]` against a `[B,1]` silently makes `[B,B]`;
Essentia needs contiguous `float32` arrays; Python `bool` and numpy `bool_` are different objects (`is True` fails
on numpy bools; cast with `bool(...)`); pytest only imports siblings when run from `scripts/synth_inversion`.

---

## 1. Facts you may rely on

### 1.1 Essentia semantics [V: upstream master source; the INSTALLED version may differ, see R6]

`HarmonicPeaks(frequencies, magnitudes, pitch)` with parameters `maxHarmonics`, `tolerance`:
- `ratio = f_i / pitch`, `h = round(ratio)` (C++ round, halves away from zero), `distance = |ratio - h|`.
  A peak is a candidate for harmonic `h` if `distance <= tolerance` and `h > 0` and `ratio` is within the harmonic
  limit. **`tolerance` is a fraction of the PITCH, a fixed window of +-tolerance*pitch Hz around every harmonic**,
  not a fraction of the harmonic's own frequency.
- Per harmonic the closest candidate wins; ties go to the larger magnitude.
- The output ALWAYS has `maxHarmonics` entries. A missing harmonic is `((h+1) * pitch, 0.0)`: an ideal frequency
  with ZERO magnitude. Such placeholders are not observed partials.
- Raises on: size mismatch, `pitch < 0`, any frequency `<= 0`, unsorted or duplicate frequencies.
  `pitch == 0` or empty input returns empty output without raising.

`Inharmonicity(frequencies, magnitudes)`:
- `f0 = frequencies[0]` (the FIRST supplied frequency, NOT the pitch you passed to HarmonicPeaks). Raises if `f0 == 0`.
- For `i >= 1`: `ratio_i = round(f_i / f0)` (so harmonic numbers come from rounding, not from list position),
  `num += |f_i - ratio_i * f0| * m_i^2`; `den = m_0^2 + sum(m_i^2)`.
- Result `= num / (den * f0)` if `den != 0`, else **`1.0`**. **Empty input returns `0.0`.**
- Raises on: size mismatch, unsorted, duplicate frequencies.

Consequences you must respect:
- **C1** For non-degenerate input every term is at most `0.5*f0`, so the value lies in `[0, 0.5]`. `1.0` appears only
  for the all-zero-magnitude sentinel. Neither `0.0` for empty input nor `1.0` for all-zero magnitudes is a label.
- **C2** The descriptor is measured against the FIRST SELECTED peak. If harmonic 1 is present but detuned, `f0`
  moves with it. If harmonic 1 is missing, `f0` is the ideal pitch placeholder.
- **C3** Magnitude-squared weighting, normalized by `f0`. Uniform amplitude scaling and uniform frequency scaling
  (with `f0`) leave the value unchanged.
- **C4** Selection dropout: a partial displaced by more than `tolerance*pitch` is silently removed by
  `HarmonicPeaks`, and the value then looks MORE harmonic, not less.
- **C5** Zero-magnitude placeholders add nothing to `num` or `den`.

### 1.2 What the code at `51c603b` does [V unless marked]

Line numbers are at `51c603b`.
1. `inharmonicity_target.py:50-54` analyses the CENTRE 4096 samples (92.9 ms at 44.1 kHz) of the rendered clip, and
   `:65` takes f0 from `midi_note`, the pitch of the FIRST note only.
2. `surge_spec_v3.py:523-528` (`render_patch`) sends MIDI events with `time` values 0.0, 0.125, 0.125, 0.250. Its
   comments say "0.250 absolute" and "0.500 absolute". **Pedalboard reads `time` as ABSOLUTE seconds from the start
   of the buffer** (`sampleIndex = time * sampleRate`, no accumulation; read in `pedalboard/ExternalPlugin.h`).
   So what actually plays is note 1 (`midi`) 0.000-0.125 s, note 2 (`midi+12`) 0.125-0.250 s, then release tails to
   0.8 s. The comments are wrong. [V from source text; confirm empirically in S0.]
3. Consequence: the centre window (0.354-0.446 s) lies in the RELEASE TAIL, 104-196 ms after the last note-off,
   with f0 taken from the first note while the tail belongs to the second (one octave up). The embedding `za` is the
   CLS summary of all 81 mel frames of the whole 0.8 s phrase [V: `synth_jepa_model.py:36-91`]. So the label is
   measured on a different segment, in a different pitch, than what the encoder sees. (`note_dur` is also unused by
   `render_patch`.)
4. The render is mono, 44.1 kHz, peak-normalized to 1.0 (`:529-531`).
5. `synth_jepa_model.py:251` builds the head unconditionally, BEFORE `audio_encoder`, so it consumes random numbers
   that shift every later parameter's initialization; and line 1 puts the import above the module docstring.
6. `train_realistic_bass_v3.py:667,671` always adds `+ i_loss` (no flag, no weight); `:127` computes the label inside
   the DataLoader worker; `:138-143` `render_fixed_set` declares inharmonicity lists and never fills them.
7. Synth-JEPA here has NO EMA target encoder in the loss path: its stop-gradients are `zp.detach()` / `za.detach()`
   inside `forward` (`:288-289`). The EMA in the training script only shadows the online weights after each step.
   So "no gradient into the target/EMA encoder" is satisfied by construction; the old test's `target_embed` mock
   was never connected to anything (its assertion was vacuous).
8. Frequency reach: `max_harmonics * f0` is the highest frequency the descriptor can see. With `f0 = 41 Hz` and 20
   harmonics that is 820 Hz. Energy above it counts as "excluded" in the coverage diagnostic even if perfectly
   harmonic. Print the real f0 range from `NOTE_LOW` / `NOTE_HIGH` in S0 before Kim chooses `max_harmonics` (D3).

### 1.3 Two worked examples (check them by hand)

- **Dropout labelled harmonic.** Peaks `[100, 200, 250, 350]` Hz, magnitudes `[1, .5, .5, .5]`, pitch 100,
  tolerance 0.2. 250 gives ratio 2.5 and 350 gives 3.5: both are 0.5 away from an integer, so both are dropped.
  Selected = {100, 200}. Value = 0.0. Coverage = `1.25/1.75 = 0.714`. The old policy (`coverage >= 0.5`, two
  partials) therefore reported `valid = True, value = 0.0`: "perfectly harmonic" for a sound whose upper partials
  are all displaced. Excluded fraction `= 1 - 1.25/1.75 = 0.2857`.
- **Hand value.** Peaks `[100, 200, 305]`, magnitudes `[1, .5, .5]`. `f0 = 100`; 305 gives `round(3.05) = 3`,
  deviation 5 Hz; weights 1, .25, .25; `num = 5 * .25 = 1.25`; `den = 1.5`; value `= 1.25 / (1.5 * 100) = 0.008333`.

---

## 2. Defects to fix

| ID | Where (at 51c603b) | Problem | Outcome | Step |
|---|---|---|---|---|
| B1 | target `:50-65`, `surge_spec_v3.py:523-528`, train `:127` | Label from the wrong time window and the wrong pitch; scope mismatch with the embedding (1.2 items 1-3) | Windows come from the same timeline `render_patch` plays; per-note f0; pooled by an explicit rule; field named for what it is | S6 |
| B2 | target `:88-97` | Coverage rule admits dropout (1.3); both rules overwrite each other's reason | Excluded-energy gate with a configured limit; every failing gate recorded; first reason kept | S4 |
| B3 | target `:59-81` | No peak-array entry point; Essentia can raise inside a DataLoader worker; gates run after the call; result finiteness unchecked; `sample_rate` not passed to `SpectralPeaks`; no `analysis_configuration_id`; thresholds hard-coded | Pure `inharmonicity_from_peaks`, explicit pre-checks mirroring Essentia's raises, gates BEFORE computing, config object with id | S4 |
| B4 | aux head `:25-52` | `loss * mask` lets `NaN` through; no shape guard (silent broadcast); host-sync branch | NaN-safe masking, shape guard, branch-free zero | S3 |
| B5 | model `:1,251`; train `:667,671` | Always on; shifts RNG; old checkpoints; no weight | Off by default (`--inharm_weight 0`), head built last and only when enabled | S5 |
| B6 | train `:138-143`, loop | No valid-rate or reason monitoring; no held-out check | `valid_frac` logged; offline report script; held-out eval proposed to Kim, not built | S7 |
| B7 | `test_inharmonicity_target.py`, `test_gradient_boundaries.py` | Not tests: no assertions on the key safeguards, run at import, zero collected, vacuous check | Replace with the provided files | S2 |

---

## 3. Decisions reserved for Kim [D]

Ask for each; record the answer in the report. Provide recommendations, never adopt them yourself.

- **D1 Label scope.** The embedding spans the whole phrase; no single segment is "exact". *Recommended:* one analysis
  window per played note (inside its sustain, ending at its note-off), per-note f0, pooled by the mean of VALID
  windows, label valid only if `min_valid_windows` (recommended 2) windows are valid. Name the field
  `inharmonicity_note_windows`; do not call it a clip descriptor. The provided `test_inharmonicity_phrase.py` is
  valid only for this choice.
- **D2 Fundamental.** MIDI note is nominal, not measured: oscillator octaves, unison detune, FM and sync move the
  true fundamental. *Recommended:* nominal f0 PLUS `require_fundamental_peak = true` (a spectral peak must exist
  within tolerance of the nominal pitch); log a YIN estimate as a diagnostic only until Kim has seen the distribution.
- **D3 Thresholds** (config JSON, no defaults in code): `tolerance`, `max_harmonics`, `min_selected_partials`,
  `max_excluded_energy_fraction`, `attack_guard_s`, `min_valid_windows`, `silence_peak_abs`, `max_level_change_db`
  (null = diagnostic only), `frame_size`, `sample_rate`, `min_partial_relative_db` (a selected peak counts as a
  partial only if it is within this many dB of the strongest input peak; no continuity value exists, the old code
  counted any nonzero magnitude). "Continuity" values already in the code: tolerance 0.2,
  max_harmonics 20, min partials 2, silence 1e-4, frame 4096, and an excluded limit of 0.5 (the old coverage 0.5),
  which is KNOWN to admit the worked example in 1.3. Do not propose 0.5.
- **D4 Auxiliary weight.** Default 0 (off). Kim sets any other value; log aux versus JEPA loss at least.
- **D5 Target and head scale.** Real values lie in `[0, 0.5]` (C1) while the head ends in a sigmoid (range 0-1,
  starts near 0.5). Options: keep, or fit a mean/std on a pilot sample of valid train-prior draws and use a linear
  head. Kim decides; do not change the head until told.
- **D6 Render timeline.** The code plays 0.000-0.125 / 0.125-0.250 s; the comments intend 0.250 / 0.500. Changing
  it alters the training distribution and invalidates existing checkpoints. *Default: leave the render as is.*

---

## 4. Required API: `scripts/synth_inversion/inharmonicity_target.py`

Exact names; the provided tests import them. Keep `Reason` strings exactly; tests compare to the constants.

```python
EXTRACTOR_VERSION = "inharm-2"      # bump on any behaviour change

class Reason:
    NONFINITE_INPUT = "nonfinite input"
    SIZE_MISMATCH = "frequency and magnitude arrays differ in length"
    NO_PEAKS = "no spectral peaks"
    BAD_FUNDAMENTAL = "fundamental is not positive"
    UNSORTED_OR_DUPLICATE = "peak frequencies not strictly ascending and positive"
    ESSENTIA_ERROR = "essentia raised"
    NO_FUNDAMENTAL_PEAK = "no spectral peak near the nominal fundamental"
    TOO_FEW_PARTIALS = "too few selected partials"
    EXCLUDED_ENERGY = "excluded energy fraction above limit"
    NONFINITE_RESULT = "nonfinite result"
    SILENCE = "silence"
    WINDOW_LENGTH = "window length differs from frame_size"
    LEVEL_UNSTABLE = "level change above limit"
    TOO_FEW_VALID_WINDOWS = "too few valid note windows"
    ALL = frozenset({...all of the above...})
```

`InharmonicityConfig`: frozen dataclass, **every field required, no defaults**: `sample_rate:int`, `frame_size:int`,
`max_harmonics:int`, `tolerance:float`, `min_selected_partials:int`, `max_excluded_energy_fraction:float`,
`require_fundamental_peak:bool`, `silence_peak_abs:float`, `max_level_change_db:Optional[float]`,
`attack_guard_s:float`, `min_valid_windows:int`, `min_partial_relative_db:float` (twelve fields). Methods: `from_json(path)` (strict: exactly these keys, else
`ValueError`), `config_id()` = first 16 hex chars of `sha256(json.dumps({**asdict(self), "extractor_version":
EXTRACTOR_VERSION, "essentia_version": essentia.__version__}, sort_keys=True))`.

**Result dict** (`inharmonicity_from_peaks` and `process_window` return exactly these keys, invalid results too;
`pool_note_results` and `process_phrase` add `n_valid_windows` and `window_results`, and set `fundamental_hz` to
`None`, `fundamental_source` to `"per_note_midi"`, and every diagnostic except `tolerance` and `max_harmonics` to
`None`):
`value` (float; `nan` when invalid), `valid` (Python bool), `rejection_reason` (first reason or None),
`all_rejection_reasons` (list, gate order), `fundamental_hz`, `fundamental_source`, `analysis_configuration_id`,
`coverage_diagnostics` with exactly: `input_peak_count`, `selected_peak_count` (an `int`: the number of selected
INFORMATIVE partials, meaning magnitude > 0 AND at least `max(input magnitudes) * 10 ** (-min_partial_relative_db
/ 20)`; this keeps tiny noise peaks that happen to fall inside a harmonic window from counting as partials),
`total_peak_energy`, `selected_peak_energy`, `selected_energy_fraction`,
`excluded_energy_fraction`, `tolerance`, `max_harmonics`, `first_selected_hz`, `level_change_db`. A diagnostic that
was not computed is `None`, never `0`. Energies are sums of magnitude squared in float64.

**`inharmonicity_from_peaks(freqs, mags, f0_hz, cfg, fundamental_source="midi_note")`** order of work:
1. Pre-checks, each returns immediately: non-finite anywhere (including `f0_hz`) -> `NONFINITE_INPUT`; lengths
   differ -> `SIZE_MISMATCH`; empty -> `NO_PEAKS`; `f0_hz <= 0` -> `BAD_FUNDAMENTAL`; frequencies not strictly
   ascending or any `<= 0` -> `UNSORTED_OR_DUPLICATE`. These mirror Essentia's own raises, so Essentia never raises.
2. `HarmonicPeaks(maxHarmonics, tolerance)` on `float32` copies (never mutate the caller's arrays). Build the
   algorithm objects once per `(max_harmonics, tolerance)` per process (`functools.lru_cache`).
3. Diagnostics. Evaluate ALL gates, in this order, collecting reasons: `NO_FUNDAMENTAL_PEAK` (only if
   `require_fundamental_peak` and the first selected magnitude is 0); `TOO_FEW_PARTIALS` (`selected_peak_count <
   min_selected_partials`); `EXCLUDED_ENERGY` (`excluded_energy_fraction` is not `None` and `> limit`). If total
   energy is 0 the fractions are `None` and `TOO_FEW_PARTIALS` applies.
4. If any gate failed: `valid=False`, `value=nan`, do NOT call `Inharmonicity`.
5. Else call `Inharmonicity` on the selected arrays; non-finite -> `NONFINITE_RESULT`. A raise (should be
   unreachable) -> `ESSENTIA_ERROR`, catching only the type S1 recorded.

**`InharmonicityExtractor(cfg)`**:
- `process_window(window, f0_hz, fundamental_source)` order: non-finite audio -> `NONFINITE_INPUT`; `len(window) !=
  frame_size` -> `WINDOW_LENGTH` (never crop or pad); `max|x| < silence_peak_abs` -> `SILENCE`; then
  `Windowing('hann')` -> `Spectrum()` -> `SpectralPeaks(sampleRate=cfg.sample_rate, orderBy='frequency',
  magnitudeThreshold=1e-5, minFrequency=20.0, maxFrequency=cfg.sample_rate/2.0, maxPeaks=100)` -> the peaks function.
  Then `level_change_db = 20*log10((rms(second half)+1e-12)/(rms(first half)+1e-12))` always into the diagnostics;
  if `max_level_change_db` is not `None` and `abs(level) >` it: append `LEVEL_UNSTABLE`, set invalid, `value=nan`.
- `process_phrase(audio, midi_note)` (S6): `spans = surge_spec_v3.phrase_note_spans(midi_note)` (import it INSIDE the
  method), `note_windows`, `process_window` per window, `pool_note_results`.

**`note_windows(spans, cfg)`** -> list the length of `spans`; each item is `None` or `{"note", "f0_hz", "start_sample",
"end_sample"}`: `end_sample = round(off_s * sample_rate)`, `start_sample = end_sample - frame_size`, `f0_hz = 440 *
2 ** ((note - 69) / 12)`; `None` if `start_sample < round((on_s + attack_guard_s) * sample_rate)` (drop, never shorten).

**`pool_note_results(window_results, cfg)`**: mean of the VALID windows' values; `n_valid_windows`; valid only if
`n_valid_windows >= min_valid_windows` (else `TOO_FEW_VALID_WINDOWS`, value `nan`); `None` items count as invalid;
never count an invalid window as zero; include the per-window results under `window_results`.

**`to_training_pair(result)`** -> `(value, 1.0)` if valid, `(0.0, 0.0)` if invalid; a result with `valid=True` and a
non-finite value raises `ValueError`.

**`RejectionStats`**: `add(result, midi_note)`, `summary()` -> `{"n","n_valid","valid_rate","by_reason","valid_rate_by_note"}`
(`by_reason` counts the `rejection_reason` of invalid results only), `reset()`.

**Loss** (`inharmonicity_aux_head.py`), reference sketch, UNTESTED:

```python
def masked_inharmonicity_loss(pred, target, valid_mask):
    if not (pred.shape == target.shape == valid_mask.shape):
        raise ValueError("pred, target and valid_mask must have identical shapes")
    target = target.detach()
    valid = valid_mask.detach() != 0                       # bool or 0/1 float
    safe = torch.where(valid, target, torch.zeros_like(target))   # NaN/inf never reach the arithmetic
    if not torch.isfinite(safe).all():
        raise ValueError("non-finite target on a row flagged valid")
    sq = (pred - safe) ** 2
    masked = torch.where(valid, sq, torch.zeros_like(sq))
    n = valid.sum().to(pred.dtype)
    return masked.sum() / n.clamp_min(1.0), n               # all-invalid -> exact 0 that keeps its graph
```

---

## 5. Work plan

Run all pytest commands from `scripts/synth_inversion` with `REQUIRE_ALL=1`. Commit after every step.

- **S0a Get the base commit. Nothing else may start before this works.** The commit exists on GitHub. This prints
  the full SHA above (checked October 10, 2026; the branch tip IS that commit):
  `git ls-remote https://github.com/Taikakim/audio-tools-avp.git refs/heads/feature/surge-xt-neural-inversion`.
  Work through the routes in order, stop at the first that succeeds, and paste the output of every command you run:
  1. `git remote -v`. One remote must point at `Taikakim/audio-tools-avp` (HTTPS or SSH). Use ITS name wherever
     this step says `<remote>`; it may be `avp`, not `origin`. If none does, you are in the wrong clone (for example
     a stable-audio-tools clone, or avp-audio-craft). Clone the right repo into a NEW directory:
     `git clone https://github.com/Taikakim/audio-tools-avp.git`.
  2. `git fetch <remote> +refs/heads/feature/surge-xt-neural-inversion:refs/remotes/<remote>/feature/surge-xt-neural-inversion`.
     Use the explicit refspec: in a single-branch clone a bare `git fetch <remote> <branch>` only fills `FETCH_HEAD`
     and leaves nothing to switch to.
  3. Check: `git cat-file -t 51c603b8c84ded947537b93b53af1ced16f44ae8` must print `commit`.
  4. No network (a sandboxed agent), or step 2 fails: use `inharm-base.bundle` from this folder. It holds the 37
     commits after `origin/main` and requires commit `018ae1a209922ce6276f57a3faa3377e89c90170` (the tip of `main`
     on October 10, 2026), which your clone must already have. Run `git bundle verify <path>/inharm-base.bundle`,
     then `git fetch <path>/inharm-base.bundle refs/remotes/origin/feature/surge-xt-neural-inversion:refs/heads/inharm-base-branch`,
     then repeat check 3.
  5. No clone at all, or the bundle's prerequisite is missing: use `synth_inversion_at_51c603b.zip` (the files of
     `scripts/synth_inversion` at that commit, 482 KB). Make a NEW directory, extract the zip into it, then
     `git init`, `git add -A`, and `git commit -m "base: scripts/synth_inversion at 51c603b8c84ded947537b93b53af1ced16f44ae8"`.
     That commit is your base. You cannot diff against the real history in this mode: say so in the report.
  Then `git tag inharm-base <the base commit>` and use `inharm-base` in every later command. Paste
  `git log -1 --format=%H inharm-base`; in routes 1-4 it must print exactly the full SHA above. Never substitute a
  different commit, even if a branch with the same name exists with different content.
- **S0 Baseline (no code changes).** Branch (R1). Print `python --version`, torch / numpy / essentia versions.
  Print `NOTE_LOW`, `NOTE_HIGH`, the f0 range they imply, and `20 * f0` for both ends and for the `+12` note. Copy
  `provided/measure_phrase_timeline.py` and `provided/make_inharmonicity_fixtures.py` into
  `scripts/synth_inversion/`. Run `python measure_phrase_timeline.py --hash-out render_baseline_hashes.json` and
  paste the whole table. Do not interpret it; Kim decides D6 and D1 from it.
- **S1 Fixtures.** `python make_inharmonicity_fixtures.py`. Paste the printed table. Check it against section 1.1
  (empty input -> 0.0, all-zero magnitudes -> 1.0, the raised exception type and message for unsorted / duplicate /
  zero frequency / pitch < 0 / size mismatch). Any mismatch: STOP (R11). Note the exception TYPE for R8. Commit the
  generator and the JSON.
- **S2 Tests in, red.** Copy the five `provided/test_*.py` files byte for byte. `git rm` the two old test files
  (B7). Run them: they must FAIL on import or assertion. Paste the failing summary (proof the tests are real).
- **S3 Loss (B4).** Implement `masked_inharmonicity_loss` per section 4. Green: `test_inharmonicity_loss.py` (all).
- **S4 Extractor (B2, B3).** Rewrite `inharmonicity_target.py` per section 4 (everything except `process_phrase`,
  `note_windows`, `pool_note_results`, `to_training_pair`, `RejectionStats`, which are S6). Green:
  `test_inharmonicity_peaks.py` and `test_inharmonicity_waveform.py` (run the latter with `-s`; paste the SNR lines).
  If the parity tests disagree with a hand-computed constant: STOP (R6).
- **S5 Model and training flag (B5).** In `SynthJEPA.__init__` add `inharmonicity_head: bool = False` as the LAST
  parameter; build the head as the LAST statement, only if the flag is true; move the import below the docstring.
  In the training script add `--inharm_weight` (float, default 0.0) and `--inharm_config` (path); weight > 0 without
  a config is a parse error. Weight 0: no head, no extractor, no loss term, dataset returns `(…, 0.0, 0.0)`.
  Weight > 0: `jepa_kwargs["inharmonicity_head"] = True`, loss term `args.inharm_weight * i_loss` in BOTH loss lines.
  Paste `grep -n "SynthJEPA(" -r scripts/` and confirm every call site; check how `FORMAT_VERSION` and the
  checkpoint loader treat the new keys (R11). Green: `test_inharmonicity_model_integration.py`.
- **S6 Alignment (B1). Needs Kim's answers to D1, D3 and D6 first.** In `surge_spec_v3.py` add
  `phrase_events(midi_note) -> [(type, note, velocity, time_s)]` and `phrase_note_spans(midi_note) ->
  [{"note","on_s","off_s"}]`; make `render_patch` build its messages from `phrase_events` (same values as today,
  see the test); correct the comment to state the ACTUAL absolute times. Implement the S6 items of section 4 and
  switch the dataset to `process_phrase` + `to_training_pair`, only when `inharm_weight > 0`. Green:
  `test_inharmonicity_phrase.py`. Then `python measure_phrase_timeline.py --hash-check render_baseline_hashes.json`
  must report identical (or the baseline must have said "not deterministic").
- **S7 Monitoring (B6).** Log `valid_frac = inharm_valid.mean()` with the existing log line. Create
  `report_inharmonicity_validity.py`: render N patches through the dataset path, feed `RejectionStats`, print the
  summary and the per-note valid rate; no asserts. Do NOT edit `validate()` or `render_fixed_set`: write a
  one-paragraph proposal for a held-out descriptor error against a constant-prediction baseline and ask Kim.
- **S8 Verify and report.** Run all five test files, `REQUIRE_ALL=1`, 0 skipped. Run
  `diff` of each `provided/*` file against its copy: empty. `git diff --stat inharm-base` shows only allow-listed
  files. Fill in section 7. Do NOT push.

---

## 6. Acceptance checklist (all must be true, each backed by pasted output)

1. Five test files green, 0 skipped, with `REQUIRE_ALL=1` on the training machine.
2. Provided files byte-identical to the copies (`diff` output pasted).
3. `--inharm_weight 0` training path: no head keys in the state dict, JEPA losses equal to baseline (the provided
   integration test), and a short smoke run of the training script reaches the loop and logs a step.
4. `render_patch` audio unchanged (hash check) or events test green if the render was not deterministic.
5. No file outside the allow-list changed.
6. Every [D] answer recorded in the report; no threshold invented.

---

## 7. Report template (copy into your final reply)

```
ENVIRONMENT   python / torch / numpy / essentia versions; machine
S0a route used (1-5) and every command run; output of `git log -1 --format=%H inharm-base`
S0  commit <sha>; f0 range and 20*f0 values; timeline table (pasted); render deterministic: yes/no
S1  commit <sha>; essentia version in fixtures; fixture table (pasted); mismatches with spec 1.1: none / list
S2  commit <sha>; red summary line (pasted)
S3  commit <sha>; pytest summary line (pasted)
S4  commit <sha>; pytest summary lines for peaks and waveform; the SNR table (pasted)
S5  commit <sha>; pytest summary; SynthJEPA( call sites; FORMAT_VERSION / loader finding
S6  BLOCKED-ON-KIM or commit <sha>; pytest summary; hash-check result
S7  commit <sha>; sample output of report_inharmonicity_validity.py; proposal paragraph for held-out eval
S8  final test run (pasted); provided-file diff (pasted); git diff --stat inharm-base (pasted)
DECISIONS     D1..D6: answer received / still open
DEVIATIONS    must be "none", or each one listed with the reason and who approved it
UNEXPECTED    anything surprising, including anything in this spec that looked wrong
```
