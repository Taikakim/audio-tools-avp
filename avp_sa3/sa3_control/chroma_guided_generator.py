"""Part 3 — Chroma (HPCP) guidance: the *development* axis for long-form SA3.

A chroma target that **moves per window** == a chord progression == melodic/harmonic
development. Drives the trained 12-channel HPCP LatCH **guidance** head
(``latch_weights_sa3_medium/latch_sa3_hpcp_best.pt``) via TFG
(``sample_flow_euler_multi_latch_guided``), which is *separate* from the onset
adapter (a forward mod via ``use_control_context``). They compose: the guided
sampler's DiT forwards run inside the onset control context, and the adapter is
differentiable, so guidance gradients flow through it cleanly.

What is PURE + unit-tested in this pass (no GPU):
  * ``chord_to_chroma``   — chord name / pc-set -> L2-normalized (12,) template.
  * ``parse_progression`` — '0:Am|32:F|64:C' -> [(0.0,'Am'),(32.0,'F'),(64.0,'C')].
  * ``ChromaSchedule``    — time-varying (1,12,n_frames) target on the latent grid,
                            with head-aware standardization.

What is STUB-THIS-PASS (importable but GPU/model-bound — see ``ChromaGuidedGenerator``):
  * ``ChromaGuidedGenerator.generate`` — needs the SA3 model loaded **fp32**
    (MASTER §5: TFG/LatCH guidance must run fp32; fp16 clashes with backprop grad
    dtypes) + the GPU. The interface and the schedule are final; runtime quality
    (rho/mu ~48-96 sweep, window length, steps, fp32-VRAM on 16 GB) needs a launch
    session. Judge by Audiobox CE / spread, never by rank-corr (the corr=1.0 mirage).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pitch-class & chord-template tables (pure, no torch needed)
# ---------------------------------------------------------------------------

# Note name -> pitch class (0..11). Both sharps and flats.
_NOTE_TO_PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11, "B#": 0,
}

# Chord quality -> intervals (semitones from root). Order: longest suffix wins.
_QUALITY_INTERVALS = {
    "": (0, 4, 7),            # major triad
    "maj": (0, 4, 7),
    "major": (0, 4, 7),
    "m": (0, 3, 7),           # minor triad
    "min": (0, 3, 7),
    "minor": (0, 3, 7),
    "-": (0, 3, 7),
    "dim": (0, 3, 6),
    "o": (0, 3, 6),
    "aug": (0, 4, 8),
    "+": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "sus": (0, 5, 7),
    "5": (0, 7),              # power chord
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "min6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),       # dominant 7
    "maj7": (0, 4, 7, 11),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),    # half-diminished
    "9": (0, 4, 7, 10, 2),
    "maj9": (0, 4, 7, 11, 2),
    "m9": (0, 3, 7, 10, 2),
    "min9": (0, 3, 7, 10, 2),
    "add9": (0, 4, 7, 2),
}

# Roman-numeral degree -> (scale-step semitone offset in a major scale, is_minor).
# Used only when a progression entry is a degree and a `key` is supplied.
_DEGREE_OFFSETS = {
    "I": (0, False), "II": (2, False), "III": (4, False), "IV": (5, False),
    "V": (7, False), "VI": (9, False), "VII": (11, False),
    "i": (0, True), "ii": (2, True), "iii": (4, True), "iv": (5, True),
    "v": (7, True), "vi": (9, True), "vii": (11, True),
}


def _root_to_pc(token: str) -> Optional[int]:
    """'C', 'F#', 'Bb' -> pitch class; None if not a note name."""
    t = token.strip()
    if not t:
        return None
    head = t[0].upper()
    rest = t[1:2]
    if rest in ("#", "b", "B"):
        key = (head + ("#" if rest == "#" else "b")).upper()
        return _NOTE_TO_PC.get(key)
    return _NOTE_TO_PC.get(head)


def _resolve_degree(name: str, key: str) -> Optional[tuple[int, str]]:
    """Roman-numeral degree relative to `key` -> (root_pc, quality_suffix).

    Lowercase numeral -> minor, uppercase -> major. A trailing 'o' -> dim.
    Returns None if `name` is not a recognised degree.
    """
    base = name
    dim = False
    if base.endswith("o") and base[:-1] in _DEGREE_OFFSETS:
        base, dim = base[:-1], True
    if base not in _DEGREE_OFFSETS:
        return None
    key_pc = _root_to_pc(key)
    if key_pc is None:
        return None
    offset, is_minor = _DEGREE_OFFSETS[base]
    root_pc = (key_pc + offset) % 12
    quality = "dim" if dim else ("m" if is_minor else "")
    return root_pc, quality


def chord_to_chroma(name: str, *, sharpness: float = 1.0,
                    key: Optional[str] = None) -> np.ndarray:
    """Chord name / pc-set / degree -> ``np.ndarray(12)`` in [0,1], L2-normalized.

    Accepted ``name`` forms:
      * Note-name chords: ``'Am'``, ``'C'``, ``'Dm7'``, ``'F#'``, ``'Gmaj7'``, ``'Bb7'``.
      * Raw pitch-class sets: ``'0,4,7'`` (any comma/space separated ints mod 12).
      * Roman-numeral degrees (only if ``key`` given): ``'I'``, ``'vi'``, ``'viio'``.

    ``sharpness`` scales non-chord-tone leakage: ``1.0`` (default) = a hard template
    (chord tones 1.0, others 0.0); ``< 1.0`` lets the off-tones leak in by
    ``max(0, 1 - sharpness)``; ``> 1.0`` is clamped to the hard template.

    The returned vector is L2-normalized (chroma is a *direction* — the head was
    trained with cosine/smooth_l1, and the guided sampler treats the target as a
    direction). An all-zero request (no valid tones) returns the zero vector.
    """
    name = (name or "").strip()
    pcs: list[int] = []

    # raw pc-set: any token that is purely digits/commas/spaces
    stripped = name.replace(",", " ").split()
    if stripped and all(tok.lstrip("-").isdigit() for tok in stripped):
        pcs = [int(tok) % 12 for tok in stripped]
    else:
        root_pc: Optional[int] = None
        intervals: Optional[Sequence[int]] = None

        # try roman-numeral degree first (only meaningful with a key)
        deg = _resolve_degree(name, key) if key else None
        if deg is not None:
            root_pc, quality = deg
            intervals = _QUALITY_INTERVALS.get(quality, _QUALITY_INTERVALS[""])
        else:
            # note-name chord: split root (1-2 chars) from quality suffix
            if name:
                if len(name) >= 2 and name[1] in ("#", "b", "B") and name[1] != "":
                    root_tok, quality = name[:2], name[2:]
                else:
                    root_tok, quality = name[:1], name[1:]
                root_pc = _root_to_pc(root_tok)
                intervals = _QUALITY_INTERVALS.get(quality)
                if intervals is None:
                    # unknown quality -> treat as bare root major triad
                    intervals = _QUALITY_INTERVALS[""]
        if root_pc is not None and intervals is not None:
            pcs = [(root_pc + iv) % 12 for iv in intervals]

    vec = np.zeros(12, dtype=np.float64)
    if pcs:
        leak = max(0.0, 1.0 - float(sharpness))
        vec[:] = leak
        for pc in pcs:
            vec[pc] = 1.0
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec = vec / n
    return vec.astype(np.float32)


def parse_progression(arg: str) -> "str | list[tuple[float, str]]":
    """'0:Am|32:F|64:C|96:G' -> [(0.0,'Am'),(32.0,'F'),(64.0,'C'),(96.0,'G')].

    Mirrors the ``'t:value'`` schedule shape (same convention as the prompt arc).
    A single chord with no leading 'float:' (e.g. ``'Am'``) returns the bare string.
    Treated as a schedule ONLY if every '|'-segment starts with 'float:'. Never
    raises; falls back to the single-string on any parse miss.
    """
    arg = (arg or "").strip()
    if not arg:
        return arg
    if "|" not in arg and ":" not in arg:
        return arg
    segments = arg.split("|")
    entries: list[tuple[float, str]] = []
    for seg in segments:
        if ":" not in seg:
            return arg  # not a schedule
        t_str, _, val = seg.partition(":")
        try:
            t = float(t_str.strip())
        except ValueError:
            return arg  # leading token not a float -> single prompt (colon-safe)
        entries.append((t, val.strip()))
    if not entries:
        return arg
    return entries


class ChromaSchedule:
    """Time-varying 12-d chroma target = a chord progression on the latent grid.

    Pure / model-free. Holds a list of ``(t_sec, chord_name)`` changes and renders a
    per-frame ``(1, 12, n_frames)`` target for any window, standardizing to match how
    the HPCP head was trained when ``head_metadata`` says so.
    """

    def __init__(self, progression: Sequence[tuple[float, str]], *, fps: float,
                 head_metadata: Optional[dict] = None,
                 sharpness: float = 1.0, key: Optional[str] = None,
                 blend_sec: float = 0.0) -> None:
        """progression: ``[(t_sec, chord_name), ...]``; sorted internally, first t
        must be ``0.0`` (a chord must be defined from the start).

        head_metadata: a LatCH ``.metadata`` dict. If it has
        ``standardized``/``std_mean``/``std_std`` the target is standardized
        ``(x - std_mean) / std_std`` (scalar), exactly as ``model.py`` does before
        handing targets to the guided sampler.

        sharpness/key: forwarded to ``chord_to_chroma``.
        blend_sec: if > 0, linearly cross-blend the chroma across a chord boundary
        over this many seconds (centered on the change); 0 = hard step.
        """
        if not progression:
            raise ValueError("progression must have at least one (t, chord) entry")
        prog = sorted(((float(t), str(c)) for t, c in progression), key=lambda e: e[0])
        if abs(prog[0][0]) > 1e-9:
            raise ValueError(
                f"progression must start at t=0.0 (got {prog[0][0]}); a chord must "
                "be defined from the start of the render")
        self.progression = prog
        self.fps = float(fps)
        self.head_metadata = head_metadata or {}
        self.sharpness = float(sharpness)
        self.key = key
        self.blend_sec = float(blend_sec)

        # precompute the (12,) chroma per change, plus change times
        self._times = np.array([t for t, _ in prog], dtype=np.float64)
        self._chromas = np.stack(
            [chord_to_chroma(c, sharpness=self.sharpness, key=self.key) for _, c in prog],
            axis=0,
        ).astype(np.float64)  # (n_changes, 12)

        meta = self.head_metadata
        self._standardize = bool(meta.get("standardized"))
        self._std_mean = float(meta.get("std_mean", 0.0))
        self._std_std = float(meta.get("std_std", 1.0)) or 1.0

    def chroma_at(self, t_sec: float) -> np.ndarray:
        """The (12,) chroma active at absolute time ``t_sec`` (unstandardized).

        Step by default; if ``blend_sec > 0`` a linear cross-blend is applied
        symmetrically over a window centered on the nearest chord change.
        """
        idx = int(np.searchsorted(self._times, t_sec + 1e-9, side="right") - 1)
        idx = max(0, min(idx, len(self._times) - 1))
        if self.blend_sec <= 0.0:
            return self._chromas[idx]
        half = self.blend_sec / 2.0
        # Find a chord boundary (between segment k-1 and k) within `half` of t.
        for k in range(1, len(self._times)):
            tc = self._times[k]
            if tc - half <= t_sec < tc + half:
                frac = (t_sec - (tc - half)) / max(self.blend_sec, 1e-9)
                v = (1.0 - frac) * self._chromas[k - 1] + frac * self._chromas[k]
                n = float(np.linalg.norm(v))
                return v / n if n > 0 else v
        return self._chromas[idx]

    def target_numpy(self, t_start_sec: float, n_frames: int) -> np.ndarray:
        """``(12, n_frames)`` float32 target (standardized if the head says so)."""
        frame_times = t_start_sec + np.arange(n_frames, dtype=np.float64) / self.fps
        out = np.empty((12, n_frames), dtype=np.float64)
        for f, ft in enumerate(frame_times):
            out[:, f] = self.chroma_at(float(ft))
        if self._standardize:
            out = (out - self._std_mean) / self._std_std
        return out.astype(np.float32)

    def target(self, t_start_sec: float, n_frames: int,
               device="cpu", dtype=None):
        """``(1, 12, n_frames)`` torch tensor for the window covering
        ``[t_start_sec, t_start_sec + n_frames/fps)``.

        Resolves chord changes within the window (step, or linear blend if
        ``blend_sec > 0``) and standardizes when ``head_metadata`` requires it.
        ``torch`` is imported lazily so the pure ``target_numpy`` path stays
        import-light for CPU unit tests.
        """
        import torch
        if dtype is None:
            dtype = torch.float32
        arr = self.target_numpy(t_start_sec, n_frames)
        t = torch.from_numpy(arr).to(device=device, dtype=dtype)
        return t.unsqueeze(0)  # (1, 12, n_frames)


# ---------------------------------------------------------------------------
# Loader (thin wrapper; importable, exercised only when a checkpoint exists)
# ---------------------------------------------------------------------------

def load_chroma_head(path: str, device="cpu"):
    """Load the HPCP LatCH guidance head via the canonical SA3 loader.

    Returns a ``LatCH`` with ``.metadata`` attached (``std_mean``/``std_std``/
    ``standardized``/``loss_type``/``t_injection``/...). Use this metadata to build
    a matching ``ChromaSchedule(head_metadata=head.metadata)``.

    NOTE (MASTER §5): for TFG guidance the head **and** the model must run fp32.
    ``load_latch_from_checkpoint`` loads fp32 by default; do not half() it.
    """
    from stable_audio_3.models.latch import load_latch_from_checkpoint
    return load_latch_from_checkpoint(path, device=device)


# ---------------------------------------------------------------------------
# ChromaGuidedGenerator — STUB-THIS-PASS runtime (importable, GPU/model-bound)
# ---------------------------------------------------------------------------

class ChromaGuidedGenerator:
    """Inpaint-continuation + HPCP TFG guidance. Same ``ChunkGenerator`` contract.

    Builds the same inpaint ``cond_inputs`` as ``InpaintContinuationGenerator``
    (mask + masked_input from the prefix), then runs the gradient-enabled sampler
    ``sample_flow_euler_multi_latch_guided`` with one HPCP guide whose target
    **moves per window** (the development axis).

    The generate() *runtime* is STUB-THIS-PASS: it needs the SA3 model loaded
    **fp32** on the GPU (MASTER §5). The interface, the cond build, the guide dict,
    and the time bookkeeping are final; only on-GPU validation (rho/mu sweep,
    fp32-VRAM on 16 GB, window/steps tuning) remains. The class subclasses
    ``ChunkGenerator`` when ``stable_audio_3`` is importable, else stays a plain
    object so the pure ``ChromaSchedule`` tests need no SA3 install.
    """

    def __init__(self, model, hpcp_head, chroma_schedule: ChromaSchedule, *,
                 steps: int = 50, cfg_scale: float = 6.0,
                 rho: float = 64.0, mu: float = 64.0, gamma: float = 0.3,
                 n_iter: int = 4, start_pct: float = 0.0, end_pct: float = 1.0,
                 weight: float = 1.0, loss_type: Optional[str] = None) -> None:
        """model: a ``StableAudioModel`` (``.model`` is the DiT wrapper).
        hpcp_head: ``load_chroma_head(latch_sa3_hpcp_best.pt)`` (fp32, on cuda).
        chroma_schedule: a ``ChromaSchedule`` built with ``head_metadata=head.metadata``.
        rho/mu: variance/mean guidance strengths (the MASTER §5 'gain ~48-96' knob).
        loss_type: defaults to the head's trained ``loss_type`` (smooth_l1 for the
            shipped HPCP head) if available, else 'cosine' (chroma is a direction).
        """
        self.model = model
        self.inner = getattr(model, "model", None)
        self.head = hpcp_head
        self.schedule = chroma_schedule
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.rho = float(rho)
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.n_iter = int(n_iter)
        self.start_pct = float(start_pct)
        self.end_pct = float(end_pct)
        self.weight = float(weight)
        meta = getattr(hpcp_head, "metadata", {}) or {}
        self.loss_type = loss_type or meta.get("loss_type") or "cosine"
        self.huber_beta = float(meta.get("huber_beta") or 1.0)
        # latent fps (frames/sec) — used to index the chroma schedule by output time
        if self.inner is not None:
            self.fps = (self.inner.sample_rate
                        / self.inner.pretransform.downsampling_ratio)
        else:
            self.fps = self.schedule.fps
        self._t_emitted_sec = 0.0  # output time (frames emitted minus prefix) / fps

    def reset_time(self) -> None:
        """Reset the emitted-time cursor (call before a fresh render)."""
        self._t_emitted_sec = 0.0

    def _cond(self, prompt, prefix_latents, prefix_frames, n_frames, device, dtype, batch=1):
        """Inpaint cond build (copied from ``InpaintContinuationGenerator._cond``;
        we do NOT subclass it because that class hardcodes ``sample_diffusion``)."""
        import torch
        win_seconds = n_frames / self.fps
        conditioning, _ = self.model._build_conditioning_dicts(prompt, None, win_seconds, batch)
        ct = self.inner.conditioner(conditioning, device)
        C = self.inner.io_channels
        mask = torch.zeros((batch, 1, n_frames), device=device)
        masked_input = torch.zeros((batch, C, n_frames), device=device)
        if prefix_latents is not None and prefix_frames > 0:
            mask[:, :, :prefix_frames] = 1.0
            masked_input[:, :, :prefix_frames] = prefix_latents[..., :prefix_frames].to(device)
        ct["inpaint_mask"] = [mask]
        ct["inpaint_masked_input"] = [masked_input]
        ci = self.inner.get_conditioning_inputs(ct)
        return {k: (v.type(dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}, conditioning

    def generate(self, prompt: str, prefix_latents, prefix_frames: int,
                 n_frames: int, seed: int):
        """STUB-THIS-PASS runtime (needs the SA3 model + GPU, fp32).

        Steps (final design, matches the guided path in ``stable_audio_3/model.py``):
          1. cond_inputs, conditioning = self._cond(...)  (inpaint mask + masked_input).
          2. target = self.schedule.target(self._t_emitted_sec, n_frames, device)
             (already standardized to the head's std_mean/std_std).
          3. guides = [{head, target, weight, start_pct, end_pct, loss_type, huber_beta}].
          4. sigmas = build_schedule(steps=..., dist_shift=inner.sampling_dist_shift, ...).
          5. latents = sample_flow_euler_multi_latch_guided(
                 inner.model, noise, sigmas, guides,
                 rho=self.rho, mu=self.mu, gamma=self.gamma, n_iter=self.n_iter,
                 cfg_scale=self.cfg_scale, **cond_inputs)
          6. advance self._t_emitted_sec by (n_frames - prefix_frames)/fps; return latents.float().

        The body below is wired but exercised only on the GPU/model in a launch
        session; the pure logic (cond build, target build, guide dict, time advance)
        is implemented so the launch session only needs to validate rho/mu + VRAM.
        """
        import torch
        from stable_audio_3.inference.latch_guided import sample_flow_euler_multi_latch_guided
        from stable_audio_3.inference.sampling import build_schedule

        device = next(self.inner.model.parameters()).device
        dtype = next(self.inner.model.parameters()).dtype

        cond_inputs, conditioning = self._cond(
            prompt, prefix_latents, prefix_frames, n_frames, device, dtype)

        target = self.schedule.target(self._t_emitted_sec, n_frames,
                                      device=device, dtype=torch.float32)

        guides = [{
            "head": self.head,
            "target": target,
            "weight": self.weight,
            "start_pct": self.start_pct,
            "end_pct": self.end_pct,
            "loss_type": self.loss_type,
            "huber_beta": self.huber_beta,
        }]

        sigmas = build_schedule(
            steps=self.steps, sigma_max=1.0,
            dist_shift=self.inner.sampling_dist_shift,
            fallback_seq_len=n_frames, include_endpoint=True, device=str(device),
        )

        torch.manual_seed(seed)
        noise = torch.randn(1, self.inner.io_channels, n_frames, device=device, dtype=dtype)

        latents = sample_flow_euler_multi_latch_guided(
            self.inner.model, noise, sigmas, guides,
            rho=self.rho, mu=self.mu, gamma=self.gamma, n_iter=self.n_iter,
            cfg_scale=self.cfg_scale, **cond_inputs,
        )

        self._t_emitted_sec += (n_frames - prefix_frames) / self.fps
        return latents.float()


# Make ChromaGuidedGenerator a proper ChunkGenerator subclass when SA3 is present,
# without forcing the import for the pure ChromaSchedule unit tests.
try:  # pragma: no cover - import wiring, exercised only with stable_audio_3 installed
    from stable_audio_3.inference.longform import ChunkGenerator as _ChunkGenerator

    class ChromaGuidedGenerator(ChromaGuidedGenerator, _ChunkGenerator):  # type: ignore[no-redef]
        __doc__ = ChromaGuidedGenerator.__doc__
except Exception:  # stable_audio_3 not importable (e.g. pure CPU test venv)
    pass
