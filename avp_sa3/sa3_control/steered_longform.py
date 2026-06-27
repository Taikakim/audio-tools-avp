"""Steered long-form generation: wrap the longform ChunkGenerator with a per-window
onset-density control schedule. Single-prompt sweeps only (no prompt transitions)."""
import torch
from sa3_control.adapters import ControlContext, use_control_context


class SteeredGenerator:
    """Tracks output time from its own frame accounting and applies the scheduled
    onset scalar via use_control_context around each window's inner generate."""

    def __init__(self, inner, schedule, encoder, mean, std, gain, fps, cfg_scale, device, dtype):
        self.inner = inner
        self.sched = schedule
        self.enc = encoder
        self.mean = float(mean); self.std = float(std); self.gain = float(gain)
        self.fps = float(fps); self.cfg = float(cfg_scale)
        self.device = device; self.dtype = dtype
        self._frames_before = 0
        self.applied = []  # (t_sec, raw_density) per window

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        t = self._frames_before / self.fps
        raw = self.sched.resolve(t)
        self.applied.append((t, raw))
        s = torch.tensor([(raw - self.mean) / self.std], device=self.device, dtype=self.dtype)
        ctrl = self.enc(s)
        cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if self.cfg != 1.0 else ctrl
        with use_control_context(ControlContext(cc, gain=self.gain)):
            out = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
        self._frames_before += (n_frames - prefix_frames)
        return out
