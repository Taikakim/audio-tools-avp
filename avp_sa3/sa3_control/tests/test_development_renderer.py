"""CPU/model-free unit tests for Part 1 continuation modes (development_renderer).

Stitching + seed/noise bookkeeping only — no GPU, no SA3 model. Uses
``stable_audio_3.inference.longform.FakeChunkGenerator`` (deterministic: emits the
call-index as a constant tensor, honours the prefix clamp when prefix_frames>0).
"""
import math

import pytest
import torch

from stable_audio_3.inference.longform import (
    LongFormRenderer, FakeChunkGenerator, CrossfadeStitcher, PromptSchedule,
)
from sa3_control.development_renderer import (
    DevelopmentRenderer, SDEditContinuationGenerator,
)

CHANNELS = 4
FPS = 10.0
WINDOW = 30      # frames
OVERLAP = 5
TOTAL = 100      # frames


# ---- recording fakes -------------------------------------------------------

class RecordingFake(FakeChunkGenerator):
    """FakeChunkGenerator that records the prefix_frames seen per generate()."""
    def __init__(self, channels):
        super().__init__(channels)
        self.prefix_seen = []
        self.seeds = []

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        self.prefix_seen.append(prefix_frames)
        self.seeds.append(seed)
        return super().generate(prompt, prefix_latents, prefix_frames, n_frames, seed)


class SetTotalWindowsFake(RecordingFake):
    """Recording fake that also exposes set_total_windows (sdedit contract)."""
    def __init__(self, channels):
        super().__init__(channels)
        self.total_windows_calls = []

    def set_total_windows(self, n):
        self.total_windows_calls.append(int(n))


def _sched(prompt="p"):
    return PromptSchedule(prompt)


# ---- clamp == base regression gate ----------------------------------------

def test_clamp_matches_base_output():
    base = LongFormRenderer(FakeChunkGenerator(CHANNELS), CHANNELS, FPS, WINDOW, OVERLAP)
    dev = DevelopmentRenderer(FakeChunkGenerator(CHANNELS), CHANNELS, FPS, WINDOW, OVERLAP,
                              continuation_mode="clamp")
    out_base = base.render_latents(_sched(), TOTAL, base_seed=7)
    out_dev = dev.render_latents(_sched(), TOTAL, base_seed=7)
    assert out_base.shape == out_dev.shape == (1, CHANNELS, TOTAL)
    assert torch.equal(out_base, out_dev)
    # drift_log has one entry per window in both
    assert len(base.drift_log) == len(dev.drift_log)


def test_clamp_prefix_is_overlap_after_first_window():
    gen = RecordingFake(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="clamp")
    dev.render_latents(_sched(), TOTAL)
    assert gen.prefix_seen[0] == 0
    assert all(p == OVERLAP for p in gen.prefix_seen[1:])


# ---- crossfade: fresh seed (prefix 0) every window + slerp join ------------

def test_crossfade_forces_prefix_zero_every_window():
    gen = RecordingFake(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="crossfade")
    out = dev.render_latents(_sched(), TOTAL)
    assert out.shape == (1, CHANNELS, TOTAL)
    assert torch.isfinite(out).all()
    assert all(p == 0 for p in gen.prefix_seen)


def test_crossfade_join_length_and_window_advance():
    # crossfade_overlap_frac=1.0 -> n_join == OVERLAP; each window advances WINDOW-OVERLAP
    gen = FakeChunkGenerator(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="crossfade",
                              crossfade_overlap_frac=1.0)
    out = dev.render_latents(_sched(), TOTAL)
    assert out.shape[-1] == TOTAL
    # number of windows generated == number of drift_log entries
    n_windows = len(dev.drift_log)
    # first window adds WINDOW, each subsequent adds (WINDOW - n_join)
    expected = math.ceil(max(0, TOTAL - WINDOW) / (WINDOW - OVERLAP)) + 1
    assert n_windows == expected


def test_crossfade_partial_overlap_frac():
    # frac=0.5 -> n_join = round(0.5*OVERLAP)=2 (>=1), windows advance more per step
    gen = FakeChunkGenerator(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="crossfade",
                              crossfade_overlap_frac=0.5)
    out = dev.render_latents(_sched(), TOTAL)
    assert out.shape[-1] == TOTAL
    assert torch.isfinite(out).all()


def test_crossfade_join_actually_slerps_boundary():
    # With two distinct-valued fake windows, the transition_join region must differ
    # from a hard cut (proves transition_join ran on the overlap, not a clamp).
    gen = FakeChunkGenerator(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="crossfade",
                              crossfade_overlap_frac=1.0)
    out = dev.render_latents(_sched(), TOTAL)
    # FakeChunkGenerator emits constant value == call index; a pure concat would
    # contain only integer plateaus. The slerp blends across the join, producing
    # at least one intermediate (non-integer) value.
    flat = out.flatten()
    has_blend = bool(((flat != flat.round()).any()).item())
    assert has_blend


# ---- sdedit: set_total_windows + init path ---------------------------------

def test_sdedit_calls_set_total_windows_and_init_path():
    gen = SetTotalWindowsFake(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode="sdedit")
    out = dev.render_latents(_sched(), TOTAL)
    assert out.shape == (1, CHANNELS, TOTAL)
    # set_total_windows called exactly once, with the renderer's estimate
    assert gen.total_windows_calls == [dev.estimate_total_windows(TOTAL)]
    # sdedit uses the same prefix decisions as clamp -> init path engaged after win0
    assert gen.prefix_seen[0] == 0
    assert all(p == OVERLAP for p in gen.prefix_seen[1:])


# ---- all modes: reach total, finite, drift_log length ----------------------

@pytest.mark.parametrize("mode", ["clamp", "sdedit", "crossfade"])
def test_all_modes_reach_total_and_finite(mode):
    gen = SetTotalWindowsFake(CHANNELS)
    dev = DevelopmentRenderer(gen, CHANNELS, FPS, WINDOW, OVERLAP, continuation_mode=mode)
    out = dev.render_latents(_sched(), TOTAL)
    assert out.shape == (1, CHANNELS, TOTAL)
    assert torch.isfinite(out).all()
    assert len(dev.drift_log) == len(gen.prefix_seen) >= 1


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        DevelopmentRenderer(FakeChunkGenerator(CHANNELS), CHANNELS, FPS, WINDOW, OVERLAP,
                            continuation_mode="bogus")


# ---- SDEditContinuationGenerator pure bookkeeping --------------------------

def test_init_noise_level_decays_start_to_end():
    g = SDEditContinuationGenerator(model=None, init_noise_start=0.85, init_noise_end=0.55,
                                    total_windows=5)
    assert g.init_noise_level(0) == pytest.approx(0.85)
    assert g.init_noise_level(4) == pytest.approx(0.55)
    mid = g.init_noise_level(2)
    assert 0.55 < mid < 0.85
    # monotone non-increasing
    levels = [g.init_noise_level(k) for k in range(5)]
    assert all(a >= b - 1e-9 for a, b in zip(levels, levels[1:]))


def test_init_noise_level_clamps_past_last_window():
    g = SDEditContinuationGenerator(model=None, init_noise_start=0.8, init_noise_end=0.5,
                                    total_windows=3)
    # k beyond total_windows-1 -> clamped at end (max(0, ...))
    assert g.init_noise_level(10) == pytest.approx(0.5)


def test_init_noise_level_handles_none_total_windows():
    g = SDEditContinuationGenerator(model=None, init_noise_start=0.8, init_noise_end=0.5)
    assert g.init_noise_level(0) == pytest.approx(0.8)
    # denom=1 -> window1 already at end
    assert g.init_noise_level(1) == pytest.approx(0.5)


def test_set_total_windows_updates_decay():
    g = SDEditContinuationGenerator(model=None, init_noise_start=0.9, init_noise_end=0.5)
    g.set_total_windows(11)
    assert g.total_windows == 11
    assert g.init_noise_level(5) == pytest.approx(0.5 + 0.4 * (1 - 5 / 10))


def test_build_init_data_tiles_short_prefix():
    g = SDEditContinuationGenerator(model=None)
    prefix = torch.arange(2 * CHANNELS, dtype=torch.float32).view(1, CHANNELS, 2)
    init = g._build_init_data(prefix, n_frames=7)
    assert init.shape == (1, CHANNELS, 7)
    # tiling repeats the prefix along time: column 0 == column 2 == column 4
    assert torch.equal(init[..., 0], init[..., 2])
    assert torch.equal(init[..., 2], init[..., 4])


def test_build_init_data_trims_long_prefix():
    g = SDEditContinuationGenerator(model=None)
    prefix = torch.randn(1, CHANNELS, 10)
    init = g._build_init_data(prefix, n_frames=4)
    assert init.shape == (1, CHANNELS, 4)
    assert torch.equal(init, prefix[..., :4])


# ---- SDEditContinuationGenerator.generate routing (injected fakes) ---------

class _FakeClamp:
    def __init__(self):
        self.calls = []
    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        self.calls.append((prefix_frames, n_frames, seed))
        return torch.zeros(1, CHANNELS, n_frames)


class _FakeReanchor:
    def __init__(self):
        self.calls = []
    def reanchor(self, latents, sigma_peak, prompt, seed):
        self.calls.append((latents.shape[-1], float(sigma_peak), seed))
        return torch.ones(1, CHANNELS, latents.shape[-1])


def test_generate_routes_window0_to_clamp_and_rest_to_reanchor():
    g = SDEditContinuationGenerator(model=None, init_noise_start=0.85, init_noise_end=0.55,
                                    total_windows=3)
    g.clamp = _FakeClamp()
    g.reanchor = _FakeReanchor()

    # window 0: no prefix -> fresh clamp
    out0 = g.generate("p", None, 0, WINDOW, seed=0)
    assert out0.shape == (1, CHANNELS, WINDOW)
    assert len(g.clamp.calls) == 1 and len(g.reanchor.calls) == 0
    assert g.sigma_log[0] == (0, None)

    # window 1: prefix present -> reanchor at init_noise_level(1)
    prefix = torch.randn(1, CHANNELS, OVERLAP)
    out1 = g.generate("p", prefix, OVERLAP, WINDOW, seed=1)
    assert out1.shape == (1, CHANNELS, WINDOW)
    assert len(g.reanchor.calls) == 1
    n_frames_seen, sigma_seen, seed_seen = g.reanchor.calls[0]
    assert n_frames_seen == WINDOW and seed_seen == 1
    assert sigma_seen == pytest.approx(g.init_noise_level(1))
    assert g.sigma_log[1] == (1, pytest.approx(g.init_noise_level(1)))
