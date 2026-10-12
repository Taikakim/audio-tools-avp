# Owner: KUANG (created 2026-10-11)
"""Bounded onset snapping: align a MIDI note onset with the true onset in a rendered clip or a stem.

Why (review B13, research Q3): attack descriptors and two-anchor windows are measured from the onset. MIDI onsets
from transcription are off by tens of milliseconds, which is more than a 40 ms attack window. For a single triggered
note a bounded search is cheaper and less able to "explain away" a wrong envelope than DTW.

Method (numpy only):
  1. a 1 ms hop RMS envelope with a window that ENDS at the frame time, so the envelope rises at the true onset;
  2. a spectral-flux onset strength (log-magnitude, positive differences, frequency max-filter as in SuperFlux);
  3. search only [expected - search_s, expected + search_s]: take the strongest flux peak p;
  4. take the local floor before p and the local peak after p; the onset is the last time before p at which the
     envelope is still below floor + rise_frac * (peak - floor) (the "10% point" of the rise);
  5. if there is no real rise in the window (a note that starts at sample 0, or a window with no event), return the
     expected onset with confidence 0. The caller decides what to do with low confidence.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class OnsetResult:
    onset_s: float
    offset_s: float          # onset_s - expected_s (the correction applied)
    confidence: float        # 0 = no usable rise found, 1 = a full-scale rise


def rms_envelope(x: np.ndarray, sr: int, hop_s: float = 0.001, win_s: float = 0.004):
    """RMS envelope. Each value uses the window that ENDS at its time stamp. Returns (rms, times_s)."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(round(hop_s * sr)))
    win = max(hop, int(round(win_s * sr)))
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    ends = np.arange(win, len(x) + 1, hop)
    if len(ends) == 0:
        ends = np.array([len(x)])
        win = len(x)
    starts = ends - win
    rms = np.sqrt(np.maximum(c[ends] - c[starts], 0.0) / win)
    return rms, ends / sr


def spectral_flux(x: np.ndarray, sr: int, n_fft: int = 512, hop_s: float = 0.002, max_filter_bins: int = 1,
                  gamma: float = 100.0):
    """Positive spectral flux of log-compressed magnitude with a frequency max-filter on the previous frame.
    Returns (flux, times_s) with the time stamp at the END of each analysis window."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(round(hop_s * sr)))
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * win[None, :], axis=1))
    lm = np.log1p(gamma * mag / n_fft)
    prev = lm[:-1]
    if max_filter_bins > 0:
        p = np.pad(prev, ((0, 0), (max_filter_bins, max_filter_bins)), mode="edge")
        prev = np.max(np.stack([p[:, i:i + prev.shape[1]] for i in range(2 * max_filter_bins + 1)]), axis=0)
    flux = np.concatenate([[0.0], np.maximum(lm[1:] - prev, 0.0).sum(axis=1)])
    times = (hop * np.arange(n_frames) + n_fft) / sr
    return flux, times


def snap_onset(x: np.ndarray, sr: int, expected_s: float, search_s: float = 0.05, rise_frac: float = 0.1,
               floor_back_s: float = 0.08, peak_fwd_s: float = 0.06, min_rise: float = 0.1,
               persist_s: float = 0.002) -> OnsetResult:
    """Snap `expected_s` to the true onset inside +-search_s. See the module docstring.

    persist_s: the envelope must stay above the rise threshold for this long (rejects noise spikes).
    A clip that already sounds at its first frame (a note that starts at sample 0, as in the training renders)
    has no rise to find: the result is `expected_s` with confidence 0.

    Known limit (measured on the stand-in synth, tests/probe): the 10% point of a slow attack on a low note is
    ambiguous by up to about one waveform period (a few ms to about 14 ms for a 40 ms attack at 41 Hz). Two clips
    of the SAME note, aligned the same way, share that bias, so relative alignment stays consistent.
    """
    env, t_env = rms_envelope(x, sr)
    flux, t_flux = spectral_flux(x, sr)
    lo, hi = max(expected_s - search_s, 0.0), expected_s + search_s
    m = (t_flux >= lo) & (t_flux <= hi)
    if not m.any():
        return OnsetResult(expected_s, 0.0, 0.0)
    sel = np.where(m)[0]
    p = float(t_flux[sel[np.argmax(flux[sel])]])
    # the flux frame is stamped at its window end; the rise that caused it is within the analysis window before p
    i_after = np.where((t_env >= p) & (t_env <= p + peak_fwd_s))[0]
    i_before = np.where((t_env >= max(p - floor_back_s, 0.0)) & (t_env <= p))[0]
    if len(i_after) == 0 or len(i_before) == 0:
        return OnsetResult(expected_s, 0.0, 0.0)
    peak = float(env[i_after].max())
    if i_before[0] == 0 and env[0] > rise_frac * peak:        # already sounding at the first frame: no rise in this clip
        return OnsetResult(expected_s, 0.0, 0.0)
    i_floor = int(i_before[np.argmin(env[i_before])])        # earliest minimum of the local floor
    floor = float(env[i_floor])
    if peak <= 0.0 or (peak - floor) < min_rise * peak:
        return OnsetResult(expected_s, 0.0, 0.0)
    thr = floor + rise_frac * (peak - floor)
    # the rise point is the first envelope sample above the threshold after the floor (NOT the flux peak: for a slow
    # attack the flux peak comes early in the rise, and its window is stamped at its end)
    hop_s = float(t_env[1] - t_env[0]) if len(t_env) > 1 else 0.001
    n_persist = max(1, int(round(persist_s / hop_s)))
    above = env[i_floor:] > thr
    onset = float(t_env[i_floor])
    for j in range(len(above)):
        if above[j:j + n_persist].all() and len(above[j:j + n_persist]) == n_persist:
            onset = float(t_env[i_floor + j])
            break
    return OnsetResult(onset, onset - expected_s, float(np.clip((peak - floor) / peak, 0.0, 1.0)))
