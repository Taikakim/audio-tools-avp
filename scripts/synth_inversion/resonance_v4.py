"""Resonance, centre frequency and bandwidth from a harmonic tone (research Q2; review B12).

Problem: a bass note has harmonics every f0 = 41 to 147 Hz. A filter resonance narrower than that spacing is
SAMPLED by one or two harmonics: its height and centre are estimable, its width (Q) is not. v4 must not train
on a "true Q" label that the audio cannot determine, so the card reports `q_censored` and a lower bound.

Method (numpy only):
  1. average power spectrum over a sustain segment (4096-point Hann by default);
  2. harmonic amplitudes: the peak magnitude within +-max(rel_tol * f0, one bin) of k * f0, k = 1..K;
  3. a smooth baseline fitted to log-amplitude versus LOG harmonic index (so the steep first harmonics of a
     sawtooth do not look like a peak), by asymmetric least squares (Whittaker smoother whose weights push the
     baseline under the peaks), with a stiffness given in octaves of harmonic index;
  4. height = the largest excess over the baseline; centre = that harmonic's frequency;
  5. width = full width at HALF HEIGHT IN dB (not the -3 dB bandwidth of a filter) IF at least `min_fwhm_points`
     harmonics lie inside it; otherwise censored, with width_upper_hz = (n_elevated + 1) * f0 and
     q_like_lower_bound = centre / that.

Calibration (measured on the stand-in, test_v4_components.py): height, centre and the ORDER of width follow the filter's
Q; width_hz and q_like are NOT calibrated to the filter's Q (the half-height-in-dB width of a two-pole low-pass is
about 1.2 to 5 times its fc / Q). Use them for ordering and for the censoring flag, not as a Q label.

`brief_prominence_db` reimplements the rolling-median prominence of DESIGN_BRIEF_V4.md section 2.3 for comparison
only (see test_v4_components.py::test_brief_prominence_measures_the_comb).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _avg_power_spectrum(seg: np.ndarray, sr: int, n_fft: int, hop: int):
    seg = np.asarray(seg, dtype=np.float64)
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    n_frames = 1 + (len(seg) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    p = np.abs(np.fft.rfft(seg[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2
    return np.fft.rfftfreq(n_fft, 1.0 / sr), p.mean(axis=0)


def harmonic_amplitudes(x, sr: int, f0: float, t_start: float = 0.2, t_end: Optional[float] = None,
                        n_fft: int = 4096, rel_tol: float = 0.15, fmax: float = 8000.0, floor_db_below_peak: float = 70.0):
    """Returns (freqs_hz [K], amp_db [K], valid [K]). Harmonics whose amplitude is within `floor_db_below_peak` of the
    strongest are valid. `t_start` skips the attack."""
    x = np.asarray(x, dtype=np.float64)
    i0 = int(round(t_start * sr))
    i1 = len(x) if t_end is None else int(round(t_end * sr))
    seg = x[i0:i1]
    f, p = _avg_power_spectrum(seg, sr, n_fft, max(1, n_fft // 4))
    df = f[1] - f[0]
    k_max = int(np.floor(min(fmax, 0.45 * sr) / f0))
    freqs, amps = [], []
    for k in range(1, k_max + 1):
        half = max(rel_tol * f0, df)
        lo, hi = int(np.floor((k * f0 - half) / df)), int(np.ceil((k * f0 + half) / df))
        lo, hi = max(lo, 0), min(hi, len(p) - 1)
        freqs.append(k * f0)
        amps.append(np.sqrt(p[lo:hi + 1].max()))
    amp_db = 20.0 * np.log10(np.asarray(amps) + 1e-12)
    valid = amp_db >= amp_db.max() - floor_db_below_peak
    return np.asarray(freqs), amp_db, valid


def asls_baseline(y: np.ndarray, lam: float, p: float = 0.01, n_iter: int = 30) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers and Boelens): a second-difference-penalized smoother whose weights are
    p above the fit and 1 - p below, so the baseline stays under the peaks."""
    n = len(y)
    d = np.diff(np.eye(n), 2, axis=0)
    h = lam * (d.T @ d)
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        z = np.linalg.solve(np.diag(w) + h, w * y)
        w_new = p * (y > z) + (1.0 - p) * (y <= z)
        if np.array_equal(w_new, w):
            break
        w = w_new
    return z


def log_index_baseline(amp_db: np.ndarray, valid: np.ndarray, smooth_octaves: float = 1.5, p: float = 0.01) -> np.ndarray:
    """Baseline of `amp_db` over harmonic index k = 1..K, fitted on a uniform LOG-k grid and mapped back. Invalid
    harmonics are ignored. `smooth_octaves` is the approximate period (in octaves of harmonic index) below which
    structure is treated as a peak rather than as roll-off."""
    k = np.arange(1, len(amp_db) + 1, dtype=np.float64)
    kv, yv = k[valid], amp_db[valid]
    if len(kv) < 5:
        return np.full(len(amp_db), float(np.max(amp_db)))
    n_grid = int(max(64, 32 * np.ceil(np.log2(max(kv[-1], 2.0)))))
    grid = np.linspace(np.log2(kv[0]), np.log2(kv[-1]), n_grid)
    y_grid = np.interp(grid, np.log2(kv), yv)
    step = grid[1] - grid[0]
    lam = (smooth_octaves / step / (2.0 * np.pi)) ** 4
    z = asls_baseline(y_grid, lam, p=p)
    return np.interp(np.log2(k), grid, z)


@dataclass
class ResonanceCard:
    f0_hz: float
    n_harmonics: int
    has_peak: bool
    peak_freq_hz: Optional[float]
    peak_harmonic: Optional[int]
    height_db: float
    width_hz: Optional[float]          # None when censored
    q_like: Optional[float]                 # None when censored
    q_censored: bool
    n_elevated: int                        # harmonics at or above half height
    width_upper_hz: Optional[float]    # set when censored
    q_like_lower_bound: Optional[float]         # set when censored
    freqs_hz: np.ndarray = field(repr=False, default=None)
    amp_db: np.ndarray = field(repr=False, default=None)
    baseline_db: np.ndarray = field(repr=False, default=None)


def resonance_card_from_amplitudes(freqs_hz, amp_db, f0: float, valid=None, smooth_octaves: float = 1.5,
                                   min_height_db: float = 3.0, min_fwhm_points: int = 3) -> ResonanceCard:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    amp_db = np.asarray(amp_db, dtype=np.float64)
    valid = np.ones(len(amp_db), bool) if valid is None else np.asarray(valid, bool)
    base = log_index_baseline(amp_db, valid, smooth_octaves)
    resid = np.where(valid, amp_db - base, -np.inf)
    ip = int(np.argmax(resid))
    height = float(resid[ip])
    if not np.isfinite(height) or height < min_height_db:
        return ResonanceCard(f0, len(amp_db), False, None, None, max(height, 0.0) if np.isfinite(height) else 0.0, None, None,
                             False, 0, None, None, freqs_hz, amp_db, base)
    half = height / 2.0
    # contiguous run of harmonics at or above half height around the peak
    lo = ip
    while lo - 1 >= 0 and valid[lo - 1] and resid[lo - 1] >= half:
        lo -= 1
    hi = ip
    while hi + 1 < len(resid) and valid[hi + 1] and resid[hi + 1] >= half:
        hi += 1
    n_el = hi - lo + 1
    at_edge = (lo == 0) or (hi == len(resid) - 1) or (lo - 1 >= 0 and not valid[lo - 1]) or (hi + 1 < len(resid) and not valid[hi + 1])
    centre = float(freqs_hz[ip])
    if n_el >= min_fwhm_points and not at_edge:
        # interpolate the half-height crossings between harmonics
        def cross(i_in, i_out):
            r_in, r_out = resid[i_in], resid[i_out]
            fr = (r_in - half) / max(r_in - r_out, 1e-12)
            return freqs_hz[i_in] + fr * (freqs_hz[i_out] - freqs_hz[i_in])
        f_lo = cross(lo, lo - 1)
        f_hi = cross(hi, hi + 1)
        bw = float(f_hi - f_lo)
        return ResonanceCard(f0, len(amp_db), True, centre, ip + 1, height, bw, centre / bw if bw > 0 else None, False, n_el,
                             None, None, freqs_hz, amp_db, base)
    upper = (n_el + 1) * f0
    return ResonanceCard(f0, len(amp_db), True, centre, ip + 1, height, None, None, True, n_el, float(upper),
                         float(centre / upper), freqs_hz, amp_db, base)


def resonance_card(x, sr: int, f0: float, t_start: float = 0.2, t_end: Optional[float] = None, n_fft: int = 4096,
                   smooth_octaves: float = 1.5, min_height_db: float = 3.0) -> ResonanceCard:
    freqs, amp_db, valid = harmonic_amplitudes(x, sr, f0, t_start=t_start, t_end=t_end, n_fft=n_fft)
    return resonance_card_from_amplitudes(freqs, amp_db, f0, valid, smooth_octaves, min_height_db)


def brief_prominence_db(x, sr: int, t_start: float = 0.2, n_fft: int = 4096, window_hz: float = 500.0,
                        floor_hz: float = 800.0) -> float:
    """DESIGN_BRIEF_V4.md section 2.3, reimplemented for comparison: log spectrum, rolling median over `window_hz`,
    prominence = max(0, S - B) for f > floor_hz; returns the largest prominence in dB."""
    f, p = _avg_power_spectrum(np.asarray(x, dtype=np.float64)[int(round(t_start * sr)):], sr, n_fft, n_fft // 4)
    s = 10.0 * np.log10(p + 1e-20)
    df = f[1] - f[0]
    half = max(1, int(round(window_hz / 2.0 / df)))
    pad = np.pad(s, (half, half), mode="edge")
    med = np.array([np.median(pad[i:i + 2 * half + 1]) for i in range(len(s))])
    prom = np.maximum(0.0, s - med)
    return float(prom[f > floor_hz].max())
