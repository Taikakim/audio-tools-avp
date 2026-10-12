# Owner: KUANG (created 2026-10-11)
"""Objective "buzz" measures for sync, FM and detune (research Q9; review B9, B15).

Brief section 5 item 3 ("zero unmodulated sync/FM buzz") is a listening criterion. These are numpy measures that
give it a number, so it can be tracked per clip and over time.

  comb_residual            share of power in [lo, hi] that is NOT within +-half_width of a harmonic k * f0
  comb_residual_timeline   the same per frame: separates STATIC buzz (constant) from ENVELOPED buzz (decays)
  harmonic_wobble_db       detrended frame-to-frame variation of the first K harmonic amplitudes

Relationship to the aux inharmonicity extractor: `inharmonicity_target.py` already reports an
`excluded_energy_fraction` per window (coverage_diagnostics). That is the same idea computed from Essentia peaks.
Use this numpy version where Essentia is unavailable (laptop, CI) and to cross-check the extractor. The two should
agree in rank; they are not numerically identical (peak picking against a band mask).

Limits (documented, tested where measurable): intentional detune, unison, chorus, noise oscillators and room tails
also add off-comb power. The measure says "off-comb energy", not "wrong". Low notes need a long FFT: the half-width
is max(band_rel * f0, 2 bins) and `resolution_ok` is False if that exceeds 45% of f0 (the comb would cover everything).
"""
import numpy as np


def _adaptive_n_fft(sr: int, f0: float, max_len: int, min_fft: int = 1024, max_fft: int = 16384) -> int:
    n = 1 << int(np.ceil(np.log2(max(sr * 6.0 / f0, min_fft))))
    n = int(min(max(n, 4096), max_fft))
    while n > max_len and n > min_fft:
        n //= 2
    return n


def comb_residual(x, sr: int, f0: float, t_start: float = 0.2, t_end=None, lo_hz: float = 100.0, hi_hz: float = 6000.0,
                  band_rel: float = 0.15) -> dict:
    x = np.asarray(x, dtype=np.float64)
    seg = x[int(round(t_start * sr)): (len(x) if t_end is None else int(round(t_end * sr)))]
    n_fft = _adaptive_n_fft(sr, f0, len(seg))
    hop = max(1, n_fft // 4)
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    n_frames = 1 + (len(seg) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    p = (np.abs(np.fft.rfft(seg[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    df = f[1] - f[0]
    half = max(band_rel * f0, 2.0 * df)
    k = np.rint(f / f0)
    on_comb = (k >= 1) & (np.abs(f - k * f0) <= half)
    band = (f >= lo_hz) & (f <= hi_hz)
    tot = p[band].sum()
    res = float(p[band & ~on_comb].sum() / tot) if tot > 0 else float("nan")
    return {"residual": res, "n_fft": n_fft, "half_width_hz": float(half), "resolution_ok": bool(half < 0.45 * f0)}


def comb_residual_timeline(x, sr: int, f0: float, frame_s: float = 0.2, hop_s: float = 0.1, **kw):
    """Returns (times_s [frame centres], residual [per frame]). frame_s must hold the adaptive FFT for f0."""
    x = np.asarray(x, dtype=np.float64)
    fl, hl = int(round(frame_s * sr)), int(round(hop_s * sr))
    times, vals = [], []
    for s in range(0, max(1, len(x) - fl + 1), hl):
        r = comb_residual(x[s:s + fl], sr, f0, t_start=0.0, **kw)
        times.append((s + fl / 2) / sr)
        vals.append(r["residual"])
    return np.asarray(times), np.asarray(vals)


def harmonic_wobble_db(x, sr: int, f0: float, n_harm: int = 8, frame_s: float = 0.05, hop_s: float = 0.025,
                       t_start: float = 0.05) -> float:
    """Median over harmonics of the standard deviation (dB) of the detrended harmonic amplitude over time."""
    x = np.asarray(x, dtype=np.float64)[int(round(t_start * sr)):]
    n_fft = 2048
    fl, hl = int(round(frame_s * sr)), int(round(hop_s * sr))
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    df = f[1] - f[0]
    rows = []
    for s in range(0, max(1, len(x) - fl + 1), hl):
        seg = x[s:s + fl]
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n_fft))
        amps = []
        for k in range(1, n_harm + 1):
            half = max(0.2 * f0, df)
            i0, i1 = int(np.floor((k * f0 - half) / df)), int(np.ceil((k * f0 + half) / df))
            amps.append(20 * np.log10(sp[max(i0, 0):i1 + 1].max() + 1e-12))
        rows.append(amps)
    a = np.asarray(rows)
    if len(a) < 5:
        return float("nan")
    t = np.arange(len(a))
    stds = []
    for j in range(a.shape[1]):
        c = np.polyfit(t, a[:, j], 2)
        stds.append(np.std(a[:, j] - np.polyval(c, t)))
    return float(np.median(stds))
