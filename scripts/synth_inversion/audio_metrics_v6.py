# Owner: KUANG / KIM (created 2026-10-19)
"""Audio metrics suite for V6.1: extends audio_metrics_v5 with:
1. compute_spectral_shoulder: measures the filter knee / shoulder frequency in Hz, decoupling filter cutoff from narrow resonance peaks.
2. compute_harmonic_parity: measures even/odd harmonic energy ratio, identifying true square waves vs asymmetric saws.
3. guided_cutoff_shift: computes directed cutoff perturbation scalar based on spectral error.

KUANG 2026-10-19 (V6_REVIEW.md section 9): both spectral measures used a fixed FFT length on the raw signal,
rfft(y, n=8192) / rfft(y, n=4096), which keeps only the first 186 ms / 93 ms of the clip. On a real stem that is
whatever happens first (often silence or another note). compute_harmonic_parity now takes an explicit window and an
FFT that covers it; parity_over_notes measures each transcribed note at its own f0 and takes the median;
compute_spectral_shoulder averages the spectrum over the whole clip (Welch).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import audio_metrics_v5 as _v5
from audio_metrics_v5 import (  # noqa: F401 (re-exported)
    SR, MEL_KW, OCTAVE_EDGES_HZ, SM_BANDS_HZ, logmel, ssim_map, logmel_l1,
    mel_ssim_dist, mel_ssim_gated_dist, flatness_gated_err, score_many,
    METRICS, DEFAULT_EVAL,
)

def get_centroid(y, sr=SR) -> float:
    """Spectral centroid center-of-mass in Hz."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    sum_spec = float(spec.sum())
    if sum_spec <= 1e-9:
        return 0.0
    return float((freqs * spec).sum() / sum_spec)

def compute_spectral_shoulder(y, sr=SR, n_fft=4096, drop_db=8.0) -> float:
    """Compute the filter roll-off shoulder (knee frequency in Hz) invariant to narrow resonance peaks.
    Extracts the smoothed spectral envelope and identifies the inflection / transition point.
    The magnitude spectrum is averaged over the whole clip (Welch, n_fft frames), not taken from its first n_fft
    samples. Not validated on Surge (no ladder check yet); not used by the V6.1 scripts."""
    from scipy.signal import welch
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    freqs, psd = welch(y, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    log_mag = 10.0 * np.log10(psd + 1e-14)                       # power -> dB (same scale as 20 log10 |X|)
    
    # Restrict to audible bass/synth range 80 Hz to 14 kHz
    valid = (freqs >= 80) & (freqs <= 14000)
    f_sub = freqs[valid]
    m_sub = log_mag[valid]
    
    # Smooth with Gaussian envelope to remove harmonic comb ripples
    env = gaussian_filter1d(m_sub, sigma=18)
    
    # Reference passband energy: median of 100 Hz to 500 Hz
    pass_mask = (f_sub >= 100) & (f_sub <= 500)
    if np.any(pass_mask):
        ref_db = float(np.median(env[pass_mask]))
    else:
        ref_db = float(np.max(env))
        
    target_db = ref_db - drop_db
    
    # Find knee: first frequency past 150 Hz that permanently drops below target_db
    after_bass = np.where(f_sub >= 150)[0]
    below = np.where(env[after_bass] <= target_db)[0]
    if len(below) == 0:
        return float(f_sub[-1])
        
    k = after_bass[below[0]]
    if k == 0:
        return float(f_sub[0])
        
    f0, f1 = f_sub[k-1], f_sub[k]
    y0, y1 = env[k-1], env[k]
    if abs(y1 - y0) < 1e-6:
        return float(f0)
        
    f_knee = f0 + (target_db - y0) * (f1 - f0) / (y1 - y0)
    return float(np.clip(f_knee, 100.0, 14000.0))

def harmonic_powers(y, f0_hz, sr=SR, n_harmonics=10, start_s=0.0, dur_s=None):
    """Power around harmonics 1..n_harmonics (+-15 % of f0) over the window [start_s, start_s + dur_s]
    (KUANG 2026-10-20). -> array (length <= n_harmonics, stops at Nyquist), or None for an empty window."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    a = int(round(start_s * sr))
    b = len(y) if dur_s is None else min(len(y), a + int(round(dur_s * sr)))
    y = y[a:b]
    if len(y) == 0:
        return None
    n_fft = max(8192, 1 << int(np.ceil(np.log2(len(y)))))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bw = f0_hz * 0.15
    out = []
    for h in range(1, n_harmonics + 1):
        if h * f0_hz >= sr / 2.0:
            break
        mask = (freqs >= h * f0_hz - bw) & (freqs <= h * f0_hz + bw)
        out.append(float(np.sum(spec[mask] ** 2)) if np.any(mask) else 0.0)
    return np.array(out)


def harmonic_richness(y, f0_hz, sr=SR, **kw) -> float:
    """Power in harmonics 2..10 over the fundamental's. A filtered-down bass (nearly a sine) has parity ~0 like a
    square, because it has hardly any harmonics at all: parity only says "odd only" when this is >= ~0.05
    (a square's harmonics 3..9 hold 0.18 of the fundamental's power, a saw's 2..10 0.55)."""
    p = harmonic_powers(y, f0_hz, sr, **kw)
    if p is None or len(p) < 2:
        return float("nan")
    return float(p[1:].sum() / (p[0] + 1e-12))


def odd_harmonic_slope(y, f0_hz, sr=SR, **kw) -> float:
    """Slope of log power against log harmonic number over the odd harmonics 1, 3, 5, 7, 9. A square (or saw) has
    amplitudes 1/k -> slope ~-2; a triangle 1/k^2 -> ~-4. Parity alone cannot tell the two apart (both odd only)."""
    p = harmonic_powers(y, f0_hz, sr, **kw)
    if p is None or len(p) < 5:
        return float("nan")
    k = np.arange(1, len(p) + 1)[0::2]
    return float(np.polyfit(np.log(k), np.log(p[0::2] + 1e-12), 1)[0])


def compute_harmonic_parity(y, f0_hz, sr=SR, n_harmonics=10, start_s=0.0, dur_s=None) -> float:
    """Compute even-to-odd harmonic power ratio over the window [start_s, start_s + dur_s] (default: to the end).
    Mathematical reference:
      Pure square wave (odd harmonics only) -> ~0.00 to 0.05
      Pure saw wave (all harmonics 1/h)      -> ~0.29 (10 harmonics)
    Returns ratio: values < 0.12 strongly indicate square wave character.
    The window must hold ONE note at f0_hz: a note an octave up puts its odd harmonics on f0's even ones (the
    training phrase plays note, then note + 12, from 0.25 s: use dur_s=0.24 there). The FFT covers the window
    (at least 8192 points, zero padded), so nothing beyond n_fft is dropped silently."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    a = int(round(start_s * sr))
    b = len(y) if dur_s is None else min(len(y), a + int(round(dur_s * sr)))
    y = y[a:b]
    if len(y) == 0:
        return float("nan")
    n_fft = max(8192, 1 << int(np.ceil(np.log2(len(y)))))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    even_energy = 0.0
    odd_energy = 0.0
    bw = f0_hz * 0.15 # 15% window around each harmonic
    
    for h in range(1, n_harmonics + 1):
        target_f = h * f0_hz
        if target_f >= sr / 2.0:
            break
        mask = (freqs >= target_f - bw) & (freqs <= target_f + bw)
        if np.any(mask):
            e = float(np.sum(spec[mask]**2))
            if h % 2 == 0:
                even_energy += e
            else:
                odd_energy += e
                
    return float(even_energy / (odd_energy + 1e-7))

def guided_cutoff_shift(c_target_hz: float, c_pred_hz: float, max_shift: float = 0.25) -> float:
    """Compute normalized cutoff adjustment scalar in [-max_shift, max_shift]."""
    delta_hz = c_target_hz - c_pred_hz
    # 2500 Hz spans roughly the full audible lowpass sweep in Surge parameter space [0, 1]
    raw_shift = delta_hz / 2500.0
    return float(np.clip(raw_shift, -max_shift, max_shift))


MIN_RICHNESS = 0.05


def parity_over_notes(y, notes, sr=SR, max_dur_s=0.25, min_dur_s=0.06, min_richness=MIN_RICHNESS):
    """Median harmonic parity over transcribed notes ({"note", "start", "dur"}), each measured on its own window
    (up to max_dur_s, notes shorter than min_dur_s skipped) at its own f0. Notes with harmonic_richness below
    min_richness are skipped (nearly a sine: parity ~0 says nothing about the waveform; None keeps them).
    nan if no note qualifies."""
    vals = []
    for n in notes:
        d = min(float(n["dur"]), max_dur_s)
        if d < min_dur_s:
            continue
        f0 = 440.0 * 2.0 ** ((n["note"] - 69.0) / 12.0)
        kw = dict(start_s=float(n["start"]), dur_s=d)
        if min_richness is not None and not harmonic_richness(y, f0, sr, **kw) >= min_richness:
            continue
        v = compute_harmonic_parity(y, f0, sr, **kw)
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def phrase_parity(y, midi, sr=SR, min_richness=MIN_RICHNESS):
    """Parity of a training-phrase render (note at 0 s, note + 12 at 0.25 s): the first note only; nan when the
    note has too few harmonics to tell (harmonic_richness < min_richness)."""
    f0, kw = 440.0 * 2.0 ** ((midi - 69.0) / 12.0), dict(start_s=0.0, dur_s=0.24)
    if min_richness is not None and not harmonic_richness(y, f0, sr, **kw) >= min_richness:
        return float("nan")
    return compute_harmonic_parity(y, f0, sr, **kw)


# ------------------------------------------------------------------------------------------------------------------
# KUANG 2026-10-20: Kim's listening-test ideas (HUMAN_EVALUATION_SYNTHESIS.md, stems 17, 22, 24).
#  - band limit: separation leaves hiss above ~4 kHz that full-band losses reward filling (lowpass);
#  - band hierarchy sub > body > air: a lower band is matched first and may not be traded for a higher one
#    (band_dists, LexiScore, lexi_no_worse);
#  - shoulder position as one number (spectral_emd_oct), the |centroid ratio| generalised;
#  - resonance: a few harmonics standing above their neighbours vs a smooth harmonic envelope (resonance_excess_db).
BANDS_HZ = (("sub", 0.0, 120.0), ("body", 120.0, 2500.0), ("air", 2500.0, 4000.0))


def lowpass(y, sr=SR, hz=4000.0, order=8):
    """Zero-phase Butterworth lowpass (sosfiltfilt) along the last axis."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, hz / (sr / 2.0), output="sos")
    return sosfiltfilt(sos, np.asarray(y, np.float64), axis=-1).astype(np.float32)


def band_dists(pred, real, sr=SR, bands=BANDS_HZ, floor_db=-80.0):
    """Per band: mean |dB difference| of the STFT magnitudes, 1/f weighted inside the band (each octave counts the
    same, so the top of the body band does not outvote its bottom). Both signals are taken as they are (normalise
    them first); magnitudes are floored floor_db below the real's peak. Sub band n_fft 8192 (5.4 Hz bins at
    44.1 kHz), others 2048. -> {band name: dB}."""
    pred = np.asarray(pred, np.float64)
    real = np.asarray(real, np.float64)
    n = min(len(pred), len(real))
    pred, real = pred[:n], real[:n]
    out = {}
    for name, lo, hi in bands:
        n_fft = 8192 if hi <= 250.0 else 2048
        if n < n_fft:
            pred_, real_ = np.pad(pred, (0, n_fft - n)), np.pad(real, (0, n_fft - n))
        else:
            pred_, real_ = pred, real
        from scipy.signal import stft
        f, _, P = stft(pred_, fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
        _, _, R = stft(real_, fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
        sel = (f >= max(lo, 20.0)) & (f < hi)
        if not np.any(sel):
            out[name] = 0.0
            continue
        ref = 20.0 * np.log10(np.abs(R).max() + 1e-12)
        dp = np.maximum(20.0 * np.log10(np.abs(P[sel]) + 1e-12), ref + floor_db)
        dr = np.maximum(20.0 * np.log10(np.abs(R[sel]) + 1e-12), ref + floor_db)
        w = 1.0 / f[sel]
        out[name] = float((np.abs(dp - dr).mean(axis=1) * w).sum() / w.sum())
    return out


class LexiScore:
    """Band-hierarchical score: compared band by band in priority order (sub, body, air); a band decides only when
    the difference exceeds rel * max(a, b) + abs_db, otherwise the next band (and finally `total`, the composite
    score with the stereo / penalty terms) decides. Not transitive (tolerances), which is fine for the greedy
    argmin / hill climbing it is used in. float() is the total (for logs and the JSON)."""

    def __init__(self, bands, total, rel=0.05, abs_db=0.2):
        self.bands, self.total, self.rel, self.abs_db = tuple(float(b) for b in bands), float(total), rel, abs_db

    def _cmp(self, other):
        if not isinstance(other, LexiScore):
            o = float(other)
            return -1 if self.total < o else (1 if self.total > o else 0)
        for a, b in zip(self.bands, other.bands):
            tol = self.rel * max(a, b) + self.abs_db
            if a < b - tol:
                return -1
            if a > b + tol:
                return 1
        return -1 if self.total < other.total else (1 if self.total > other.total else 0)

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __float__(self):
        return self.total

    def __format__(self, spec):
        return format(self.total, spec)

    def __round__(self, nd=None):
        return round(self.total, nd)

    def __repr__(self):
        return f"LexiScore(bands={tuple(round(b, 3) for b in self.bands)}, total={self.total:.4f})"


def lexi_no_worse(new, old, rel=0.05, abs_db=0.2):
    """Pareto check with tolerance (Occam under the hierarchy): no band of `new` is worse than old's by more than
    rel * old + abs_db."""
    return all(a <= b * (1.0 + rel) + abs_db for a, b in zip(new.bands, old.bands))


def spectral_emd_oct(pred, real, sr=SR, fmin=30.0, fmax=4000.0, n_bins=240):
    """Earth mover's distance (Wasserstein-1) between the two long-term power spectra, on a log2 frequency axis
    from fmin to fmax: how many octaves of spectral mass have to move. 0 = same balance. Welch, n_fft 8192, on the
    magnitude (sqrt PSD; power is dominated by the fundamental on a bass and hardly sees the filter)."""
    from scipy.signal import welch
    grid = np.linspace(np.log2(fmin), np.log2(fmax), n_bins)
    cdfs = []
    for y in (pred, real):
        y = np.asarray(y, np.float64)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if len(y) < 8192:
            y = np.pad(y, (0, 8192 - len(y)))
        f, p = welch(y, fs=sr, nperseg=8192, noverlap=4096)
        keep = f > 0
        q = np.interp(grid, np.log2(f[keep]), np.sqrt(p[keep]))      # magnitude, as the centroid uses
        q = q / (q.sum() + 1e-20)
        cdfs.append(np.cumsum(q))
    return float(np.abs(cdfs[0] - cdfs[1]).sum() * (grid[1] - grid[0]))


def resonance_excess_db(y, f0_hz, sr=SR, fmax=4000.0, sigma_oct=0.5, start_s=0.0, dur_s=None):
    """Kim's resonance measure (stem 17): how far the loudest harmonic stands above the Gaussian average (sigma
    sigma_oct octaves, on log2 frequency, itself excluded) of the harmonics around it, in dB. A plain filtered saw
    or square has a smooth harmonic envelope (~0..3 dB); a resonant filter puts a few harmonics near the cutoff
    6..20 dB above it. Odd-only waveforms are measured over the odd harmonics (parity < 0.12), so the missing even
    harmonics are not read as dips. -> (excess dB, frequency Hz of that harmonic); (nan, nan) with < 4 harmonics."""
    n_h = int(fmax // f0_hz)
    p = harmonic_powers(y, f0_hz, sr, n_harmonics=max(n_h, 1), start_s=start_s, dur_s=dur_s)
    if p is None or len(p) < 4:
        return float("nan"), float("nan")
    k = np.arange(1, len(p) + 1)
    if len(p) >= 4 and p[1::2].sum() / (p[0::2].sum() + 1e-12) < 0.12:
        k, p = k[0::2], p[0::2]
    if len(p) < 4:
        return float("nan"), float("nan")
    if p[1:].sum() / (p[0] + 1e-12) < 0.01:                # (nearly) a sine: no harmonic envelope to stand above
        return float("nan"), float("nan")
    db = 10.0 * np.log10(p + 1e-20)
    db = np.maximum(db, db.max() - 60.0)
    lk = np.log2(k)
    best, at = -np.inf, float("nan")
    for i in range(1, len(p)):                             # the fundamental is not a resonance
        w = np.exp(-0.5 * ((lk - lk[i]) / sigma_oct) ** 2) / k     # 1/k: each octave counts once (k per octave)
        w[i] = 0.0
        if np.count_nonzero(w > 1e-3 * w.max()) < 3:
            continue
        # local weighted LINEAR fit of dB against log2 k (not a mean): a sloped envelope (-6 dB / octave) and the
        # one-sided neighbourhood near the fundamental do not read as a peak
        coef = np.polyfit(lk - lk[i], db, 1, w=np.sqrt(w))
        ex = db[i] - float(coef[1])
        if ex > best:
            best, at = ex, float(k[i] * f0_hz)
    return float(best), at


def resonance_over_notes(y, notes, sr=SR, max_dur_s=0.25, min_dur_s=0.06, **kw):
    """Median resonance_excess_db over transcribed notes (as parity_over_notes). -> (dB, Hz), nan if none."""
    vals = []
    for n in notes:
        d = min(float(n["dur"]), max_dur_s)
        if d < min_dur_s:
            continue
        f0 = 440.0 * 2.0 ** ((n["note"] - 69.0) / 12.0)
        e, f = resonance_excess_db(y, f0, sr, start_s=float(n["start"]), dur_s=d, **kw)
        if np.isfinite(e):
            vals.append((e, f))
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[len(vals) // 2]


class DecayMultiScaleSTFTLoss:
    """audio_utils.MultiScaleSTFTLoss with every bin weighted by w(f) = (1 + f / knee)^-gamma (V6_1_PLAN.md 4.3,
    Antigravity 2026-10-20), in both the spectral convergence and the log-magnitude L1. gamma = 0 gives the plain
    loss back. Same FFT sizes; numbers are not comparable with the unweighted loss (report which one was used).
    Note: at n_fft 256 the first bin is 172 Hz wide, so the small resolutions cannot see the sub band at all; the
    weighting changes how much the air counts, it does not add sub-bass resolution (band_dists uses n_fft 8192)."""

    FFT_SIZES = (2048, 1024, 512, 256)

    def __init__(self, knee_hz=900.0, gamma=1.25, sr=SR):
        self.knee_hz, self.gamma, self.sr = float(knee_hz), float(gamma), sr

    def weights(self, n_fft):
        f = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        return (1.0 + f / self.knee_hz) ** (-self.gamma)

    def __call__(self, x, y) -> float:
        import torch
        x = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)
        y = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(1, -1)
        n = min(x.shape[1], y.shape[1])
        if n <= max(self.FFT_SIZES) // 2:
            raise ValueError(f"need > {max(self.FFT_SIZES) // 2} samples to score, got {n}")
        x, y = x[:, :n], y[:, :n]
        total = 0.0
        for n_fft in self.FFT_SIZES:
            window = torch.hann_window(n_fft)
            X = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs() + 1e-7
            Y = torch.stft(y, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs() + 1e-7
            w = torch.as_tensor(self.weights(n_fft), dtype=torch.float32).reshape(1, -1, 1)
            sc = torch.norm(w * (Y - X), p="fro") / (torch.norm(w * Y, p="fro") + 1e-7)
            log_l1 = torch.sum(w * torch.abs(torch.log(Y) - torch.log(X))) / (torch.sum(w.expand_as(X)) + 1e-12)
            total += float(sc + log_l1)
        return total / len(self.FFT_SIZES)


def spectral_emd_linear(pred, real, sr=SR, fmax=None):
    """The V6_1_PLAN.md 4.2 form as written: CDF of POWER over linear-frequency bins (rfft of the whole clip), L1 of
    the CDF difference integrated over Hz, divided by fmax (so 0..1). Kept to reproduce Antigravity's stem 24 numbers;
    spectral_emd_oct (magnitude, log2 axis) is the one stored by the eval, because a bass's power is nearly all in
    the fundamental and the power CDF barely moves with the filter (55 Hz saw, 2-pole lowpass 800 Hz vs none:
    0.011 octaves on power, 0.098 on magnitude)."""
    p = [np.abs(np.fft.rfft(np.asarray(v, np.float64))) ** 2 for v in (pred, real)]
    n = min(len(p[0]), len(p[1]))
    f = np.fft.rfftfreq(2 * (n - 1), 1.0 / sr)[:n]
    keep = f <= (fmax or sr / 2)
    c = [np.cumsum(q[:n][keep]) / (q[:n][keep].sum() + 1e-20) for q in p]
    df = f[1] - f[0]
    return float(np.abs(c[0] - c[1]).sum() * df / (f[keep][-1] + 1e-12))
