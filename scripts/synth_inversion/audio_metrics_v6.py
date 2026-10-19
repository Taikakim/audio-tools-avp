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


def parity_over_notes(y, notes, sr=SR, max_dur_s=0.25, min_dur_s=0.06):
    """Median harmonic parity over transcribed notes ({"note", "start", "dur"}), each measured on its own window
    (up to max_dur_s, notes shorter than min_dur_s skipped) at its own f0. nan if no note qualifies."""
    vals = []
    for n in notes:
        d = min(float(n["dur"]), max_dur_s)
        if d < min_dur_s:
            continue
        f0 = 440.0 * 2.0 ** ((n["note"] - 69.0) / 12.0)
        v = compute_harmonic_parity(y, f0, sr, start_s=float(n["start"]), dur_s=d)
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def phrase_parity(y, midi, sr=SR):
    """Parity of a training-phrase render (note at 0 s, note + 12 at 0.25 s): the first note only."""
    return compute_harmonic_parity(y, 440.0 * 2.0 ** ((midi - 69.0) / 12.0), sr, start_s=0.0, dur_s=0.24)
