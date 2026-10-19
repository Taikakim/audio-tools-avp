# Owner: KUANG / KIM (created 2026-10-19)
"""Audio metrics suite for V6.1: extends audio_metrics_v5 with:
1. compute_spectral_shoulder: measures the filter knee / shoulder frequency in Hz, decoupling filter cutoff from narrow resonance peaks.
2. compute_harmonic_parity: measures even/odd harmonic energy ratio, identifying true square waves vs asymmetric saws.
3. guided_cutoff_shift: computes directed cutoff perturbation scalar based on spectral error.
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
    Extracts the smoothed spectral envelope and identifies the inflection / transition point."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    
    spec = np.abs(np.fft.rfft(y, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    log_mag = 20.0 * np.log10(spec + 1e-7)
    
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

def compute_harmonic_parity(y, f0_hz, sr=SR, n_harmonics=10) -> float:
    """Compute even-to-odd harmonic power ratio.
    Mathematical reference:
      Pure square wave (odd harmonics only) -> ~0.00 to 0.05
      Pure saw wave (all harmonics 1/h)      -> ~0.33
    Returns ratio: values < 0.12 strongly indicate square wave character."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
        
    n_fft = 8192
    spec = np.abs(np.fft.rfft(y, n=n_fft))
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
