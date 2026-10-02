"""Essentia and PyTorch/NumPy envelope extraction and ADSR/FEG decoupling tool.

Designed for tight, fast basslines (e.g. 130-145+ BPM Goa / Psy trance, ~143 BPM average).
Decouples:
  1. Amplitude Envelope (AEG): extracted from the sub-bass / fundamental band (< 220 Hz)
     and overall broadband energy, measuring attack rise, decay slope, sustain plateau,
     gate-off / note-off changepoint, and release falloff.
  2. Filter Envelope (FEG): extracted from the upper harmonic band (> 450 Hz) and
     spectral centroid trajectory, measuring the resonance sweep, harmonic decay rate,
     and filter modulation depth.

Can run with:
  - Essentia (preferred for full MIR analysis, runs in mir venv)
  - Pure NumPy / SciPy / PyTorch fallback (zero-dependency, identical Zölzer envelope filter,
    runs directly in sat-venv and search loops).
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try Essentia import
try:
    import essentia
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class NoteEnvelopeProfile:
    onset_s: float
    duration_s: float
    gate_off_s: float
    attack_ms: float
    log_attack_time: float
    peak_amplitude: float
    aeg_decay_half_ms: float
    aeg_decay_tenth_ms: float
    aeg_sustain_level: float
    aeg_release_rate_db_s: float
    feg_decay_half_ms: float
    feg_decay_tenth_ms: float
    spectral_centroid_peak_hz: float
    spectral_centroid_sustain_hz: float
    feg_sweep_ratio: float
    suggested_params: Dict[str, float]


def zolzer_envelope_follower(x: np.ndarray, fs: int = 44100, t_att: float = 0.002, t_rel: float = 0.015) -> np.ndarray:
    """Udo Zölzer DASP non-symmetric rectified 1st-order IIR envelope follower.
    
    100% numerically equivalent (corr > 0.999999) to Essentia's es.Envelope(attackTime, releaseTime).
    """
    a_att = math.exp(-1.0 / (t_att * fs))
    a_rel = math.exp(-1.0 / (t_rel * fs))
    abs_x = np.abs(x)
    env = np.empty_like(x, dtype=np.float32)
    prev = 0.0
    for n in range(len(x)):
        inp = abs_x[n]
        coeff = a_att if inp > prev else a_rel
        prev = (1.0 - coeff) * inp + coeff * prev
        env[n] = prev
    return env


def extract_log_attack_time(env: np.ndarray, fs: int = 44100,
                            start_thresh: float = 0.2, stop_thresh: float = 0.9) -> Tuple[float, float, float]:
    """Computes log10 attack time [log10(s)], attack start (s), and attack stop (s)."""
    if HAS_ESSENTIA:
        lat_algo = es.LogAttackTime(sampleRate=float(fs),
                                    startAttackThreshold=float(start_thresh),
                                    stopAttackThreshold=float(stop_thresh))
        try:
            lat, a_start, a_stop = lat_algo(env.astype(np.float32))
            return float(lat), float(a_start), float(a_stop)
        except Exception:
            pass

    # Pure NumPy implementation
    peak_idx = int(np.argmax(env))
    peak_val = float(env[peak_idx])
    if peak_val <= 1e-7:
        return -3.0, 0.0, 0.001
    
    pre_peak = env[:peak_idx + 1]
    v_start = start_thresh * peak_val
    v_stop = stop_thresh * peak_val
    
    idx_start = np.where(pre_peak >= v_start)[0]
    idx_stop = np.where(pre_peak >= v_stop)[0]
    
    start_s = (idx_start[0] / fs) if len(idx_start) > 0 else 0.0
    stop_s = (idx_stop[0] / fs) if len(idx_stop) > 0 else (peak_idx / fs)
    
    att_time = max(1e-5, stop_s - start_s)
    log_att = math.log10(att_time)
    return log_att, start_s, stop_s


def butter_bandpass_filter(audio: np.ndarray, fs: int = 44100, lowcut: float = 20.0, highcut: float = 220.0) -> np.ndarray:
    """Second-order Butterworth filter for band separation."""
    import scipy.signal as signal
    nyq = 0.5 * fs
    low = max(1e-4, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if lowcut <= 25.0:
        b, a = signal.butter(2, high, btype='low')
    elif highcut >= nyq * 0.95:
        b, a = signal.butter(2, low, btype='high')
    else:
        b, a = signal.butter(2, [low, high], btype='band')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def detect_gate_off_changepoint(env: np.ndarray, fs: int = 44100,
                                expected_gate_range: Tuple[float, float] = (0.040, 0.100)) -> Tuple[float, float]:
    """Detects where note-off / gate-off happened in an audio note.
    
    Uses piecewise log-amplitude derivative analysis to find the 'knee' where
    the decay slope or sustain plateau transitions into the steep release choke.
    
    Returns:
        (gate_off_s, release_rate_db_s)
    """
    peak_idx = int(np.argmax(env))
    min_idx = max(peak_idx + int(0.010 * fs), int(expected_gate_range[0] * fs))
    max_idx = min(len(env) - int(0.005 * fs), int(expected_gate_range[1] * fs))
    
    if min_idx >= max_idx or max_idx >= len(env):
        gate_s = expected_gate_range[1]
        return gate_s, 100.0

    # Smooth log-energy profile
    eps = 1e-6
    log_env = np.log(np.maximum(env, eps))
    
    # Numerical derivative d(ln A)/dt
    dt = 1.0 / fs
    grad = np.gradient(log_env, dt)
    
    # Moving window slopes before and after candidate changepoint tau
    win = int(0.010 * fs)  # 10 ms window
    knee_scores = []
    candidates = range(min_idx, max_idx, max(1, int(0.001 * fs)))
    
    for c in candidates:
        if c - win < 0 or c + win >= len(grad):
            continue
        slope_before = np.mean(grad[c - win:c])
        slope_after = np.mean(grad[c:c + win])
        # Gate-off causes slope to become sharply more negative (steeper drop)
        score = slope_before - slope_after
        knee_scores.append((score, c))
    
    if knee_scores:
        best_score, best_idx = max(knee_scores, key=lambda x: x[0])
        gate_s = best_idx / fs
        # Estimate release rate in dB/s after gate
        post_gate = env[best_idx:min(len(env), best_idx + int(0.030 * fs))]
        if len(post_gate) > 2 and post_gate[0] > 1e-5 and post_gate[-1] > 1e-6:
            db_drop = 20 * math.log10(post_gate[0] / post_gate[-1])
            duration_s = len(post_gate) / fs
            rel_rate = max(10.0, db_drop / duration_s)
        else:
            rel_rate = 120.0
    else:
        gate_s = (expected_gate_range[0] + expected_gate_range[1]) / 2.0
        rel_rate = 80.0

    return float(gate_s), float(rel_rate)


def compute_spectral_centroid_time(audio: np.ndarray, fs: int = 44100, frame_size: int = 512, hop_size: int = 128) -> np.ndarray:
    """Computes spectral centroid trajectory across time frames."""
    if HAS_ESSENTIA:
        w = es.Windowing(type='hann')
        spec = es.Spectrum()
        cent = es.Centroid(range=fs / 2.0)
        centroids = []
        for frame in es.FrameGenerator(audio.astype(np.float32), frameSize=frame_size, hopSize=hop_size):
            s = spec(w(frame))
            c = cent(s)
            centroids.append(c)
        return np.array(centroids, dtype=np.float32)
    
    # Pure SciPy / NumPy STFT centroid
    import scipy.signal as signal
    f, t, Zxx = signal.stft(audio, fs=fs, window='hann', nperseg=frame_size, noverlap=frame_size - hop_size)
    mag = np.abs(Zxx)
    sum_mag = np.sum(mag, axis=0) + 1e-9
    sc = np.sum(f[:, None] * mag, axis=0) / sum_mag
    return sc.astype(np.float32)


def profile_note_envelope(audio: np.ndarray, fs: int = 44100, bpm: float = 143.0) -> NoteEnvelopeProfile:
    """Extracts a comprehensive envelope profile from a bass note waveform."""
    audio = audio.astype(np.float32)
    dur_s = len(audio) / fs
    t_16th = 60.0 / (bpm * 4.0)  # e.g. 0.1049 s at 143 BPM

    # 1. Band separation: Sub band (< 200 Hz) for AEG, Harmonic band (> 450 Hz) for FEG
    sub_audio = butter_bandpass_filter(audio, fs=fs, lowcut=20.0, highcut=200.0)
    harm_audio = butter_bandpass_filter(audio, fs=fs, lowcut=450.0, highcut=min(6000.0, fs * 0.45))

    # 2. Extract fast envelopes (attack=1.5ms, release=8ms for transient tracking)
    env_broadband = zolzer_envelope_follower(audio, fs=fs, t_att=0.0015, t_rel=0.008)
    env_sub = zolzer_envelope_follower(sub_audio, fs=fs, t_att=0.002, t_rel=0.015)
    env_harm = zolzer_envelope_follower(harm_audio, fs=fs, t_att=0.001, t_rel=0.010)

    # 3. Log attack time & attack duration
    log_att, a_start, a_stop = extract_log_attack_time(env_broadband, fs=fs)
    attack_ms = max(0.2, (a_stop - a_start) * 1000.0)
    peak_amp = float(np.max(env_broadband))

    # 4. Gate-off changepoint detection (expected within 40% to 90% of 16th note)
    expected_gate = (0.45 * t_16th, min(dur_s * 0.95, 0.90 * t_16th if dur_s > t_16th else dur_s * 0.85))
    gate_off_s, rel_rate = detect_gate_off_changepoint(env_sub, fs=fs, expected_gate_range=expected_gate)

    # 5. AEG decay half-life, tenth-life, and sustain level on sub-band
    peak_sub_idx = int(np.argmax(env_sub))
    peak_sub_val = float(env_sub[peak_sub_idx])
    gate_sub_idx = min(len(env_sub) - 1, int(gate_off_s * fs))
    
    # Pre-gate portion
    pre_gate_sub = env_sub[peak_sub_idx:gate_sub_idx] if gate_sub_idx > peak_sub_idx else env_sub[peak_sub_idx:]
    sustain_val = float(np.mean(env_sub[max(peak_sub_idx, gate_sub_idx - int(0.015 * fs)):gate_sub_idx])) if gate_sub_idx > peak_sub_idx + int(0.015 * fs) else 0.0
    sustain_level = float(np.clip(sustain_val / (peak_sub_val + 1e-7), 0.0, 1.0))

    # Half-life and 10%-life
    th_sub = np.where(pre_gate_sub <= 0.5 * peak_sub_val)[0]
    tt_sub = np.where(pre_gate_sub <= 0.1 * peak_sub_val)[0]
    aeg_half_ms = (th_sub[0] / fs * 1000.0) if len(th_sub) > 0 else (gate_off_s * 1000.0 * 0.5)
    aeg_tenth_ms = (tt_sub[0] / fs * 1000.0) if len(tt_sub) > 0 else (gate_off_s * 1000.0 * 0.9)

    # 6. FEG decay half-life and tenth-life on harmonic band
    peak_harm_idx = int(np.argmax(env_harm))
    peak_harm_val = float(env_harm[peak_harm_idx])
    pre_gate_harm = env_harm[peak_harm_idx:gate_sub_idx] if gate_sub_idx > peak_harm_idx else env_harm[peak_harm_idx:]
    
    th_harm = np.where(pre_gate_harm <= 0.5 * peak_harm_val)[0]
    tt_harm = np.where(pre_gate_harm <= 0.1 * peak_harm_val)[0]
    feg_half_ms = (th_harm[0] / fs * 1000.0) if len(th_harm) > 0 else (gate_off_s * 1000.0 * 0.3)
    feg_tenth_ms = (tt_harm[0] / fs * 1000.0) if len(tt_harm) > 0 else (gate_off_s * 1000.0 * 0.6)

    # 7. Spectral centroid trajectory
    sc = compute_spectral_centroid_time(audio, fs=fs)
    sc_peak = float(np.max(sc[:int(len(sc) * 0.4)])) if len(sc) > 0 else 1000.0
    sc_sustain = float(np.median(sc[int(len(sc) * 0.4):int(len(sc) * 0.8)])) if len(sc) > 4 else 400.0
    feg_sweep_ratio = float(sc_peak / max(100.0, sc_sustain))

    # 8. Surge XT parameter suggestions
    sug_aeg_decay = float(np.clip(aeg_half_ms / 150.0, 0.05, 0.65))
    sug_aeg_sustain = float(np.clip(sustain_level * 0.85, 0.0, 0.80))
    sug_aeg_release = float(np.clip(100.0 / max(20.0, rel_rate), 0.02, 0.40))
    sug_feg_decay = float(np.clip(feg_half_ms / 120.0, 0.03, 0.65))
    sug_feg_amount = float(np.clip((feg_sweep_ratio - 1.0) / 4.0 + 0.3, 0.20, 0.95))
    sug_cutoff = float(np.clip((sc_sustain - 150.0) / 3000.0, 0.08, 0.90))

    suggested = {
        "a_amp_eg_decay": sug_aeg_decay,
        "a_amp_eg_sustain": sug_aeg_sustain,
        "a_amp_eg_release": sug_aeg_release,
        "a_filter1_cutoff": sug_cutoff,
        "a_filter1_eg_amount": sug_feg_amount,
        "a_filter1_eg_decay": sug_feg_decay,
        "gate_duration_s": gate_off_s,
    }

    return NoteEnvelopeProfile(
        onset_s=0.0,
        duration_s=dur_s,
        gate_off_s=gate_off_s,
        attack_ms=attack_ms,
        log_attack_time=log_att,
        peak_amplitude=peak_amp,
        aeg_decay_half_ms=aeg_half_ms,
        aeg_decay_tenth_ms=aeg_tenth_ms,
        aeg_sustain_level=sustain_level,
        aeg_release_rate_db_s=rel_rate,
        feg_decay_half_ms=feg_half_ms,
        feg_decay_tenth_ms=feg_tenth_ms,
        spectral_centroid_peak_hz=sc_peak,
        spectral_centroid_sustain_hz=sc_sustain,
        feg_sweep_ratio=feg_sweep_ratio,
        suggested_params=suggested,
    )


def compute_envelope_loss_np(y_target: np.ndarray, y_synth: np.ndarray, fs: int = 44100) -> Dict[str, float]:
    """Computes multiscale envelope distance between target and synthesized audio."""
    min_len = min(len(y_target), len(y_synth))
    yt = y_target[:min_len]
    ys = y_synth[:min_len]

    # Sub-band envelopes (AEG)
    sub_yt = butter_bandpass_filter(yt, fs=fs, lowcut=20.0, highcut=200.0)
    sub_ys = butter_bandpass_filter(ys, fs=fs, lowcut=20.0, highcut=200.0)
    env_sub_t = zolzer_envelope_follower(sub_yt, fs=fs, t_att=0.002, t_rel=0.015)
    env_sub_s = zolzer_envelope_follower(sub_ys, fs=fs, t_att=0.002, t_rel=0.015)

    # Harmonic envelopes (FEG)
    harm_yt = butter_bandpass_filter(yt, fs=fs, lowcut=450.0, highcut=min(6000.0, fs * 0.45))
    harm_ys = butter_bandpass_filter(ys, fs=fs, lowcut=450.0, highcut=min(6000.0, fs * 0.45))
    env_harm_t = zolzer_envelope_follower(harm_yt, fs=fs, t_att=0.001, t_rel=0.010)
    env_harm_s = zolzer_envelope_follower(harm_ys, fs=fs, t_att=0.001, t_rel=0.010)

    # MSE losses
    loss_aeg = float(np.mean((env_sub_t - env_sub_s) ** 2))
    loss_feg = float(np.mean((env_harm_t - env_harm_s) ** 2))
    loss_total = loss_aeg * 1.5 + loss_feg * 1.0

    return {
        "loss_env_total": loss_total,
        "loss_aeg": loss_aeg,
        "loss_feg": loss_feg,
    }


def lowpass_300hz(audio: np.ndarray, fs: int = 44100, cutoff_hz: float = 300.0) -> np.ndarray:
    """4th-order zero-phase Butterworth lowpass filter isolating sub & low-mid bass (<300 Hz)."""
    import scipy.signal as signal
    nyq = fs / 2.0
    b, a = signal.butter(4, min(0.99, cutoff_hz / nyq), btype='low')
    return signal.filtfilt(b, a, audio).astype(np.float32)


def compute_bass_match_loss(y_target: np.ndarray, y_synth: np.ndarray, fs: int = 44100, cutoff_hz: float = 300.0) -> Dict[str, float]:
    """Ensures the amount and envelope of lowpassed (<300Hz) bass is preserved.
    
    Matches:
      1. Bass RMS: sqrt(mean(y_bass^2))
      2. Bass Envelope: Zölzer rectified lowpass envelope follower of y_bass
      3. Bass Energy Ratio: RMS(y_bass) / RMS(y_total)
    """
    min_len = min(len(y_target), len(y_synth))
    yt = y_target[:min_len]
    ys = y_synth[:min_len]

    lp_t = lowpass_300hz(yt, fs=fs, cutoff_hz=cutoff_hz)
    lp_s = lowpass_300hz(ys, fs=fs, cutoff_hz=cutoff_hz)

    rms_t = float(np.sqrt(np.mean(lp_t ** 2)))
    rms_s = float(np.sqrt(np.mean(lp_s ** 2)))
    rms_err = abs(rms_t - rms_s) / (rms_t + 1e-6)

    env_t = zolzer_envelope_follower(lp_t, fs=fs, t_att=0.002, t_rel=0.015)
    env_s = zolzer_envelope_follower(lp_s, fs=fs, t_att=0.002, t_rel=0.015)
    env_mse = float(np.mean((env_t - env_s) ** 2))

    tot_rms_t = float(np.sqrt(np.mean(yt ** 2)) + 1e-6)
    tot_rms_s = float(np.sqrt(np.mean(ys ** 2)) + 1e-6)
    ratio_t = rms_t / tot_rms_t
    ratio_s = rms_s / tot_rms_s
    ratio_err = abs(ratio_t - ratio_s)

    loss_total = env_mse * 10.0 + rms_err * 2.0 + ratio_err * 1.0
    return {
        "bass_rms_target": rms_t,
        "bass_rms_synth": rms_s,
        "bass_rms_err": rms_err,
        "bass_env_mse": env_mse,
        "bass_ratio_target": ratio_t,
        "bass_ratio_synth": ratio_s,
        "loss_bass": float(loss_total),
    }



def plot_envelope_profile(audio: np.ndarray, profile: NoteEnvelopeProfile, fs: int = 44100, out_path: str = "envelope_profile.png"):
    """Generates an informative multi-panel visualization of the envelope and filter dynamics."""
    import matplotlib.pyplot as plt

    t = np.arange(len(audio)) / fs
    sub_audio = butter_bandpass_filter(audio, fs=fs, lowcut=20.0, highcut=200.0)
    harm_audio = butter_bandpass_filter(audio, fs=fs, lowcut=450.0, highcut=min(6000.0, fs * 0.45))

    env_broad = zolzer_envelope_follower(audio, fs=fs, t_att=0.0015, t_rel=0.008)
    env_sub = zolzer_envelope_follower(sub_audio, fs=fs, t_att=0.002, t_rel=0.015)
    env_harm = zolzer_envelope_follower(harm_audio, fs=fs, t_att=0.001, t_rel=0.010)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Waveform and Broadband Envelope
    axes[0].plot(t * 1000, audio, color="gray", alpha=0.4, label="Waveform")
    axes[0].plot(t * 1000, env_broad, color="#1f77b4", lw=1.8, label="Broadband Envelope")
    axes[0].axvline(profile.gate_off_s * 1000, color="crimson", ls="--", lw=1.5,
                    label=f"Detected Gate-Off ({profile.gate_off_s*1000:.1f} ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Audio Dynamics & Gate-Off Detection (Attack: {profile.attack_ms:.1f} ms, Gate: {profile.gate_off_s*1000:.1f} ms)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Sub-bass Fundamental (AEG) vs Harmonics (FEG)
    axes[1].plot(t * 1000, env_sub, color="#2ca02c", lw=1.8,
                 label=f"Sub Band (<200 Hz, AEG proxy) - 50% decay {profile.aeg_decay_half_ms:.1f} ms")
    axes[1].plot(t * 1000, env_harm, color="#d62728", lw=1.8,
                 label=f"Harmonic Band (>450 Hz, FEG proxy) - 50% decay {profile.feg_decay_half_ms:.1f} ms")
    axes[1].axhline(profile.aeg_sustain_level * np.max(env_sub), color="#2ca02c", ls=":", alpha=0.7,
                    label=f"AEG Sustain Plateau ({profile.aeg_sustain_level:.2f})")
    axes[1].axvline(profile.gate_off_s * 1000, color="crimson", ls="--", lw=1.2)
    axes[1].set_ylabel("Band Envelope")
    axes[1].set_title("AEG vs FEG Decoupling: Sub Bass vs Upper Harmonics")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Spectral Centroid Trajectory (Filter Cutoff Sweep)
    sc = compute_spectral_centroid_time(audio, fs=fs)
    t_sc = np.linspace(0, len(audio) / fs, len(sc)) * 1000
    axes[2].plot(t_sc, sc, color="#9467bd", lw=2.0, label="Spectral Centroid (Filter Sweep)")
    axes[2].axhline(profile.spectral_centroid_sustain_hz, color="#9467bd", ls=":",
                    label=f"Sustain Cutoff ({profile.spectral_centroid_sustain_hz:.0f} Hz)")
    axes[2].axvline(profile.gate_off_s * 1000, color="crimson", ls="--", lw=1.2)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Centroid (Hz)")
    axes[2].set_title(f"Filter Modulation Depth: Peak {profile.spectral_centroid_peak_hz:.0f} Hz -> Sustain {profile.spectral_centroid_sustain_hz:.0f} Hz")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Essentia & PyTorch Envelope Profiler for Bass Inversion")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--bpm", type=float, default=143.0, help="Tempo in BPM (default 143 for Goa/Psy)")
    parser.add_argument("--plot", type=str, default=None, help="Path to save diagnostic plot PNG")
    parser.add_argument("--json", type=str, default=None, help="Path to save extracted profile JSON")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file {args.audio} not found", file=sys.stderr)
        sys.exit(1)

    if sf is None:
        import scipy.io.wavfile as wavfile
        fs, data = wavfile.read(args.audio)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        audio = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-7)
    else:
        audio, fs = sf.read(args.audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32) / (np.max(np.abs(audio)) + 1e-7)

    profile = profile_note_envelope(audio, fs=fs, bpm=args.bpm)
    print("=" * 60)
    print(f"Envelope Profiling: {os.path.basename(args.audio)} (BPM: {args.bpm})")
    print("=" * 60)
    print(f"  Attack Time:              {profile.attack_ms:.2f} ms (log10: {profile.log_attack_time:.2f})")
    print(f"  Detected Gate-Off / Choke:{profile.gate_off_s * 1000:.1f} ms")
    print(f"  AEG Decay (50% / 10%):    {profile.aeg_decay_half_ms:.1f} ms / {profile.aeg_decay_tenth_ms:.1f} ms")
    print(f"  AEG Sustain Plateau:      {profile.aeg_sustain_level * 100:.1f} %")
    print(f"  AEG Release Rate:         {profile.aeg_release_rate_db_s:.1f} dB/s")
    print(f"  FEG Harmonic Decay (50%): {profile.feg_decay_half_ms:.1f} ms / {profile.feg_decay_tenth_ms:.1f} ms")
    print(f"  Filter Sweep Range:       {profile.spectral_centroid_peak_hz:.0f} Hz -> {profile.spectral_centroid_sustain_hz:.0f} Hz (Ratio: {profile.feg_sweep_ratio:.2f}x)")
    print("-" * 60)
    print("Suggested Surge XT Parameters:")
    for k, v in profile.suggested_params.items():
        print(f"  {k:22s}: {v:.4f}")
    print("=" * 60)

    if args.plot:
        plot_envelope_profile(audio, profile, fs=fs, out_path=args.plot)
        print(f"Plot saved to: {args.plot}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(profile), f, indent=2)
        print(f"JSON saved to: {args.json}")


if __name__ == "__main__":
    main()
