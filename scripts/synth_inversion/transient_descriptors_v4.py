"""Attack and decay descriptors measured from an onset (research Q3; review B13, B15).

All numpy. Call `snap_onset` first (onset_alignment_v4) and pass its onset, or pass 0.0 for a note that starts at
sample 0. Descriptors:

  log_attack_time        log10 of the 10% to 90% rise time of the RMS envelope (seconds), standard LAT
  peak_time_s            time from the onset to the envelope peak (first 300 ms)
  temporal_centroid_s    envelope-weighted mean time after the onset
  early_decay_db_per_s   slope of the log envelope from peak + 20 ms to peak + 120 ms (negative = decaying)
  late_decay_db_per_s    slope from peak + 250 ms to the end (NaN if shorter than 5 frames)
  attack_centroid_hz     spectral centroid of the first `att_s` (40 ms) after the onset
  sustain_centroid_hz    spectral centroid of the last (1 - sus_start_frac) of the remaining clip
  centroid_ratio         attack / sustain centroid (about 1 for a static filter, above 1 when the filter closes)
  band_peak_time_ms      per band (low < 200 Hz, mid 200 to 2 000 Hz, high > 2 000 Hz): time of the band-energy peak
  band_drop_db           per band: peak level minus the level at the sustain window

Envelope window: 4 ms RMS. For a low note (period 7 to 24 ms) the envelope follows the waveform within a cycle;
descriptors are therefore most reliable when the candidate and the target are the SAME note and are compared
with the same code (the use in the objectives and probes).
"""
import numpy as np

from onset_alignment_v4 import rms_envelope

BANDS = {"low": (20.0, 200.0), "mid": (200.0, 2000.0), "high": (2000.0, 16000.0)}


def _slope_db_per_s(t, env, t0, t1):
    m = (t >= t0) & (t <= t1) & (env > 0)
    if m.sum() < 5:
        return float("nan")
    y = 20.0 * np.log10(env[m] + 1e-12)
    return float(np.polyfit(t[m], y, 1)[0])


def _centroid_hz(seg, sr, fmin=40.0, fmax=12000.0):
    if len(seg) < 64:
        return float("nan")
    w = np.hanning(len(seg))
    p = np.abs(np.fft.rfft(seg * w, n=max(4096, 1 << int(np.ceil(np.log2(len(seg)))))) ) ** 2
    f = np.fft.rfftfreq(max(4096, 1 << int(np.ceil(np.log2(len(seg))))), 1.0 / sr)
    m = (f >= fmin) & (f <= fmax)
    tot = p[m].sum()
    return float((f[m] * p[m]).sum() / tot) if tot > 0 else float("nan")


def _band_energy_db(x, sr, n_fft=512, hop_s=0.002):
    hop = max(1, int(round(hop_s * sr)))
    xp = np.pad(np.asarray(x, dtype=np.float64), (0, max(0, n_fft - len(x))))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    p = np.abs(np.fft.rfft(xp[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (hop * np.arange(n_frames) + n_fft / 2) / sr
    out = {}
    for name, (lo, hi) in BANDS.items():
        e = p[:, (f >= lo) & (f < hi)].sum(axis=1)
        out[name] = 10.0 * np.log10(e + 1e-12)
    return out, times


def attack_decay_descriptors(x, sr: int, onset_s: float = 0.0, att_s: float = 0.04, sus_start_frac: float = 0.25,
                             peak_search_s: float = 0.3) -> dict:
    x = np.asarray(x, dtype=np.float64)
    env, t = rms_envelope(x, sr)
    on = float(onset_s)
    end = len(x) / sr
    win = (t >= on) & (t <= min(on + peak_search_s, end))
    if not win.any() or env[win].max() <= 0:
        return {k: float("nan") for k in ("log_attack_time", "peak_time_s", "temporal_centroid_s", "early_decay_db_per_s",
                                          "late_decay_db_per_s", "attack_centroid_hz", "sustain_centroid_hz",
                                          "centroid_ratio")} | {"band_peak_time_ms": {}, "band_drop_db": {}}
    iw = np.where(win)[0]
    ip = int(iw[np.argmax(env[iw])])
    peak, t_peak = float(env[ip]), float(t[ip])
    seg = env[(t >= on) & (t <= t_peak)]
    ts = t[(t >= on) & (t <= t_peak)]
    t10 = ts[np.argmax(seg >= 0.1 * peak)] if (seg >= 0.1 * peak).any() else on
    t90 = ts[np.argmax(seg >= 0.9 * peak)] if (seg >= 0.9 * peak).any() else t_peak
    lat = float(np.log10(max(t90 - t10, 1e-4)))
    after = (t >= on)
    tc = float((t[after] * env[after]).sum() / max(env[after].sum(), 1e-12) - on)
    early = _slope_db_per_s(t, env, t_peak + 0.02, t_peak + 0.12)   # skip the first 20 ms: the peak is a waveform crest
    late = _slope_db_per_s(t, env, t_peak + 0.25, end)

    i_on = int(round(on * sr))
    i_att = min(len(x), i_on + int(round(att_s * sr)))
    rem = len(x) - i_on
    i_sus = i_on + int(round(sus_start_frac * rem))
    c_att = _centroid_hz(x[i_on:i_att], sr)
    c_sus = _centroid_hz(x[i_sus:], sr)
    ratio = c_att / c_sus if (np.isfinite(c_att) and np.isfinite(c_sus) and c_sus > 0) else float("nan")

    be, tb = _band_energy_db(x[i_on:], sr)
    bpt, bdrop = {}, {}
    sus_t0 = sus_start_frac * (rem / sr)
    for name, e in be.items():
        k = int(np.argmax(e[tb <= peak_search_s])) if (tb <= peak_search_s).any() else 0
        bpt[name] = float(tb[k] * 1000.0)
        sus = e[tb >= sus_t0]
        bdrop[name] = float(e[k] - sus.mean()) if len(sus) else float("nan")
    return {"log_attack_time": lat, "peak_time_s": float(t_peak - on), "temporal_centroid_s": tc,
            "early_decay_db_per_s": early, "late_decay_db_per_s": late,
            "attack_centroid_hz": c_att, "sustain_centroid_hz": c_sus, "centroid_ratio": float(ratio),
            "band_peak_time_ms": bpt, "band_drop_db": bdrop}
