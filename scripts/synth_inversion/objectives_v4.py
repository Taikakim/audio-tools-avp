# Owner: KUANG (created 2026-10-11)
"""Audio-domain objectives for INFERENCE-TIME reranking and refinement (not for training; review B1).

Surge XT is not differentiable, so these never need a gradient. They plug into `refine_v3.Refiner` (its `objective`)
and `inference_v3.predict_and_rerank_candidates` as additional terms. torch is used only for the STFT and the Slaney
mel filterbank (torchaudio.functional); everything else is numpy.

  log_mel_db                      amplitude-referenced log-mel (dB), any (n_fft, hop, n_mels, fmin, fmax)
  logmel_l1                       baseline: global log-mel L1 on the model's front end (128 bands, n_fft 1024, hop 441)
  mss_l1                          baseline: multi-scale STFT log-magnitude L1 (n_fft 512, 1024, 2048, 4096)
  TwoAnchorObjective              the brief's section 2.1 as a scorer: attack window + sustain window + global,
                                  each onset-aligned, weights (4, 2, 1) by default (the brief's values; UNABLATED)
  harmonic_envelope_distance      L1 (dB) between harmonic-index amplitude vectors (pitch-resolved, comb-aware)

Window layouts avoid the redundant low mel bands measured for the current front end (28 of 128 bands fed by at most
one FFT bin): the attack branch uses n_fft 512 with 32 bands from 100 Hz, the sustain branch n_fft 4096 with 96 bands
from 30 Hz.
"""
from functools import lru_cache

import numpy as np
import torch
import torchaudio.functional as AF

from onset_alignment_v4 import snap_onset
from resonance_v4 import harmonic_amplitudes

ATTACK_CFG = dict(n_fft=512, hop=110, n_mels=32, fmin=100.0, fmax=16000.0)
SUSTAIN_CFG = dict(n_fft=4096, hop=441, n_mels=96, fmin=30.0, fmax=16000.0)
GLOBAL_CFG = dict(n_fft=1024, hop=441, n_mels=128, fmin=20.0, fmax=16000.0)


@lru_cache(maxsize=32)
def _fbank(n_fft, n_mels, fmin, fmax, sr):
    return AF.melscale_fbanks(n_fft // 2 + 1, fmin, fmax, n_mels, sr, norm="slaney", mel_scale="slaney")


def log_mel_db(x, sr: int, n_fft: int, hop: int, n_mels: int, fmin: float, fmax: float, floor_db: float = -100.0):
    """Amplitude-referenced log-mel in dB: a unit-amplitude sine has power 0.25 in its bin. Returns [n_mels, frames]."""
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if t.numel() < n_fft:
        t = torch.nn.functional.pad(t, (0, n_fft - t.numel()))
    win = torch.hann_window(n_fft)
    sp = torch.stft(t, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, return_complex=True)
    pw = (sp.abs() / win.sum()) ** 2                                   # [freq, frames]
    mel = _fbank(n_fft, n_mels, fmin, fmax, sr).T @ pw                  # [n_mels, frames]
    return (10.0 * torch.log10(mel.clamp_min(10 ** (floor_db / 10.0)))).numpy()


def _peak_norm(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def logmel_l1(cand, target, sr: int = 44100) -> float:
    a, b = _peak_norm(cand), _peak_norm(target)
    n = min(len(a), len(b))
    return float(np.mean(np.abs(log_mel_db(a[:n], sr, **GLOBAL_CFG) - log_mel_db(b[:n], sr, **GLOBAL_CFG))))


def mss_l1(cand, target, sr: int = 44100, sizes=(512, 1024, 2048, 4096)) -> float:
    a, b = _peak_norm(cand), _peak_norm(target)
    n = min(len(a), len(b))
    tot = 0.0
    for s in sizes:
        cfg = dict(n_fft=s, hop=s // 4, n_mels=min(128, s // 8), fmin=30.0 if s >= 2048 else 100.0, fmax=16000.0)
        tot += float(np.mean(np.abs(log_mel_db(a[:n], sr, **cfg) - log_mel_db(b[:n], sr, **cfg))))
    return tot / len(sizes)


class TwoAnchorObjective:
    """Attack window + sustain window + global, each measured from the clip's own snapped onset.

    cand / target: mono arrays at `sr`. The target's expected onset (from MIDI) is `expected_onset_s`; the candidate is
    a render, so its expected onset is `cand_onset_s` (0.0 for the training renders). Low confidence from `snap_onset`
    (no rise found) falls back to the expected onset; the result carries both confidences.
    """

    def __init__(self, sr: int = 44100, att_s: float = 0.04, sus_start_frac: float = 0.25, weights=(4.0, 2.0, 1.0),
                 snap: bool = True, search_s: float = 0.05, pad_s: float = 0.1):
        self.sr, self.att_s, self.sus_start_frac = sr, att_s, sus_start_frac
        self.weights, self.snap, self.search_s, self.pad_s = tuple(weights), snap, search_s, pad_s

    def _onset(self, x, expected):
        if not self.snap:
            return expected, 1.0
        r = snap_onset(x, self.sr, expected, search_s=self.search_s)
        return r.onset_s, r.confidence

    def components(self, cand, target, expected_onset_s: float = 0.0, cand_onset_s: float = 0.0) -> dict:
        sr = self.sr
        a, b = _peak_norm(cand), _peak_norm(target)
        # A clip that starts at sample 0 (the training renders) has no rise to snap to. Padding BOTH clips with the same
        # lead-in silence makes the candidate and the target get the same onset treatment, so their measurement bias cancels.
        pad = np.zeros(int(round(self.pad_s * sr)))
        a, b = np.concatenate([pad, a]), np.concatenate([pad, b])
        oa, ca = self._onset(a, cand_onset_s + self.pad_s)
        ob, cb = self._onset(b, expected_onset_s + self.pad_s)
        ia, ib = int(round(oa * sr)), int(round(ob * sr))
        a, b = a[ia:], b[ib:]                                       # both start at their own onset
        oa, ob = oa - self.pad_s, ob - self.pad_s
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        n_att = int(round(self.att_s * sr))
        i_sus = int(round(self.sus_start_frac * n))
        att = float(np.mean(np.abs(log_mel_db(a[:n_att], sr, **ATTACK_CFG) - log_mel_db(b[:n_att], sr, **ATTACK_CFG))))
        sus = float(np.mean(np.abs(log_mel_db(a[i_sus:], sr, **SUSTAIN_CFG) - log_mel_db(b[i_sus:], sr, **SUSTAIN_CFG))))
        glob = float(np.mean(np.abs(log_mel_db(a, sr, **GLOBAL_CFG) - log_mel_db(b, sr, **GLOBAL_CFG))))
        return {"attack": att, "sustain": sus, "global": glob, "onset_cand_s": oa, "onset_target_s": ob,
                "confidence_cand": ca, "confidence_target": cb}

    def __call__(self, cand, target, expected_onset_s: float = 0.0, cand_onset_s: float = 0.0) -> float:
        c = self.components(cand, target, expected_onset_s, cand_onset_s)
        w = self.weights
        return (w[0] * c["attack"] + w[1] * c["sustain"] + w[2] * c["global"]) / sum(w)


def harmonic_envelope_distance(cand, target, f0: float, sr: int = 44100, t_start: float = 0.2, n_fft: int = 4096) -> float:
    """Mean absolute difference (dB) of harmonic amplitudes at k * f0, after removing the overall level difference.
    Harmonics invalid in BOTH clips are skipped."""
    fa, aa, va = harmonic_amplitudes(_peak_norm(cand), sr, f0, t_start=t_start, n_fft=n_fft)
    fb, ab, vb = harmonic_amplitudes(_peak_norm(target), sr, f0, t_start=t_start, n_fft=n_fft)
    m = va | vb
    if not m.any():
        return float("nan")
    da, db = aa[m], ab[m]
    return float(np.mean(np.abs((da - da.max()) - (db - db.max()))))
