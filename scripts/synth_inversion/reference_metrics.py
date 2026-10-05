"""Paper-comparable sound-matching metrics, ported from the reference implementation.

Source: Hayes et al., synth-permutations (ISMIR 2025), scripts/compute_audio_metrics.py
(local checkout: /home/kim/Projects/synth-permutations). Synth-JEPA (arXiv:2609.31024, Sec. 3.3)
reports MSS and wMFCC "following [1]", i.e. these definitions.

Why this module exists: audio_utils.MultiScaleSTFTLoss and audio_utils.compute_wmfcc are
DIFFERENT metrics that share the names. MultiScaleSTFTLoss = linear-STFT spectral convergence +
log-magnitude L1 at 4 FFT sizes; compute_wmfcc = librosa-default MFCCs (46 ms window) with a
Euclidean DTW normalised by path length. Both are fine as internal scores, but their numbers are
NOT comparable to the paper's Table 1. Use mss() / wmfcc() here when comparing to the paper.

  mss    log-Mel L1 (dB) at three scales: (10, 25, 100) ms windows, (5, 10, 50) ms hops,
         (32, 64, 128) mels, each ref=np.max, averaged over scales.
  wmfcc  DTW over 20 MFCCs (50 ms window, 10 ms hop, 128 mels), per-frame mean-L1 local cost,
         symmetric2 step pattern, normalised by (N + M) -- dtw-python's normalizedDistance.
  rms_env_cos  cosine similarity of RMS envelopes (50 ms window, 25 ms hop); higher is better.
"""
import numpy as np

SAMPLE_RATE = 44100
MEL_PARAMS = ((10, 5, 32), (25, 10, 64), (100, 50, 128))  # (window ms, hop ms, n_mels)


def _mono(y):
    y = np.asarray(y, dtype=np.float32)
    return y.mean(axis=0) if y.ndim == 2 and y.shape[0] <= 2 else (y.mean(axis=1) if y.ndim == 2 else y)


def mel_specs(y, sr=SAMPLE_RATE):
    import librosa
    out = []
    for win_ms, hop_ms, n_mels in MEL_PARAMS:
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=int(win_ms * sr / 1000),
                                              hop_length=int(hop_ms * sr / 1000), window="hann")
        out.append(librosa.power_to_db(spec, ref=np.max))
    return out


def mss(target, pred, sr=SAMPLE_RATE) -> float:
    t, p = _mono(target), _mono(pred)
    n = min(len(t), len(p))
    return float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(mel_specs(t[:n], sr), mel_specs(p[:n], sr))]))


def mfcc(y, sr=SAMPLE_RATE):
    import librosa
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=int(0.05 * sr), hop_length=int(0.01 * sr), n_mels=128)


def wmfcc(target, pred, sr=SAMPLE_RATE) -> float:
    import librosa
    a, b = mfcc(_mono(target), sr), mfcc(_mono(pred), sr)          # [20, N], [20, M]
    cost = np.mean(np.abs(a[:, :, None] - b[:, None, :]), axis=0)  # [N, M] mean-L1 per frame pair
    # symmetric2: diagonal step costs 2x the local cost, horizontal/vertical 1x
    D, _ = librosa.sequence.dtw(C=cost, step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
                                weights_mul=np.array([2.0, 1.0, 1.0]))
    return float(D[-1, -1] / (a.shape[1] + b.shape[1]))


def rms_env_cos(target, pred, sr=SAMPLE_RATE) -> float:
    import librosa
    w, h = int(0.05 * sr), int(0.025 * sr)
    t = librosa.feature.rms(y=_mono(target), frame_length=w, hop_length=h)[0]
    p = librosa.feature.rms(y=_mono(pred), frame_length=w, hop_length=h)[0]
    n = min(len(t), len(p))
    t, p = t[:n], p[:n]
    return float(np.dot(t, p) / (np.linalg.norm(t) * np.linalg.norm(p) + 1e-12))
