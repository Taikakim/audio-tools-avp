"""Shared audio front-end, target preparation, scoring metric and patch export.

One copy of each so that numbers from different scripts are comparable: before this
module, evaluate_200k_inversion.py and match_untitled_note.py each carried their own
MultiScaleSTFTLoss with DIFFERENT FFT sizes, so their losses could not be compared.
"""
import json
import math
import warnings

import numpy as np

from surge_spec import AUDIO_LEN, DURATION_S, NOTE_DUR_RANGE, SAMPLE_RATE, describe_patch

MEL_KW = dict(n_mels=128, n_fft=1024, hop_length=441, fmin=20, fmax=16000)
MEL_SHAPE = (128, AUDIO_LEN // MEL_KW["hop_length"] + 1)  # (128, 81)


def make_mel_spec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Normalised log-mel, the model input. Refuses anything but 44.1 kHz / 0.8 s,
    because the CNN's adaptive pooling would otherwise accept a mis-scaled input silently."""
    import librosa

    if sr != SAMPLE_RATE:
        raise ValueError(f"make_mel_spec expects {SAMPLE_RATE} Hz audio, got {sr}; use prepare_target()")
    if len(audio) != AUDIO_LEN:
        raise ValueError(f"make_mel_spec expects {AUDIO_LEN} samples, got {len(audio)}; use prepare_target()")
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, **MEL_KW)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_norm = np.clip((spec_db + 80.0) / 40.0 - 1.0, -1.0, 1.0)
    assert spec_norm.shape == MEL_SHAPE, spec_norm.shape
    return spec_norm.astype(np.float32)


def prepare_target(path: str, trim_db: float = 40.0, preroll_ms: float = 2.0, note_dur: float = None):
    """Load a real target the way the training data looks.

    mono -> resample to 44.1 kHz (polyphase; never sox) -> trim leading silence so the
    onset sits at t~0 like every training render -> crop/pad to 0.8 s -> peak-normalise.
    Returns (audio[AUDIO_LEN], info) and warns when the clip is outside the training
    distribution (onset offset, note length) so out-of-distribution numbers are flagged.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    info = {"path": path, "orig_sr": int(sr), "orig_len_s": len(y) / sr}
    if sr != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)

    peak = float(np.max(np.abs(y))) + 1e-12
    above = np.flatnonzero(np.abs(y) > peak * 10 ** (-trim_db / 20))
    onset = int(above[0]) if len(above) else 0
    start = max(0, onset - int(preroll_ms * 1e-3 * SAMPLE_RATE))
    y = y[start:]
    info["trimmed_leading_ms"] = 1000.0 * start / SAMPLE_RATE
    info["content_len_s"] = len(y) / SAMPLE_RATE
    # Score only where the target has real content: past the end of a clip cut from a track
    # the truth is unknown, not silence, so the zero padding must not be scored.
    info["score_len"] = int(min(AUDIO_LEN, len(y)))

    y = np.pad(y, (0, max(0, AUDIO_LEN - len(y))))[:AUDIO_LEN]
    y = (y / (np.max(np.abs(y)) + 1e-7)).astype(np.float32)

    ood = []
    if info["content_len_s"] < DURATION_S:
        ood.append(f"clip has {info['content_len_s']*1000:.0f} ms of content; training windows are "
                   f"{DURATION_S*1000:.0f} ms with the note tail inside")
    if note_dur is not None and not (NOTE_DUR_RANGE[0] <= note_dur <= NOTE_DUR_RANGE[1]):
        ood.append(f"note_dur {note_dur*1000:.0f} ms is outside the training range "
                   f"{NOTE_DUR_RANGE[0]*1000:.0f}-{NOTE_DUR_RANGE[1]*1000:.0f} ms")
    info["out_of_distribution"] = ood
    for msg in ood:
        warnings.warn(f"{path}: OUT OF DISTRIBUTION: {msg}")
    return y, info


class MultiScaleSTFTLoss:
    """Spectral convergence + log-magnitude L1, averaged over 4 resolutions.

    THE scoring metric for every synth-inversion script. Lower is better; 0 = identical.
    """

    FFT_SIZES = (2048, 1024, 512, 256)

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
            hop = n_fft // 4
            window = torch.hann_window(n_fft)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs() + 1e-7
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs() + 1e-7
            sc = torch.norm(Y - X, p="fro") / (torch.norm(Y, p="fro") + 1e-7)
            log_l1 = torch.mean(torch.abs(torch.log(Y) - torch.log(X)))
            total += float(sc + log_l1)
        return total / len(self.FFT_SIZES)


def spectral_centroid(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    import librosa

    return float(np.mean(librosa.feature.spectral_centroid(y=np.asarray(y, dtype=np.float32), sr=sr)))


def save_patch(plugin, patch: dict, path_stem: str, extra: dict = None) -> None:
    """Export a matched patch.

    Writes <stem>.json (every Surge raw value we set, human-readable) and
    <stem>.pedalboard_state (the plugin's full state blob from pedalboard's raw_state).
    The state blob restores exactly via `plugin.raw_state = open(p, 'rb').read()` in
    pedalboard. It is NOT a .vstpreset file (that format wraps the component state in a
    VST3 chunk container); earlier versions mislabelled it as one.
    """
    record = {"patch": describe_patch(patch)}
    if extra:
        record.update(extra)
    with open(f"{path_stem}.json", "w") as f:
        json.dump(record, f, indent=2)
    with open(f"{path_stem}.pedalboard_state", "wb") as f:
        f.write(plugin.raw_state)
