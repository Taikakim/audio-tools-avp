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


class ExactGpuMel:
    """GPU-accelerated Mel Spectrogram module that exactly matches make_mel_spec() (librosa).

    Matches librosa's Slaney filterbank, Slaney area normalization, constant (zero) padding
    of n_fft // 2, power_to_db with top_db=80.0, and [-1, 1] clipping within 3.7e-5 max abs diff.
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = 1024, hop_length: int = 441,
                 n_mels: int = 128, fmin: float = 20.0, fmax: float = 16000.0, device: str = "cpu"):
        import librosa
        import torch

        self.device = torch.device(device)
        self.n_fft = n_fft
        self.hop_length = hop_length

        fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False, norm="slaney")
        self.mel_fb = torch.from_numpy(fb).float().to(self.device)  # [n_mels, n_fft//2 + 1]
        win = librosa.filters.get_window("hann", n_fft, fftbins=True)
        self.window = torch.from_numpy(win).float().to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.mel_fb = self.mel_fb.to(self.device)
        self.window = self.window.to(self.device)
        return self

    def __call__(self, audio_tensor):
        """audio_tensor: [B, T] or [T] float32 on self.device -> [B, 128, 81] in [-1, 1]."""
        import torch

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        B, T = audio_tensor.shape
        pad_size = self.n_fft // 2

        # Librosa pad_mode='constant'
        x_padded = torch.nn.functional.pad(audio_tensor, (pad_size, pad_size), mode="constant", value=0.0)
        frames = x_padded.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        windowed = frames * self.window
        stft = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
        power = stft.abs().pow(2).transpose(-2, -1)  # [B, 513, num_frames]
        mel = torch.matmul(self.mel_fb, power)        # [B, 128, num_frames]

        ref = torch.amax(mel, dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        spec_db = 10.0 * torch.log10(mel.clamp_min(1e-10)) - 10.0 * torch.log10(ref)
        spec_db = torch.maximum(spec_db, spec_db.amax(dim=(-2, -1), keepdim=True) - 80.0)
        spec_norm = torch.clamp((spec_db + 80.0) / 40.0 - 1.0, -1.0, 1.0)
        return spec_norm


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


def compute_wmfcc(ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 20) -> float:
    """Warped MFCC distance (wMFCC) using Dynamic Time Warping (DTW).
    
    Standard metric used in synth matching benchmarks (Synth-JEPA, Barkan et al.).
    Computes MFCC sequences and finds the cost along the optimal time-warping path.
    """
    import librosa

    r = np.asarray(ref_audio, dtype=np.float32)
    s = np.asarray(syn_audio, dtype=np.float32)
    mfcc_r = librosa.feature.mfcc(y=r, sr=sr, n_mfcc=n_mfcc)
    mfcc_s = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=n_mfcc)
    cost_matrix, wp = librosa.sequence.dtw(mfcc_r, mfcc_s, metric="euclidean")
    return float(cost_matrix[-1, -1] / max(1, len(wp)))


def save_patch(plugin, patch: dict, path_stem: str, extra: dict = None, copy_to_user_dir: bool = True) -> None:
    """Export a matched patch for DAW workflows.

    Writes:
      1. <stem>.json: human-readable configuration parameters.
      2. <stem>.pedalboard_state: pedalboard raw_state blob.
      3. <stem>.vstpreset: native Steinberg VST3 preset container directly loadable
         in Bitwig, Ableton, FL Studio, Reaper, Cubase, etc.
      4. Optionally installs <stem>.vstpreset directly into
         ~/Documents/Surge XT/Patches/AI Inversions/ for instant access in Surge's patch browser.
    """
    import os

    record = {"patch": describe_patch(patch)}
    if extra:
        record.update(extra)
    with open(f"{path_stem}.json", "w") as f:
        json.dump(record, f, indent=2)

    # 1. Pedalboard raw state
    with open(f"{path_stem}.pedalboard_state", "wb") as f:
        f.write(plugin.raw_state)

    # 2. Native .vstpreset container
    vstpreset_path = f"{path_stem}.vstpreset"
    preset_bytes = getattr(plugin, "preset_data", None)
    if preset_bytes:
        with open(vstpreset_path, "wb") as f:
            f.write(preset_bytes)

        # 3. Copy to Surge XT DAW user directory
        if copy_to_user_dir:
            user_patches_dir = os.path.expanduser("~/Documents/Surge XT/Patches/AI Inversions")
            try:
                os.makedirs(user_patches_dir, exist_ok=True)
                preset_name = os.path.basename(vstpreset_path)
                target_dest = os.path.join(user_patches_dir, preset_name)
                with open(target_dest, "wb") as f:
                    f.write(preset_bytes)
            except Exception as e:
                warnings.warn(f"Could not copy {vstpreset_path} to Surge XT user directory: {e}")
