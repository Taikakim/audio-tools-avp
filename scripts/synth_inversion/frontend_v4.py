"""Dual-resolution log-mel front end (research Q4; review B4). NOT wired into the trainer.

Why: the v3 front end is n_fft 1024, hop 441, 128 mel bands from 20 Hz. Bins are 43 Hz wide, so the lowest bands are not
independent measurements (measured with torchaudio's Slaney filterbank: 28 of the 128 bands are fed by at most one FFT
bin, the first band with two or more bins is near 100 Hz). Bass fundamentals are 41 to 147 Hz. A resonance narrower than
about 90 Hz, and the harmonic comb itself, are not resolved; a 30 ms attack spans about 3 frames.

Two branches, band layouts chosen so that no band is fed by fewer than two FFT bins (checked in the tests):
  short  n_fft 512, hop 110 (2.5 ms), 32 bands, 100 Hz to 16 kHz  -> [B, 32, 321]   attack and transients
  long   n_fft 4096, hop 441 (10 ms), 96 bands, 30 Hz to 16 kHz    -> [B, 96, 81]    low-frequency and harmonic structure

Feature scale: dB re full scale, clamped to `top_db` below the clip maximum, then mapped to [-1, 1]. This is NOT the
trainer's `ExactGpuMel` scaling; if this front end is wired into a trainer, make the two consistent and re-run the
normalizer statistics. The temporal encoder (models_v4) takes one 2-D input; feeding two branches needs either two
encoders or a resample of the short branch to the long branch's frame grid. That wiring is a trainer change and is not
done here.
"""
import torch
import torch.nn as nn
import torchaudio.functional as AF


class _MelBranch(nn.Module):
    def __init__(self, sr: int, n_fft: int, hop: int, n_mels: int, fmin: float, fmax: float, top_db: float = 80.0):
        super().__init__()
        self.n_fft, self.hop, self.top_db = n_fft, hop, top_db
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", AF.melscale_fbanks(n_fft // 2 + 1, fmin, fmax, n_mels, sr, norm="slaney",
                                                      mel_scale="slaney").T.contiguous(), persistent=False)

    def forward(self, audio):                                   # audio [B, N]
        sp = torch.stft(audio, self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        pw = (sp.abs() / self.window.sum()) ** 2                # [B, freq, frames]
        mel = torch.einsum("mf,bft->bmt", self.fb, pw)
        db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        top = db.amax(dim=(1, 2), keepdim=True)
        db = torch.maximum(db, top - self.top_db)
        return (db - top) / self.top_db * 2.0 + 1.0             # [-1, 1]: 1 at the clip maximum


class DualMel(nn.Module):
    def __init__(self, sr: int = 44100):
        super().__init__()
        self.short = _MelBranch(sr, 512, 110, 32, 100.0, 16000.0)
        self.long = _MelBranch(sr, 4096, 441, 96, 30.0, 16000.0)

    def forward(self, audio):
        """audio [B, N] (or [N]) -> (short [B, 32, T_s], long [B, 96, T_l])."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return self.short(audio), self.long(audio)
