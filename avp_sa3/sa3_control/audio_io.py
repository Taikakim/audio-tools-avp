"""Audio file writing that won't clip — and doesn't depend on torchcodec.

`torchaudio.save` now routes through the **torchcodec** backend, which (a) isn't
installed in every venv (the ROCm 7.14 stack lacks it -> ImportError on save) and
(b) clips fp16 / out-of-range float when it is. SA3 output regularly peaks above
1.0. So we write through **soundfile** instead: float32 -> peak-normalize -> clamp
-> 16-bit PCM (unambiguous, backend-independent). Matches the SA gradio GUI's
normalize intent.
"""

import soundfile as sf
import torch


def save_audio(path, audio, sr, normalize=True):
    """Write `audio` (C,T) | (1,T) | (T,) as 16-bit PCM WAV at `sr` via soundfile.

    normalize=True peak-normalizes so SA3's >1.0 peaks don't clip (what the GUI does);
    set False to keep relative level and only clamp.
    """
    a = audio.detach().to(torch.float32).cpu()
    if a.dim() == 1:
        a = a.unsqueeze(0)                       # (T,) -> (1, T)
    peak = torch.max(torch.abs(a))
    if normalize and peak > 1e-6:
        a = a / peak * 0.8913                    # loudest sample -> -1 dBFS (headroom; avoids
                                                 # inter-sample clipping on lossy/DAC playback)
    a = a.clamp(-1.0, 1.0)
    arr = a.transpose(0, 1).contiguous().numpy()  # soundfile wants (frames, channels)
    sf.write(str(path), arr, int(sr), subtype="PCM_16")
