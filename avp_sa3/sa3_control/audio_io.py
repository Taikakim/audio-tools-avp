"""Audio file writing that won't clip.

File handlers — including the new torchcodec backend torchaudio routes through —
expect sample values in [-1, 1] and clip/distort fp16 or out-of-range float input.
SA3 model output regularly peaks above 1.0, so writing it raw clips. Match the SA
gradio GUI fix: float32 -> peak-normalize -> clamp -> int16 PCM (unambiguous).
"""

import torch
import torchaudio


def save_audio(path, audio, sr, normalize=True):
    """Write `audio` (C,T) | (1,T) | (T,) as int16 PCM at `sr`.

    normalize=True peak-normalizes so SA3's >1.0 peaks don't clip (what the GUI does);
    set False to keep relative level and only clamp.
    """
    a = audio.detach().to(torch.float32).cpu()
    if a.dim() == 1:
        a = a.unsqueeze(0)                       # (T,) -> (1, T)
    peak = torch.max(torch.abs(a))
    if normalize and peak > 1e-6:
        a = a / peak                             # loudest sample -> 1.0, no clipping
    a = a.clamp(-1.0, 1.0).mul(32767).to(torch.int16)
    torchaudio.save(str(path), a, int(sr))
