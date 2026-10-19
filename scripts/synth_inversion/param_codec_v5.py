"""Converts between the 60-d ordinal storage and the 126-d onehot layout for v5 models.

Categoricals (16 total):
- 13 original v4 synth categoricals
- osc_1_octave (7 classes: -3 to +3)
- osc_2_octave (7 classes: -3 to +3)
- osc_3_octave (7 classes: -3 to +3)

Continuous (44 total).
Total encoded dim: 44 + 82 = 126.
"""
import torch
import torch.nn.functional as F

from surge_spec_v5 import (
    CAT_INDICES, CONT_INDICES, NUM_PARAMS, OCTAVE_THRESHOLDS, PARAM_INDEX
)

ENCODING_V1 = "ordinal_v1"
ENCODING_V2 = "onehot_v2"

CAT_SIZES = [k for _, k in CAT_INDICES]
N_CONT = len(CONT_INDICES)
ENCODED_DIM = N_CONT + sum(CAT_SIZES)  # 44 + 82 = 126

OCTAVE_PARAM_INDICES = {
    PARAM_INDEX["osc_1_octave"],
    PARAM_INDEX["osc_2_octave"],
    PARAM_INDEX["osc_3_octave"],
}


def _cont_idx(device):
    return torch.tensor(CONT_INDICES, device=device, dtype=torch.long)


def target_classes(p60: torch.Tensor):
    """Ordinal storage -> list of class-index tensors, one per categorical."""
    thresh = torch.tensor(OCTAVE_THRESHOLDS, device=p60.device, dtype=p60.dtype)
    classes = []
    for i, k in CAT_INDICES:
        if i in OCTAVE_PARAM_INDICES:
            cls = torch.bucketize(p60[:, i].contiguous(), thresh).clamp(0, k - 1)
        else:
            cls = torch.round(p60[:, i] * (k - 1)).clamp(0, k - 1).long()
        classes.append(cls)
    return classes


def encode(p60: torch.Tensor) -> torch.Tensor:
    """[B, 60] ordinal -> [B, ENCODED_DIM=126] = [continuous | one-hot blocks]."""
    parts = [p60.index_select(1, _cont_idx(p60.device))]
    for cls, (_, k) in zip(target_classes(p60), CAT_INDICES):
        parts.append(F.one_hot(cls, k).to(p60.dtype))
    return torch.cat(parts, dim=1)


def split_encoded(enc: torch.Tensor):
    """[B, ENCODED_DIM] -> (continuous [B, N_CONT], list of [B, k] blocks)."""
    cont = enc[:, :N_CONT]
    blocks, o = [], N_CONT
    for k in CAT_SIZES:
        blocks.append(enc[:, o:o + k])
        o += k
    return cont, blocks


def assemble(cont: torch.Tensor, classes) -> torch.Tensor:
    """(continuous values, class indices) -> [B, 60] ordinal storage layout."""
    p60 = torch.zeros(cont.shape[0], NUM_PARAMS, device=cont.device, dtype=cont.dtype)
    p60[:, _cont_idx(cont.device)] = cont
    for cls, (i, k) in zip(classes, CAT_INDICES):
        p60[:, i] = cls.to(cont.dtype) / (k - 1)
    return p60


def decode(enc: torch.Tensor) -> torch.Tensor:
    """[B, ENCODED_DIM] (flow output or logits) -> [B, 60] ordinal, argmax per categorical."""
    cont, blocks = split_encoded(enc)
    return assemble(cont.clamp(0.0, 1.0), [b.argmax(dim=1) for b in blocks])


def encoded_weights(w60: torch.Tensor) -> torch.Tensor:
    """Expand 60 per-parameter weights to the encoded layout."""
    parts = [w60.index_select(0, _cont_idx(w60.device))]
    for i, k in CAT_INDICES:
        parts.append(w60[i].repeat(k))
    return torch.cat(parts)


def per_param_error(pred60: torch.Tensor, true60: torch.Tensor) -> torch.Tensor:
    """[B, 60] error: squared error on continuous params, 0/1 misclassification on categoricals."""
    err = (pred60 - true60) ** 2
    for (i, k), pc, tc in zip(CAT_INDICES, target_classes(pred60), target_classes(true60)):
        err[:, i] = (pc != tc).to(err.dtype)
    return err
