"""Converts between the h5's 23-d ordinal storage and what the models learn.

Categorical parameters (filter circuit, unison, waveshaper) have no meaningful order. The
v1 models regressed them as ordinal numbers (class / (n-1)) with MSE and rounded at decode,
so an uncertain prediction averaged to the middle of an arbitrary list and rounded to
whichever circuit happened to sit there. v2 ("onehot_v2") instead:
  * ResMLP: sigmoid head for continuous params + one logits head per categorical (CE loss).
  * Flow:   flows over [continuous params, one-hot slots per categorical]; decode = argmax.
"""
import torch
import torch.nn.functional as F

from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS

ENCODING_V1 = "ordinal_v1"
ENCODING_V2 = "onehot_v2"

CAT_SIZES = [k for _, k in CAT_INDICES]
N_CONT = len(CONT_INDICES)
ENCODED_DIM = N_CONT + sum(CAT_SIZES)


def _cont_idx(device):
    return torch.tensor(CONT_INDICES, device=device, dtype=torch.long)


def target_classes(p23: torch.Tensor):
    """Ordinal storage -> list of class-index tensors, one per categorical."""
    return [torch.round(p23[:, i] * (k - 1)).clamp(0, k - 1).long() for i, k in CAT_INDICES]


def encode(p23: torch.Tensor) -> torch.Tensor:
    """[B, 23] ordinal -> [B, ENCODED_DIM] = [continuous | one-hot blocks]."""
    parts = [p23.index_select(1, _cont_idx(p23.device))]
    for cls, (_, k) in zip(target_classes(p23), CAT_INDICES):
        parts.append(F.one_hot(cls, k).to(p23.dtype))
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
    """(continuous values, class indices) -> [B, 23] ordinal storage layout."""
    p23 = torch.zeros(cont.shape[0], NUM_PARAMS, device=cont.device, dtype=cont.dtype)
    p23[:, _cont_idx(cont.device)] = cont
    for cls, (i, k) in zip(classes, CAT_INDICES):
        p23[:, i] = cls.to(cont.dtype) / (k - 1)
    return p23


def decode(enc: torch.Tensor) -> torch.Tensor:
    """[B, ENCODED_DIM] (flow output or logits) -> [B, 23] ordinal, argmax per categorical."""
    cont, blocks = split_encoded(enc)
    return assemble(cont.clamp(0.0, 1.0), [b.argmax(dim=1) for b in blocks])


def encoded_weights(w23: torch.Tensor) -> torch.Tensor:
    """Expand 23 per-parameter weights to the encoded layout (each one-hot slot inherits
    its categorical's weight)."""
    parts = [w23.index_select(0, _cont_idx(w23.device))]
    for i, k in CAT_INDICES:
        parts.append(w23[i].repeat(k))
    return torch.cat(parts)


def per_param_error(pred23: torch.Tensor, true23: torch.Tensor) -> torch.Tensor:
    """[B, 23] error that means the same thing for every model: squared error on
    continuous params, 0/1 misclassification on categoricals."""
    err = (pred23 - true23) ** 2
    for (i, k), pc, tc in zip(CAT_INDICES, target_classes(pred23), target_classes(true23)):
        err[:, i] = (pc != tc).to(err.dtype)
    return err
