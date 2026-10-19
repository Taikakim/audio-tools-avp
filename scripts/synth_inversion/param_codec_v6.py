# Owner: KUANG (created 2026-10-19)
"""v6 storage (59, surge_spec_v6) <-> the flow's encoded layout (114 = 43 continuous + 71 one-hot).

The note (storage index 0) is NOT encoded: it is the flow's condition. Functions that rebuild a storage vector
(assemble, decode) therefore take the note as an argument. jepa_cont() rebuilds the JEPA's continuous input
(storage CONT_INDICES order, note first) from the flow's continuous block and the note.
"""
import torch
import torch.nn.functional as F

from surge_spec_v6 import CAT_INDICES, CONT_INDICES, FLOW_CONT_INDICES, NUM_PARAMS, PARAM_INDEX

ENCODING_V1 = "ordinal_v1"
ENCODING_V2 = "onehot_v2"

CAT_SIZES = [k for _, k in CAT_INDICES]
N_CONT = len(FLOW_CONT_INDICES)                       # 43
ENCODED_DIM = N_CONT + sum(CAT_SIZES)                 # 114
NOTE_INDEX = PARAM_INDEX["midi_note"]
assert CONT_INDICES[0] == NOTE_INDEX and CONT_INDICES[1:] == FLOW_CONT_INDICES, "jepa_cont() relies on this order"


def _idx(ix, device):
    return torch.tensor(ix, device=device, dtype=torch.long)


def note_of(p):
    """[B, 59] storage -> [B] normalised note (the flow's condition)."""
    return p[:, NOTE_INDEX]


def target_classes(p):
    return [torch.round(p[:, i] * (k - 1)).clamp(0, k - 1).long() for i, k in CAT_INDICES]


def encode(p):
    """[B, 59] storage -> [B, 114] = [flow continuous | one-hot blocks]."""
    parts = [p.index_select(1, _idx(FLOW_CONT_INDICES, p.device))]
    for cls, (_, k) in zip(target_classes(p), CAT_INDICES):
        parts.append(F.one_hot(cls, k).to(p.dtype))
    return torch.cat(parts, dim=1)


def split_encoded(enc):
    cont = enc[:, :N_CONT]
    blocks, o = [], N_CONT
    for k in CAT_SIZES:
        blocks.append(enc[:, o:o + k])
        o += k
    return cont, blocks


def assemble(cont, classes, note):
    """(flow continuous [B, 43], class indices, note [B]) -> [B, 59] storage."""
    p = torch.zeros(cont.shape[0], NUM_PARAMS, device=cont.device, dtype=cont.dtype)
    p[:, _idx(FLOW_CONT_INDICES, cont.device)] = cont
    p[:, NOTE_INDEX] = torch.as_tensor(note, dtype=cont.dtype, device=cont.device).expand(cont.shape[0])
    for cls, (i, k) in zip(classes, CAT_INDICES):
        p[:, i] = cls.to(cont.dtype) / (k - 1)
    return p


def decode(enc, note):
    """[B, 114] flow output -> [B, 59] storage (argmax per categorical; the note is the condition it was given)."""
    cont, blocks = split_encoded(enc)
    return assemble(cont.clamp(0.0, 1.0), [b.argmax(dim=1) for b in blocks], note)


def jepa_cont(flow_cont, note):
    """JEPA continuous input in storage order (note first), in [0, 1] units."""
    return torch.cat([torch.as_tensor(note, dtype=flow_cont.dtype, device=flow_cont.device)
                      .expand(flow_cont.shape[0])[:, None], flow_cont], dim=1)


def encoded_weights(w):
    """59 per-parameter weights -> encoded layout weights (the note gets none: it is not predicted)."""
    parts = [w.index_select(0, _idx(FLOW_CONT_INDICES, w.device))]
    for i, k in CAT_INDICES:
        parts.append(w[i].repeat(k))
    return torch.cat(parts)


def per_param_error(pred, true):
    err = (pred - true) ** 2
    for (i, _), pc, tc in zip(CAT_INDICES, target_classes(pred), target_classes(true)):
        err[:, i] = (pc != tc).to(err.dtype)
    return err
