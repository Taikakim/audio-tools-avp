"""Test for --force-scalar routing in build_fusion_param_groups (train_lora exposure).

`build_fusion_param_groups` already accepts `force_scalar` (a list of regexes matched
with `.search` against each param NAME); matching params are forced onto the scalar
(AdamW) path regardless of shape. This is the escape hatch for 2D matrices that
misbehave under the spectral (NS5/Muon) update.

`scripts/train_lora.py` exposes it via `--force-scalar RE[,RE...]` and the convenience
`--force-scalar-output`, which expands to the SA3-DiT regexes for the final output
projection and the AdaLN modulation emitter — both of which otherwise route SPECTRAL
(2D, min-dim >= 128):

    FINAL OUTPUT PROJECTION : r"\\.project_out\\."          (transformer.project_out.weight, [dim_out, dim])
    ADALN MODULATION EMITTER: r"\\.global_cond_embedder\\.2\\."  (global_cond_embedder.2.weight, [6*dim, dim])

This test pins that those two land in SCALAR when the regexes are passed, that a normal
hidden matrix stays SPECTRAL, and that force_scalar=[] leaves every 2D (min>=128) matrix
in spectral (the off/byte-identical case).

Runnable directly (pytest is not installed in sat-venv):
    sat-venv/bin/python tests/test_force_scalar.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from stable_audio_tools.training.fusion_groups import build_fusion_param_groups

# The EXACT regexes --force-scalar-output expands to (mirrors train_lora.py).
FORCE_SCALAR_OUTPUT = [r"\.project_out\.", r"\.global_cond_embedder\.2\."]

DIM = 256  # >= MIN_SPECTRAL_DIM (128), so every matrix below defaults to spectral


class _TinyDiT(nn.Module):
    """Minimal module whose param names match the SA3-DiT recon regexes. Every weight
    is a 2D matrix with min(shape) >= 128, so WITHOUT force_scalar all route spectral."""

    def __init__(self):
        super().__init__()
        # final output projection: transformer.project_out.weight, [dim_out, dim]
        self.transformer = nn.Module()
        self.transformer.project_out = nn.Linear(DIM, DIM, bias=False)
        # AdaLN modulation emitter: global_cond_embedder is Sequential(Linear, SiLU, Linear);
        # the `.2` Linear emits [6*dim, dim] (scale/shift/gate x {self, ff}).
        self.transformer.global_cond_embedder = nn.Sequential(
            nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, 6 * DIM)
        )
        # a normal hidden matrix that must STAY spectral
        self.transformer.layers = nn.ModuleList([nn.Module()])
        self.transformer.layers[0].hidden = nn.Linear(DIM, DIM, bias=False)

    def named_parameters(self, *a, **k):  # ensure stable, full-path names
        return super().named_parameters(*a, **k)


def _group_names(groups, gtype):
    for g in groups:
        if g["group_type"] == gtype:
            return set(g["param_names"])
    return set()


def _names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def test_force_scalar_output_routes_output_and_adaln_to_scalar():
    """--force-scalar-output regexes force project_out + global_cond_embedder.2 to SCALAR."""
    model = _TinyDiT()
    groups = build_fusion_param_groups(model, force_scalar=FORCE_SCALAR_OUTPUT)
    scalar = _group_names(groups, "scalar")
    spectral = _group_names(groups, "spectral")

    proj = [n for n in _names(model) if ".project_out." in n]
    adaln = [n for n in _names(model) if ".global_cond_embedder.2." in n]
    assert proj and adaln, "test module missing expected param names"

    for n in proj + adaln:
        assert n in scalar, f"{n} should be forced SCALAR, found in {'spectral' if n in spectral else '?'}"
        assert n not in spectral


def test_hidden_matrix_stays_spectral_under_force_scalar():
    """A normal hidden 2D matrix (min>=128) is NOT matched by the output regexes -> SPECTRAL."""
    model = _TinyDiT()
    groups = build_fusion_param_groups(model, force_scalar=FORCE_SCALAR_OUTPUT)
    spectral = _group_names(groups, "spectral")
    scalar = _group_names(groups, "scalar")

    hidden = [n for n in _names(model) if ".layers.0.hidden." in n]
    assert hidden, "test module missing hidden param"
    for n in hidden:
        assert n in spectral, f"{n} should stay SPECTRAL, found in {'scalar' if n in scalar else '?'}"
        assert n not in scalar
    # the input-side AdaLN weight (.0.weight) must NOT be forced (regex targets only .2.)
    # -- it is 2D min>=128 so it stays SPECTRAL. (Its 1D bias routes scalar by shape, as
    # do all biases; the regex is intentionally weight-agnostic and matches neither here.)
    adaln_in = [n for n in _names(model) if ".global_cond_embedder.0.weight" in n]
    assert adaln_in, "test module missing AdaLN input Linear weight"
    for n in adaln_in:
        assert n in spectral, f"{n} (AdaLN input Linear) should stay SPECTRAL"


def test_empty_force_scalar_leaves_all_2d_spectral():
    """Off case: force_scalar=[] => every 2D (min>=128) weight routes SPECTRAL (byte-identical)."""
    model = _TinyDiT()
    groups = build_fusion_param_groups(model, force_scalar=[])
    spectral = _group_names(groups, "spectral")
    scalar = _group_names(groups, "scalar")

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and min(p.shape) >= 128:
            assert n in spectral, f"{n} should be SPECTRAL with empty force_scalar"
            assert n not in scalar
    # specifically: the output projection + AdaLN emitter WEIGHTS route spectral when off
    for n in _names(model):
        if ".project_out.weight" in n or ".global_cond_embedder.2.weight" in n:
            assert n in spectral, f"{n} should be SPECTRAL with empty force_scalar (off case)"
    # only 1D params (biases) land in scalar here; no 2D matrix does
    for n in scalar:
        p = dict(model.named_parameters())[n]
        assert p.ndim == 1, f"{n} unexpectedly in scalar with empty force_scalar (ndim={p.ndim})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    # Diagnostic: show the routing under --force-scalar-output
    m = _TinyDiT()
    gs = build_fusion_param_groups(m, force_scalar=FORCE_SCALAR_OUTPUT)
    for g in gs:
        print(f"  [{g['group_type']:>8}] {sorted(g['param_names'])}")
    sys.exit(1 if failed else 0)
