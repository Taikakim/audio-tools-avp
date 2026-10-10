"""Model-integration contract: the head is OFF by default, adds only its own keys, and cannot change the JEPA.

PROVIDED BY THE SPEC. Do NOT edit assertions or constants. A construction error that is not about the
inharmonicity head (for example a head-count constraint in the encoders) is a TEST-SIDE problem: STOP and report.
Run from scripts/synth_inversion:   REQUIRE_ALL=1 python -m pytest test_inharmonicity_model_integration.py -q
"""
import importlib
import os

import pytest


def need(mod):
    try:
        return importlib.import_module(mod)
    except ImportError:
        if os.environ.get("REQUIRE_ALL") == "1":
            raise
        pytest.skip(f"{mod} not installed", allow_module_level=True)


torch = need("torch")
need("numpy")

from inharmonicity_aux_head import masked_inharmonicity_loss  # noqa: E402
from surge_spec_v3 import CAT_INDICES, CONT_INDICES  # noqa: E402
from synth_jepa_model import SynthJEPA  # noqa: E402

SMALL = dict(embed_dim=64, predictor_hidden=128, num_audio_layers=1, num_param_layers=1,
             num_slices_sigreg=8, ff_dim=128, in_frames=81)


def build(flag, seed=0):
    torch.manual_seed(seed)
    return SynthJEPA(**SMALL, inharmonicity_head=flag)


def inputs(batch=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    mel = torch.randn(batch, 128, 81, generator=g)
    cont = torch.rand(batch, len(CONT_INDICES), generator=g) * 2 - 1
    cats = [torch.nn.functional.one_hot(torch.randint(0, k, (batch,), generator=g), k).float()
            for _, k in CAT_INDICES]
    return mel, cont, cats


def test_the_head_is_disabled_by_default():
    m = SynthJEPA(**SMALL)
    assert not hasattr(m, "inharmonicity_head")
    assert not any(k.startswith("inharmonicity_head") for k in m.state_dict())


def test_flag_adds_only_head_keys():
    off, on = build(False).state_dict(), build(True).state_dict()
    assert not any(k.startswith("inharmonicity_head") for k in off)
    extra = set(on) - set(off)
    assert extra and all(k.startswith("inharmonicity_head") for k in extra)
    assert set(off) <= set(on)


def test_all_other_parameters_are_identical_with_and_without_the_head():
    off, on = build(False).state_dict(), build(True).state_dict()
    for k in off:
        assert torch.equal(off[k], on[k]), k  # the head must not consume RNG before other modules initialise


def test_jepa_losses_and_embeddings_are_unchanged_by_the_head():
    m_off, m_on = build(False), build(True)
    mel, cont, cats = inputs()
    with torch.no_grad():
        out_off = m_off(mel, cont, cats, sigreg_generator=torch.Generator().manual_seed(7))
        out_on = m_on(mel, cont, cats, sigreg_generator=torch.Generator().manual_seed(7))
    assert len(out_off) == len(out_on) == 6
    for a, b in zip(out_off, out_on):
        assert torch.allclose(a, b, atol=1e-7, rtol=0)


def test_old_checkpoints_load_loudly_not_silently():
    m_off, m_on = build(False), build(True)
    sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
    m_off.load_state_dict(sd_off, strict=True)  # same flag: strict load works
    res = m_off.load_state_dict(sd_on, strict=False)  # head checkpoint into a no-head model
    assert res.missing_keys == []
    assert res.unexpected_keys and all(k.startswith("inharmonicity_head") for k in res.unexpected_keys)
    res2 = m_on.load_state_dict(sd_off, strict=False)  # old checkpoint into a head model
    assert res2.unexpected_keys == []
    assert res2.missing_keys and all(k.startswith("inharmonicity_head") for k in res2.missing_keys)


def test_the_auxiliary_gradient_reaches_the_audio_encoder_and_nothing_else():
    m = build(True)
    mel, cont, cats = inputs()
    *_, za, zp = m(mel, cont, cats, sigreg_generator=torch.Generator().manual_seed(7))
    pred = m.inharmonicity_head(za)
    target = torch.tensor([0.05, 0.40, 0.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    loss, _ = masked_inharmonicity_loss(pred, target, valid)
    loss.backward()
    enc = [p.grad for p in m.audio_encoder.parameters() if p.grad is not None]
    assert enc and all(torch.isfinite(g).all() for g in enc)
    assert any(float(g.abs().sum()) > 0 for g in enc)
    for name, p in m.named_parameters():
        if name.startswith(("param_encoder", "f_a2p", "f_p2a")):
            assert p.grad is None, name
