"""Loss + gradient-boundary contract for the inharmonicity auxiliary head. Torch only.

PROVIDED BY THE SPEC. Do NOT edit assertions, constants or parametrize lists. If a test fails for a reason that
is not in the code under test, STOP and report the traceback.
Run from scripts/synth_inversion:   REQUIRE_ALL=1 python -m pytest test_inharmonicity_loss.py -q
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

from inharmonicity_aux_head import InharmonicityHead, masked_inharmonicity_loss  # noqa: E402


def _batch():
    pred = torch.tensor([0.30, 0.10, 0.45, 0.20], requires_grad=True)
    target = torch.tensor([0.10, 0.20, 0.00, 0.00])
    valid = torch.tensor([True, True, False, False])
    return pred, target, valid


def test_perfect_prediction_is_zero():
    target = torch.tensor([0.10, 0.20, 0.30])
    pred = target.clone().requires_grad_(True)
    valid = torch.tensor([True, True, True])
    loss, n = masked_inharmonicity_loss(pred, target, valid)
    assert float(loss) == 0.0
    assert float(n) == 3.0


def test_moving_toward_the_target_decreases_the_loss():
    target = torch.tensor([0.10, 0.20, 0.30])
    valid = torch.tensor([True, True, True])
    far = torch.tensor([0.40, 0.00, 0.50])
    near = far + 0.5 * (target - far)
    l_far, _ = masked_inharmonicity_loss(far, target, valid)
    l_near, _ = masked_inharmonicity_loss(near, target, valid)
    assert float(l_near) < float(l_far)


def test_a_higher_prediction_is_not_rewarded_independently_of_the_target():
    valid = torch.tensor([True])
    lo, hi = torch.tensor([0.1]), torch.tensor([0.9])
    t_low, t_high = torch.tensor([0.1]), torch.tensor([0.9])
    assert float(masked_inharmonicity_loss(lo, t_low, valid)[0]) < float(masked_inharmonicity_loss(hi, t_low, valid)[0])
    assert float(masked_inharmonicity_loss(hi, t_high, valid)[0]) < float(masked_inharmonicity_loss(lo, t_high, valid)[0])


def test_invalid_rows_get_exactly_zero_gradient_and_valid_rows_nonzero():
    pred, target, valid = _batch()
    loss, n = masked_inharmonicity_loss(pred, target, valid)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.all(pred.grad[2:] == 0)
    assert torch.all(pred.grad[:2] != 0)
    assert float(n) == 2.0


def test_changing_invalid_target_values_changes_neither_loss_nor_gradient():
    pred1, target, valid = _batch()
    pred2 = pred1.detach().clone().requires_grad_(True)
    t2 = target.clone()
    t2[2] = float("nan")
    t2[3] = float("inf")
    l1, _ = masked_inharmonicity_loss(pred1, target, valid)
    l2, _ = masked_inharmonicity_loss(pred2, t2, valid)
    l1.backward()
    l2.backward()
    assert torch.isfinite(l2)
    assert float(l1) == float(l2)
    assert torch.equal(pred1.grad, pred2.grad)


def test_a_row_flagged_valid_with_a_nonfinite_target_raises():
    pred = torch.tensor([0.1, 0.2], requires_grad=True)
    target = torch.tensor([0.1, float("nan")])
    valid = torch.tensor([True, True])
    with pytest.raises(ValueError):
        masked_inharmonicity_loss(pred, target, valid)


def test_all_invalid_batch_is_a_finite_zero_that_keeps_its_graph():
    pred = torch.tensor([0.3, 0.6], requires_grad=True)
    target = torch.tensor([float("nan"), 0.0])
    valid = torch.tensor([False, False])
    loss, n = masked_inharmonicity_loss(pred, target, valid)
    assert float(loss) == 0.0 and torch.isfinite(loss)
    assert float(n) == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.all(pred.grad == 0) and torch.isfinite(pred.grad).all()


def test_shape_mismatch_raises_instead_of_silently_broadcasting():
    pred = torch.zeros(4)
    valid = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError):
        masked_inharmonicity_loss(pred, torch.zeros(4, 1), valid)
    with pytest.raises(ValueError):
        masked_inharmonicity_loss(pred, torch.zeros(4), torch.ones(4, 1, dtype=torch.bool))


def test_target_and_mask_never_receive_gradients_even_if_they_require_them():
    pred = torch.tensor([0.3, 0.1], requires_grad=True)
    target = torch.tensor([0.1, 0.2], requires_grad=True)
    mask = torch.tensor([1.0, 1.0], requires_grad=True)  # a float 0/1 mask must be accepted
    loss, _ = masked_inharmonicity_loss(pred, target, mask)
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None
    assert mask.grad is None


def test_head_and_online_embedding_get_finite_nonzero_gradients_and_invalid_rows_get_none():
    torch.manual_seed(0)
    head = InharmonicityHead(embed_dim=16, hidden_dim=8)
    emb = torch.randn(4, 16, requires_grad=True)
    target = torch.tensor([0.05, 0.40, 0.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    pred = head(emb)
    assert pred.shape == (4,)
    loss, _ = masked_inharmonicity_loss(pred, target, valid)
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert any(float(p.grad.abs().sum()) > 0 for p in head.parameters())
    assert torch.isfinite(emb.grad).all()
    assert float(emb.grad[:2].abs().sum()) > 0
    assert torch.all(emb.grad[2:] == 0)
