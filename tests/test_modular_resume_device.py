"""Resuming a ModularOptimizer whose state holds preconditioner OBJECTS (Shampoo, SOAP).

torch's Optimizer.load_state_dict moves the tensors it sees directly in each param's state to the
param's device, but not tensors held as attributes of objects stored there. A checkpoint is loaded
onto the CPU first, so a resumed Shampoo run had its C/P matrices on the CPU while the params and
gradients were on the GPU, and the first optimizer step died with "Expected all tensors to be on the
same device" (goa3_avp_r128_shampoo_b16_3e4_2026-09-25, resumed at step 6340, 2026-09-25).
"""
import io

import pytest
import torch
import torch.nn as nn

from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU: the bug is a CPU/GPU mismatch")


def _model(device):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 96), nn.Linear(96, 32)).to(device)
    return m


def _step(m, opt):
    x = torch.randn(8, 64, device=next(m.parameters()).device)
    opt.zero_grad()
    m(x).pow(2).mean().backward()
    opt.step()


@cuda
@pytest.mark.parametrize("whitening", ["shampoo", "soap"])
def test_resume_moves_preconditioner_tensors_to_param_device(whitening):
    dev = "cuda"
    m = _model(dev)
    opt = ModularOptimizer(build_modular_param_groups(m, default_whitening=whitening), lr=1e-3,
                           precond_update_freq=1)
    for _ in range(3):
        _step(m, opt)
    held = [s["preconditioner"] for s in opt.state.values() if s.get("preconditioner") is not None]
    assert held, f"{whitening}: test setup produced no preconditioner objects"

    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location="cpu", weights_only=False)  # what a checkpoint resume does

    m2 = _model(dev)
    opt2 = ModularOptimizer(build_modular_param_groups(m2, default_whitening=whitening), lr=1e-3,
                            precond_update_freq=1)
    opt2.load_state_dict(sd)
    for s in opt2.state.values():
        pc = s.get("preconditioner")
        if pc is None:
            continue
        for k, v in vars(pc).items():
            if torch.is_tensor(v):
                assert v.device.type == "cuda", f"{whitening}: preconditioner.{k} left on {v.device}"
    _step(m2, opt2)  # raised "Expected all tensors to be on the same device" before the fix


def test_step_count_survives_state_dict_roundtrip():
    """_step_count drives LR warmup, the Schedule-Free c_warmup burn-in, and the momentum bias
    corrections. It was a plain attribute, so a resume restarted it at 0: SF averaging went
    INERT (ck=1, x overwritten by z every step), LR re-warmed from 0, and the restored momentum
    was divided by 1-beta1 = 0.1 on the first step. Seen resuming the shampoo run at 6340."""
    m = _model("cpu")
    opt = ModularOptimizer(build_modular_param_groups(m), lr=1e-3, schedule_free=True, sf_c_warmup=2)
    for _ in range(5):
        _step(m, opt)
    assert opt.get_step_count() == 5
    sd = opt.state_dict()
    m2 = _model("cpu")
    opt2 = ModularOptimizer(build_modular_param_groups(m2), lr=1e-3, schedule_free=True, sf_c_warmup=2)
    opt2.load_state_dict(sd)
    assert opt2.get_step_count() == 5


def test_old_checkpoint_without_step_count_loads_and_can_be_set():
    """Checkpoints written before the fix carry no step count; loading one must not fail, and
    the trainer then sets it from Lightning's restored global_step."""
    m = _model("cpu")
    opt = ModularOptimizer(build_modular_param_groups(m), lr=1e-3)
    for _ in range(3):
        _step(m, opt)
    sd = opt.state_dict()
    sd.pop("modular_step_count", None)  # what a pre-fix checkpoint looks like
    opt2 = ModularOptimizer(build_modular_param_groups(_model("cpu")), lr=1e-3)
    opt2.load_state_dict(sd)
    assert opt2.get_step_count() == 0
    opt2.set_step_count(6340)
    assert opt2.get_step_count() == 6340
