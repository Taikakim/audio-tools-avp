"""CPU/model-free unit tests for the Part-5 MERT continuation reward scaffold."""

import json

import numpy as np
import torch

from sa3_control.mert_reward.reward import (
    RewardSpec,
    composite_reward,
    composite_reward_spec,
)
from sa3_control.mert_reward.model import MERTRewardModel, AUX_FIELDS
from sa3_control.mert_reward.dataset import build_reward_dataset, load_bestof_logs
from sa3_control.mert_reward import train as train_mod


# ----------------------------- reward math --------------------------------- #

def test_reward_zero_at_band_center():
    # melody penalty is zero when melody_sim == band_center; ce/rhythm pass through.
    r = composite_reward(
        ce=5.0, rhythm_sim=0.9, melody_sim=0.675,
        w_ce=1.0, w_rhythm=1.0, w_melody=1.0, band_center=0.675,
    )
    assert abs(r - (5.0 + 0.9)) < 1e-9


def test_reward_band_penalty_sign_symmetric():
    # Deviating either side of the band centre lowers the reward equally.
    base = composite_reward(0.0, 0.0, 0.675, w_ce=1, w_rhythm=1, w_melody=2.0, band_center=0.675)
    below = composite_reward(0.0, 0.0, 0.575, w_ce=1, w_rhythm=1, w_melody=2.0, band_center=0.675)
    above = composite_reward(0.0, 0.0, 0.775, w_ce=1, w_rhythm=1, w_melody=2.0, band_center=0.675)
    assert base == 0.0
    assert below < base and above < base
    assert abs(below - above) < 1e-9          # symmetric
    assert abs(below - (-2.0 * 0.1)) < 1e-9   # penalty = w_melody * |dev|


def test_reward_higher_ce_and_rhythm_increase_reward():
    lo = composite_reward(1.0, 0.1, 0.675, w_ce=1, w_rhythm=1, w_melody=1, band_center=0.675)
    hi = composite_reward(2.0, 0.8, 0.675, w_ce=1, w_rhythm=1, w_melody=1, band_center=0.675)
    assert hi > lo


def test_reward_spec_matches_explicit_and_defaults():
    spec = RewardSpec()  # defaults band 0.55..0.80, center 0.675
    assert abs(spec.band_center - 0.675) < 1e-9
    assert abs(spec.band_width - 0.25) < 1e-9
    r_spec = composite_reward_spec(3.0, 0.7, 0.6, spec)
    r_expl = composite_reward(
        3.0, 0.7, 0.6,
        w_ce=spec.w_ce, w_rhythm=spec.w_rhythm, w_melody=spec.w_melody,
        band_center=spec.band_center,
    )
    assert abs(r_spec - r_expl) < 1e-12


def test_reward_spec_from_band_midpoint():
    spec = RewardSpec.from_band(0.5, 0.9, w_melody=2.0)
    assert abs(spec.band_center - 0.7) < 1e-9
    assert spec.w_melody == 2.0


# ----------------------------- model forward ------------------------------- #

def test_model_forward_shape():
    emb_dim = 16  # tiny for CPU
    B = 5
    model = MERTRewardModel(emb_dim=emb_dim, hidden=32, aux=True)
    prev = torch.randn(B, 2 * emb_dim)
    cand = torch.randn(B, 2 * emb_dim)
    out = model(prev, cand)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_model_forward_aux_shape():
    emb_dim = 16
    B = 4
    model = MERTRewardModel(emb_dim=emb_dim, hidden=32, aux=True)
    prev = torch.randn(B, 2 * emb_dim)
    cand = torch.randn(B, 2 * emb_dim)
    reward, aux = model.forward_aux(prev, cand)
    assert reward.shape == (B,)
    assert aux is not None and aux.shape == (B, len(AUX_FIELDS))


def test_model_no_aux_returns_none():
    model = MERTRewardModel(emb_dim=8, hidden=16, aux=False)
    prev = torch.randn(3, 16)
    cand = torch.randn(3, 16)
    reward, aux = model.forward_aux(prev, cand)
    assert reward.shape == (3,)
    assert aux is None


def test_model_rejects_wrong_emb_dim():
    model = MERTRewardModel(emb_dim=16, hidden=32)
    bad = torch.randn(2, 7)  # not 2*emb_dim
    try:
        model(bad, bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong embedding width")


# ----------------------------- dataset schema ------------------------------ #

def _fake_embedder(emb_dim):
    # Deterministic: returns mid/upper vectors derived from the path hash.
    def _embed(wav_path):
        h = abs(hash(wav_path)) % 1000
        mid = np.full(emb_dim, h * 0.001, dtype=np.float32)
        upper = np.full(emb_dim, -h * 0.001, dtype=np.float32)
        return {"mid": mid, "upper": upper}
    return _embed


def test_build_reward_dataset_schema(tmp_path):
    emb_dim = 8
    log = [
        # first window: prev_wav is None (CE-only continuation)
        {"prev_wav": None, "cand_wav": "/w/win0_cand0.wav",
         "ce": 7.2, "rhythm_sim": 1.0, "melody_sim": 0.675, "reward": 7.2},
        {"prev_wav": "/w/win0_best.wav", "cand_wav": "/w/win1_cand0.wav",
         "ce": 6.5, "rhythm_sim": 0.83, "melody_sim": 0.71, "reward": 8.93},
        {"prev_wav": "/w/win0_best.wav", "cand_wav": "/w/win1_cand1.wav",
         "ce": 6.0, "rhythm_sim": 0.60, "melody_sim": 0.40, "reward": 6.32},
    ]
    log_path = tmp_path / "bestof.json"
    log_path.write_text(json.dumps(log))
    out_npz = tmp_path / "reward_ds.npz"

    ret = build_reward_dataset(
        [str(log_path)], str(out_npz),
        embedder=_fake_embedder(emb_dim), emb_dim=emb_dim,
    )
    assert ret == str(out_npz)

    data = np.load(out_npz)
    assert set(data.files) == {"X", "y", "ce", "rhythm_sim", "melody_sim"}
    n = len(log)
    assert data["X"].shape == (n, 4 * emb_dim)
    assert data["y"].shape == (n,)
    assert data["X"].dtype == np.float32
    # reward + components copied through faithfully
    np.testing.assert_allclose(data["y"], [7.2, 8.93, 6.32], rtol=0, atol=1e-5)
    np.testing.assert_allclose(data["ce"], [7.2, 6.5, 6.0], atol=1e-5)
    np.testing.assert_allclose(data["melody_sim"], [0.675, 0.71, 0.40], atol=1e-5)
    # first row's prev half is the zero vector (missing prev_wav)
    assert np.all(data["X"][0, : 2 * emb_dim] == 0.0)
    assert np.any(data["X"][1, : 2 * emb_dim] != 0.0)


def test_load_bestof_logs_accepts_dict_and_list(tmp_path):
    list_log = tmp_path / "a.json"
    list_log.write_text(json.dumps([{"cand_wav": "x", "ce": 1, "rhythm_sim": 1,
                                     "melody_sim": 1, "reward": 1}]))
    dict_log = tmp_path / "b.json"
    dict_log.write_text(json.dumps({"log": [{"cand_wav": "y", "ce": 2, "rhythm_sim": 2,
                                             "melody_sim": 2, "reward": 2}]}))
    entries = load_bestof_logs([str(list_log), str(dict_log)])
    assert len(entries) == 2
    assert entries[0]["cand_wav"] == "x" and entries[1]["cand_wav"] == "y"


# ----------------------------- loss / train step --------------------------- #

def test_reward_loss_zero_when_perfect():
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, 2.0, 3.0])
    assert float(train_mod.reward_loss(pred, tgt)) == 0.0


def test_reward_loss_adds_aux_term():
    pred = torch.tensor([1.0, 2.0])
    tgt = torch.tensor([1.0, 2.0])  # reward MSE == 0
    pred_aux = torch.zeros(2, 3)
    tgt_aux = torch.ones(2, 3)      # aux MSE == 1
    loss = train_mod.reward_loss(pred, tgt, pred_aux, tgt_aux, aux_weight=0.5)
    assert abs(float(loss) - 0.5) < 1e-6


def test_train_step_reduces_loss_on_tiny_batch():
    emb_dim = 8
    model = MERTRewardModel(emb_dim=emb_dim, hidden=16, aux=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    x = torch.randn(6, 4 * emb_dim)
    y = torch.randn(6)
    aux = torch.randn(6, 3)
    first = train_mod.train_step(model, opt, x, y, aux)
    for _ in range(50):
        last = train_mod.train_step(model, opt, x, y, aux)
    assert last < first  # overfits the tiny batch
