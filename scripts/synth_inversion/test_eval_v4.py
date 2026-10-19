# Owner: KUANG (created 2026-10-12)
"""Tests for the v4 evaluation tools (eval_stats_v4, heldout_eval_v4, compare_runs_v4, stem_compare_v4,
listening_ab_v4). CPU only, no Surge: rendering is the only part not exercised.

The two strongest checks are parity checks against the trainer's OWN code:
  * draw_items / draw_ladders reproduce render_fixed_set's and render_ladder_set's RNG streams exactly;
  * score_items / score_ladders reproduce validate() and ladder_validation() on the same weights and inputs.
The trainer imports Essentia at module level; off the training machine the essentia/mido fakes under
$TEMP/inharm-review/fakes are used when present, otherwise the parity tests skip.

Run: python -m pytest test_eval_v4.py -q
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import compare_runs_v4 as cr  # noqa: E402
import eval_stats_v4 as st  # noqa: E402
import heldout_eval_v4 as he  # noqa: E402
import listening_ab_v4 as ab  # noqa: E402
import stem_compare_v4 as sc  # noqa: E402


# ------------------------------------------------------------------------------------------------ statistics
def _clustered(rng, k=20, m=25, cluster_sd=1.0, item_sd=1.0, mean=0.0):
    c = np.repeat(np.arange(k), m)
    x = mean + rng.normal(0, cluster_sd, k)[c] + rng.normal(0, item_sd, k * m)
    return x, c


def test_fast_bootstrap_matches_generic():
    rng = np.random.default_rng(0)
    x, c = _clustered(rng, k=12, m=9)
    fast = st.cluster_bootstrap_means(x, c, n_boot=4000, seed=1)
    slow = st.cluster_bootstrap(np.mean, [x], c, n_boot=4000, seed=2)
    assert abs(fast.mean() - slow.mean()) < 0.03
    assert abs(fast.std() - slow.std()) / slow.std() < 0.08


def test_cluster_ci_keeps_coverage_where_iid_ci_does_not():
    rng = np.random.default_rng(3)
    cov_cluster = cov_iid = 0
    sims = 150
    for s in range(sims):
        x, c = _clustered(rng, k=20, m=25)
        r = st.mean_ci(x, c, n_boot=600, seed=s)
        cov_cluster += r["lo"] <= 0.0 <= r["hi"]
        cov_iid += abs(x.mean()) <= 1.96 * r["iid_se"]
    assert cov_cluster / sims > 0.85          # nominal 0.95, percentile bootstrap with 20 clusters runs a bit low
    assert cov_iid / sims < 0.6               # the naive formula is badly overconfident on clustered items


def test_design_effect_reflects_clustering():
    rng = np.random.default_rng(4)
    x, c = _clustered(rng, k=20, m=25, cluster_sd=1.0, item_sd=1.0)
    assert st.mean_ci(x, c, n_boot=2000)["design_effect"] > 5
    y, c2 = _clustered(rng, k=20, m=25, cluster_sd=0.0, item_sd=1.0)
    assert st.mean_ci(y, c2, n_boot=2000)["design_effect"] < 2


def test_paired_delta_uses_the_pairing():
    rng = np.random.default_rng(5)
    base, c = _clustered(rng, k=20, m=25, cluster_sd=2.0, item_sd=1.0)
    a = base + 0.1 + rng.normal(0, 0.05, base.size)
    r = st.paired_delta(a, base, c, n_boot=4000)
    assert r["lo"] > 0 and abs(r["estimate"] - 0.1) < 0.02     # shared difficulty cancels in the pair


def test_paired_delta_drops_nan_pairs_together():
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([0.0, 1.0, np.nan, 3.0])
    r = st.paired_delta(a, b, [0, 1, 2, 3], n_boot=200)
    assert r["n_items"] == 2 and abs(r["estimate"] - 1.0) < 1e-12


def test_sign_test_thresholds_for_24_stems():
    assert st.min_wins(24) == 18
    assert abs(st.sign_test(18, 6) - 0.0227) < 1e-3
    assert abs(st.sign_test(17, 7) - 0.0639) < 1e-3
    assert st.sign_test(0, 0) == 1.0


def test_mcnemar_and_holm():
    r = st.mcnemar_exact([1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1])
    assert (r["a_only"], r["b_only"]) == (2, 1)
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_verdict_rules():
    res = {"estimate": 0.05, "lo": 0.01, "hi": 0.09}
    assert st.verdict(res, True) == "established better"
    assert st.verdict(res, False) == "established worse"
    assert st.verdict(res, True, gauge=0.06) == "below gauge"
    assert st.verdict(res, True, min_effect=0.1) == "below gauge"
    assert st.verdict({"estimate": 0.05, "lo": -0.01, "hi": 0.09}, True) == "not established"


def test_report_binomial_number_and_gauge():
    assert abs(st.binomial_se(0.21, 512) - 0.0180) < 1e-4
    g = st.replicate_gauge([0.018, -0.012, None])
    assert g["n_pairs"] == 2 and abs(g["max_abs"] - 0.018) < 1e-12
    assert st.min_detectable_delta(0.01) == pytest.approx(0.028, abs=1e-3)


# ------------------------------------------------------------------------------------- synthetic manifold
@pytest.fixture
def prior(tmp_path):
    from realistic_bass_prior_v3 import MANIFOLD_FORMAT, RealisticBassPrior
    from surge_spec_v3 import CATEGORICAL, PARAM_INDEX, PARAM_NAMES
    rng = np.random.default_rng(0)
    n, n_val = 48, 14
    P = rng.uniform(0.05, 0.95, size=(n, len(PARAM_NAMES)))
    for name, k in CATEGORICAL.items():
        P[:, PARAM_INDEX[name]] = rng.integers(0, k, n) / max(1, k - 1)
    for col in ("sync", "fm_depth", "sub_mix"):
        P[rng.random(n) < 0.4, PARAM_INDEX[col]] = 0.0
    for fx in ("chorus_mix", "delay_mix", "delay_fb"):
        P[:, PARAM_INDEX[fx]] = 0.0
    is_val = np.zeros(n, dtype=bool)
    is_val[:n_val] = True
    path = tmp_path / "manifold.npz"
    np.savez(path, format_version=MANIFOLD_FORMAT, param_names=np.array(PARAM_NAMES), params=P, is_val=is_val,
             calibrated=True)
    return RealisticBassPrior(str(path), split="val")


def test_draw_items_reproduces_the_trainer_stream(prior):
    d = he.draw_items(prior, 60, seed=1)
    rng = np.random.RandomState(1)
    for i in range(60):
        _, ref, midi, dur = prior.sample_patch_and_midi(rng)
        assert np.array_equal(ref.astype(np.float32), d["vecs"][i]) and midi == d["midi"][i]
        assert abs(dur - d["dur"][i]) < 1e-6
    he.verify_stream(prior, 1, d["vecs"])
    assert d["preset"].min() >= 0 and d["preset"].max() < prior.N


def test_recorded_preset_is_the_archetype(prior):
    from surge_spec_v3 import CATEGORICAL, PARAM_INDEX
    cat = [PARAM_INDEX[n] for n in CATEGORICAL]
    d = he.draw_items(prior, 80, seed=7)
    for v, k in zip(d["vecs"], d["preset"]):
        assert np.allclose(v[cat], prior.params[k][cat].astype(np.float32), atol=1e-6)


def test_draw_ladders_reproduces_render_ladder_set(prior, monkeypatch):
    import surge_spec_v3
    import training_controls_v3 as tc
    monkeypatch.setattr(surge_spec_v3, "render_patch",
                        lambda synth, patch, midi, dur: np.full(8, float(midi) + dur, dtype=np.float32))
    audio, vecs, axis = tc.render_ladder_set(None, prior, 3, 6, seed=3)
    d = he.draw_ladders(prior, 3, 6, seed=3)
    assert np.array_equal(vecs.numpy(), d["vecs"]) and np.array_equal(axis, d["axis"])
    assert np.allclose(audio.numpy()[:, 0], d["midi"] + d["dur"])
    assert len(d["preset"]) == 4 * 3


# ------------------------------------------------------------------------------------------------- scoring
def _trainer():
    fakes = os.path.join(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")), "inharm-review", "fakes")
    if os.path.isdir(fakes) and fakes not in sys.path:
        sys.path.insert(0, fakes)
    try:
        import train_realistic_bass_v3 as T
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"trainer not importable here: {e}")
    return T


def _models(v4=False, aux=True, seed=0):
    import param_codec_v3 as codec
    from synth_jepa_model import SynthJEPA
    from synth_jepa_search_v3 import WelfordNormalizer
    torch.manual_seed(seed)
    from surge_spec_v3 import CAT_INDICES, CONT_INDICES
    kw = dict(embed_dim=64, predictor_hidden=64, num_audio_layers=1, num_param_layers=1, num_slices_sigreg=8,
              ff_dim=64, in_frames=81, inharmonicity_head=aux,
              num_continuous=len(CONT_INDICES), cat_sizes=[k for _, k in CAT_INDICES])   # the v3 trainer's sizes
    jepa = SynthJEPA(**kw).eval()
    norm = WelfordNormalizer(n_mels=128)
    norm.update(np.random.default_rng(seed).normal(size=(16, 128, 81)))
    norm.freeze()
    if v4:
        from models_v4 import build_model_v4
        flow = build_model_v4("flow", hidden_dim=32, num_layers=2, encoder="temporal_asp").eval()
    else:
        from models_v3 import build_model
        flow = build_model("flow", 32, 2, encoding=codec.ENCODING_V2, time_scale=1000.0).eval()
    return jepa, norm, flow, kw


def _w_enc():
    import param_codec_v3 as codec
    from surge_spec_v3 import domain_weights
    return codec.encoded_weights(torch.from_numpy(domain_weights()))


@pytest.mark.parametrize("v4", [False, True])
def test_score_items_reproduces_validate(prior, v4):
    T = _trainer()
    jepa, norm, flow, _ = _models(v4=v4)
    d = he.draw_items(prior, 300, seed=1)
    mel = torch.randn(300, 128, 81)
    w = _w_enc()
    ref = T.validate(jepa, flow, norm, lambda x: x, mel, torch.from_numpy(d["vecs"]), w, 1.0, "cpu")
    got = he.score_items(jepa, norm, flow, mel, d["vecs"], w, "cpu", gallery=300, sample_draws=2, sample_steps=3)
    assert abs(np.mean(got["rank_a2p"] == 0) - ref["r_a2p"]) < 1e-9
    assert abs(np.mean(got["rank_p2a"] == 0) - ref["r_p2a"]) < 1e-9
    assert abs(float(np.mean(got["flow_mse"])) - ref["flow"]) < 1e-5
    assert got["cont_abs_err"].shape[0] == 300 and np.isfinite(got["cont_abs_err"]).all()
    assert set(np.unique(got["cat_correct"])) <= {0.0, 1.0}
    assert np.isfinite(got["aux_pred"]).all()


def test_score_ladders_reproduces_ladder_validation(prior):
    import param_codec_v3 as codec
    import training_controls_v3 as tc
    jepa, norm, flow, _ = _models()
    d = he.draw_ladders(prior, 2, 6, seed=3)
    mel = torch.randn(len(d["vecs"]), 128, 81)
    ref = tc.ladder_validation(jepa, flow, norm, lambda x: x, (mel, torch.from_numpy(d["vecs"]), d["axis"]), 6,
                               "cpu", codec)
    lad = he.score_ladders(jepa, norm, flow, mel, d["vecs"], d["axis"], 6, "cpu")
    means = he.ladder_means(lad)
    assert abs(means["ladder_flow_rho_mean"] - ref["ladder_flow_rho_mean"]) < 1e-9
    assert abs(means["ladder_jepa_rho_mean"] - ref["ladder_jepa_rho_mean"]) < 1e-9
    assert len(lad["flow_rho"]) == 8


def test_block_ranks_and_galleries():
    q = torch.eye(5)
    g = torch.eye(5)
    g[1] = -q[1]                     # item 1's partner moves away: both other items of its block are now closer
    ranks, sizes = he.block_ranks(q, g, block=3)
    assert ranks[0] == 0 and ranks[2] == 0 and list(sizes) == [3, 3, 3, 2, 2]
    assert ranks[1] == 2             # only items of its own block count (block of 3)


@pytest.mark.parametrize("v4", [False, True])
def test_load_kind_round_trips_the_trainer_exports(tmp_path, prior, v4):
    jepa, norm, flow, kw = _models(v4=v4, seed=1)
    meta = {"step": 12000, "prior_md5": "abc"}
    torch.save({"model_state_dict": jepa.state_dict(), "model_kwargs": kw, "normalizer": norm.state_dict(),
                "prior_bounds": {}, **meta}, tmp_path / "jepa_ema_latest.pt")
    torch.save({"model_state": flow.state_dict(), "prior_bounds": {},
                "args": {"model_type": "flow", "param_encoding": "onehot_v2", "time_scale": 1000.0}, **meta},
               tmp_path / "flow_ema_latest.pt")
    j2, n2, f2, info = he.load_kind(str(tmp_path), "ema", "latest", "cpu")
    assert info["step"] == 12000 and info["aux_head"] and type(f2).__name__ == type(flow).__name__
    d = he.draw_items(prior, 40, seed=2)
    mel = torch.randn(40, 128, 81)
    w = _w_enc()
    a = he.score_items(jepa, norm, flow, mel, d["vecs"], w, "cpu", gallery=40, sample_draws=2, sample_steps=3)
    b = he.score_items(j2, n2, f2, mel, d["vecs"], w, "cpu", gallery=40, sample_draws=2, sample_steps=3)
    assert np.array_equal(a["rank_a2p"], b["rank_a2p"]) and np.allclose(a["flow_mse"], b["flow_mse"])
    assert he.load_kind(str(tmp_path), "online", "latest", "cpu") is None


def test_cli_end_to_end_without_surge(tmp_path, prior, monkeypatch, capsys):
    """heldout_eval_v4.main() -> compare_runs_v4.main() on exported random models; only the Surge render is faked."""
    from realistic_bass_prior_v3 import manifold_md5
    from surge_spec_v3 import AUDIO_LEN
    monkeypatch.setattr(he, "render_items", lambda patches, midi, dur, plugin:
                        np.random.default_rng(len(patches)).normal(0, 0.1, (len(patches), AUDIO_LEN)).astype(np.float32))
    md5 = manifold_md5(prior.path)
    for i, (name, v4) in enumerate((("A1", False), ("A2", True), ("A3", True))):
        rd = tmp_path / name
        rd.mkdir()
        jepa, norm, flow, kw = _models(v4=v4, seed=i)
        for kind in ("online", "ema"):
            meta = {"step": 120, "prior_md5": md5}
            torch.save({"model_state_dict": jepa.state_dict(), "model_kwargs": kw, "normalizer": norm.state_dict(),
                        "prior_bounds": {}, **meta}, rd / f"jepa_{kind}_latest.pt")
            torch.save({"model_state": flow.state_dict(), "prior_bounds": {}, **meta,
                        "args": {"model_type": "flow", "param_encoding": "onehot_v2", "time_scale": 1000.0}},
                       rd / f"flow_{kind}_latest.pt")
        (rd / "run_meta.json").write_text(json.dumps({
            "args": {"manifold": prior.path, "plugin": "none", "val_size": 64, "ladder_val_anchors": 2,
                     "ladder_steps": 6, "inharm_weight": 0, "seed": None, "flow_encoder": "temporal_asp" if v4 else "conv"},
            "prior": {"md5": md5}}))
        (rd / "val_kinds.jsonl").write_text(json.dumps({"step": 120, "ema/heldout/r_a2p": 0.1}) + "\n")
    out = tmp_path / "eval"
    rc = he.main(["--run", f"A1={tmp_path / 'A1'}", "--run", f"A2={tmp_path / 'A2'}", "--run", f"A3={tmp_path / 'A3'}",
                  "--kinds", "online", "ema", "--out", str(out), "--device", "cpu", "--extra_n", "64",
                  "--extra_ladder_anchors", "2", "--sample_draws", "2", "--sample_steps", "3"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "parity r_a2p" in printed                                   # the logged record was found and compared
    files = sorted(os.listdir(out))
    assert "base__A3__ema.npz" in files and "ladders_extra__A2__online.npz" in files and "meta__A1__ema.json" in files
    assert len(os.listdir(out / "sets")) == 4
    he.main(["--run", f"A1={tmp_path / 'A1'}", "--kinds", "ema", "--out", str(out), "--device", "cpu",
             "--extra_n", "64", "--extra_ladder_anchors", "2", "--sample_draws", "2", "--sample_steps", "3"])
    assert "rendered" not in capsys.readouterr().out                   # second call reuses the cached sets
    plan = {"name": "smoke", "set": "extra", "ladder_set": "ladders_extra", "kinds": ["ema", "online"], "n_boot": 500,
            "contrasts": [{"name": "asp", "module": "flow", "a": ["A2", "A3"], "b": ["A1"], "primary": ["flow_mse"],
                           "secondary": ["ladder_flow_rho"]}], "replicates": {"flow": [["A2", "A3"]]}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert cr.main(["--dir", str(out), "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "r.md")]) == 0
    assert "## Overall" in (tmp_path / "r.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------ compare_runs_v4
def _fake_run_files(directory, run, kind, n=400, k=20, flow_shift=0.0, jepa_mrr_shift=0.0, seed=0):
    from surge_spec_v3 import CAT_INDICES, CONT_INDICES, PARAM_NAMES
    rng = np.random.default_rng(seed)
    preset = np.repeat(np.arange(k), n // k)
    base_rank = np.random.default_rng(99).integers(0, 30, n)                 # shared item difficulty
    shift = np.where(rng.random(n) < abs(jepa_mrr_shift), -base_rank, 0) if jepa_mrr_shift else 0
    rank = np.clip(base_rank + rng.integers(-2, 3, n) + shift, 0, None)
    item_err = np.random.default_rng(98).uniform(0.1, 0.3, n)
    res = {"rank_a2p": rank, "rank_p2a": np.clip(base_rank + rng.integers(-2, 3, n), 0, None),
           "gallery": np.full(n, 512), "flow_mse": item_err + flow_shift + rng.normal(0, 0.005, n),
           "flow_mae": item_err + rng.normal(0, 0.005, n), "cont_idx": np.array(CONT_INDICES),
           "cat_idx": np.array([i for i, _ in CAT_INDICES]),
           "cont_abs_err": rng.uniform(0, 0.2, (n, len(CONT_INDICES))).astype(np.float32),
           "cont_abs_err_draw0": rng.uniform(0, 0.2, (n, len(CONT_INDICES))).astype(np.float32),
           "cat_correct": (rng.random((n, len(CAT_INDICES))) < 0.7).astype(np.float32),
           "aux_pred": rng.normal(0, 1, n).astype(np.float32), "preset": preset, "midi": np.full(n, 36),
           "items_md5": np.array("items-md5")}
    np.savez(os.path.join(directory, f"base__{run}__{kind}.npz"), **res)
    L = 4 * 8
    lad = {"axis": np.repeat(np.arange(4), 8), "jepa_rho": rng.uniform(0.6, 0.9, L), "flow_rho": rng.uniform(0.3, 0.6, L),
           "flow_mae": rng.uniform(0, 0.1, L), "preset": np.random.default_rng(97).integers(0, k, L),
           "items_md5": np.array("ladder-md5")}
    np.savez(os.path.join(directory, f"ladders__{run}__{kind}.npz"), **lad)
    with open(os.path.join(directory, f"meta__{run}__{kind}.json"), "w") as f:
        json.dump({"run": run, "kind": kind, "info": {"step": 12000, "flow_class": "X", "aux_head": True},
                   "run_args": {"seed": None}, "parity": {}, "param_names": list(PARAM_NAMES)}, f)


def test_compare_runs_end_to_end(tmp_path):
    d = str(tmp_path)
    _fake_run_files(d, "A1", "ema", flow_shift=0.05, seed=1)        # conv flow: worse flow loss
    _fake_run_files(d, "A2", "ema", flow_shift=0.0, seed=2)
    _fake_run_files(d, "A3", "ema", flow_shift=0.0, seed=3)
    plan = {"name": "t", "set": "base", "ladder_set": "ladders", "kinds": ["ema"], "n_boot": 2000,
            "contrasts": [
                {"name": "aux", "module": "jepa", "a": ["A1", "A3"], "b": ["A2"], "primary": ["mrr_a2p"],
                 "secondary": ["ladder_jepa_rho"]},
                {"name": "asp", "module": "flow", "a": ["A2", "A3"], "b": ["A1"], "primary": ["flow_mse"],
                 "secondary": ["param_err:cutoff", "ladder_flow_rho:cutoff_low", "cat_acc"]}],
            "replicates": {"jepa": [["A1", "A3"]], "flow": [["A2", "A3"]]}, "min_effect": {"flow_mse": 0.005}}
    rows = cr.run_plan(cr.Store(d), plan)
    by = {(r["contrast"], r["metric"]): r for r in rows}
    asp = by[("asp", "flow_mse")]
    assert asp["estimate"] < -0.04 and asp["verdict"] == "established better"
    assert asp["gauge_pairs"] == 1 and asp["gauge"] < 0.01
    assert not by[("aux", "mrr_a2p")]["verdict"].startswith("established")
    md = cr.report_md(plan, rows, cr.Store(d))
    assert "asp" in md and "established better" in md


def test_compare_runs_refuses_misaligned_items(tmp_path):
    d = str(tmp_path)
    _fake_run_files(d, "A1", "ema", seed=1)
    _fake_run_files(d, "A2", "ema", seed=2)
    z = dict(np.load(os.path.join(d, "base__A2__ema.npz")))
    z["items_md5"] = np.array("other")
    np.savez(os.path.join(d, "base__A2__ema.npz"), **z)
    plan = {"set": "base", "kinds": ["ema"], "contrasts": [{"name": "x", "a": ["A1"], "b": ["A2"],
                                                            "primary": ["flow_mse"]}]}
    with pytest.raises(RuntimeError):
        cr.run_plan(cr.Store(d), plan)


def test_metric_extraction():
    from surge_spec_v3 import PARAM_NAMES
    res = {"rank_a2p": np.array([0, 1, 4, 9]), "inharm": np.array([0.1, 0.2, 0.0, 0.3]),
           "inharm_valid": np.array([1, 1, 0, 1]), "aux_pred": np.array([0.1, 0.0, 5.0, 0.3]),
           "preset": np.zeros(4)}
    assert list(cr.per_item(res, "hit_a2p", PARAM_NAMES)) == [1, 0, 0, 0]
    assert np.allclose(cr.per_item(res, "mrr_a2p", PARAM_NAMES), [1, 0.5, 0.2, 0.1])
    assert list(cr.per_item(res, "top5_a2p", PARAM_NAMES)) == [1, 1, 1, 0]
    aux = cr.per_item(res, "aux_sqerr", PARAM_NAMES)
    assert np.isnan(aux[2]) and abs(aux[1] - 0.04) < 1e-12
    with pytest.raises(KeyError):
        cr.metric_info("nonsense")


# ------------------------------------------------------------------------------------------ stem_compare_v4
def test_stem_compare_selects_on_dev_and_reports_on_test(tmp_path):
    rng = np.random.default_rng(0)
    stems = [f"s{i:02d}" for i in range(24)]
    rows = []
    for s in stems:
        base = rng.uniform(6, 9)
        for seed in range(3):
            rows.append({"stem": s, "system": "ema/tau1.00/raw", "seed": seed, "metric": "mss", "value": base + rng.normal(0, .05)})
            rows.append({"stem": s, "system": "ema/tau0.50/raw", "seed": seed, "metric": "mss", "value": base - 0.3 + rng.normal(0, .05)})
            rows.append({"stem": s, "system": "ema/tau0.15/raw", "seed": seed, "metric": "mss", "value": base + 0.5 + rng.normal(0, .05)})
    table = sc.pivot(rows, "mss")
    dev, test = sc.split_stems(stems, seed=0)
    assert len(dev) == 12 and not set(dev) & set(test)
    best, means, _ = sc.select_best(table, "ema/tau", dev)
    assert best == "ema/tau0.50/raw"
    c = sc.compare(table, best, "ema/tau1.00/raw", test)
    assert c["wins"] == 12 and c["sign_p"] < 0.001 and c["hi"] < 0
    assert c["median_seed_sd"] < 0.1


def test_stem_compare_cli(tmp_path, capsys):
    import csv as _csv
    rng = np.random.default_rng(1)
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["stem", "system", "seed", "metric", "value"])
        for i in range(10):
            b = rng.uniform(6, 9)
            for seed in range(2):
                w.writerow([f"s{i}", "base", seed, "mss", b + rng.normal(0, .05)])
                w.writerow([f"s{i}", "fam/x", seed, "mss", b - .2 + rng.normal(0, .05)])
                w.writerow([f"s{i}", "fam/y", seed, "mss", b + .2 + rng.normal(0, .05)])
    out = tmp_path / "r.md"
    assert sc.main(["--csv", str(path), "--baseline", "base", "--select", "best=fam/", "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "-> **fam/x**" in md and "[test]" in md and "[all]" in md


def test_from_multitemp_parsing(tmp_path):
    res = {"ema": {"stem_01": {"temps": {"t0.15": {"tau": 0.15, "stft_phrase_raw": 7.1, "stft_phrase_harm": 7.3}}}},
           "online": {"stem_01": {"temps": {"t1.00": {"tau": 1.0, "stft_phrase_raw": 6.9, "stft_phrase_harm": None}}}}}
    p = tmp_path / "results.json"
    p.write_text(json.dumps(res))
    rows = sc.from_multitemp(str(p))
    names = sorted(r["system"] for r in rows)
    assert names == ["ema/tau0.15/harm", "ema/tau0.15/raw", "online/tau1.00/raw"]


# ------------------------------------------------------------------------------------------ listening_ab_v4
def test_listening_analysis():
    key = {"trials": {"t001": {"stem": "s1", "A": "X", "B": "Y", "check": False},
                      "t002": {"stem": "s2", "A": "Y", "B": "X", "check": False},
                      "t003": {"stem": "s3", "A": "X", "B": "Y", "check": False},
                      "t004": {"stem": "s1", "A": "__reference__", "B": "__degraded__", "check": True}}}
    answers = [{"trial": "t001", "choice": "A"}, {"trial": "t002", "choice": "B"}, {"trial": "t003", "choice": "B"},
               {"trial": "t004", "choice": "A"}]
    res = ab.analyze_answers(key, answers)
    assert res["checks_passed"] == 1 and res["checks_total"] == 1
    p = res["pairs"][0]
    assert (p["x"], p["y"]) == ("X", "Y") and abs(p["x_pref"] - 2 / 3) < 1e-12
    assert p["stem_wins_x"] == 2 and p["stem_wins_y"] == 1


def test_listening_make_writes_a_blind_set(tmp_path):
    sf = pytest.importorskip("soundfile")
    clips, refs = tmp_path / "clips", tmp_path / "refs"
    clips.mkdir()
    refs.mkdir()
    sr = 22050
    t = np.arange(sr) / sr
    for s in ("s1", "s2"):
        sf.write(refs / f"{s}.wav", 0.3 * np.sin(2 * np.pi * 55 * t), sr)
        for sysname, f in (("X", 55.0), ("Y", 58.0)):
            sf.write(clips / f"{s}__{sysname}.wav", 0.1 * np.sin(2 * np.pi * f * t), sr)
    out = tmp_path / "ab"
    ab.main(["make", "--clips", str(clips), "--refs", str(refs), "--pair", "X", "Y", "--checks", "1",
             "--seed", "0", "--out", str(out)])
    trials = json.loads((out / "trials.json").read_text())
    key = json.loads((out / "key.json").read_text())
    assert len(trials) == 3 and sum(v["check"] for v in key["trials"].values()) == 1
    public = json.dumps(trials) + (out / "index.html").read_text(encoding="utf-8")
    assert '"X"' not in public and '"Y"' not in public and "__X" not in public   # the page does not reveal systems
    for tr in trials:
        x, _ = sf.read(out / tr["A"])
        assert np.max(np.abs(x)) <= 0.99
