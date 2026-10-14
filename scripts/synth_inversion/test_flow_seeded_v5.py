# Owner: KUANG (created 2026-10-15)
"""CPU tests for flow_seeded_search_v5 (tiny random models, no Surge, no GPU). Run from scripts/synth_inversion:

    python -m pytest test_flow_seeded_v5.py -q

They check the mechanics (budget, note pinning, box and couplings, never worse than the seeds, determinism, batched ==
one by one, the audio_eval_v5 integration). They say nothing about audio quality or speed on real checkpoints.
"""
import numpy as np
import pytest
import torch

from test_v5 import BankPrior, _mel, _save_flow, fake_render  # noqa: F401  (also sets up the essentia fakes path)


def _jepa(path=None, seed=0):
    from synth_jepa_model import SynthJEPA
    from synth_jepa_search_v3 import WelfordNormalizer
    torch.manual_seed(seed)
    kw = dict(embed_dim=32, ff_dim=64, num_audio_layers=1, num_param_layers=1, predictor_hidden=32,
              num_slices_sigreg=8, in_frames=81)
    j = SynthJEPA(**kw).eval()
    norm = WelfordNormalizer()
    norm.update(np.random.default_rng(0).standard_normal((8, 128, 81)).astype(np.float32))
    norm.freeze()
    if path is not None:
        torch.save({"model_state_dict": j.state_dict(), "model_kwargs": kw, "normalizer": norm.state_dict(),
                    "prior_bounds": None}, path)
    return j, norm


def _searcher(tmp_path):
    from flow_seeded_search_v5 import FlowSeededSearcher
    j, norm = _jepa()
    flow = _save_flow(tmp_path / "f.pt").eval()
    return FlowSeededSearcher(j, norm, flow, device="cpu")


def test_schedule_fits_the_budget():
    from flow_seeded_search_v5 import FlowSeededSearcher
    for budget in (256, 512, 1024):
        pools, per = FlowSeededSearcher.schedule(budget, 32)
        used = 32 + sum(pools) + per * sum(pools)
        assert pools == [32, 16, 8, 4] and used <= budget and used + sum(pools) > budget


def test_search_respects_budget_note_box_and_couplings(tmp_path):
    from surge_spec_v3 import CONT_BOUNDS, NOTE_HIGH, NOTE_LOW, PARAM_INDEX as I, WAVESHAPER_TYPES
    s = _searcher(tmp_path)
    midi = np.array([30, 41, 50])
    out = s.search(_mel(3), midi, budget=256, n_seeds=16, random_frac=0.25, steps=3, seed=0)
    v = out["vectors"]
    assert v.shape[0] == 3 and out["n_evals_per_target"] <= 256
    assert np.allclose(v[:, I["midi_note"]], (midi - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW), atol=1e-6)
    for name, (lo, hi) in CONT_BOUNDS.items():
        assert np.all(v[:, I[name]] >= lo - 1e-6) and np.all(v[:, I[name]] <= hi + 1e-6), name
    no_unison = v[:, I["unison"]] < 0.5
    assert np.all(v[no_unison, I["unison_detune"]] == 0.0)
    no_ws = v[:, I["waveshaper_type"]] < 0.5 / (len(WAVESHAPER_TYPES) - 1)
    assert np.all(v[no_ws, I["drive"]] == 0.0)
    assert np.all(out["d_jepa"] <= out["seed_best"] + 1e-7)     # never worse than the best seed


def test_returned_score_matches_the_returned_vector(tmp_path):
    s = _searcher(tmp_path)
    mel, midi = _mel(2), np.array([36, 40])
    out = s.search(mel, midi, budget=128, n_seeds=8, steps=3, seed=1)
    cont, cats = s.vectors_to_search(out["vectors"])
    with torch.no_grad():
        za = s.jepa.encode_audio(s.normalizer.normalize(mel))
        d = s.objective(za, cont, cats).numpy()
    np.testing.assert_allclose(d, out["d_jepa"], rtol=1e-4, atol=1e-6)


def test_deterministic_and_batched_equals_one_by_one(tmp_path):
    from flow_sampling_v5 import sample_candidates
    s = _searcher(tmp_path)
    mel, midi = _mel(3), np.array([33, 38, 45])
    seeds = sample_candidates(s.flow, mel, midi, k=8, steps=3, seed=0)
    a = s.search(mel, midi, budget=160, n_seeds=8, seeds=seeds)
    b = s.search(mel, midi, budget=160, n_seeds=8, seeds=seeds)
    assert np.array_equal(a["vectors"], b["vectors"])
    for i in range(3):
        one = s.search(mel[i:i + 1], midi[i:i + 1], budget=160, n_seeds=8, seeds=seeds[i:i + 1])
        np.testing.assert_allclose(one["d_jepa"], a["d_jepa"][i:i + 1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(one["vectors"], a["vectors"][i:i + 1], atol=1e-4)


def test_search_improves_on_the_seeds_somewhere(tmp_path):
    s = _searcher(tmp_path)
    out = s.search(_mel(4), np.array([30, 35, 40, 45]), budget=512, n_seeds=16, steps=3, seed=2)
    assert np.all(out["d_jepa"] <= out["seed_best"]) and np.any(out["d_jepa"] < out["seed_best"] - 1e-6)


def test_audio_eval_runs_seeded_systems(tmp_path):
    import audio_eval_v5 as ae
    from render_pool_v5 import RenderPool
    _save_flow(tmp_path / "f.pt", cond_noise=False)
    _jepa(tmp_path / "j.pt")
    s = ae.main(["--manifold", "fake.npz", "--flow", f"A={tmp_path / 'f.pt'}",
                 "--seeded", f"S={tmp_path / 'f.pt'},{tmp_path / 'j.pt'}", "--seeded_budgets", "64", "128",
                 "--seeded_seeds", "8", "--n", "4", "--k", "2", "--steps", "2", "--metrics", "logmel_l1",
                 "--workers", "0", "--listen", "0", "--device", "cpu", "--out", str(tmp_path / "ev")],
                pool=RenderPool(render_fn=fake_render), prior=BankPrior())
    assert {"S@64", "S@128", "A/draw0", "A/best2"} <= set(s["systems"])
    assert set(s["meta"]["seconds_per_target_incl_render"]) == {"A", "S@64", "S@128"}
