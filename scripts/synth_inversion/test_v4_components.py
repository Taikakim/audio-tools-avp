"""CPU tests for the v4 components. No Surge, no Essentia, no GPU. Run from scripts/synth_inversion:

    python -m pytest test_v4_components.py -q

REQUIRE_ALL=1 makes a missing torch / torchaudio an error instead of a skip (same convention as the inharmonicity tests).

What these tests are: unit tests on a stand-in synth (standin_synth_v4.py) and on random tensors. They show that each
component does what its docstring says in a controlled setting. They do NOT show that a component helps on Surge audio
or on real stems; that needs the probe harness (probe_metrics_v4.py) on the training machine and listening tests.
"""
import importlib
import os

import numpy as np
import pytest
from scipy.stats import spearmanr

SR = 44100


def need(mod):
    try:
        return importlib.import_module(mod)
    except ImportError:
        if os.environ.get("REQUIRE_ALL") == "1":
            raise
        pytest.skip(f"{mod} not installed", allow_module_level=True)


torch = need("torch")
need("torchaudio")

from standin_synth_v4 import StandinPatch, _lowpass_gain, render  # noqa: E402
from onset_alignment_v4 import snap_onset  # noqa: E402
from transient_descriptors_v4 import attack_decay_descriptors as describe  # noqa: E402
from resonance_v4 import brief_prominence_db, resonance_card, resonance_card_from_amplitudes  # noqa: E402
from buzz_v4 import comb_residual, comb_residual_timeline  # noqa: E402
from objectives_v4 import (GLOBAL_CFG, TwoAnchorObjective, harmonic_envelope_distance, log_mel_db,  # noqa: E402
                           logmel_l1, mss_l1)
from frontend_v4 import DualMel  # noqa: E402

# models_v3 imports surge_spec_v3 -> pedalboard may be absent on a laptop; the model tests skip then
try:
    import models_v3  # noqa: E402
    import models_v4  # noqa: E402
    HAVE_MODELS = True
except ImportError:
    if os.environ.get("REQUIRE_ALL") == "1":
        raise
    HAVE_MODELS = False
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="models_v3 dependencies not installed")


# ----------------------------------------------------------------------------------------------------------------------
# stand-in synth
# ----------------------------------------------------------------------------------------------------------------------
def test_standin_is_deterministic_normalized_and_silent_before_the_onset():
    p = StandinPatch(f0=55.0, phase_seed=3)
    a, b = render(p, onset_s=0.1), render(p, onset_s=0.1)
    assert np.array_equal(a, b)
    assert abs(float(np.max(np.abs(a))) - 1.0) < 1e-6
    assert float(np.max(np.abs(a[: int(0.099 * SR)]))) < 1e-6


# ----------------------------------------------------------------------------------------------------------------------
# onset alignment
# ----------------------------------------------------------------------------------------------------------------------
def test_snap_onset_result_does_not_depend_on_the_expected_onset():
    x = render(StandinPatch(f0=55.0, attack_s=0.003, amp_decay_s=0.2), onset_s=0.10, note_dur=0.4)
    found = [snap_onset(x, SR, 0.10 + s).onset_s for s in (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)]
    assert max(found) - min(found) <= 0.001 + 1e-9
    assert abs(found[3] - 0.10) <= 0.003


def test_snap_onset_survives_noise():
    rng = np.random.default_rng(0)
    x = render(StandinPatch(f0=98.0, attack_s=0.003, amp_decay_s=0.2), onset_s=0.10, note_dur=0.4)
    x = x + 1e-2 * rng.standard_normal(len(x)).astype(np.float32)             # -40 dB re full scale
    r = snap_onset(x, SR, 0.13)
    assert abs(r.onset_s - 0.10) <= 0.004 and r.confidence > 0.5


def test_snap_onset_is_consistent_across_patches_of_the_same_note():
    # relative alignment is what the objectives need: same note and attack, different decay -> same onset
    on = [snap_onset(render(StandinPatch(f0=55.0, attack_s=0.005, amp_decay_s=d, phase_seed=1), onset_s=0.1, note_dur=0.5),
                     SR, 0.12).onset_s for d in (0.05, 0.2, 0.6)]
    assert max(on) - min(on) <= 0.0015


def test_snap_onset_reports_no_confidence_when_there_is_no_rise():
    r = snap_onset(render(StandinPatch(f0=55.0), onset_s=0.0), SR, 0.0)          # already sounding at sample 0
    assert r.confidence == 0.0 and r.onset_s == 0.0
    r = snap_onset(np.zeros(35280, np.float32), SR, 0.1)
    assert r.confidence == 0.0 and r.onset_s == 0.1


# ----------------------------------------------------------------------------------------------------------------------
# transient descriptors
# ----------------------------------------------------------------------------------------------------------------------
def test_decay_slopes_follow_the_amplitude_time_constant():
    taus = (0.05, 0.1, 0.2, 0.4)
    early, late = [], []
    for tau in taus:
        d = describe(render(StandinPatch(f0=98.0, attack_s=0.003, amp_decay_s=tau, amp_sustain=0.0, fc_hz=3000, phase_seed=1),
                            note_dur=0.8), SR, 0.0)
        ideal = -8.686 / tau
        early.append(d["early_decay_db_per_s"])
        late.append(d["late_decay_db_per_s"])
        assert abs(d["late_decay_db_per_s"] - ideal) / abs(ideal) < 0.03
        assert abs(d["early_decay_db_per_s"] - ideal) / abs(ideal) < 0.12
    assert early == sorted(early, reverse=True) or all(a < b for a, b in zip(early, early[1:]))   # slower decay -> shallower slope


def test_log_attack_time_orders_the_attack_where_it_is_resolvable():
    # documented limit: attacks much shorter than one waveform period are not resolved on a low note
    lat = [describe(render(StandinPatch(f0=55.0, attack_s=a, amp_decay_s=0.3, phase_seed=1)), SR, 0.0)["log_attack_time"]
           for a in (0.010, 0.020, 0.040)]
    assert lat[0] < lat[1] < lat[2]
    lat = [describe(render(StandinPatch(f0=146.8, attack_s=a, amp_decay_s=0.3, phase_seed=1)), SR, 0.0)["log_attack_time"]
           for a in (0.002, 0.010, 0.040)]
    assert lat[0] < lat[1] < lat[2]


def test_centroid_ratio_separates_a_closing_filter_from_a_static_one():
    static = describe(render(StandinPatch(f0=55.0, fc_hz=600, fc_env_oct=0.0, phase_seed=1)), SR, 0.0)
    closing = describe(render(StandinPatch(f0=55.0, fc_hz=600, fc_env_oct=2.5, fc_decay_s=0.06, phase_seed=1)), SR, 0.0)
    assert abs(static["centroid_ratio"] - 1.0) < 0.15
    assert closing["centroid_ratio"] > 1.3
    assert closing["band_drop_db"]["high"] > static["band_drop_db"]["high"] + 10.0


def test_descriptors_do_not_crash_on_silence():
    d = describe(np.zeros(35280, np.float32), SR, 0.0)
    assert np.isnan(d["log_attack_time"]) and d["band_peak_time_ms"] == {}


# ----------------------------------------------------------------------------------------------------------------------
# resonance
# ----------------------------------------------------------------------------------------------------------------------
QS = [0.7, 1.0, 2.0, 4.0, 6.0, 10.0, 14.0, 20.0, 30.0]


def _analytic(f0, fc=1000.0):
    ks = np.arange(1, int(8000 // f0) + 1)
    fr = ks * f0
    return fr, {q: 20 * np.log10(_lowpass_gain(fr, fc, q) / ks) for q in QS}


@pytest.mark.parametrize("f0", [41.2, 82.4, 146.8])
def test_resonance_height_orders_by_q_and_centre_tracks_the_cutoff(f0):
    fr, amps = _analytic(f0)
    cards = [resonance_card_from_amplitudes(fr, amps[q], f0) for q in QS]
    h = [c.height_db for c in cards]
    assert spearmanr(QS, h)[0] > 0.99
    for q, c in zip(QS, cards):
        if q >= 2.0:
            assert c.has_peak and abs(c.peak_freq_hz - 1000.0) <= f0 * 1.0
    assert not cards[0].has_peak                                               # Q = 0.7 has no resonance


def test_resonance_width_is_resolved_on_a_low_note_and_ordered():
    fr, amps = _analytic(41.2)
    cards = [resonance_card_from_amplitudes(fr, amps[q], 41.2) for q in QS if q >= 2.0]
    assert all(not c.q_censored for c in cards)
    widths = [c.width_hz for c in cards]
    assert all(a > b for a, b in zip(widths, widths[1:]))                      # narrower resonance -> narrower width


def test_narrow_resonance_on_a_high_note_is_censored_not_labelled():
    fr, amps = _analytic(146.8)
    c = resonance_card_from_amplitudes(fr, amps[20.0], 146.8)
    assert c.has_peak and c.q_censored and c.width_hz is None and c.q_like is None
    assert c.q_like_lower_bound is not None and c.width_upper_hz == pytest.approx((c.n_elevated + 1) * 146.8)


@pytest.mark.parametrize("f0", [55.0, 146.8])
def test_resonance_card_from_audio_orders_by_q(f0):
    h = [resonance_card(render(StandinPatch(f0=f0, fc_hz=1000.0, q=q, amp_decay_s=0.5, amp_sustain=0.8, phase_seed=1),
                               note_dur=0.7), SR, f0).height_db for q in (1.0, 4.0, 10.0, 20.0)]
    assert all(a < b for a, b in zip(h, h[1:]))


def test_brief_prominence_measures_the_comb_not_the_resonance():
    """Review B12: the rolling-median prominence of DESIGN_BRIEF_V4.md 2.3 on a 41 Hz note is the harmonic-to-valley
    contrast. It is large without any resonance and barely changes with Q, while the harmonic-envelope height does."""
    f0 = 41.2
    prom, height = [], []
    for q in (0.7, 4.0, 20.0):
        x = render(StandinPatch(f0=f0, fc_hz=1000.0, q=q, amp_decay_s=0.5, amp_sustain=0.8, phase_seed=1), note_dur=0.7)
        prom.append(brief_prominence_db(x, SR))
        height.append(resonance_card(x, SR, f0).height_db)
    assert prom[0] > 15.0                                                      # a "resonance" with Q = 0.7, i.e. none
    assert max(prom) - min(prom) < 2.0
    assert max(height) - min(height) > 15.0


# ----------------------------------------------------------------------------------------------------------------------
# buzz
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("f0", [82.4, 146.8])
def test_comb_residual_separates_clean_inharmonic_and_integer_fm(f0):
    base = dict(f0=f0, fc_hz=2500, q=0.7, amp_decay_s=0.5, amp_sustain=0.8, phase_seed=1)
    clean = comb_residual(render(StandinPatch(**base)), SR, f0)
    fm = comb_residual(render(StandinPatch(**base, fm_ratio=1.37, fm_depth=1.0)), SR, f0)
    fm_light = comb_residual(render(StandinPatch(**base, fm_ratio=1.37, fm_depth=0.3)), SR, f0)
    integer = comb_residual(render(StandinPatch(**base, fm_ratio=2.0, fm_depth=1.0)), SR, f0)
    assert clean["residual"] < 0.01 and clean["resolution_ok"]
    assert fm["residual"] > 0.2 and fm["residual"] > 20 * clean["residual"]
    assert fm["residual"] > fm_light["residual"] > 3 * clean["residual"]       # scales with the FM depth
    assert integer["residual"] < 0.02                                          # integer ratio stays on the comb


def test_comb_residual_timeline_separates_static_from_enveloped_buzz():
    f0 = 110.0
    base = dict(f0=f0, fc_hz=2500, q=0.7, amp_decay_s=0.5, amp_sustain=0.8, phase_seed=1, fm_ratio=1.37, fm_depth=1.0)
    _, static = comb_residual_timeline(render(StandinPatch(**base)), SR, f0)
    _, env = comb_residual_timeline(render(StandinPatch(**base, fm_decay_s=0.12)), SR, f0)
    assert 0.8 < static[-1] / static[0] < 1.25
    assert env[0] > 10 * env[-1]


def test_comb_residual_flags_a_comb_it_cannot_resolve():
    x = render(StandinPatch(f0=41.2, fc_hz=2500, phase_seed=1), note_dur=0.8)[: int(0.12 * SR)]
    assert comb_residual(x, SR, 41.2, t_start=0.0)["resolution_ok"] is False   # a 0.12 s segment cannot resolve 41 Hz


# ----------------------------------------------------------------------------------------------------------------------
# objectives
# ----------------------------------------------------------------------------------------------------------------------
BASE = dict(f0=55.0, fc_hz=500.0, fc_env_oct=2.5, q=1.5, amp_decay_s=0.2, amp_sustain=0.3, attack_s=0.003, phase_seed=1)


def test_log_mel_matches_the_trainer_front_end_shape():
    assert log_mel_db(np.zeros(35280, np.float32), SR, **GLOBAL_CFG).shape == (128, 81)


def test_objectives_are_zero_for_identical_audio_and_follow_a_graded_decay_mismatch():
    tgt = render(StandinPatch(**BASE, fc_decay_s=0.06))
    ta = TwoAnchorObjective()
    assert ta(tgt, tgt) == pytest.approx(0.0, abs=1e-6)
    assert logmel_l1(tgt, tgt) == pytest.approx(0.0, abs=1e-6) and mss_l1(tgt, tgt) == pytest.approx(0.0, abs=1e-6)
    ds = (0.02, 0.04, 0.1, 0.2, 0.4)
    cands = [render(StandinPatch(**BASE, fc_decay_s=d)) for d in ds]
    mis = [abs(np.log(d / 0.06)) for d in ds]
    for fn in (ta, logmel_l1, mss_l1):
        assert spearmanr(mis, [fn(c, tgt) for c in cands])[0] > 0.85


def test_a_static_hedge_scores_far_worse_than_a_near_true_envelope():
    tgt = render(StandinPatch(**BASE, fc_decay_s=0.06))
    hedge = render(StandinPatch(**{**BASE, "fc_env_oct": 0.0, "fc_hz": 500.0 * 2 ** 1.25}))
    near = render(StandinPatch(**BASE, fc_decay_s=0.07))
    ta = TwoAnchorObjective()
    assert ta(hedge, tgt) > 5.0 * ta(near, tgt)


def test_onset_snapping_removes_a_time_offset_that_defeats_plain_log_mel():
    cand = render(StandinPatch(**BASE, fc_decay_s=0.06))
    shifted = render(StandinPatch(**BASE, fc_decay_s=0.06), onset_s=0.06, note_dur=0.5)
    snapped = TwoAnchorObjective()(cand, shifted, expected_onset_s=0.035)      # MIDI onset 25 ms early
    unsnapped = TwoAnchorObjective(snap=False)(cand, shifted, expected_onset_s=0.0)
    assert snapped < 1.0 and snapped < 0.2 * unsnapped and snapped < 0.5 * logmel_l1(cand, shifted)


def test_harmonic_envelope_distance_grows_with_the_cutoff_mismatch():
    tgt = render(StandinPatch(f0=82.4, fc_hz=1000.0, q=4.0, phase_seed=1), note_dur=0.7)
    d = [harmonic_envelope_distance(render(StandinPatch(f0=82.4, fc_hz=fc, q=4.0, phase_seed=1), note_dur=0.7), tgt, 82.4)
         for fc in (1000.0, 1400.0, 2000.0, 4000.0)]
    assert d[0] == pytest.approx(0.0, abs=1e-6) and all(a < b for a, b in zip(d, d[1:]))


# ----------------------------------------------------------------------------------------------------------------------
# front end
# ----------------------------------------------------------------------------------------------------------------------
def _thin_bands(n_fft, n_mels, fmin, fmax):
    import torchaudio.functional as AF
    fb = AF.melscale_fbanks(n_fft // 2 + 1, fmin, fmax, n_mels, SR, norm="slaney", mel_scale="slaney")
    return int(((fb > 0).sum(dim=0) <= 1).sum())


def test_dual_mel_shapes_range_and_no_redundant_low_bands():
    s, l = DualMel()(torch.randn(2, 35280) * 0.1)
    assert tuple(s.shape) == (2, 32, 321) and tuple(l.shape) == (2, 96, 81)
    assert float(s.max()) <= 1.0 + 1e-6 and float(l.min()) >= -1.0 - 1e-6
    assert _thin_bands(512, 32, 100.0, 16000.0) == 0 and _thin_bands(4096, 96, 30.0, 16000.0) == 0


def test_the_current_front_end_has_redundant_low_bands():
    """Review B4 / research report check: 28 of 128 bands are fed by at most one FFT bin (torchaudio Slaney filterbank,
    which is not ExactGpuMel itself)."""
    assert _thin_bands(1024, 128, 20.0, 16000.0) >= 20


# ----------------------------------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------------------------------
POOLS = ("temporal_asp", "temporal_segments", "temporal_queries")


@needs_models
@pytest.mark.parametrize("kind", POOLS)
def test_temporal_encoder_shape_tokens_and_gradients(kind):
    torch.manual_seed(0)
    enc = models_v4.make_encoder_v4(kind, 512)
    mel = torch.randn(3, 1, 128, 81)
    out = enc(mel)
    assert tuple(out.shape) == (3, 512) and enc.n_tokens == 41 and enc.n_att == 2 and enc.sus_start == 10
    assert torch.allclose(enc.eval()(mel), enc.eval()(mel.squeeze(1)))          # 3-D and 4-D input agree
    enc.train()
    enc(mel).pow(2).sum().backward()
    assert all(p.grad is not None for p in enc.parameters())
    # Not time-invariant. This is a sanity check only: on random-noise mels at initialisation the v3 conv encoder reacts
    # to a flip at least as much (ratio about 1.0 against 0.4 to 0.6 here), so it is NOT evidence that v4 uses temporal
    # order better. That question needs training: probe_encoders_toy_v4.py.
    enc.eval()
    with torch.no_grad():
        other = torch.randn_like(mel)
        ratio = float((enc(mel.flip(-1)) - enc(mel)).norm() / (enc(other) - enc(mel)).norm())
    assert ratio > 0.2


@needs_models
def test_flow_v4_matches_the_v3_interface_and_film_starts_as_identity():
    torch.manual_seed(0)
    m = models_v4.build_model_v4(hidden_dim=64, num_layers=2, cond_noise=True, encoder="temporal_segments", film=True,
                                 enc_kwargs=dict(d_model=32, n_heads=2))
    mel = torch.randn(4, 1, 128, 81)
    x, t, tau = torch.rand(4, m.param_dim), torch.rand(4), torch.ones(4)
    y = m(x, t, mel=mel, tau=tau)
    assert y.shape == (4, m.param_dim)
    film, m.film = m.film, None
    assert torch.allclose(y, m(x, t, mel=mel, tau=tau), atol=1e-6)
    m.film = film
    # one flow-matching loss step, as flow_loss does it
    x1, x0 = torch.rand(4, m.param_dim), torch.randn(4, m.param_dim)
    xt, target = m.path(x0, x1, t)
    loss = (m(xt, t, mel=mel, tau=tau) - target).abs().mean()
    loss.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    m.eval()
    s = m.sample(mel, num_steps=3)
    assert s.shape == (4, m.param_dim) and float(s.min()) >= 0.0 and float(s.max()) <= 1.0


@needs_models
def test_build_model_v4_conv_is_the_v3_flow_and_state_dicts_do_not_cross_load():
    v3 = models_v4.build_model_v4(hidden_dim=64, num_layers=2, encoder="conv")
    assert type(v3) is models_v3.FlowMatchingResMLP
    v4 = models_v4.build_model_v4(hidden_dim=64, num_layers=2, encoder="temporal_asp", enc_kwargs=dict(d_model=32, n_heads=2))
    with pytest.raises(RuntimeError):
        v3.load_state_dict(v4.state_dict())


@needs_models
@pytest.mark.parametrize("kind,film", [("temporal_asp", False), ("temporal_segments", True), ("temporal_queries", False)])
def test_load_inverter_v4_round_trip(tmp_path, kind, film):
    torch.manual_seed(1)
    m = models_v4.build_model_v4(hidden_dim=64, num_layers=2, cond_noise=True, encoder=kind, film=film,
                                 enc_kwargs=dict(d_model=32, n_heads=2)).eval()
    p = tmp_path / "flow.pt"
    torch.save({"model_state": m.state_dict(), "args": {"model_type": "flow"}}, p)
    r = models_v4.load_inverter_v4(p, v4_config=m.v4_config)
    mel, x, t = torch.randn(2, 1, 128, 81), torch.rand(2, m.param_dim), torch.rand(2)
    assert torch.allclose(m(x, t, mel=mel, tau=torch.ones(2)), r(x, t, mel=mel, tau=torch.ones(2)), atol=1e-6)


@needs_models
def test_load_inverter_v4_inference_without_a_config_and_v3_passthrough(tmp_path):
    torch.manual_seed(2)
    m = models_v4.build_model_v4(hidden_dim=64, num_layers=2, encoder="temporal_segments", enc_kwargs=dict(d_model=256)).eval()
    p = tmp_path / "v4.pt"
    torch.save({"model_state": m.state_dict(), "args": {}}, p)
    r = models_v4.load_inverter_v4(p)
    assert r.v4_config["encoder"] == "temporal_segments" and r.film is None
    v3 = models_v3.build_model("flow", 64, 2).eval()
    p3 = tmp_path / "v3.pt"
    import param_codec_v3 as codec
    torch.save({"model_state": v3.state_dict(),
                "args": {"model_type": "flow", "param_encoding": codec.ENCODING_V2, "time_scale": 1000.0}}, p3)
    assert type(models_v4.load_inverter_v4(p3)) is models_v3.FlowMatchingResMLP


@needs_models
def test_parameter_budget_is_modest():
    v3 = models_v3.build_model("flow", 512, 8, cond_noise=True)
    v4 = models_v4.build_model_v4(hidden_dim=512, num_layers=8, cond_noise=True, encoder="temporal_asp")
    extra = sum(p.numel() for p in v4.parameters()) - sum(p.numel() for p in v3.parameters())
    assert 0 < extra < 0.35 * sum(p.numel() for p in v3.parameters())


# ----------------------------------------------------------------------------------------------------------------------
# inference glue
# ----------------------------------------------------------------------------------------------------------------------
def test_rank_candidates_puts_the_near_true_candidate_first_and_explains_itself():
    from inference_v4 import candidate_scores, rank_candidates
    f0 = 55.0
    base = dict(f0=f0, fc_hz=1000.0, fc_env_oct=2.0, fc_decay_s=0.08, q=6.0, amp_decay_s=0.3, amp_sustain=0.4, phase_seed=1)
    tgt = render(StandinPatch(**base))
    cands = [render(StandinPatch(**{**base, "q": 0.7})),                           # no resonance
             render(StandinPatch(**{**base, "fc_decay_s": 0.09})),                 # near true
             render(StandinPatch(**{**base, "fc_env_oct": 0.0})),                  # static filter
             render(StandinPatch(**base, fm_ratio=1.37, fm_depth=1.0))]            # buzz
    order, rows = rank_candidates(cands, tgt, f0, weights={"two_anchor": 1.0, "height_delta_db": 0.5, "comb_delta": 5.0},
                                  explain=True)
    assert order[0] == 1
    assert rows[0]["height_delta_db"] > rows[1]["height_delta_db"]                 # the resonance term sees Q = 0.7
    assert rows[3]["comb_delta"] > 10 * rows[1]["comb_delta"]                      # the buzz term sees the FM
    assert candidate_scores(tgt, tgt, f0)["two_anchor"] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        rank_candidates(cands, tgt, f0, weights={"made_up": 1.0})


# ----------------------------------------------------------------------------------------------------------------------
# integration with the existing Refiner (needs librosa, which refine_v3 imports through reference_metrics)
# ----------------------------------------------------------------------------------------------------------------------
def test_two_anchor_objective_drops_into_the_existing_refiner_and_is_restored():
    try:
        import refine_v3
        import probe_refine_v4 as pr
    except ImportError:
        if os.environ.get("REQUIRE_ALL") == "1":
            raise
        pytest.skip("refine_v3 dependencies (librosa) not installed")
    original = refine_v3.objective
    target = pr.render_axes(pr.TRUTH)
    start = {**pr.TRUTH, "cutoff": 0.65, "aeg_decay": 0.35}
    with pr.use_objective(pr.objective_for("two_anchor")):
        assert refine_v3.objective is not original
        r = refine_v3.Refiner(pr.render_axes, target)
        out, final = r.refine(start, axes=[("cutoff", 0.15), ("aeg_decay", 0.15)], passes=1, n_coarse=3, n_fine=2)
    assert refine_v3.objective is original                                         # restored
    assert set(out) == set(start) and r.n_renders > 5
    assert final <= r.trace[0][2] + 1e-9                                           # the Refiner only accepts improvements
