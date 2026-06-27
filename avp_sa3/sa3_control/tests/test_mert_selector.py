"""Unit tests for sa3_control.mert_selector -- pure scoring math with synthetic
embeddings/CE. No model, no GPU. The MERTEmbedder/audiobox_ce model path is left
for an integration session (see 'gaps')."""
import numpy as np
import pytest

from sa3_control.mert_selector import (
    RewardWeights,
    best_index,
    best_of,
    composite_score,
    cosine_sim,
    melody_band_penalty,
    score_continuation,
    score_from_embeddings,
)


# --------------------------------------------------------------------------- #
# Fakes: a deterministic embedder + CE that bypass MERT/Audiobox entirely.
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Maps a 1-D wav array to {'mid','upper'} embeddings deterministically.
    We encode the desired (mid, upper) vectors directly in the wav as two halves."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def embed(self, wav, sr):
        self.calls.append((np.asarray(wav).shape, sr))
        v = np.asarray(wav, dtype=np.float64).ravel()
        mid = v[: self.dim]
        upper = v[self.dim : 2 * self.dim]
        return {"mid": mid, "upper": upper}


def make_wav(mid, upper):
    return np.concatenate([np.asarray(mid, float), np.asarray(upper, float)])


# --------------------------------------------------------------------------- #
# cosine_sim
# --------------------------------------------------------------------------- #
def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_sim([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_scale_invariant():
    assert cosine_sim([1.0, 2.0], [3.0, 6.0]) == pytest.approx(1.0)


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_sim([1.0, 2.0], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# melody band penalty + composite score
# --------------------------------------------------------------------------- #
def test_band_penalty_zero_at_center():
    assert melody_band_penalty(0.675, 0.675) == pytest.approx(0.0)


def test_band_penalty_symmetric():
    assert melody_band_penalty(0.5, 0.675) == pytest.approx(melody_band_penalty(0.85, 0.675))


def test_composite_score_formula():
    w = RewardWeights(w_ce=1.0, w_rhythm=2.0, w_melody=3.0, band_center=0.675)
    # score = 1*7 + 2*0.8 - 3*|0.6-0.675| = 7 + 1.6 - 0.225 = 8.375
    assert composite_score(7.0, 0.8, 0.6, weights=w) == pytest.approx(8.375)


def test_composite_melody_penalized_both_sides():
    """melody_sim ABOVE the band (the loop) is penalized just like below it."""
    w = RewardWeights(w_ce=0.0, w_rhythm=0.0, w_melody=1.0, band_center=0.675)
    below = composite_score(0.0, 0.0, 0.55, weights=w)
    at_center = composite_score(0.0, 0.0, 0.675, weights=w)
    above = composite_score(0.0, 0.0, 0.80, weights=w)
    assert at_center == pytest.approx(0.0)
    assert below < at_center and above < at_center
    assert below == pytest.approx(above)  # symmetric band of +-0.125


# --------------------------------------------------------------------------- #
# score_from_embeddings
# --------------------------------------------------------------------------- #
def test_score_from_embeddings_keys_and_values():
    w = RewardWeights()
    prev_mid = np.array([1.0, 0.0, 0.0, 0.0])
    prev_upper = np.array([0.0, 1.0, 0.0, 0.0])
    cand_mid = np.array([1.0, 0.0, 0.0, 0.0])  # identical groove -> rhythm_sim 1
    cand_upper = np.array([0.0, 1.0, 0.0, 0.0])  # identical melody -> melody_sim 1
    out = score_from_embeddings(prev_mid, prev_upper, cand_mid, cand_upper, 7.0, weights=w)
    assert set(out) == {"ce", "rhythm_sim", "melody_sim", "score"}
    assert out["rhythm_sim"] == pytest.approx(1.0)
    assert out["melody_sim"] == pytest.approx(1.0)
    # score = 1*7 + 1*1 - 1*|1-0.675|
    assert out["score"] == pytest.approx(7.0 + 1.0 - abs(1.0 - w.band_center))


def test_score_from_embeddings_no_reference_is_ce_only():
    w = RewardWeights()
    out = score_from_embeddings(None, None, np.ones(4), np.ones(4), 6.5, weights=w)
    assert out["rhythm_sim"] == 0.0
    assert out["melody_sim"] == pytest.approx(w.band_center)  # zero melody penalty
    assert out["score"] == pytest.approx(6.5)  # CE-only


# --------------------------------------------------------------------------- #
# best_index
# --------------------------------------------------------------------------- #
def test_best_index_argmax():
    scores = [{"score": 1.0}, {"score": 5.0}, {"score": 3.0}]
    assert best_index(scores) == 1


def test_best_index_ties_lowest():
    scores = [{"score": 2.0}, {"score": 2.0}]
    assert best_index(scores) == 0


def test_best_index_empty_raises():
    with pytest.raises(ValueError):
        best_index([])


# --------------------------------------------------------------------------- #
# score_continuation (full path through injected fakes)
# --------------------------------------------------------------------------- #
def test_score_continuation_with_fakes():
    emb = FakeEmbedder(dim=2)
    prev = make_wav([1.0, 0.0], [0.0, 1.0])
    cand = make_wav([1.0, 0.0], [1.0, 0.0])  # same groove, orthogonal melody
    out = score_continuation(
        prev, cand, embedder=emb, ce_fn=lambda w: 8.0, weights=RewardWeights()
    )
    assert out["ce"] == 8.0
    assert out["rhythm_sim"] == pytest.approx(1.0)
    assert out["melody_sim"] == pytest.approx(0.0)
    # both prev and cand embedded
    assert len(emb.calls) == 2


def test_score_continuation_no_prev_is_ce_only():
    emb = FakeEmbedder(dim=2)
    cand = make_wav([1.0, 0.0], [1.0, 0.0])
    out = score_continuation(None, cand, embedder=emb, ce_fn=lambda w: 5.0)
    assert out["score"] == pytest.approx(5.0)
    assert len(emb.calls) == 1  # only the candidate embedded


# --------------------------------------------------------------------------- #
# best_of: the headline selector
# --------------------------------------------------------------------------- #
def test_best_of_picks_in_band_over_loop():
    """Identity weight scenario: a candidate that loops (melody_sim ~1) should lose
    to one that develops within the band, when CE/rhythm are equal."""
    emb = FakeEmbedder(dim=2)
    w = RewardWeights(w_ce=1.0, w_rhythm=1.0, w_melody=4.0, band_center=0.7)
    prev = make_wav([1.0, 0.0], [1.0, 0.0])
    loop = make_wav([1.0, 0.0], [1.0, 0.0])        # melody_sim 1.0 (literal repeat)
    develop = make_wav([1.0, 0.0], [0.7, 0.714])   # melody_sim ~ in-band
    bi, scores = best_of(
        prev, [loop, develop], embedder=emb, ce_fn=lambda x: 7.0, weights=w,
        return_scores=True,
    )
    assert scores[0]["melody_sim"] == pytest.approx(1.0)
    assert abs(scores[1]["melody_sim"] - 0.7) < 0.05
    assert bi == 1  # the developing candidate wins


def test_best_of_reference_embedded_once():
    emb = FakeEmbedder(dim=2)
    prev = make_wav([1.0, 0.0], [0.0, 1.0])
    cands = [make_wav([1.0, 0.0], [0.0, 1.0]) for _ in range(3)]
    best_of(prev, cands, embedder=emb, ce_fn=lambda x: 6.0)
    # 1 reference + 3 candidates = 4 embeds (not 6)
    assert len(emb.calls) == 4


def test_best_of_ce_drives_first_window():
    emb = FakeEmbedder(dim=2)
    ces = {0: 6.0, 1: 9.0, 2: 7.0}
    counter = {"i": -1}

    def ce_fn(_):
        counter["i"] += 1
        return ces[counter["i"]]

    cands = [make_wav([1.0, 0.0], [1.0, 0.0]) for _ in range(3)]
    bi, scores = best_of(None, cands, embedder=emb, ce_fn=ce_fn, return_scores=True)
    assert bi == 1  # highest CE wins when there is no reference
    assert all(s["rhythm_sim"] == 0.0 for s in scores)


def test_best_of_returns_int_index():
    emb = FakeEmbedder(dim=2)
    cands = [make_wav([1.0, 0.0], [0.0, 1.0])]
    bi = best_of(make_wav([1.0, 0.0], [0.0, 1.0]), cands, embedder=emb, ce_fn=lambda x: 5.0)
    assert isinstance(bi, int) and bi == 0


def test_best_of_empty_raises():
    with pytest.raises(ValueError):
        best_of(None, [], embedder=FakeEmbedder())


def test_best_of_scores_carry_paths_for_str_inputs():
    emb = FakeEmbedder(dim=2)
    # str candidates would normally hit torchaudio.load; here we only check that the
    # path is threaded through. Use the array form to avoid disk I/O, and assert None.
    cands = [make_wav([1.0, 0.0], [0.0, 1.0])]
    _, scores = best_of(None, cands, embedder=emb, ce_fn=lambda x: 5.0, return_scores=True)
    assert scores[0]["path"] is None  # array inputs -> no path
