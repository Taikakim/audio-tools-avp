"""CMuon-style AdaLN chunking for FusionOpt (arXiv 2608.02502).

The fused AdaLN modulation projection (SA3's `global_cond_embedder.2`) is a
[k*dim, dim] tensor whose k output row-blocks emit functionally-distinct
scale/shift/gate signals (self + ff, k=6) but are fused into one matmul for
efficiency. Running Newton-Schulz on the WHOLE fused tensor makes its columns
orthonormal ACROSS the k blocks — i.e. it couples the k independent subspaces:
each individual d x d row-block then has singular values ~1/sqrt(k), not 1, so
no single modulation sub-matrix is orthogonalised. CMuon's fix is to chunk the
update along the output dim into its k sub-blocks BEFORE NS5, orthogonalise each
independently, and concatenate back.

The repo already does exactly this for fused attention up-projections via
`split_qkv` / `spectral_split` / `_block_count` (the optimiser's chunked-NS5 path
at fusion_opt.py step 5). This test pins the NEW `split_adaln` extension that
routes the AdaLN emitter into the SAME machinery:

  (a) build_fusion_param_groups(split_adaln=True) tags the AdaLN param with
      block_count == k, and leaves it 1 (no spectral_split key) without the flag;
  (b) on a synthetic fused [k*d, d] matrix, chunked orthogonalisation makes each
      d x d sub-block individually orthogonal-ish (per-block singular values ~1
      after NS5) whereas whole-tensor NS5 does NOT (per-block SVs ~1/sqrt(k));
  (c) non-AdaLN spectral params (e.g. project_out) are unaffected (block_count 1),
      and the flag composes through a real FusionOpt.step without error.

Runnable directly (pytest is not installed in sat-venv):
    sat-venv/bin/python tests/test_cmuon_adaln.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from stable_audio_tools.training.fusion_opt import FusionOpt, newton_schulz_5
from stable_audio_tools.training.fusion_groups import (
    build_fusion_param_groups,
    _adaln_block_count,
    _block_count,
)

DIM = 128   # >= MIN_SPECTRAL_DIM so the fused AdaLN Linear routes spectral
K = 6       # scale/shift/gate x {self, ff} — the SA3 fused count


class _FakeDiT(nn.Module):
    """Minimal module reproducing the SA3 param NAMES the routing keys on:
      transformer.global_cond_embedder.2.weight  -> [K*DIM, DIM]  (fused AdaLN emitter)
      transformer.global_cond_embedder.0.weight  -> [DIM, 64]     (input Linear, NOT the emitter)
      transformer.project_out.weight             -> [DIM, DIM]    (non-AdaLN spectral)
    """

    def __init__(self, dim=DIM, k=K):
        super().__init__()
        inner = nn.Module()
        inner.global_cond_embedder = nn.Sequential(
            nn.Linear(64, dim),          # .0  -> [dim, 64]  (min-dim 64 -> scalar)
            nn.SiLU(),                   # .1
            nn.Linear(dim, k * dim),     # .2  -> [k*dim, dim] fused AdaLN emitter
        )
        inner.project_out = nn.Linear(dim, dim, bias=False)  # [dim, dim] spectral, NOT fused
        self.transformer = inner


def _find(names, needle):
    for i, n in enumerate(names):
        if needle in n:
            return i
    raise AssertionError(f"{needle!r} not found in {names}")


def test_split_adaln_tags_block_count_k():
    """(a) split_adaln=True tags the AdaLN emitter with spectral_split == K; the
    non-AdaLN spectral param (project_out) stays 1."""
    model = _FakeDiT()
    groups = build_fusion_param_groups(model, split_adaln=True)
    spectral = next(g for g in groups if g["group_type"] == "spectral")
    names = spectral["param_names"]
    splits = spectral["spectral_split"]

    adaln_i = _find(names, ".global_cond_embedder.2.")
    proj_i = _find(names, ".project_out.")
    assert splits[adaln_i] == K, f"AdaLN block_count {splits[adaln_i]} != {K}"
    assert splits[proj_i] == 1, f"project_out block_count {splits[proj_i]} != 1"
    # every block count is a positive int
    assert all(isinstance(s, int) and s >= 1 for s in splits), splits


def test_off_by_default_is_byte_identical():
    """(a) Without the flag there is NO spectral_split key (effective block_count 1
    everywhere) — the routing is byte-identical to pre-CMuon."""
    model = _FakeDiT()
    groups = build_fusion_param_groups(model)  # no flags
    spectral = next(g for g in groups if g["group_type"] == "spectral")
    assert "spectral_split" not in spectral, "spectral_split must be absent when both flags off"

    # And the helper itself: AdaLN emitter -> K, non-AdaLN -> 1.
    embedder = model.transformer.global_cond_embedder
    emitter_w = embedder[2].weight            # [K*DIM, DIM]
    input_w = embedder[0].weight              # [DIM, 64]
    proj_w = model.transformer.project_out.weight
    assert _adaln_block_count("m.transformer.global_cond_embedder.2.weight", emitter_w) == K
    assert _adaln_block_count("m.transformer.global_cond_embedder.0.weight", input_w) == 1
    assert _adaln_block_count("m.transformer.project_out.weight", proj_w) == 1


def _block_svmeans(mat, k):
    """Split [k*d, d] into k d-row blocks, return each block's mean singular value."""
    d = mat.shape[0] // k
    means = []
    for b in range(k):
        blk = mat[b * d:(b + 1) * d].float()
        means.append(float(torch.linalg.svdvals(blk).mean()))
    return means


def test_chunked_vs_whole_orthogonalisation():
    """(b) Chunked NS5 orthogonalises each d x d sub-block (per-block SVs ~1);
    whole-tensor NS5 orthogonalises ACROSS blocks (per-block SVs ~1/sqrt(k)),
    proving the subspace coupling the chunking is meant to remove."""
    torch.manual_seed(0)
    d = DIM
    m = torch.randn(K * d, d)

    # Chunked path — exactly what fusion_opt.py step-5 does for _nblk>1: NS5 each
    # d-row block, aspect scale (max(1, od/idim))**0.5 == 1 for square blocks, cat.
    bh = m.shape[0] // K
    chunked = torch.cat([newton_schulz_5(m[b * bh:(b + 1) * bh]).float() for b in range(K)], dim=0)
    # Whole-tensor path — _nblk==1: NS5 the entire [k*d, d] fused tensor.
    whole = newton_schulz_5(m).float()

    chunked_means = _block_svmeans(chunked, K)
    whole_means = _block_svmeans(whole, K)

    # NS5 pulls SVs into ~[0.7, 1.3]; chunked => every sub-block is orthogonal-ish.
    assert all(0.7 <= s <= 1.3 for s in chunked_means), \
        f"chunked per-block SV means not ~1: {chunked_means}"
    # Whole-tensor => each sub-block carries ~1/k of the energy (SVs ~1/sqrt(k)),
    # i.e. NO individual modulation sub-matrix is orthogonalised.
    expected = 1.0 / (K ** 0.5)
    assert all(s < 0.6 for s in whole_means), \
        f"whole-tensor per-block SV means unexpectedly ~1 (no coupling shown): {whole_means}"
    assert all(abs(s - expected) < 0.15 for s in whole_means), \
        f"whole-tensor per-block SV means {whole_means} not ~1/sqrt(k)={expected:.3f}"


def test_non_adaln_unaffected_and_step_runs():
    """(c) Non-AdaLN params keep block_count 1; the qkv path is untouched; and the
    split_adaln group runs a real FusionOpt.step through the chunked path w/o error."""
    # _block_count (the qkv helper) never fires on the AdaLN name — the two paths
    # are independent detectors.
    model = _FakeDiT()
    emitter_w = model.transformer.global_cond_embedder[2].weight
    assert _block_count("m.transformer.global_cond_embedder.2.weight", emitter_w, DIM) == 1, \
        "qkv _block_count must NOT match the AdaLN emitter"

    # End-to-end: build the split_adaln group, run one optimiser step. The AdaLN
    # param (spectral_split==K) must flow through step-5's chunked-NS5 branch and
    # produce a finite, changed update — reusing the existing path, no new NS5.
    groups = build_fusion_param_groups(model, split_adaln=True)
    for g in groups:
        g["lr"] = 1e-2
    opt = FusionOpt(groups, lr=1e-2, components={"ns5", "normuon", "sf"}, hot_dtype="fp32")

    before = emitter_w.detach().clone()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt.step()
    opt.eval()  # write the averaged iterate x into params

    assert torch.isfinite(emitter_w).all(), "AdaLN update produced non-finite values"
    assert not torch.equal(before, emitter_w.detach()), "AdaLN param did not update"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
