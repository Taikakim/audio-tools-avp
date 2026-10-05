"""Which audio distance agrees best with HUMAN timbre dissimilarity? (decides the refinement objective)

Uses Ben Hayes' timbre-dissimilarity-metrics (/home/kim/Projects/timbre-dissimilarity-metrics):
21 published listening-test datasets (Grey 1977 ... Zacharakis 2014), each a set of tones plus a
matrix of mean human dissimilarity ratings. For every candidate distance we compute the full
pairwise distance matrix of each dataset's tones and its Spearman correlation with the human
matrix (the Mantel statistic), plus timbremetrics' triplet agreement.

Caveat, stated up front: these are mostly ACOUSTIC instrument tones (orchestral, percussive,
FM approximations), not synth bass, so this measures general timbre alignment. Our JEPA was
trained only on Surge bass, so it is out of domain here; the hand-made distances are not.

Candidates:
  jepa_ea       L2 between Synth-JEPA audio embeddings z_a (first 0.8 s, mono, model mel)
  stft_ours     audio_utils.MultiScaleSTFTLoss (first 0.8 s) -- the current reranker term
  mss_ref       reference_metrics.mss (paper MSS, full tone)
  wmfcc_ref     reference_metrics.wmfcc (paper wMFCC, full tone)
  rms_env       1 - reference_metrics.rms_env_cos
  logmel_l1     L1 between our model's 0.8 s log-mel inputs (the paper's best-of-k selector)

Run (SAT venv):  sat-venv/bin/python scripts/synth_inversion/timbre_alignment_check.py [--jepa PATH]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/kim/Projects/timbre-dissimilarity-metrics")

import timbremetrics  # noqa: E402
from timbremetrics.utils import load_dissimilarity_matrix  # noqa: E402

import reference_metrics as rm  # noqa: E402
from audio_utils import MultiScaleSTFTLoss, make_mel_spec  # noqa: E402
from surge_spec import AUDIO_LEN, SAMPLE_RATE  # noqa: E402

DEFAULT_JEPA = "/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64/jepa_latest.pt"


def prep(audio, sr):
    import librosa
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y / (np.max(np.abs(y)) + 1e-9)
    short = np.zeros(AUDIO_LEN, np.float32)
    short[:min(AUDIO_LEN, len(y))] = y[:AUDIO_LEN]
    return y, short


def pairwise(items, fn):
    n = len(items)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fn(items[i], items[j])
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default=DEFAULT_JEPA)
    ap.add_argument("--out", default="/run/media/kim/Mantu/surge_200k_models/timbre_alignment_result.json")
    args = ap.parse_args()

    jepa = norm = None
    if args.jepa and os.path.exists(args.jepa):
        from synth_jepa_search import load_synth_jepa
        jepa, norm = load_synth_jepa(args.jepa)
    stft = MultiScaleSTFTLoss()

    candidates = ["stft_ours", "mss_ref", "wmfcc_ref", "rms_env", "logmel_l1"] + (["jepa_ea"] if jepa else [])
    per_ds = {}
    for ds in timbremetrics.list_datasets():
        target = load_dissimilarity_matrix(ds).astype(float)
        target = target + target.T
        tones = timbremetrics.get_audio(ds)
        full, short = zip(*[prep(t["audio"], t["sample_rate"]) for t in tones])
        mels = [make_mel_spec(s) for s in short]
        mats = {
            "stft_ours": pairwise(short, lambda a, b: stft(a, b)),
            "mss_ref": pairwise(full, rm.mss),
            "wmfcc_ref": pairwise(full, rm.wmfcc),
            "rms_env": pairwise(full, lambda a, b: 1.0 - rm.rms_env_cos(a, b)),
            "logmel_l1": pairwise(mels, lambda a, b: float(np.mean(np.abs(a - b)))),
        }
        if jepa:
            with torch.no_grad():
                z = jepa.encode_audio(norm.normalize(torch.from_numpy(np.stack(mels)))).numpy()
            mats["jepa_ea"] = pairwise(list(z), lambda a, b: float(np.linalg.norm(a - b)))
        iu = np.triu_indices(len(tones), 1)
        per_ds[ds] = {"n": len(tones)}
        for name, D in mats.items():
            per_ds[ds][name] = float(spearmanr(D[iu], target[iu]).correlation)
        print(ds.ljust(36), " ".join(f"{k}={per_ds[ds][k]:+.2f}" for k in candidates), flush=True)

    summary = {k: {"mean_spearman": round(float(np.mean([v[k] for v in per_ds.values()])), 3),
                   "wins": sum(1 for v in per_ds.values() if max(candidates, key=lambda c: v[c]) == k)}
               for k in candidates}
    print(json.dumps(summary, indent=1))
    json.dump({"purpose": "rank candidate audio distances by agreement with human timbre dissimilarity "
                          "(21 datasets, timbre-dissimilarity-metrics) to choose the inference-refinement objective",
               "jepa": args.jepa, "summary": summary, "per_dataset": per_ds}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
