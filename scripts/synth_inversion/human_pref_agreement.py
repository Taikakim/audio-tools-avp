# Owner: KUANG (created 2026-10-20)
"""Which automatic metric agrees with Kim's ears? Reads the listening ranks (human_preferences.json:
{stem_id: {model: rank, 1 = best}}) and the eval summary (consensus_eval_summary.json) of the same run, and reports:
  - a Friedman test over models (stems where every model was ranked): do the models differ at all?
  - per metric: the Spearman correlation between the human rank and the metric within each stem (oriented so that
    > 0 means "the metric agrees"), its median / mean over stems, a Wilcoxon signed-rank p against 0, and how often
    the metric's best model is the human's first choice (chance = 1 / number of models).

Computed on the foundation run's page (24 stems, 5 models) by hand from HUMAN_EVALUATION_SYNTHESIS.md on
2026-10-20: Friedman p = 0.73 (no model is distinguishable); phrase MSS rho median 0.05 (p = 0.35, no agreement);
|log2 centroid ratio| rho median 0.36 (p = 0.0004, 18 of 21 stems agree). The score that picks the winner does not
track what the listener hears; spectral balance does.

--recompute LISTEN_DIR adds the KUANG 2026-10-20 metrics (spectral_emd_oct, band_dists, resonance) computed on the
exact audio on the page (LISTEN_DIR/{stem}__{model}.wav against {stem}__real_phrase.wav), so the new metrics are
tested on the sounds that were ranked, not on a new eval run. Resonance needs the notes: --midi_dir (the
eval's midi_muscriptor directory) — skipped without it.

  python human_pref_agreement.py RUN/human_preferences.json RUN/consensus_eval_summary.json \\
      --recompute RUN/listen --max_eval_freq 4000 --out RUN/pref_agreement.md
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _centroid_err(m, stem):
    c, r = m.get("centroid_hz"), stem.get("real_centroid_hz")
    if c is None or r is None or c <= 0 or r <= 0:
        return None
    return abs(float(np.log2(c / r)))


def _res_err(m, stem):
    a, b = m.get("resonance_db"), stem.get("real_resonance_db")
    return None if a is None or b is None else abs(a - b)


# name -> (stem, model entry) -> value, lower = closer to the real stem
METRICS = {
    "phrase_mss": lambda m, s: m.get("phrase_mss"),
    "mel_ssim_dist": lambda m, s: m.get("mel_ssim_dist", None if m.get("mel_ssim") is None else 1 - m["mel_ssim"]),
    "abs_log2_centroid_ratio": _centroid_err,
    "spectral_emd_oct": lambda m, s: m.get("spectral_emd_oct"),
    "band_sub": lambda m, s: (m.get("band_dists") or {}).get("sub"),
    "band_body": lambda m, s: (m.get("band_dists") or {}).get("body"),
    "band_air": lambda m, s: (m.get("band_dists") or {}).get("air"),
    "abs_resonance_db_err": _res_err,
    "abs_width_delta": lambda m, s: None if m.get("width_delta") is None else abs(m["width_delta"]),
}


def recompute(summary, listen_dir, max_eval_freq=None, midi_dir=None):
    """Add spectral_emd_oct / band_dists / resonance to the summary, from the wavs on the page."""
    import soundfile as sf
    import audio_metrics_v6 as am6
    band = (lambda x: am6.lowpass(x, am6.SR, max_eval_freq)) if max_eval_freq else (lambda x: x)
    bands = tuple(b for b in am6.BANDS_HZ if b[1] < (max_eval_freq or 4000.0))
    for stem in summary:
        sid = stem["stem_id"]
        rp = os.path.join(listen_dir, f"{sid}__real_phrase.wav")
        if not os.path.exists(rp):
            continue
        real, sr = sf.read(rp)
        real = band(real.mean(axis=1) if real.ndim > 1 else real)
        real = real / (np.max(np.abs(real)) + 1e-7)
        notes = None
        if midi_dir:
            from eval_real_stems_consensus_v6 import extract_notes_from_midi, select_representative_notes
            mp = os.path.join(midi_dir, f"{sid}_muscriptor_FIX.mid")
            mp = mp if os.path.exists(mp) else os.path.join(midi_dir, f"{sid}_muscriptor.mid")
            if os.path.exists(mp):
                notes = select_representative_notes(extract_notes_from_midi(mp, max_dur=len(real) / sr))
                scale = stem.get("midi_scale", 1.0) or 1.0
                notes = [dict(n, start=n["start"] * scale, dur=n["dur"] * scale) for n in notes]
                e, f = am6.resonance_over_notes(real, notes, sr)
                stem["real_resonance_db"] = None if not np.isfinite(e) else float(e)
        for name, m in stem.get("models", {}).items():
            p = os.path.join(listen_dir, m.get("audio_file", f"{sid}__{name}.wav"))
            if not os.path.exists(p):
                continue
            y, _ = sf.read(p)
            y = band(y.mean(axis=1) if y.ndim > 1 else y)
            y = y / (np.max(np.abs(y)) + 1e-7)
            n = min(len(y), len(real))
            m["spectral_emd_oct"] = am6.spectral_emd_oct(y[:n], real[:n], sr, fmax=min(4000.0, max_eval_freq or 4000.0))
            m["band_dists"] = am6.band_dists(y[:n], real[:n], sr, bands=bands)
            if notes:
                e, _ = am6.resonance_over_notes(y, notes, sr)
                m["resonance_db"] = None if not np.isfinite(e) else float(e)
    return summary


def agreement(prefs, summary, metrics=METRICS):
    """-> (report lines, results dict)."""
    by_id = {s["stem_id"]: s for s in summary}
    stems = [sid for sid in prefs if sid in by_id]
    models = sorted({m for sid in stems for m in prefs[sid]})
    lines, res = [], {"models": models, "metrics": {}}
    full = [sid for sid in stems if all(m in prefs[sid] for m in models)]
    if len(full) >= 2 and len(models) >= 3:
        R = np.array([[prefs[sid][m] for m in models] for sid in full], float)
        fr = stats.friedmanchisquare(*R.T)
        res["friedman"] = {"n_stems": len(full), "stat": float(fr.statistic), "p": float(fr.pvalue),
                           "mean_rank": dict(zip(models, R.mean(0).round(3).tolist())),
                           "wins": dict(zip(models, (R == 1).sum(0).tolist()))}
        lines += [f"## Do the models differ? (Friedman, {len(full)} stems with all {len(models)} models ranked)", "",
                  f"chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3f}", "", "| model | mean rank | wins |", "|---|---|---|"]
        lines += [f"| {m} | {R[:, i].mean():.2f} | {int((R[:, i] == 1).sum())} |" for i, m in enumerate(models)]
        lines.append("")
    lines += ["## Which metric agrees with the listener? (within-stem Spearman, > 0 = agrees)", "",
              "| metric | stems | median rho | mean rho | agree | Wilcoxon p | metric-best = human-best |",
              "|---|---|---|---|---|---|---|"]
    for name, fn in metrics.items():
        rhos, hits, n_hit, chance = [], 0, 0, []
        for sid in stems:
            st = by_id[sid]
            pairs = [(prefs[sid][m], fn(st["models"][m], st)) for m in prefs[sid] if m in st.get("models", {})]
            pairs = [(h, v) for h, v in pairs if v is not None and np.isfinite(v)]
            if len(pairs) < 3:
                continue
            h, v = np.array(pairs, float).T
            if np.ptp(v) == 0 or np.ptp(h) == 0:
                continue
            rhos.append(float(stats.spearmanr(h, v)[0]))
            hits += int(h[np.argmin(v)] == h.min())
            n_hit += 1
            chance.append(1.0 / len(h))
        if len(rhos) < 3:
            continue
        rhos = np.array(rhos)
        p = float(stats.wilcoxon(rhos).pvalue) if np.any(rhos != 0) else 1.0
        res["metrics"][name] = {"n": len(rhos), "median_rho": float(np.median(rhos)), "mean_rho": float(rhos.mean()),
                                "agree": int((rhos > 0).sum()), "wilcoxon_p": p, "top1_hits": hits,
                                "top1_chance": float(np.sum(chance))}
        lines.append(f"| {name} | {len(rhos)} | {np.median(rhos):+.2f} | {rhos.mean():+.2f} | {(rhos > 0).sum()}/{len(rhos)} "
                     f"| {p:.4f} | {hits}/{n_hit} (chance {np.sum(chance):.1f}) |")
    lines += ["", "Small samples: with ~20 stems a median rho needs to be well above 0.3 before it means much, and "
              "several metrics are tested at once (no multiplicity correction)."]
    return lines, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefs")
    ap.add_argument("summary")
    ap.add_argument("--recompute", default=None, metavar="LISTEN_DIR")
    ap.add_argument("--max_eval_freq", type=float, default=None)
    ap.add_argument("--midi_dir", default=None)
    ap.add_argument("--out", default=None, help="markdown report (a .json with the numbers is written next to it)")
    a = ap.parse_args(argv)
    prefs = json.load(open(a.prefs))
    summary = json.load(open(a.summary))
    if a.recompute:
        summary = recompute(summary, a.recompute, a.max_eval_freq, a.midi_dir)
    lines, res = agreement(prefs, summary)
    text = "\n".join(["# Human preference vs metrics", ""] + lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main()
