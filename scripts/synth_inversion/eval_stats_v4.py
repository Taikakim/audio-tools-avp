# Owner: KUANG (created 2026-10-12)
"""Statistics for comparing inverter runs on SHARED evaluation items (EVAL_PARADIGM_V4.md).

Why this exists: the A1/A2/A3 report (V4_EXPERIMENT_REPORT.md) read single-run, single-checkpoint differences of
1-3 points as effects, and sized their noise with a binomial formula for ONE arm on 512 independent notes. That is
too optimistic three ways:

  * the held-out notes are clustered: several notes come from each held-out preset, so the effective sample size
    is closer to the number of presets than to the number of notes;
  * two arms are scored on the SAME notes, so the comparison is paired; pairing removes the shared item
    difficulty and is the right test (it is usually MORE powerful than comparing two separate CIs);
  * training-run variance (seed, data order, GPU non-determinism) is not evaluation noise at all and is only
    visible by comparing replicate runs of an identical configuration.

This module gives the paired, cluster-aware tools for the first two and a replicate gauge for the third.

Conventions
  * Per-item arrays of one comparison are aligned on the same evaluation items (same eval-set md5).
  * `clusters` holds the held-out preset index of each item; the bootstrap resamples PRESETS, not notes.
  * Deltas are a - b. `higher_is_better` says which sign is good for the metric.
  * All bootstrap functions are deterministic for a given `seed`.
"""
import math

import numpy as np
from scipy.stats import binomtest, norm


def _f64(x, name):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


def _clusters(clusters, n):
    c = np.asarray(clusters).reshape(-1)
    if c.size != n:
        raise ValueError(f"clusters has {c.size} entries for {n} items")
    uniq, inv = np.unique(c, return_inverse=True)
    return uniq, inv


def cluster_bootstrap_means(x, clusters, n_boot=10000, seed=0):
    """Bootstrap distribution of the item-weighted mean of `x`, resampling whole clusters with replacement.

    Vectorised: a resample is a vector of cluster multiplicities c ~ Multinomial(K, 1/K); its mean is
    sum_k c_k S_k / sum_k c_k N_k with S_k, N_k the cluster sums and sizes. NaN items are dropped first."""
    x = _f64(x, "x")
    uniq, inv = _clusters(clusters, x.size)
    ok = np.isfinite(x)
    if not ok.any():
        raise ValueError("x has no finite values")
    S = np.bincount(inv[ok], weights=x[ok], minlength=len(uniq))
    N = np.bincount(inv[ok], minlength=len(uniq)).astype(np.float64)
    keep = N > 0
    S, N = S[keep], N[keep]
    K = len(S)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot).astype(np.float64)
    den = counts @ N
    den[den == 0] = np.nan
    return (counts @ S) / den


def cluster_bootstrap(stat_fn, arrays, clusters, n_boot=2000, seed=0):
    """Generic (slower) cluster bootstrap: stat_fn(*resampled_arrays) for each resample of whole clusters.
    Use for statistics that are not item means (R-squared, medians, ratios)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    uniq, inv = _clusters(clusters, n)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        out[b] = stat_fn(*(a[idx] for a in arrays))
    return out


def _summary(est, boots, ci, n_items, n_clusters, iid_se):
    boots = boots[np.isfinite(boots)]
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a])
    se = float(boots.std(ddof=1))
    # Two-sided bootstrap p for "the mean is 0": share of the bootstrap distribution on the far side of 0,
    # doubled. Approximate (percentile bootstrap); floored at 1/n so it is never reported as exactly 0.
    tail = min(np.mean(boots <= 0.0), np.mean(boots >= 0.0))
    p = float(min(1.0, max(2.0 * tail, 1.0 / len(boots))))
    deff = (se / iid_se) ** 2 if iid_se > 0 else float("nan")
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "p": p, "se": se, "iid_se": float(iid_se),
            "design_effect": float(deff), "n_items": int(n_items), "n_clusters": int(n_clusters), "ci": ci}


def mean_ci(x, clusters, n_boot=10000, seed=0, ci=0.95):
    """Item mean of one arm with a cluster-bootstrap CI. `design_effect` = (cluster SE / iid SE)^2: how much the
    clustering inflates the variance relative to the naive binomial / iid formula."""
    x = _f64(x, "x")
    ok = np.isfinite(x)
    c = np.asarray(clusters).reshape(-1)
    boots = cluster_bootstrap_means(x[ok], c[ok], n_boot, seed)
    iid_se = x[ok].std(ddof=1) / math.sqrt(ok.sum()) if ok.sum() > 1 else float("nan")
    return _summary(x[ok].mean(), boots, ci, ok.sum(), len(np.unique(c[ok])), iid_se)


def paired_delta(a, b, clusters, n_boot=10000, seed=0, ci=0.95):
    """Paired comparison on shared items: mean(a - b) with a cluster-bootstrap CI and approximate p.

    Items where either side is NaN are dropped (both sides together, so the pairing is kept)."""
    a = _f64(a, "a")
    b = _f64(b, "b")
    if a.size != b.size:
        raise ValueError(f"a and b differ in length ({a.size} vs {b.size}): not the same items")
    d = a - b
    ok = np.isfinite(d)
    c = np.asarray(clusters).reshape(-1)
    boots = cluster_bootstrap_means(d[ok], c[ok], n_boot, seed)
    iid_se = d[ok].std(ddof=1) / math.sqrt(ok.sum()) if ok.sum() > 1 else float("nan")
    out = _summary(d[ok].mean(), boots, ci, ok.sum(), len(np.unique(c[ok])), iid_se)
    out["mean_a"], out["mean_b"] = float(a[ok].mean()), float(b[ok].mean())
    return out


def binomial_se(p, n):
    """The report's formula: sqrt(p(1-p)/n) for ONE arm on n independent items (kept for comparison)."""
    return math.sqrt(p * (1.0 - p) / n)


def sign_test(wins, losses):
    """Two-sided exact sign test; ties are excluded before calling (they carry no direction)."""
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    return float(binomtest(int(wins), n, 0.5).pvalue)


def min_wins(n, alpha=0.05):
    """Smallest number of wins out of n (no ties) that a two-sided exact sign test calls significant.
    n = 24 stems -> 18 at alpha 0.05 (17 gives p = 0.064)."""
    for k in range(int(math.ceil(n / 2.0)), n + 1):
        if binomtest(k, n, 0.5).pvalue <= alpha:
            return k
    return None


def mcnemar_exact(hit_a, hit_b):
    """Exact McNemar on paired binary outcomes: only discordant items count. Ignores clustering, so use it as
    a cross-check of `paired_delta`, not instead of it."""
    a = np.asarray(hit_a, dtype=bool).reshape(-1)
    b = np.asarray(hit_b, dtype=bool).reshape(-1)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    p = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"a_only": n01, "b_only": n10, "p": p}


def holm(pvals):
    """Holm step-down adjustment of a {name: p} dict -> {name: adjusted p}."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out, running = {}, 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def replicate_gauge(deltas):
    """Run-to-run noise gauge from replicate pairs: runs whose configuration is identical for the module being
    measured (for the JEPA metrics: same JEPA config; for the flow metrics: same flow config). `deltas` are their
    paired mean differences on the same items. Returns the largest |delta| (conservative with 1-2 pairs) and the
    RMS (the per-pair SD estimate of a difference)."""
    d = np.abs(np.asarray([x for x in deltas if x is not None and np.isfinite(x)], dtype=np.float64))
    if d.size == 0:
        return {"n_pairs": 0, "max_abs": float("nan"), "rms": float("nan")}
    return {"n_pairs": int(d.size), "max_abs": float(d.max()), "rms": float(math.sqrt(np.mean(d ** 2)))}


def verdict(res, higher_is_better=True, gauge=None, min_effect=0.0):
    """Decision rule (EVAL_PARADIGM_V4.md section 3):
      established  the CI excludes 0 AND |delta| >= max(min_effect, gauge)
      below gauge  the CI excludes 0 but |delta| is inside the run-to-run gauge or under the minimal effect
      not established  otherwise.
    `gauge` is replicate_gauge(...)["max_abs"] for the module, or None when no replicate exists (then the verdict
    is marked provisional by the caller: evaluation noise is controlled, training noise is not)."""
    d, lo, hi = res["estimate"], res["lo"], res["hi"]
    excludes0 = lo > 0.0 or hi < 0.0
    floor = max(float(min_effect or 0.0), float(gauge) if gauge is not None and np.isfinite(gauge) else 0.0)
    if not excludes0:
        return "not established"
    if abs(d) < floor:
        return "below gauge"
    good = (d > 0) == bool(higher_is_better)
    return "established better" if good else "established worse"


def min_detectable_delta(se, alpha=0.05, power=0.8):
    """Smallest true difference detected with the given power by a two-sided test whose estimate has standard
    error `se` (normal approximation): (z_{1-alpha/2} + z_power) * se. For planning eval-set size and replicates."""
    return float((norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)) * se)


def r2(y, yhat):
    """Coefficient of determination; NaN when y is constant or fewer than 2 values."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
