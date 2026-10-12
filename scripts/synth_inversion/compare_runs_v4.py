"""Paired, preset-clustered comparison of runs scored by heldout_eval_v4.py, against a PRE-REGISTERED plan.

Why: see EVAL_PARADIGM_V4.md. In short, every contrast is computed on the same held-out items (paired), its
uncertainty comes from resampling held-out PRESETS (the notes of one preset are not independent), and a difference
is only called an effect if it also exceeds the run-to-run gauge measured on replicate runs of the same
configuration. The plan is written BEFORE looking at the numbers; this script refuses metrics the plan did not
name as primary when it applies the multiple-comparison correction.

Plan (JSON), e.g. eval_plan_v4_aseries.json:
  {"name": ..., "set": "extra", "ladder_set": "ladders_extra", "kinds": ["ema", "online"],
   "alpha": 0.05, "n_boot": 10000,
   "contrasts": [{"name": "aux_head", "module": "jepa", "a": ["A1", "A3"], "b": ["A2"],
                  "primary": ["mrr_a2p", "mrr_p2a"], "secondary": ["hit_a2p", "ladder_jepa_rho"]}, ...],
   "replicates": {"jepa": [["A1", "A3"]], "flow": [["A2", "A3"]]},
   "min_effect": {"mrr_a2p": 0.01}}
A group of several runs is averaged per item before the paired difference (each run counts once).

Metrics (per item unless marked ladder):
  hit_a2p / hit_p2a   retrieval top-1 within the gallery (what the logs call r_a2p / r_p2a)
  mrr_a2p / mrr_p2a   reciprocal rank 1/(1 + rank): same information, far less discrete, so more power
  top5_a2p / top5_p2a top-5 retrieval
  flow_mse, flow_mae  per-item flow-matching loss (clean condition, fixed noise and t), lower is better
  param_mae           mean |error| of the flow's draw-mean over the continuous parameters (MIDI excluded)
  param_err:<name>    |error| of one continuous parameter, e.g. param_err:cutoff
  cat_acc             share of categorical parameters whose majority-vote class is right
  aux_sqerr           squared error of the inharmonicity head on items with a valid label (lower is better)
  ladder_flow_rho, ladder_jepa_rho, ladder_flow_mae   per ladder (ladder set); also :<axis>, e.g.
                      ladder_flow_rho:cutoff_low

Usage:
  python compare_runs_v4.py --dir /path/eval_v4 --plan eval_plan_v4_aseries.json --out report_v4.md
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import eval_stats_v4 as st  # noqa: E402

LADDER_AXES = ("cutoff_low", "aeg_sustain", "feg_amount", "aeg_decay")   # training_controls_v3.AXES
# name -> (higher_is_better, source)   source: "items" or "ladders"
BASE_METRICS = {
    "hit_a2p": (True, "items"), "hit_p2a": (True, "items"),
    "mrr_a2p": (True, "items"), "mrr_p2a": (True, "items"),
    "top5_a2p": (True, "items"), "top5_p2a": (True, "items"),
    "flow_mse": (False, "items"), "flow_mae": (False, "items"),
    "param_mae": (False, "items"), "cat_acc": (True, "items"), "aux_sqerr": (False, "items"),
    "ladder_flow_rho": (True, "ladders"), "ladder_jepa_rho": (True, "ladders"), "ladder_flow_mae": (False, "ladders"),
}


def metric_info(name):
    base, _, sub = name.partition(":")
    if base == "param_err":
        return False, "items"
    if base not in BASE_METRICS:
        raise KeyError(f"unknown metric {name!r} (known: {sorted(BASE_METRICS)} + param_err:<name>, ladder_*:<axis>)")
    return BASE_METRICS[base]


def per_item(res, name, param_names):
    """Per-item (or per-ladder) values of one metric from a loaded npz dict; NaN where undefined."""
    base, _, sub = name.partition(":")
    if base.startswith("hit_"):
        return (res["rank_" + base[4:]] == 0).astype(np.float64)
    if base.startswith("mrr_"):
        return 1.0 / (1.0 + res["rank_" + base[4:]].astype(np.float64))
    if base.startswith("top5_"):
        return (res["rank_" + base[5:]] < 5).astype(np.float64)
    if base in ("flow_mse", "flow_mae"):
        return res[base].astype(np.float64)
    if base == "param_mae":
        cont = list(res["cont_idx"])
        keep = [j for j, i in enumerate(cont) if param_names[i] != "midi_note"]
        return res["cont_abs_err"][:, keep].astype(np.float64).mean(axis=1)
    if base == "param_err":
        cont = list(res["cont_idx"])
        idx = param_names.index(sub)
        if idx not in cont:
            raise KeyError(f"param_err:{sub}: not a continuous parameter")
        return res["cont_abs_err"][:, cont.index(idx)].astype(np.float64)
    if base == "cat_acc":
        return res["cat_correct"].astype(np.float64).mean(axis=1)
    if base == "aux_sqerr":
        if "inharm" not in res or np.all(np.isnan(res["aux_pred"])):
            return np.full(len(res["preset"]), np.nan)
        v = np.where(res["inharm_valid"] > 0, (res["aux_pred"] - res["inharm"]) ** 2, np.nan)
        return v.astype(np.float64)
    if base.startswith("ladder_"):
        vals = res[base[len("ladder_"):]].astype(np.float64)
        if sub:
            vals = np.where(res["axis"] == LADDER_AXES.index(sub), vals, np.nan)
        return vals
    raise KeyError(name)


class Store:
    """Loads <set>__<run>__<kind>.npz files written by heldout_eval_v4.py and checks the item alignment."""

    def __init__(self, directory):
        self.dir = directory
        self.cache = {}
        meta = sorted(glob.glob(os.path.join(directory, "meta__*.json")))
        self.param_names = None
        self.meta = {}
        for p in meta:
            with open(p) as f:
                m = json.load(f)
            self.meta[(m["run"], m["kind"])] = m
            self.param_names = self.param_names or m["param_names"]

    def get(self, set_name, run, kind):
        key = (set_name, run, kind)
        if key not in self.cache:
            path = os.path.join(self.dir, f"{set_name}__{run}__{kind}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            z = np.load(path, allow_pickle=False)
            self.cache[key] = {k: z[k] for k in z.files}
        return self.cache[key]

    def values(self, set_name, runs, kind, metric):
        """Per-item metric averaged over the runs of a group; returns (values, clusters, items_md5)."""
        arrs, md5, clusters = [], None, None
        for r in runs:
            res = self.get(set_name, r, kind)
            m = str(res["items_md5"])
            if md5 is not None and m != md5:
                raise RuntimeError(f"{r}: scored on different items than the rest of its group ({set_name})")
            md5, clusters = m, res["preset"]
            arrs.append(per_item(res, metric, self.param_names))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN items (e.g. other ladder axes)
            return np.nanmean(np.stack(arrs), axis=0), clusters, md5


def run_plan(store, plan):
    kinds = plan.get("kinds", ["ema"])
    n_boot = int(plan.get("n_boot", 10000))
    alpha = float(plan.get("alpha", 0.05))
    min_eff = plan.get("min_effect", {})
    rows = []
    for c in plan["contrasts"]:
        metrics = [(m, True) for m in c.get("primary", [])] + [(m, False) for m in c.get("secondary", [])]
        for metric, primary in metrics:
            hib, src = metric_info(metric)
            set_name = plan.get("ladder_set", "ladders") if src == "ladders" else plan.get("set", "base")
            for kind in kinds:
                va, cl, ma = store.values(set_name, c["a"], kind, metric)
                vb, _, mb = store.values(set_name, c["b"], kind, metric)
                if ma != mb:
                    raise RuntimeError(f"{c['name']}: groups scored on different items")
                res = st.paired_delta(va, vb, cl, n_boot=n_boot, seed=0)
                gauges = []
                for r1, r2 in plan.get("replicates", {}).get(c.get("module", ""), []):
                    g1, gcl, _ = store.values(set_name, [r1], kind, metric)
                    g2, _, _ = store.values(set_name, [r2], kind, metric)
                    gauges.append(st.paired_delta(g1, g2, gcl, n_boot=2000, seed=1)["estimate"])
                gauge = st.replicate_gauge(gauges)
                rows.append({"contrast": c["name"], "module": c.get("module"), "metric": metric, "primary": primary,
                             "kind": kind, "set": set_name, "higher_is_better": hib, **res,
                             "gauge": gauge["max_abs"], "gauge_pairs": gauge["n_pairs"],
                             "min_effect": float(min_eff.get(metric, 0.0))})
    # Holm over the PRIMARY metrics of each contrast (per kind)
    for c in plan["contrasts"]:
        for kind in kinds:
            sel = [r for r in rows if r["contrast"] == c["name"] and r["kind"] == kind and r["primary"]]
            adj = st.holm({r["metric"]: r["p"] for r in sel})
            for r in sel:
                r["p_holm"] = adj[r["metric"]]
    for r in rows:
        gauge = r["gauge"] if r["gauge_pairs"] else None
        v = st.verdict(r, r["higher_is_better"], gauge, r["min_effect"])
        if r["primary"] and r.get("p_holm", 1.0) > alpha and v.startswith("established"):
            v = "not established (Holm)"
        if v.startswith("established") and not r["gauge_pairs"]:
            v += " (provisional: no replicate)"
        r["verdict"] = v
    # overall verdict per contrast x metric: the same 'established' call in every kind
    overall = {}
    for r in rows:
        overall.setdefault((r["contrast"], r["metric"]), []).append(r["verdict"])
    for r in rows:
        vs = overall[(r["contrast"], r["metric"])]
        same = all(v == vs[0] for v in vs) and vs[0].startswith("established")
        r["overall"] = vs[0] if same else ("mixed across kinds" if any(v.startswith("established") for v in vs)
                                           else "not established")
    return rows


def fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def report_md(plan, rows, store):
    L = [f"# {plan.get('name', 'Comparison')}", "",
         "Paired differences a - b on identical held-out items; 95% CI from resampling held-out presets; "
         "gauge = largest |difference| between replicate runs of the same configuration (EVAL_PARADIGM_V4.md).", ""]
    for kind in plan.get("kinds", ["ema"]):
        L += [f"## kind: {kind}", "",
              "| contrast | metric | a | b | a - b [95% CI] | p (Holm) | design eff. | gauge | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in rows if r["kind"] == kind]:
            star = "" if r["primary"] else " (sec.)"
            p = f"{r['p']:.3f}" + (f" ({r['p_holm']:.3f})" if "p_holm" in r else "")
            L.append(f"| {r['contrast']} | {r['metric']}{star} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | "
                     f"{fmt(r['estimate'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] | {p} | {fmt(r['design_effect'], 2)} | "
                     f"{fmt(r['gauge'])} | {r['verdict']} |")
        L.append("")
    L += ["## Overall (same call in every kind)", ""]
    seen = set()
    for r in rows:
        k = (r["contrast"], r["metric"])
        if k not in seen:
            seen.add(k)
            L.append(f"- {r['contrast']} / {r['metric']}: **{r['overall']}**")
    L += ["", "## Runs", ""]
    for (run, kind), m in sorted(store.meta.items()):
        par = "; ".join(f"{k} logged {v['logged']:.4f} vs {v['recomputed']:.4f}" for k, v in m.get("parity", {}).items())
        L.append(f"- {run}/{kind}: step {m['info'].get('step')}, seed {m['run_args'].get('seed')}, "
                 f"flow {m['info'].get('flow_class')}, aux {m['info'].get('aux_head')}; parity: {par or 'none'}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="output directory of heldout_eval_v4.py")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=None, help="markdown report (default: <dir>/report_<plan name>.md)")
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        plan = json.load(f)
    store = Store(a.dir)
    rows = run_plan(store, plan)
    md = report_md(plan, rows, store)
    out = a.out or os.path.join(a.dir, f"report_{os.path.splitext(os.path.basename(a.plan))[0]}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
