"""B.13 -- robust co-metrics over frozen TEST artifacts (PREREGISTRATION_B13.md).

No training, no inference: every number comes from `comparison_errors_test.npz` files
that already exist. Absolute errors are recovered as sqrt(se). For each population
(cut / incl), split seed and target, arms are scored against the interpolation ladder
(pchip, gp, gp_causal, persistence) and against each other with:

  M1  medAE skill       1 - median|e_a| / median|e_b|        (weighted-median cluster bootstrap)
  M2  Huber skill       1 - sum rho_d(e_a) / sum rho_d(e_b)  (rho = e^2 inside d, 2d|e| - d^2 outside;
                                                              d = sec. 8aq order-4 noise bound)
  M3  trimmed skill     MSE skill after dropping the top 1 % rows by max(se_a, se_b)
  M4  tail share        share of each arm's SSE carried by the rows M3 dropped
  M5  quantile ladder   |e| quantile ratios q_a / q_b at 50/75/90/95/99 (point only)

Guards (both abort): the MSE skill vs PCHIP recomputed from each npz must equal the dir's
comparison_metrics.json value, and the shot ids must map onto the split's test files.

Usage (repo root):
  py ces_prediction/experiments/b13_robust/rescore_b13.py
  py ... --extra huber=.b14_huber_{pop}_s{s} tnll=.b14_tnll_{pop}_s{s}   # B.14 arms
  py ... --extra ens4=.b15_ens4_{pop}_s{s}                              # B.15 arm
Output: data/.b13_summary.json (+ printed verdict table).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "ces_prediction"))
from bootstrap_compare import BOOTSTRAP_SEED, _bootstrap  # noqa: E402

SEEDS = (42, 1, 7, 123)
TARGETS = ("CES_TI", "CES_VT")
B_FULL, B_STRATA = 10000, 2000
TRIM = 0.01
PRACTICAL_EPS = 0.02
BASELINES = ("pchip", "gp", "gp_causal", "persistence")
POPS = {
    "cut": {"model": ".b1_seqv2_s{s}_i{s}", "window": ".b1_w2cut_s{s}",
            "manifest": ".b1_w2cut_split_s{s}/split_manifest.json",
            "delta": {"CES_TI": 129.9, "CES_VT": 16.4}},
    "incl": {"model": ".b5i_seqv2_s{s}", "window": ".b5i_w2_s{s}",
             "manifest": ".b1_manifest_s{s}/split_manifest.json",
             "delta": {"CES_TI": 274.3, "CES_VT": 16.4}},
}
QUANTILES = (50, 75, 90, 95, 99)


# ---- loading ------------------------------------------------------------------------

def shot_names():
    return [p.name for p in sorted(DATA.glob("s*.csv"))]


def load_arm(path, names, test_files, verify_metrics=True):
    """npz -> {target: {shot(str), dt, se_* (float64), y, is_peak}}; aborts on drift."""
    z = np.load(path)
    out = {}
    for t in TARGETS:
        idx = z[f"{t}_shot"].astype(int)
        shots = np.asarray([names[i] for i in idx])
        if not set(shots) <= set(test_files):
            raise SystemExit(f"FATAL: {path} {t}: shot ids do not map onto the split's test files")
        d = {"shot": shots, "dt": z[f"{t}_dt_ms"], "y": z[f"{t}_y_true"], "is_peak": z[f"{t}_is_peak"].astype(bool)}
        for k in ("model", *BASELINES):
            d[f"se_{k}"] = z[f"{t}_se_{k}"].astype(np.float64)
        out[t] = d
    if verify_metrics:
        cm = json.loads((path.parent / "comparison_metrics.json").read_text(encoding="utf-8"))
        for t in TARGETS:
            got = 1.0 - out[t]["se_model"].mean() / out[t]["se_pchip"].mean()
            want = cm["per_target"][t]["skill_vs_pchip"]
            # The frozen harness reduced float32 arrays; this script reduces in float64, so the
            # two agree to ~1e-5 relative, not bit-exactly. Anything beyond 1e-4 is drift.
            if abs(got - want) > 1e-4 * max(1.0, abs(want)):
                raise SystemExit(f"FATAL: {path} {t}: MSE skill {got} != recorded {want}")
    return out


# ---- statistics ---------------------------------------------------------------------

def _cluster_weights(clusters, n_boot, rng):
    uniq, inv = np.unique(clusters, return_inverse=True)
    n = len(uniq)
    idx = rng.integers(0, n, size=(n_boot, n))
    w = np.zeros((n_boot, n), dtype=np.float64)
    for b in range(n_boot):
        w[b] = np.bincount(idx[b], minlength=n)
    return inv, w


def _weighted_medians(values, inv, w, chunk=200):
    """Lower weighted median of `values` for every bootstrap weight vector."""
    order = np.argsort(values, kind="stable")
    v, c = values[order], inv[order]
    out = np.empty(len(w))
    for s in range(0, len(w), chunk):
        rw = w[s:s + chunk][:, c]                       # (chunk, N) row weights
        cum = np.cumsum(rw, axis=1)
        half = 0.5 * cum[:, -1:]
        pos = (cum >= half).argmax(axis=1)
        out[s:s + chunk] = v[pos]
    return out


def median_skill(clusters, ae_a, ae_b, rng, n_boot):
    """M1: 1 - median|e_a| / median|e_b| with a shot-cluster bootstrap CI."""
    point = 1.0 - np.median(ae_a) / np.median(ae_b)
    inv, w = _cluster_weights(clusters, n_boot, rng)
    ma, mb = _weighted_medians(ae_a, inv, w), _weighted_medians(ae_b, inv, w)
    skill = 1.0 - ma / mb
    lo, hi = np.percentile(skill, [2.5, 97.5])
    return {"skill_point": float(point), "skill_ci95": [float(lo), float(hi)],
            "pass": bool(lo > 0.0), "n_shots": int(w.shape[1])}


def huber_rho(ae, delta):
    return np.where(ae <= delta, ae ** 2, 2.0 * delta * ae - delta ** 2)


def ratio_skill(clusters, cost_a, cost_b, rng, n_boot):
    """Ratio-of-sums skill (M2 / M3) through the repo's own bootstrap at the PR4 B."""
    saved = _bootstrap.__globals__["B"]
    _bootstrap.__globals__["B"] = n_boot
    try:
        r = _bootstrap(clusters, cost_a, cost_b, rng)
    finally:
        _bootstrap.__globals__["B"] = saved
    return {k: r[k] for k in ("skill_point", "skill_ci95", "pass", "n_shots")}


def compare(clusters, se_a, se_b, delta, rng, n_boot):
    ae_a, ae_b = np.sqrt(se_a), np.sqrt(se_b)
    worst = np.maximum(se_a, se_b)
    keep = worst <= np.quantile(worst, 1.0 - TRIM)
    res = {
        "n": int(len(se_a)),
        "mse": ratio_skill(clusters, se_a, se_b, rng, n_boot),
        "m1_medae": median_skill(clusters, ae_a, ae_b, rng, n_boot),
        "m2_huber": ratio_skill(clusters, huber_rho(ae_a, delta), huber_rho(ae_b, delta), rng, n_boot),
        "m3_trimmed": ratio_skill(clusters[keep], se_a[keep], se_b[keep], rng, n_boot),
        "m4_tail_share": {"a": float(se_a[~keep].sum() / se_a.sum()), "b": float(se_b[~keep].sum() / se_b.sum()),
                          "rows_dropped": int((~keep).sum())},
        "m5_quantile_ratio": {str(q): float(np.percentile(ae_a, q) / np.percentile(ae_b, q)) for q in QUANTILES},
    }
    return res


# ---- driver -------------------------------------------------------------------------

def score_population(pop, extra, rng):
    cfg = POPS[pop]
    names = shot_names()
    per_seed, pooled_rows = {}, {}
    for s in SEEDS:
        manifest = json.loads((DATA / cfg["manifest"].format(s=s)).read_text(encoding="utf-8"))
        test_files = set(manifest["test_files"])
        arms = {"model": load_arm(DATA / cfg["model"].format(s=s) / "comparison_errors_test.npz", names, test_files),
                "window": load_arm(DATA / cfg["window"].format(s=s) / "comparison_errors_test.npz", names, test_files)}
        for name, pat in extra.items():
            p = DATA / pat.format(pop=pop, s=s) / "comparison_errors_test.npz"
            if p.exists():
                arms[name] = load_arm(p, names, test_files)
        per_seed[str(s)] = {}
        for t in TARGETS:
            base = arms["model"][t]
            # pairing guard: every arm scored the same rows (shot + dt pin the row set)
            for name, a in arms.items():
                if not (np.array_equal(a[t]["shot"], base["shot"]) and np.array_equal(a[t]["dt"], base["dt"], equal_nan=True)):
                    raise SystemExit(f"FATAL: {pop} s{s} {t}: arm {name} is not row-paired with the backbone")
            delta = cfg["delta"][t]
            entry = {}
            strata = {"all": np.ones(len(base["shot"]), bool), "peak": base["is_peak"], "nonpeak": ~base["is_peak"],
                      "dt_le_15": base["dt"] <= 15.0, "dt_gt_15": base["dt"] > 15.0}
            pairs = [(name, b) for name in arms for b in BASELINES] + [("model", "window")]
            pairs += [(name, "model") for name in extra if name in arms]
            for a_name, b_name in pairs:
                se_a = arms[a_name][t]["se_model"]
                se_b = arms[b_name][t]["se_model"] if b_name in arms else base[f"se_{b_name}"]
                key = f"{a_name}_vs_{b_name}"
                entry[key] = {}
                for st, m in strata.items():
                    if m.sum() < 200:
                        continue
                    entry[key][st] = compare(base["shot"][m], se_a[m], se_b[m], delta, rng,
                                             B_FULL if st == "all" else B_STRATA)
                pooled_rows.setdefault((t, key), []).append((base["shot"], se_a, se_b))
            per_seed[str(s)][t] = entry
        print(f"[b13] {pop} s{s} scored ({', '.join(arms)})", flush=True)
    pooled = {}
    for (t, key), parts in pooled_rows.items():
        if len(parts) < len(SEEDS):
            continue
        shots = np.concatenate([p[0] for p in parts])
        se_a = np.concatenate([p[1] for p in parts])
        se_b = np.concatenate([p[2] for p in parts])
        pooled.setdefault(t, {})[key] = compare(shots, se_a, se_b, cfg["delta"][t], rng, B_FULL)
        pooled[t][key]["n_splits"] = len(parts)
    return {"per_seed": per_seed, "pooled": pooled}


def verdicts(res):
    """PREREGISTRATION_B13.md sec. 4: Q1 (V_rot bulk advantage), Q2 (T_i bulk / tail / mixed)."""
    out = {}
    for pop, r in res.items():
        v = {}
        for t in TARGETS:
            per = [r["per_seed"][str(s)][t]["model_vs_pchip"]["all"] for s in SEEDS]
            pooled = r["pooled"][t]["model_vs_pchip"]
            m1_pass = sum(p["m1_medae"]["pass"] for p in per)
            m1_pool = pooled["m1_medae"]
            m2_sign = np.sign(pooled["m2_huber"]["skill_point"]) == np.sign(m1_pool["skill_point"])
            mse_pool = pooled["mse"]["skill_point"]
            q1 = m1_pass >= 3 and m1_pool["pass"] and bool(m2_sign)
            if t == "CES_VT":
                v[t] = {"q1_bulk_advantage": q1, "m1_splits_pass": m1_pass, "m1_pooled": m1_pool,
                        "m2_pooled": pooled["m2_huber"], "mse_pooled": pooled["mse"]}
            else:
                ci_includes0 = sum(1 for p in per if p["m1_medae"]["skill_ci95"][0] <= 0.0 <= p["m1_medae"]["skill_ci95"][1])
                if m1_pool["skill_point"] >= 0.5 * mse_pool and m1_pass == 4:
                    cls = "bulk"
                elif m1_pool["skill_point"] <= PRACTICAL_EPS or ci_includes0 >= 2:
                    cls = "tail"
                else:
                    cls = "mixed"
                v[t] = {"q2_class": cls, "m1_splits_pass": m1_pass, "m1_pooled": m1_pool,
                        "mse_pooled": pooled["mse"], "m2_pooled": pooled["m2_huber"],
                        "m5_pooled": pooled["m5_quantile_ratio"]}
        out[pop] = v
    return out


def fmt(r):
    return f"{r['skill_point']:+.3f} [{r['skill_ci95'][0]:+.3f}, {r['skill_ci95'][1]:+.3f}]{'*' if r['pass'] else ' '}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", nargs="*", default=list(POPS))
    ap.add_argument("--extra", nargs="*", default=[], help="name=dir pattern with {pop} and {s}")
    ap.add_argument("--out", default=str(DATA / ".b13_summary.json"))
    args = ap.parse_args()
    extra = dict(e.split("=", 1) for e in args.extra)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    res = {pop: score_population(pop, extra, rng) for pop in args.pops}
    summary = {"protocol": {"B": B_FULL, "B_strata": B_STRATA, "trim": TRIM, "practical_eps": PRACTICAL_EPS,
                            "delta": {p: POPS[p]["delta"] for p in args.pops}, "extra_arms": extra},
               "results": res, "verdicts": verdicts(res)}
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for pop in args.pops:
        for t in TARGETS:
            print(f"\n== {pop} {t} ==  (MSE | M1 medAE | M2 Huber | M3 trimmed | tail share a/b)")
            keys = list(res[pop]["pooled"][t])
            for key in keys:
                line = []
                for s in SEEDS:
                    e = res[pop]["per_seed"][str(s)][t][key]["all"]
                    line.append(f"s{s}: {e['mse']['skill_point']:+.3f}|{e['m1_medae']['skill_point']:+.3f}"
                                f"{'*' if e['m1_medae']['pass'] else ' '}|{e['m2_huber']['skill_point']:+.3f}"
                                f"{'*' if e['m2_huber']['pass'] else ' '}")
                p = res[pop]["pooled"][t][key]
                print(f"  {key:22s} " + "  ".join(line))
                print(f"  {'':22s} pooled MSE {fmt(p['mse'])}  M1 {fmt(p['m1_medae'])}  M2 {fmt(p['m2_huber'])}  "
                      f"M3 {fmt(p['m3_trimmed'])}  tail {p['m4_tail_share']['a']:.2f}/{p['m4_tail_share']['b']:.2f}")
    print("\n== verdicts ==")
    print(json.dumps(summary["verdicts"], indent=1, default=str)[:4000])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
