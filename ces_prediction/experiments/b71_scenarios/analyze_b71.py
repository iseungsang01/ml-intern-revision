"""B.71 readout (PREREGISTRATION_B71.md sections 3-5 and 8; addendum 1): bins, pairs, divergence and return rules,
statements S1-S5, the staleness training range and the cut table, from data/.b71/rows_r{r}_f{k}.npz (and the
filter rows kf_r{r}_f{k}.npz when B.70 phase 2 has delivered them).

Unit = (scored row, cut).  Per re-partition the five test folds are joined; a discharge is a D1 discharge of the
re-partition with at least one scored T_i row.  Intervals: the two-level bootstrap of B.70 addendum 1 A4
(stage2_b64.weights: a re-partition drawn among r2-r4, then its discharges resampled with replacement; units weighted
by the draw counts of their discharge), 20,000 draws, seed [20261171, crc32(cell name)], one stream per cell; every
estimator, pair and ratio of a cell uses the same draws.  Point = mean over the three re-partitions of the
per-re-partition value.  MSE interval = square of the RMSE bounds.

    py ces_prediction/experiments/b71_scenarios/run_b71.py analyze
    py ces_prediction/experiments/b71_scenarios/analyze_b71.py --plumb DIR   (structure test: every estimator replaced
                                                                              by 1,000 eV; nothing is read out)
Output: docs/paper/b71_scenarios.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "b64_two_stage"))
sys.path.insert(0, str(HERE))
import stage2_b64 as S64  # noqa: E402  (weights: the two-level draws of B.70 A4)
from analyze_b39 import ci  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "data" / ".b71"
OUT_JSON = REPO / "docs" / "paper" / "b71_scenarios.json"
OUT_MD = REPO / "docs" / "paper" / "b71_scenarios.md"
REG = "ces_prediction/experiments/PREREGISTRATION_B71.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B71_ADDENDUM1.md"
REPEATS = (2, 3, 4)
K = 5
SEED, N_BOOT = 20261171, 20_000
F_BINS = ((0, 10), (10, 50), (50, 100), (100, 300), (300, 1000), (1000, 3000))
G_BINS = ((0, 10), (10, 30), (30, 100), (100, 300), (300, 1000))
GAPS = (100, 300, 1000)
LO_EV, HI_EV = 0.0, 3000.0
DIV_RATIO, DIV_SHARE, RET_RATIO = 2.0, 0.05, 1.10
F_EST = ("net_U", "net_F", "net_R", "twin_U", "twin_F", "twin_R", "hold_F", "gp_F")
KF_EST = ("kf0_F", "kfx_F")
F_PAIRS = (("net_F", "twin_F"), ("net_F", "hold_F"), ("net_F", "gp_F"), ("net_F", "kfx_F"), ("net_F", "kf0_F"),
           ("net_F", "net_R"), ("net_U", "net_F"))
F_RATIOS = (("net_R", "hold_F"), ("twin_R", "hold_F"), ("net_F", "net_U"))
G_EST = ("net_G", "net_U")


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def bname(b):
    return f"({b[0]}, {b[1]}]"


# ------------------------------------------------------------------------------------------ data
def load(cache, reps, plumb=False):
    D, metas, kf_ok = {}, {}, True
    for r in reps:
        Fp, Gp, files, d1n = [], [], [], []
        for k in range(K):
            p = cache / f"rows_r{r}_f{k}.npz"
            if not p.exists():
                if plumb:
                    continue
                raise SystemExit(f"FATAL: {p} missing; run `run_b71.py rows` first")
            z = np.load(p)
            meta = json.loads(str(z["meta"]))
            metas[f"r{r}f{k}"] = meta
            fF = {c.split("__")[1]: z[c] for c in z.files if c.startswith("F__")}
            fG = {c.split("__")[1]: z[c] for c in z.files if c.startswith("G__")}
            fF["cut"] = fF["cut"] + 100_000 * k                         # cut ids unique within a re-partition
            fG["cut"] = fG["cut"] + 100_000 * k
            kp = cache / f"kf_r{r}_f{k}.npz"
            if kp.exists():
                kz = np.load(kp)
                fF["kf0_F"], fF["kfx_F"] = kz["kf0_F"], kz["kfx_F"]
            else:
                kf_ok = False
            Fp.append(fF); Gp.append(fG)
            files.append(z["d1_files"]); d1n.append(z["d1_rows_per_file"])
        if not Fp:
            continue
        keys = set.intersection(*(set(x) for x in Fp))
        F = {c: np.concatenate([x[c] for x in Fp]) for c in keys}
        G = {c: np.concatenate([x[c] for x in Gp]) for c in Gp[0]}
        u = np.unique(np.concatenate(files)[np.concatenate(d1n) > 0])
        for X in (F, G):
            if not np.all(np.isin(X["file"], u)):
                raise SystemExit(f"FATAL: r{r}: units in discharges without D1 rows")
            X["f"] = np.searchsorted(u, X["file"])
        D[r] = {"F": F, "G": G, "n": len(u)}
    if plumb:
        for d in D.values():
            for X, ests in ((d["F"], F_EST + KF_EST), (d["G"], G_EST)):
                for e in ests:
                    if e in X:
                        X[e] = np.full(len(X["y"]), 1000.0)
    return D, metas, kf_ok


# ------------------------------------------------------------------------------------------ one cell
def cell(name, D, part, mask_fn, ests, pairs, ratios, reps, nboot=N_BOOT):
    W = S64.weights(rng_for(name), reps, {r: D[r]["n"] for r in reps}, nboot)
    per, boot, extra = {}, {}, {}

    def quantities(S):
        o = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for e in ests:
                o[f"mse|{e}"] = S[f"se|{e}"] / S["cnt"]
                o[f"rmse|{e}"] = np.sqrt(o[f"mse|{e}"])
                o[f"bias|{e}"] = S[f"sg|{e}"] / S["cnt"]
                o[f"out|{e}"] = S[f"out|{e}"] / S["cnt"]
            for a, b in pairs:
                o[f"skill|{a}|{b}"] = 1 - S[f"se|{a}"] / S[f"se|{b}"]
                o[f"rmse_diff|{a}|{b}"] = o[f"rmse|{a}"] - o[f"rmse|{b}"]
            for a, b in ratios:
                o[f"ratio|{a}|{b}"] = S[f"se|{a}"] / S[f"se|{b}"]
        return o

    for r in reps:
        d = D[r][part]
        m = mask_fn(d)
        f, n = d["f"][m], D[r]["n"]
        S = {"cnt": np.bincount(f, None, n).astype(float)}
        ex = {"pairs": int(m.sum()), "discharges": int((S["cnt"] > 0).sum()),
              "cuts": int(len(np.unique(d["cut"][m])))}
        for e in ests:
            est = d[e][m]
            err = est - d["y"][m]
            S[f"se|{e}"] = np.bincount(f, err ** 2, n)
            S[f"sg|{e}"] = np.bincount(f, err, n)
            S[f"out|{e}"] = np.bincount(f, ((est < LO_EV) | (est > HI_EV) | ~np.isfinite(est)).astype(float), n)
            ex[f"median_abs|{e}"] = float(np.median(np.abs(err))) if len(err) else float("nan")
            ex[f"nonfinite|{e}"] = int((~np.isfinite(est)).sum())
        extra[r] = ex
        per[r] = {kk: float(v) for kk, v in quantities({kk: v.sum() for kk, v in S.items()}).items()}
        sel, Wr = W[r]
        for kk, v in quantities({kk: Wr @ v for kk, v in S.items()}).items():
            boot.setdefault(kk, np.full(nboot, np.nan))[sel] = v
    o = {}
    for kk in boot:
        pt = float(np.mean([per[r][kk] for r in reps]))
        lo_hi = ci(boot[kk]) if np.isfinite(boot[kk]).any() else [float("nan"), float("nan")]
        o[kk] = [pt, *lo_hi]
    for e in ests:                                            # MSE interval = square of the RMSE bounds (A4)
        rm = o[f"rmse|{e}"]
        o[f"mse|{e}"] = [o[f"mse|{e}"][0], rm[1] ** 2, rm[2] ** 2]
        o[f"median_abs|{e}"] = float(np.mean([extra[r][f"median_abs|{e}"] for r in reps]))
    o["per_repeat"] = {str(r): extra[r] for r in reps}
    o["per_repeat_point"] = {str(r): per[r] for r in reps}
    return o


# ------------------------------------------------------------------------------------------ readout
def analyze(cache=CACHE, out_json=OUT_JSON, out_md=OUT_MD, plumb=False):
    reps = REPEATS
    D, metas, kf_ok = load(cache, reps, plumb)
    reps = tuple(r for r in reps if r in D)
    ests_F = F_EST + (KF_EST if kf_ok else ())
    pairs_F = tuple(p for p in F_PAIRS if all(e in ests_F for e in p))
    res = {"registration": REG, "addendum": ADD, "n_boot": N_BOOT, "seed": SEED, "repeats": list(reps),
           "filters": "included" if kf_ok else "left out: B.70 phase 2 has not delivered the filter parameters",
           "F": {}, "G": {}}
    # units outside every bin
    res["units_outside_bins"] = {
        "F_e_gt_3000ms": {str(r): int((D[r]["F"]["e_q"] > 30_000).sum()) for r in reps},
        "F_e_le_0": {str(r): int((D[r]["F"]["e_q"] <= 0).sum()) for r in reps}}
    for b in F_BINS:
        lo, hi = b[0] * 10, b[1] * 10
        res["F"][bname(b)] = cell(f"F|{bname(b)}", D, "F", lambda d, lo=lo, hi=hi: (d["e_q"] > lo) & (d["e_q"] <= hi),
                                  ests_F, pairs_F, F_RATIOS, reps)
        print(f"F {bname(b)} done", flush=True)
    for g in GAPS:
        res["G"][str(g)] = {}
        for b in G_BINS:
            lo, hi = b[0] * 10, b[1] * 10
            res["G"][str(g)][bname(b)] = cell(
                f"G{g}|{bname(b)}", D, "G",
                lambda d, g=g, lo=lo, hi=hi: (d["g_ms"] == g) & (d["s_q"] > lo) & (d["s_q"] <= hi),
                G_EST, (("net_G", "net_U"),), (("net_G", "net_U"),), reps)
        print(f"G {g} done", flush=True)
    # divergence (R) and return (G) rules
    div = {}
    for est in ("net_R", "twin_R"):
        div[est] = {}
        for b in F_BINS:
            c = res["F"][bname(b)]
            ratio, share = c[f"ratio|{est}|hold_F"], c[f"out|{est}"][0]
            ev = np.isfinite(ratio[1])
            div[est][bname(b)] = {"ratio_mse_over_hold_F": ratio, "share_outside_0_3000_eV": share,
                                  "evaluable": bool(ev),
                                  "divergent": bool(ev and (ratio[1] > DIV_RATIO or share > DIV_SHARE))}
    res["divergence"] = div
    ret = {}
    for g in GAPS:
        rows = {}
        for b in G_BINS:
            q = res["G"][str(g)][bname(b)]["ratio|net_G|net_U"]
            rows[bname(b)] = {"ratio": q, "rule_met": bool(np.isfinite(q[2]) and q[2] < RET_RATIO)}
        names = [bname(b) for b in G_BINS]
        first = next((nb for nb in names if rows[nb]["rule_met"]), None)
        stay = next((nb for i, nb in enumerate(names) if all(rows[x]["rule_met"] for x in names[i:])), None)
        ret[str(g)] = {"bins": rows, "first_bin_meeting_rule": first, "returned_by_bin": stay}
    res["return"] = ret
    # statements (section 5): MSE reduction = skill, point [95 % interval]; a statement on a skill holds when the lower
    # bound is above 0 in every bin it names (NaN -> not established)
    sk = lambda b, a, c: res["F"][bname(b)][f"skill|{a}|{c}"]                      # noqa: E731
    above = lambda d: bool(all(np.isfinite(v[1]) and v[1] > 0 for v in d.values()))  # noqa: E731
    up_to_1000 = [b for b in F_BINS if b[1] <= 1000]
    s1 = {bname(b): sk(b, "net_F", "twin_F") for b in ((100, 300), (300, 1000))}
    s2 = {bname(b): sk(b, "net_F", "hold_F") for b in up_to_1000}
    s3 = {bname(b): div["net_R"][bname(b)] for b in F_BINS}
    s4 = {bname((300, 1000)): sk((300, 1000), "net_F", "net_R")}
    order = [bname(b) for b in G_BINS]
    s5 = {str(g): {"returned_by_bin": ret[str(g)]["returned_by_bin"], "first_bin_meeting_rule": ret[str(g)]["first_bin_meeting_rule"],
                   "ratio_G_over_U_by_bin": {b: ret[str(g)]["bins"][b]["ratio"] for b in order}} for g in GAPS}
    res["statements"] = {
        "S1": {"statement": "network F over twin F: MSE-reduction interval above 0 in (100, 300] and (300, 1000] ms",
               "skill": s1, "holds": above(s1), "expectation": "holds"},
        "S2": {"statement": "network F over hold F: interval above 0 in every bin up to 1,000 ms",
               "skill": s2, "holds": above(s2), "expectation": "holds"},
        "S3": {"statement": "the network's R is divergent in no bin up to 3,000 ms", "by_bin": s3,
               "evaluable_bins": sum(v["evaluable"] for v in s3.values()),
               "holds": bool(not any(v["divergent"] for v in s3.values())), "expectation": "holds (weak expectation)"},
        "S4": {"statement": "network F over network R: interval above 0 in (300, 1000] ms",
               "skill": s4, "holds": above(s4), "expectation": "holds"},
        "S5": {"statement": "for g = 0.1, 0.3 and 1.0 s the network has returned by (30, 100] ms", "by_gap": s5,
               "holds": bool(all(v["returned_by_bin"] is not None and order.index(v["returned_by_bin"]) <= order.index("(30, 100]")
                                 for v in s5.values())), "expectation": "holds"}}
    # training range of the staleness input and the cut table
    tr, flags = {}, {}
    for fk, m in metas.items():
        tr[fk] = m["train_range"]
    for b in F_BINS:
        # addendum 1, C13 as changed 2026-11-19 22:56: outside when the lower edge exceeds the p99.9 of ANY fold
        n_out = sum(b[0] > tr[fk]["flag_rows"]["p99.9"] for fk in tr)
        flags[bname(b)] = {"folds_outside": int(n_out), "folds": len(tr),
                           "p99.9_per_fold": {fk: tr[fk]["flag_rows"]["p99.9"] for fk in tr},
                           "outside_the_training_range": bool(n_out > 0)}
    res["training_range"] = {"per_fold": tr, "bins": flags}
    res["cuts"] = {fk: {"kept": m["kept_cuts"], "test_files": m["test_files"]} for fk, m in metas.items()}
    res["precision"] = sorted({m.get("precision", "unrecorded") for m in metas.values()})
    res["checks"] = {fk: {"check1_failed": m["check1"]["failed"], "check2": m["check2"],
                          "checks_3_to_5_pass": m.get("checks_3_to_5", {}).get("passes")} for fk, m in metas.items()}
    if plumb:
        bad = []
        for b in F_BINS:
            c = res["F"][bname(b)]
            for p in pairs_F:
                v = c[f"skill|{p[0]}|{p[1]}"]
                if np.isfinite(v[0]) and not np.allclose(v, 0.0):
                    bad.append(("skill", bname(b), p))
            for p in F_RATIOS:
                v = c[f"ratio|{p[0]}|{p[1]}"]
                if np.isfinite(v[0]) and not np.allclose(v, 1.0):
                    bad.append(("ratio", bname(b), p))
        res["plumbing"] = {"constant_estimators_give_skill_0_and_ratio_1": not bad, "violations": bad[:20]}
    out_json.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    out_md.write_text(render(res, ests_F, pairs_F), encoding="utf-8", newline="\n")
    if plumb:
        print(json.dumps(res["plumbing"]), flush=True)
    else:
        print(json.dumps({k: v["holds"] for k, v in res["statements"].items()}), flush=True)
    return res


# ------------------------------------------------------------------------------------------ markdown
LABEL = {"net": "network", "twin": "twin", "hold": "hold", "gp": "causal GP", "kf0": "KF0", "kfx": "KFX"}


def lab(e):
    a, s = e.rsplit("_", 1)
    return f"{LABEL[a]} {s}"


def render(res, ests_F, pairs_F):
    f1 = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"                       # noqa: E731
    f2 = lambda x: f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"                       # noqa: E731
    pc = lambda x: f"{100 * x[0]:+.1f} [{100 * x[1]:+.1f}, {100 * x[2]:+.1f}]"  # noqa: E731
    k3 = lambda x: f"{x[0] / 1e3:.2f} [{x[1] / 1e3:.2f}, {x[2] / 1e3:.2f}]"     # noqa: E731
    reps = res["repeats"]
    S = res["statements"]
    L = ["# B.71 — when the CES updates stop: free run, own-estimate feedback, and return", "",
         f"Registered: `{REG}`; implementation choices: `{ADD}`. CPU, frozen networks of the pooled re-partitions "
         f"r{', r'.join(map(str, reps))} (five folds each), D1 rows, T_i, cut population. Unit = (scored row, cut). "
         "e = time since the last T_i sample available before the cut (F, R); s = time since updates resumed (G). "
         f"Intervals: 95 %, two-level bootstrap (re-partition, then its discharges), {res['n_boot']:,} draws per cell; "
         "point = mean over the re-partitions. MSE reduction = 100 x (1 - MSE_a / MSE_b).", "",
         f"Filters: {res['filters']}. Arithmetic of the forward passes: {', '.join(res['precision'])} (addendum 1, C16).",
         "", "## Statements", ""]
    verdict = lambda h: "**holds**" if h else "**does not hold**"               # noqa: E731
    L += ["MSE reduction in % (point [95 % interval]); a statement on a reduction holds when every named interval lies "
          "above 0.", "", "| id | statement | readout | verdict | registered expectation |", "|---|---|---|---|---|"]
    for sid in ("S1", "S2", "S4"):
        st = S[sid]
        L.append(f"| {sid} | {st['statement']} | " + "; ".join(f"{b} ms: {pc(v)}" for b, v in st["skill"].items())
                 + f" | {verdict(st['holds'])} | {st['expectation']} |")
    st = S["S3"]
    L.append(f"| S3 | {st['statement']} | " + "; ".join(
        f"{b}: MSE(R)/MSE(hold F) {f2(v['ratio_mse_over_hold_F'])}, outside [0, 3,000] eV {100 * v['share_outside_0_3000_eV']:.2f} %"
        + (" (divergent)" if v["divergent"] else "") for b, v in st["by_bin"].items())
        + f" ({st['evaluable_bins']} of {len(st['by_bin'])} bins evaluable) | {verdict(st['holds'])} | {st['expectation']} |")
    st = S["S5"]
    L.append(f"| S5 | {st['statement']} | " + "; ".join(
        f"g = {g} ms: returned by {v['returned_by_bin'] or 'no bin'} (first bin meeting the rule {v['first_bin_meeting_rule'] or 'none'}; "
        f"MSE(G)/MSE(U) in (30, 100] {f2(v['ratio_G_over_U_by_bin']['(30, 100]'])})" for g, v in st["by_gap"].items())
        + f" | {verdict(st['holds'])} | {st['expectation']} |")
    L += ["", "S1 to S5 are judged separately, without a correction for five statements (registration section 5)."]
    L += ["", "## F and R by time since the last available value (e)", ""]
    hdr = "| e (ms) | pairs (per re-partition) | discharges | " + " | ".join(lab(e) for e in ests_F) + " |"
    for title, key, fmt in (("MSE (10³ eV²)", "mse", k3), ("RMSE (eV)", "rmse", f1), ("mean signed error (eV)", "bias", f1)):
        L += [f"**{title}**", "", hdr, "|" + "---|" * (3 + len(ests_F))]
        for b, c in res["F"].items():
            pr = c["per_repeat"]
            L.append(f"| {b} | {' / '.join(str(pr[str(r)]['pairs']) for r in reps)} | "
                     f"{' / '.join(str(pr[str(r)]['discharges']) for r in reps)} | "
                     + " | ".join(fmt(c[f"{key}|{e}"]) for e in ests_F) + " |")
        L.append("")
    L += ["**median absolute error (eV; point, mean over re-partitions)**", "",
          "| e (ms) | " + " | ".join(lab(e) for e in ests_F) + " |", "|" + "---|" * (1 + len(ests_F))]
    for b, c in res["F"].items():
        L.append(f"| {b} | " + " | ".join(f"{c[f'median_abs|{e}']:.1f}" for e in ests_F) + " |")
    L += ["", "**share of estimates outside [0, 3,000] eV**", "",
          "| e (ms) | " + " | ".join(lab(e) for e in ests_F) + " |", "|" + "---|" * (1 + len(ests_F))]
    for b, c in res["F"].items():
        L.append(f"| {b} | " + " | ".join(f"{c[f'out|{e}'][0]:.4f}" for e in ests_F) + " |")
    L += ["", "**Pairs: MSE reduction (%) and RMSE difference (eV)**", "",
          "| e (ms) | " + " | ".join(f"{lab(a)} vs {lab(b)}" for a, b in pairs_F) + " |", "|" + "---|" * (1 + len(pairs_F))]
    for b, c in res["F"].items():
        L.append(f"| {b} | " + " | ".join(f"{pc(c[f'skill|{a}|{bb}'])}; {f1(c[f'rmse_diff|{a}|{bb}'])}"
                                          for a, bb in pairs_F) + " |")
    L += ["", "**Divergence rule (R): MSE over the MSE of hold F; share outside [0, 3,000] eV**", "",
          "| e (ms) | network R: ratio | share | divergent | twin R: ratio | share | divergent | network F over U: MSE ratio |",
          "|---|---|---|---|---|---|---|---|"]
    for b in res["F"]:
        dn, dt = res["divergence"]["net_R"][b], res["divergence"]["twin_R"][b]
        L.append(f"| {b} | {f2(dn['ratio_mse_over_hold_F'])} | {dn['share_outside_0_3000_eV']:.4f} | {dn['divergent']} | "
                 f"{f2(dt['ratio_mse_over_hold_F'])} | {dt['share_outside_0_3000_eV']:.4f} | {dt['divergent']} | "
                 f"{f2(res['F'][b]['ratio|net_F|net_U'])} |")
    L += ["", "## G: the return after updates resume (s = time since c + g)", "",
          "| g (ms) | s (ms) | pairs (per re-partition) | MSE network G (10³ eV²) | MSE network U (10³ eV²) | "
          "MSE(G) / MSE(U) | rule met (upper < 1.10) | RMSE difference (eV) | median abs. error G / U (eV) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in res["G"]:
        for b, c in res["G"][g].items():
            pr = c["per_repeat"]
            L.append(f"| {g} | {b} | {' / '.join(str(pr[str(r)]['pairs']) for r in reps)} | {k3(c['mse|net_G'])} | "
                     f"{k3(c['mse|net_U'])} | {f2(c['ratio|net_G|net_U'])} | {res['return'][g]['bins'][b]['rule_met']} | "
                     f"{f1(c['rmse_diff|net_G|net_U'])} | {c['median_abs|net_G']:.1f} / {c['median_abs|net_U']:.1f} |")
    L += ["", "| g (ms) | first bin meeting the rule | returned by (rule met there and in every later bin) |", "|---|---|---|"]
    for g, v in res["return"].items():
        L.append(f"| {g} | {v['first_bin_meeting_rule']} | {v['returned_by_bin']} |")
    L += ["", "## Training range of the T_i staleness input (ms; training discharges of each fold, rows with the flag 1)", "",
          "| fold | rows | p50 | p99 | p99.9 | max | p99.9 on rows with T_i measured |", "|---|---|---|---|---|---|---|"]
    for fk, v in res["training_range"]["per_fold"].items():
        a, m = v["flag_rows"], v["measured_rows"]
        L.append(f"| {fk} | {a['n']:,} | {a['p50']:.0f} | {a['p99']:.0f} | {a['p99.9']:.0f} | {a['max']:.0f} | {m['p99.9']:.0f} |")
    folds = list(res["training_range"]["per_fold"])
    L += ["", "A bin is outside the training range when its lower edge exceeds the p99.9 of any fold (addendum 1, C13).", "",
          "| e bin (ms) | folds whose p99.9 lies below the lower edge | outside the training range | "
          + " | ".join(f"p99.9 {fk}" for fk in folds) + " |", "|" + "---|" * (3 + len(folds))]
    for b, v in res["training_range"]["bins"].items():
        L.append(f"| {b} | {v['folds_outside']} of {v['folds']} | {v['outside_the_training_range']} | "
                 + " | ".join(f"{v['p99.9_per_fold'][fk]:.0f}" for fk in folds) + " |")
    L += ["", "## Cuts and checks", "", "| fold | test discharges | kept cuts | check 1 | check 2 (share within 0.5 eV) | checks 3-5 |",
          "|---|---|---|---|---|---|"]
    for fk, v in res["cuts"].items():
        ch = res["checks"][fk]
        L.append(f"| {fk} | {v['test_files']} | {v['kept']} | {'passes' if not ch['check1_failed'] else 'FAILS'} | "
                 f"{ch['check2'].get('share_within_0.5_eV', float('nan')):.5f} | {ch['checks_3_to_5_pass']} |")
    uo = res["units_outside_bins"]
    L += ["", f"F/R units with e above 3,000 ms (outside every bin, not read): "
          + ", ".join(f"r{r}: {uo['F_e_gt_3000ms'][str(r)]:,}" for r in reps) + "."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="analyze")
    ap.add_argument("--plumb", default=None)
    args, _ = ap.parse_known_args()
    if args.plumb:
        d = Path(args.plumb)
        analyze(d, d / "plumb.json", d / "plumb.md", plumb=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
