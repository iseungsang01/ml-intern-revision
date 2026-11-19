"""B.72: the D1 result of B.70 re-read under the validation conventions of other fields
(PREREGISTRATION_B72.md with PREREGISTRATION_B72_ADDENDUM1.md).  CPU only; nothing is trained and no network is run.

R1 field metrics (R2, correlation, relative error, Murphy's decomposition); R2 one score per discharge; R3 the pooled
skill under (a) the two-level bootstrap of B.70 (reproduced draw for draw), (b) a single-level discharge bootstrap,
(c) a cluster-robust paired t-test with a linearised skill interval, (d) the corrected resampled t-test over the 15
folds, (e) Wilcoxon; R4 the model confidence set (arch.bootstrap.MCS, method R); R5 bootstrap p-values of the seven
D1 statements with Holm's correction.  The stored arrays are read through analyze_b70 (imported, not copied).

    py ces_prediction/experiments/b72_methods/run_b72.py
Output: docs/paper/b72_methods.json / .md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
THREADS = "2"
_TV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _v in _TV:
    os.environ[_v] = THREADS

import json  # noqa: E402
import sys  # noqa: E402
import time as _time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(EXP / "b70_d1"))
import analyze_b70 as AB  # noqa: E402  (sets the BLAS variables to 4 on import)

for _v in _TV:                                    # back to 2 before torch is imported (stage2_b64.ctx)
    os.environ[_v] = THREADS

S2, ST, RB, A39 = AB.S2, AB.ST, AB.RB, AB.A39
B41 = S2.B54.B41

REG = "ces_prediction/experiments/PREREGISTRATION_B72.md"
ADD = "ces_prediction/experiments/PREREGISTRATION_B72_ADDENDUM1.md"
PAPER = REPO / "docs" / "paper"
OUT_JSON, OUT_MD = PAPER / "b72_methods.json", PAPER / "b72_methods.md"
B70_JSON = PAPER / "b70_d1.json"
KF_DIR = RB.DATA / ".b70p2" / "b56"
REPS = (2, 3, 4)
K = 5
N_BOOT = 20_000
SEED = 20261172
MIN_ROWS = 20
ALPHA = 0.05
TOL_A = 0.001
T0 = _time.time()

LAB = dict(AB.LAB)
LAB.update(c="network", kf0="KF0", kfx="KFX")
SIGNED_NET = ("c2", "c", "t", "t2")
FILTERS = ("kf0", "kfx")
R1_ESTS = ("c2", "c", "t", "t2", "gp", "hold", "off", "pchip", "kf0", "kfx")
R3_PAIRS = [("c2", "gp"), ("c2", "t2"), ("c", "t"), ("c", "gp"), ("c2", "c"), ("c2", "off")]
STMT = {("c2", "gp"): "D1-1", ("c2", "t2"): "D1-2", ("c", "t"): "D1-3", ("c", "gp"): "D1-4"}
R2_PAIRS = [("c2", "gp"), ("c2", "t2"), ("c2", "c"), ("c2", "off"), ("c2", "hold"), ("c2", "kfx"), ("c", "t"), ("c", "gp")]
SET_C = ("c2", "c", "t", "t2", "gp", "hold", "kf0", "kfx")
SET_A = SET_C + ("off", "pchip")
METHODS = ("a", "b", "c", "d", "e")
MLAB = {"a": "(a) two-level bootstrap (B.70)", "b": "(b) discharge bootstrap", "c": "(c) cluster-robust paired t",
        "d": "(d) corrected resampled t (15 folds)", "e": "(e) Wilcoxon signed-rank"}
EXPECT = {"V1": "holds for D1-1, D1-2, D1-4; for D1-3 holds under (b) and (c), uncertain under (d) and (e)",
          "V2": "holds", "V3": "holds",
          "V4": "holds over the causal GP; weak expectation over the twin",
          "V5": "holds for (b) and (c); uncertain for (d)"}


def log(m):
    print(f"[{_time.time() - T0:6.0f}s] {m}", flush=True)


def rng_for(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def ci(b):
    return A39.ci(b)


# ------------------------------------------------------------------------------------------ data
def load():
    D, _ = AB.load_all()
    chk = {"stage2_vs_rescore": {}, "filters": {}}
    filt_ok = True
    for r in REPS:
        est = {e: [] for e in SIGNED_NET + FILTERS}
        for k in range(K):
            st = np.load(ST.CACHE / f"stage2_r{r}_f{k}.npz")
            zc = ST.load_npz(RB.new_dir("b39", "v2", r, k) / "comparison_errors_test.npz")
            same = bool(np.array_equal(st["shot"], zc["shot"]) and np.array_equal(st["row"], zc["row"]))
            chk["stage2_vs_rescore"][f"r{r}f{k}"] = {"same_rows": same,
                                                    "max_abs_y_diff_eV": float(np.max(np.abs(st["y"] - zc["y"])))}
            for e in SIGNED_NET:
                est[e].append(st[f"est_{e}"].astype(np.float64))
            p = KF_DIR / f"r{r}_f{k}.npz"
            rec = {"file_exists": p.exists()}
            if p.exists():
                z = np.load(p)
                rec.update(same_rows=bool(np.array_equal(z["shot"], st["shot"]) and np.array_equal(z["row"], st["row"])),
                           max_abs_y_diff_eV=float(np.max(np.abs(z["y"] - st["y"]))) if len(z["y"]) == len(st["y"]) else None,
                           rows_unlocated=int((z["gi"] < 0).sum()),
                           finite=bool(np.isfinite(z["est_kf0"]).all() and np.isfinite(z["est_kfx"]).all()))
                rec["passes"] = bool(rec["same_rows"] and rec["max_abs_y_diff_eV"] is not None
                                     and rec["max_abs_y_diff_eV"] <= 1e-6 and rec["rows_unlocated"] == 0 and rec["finite"])
            else:
                rec["passes"] = False
            chk["filters"][f"r{r}f{k}"] = rec
            filt_ok &= rec["passes"]
            for e in FILTERS:
                est[e].append(np.asarray(z[f"est_{e}"], np.float64) if rec["passes"] else np.full(len(st["y"]), np.nan))
        D[r]["est"] = {e: np.concatenate(v) for e, v in est.items()}
    chk["filters_included"] = bool(filt_ok)
    for r in REPS:
        d = D[r]
        if filt_ok:
            for e in FILTERS:
                d["se"][e] = (d["est"][e] - d["y"]) ** 2
        else:
            for e in FILTERS:
                d["est"].pop(e)
    # signed estimates against the squared errors of the statements
    chk["signed_vs_se_max_abs_eV"] = {str(r): {e: float(np.max(np.abs(np.abs(D[r]["est"][e] - D[r]["y"])
                                                                      - np.sqrt(D[r]["se"][e]))))
                                               for e in SIGNED_NET} for r in REPS}
    # the same rows, values and references in every re-partition
    key = {r: D[r]["fidx"] * 1_000_000 + D[r]["grow"] for r in REPS}
    o = {r: np.argsort(key[r], kind="stable") for r in REPS}
    same_rows = all(np.array_equal(key[r][o[r]], key[2][o[2]]) for r in REPS)
    chk["same_rows_every_repartition"] = bool(same_rows)
    chk["same_discharges_every_repartition"] = bool(all(np.array_equal(D[r]["u"], D[2]["u"]) for r in REPS))
    if same_rows:
        chk["max_abs_y_diff_across_repartitions_eV"] = float(max(np.max(np.abs(D[r]["y"][o[r]] - D[2]["y"][o[2]])) for r in REPS))
        chk["max_abs_reference_se_diff_across_repartitions"] = {
            q: float(max(np.max(np.abs(D[r]["se"][q][o[r]] - D[2]["se"][q][o[2]])) for r in REPS))
            for q in ("hold", "gp", "off", "pchip")}
    chk["min_measured_value_eV"] = float(min(D[r]["y"].min() for r in REPS))
    if not (same_rows and chk["same_discharges_every_repartition"]
            and all(v["same_rows"] and v["max_abs_y_diff_eV"] <= 1e-6 for v in chk["stage2_vs_rescore"].values())):
        raise SystemExit("FATAL: the D1 rows are not the same in the two sources or across the re-partitions")
    ests = tuple(e for e in R1_ESTS if e in D[2]["se"])
    return D, chk, ests


def per_discharge(D, ests):
    n = D[2]["n"]
    nd = {r: np.bincount(D[r]["f"], minlength=n) for r in REPS}
    if not all(np.array_equal(nd[r], nd[2]) for r in REPS):
        raise SystemExit("FATAL: n(d) differs between re-partitions")
    S = {e: np.mean([np.bincount(D[r]["f"], D[r]["se"][e], n) for r in REPS], 0) for e in ests}
    y, f = D[2]["y"], D[2]["f"]
    return {"S": S, "n": nd[2].astype(float), "Sy": np.bincount(f, y, n), "Syy": np.bincount(f, y * y, n), "N": n}


def counts(rng, n, nboot):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(np.float64)


# ------------------------------------------------------------------------------------------ R1
def r1(D, PD, ests, Wb):
    out = {}
    den = Wb @ PD["Syy"] - (Wb @ PD["Sy"]) ** 2 / (Wb @ PD["n"])
    for e in ests:
        per = {}
        for r in REPS:
            d = D[r]
            y, se = d["y"], d["se"][e]
            sst = float(np.sum((y - y.mean()) ** 2))
            q = {"R2": 1.0 - float(se.sum()) / sst}
            rel = np.sqrt(se) / y
            q["rel_mean"], q["rel_median"] = float(np.mean(rel)), float(np.median(rel))
            if e in d["est"]:
                fz = d["est"][e]
                mf, mo, sf, so = fz.mean(), y.mean(), fz.std(), y.std()
                rr = float(np.mean((fz - mf) * (y - mo)) / (sf * so))
                q.update(corr=rr, r2_corr=rr ** 2, cond_bias=float((rr - sf / so) ** 2),
                         mean_bias=float(((mf - mo) / so) ** 2), slope=float(rr * so / sf))
                q["murphy_identity_dev"] = float(abs(q["R2"] - (q["r2_corr"] - q["cond_bias"] - q["mean_bias"])))
            per[str(r)] = q
        m = {kk: float(np.mean([per[str(r)][kk] for r in REPS])) for kk in per["2"]}
        draws = 1.0 - (Wb @ PD["S"][e]) / den
        m["R2_ci"] = ci(draws)
        m["per_repeat"] = per
        out[e] = m
    return out


# ------------------------------------------------------------------------------------------ R2
def wmedian_draws(v, W):
    o = np.argsort(v, kind="stable")
    vs = v[o]
    c = np.cumsum(W[:, o], 1)
    return vs[(c >= 0.5 * c[:, -1:]).argmax(1)]


def r2(PD):
    elig = PD["n"] >= MIN_ROWS
    idx = np.flatnonzero(elig)
    W2 = counts(rng_for("R2"), len(idx), N_BOOT)
    out = {"min_rows": MIN_ROWS, "discharges_total": int(PD["N"]), "discharges_eligible": int(len(idx)),
           "discharges_left_out": int((~elig).sum()), "pairs": {}}
    for a, b in R2_PAIRS:
        if a not in PD["S"] or b not in PD["S"]:
            continue
        SA, SB, nn = PD["S"][a][idx], PD["S"][b][idx], PD["n"][idx]
        ok = SB > 0
        rho = 1.0 - SA[ok] / SB[ok]
        Wk = W2[:, ok]
        med = wmedian_draws(rho, Wk)
        share = (Wk @ (rho > 0).astype(float)) / Wk.sum(1)
        mA, mB = SA / nn, SB / nn
        ew = 1.0 - (W2 @ mA) / (W2 @ mB)
        out["pairs"][f"{a}|{b}"] = {
            "discharges_with_zero_reference": int((~ok).sum()),
            "median_rho": [float(np.median(rho)), *ci(med)],
            "share_rho_above_0": [float(np.mean(rho > 0)), *ci(share)],
            "equal_weight_mse": {"A": float(mA.mean()), "B": float(mB.mean())},
            "equal_weight_skill": [float(1.0 - mA.mean() / mB.mean()), *ci(ew)],
            "pooled_skill_all_discharges": float(1.0 - PD["S"][a].sum() / PD["S"][b].sum()),
            "quartiles_rho": [float(x) for x in np.quantile(rho, [0.25, 0.5, 0.75])]}
    return out


# ------------------------------------------------------------------------------------------ R3 / R5 draws
def skill_draws(D, mask_fn, pairs, rng, nboot=N_BOOT):
    """stage2_b64.cell restricted to the MSE skill, keeping the draws: the same weights (drawn first, as cell does),
    the same per-discharge sums and the same products, so value and interval equal cell's."""
    W = S2.weights(rng, REPS, {r: D[r]["n"] for r in REPS}, nboot)
    need = sorted({e for p in pairs for e in p})
    boot = {p: np.empty(nboot) for p in pairs}
    per = {p: {} for p in pairs}
    for r in REPS:
        d = D[r]
        m = mask_fn(r)
        f, n = d["f"][m], d["n"]
        S = {e: np.bincount(f, d["se"][e][m], n) for e in need}
        sel, Wr = W[r]
        WS = {e: Wr @ S[e] for e in need}
        for a, b in pairs:
            per[(a, b)][str(r)] = float(1 - S[a].sum() / S[b].sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[(a, b)][sel] = 1 - WS[a] / WS[b]
    val = {p: [float(np.mean(list(per[p].values()))), *ci(boot[p])] for p in pairs}
    return val, per, boot


def median_ratio_draws(D, rng, nboot=N_BOOT):
    """stage2_b64.medians for the two-stage estimator and hold only, keeping the draws (same weights)."""
    W = S2.weights(rng, REPS, {r: D[r]["n"] for r in REPS}, nboot)
    bq = {e: np.empty(nboot) for e in ("c2", "hold")}
    pt = {e: [] for e in ("c2", "hold")}
    for r in REPS:
        d = D[r]
        sel, Wr = W[r]
        for e in ("c2", "hold"):
            ae = np.sqrt(d["se"][e])
            bq[e][sel] = B41.wquant(ae, d["f"], Wr, [0.5])[:, 0]
            pt[e].append(float(np.median(ae)))
        log(f"medians r{r} done")
    draws = bq["c2"] / bq["hold"]
    return [float(np.mean([x / y for x, y in zip(pt["c2"], pt["hold"])])), *ci(draws)], draws


def r3(D, PD, Wb, a_val, a_per):
    out = {}
    n = PD["N"]
    tq = stats.t.ppf(0.975, n - 1)
    for a, b in R3_PAIRS:
        SA, SB = PD["S"][a], PD["S"][b]
        q = {"a": {"point": a_val[(a, b)][0], "ci": a_val[(a, b)][1:], "per_repeat": a_per[(a, b)]}}
        # (b)
        db = 1.0 - (Wb @ SA) / (Wb @ SB)
        q["b"] = {"point": float(1.0 - SA.sum() / SB.sum()), "ci": ci(db)}
        # (c)
        Dd = SB - SA
        tstat = float(Dd.mean() / (Dd.std(ddof=1) / np.sqrt(n)))
        s_hat = float(Dd.sum() / SB.sum())
        se_lin = float(np.std(Dd - s_hat * SB, ddof=1) / (np.sqrt(n) * SB.mean()))
        q["c"] = {"point": s_hat, "ci": [s_hat - tq * se_lin, s_hat + tq * se_lin], "t": tstat,
                  "p": float(2 * stats.t.sf(abs(tstat), n - 1)), "se_linearised": se_lin, "n": int(n)}
        # (d)
        fs = []
        for r in REPS:
            d = D[r]
            for k in range(K):
                m = d["fold"] == k
                fs.append(float(1.0 - d["se"][a][m].sum() / d["se"][b][m].sum()))
        fs = np.array(fs)
        mb = float(fs.mean())
        var = (1.0 / len(fs) + 1.0 / 4.0) * float(fs.var(ddof=1))
        t14 = stats.t.ppf(0.975, len(fs) - 1)
        td = mb / np.sqrt(var)
        q["d"] = {"point": mb, "ci": [mb - t14 * np.sqrt(var), mb + t14 * np.sqrt(var)], "t": float(td),
                  "p": float(2 * stats.t.sf(abs(td), len(fs) - 1)), "fold_skills": fs.tolist(),
                  "sd_fold_skills": float(fs.std(ddof=1)), "uncorrected_ci": [mb - t14 * np.sqrt(var / (1 + 15 / 4)),
                                                                              mb + t14 * np.sqrt(var / (1 + 15 / 4))]}
        # (e)
        w = stats.wilcoxon(Dd, alternative="two-sided")
        nz = Dd[Dd != 0]
        rk = stats.rankdata(np.abs(nz))
        tp, tm = float(rk[nz > 0].sum()), float(rk[nz < 0].sum())
        q["e"] = {"p": float(w.pvalue), "statistic": float(w.statistic), "T_plus": tp, "T_minus": tm,
                  "zero_differences": int((Dd == 0).sum()), "share_D_above_0": float(np.mean(Dd > 0))}
        wa = q["a"]["ci"][1] - q["a"]["ci"][0]
        for mth in ("a", "b", "c", "d"):
            q[mth]["width"] = float(q[mth]["ci"][1] - q[mth]["ci"][0])
            q[mth]["width_over_a"] = float(q[mth]["width"] / wa)
            q[mth]["holds"] = bool(q[mth]["ci"][0] > 0)
        q["e"]["holds"] = bool(q["e"]["p"] < ALPHA and tp > tm)
        out[f"{a}|{b}"] = q
    return out


# ------------------------------------------------------------------------------------------ R4
def mcs(PD, members, loss, name):
    import pandas as pd
    import arch
    from arch.bootstrap import MCS
    cols = [e for e in members if e in PD["S"]]
    L = np.column_stack([PD["S"][e] if loss == 1 else PD["S"][e] / PD["n"] for e in cols])
    df = pd.DataFrame(L, columns=[LAB[e] for e in cols])
    m = MCS(df, size=ALPHA, reps=N_BOOT, block_size=1, method="R", bootstrap="stationary", seed=rng_for(name))
    m.compute()
    pv = m.pvalues["Pvalue"]
    inv = {LAB[e]: e for e in cols}
    return {"implementation": f"arch {arch.__version__}, MCS(size=0.05, reps={N_BOOT}, block_size=1, method='R', "
                              "bootstrap='stationary')",
            "estimators": cols, "elimination_order": [inv[x] for x in pv.index],
            "pvalues": {inv[x]: float(v) for x, v in pv.items()},
            "included": [inv[x] for x in m.included], "mean_loss": {e: float(L[:, j].mean()) for j, e in enumerate(cols)}}


# ------------------------------------------------------------------------------------------ R5
def boot_p(draws, null):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    p = 2.0 * min(np.mean(d <= null), np.mean(d >= null))
    return float(min(1.0, max(p, 1.0 / (N_BOOT + 1)))), int(len(d))


def holm(ps):
    keys = sorted(ps, key=lambda k_: ps[k_])
    m = len(keys)
    out, run = {}, 0.0
    for j, k_ in enumerate(keys):
        run = max(run, min(1.0, (m - j) * ps[k_]))
        out[k_] = run
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    B70 = json.loads(B70_JSON.read_text(encoding="utf-8"))
    D, chk, ests = load()
    log(f"loaded; estimators {ests}; filters included {chk['filters_included']}")
    PD = per_discharge(D, ests)
    Wb = counts(rng_for("R3b"), PD["N"], N_BOOT)
    R1 = r1(D, PD, ests, Wb)
    log("R1 done")
    R2 = r2(PD)
    log("R2 done")
    a_val, a_per, a_boot = skill_draws(D, lambda r: np.ones(len(D[r]["y"]), bool), R3_PAIRS, AB.rng_for("all"))
    chk["a_vs_b70"] = {f"{a}|{b}": {"b72": a_val[(a, b)], "b70": B70["all_rows"][f"skill_mse|{a}|{b}"],
                                    "max_abs_diff": float(max(abs(x - y) for x, y in
                                                              zip(a_val[(a, b)], B70["all_rows"][f"skill_mse|{a}|{b}"])))}
                       for a, b in R3_PAIRS}
    chk["a_reproduces_b70"] = bool(all(v["max_abs_diff"] <= TOL_A for v in chk["a_vs_b70"].values()))
    R3 = r3(D, PD, Wb, a_val, a_per)
    log("R3 done")
    R4 = {}
    for setn, mem in (("C", SET_C), ("A", SET_A)):
        for loss in (1, 2):
            nm = f"R4|set {setn}|loss {loss}"
            R4[f"{setn}|{loss}"] = mcs(PD, mem, loss, nm)
            log(f"MCS set {setn} loss {loss}: {R4[f'{setn}|{loss}']['included']}")
    # R5: D1-5 and D1-6 draws (B.70's streams)
    v5, d5 = median_ratio_draws(D, AB.rng_for("medians"))
    chk["d1_5_vs_b70"] = {"b72": v5, "b70": B70["statements"]["D1-5"]["value"],
                          "max_abs_diff": float(max(abs(x - y) for x, y in zip(v5, B70["statements"]["D1-5"]["value"])))}
    C = S2.ctx()
    mix, flags = AB.mixture(C, D)
    chk["mixture_flags_vs_b70"] = {str(r): {"b72": int(flags[r].sum()),
                                            "b70": B70["mixture"]["per_repeat"][str(r)]["scored_rows_flagged"]} for r in REPS}
    keep = {r: ~flags[r] for r in REPS}
    v6, _p6, d6 = skill_draws(D, lambda r: keep[r], [("c2", "gp"), ("c2", "t2"), ("c", "gp"), ("c", "t"), ("c2", "c"),
                                                     ("c2", "hold")], AB.rng_for("D1-6"))
    chk["d1_6_vs_b70"] = {
        "D1-1 form": {"b72": v6[("c2", "gp")], "b70": B70["statements"]["D1-6"]["value_D1_1"]},
        "D1-2 form": {"b72": v6[("c2", "t2")], "b70": B70["statements"]["D1-6"]["value_D1_2"]}}
    for v in chk["d1_6_vs_b70"].values():
        v["max_abs_diff"] = float(max(abs(x - y) for x, y in zip(v["b72"], v["b70"])))
    chk["d1_5_d1_6_reproduce_b70"] = bool(chk["d1_5_vs_b70"]["max_abs_diff"] <= TOL_A
                                          and all(v["max_abs_diff"] <= TOL_A for v in chk["d1_6_vs_b70"].values()))
    log("D1-5 / D1-6 draws done")
    R5 = {"tests": {}}
    for (a, b), s in STMT.items():
        p, nd = boot_p(a_boot[(a, b)], 0.0)
        R5["tests"][s] = {"form": f"{LAB[a]} over {LAB[b]}, MSE skill", "value": a_val[(a, b)], "null": 0.0, "p": p,
                          "draws": nd, "draws_at_or_below_null": int(np.sum(a_boot[(a, b)] <= 0))}
    p, nd = boot_p(d5, 1.0)
    R5["tests"]["D1-5"] = {"form": "median |e|, two-stage estimator / hold", "value": v5, "null": 1.0, "p": p, "draws": nd,
                           "draws_at_or_above_null": int(np.sum(d5 >= 1.0))}
    for nm_, pr in (("D1-6 (D1-1 form)", ("c2", "gp")), ("D1-6 (D1-2 form)", ("c2", "t2"))):
        p, nd = boot_p(d6[pr], 0.0)
        R5["tests"][nm_] = {"form": f"{LAB[pr[0]]} over {LAB[pr[1]]}, MSE skill, rows the mixture does not flag",
                            "value": v6[pr], "null": 0.0, "p": p, "draws": nd,
                            "draws_at_or_below_null": int(np.sum(d6[pr] <= 0))}
    hp = holm({k_: v["p"] for k_, v in R5["tests"].items()})
    for k_, v in R5["tests"].items():
        v["p_holm"] = hp[k_]
    # ---- statements
    V = {}
    v1 = {}
    for (a, b), s in STMT.items():
        q = R3[f"{a}|{b}"]
        v1[s] = {m_: bool(q[m_]["holds"]) for m_ in ("b", "c", "d", "e")}
    V["V1"] = {"text": "each of D1-1 to D1-4 keeps its verdict under (b), (c), (d) and (e)", "per_statement": v1,
               "holds": bool(all(all(x.values()) for x in v1.values()))}
    mc = R4["C|1"]
    others = {e: p for e, p in mc["pvalues"].items() if e != "c2"}
    V["V2"] = {"text": "the 95 % model confidence set of set C is the two-stage estimator alone",
               "members_loss1": mc["included"], "members_loss2": R4["C|2"]["included"],
               "largest_p_of_the_others_loss1": float(max(others.values())),
               "holds": bool(mc["included"] == ["c2"])}
    V["V3"] = {"text": "after Holm's correction every one of the seven remains below 0.05",
               "largest_adjusted_p": float(max(hp.values())), "at": max(hp, key=hp.get),
               "holds": bool(max(hp.values()) < ALPHA)}
    g, t2 = R2["pairs"]["c2|gp"]["median_rho"], R2["pairs"]["c2|t2"]["median_rho"]
    V["V4"] = {"text": "the median per-discharge MSE reduction of the two-stage estimator lies above zero (interval), "
                       "over the causal GP and over the two-stage twin",
               "over_gp": g, "over_t2": t2, "holds_over_gp": bool(g[1] > 0), "holds_over_t2": bool(t2[1] > 0),
               "holds": bool(g[1] > 0 and t2[1] > 0)}
    ratios = {f"{STMT[(a, b)]} {m_}": R3[f"{a}|{b}"][m_]["width_over_a"] for (a, b) in STMT for m_ in ("b", "c", "d")}
    V["V5"] = {"text": "no interval of (b), (c) or (d) is more than 1.25 times as wide as (a) for D1-1 to D1-4",
               "width_ratios": ratios, "largest": float(max(ratios.values())), "at": max(ratios, key=ratios.get),
               "holds": bool(max(ratios.values()) <= 1.25)}
    for k_ in V:
        V[k_]["expected"] = EXPECT[k_]
    chk["murphy_identity_max_dev"] = float(max(v["per_repeat"][str(r)].get("murphy_identity_dev", 0.0)
                                              for v in R1.values() for r in REPS))
    o = {"registration": REG, "addendum": ADD, "n_boot": N_BOOT, "seed": SEED, "repeats": list(REPS),
         "estimators": list(ests), "filters_included": chk["filters_included"], "statements": V,
         "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "checks": chk,
         "mixture_fit_ranges": mix["fit_ranges"], "runtime_s": round(_time.time() - T0, 1)}
    OUT_JSON.write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(render(o), encoding="utf-8", newline="\n")
    log("written " + str(OUT_MD) + " | " + ", ".join(f"{k_} {'holds' if v['holds'] else 'does not hold'}"
                                                     for k_, v in V.items()))


# ------------------------------------------------------------------------------------------ render
def s3(x):
    return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def f3(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def pct(x):
    return f"{100 * x[0]:+.1f} [{100 * x[1]:+.1f}, {100 * x[2]:+.1f}]"


def pf(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def hv(b):
    return "holds" if b else "does not hold"


def render(o):
    V, R1, R2, R3, R4, R5, chk = (o[k] for k in ("statements", "R1", "R2", "R3", "R4", "R5", "checks"))
    L = ["# B.72 — the D1 result under other validation conventions", "",
         f"Registered: {o['registration']} (with {o['addendum']}). CPU only: the stored D1 estimates and squared errors "
         "of B.70 (T_i, pooled r2-r4, 222,672 rows per re-partition, 635 discharges) are re-read; nothing is trained and "
         "no network is run. Skill of A over B = 1 - (sum of squared errors of A) / (that of B) = the MSE reduction. "
         f"S(m, d) = the sum of squared errors of estimator m over the D1 rows of discharge d, mean over r2-r4. "
         f"{o['n_boot']:,} draws for every bootstrap; 95 % intervals as [2.5, 97.5] percentiles unless stated. "
         f"Filters KF0 and KFX: {'included (B.70 phase 2 stored their estimates on the D1 rows; every fold maps exactly)' if o['filters_included'] else 'left out (their stored rows do not map onto the D1 rows)'}.",
         "", "## Statements", "",
         "| statement | value | verdict | registered expectation |", "|---|---|---|---|"]
    v = V["V1"]
    cells = "; ".join(f"{s}: " + ", ".join(f"({m_}) {hv(x)}" for m_, x in d.items()) for s, d in v["per_statement"].items())
    L.append(f"| **V1** {v['text']} | {cells} | **{hv(v['holds'])}** | {v['expected']} |")
    v = V["V2"]
    L.append(f"| **V2** {v['text']} | set C, loss S(m, d): {{{', '.join(LAB[e] for e in v['members_loss1'])}}}; largest MCS "
             f"p-value of the others {pf(v['largest_p_of_the_others_loss1'])}; loss S(m, d) / n(d): "
             f"{{{', '.join(LAB[e] for e in v['members_loss2'])}}} | **{hv(v['holds'])}** | {v['expected']} |")
    v = V["V3"]
    L.append(f"| **V3** {v['text']} | largest Holm-adjusted p {pf(v['largest_adjusted_p'])} ({v['at']}) | "
             f"**{hv(v['holds'])}** | {v['expected']} |")
    v = V["V4"]
    L.append(f"| **V4** {v['text']} | median rho(d) over the causal GP {s3(v['over_gp'])} ({hv(v['holds_over_gp'])}); "
             f"over the two-stage twin {s3(v['over_t2'])} ({hv(v['holds_over_t2'])}) | **{hv(v['holds'])}** | {v['expected']} |")
    v = V["V5"]
    L.append(f"| **V5** {v['text']} | largest width ratio {v['largest']:.2f} ({v['at']}); "
             + ", ".join(f"{k_} {x:.2f}" for k_, x in v["width_ratios"].items())
             + f" | **{hv(v['holds'])}** | {v['expected']} |")
    # R1
    L += ["", "## R1. Field metrics (all D1 rows; point = mean over r2-r4)", "",
          "R² = 1 - SSE / SST, SST about the mean of the measured values (sample variance of T_i on the D1 rows); interval "
          "from the discharge bootstrap of R3 (b). Relative error = |estimate - measured| / measured. Murphy's terms (moments "
          "divided by N): R² = r² - conditional bias - mean bias; conditional bias = (r - s_f / s_o)²; mean bias = "
          "((mean f - mean o) / s_o)²; slope = regression slope of measured on estimated (r s_o / s_f). Correlation and "
          "Murphy's terms need the estimate with its sign; for hold and the GP / PCHIP references only squared errors "
          "are stored (-).", "",
          "| estimator | R² [95 %] | correlation r | mean relative error | median relative error | r² | conditional bias | "
          "mean bias | slope |", "|---|---|---|---|---|---|---|---|---|"]
    for e in o["estimators"]:
        q = R1[e]
        rr = (f"{q['corr']:.4f} | " if "corr" in q else "- | ")
        mur = (f"{q['r2_corr']:.4f} | {q['cond_bias']:.5f} | {q['mean_bias']:.5f} | {q['slope']:.3f}"
               if "corr" in q else "- | - | - | -")
        L.append(f"| {LAB[e]} | {q['R2']:.4f} [{q['R2_ci'][0]:.4f}, {q['R2_ci'][1]:.4f}] | {rr}{q['rel_mean']:.4f} | "
                 f"{q['rel_median']:.4f} | {mur} |")
    # R2
    L += ["", "## R2. One score per discharge", "",
          f"Discharges with at least {R2['min_rows']} D1 rows: {R2['discharges_eligible']} of {R2['discharges_total']} "
          f"({R2['discharges_left_out']} left out). rho(d) = 1 - S(A, d) / S(B, d), the MSE reduction of discharge d. "
          "Intervals: the eligible discharges resampled, 20,000 draws. Equal-weight skill = 1 - mean_d MSE_A(d) / "
          "mean_d MSE_B(d), MSE(d) = S(m, d) / n(d). Pooled skill = 1 - sum_d S(A, d) / sum_d S(B, d) over all 635 "
          "discharges (point, for comparison).", "",
          "| A over B | median rho(d) [95 %] | quartiles of rho(d) | share of discharges with rho(d) > 0 [95 %] | "
          "equal-weight MSE skill [95 %] | equal-weight MSE reduction (%) | pooled skill (point) |",
          "|---|---|---|---|---|---|---|"]
    for k_, q in R2["pairs"].items():
        a, b = k_.split("|")
        L.append(f"| {LAB[a]} over {LAB[b]} | {s3(q['median_rho'])} | {q['quartiles_rho'][0]:+.3f} / "
                 f"{q['quartiles_rho'][1]:+.3f} / {q['quartiles_rho'][2]:+.3f} | {f3(q['share_rho_above_0'])} | "
                 f"{s3(q['equal_weight_skill'])} | {pct(q['equal_weight_skill'])} | {q['pooled_skill_all_discharges']:+.3f} |")
    # R3
    L += ["", "## R3. The pooled MSE skill under five inference methods", "",
          "(a) the two-level bootstrap of B.70, draw for draw; (b) discharges resampled on S(m, d); (c) D(d) = S(B, d) - "
          "S(A, d), t-test over the 635 discharges, skill interval by linearising the ratio; (d) the skill of each of "
          "the 15 folds, variance of the mean (1/15 + 1/4) s², Student t with 14 degrees of freedom; (e) Wilcoxon "
          "signed-rank on D(d). Verdict at 5 %: the interval lies above zero; for (e) p < 0.05 with the positive "
          "differences carrying the larger rank sum. Width relative to (a) = width of the interval / width of (a)'s.", "",
          "| A over B (statement) | method | point | 95 % interval | p | width / (a) | verdict |",
          "|---|---|---|---|---|---|---|"]
    for k_, q in R3.items():
        a, b = k_.split("|")
        nm = f"{LAB[a]} over {LAB[b]}" + (f" ({STMT[(a, b)]})" if (a, b) in STMT else "")
        for m_ in METHODS:
            x = q[m_]
            if m_ == "e":
                L.append(f"| {nm} | {MLAB[m_]} | T+ {x['T_plus']:,.0f}, T- {x['T_minus']:,.0f} (share D(d) > 0: "
                         f"{x['share_D_above_0']:.3f}) | - | {pf(x['p'])} | - | {hv(x['holds'])} |")
            else:
                p = pf(x["p"]) if "p" in x else "-"
                L.append(f"| {nm} | {MLAB[m_]} | {x['point']:+.3f} | [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {p} | "
                         f"{x['width_over_a']:.2f} | {hv(x['holds'])} |")
    L += ["", "Fold skills of (d) (r2 f0-f4, r3 f0-f4, r4 f0-f4) and their sd:", ""]
    for k_, q in R3.items():
        a, b = k_.split("|")
        L.append(f"- {LAB[a]} over {LAB[b]}: " + ", ".join(f"{x:+.3f}" for x in q["d"]["fold_skills"])
                 + f"; sd {q['d']['sd_fold_skills']:.3f}; without the correction the interval would be "
                 f"[{q['d']['uncorrected_ci'][0]:+.3f}, {q['d']['uncorrected_ci'][1]:+.3f}].")
    # R4
    impl = R4["C|1"]["implementation"]
    L += ["", "## R4. Model confidence set (95 %)", "",
          f"Implementation: {impl}; the stationary bootstrap with block length 1 resamples discharges independently. "
          "One row per discharge (635). Loss 1 = S(m, d); loss 2 = S(m, d) / n(d) (the per-discharge MSE). An estimator "
          "is in the set when its MCS p-value is at least 0.05; the MCS p-value is the p-value at which it leaves "
          "(running maximum over the eliminations; 1 for the last one left).", "",
          "| estimator | set C, loss 1 | set C, loss 2 | set A, loss 1 | set A, loss 2 |", "|---|---|---|---|---|"]
    allm = list(dict.fromkeys(R4["A|1"]["estimators"]))
    for e in allm:
        cells = []
        for key in ("C|1", "C|2", "A|1", "A|2"):
            q = R4[key]
            if e not in q["pvalues"]:
                cells.append("-")
            else:
                pos = q["elimination_order"].index(e) + 1
                cells.append(f"{pf(q['pvalues'][e])} (leaves {pos} of {len(q['estimators'])})" if e not in q["included"]
                             else f"**in the set** (p {pf(q['pvalues'][e])})")
        L.append(f"| {LAB[e]} | " + " | ".join(cells) + " |")
    for key, nm in (("C|1", "set C, loss 1"), ("C|2", "set C, loss 2"), ("A|1", "set A, loss 1"), ("A|2", "set A, loss 2")):
        L.append("")
        L.append(f"- {nm}: members {{{', '.join(LAB[e] for e in R4[key]['included'])}}}; elimination order "
                 + " -> ".join(LAB[e] for e in R4[key]["elimination_order"]) + ".")
    # R5
    L += ["", "## R5. The seven D1 statements: bootstrap p-values and Holm's correction", "",
          "Two-sided p = 2 x the smaller share of the two-level draws (B.70's draws, reproduced) on either side of the "
          f"null value, not below 1 / {N_BOOT + 1:,}. Holm over the seven.", "",
          "| statement | form | value [95 %] | null | raw p | Holm-adjusted p | adjusted p < 0.05 |",
          "|---|---|---|---|---|---|---|"]
    for k_, q in R5["tests"].items():
        val = f3(q["value"]) if k_ == "D1-5" else s3(q["value"])
        L.append(f"| {k_} | {q['form']} | {val} | {q['null']:g} | {pf(q['p'])} | {pf(q['p_holm'])} | "
                 f"{'yes' if q['p_holm'] < ALPHA else 'no'} |")
    # checks
    L += ["", "## Checks", ""]
    L.append(f"- (a) reproduces `docs/paper/b70_d1.json` (points and bounds, tolerance {TOL_A}): "
             f"{'passes' if chk['a_reproduces_b70'] else 'FAILS'}; largest difference "
             f"{max(v['max_abs_diff'] for v in chk['a_vs_b70'].values()):.1e}.")
    L.append(f"- D1-5 and the two D1-6 forms reproduce `b70_d1.json` (tolerance {TOL_A}): "
             f"{'passes' if chk['d1_5_d1_6_reproduce_b70'] else 'FAILS'}; D1-5 difference "
             f"{chk['d1_5_vs_b70']['max_abs_diff']:.1e}, D1-6 differences "
             + ", ".join(f"{k_} {v['max_abs_diff']:.1e}" for k_, v in chk["d1_6_vs_b70"].items())
             + "; flagged rows per re-partition " + ", ".join(f"r{r} {v['b72']:,} (B.70 {v['b70']:,})"
                                                             for r, v in chk["mixture_flags_vs_b70"].items()) + ".")
    s2r = chk["stage2_vs_rescore"]
    L.append(f"- Rows and measured values of the stage-2 files and the D1 re-scorings: same rows in "
             f"{sum(v['same_rows'] for v in s2r.values())} of {len(s2r)} folds; largest |y difference| "
             f"{max(v['max_abs_y_diff_eV'] for v in s2r.values()):.1e} eV.")
    L.append(f"- Same D1 rows and discharges in r2, r3, r4: {chk['same_rows_every_repartition']} / "
             f"{chk['same_discharges_every_repartition']}; largest |y difference| across re-partitions "
             f"{chk.get('max_abs_y_diff_across_repartitions_eV', float('nan')):.1e} eV; reference squared errors across "
             "re-partitions, largest difference: " + ", ".join(
                 f"{LAB[q]} {x:.1e}" for q, x in chk.get("max_abs_reference_se_diff_across_repartitions", {}).items()) + ".")
    L.append("- Signed estimates against the squared errors of the statements, largest | |estimate - y| - sqrt(se) | (eV): "
             + "; ".join(f"r{r} " + ", ".join(f"{LAB[e]} {x:.1e}" for e, x in v.items())
                         for r, v in chk["signed_vs_se_max_abs_eV"].items()) + ".")
    fl = chk["filters"]
    L.append(f"- Filters: {sum(v['passes'] for v in fl.values())} of {len(fl)} folds map exactly onto the D1 rows "
             f"(same shot and row arrays, y within 1e-6 eV, every row located, every estimate finite) -> "
             f"{'included' if chk['filters_included'] else 'left out'}.")
    L.append(f"- Murphy identity R² = r² - conditional bias - mean bias: largest deviation {chk['murphy_identity_max_dev']:.1e}.")
    L.append(f"- Smallest measured T_i on the D1 rows: {chk['min_measured_value_eV']:.1f} eV (relative errors defined).")
    L.append(f"- Run time {o['runtime_s']:.0f} s (CPU, 2 threads).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
