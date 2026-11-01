"""B.26 (PREREGISTRATION_B26.md): TRAIN-only pre-check battery -- no model, no VAL/TEST row.

A1-A4: photon-statistics consistency of the 10 ms scatter (second differences within blocks).
E1/E1b: ambiguous jump-weight fraction of the B.19 hold-or-jump t filter, and the TRAIN one-step
median AE of the two-component posterior median against the merged mean the filter carries.
E2: rank correlation of T_i and V_rot increments on the largest-|dT_i| rows.
Output: data/.b26_precheck.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/precheck_b26.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_vrot_jump_b23 import DATA, MANIFEST, SEEDS, log_t  # noqa: E402
sys.path.insert(0, str(HERE.parent / "seq"))
from seq_data import N_FAST_CHANNELS, build_blocks, fit_stats, load_grid_files  # noqa: E402

CUT_EV = {"cut": 3000.0, "incl": 0.0}
KB_EV, M_C = 1.602176634e-19, 12 * 1.66053906660e-27
NU, SIGMA_D, P0 = 3.0, 0.05, 1.0
BULK_SIGMA_N = {"CES_TI_cut": 0.116, "CES_TI_incl": 0.071, "CES_VT": 0.060}
REG = {"lam": 0.05, "sigma_j": 1.0}
COL_CARRIED = (N_FAST_CHANNELS + 1, N_FAST_CHANNELS + 4)


def rsd(v):
    v = np.asarray(v, float)
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def sigma_v_kms(t_ev):
    return np.sqrt(np.asarray(t_ev, float) * KB_EV / M_C) / 1e3


def mix_median(w1, m0, v0, m1, v1, iters=60):
    """Median of (1 - w1) N(m0, v0) + w1 N(m1, v1), elementwise, by bisection."""
    s0, s1 = np.sqrt(v0), np.sqrt(v1)
    lo = np.minimum(m0 - 8 * s0, m1 - 8 * s1)
    hi = np.maximum(m0 + 8 * s0, m1 + 8 * s1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = (1 - w1) * ndtr((mid - m0) / s0) + w1 * ndtr((mid - m1) / s1)
        lo, hi = np.where(c < 0.5, mid, lo), np.where(c < 0.5, hi, mid)
    return 0.5 * (lo + hi)


def filter_e1(carried, obs_prev, lam, j, r):
    """B.19 hold-or-jump t recursion (= fit_vrot_jump_b23.scan_loglik, one hypothesis) returning the
    jump weights of scored updates and the next-observation |error| of merged mean vs posterior median."""
    B, L = carried.shape
    x, P, h = np.zeros(B), np.full(B, P0), np.zeros(B)
    med = np.zeros(B)
    seen = np.zeros(B, bool)
    w_all, e_mean, e_med = [], [], []
    for t in range(L):
        if t > 0:
            upd = obs_prev[:, t] > 0.5
            if upd.any():
                y = carried[:, t]
                score = upd & seen
                e_mean.append(np.abs(y - x)[score]); e_med.append(np.abs(y - med)[score])
                e = y - x
                e2 = e * e
                S0, S1 = P + r, P + j + r
                hh = np.clip(h, 1e-6, 1 - 1e-6)
                ll = log_t(e2, S1, NU) - log_t(e2, S0, NU)
                w1 = 1.0 / (1.0 + np.exp(-(np.log(hh) - np.log1p(-hh) + ll)))
                w_all.append(w1[score])
                R0 = r * (NU + e2 / S0) / (NU + 1)
                R1 = r * (NU + e2 / S1) / (NU + 1)
                T0, T1 = P + R0, P + j + R1
                x0, x1 = x + P / T0 * e, x + (P + j) / T1 * e
                v0, v1 = P * R0 / T0, (P + j) * R1 / T1
                xm = (1 - w1) * x0 + w1 * x1
                Pm = (1 - w1) * (v0 + (x0 - xm) ** 2) + w1 * (v1 + (x1 - xm) ** 2)
                mm = mix_median(w1, x0, v0, x1, v1)
                x, P, med = np.where(upd, xm, x), np.where(upd, Pm, P), np.where(upd, mm, med)
                h = np.where(upd, 0.0, h)
                seen |= upd
        P = P + SIGMA_D ** 2
        h = 1 - (1 - h) * (1 - lam)
    cat = lambda z: np.concatenate(z) if z else np.zeros(0)
    return cat(w_all), cat(e_mean), cat(e_med)


def series(blocks, t):
    L = max(len(b["y"]) for b in blocks)
    carried, obs_prev = np.zeros((len(blocks), L)), np.zeros((len(blocks), L))
    for i, b in enumerate(blocks):
        n = len(b["y"])
        carried[i, :n] = b["x"][:, COL_CARRIED[t]]
        obs_prev[i, 1:n] = b["mask"][:-1, t]
    return carried, obs_prev


def photon_rows(file_blocks, mean, std):
    """Per-row second differences / increments in physical units, with the shot index."""
    out = {k: [] for k in ("shot", "eT", "eV", "T", "both", "dT", "dV")}
    for si, blocks in enumerate(file_blocks):
        for b in blocks:
            y = b["y"] * std + mean
            m = b["mask"] > 0.5
            n = len(y)
            for i in range(1, n - 1):
                tri = m[i - 1:i + 2]
                okT, okV = tri[:, 0].all(), tri[:, 1].all()
                if not (okT or okV):
                    continue
                out["shot"].append(si)
                out["eT"].append(y[i, 0] - 0.5 * (y[i - 1, 0] + y[i + 1, 0]) if okT else np.nan)
                out["eV"].append(y[i, 1] - 0.5 * (y[i - 1, 1] + y[i + 1, 1]) if okV else np.nan)
                out["T"].append(y[i, 0] if m[i, 0] else np.nan)
                out["both"].append(bool(okT and okV))
            for i in range(1, n):
                if m[i - 1].all() and m[i].all():
                    out["dT"].append(y[i, 0] - y[i - 1, 0]); out["dV"].append(y[i, 1] - y[i - 1, 1])
    return {k: np.asarray(v, float) for k, v in out.items()}


def one_split(pop, s, grid, dims):
    names = [n for n in json.loads((DATA / MANIFEST[pop].format(s=s) / "split_manifest.json")
                                   .read_text(encoding="utf-8"))["train_files"] if n in grid]
    stats = fit_stats(grid, dims, names)
    mean = np.asarray(stats["target"]["mean"], float)
    std = np.asarray(stats["target"]["std"], float)
    file_blocks = [build_blocks(grid[n], dims, stats, per_shot_norm=True) for n in names]
    blocks = [b for fb in file_blocks for b in fb]
    rec = {"n_files": len(names)}

    pr = photon_rows(file_blocks, mean, std)
    both = pr["both"] > 0.5
    # A1: per-shot ratio
    R = []
    for si in np.unique(pr["shot"][both]):
        k = both & (pr["shot"] == si)
        if k.sum() < 30:
            continue
        tbar = float(np.mean(pr["T"][k]))
        if tbar <= 0:
            continue
        sv = rsd(pr["eV"][k])
        if sv <= 0:
            continue
        R.append((rsd(pr["eT"][k]) / tbar) / (sv / float(sigma_v_kms(tbar))))
    rec["A1"] = {"n_shots": len(R), "R_median": float(np.median(R)) if R else None,
                 "R_iqr": [float(np.quantile(R, 0.25)), float(np.quantile(R, 0.75))] if R else None}
    # A2: |e_V| scale vs T_i deciles
    kv = np.isfinite(pr["eV"]) & np.isfinite(pr["T"]) & (pr["T"] > 0)
    T, eV = pr["T"][kv], pr["eV"][kv]
    edges = np.quantile(T, np.linspace(0, 1, 11))
    xs, ys = [], []
    for d in range(10):
        k = (T >= edges[d]) & ((T < edges[d + 1]) if d < 9 else (T <= edges[d + 1]))
        if k.sum() >= 50:
            xs.append(math.log(float(np.median(T[k])))); ys.append(math.log(1.4826 * float(np.median(np.abs(eV[k])))))
    rec["A2"] = {"slope": float(np.polyfit(xs, ys, 1)[0]), "n_rows": int(kv.sum()),
                 "scale_by_decile_kms": [math.exp(v) for v in ys], "T_by_decile_ev": [math.exp(v) for v in xs]}
    # A3: shared scale and sign on the same rows
    eT, eVb = pr["eT"][both], pr["eV"][both]
    rec["A3"] = {"n_rows": int(both.sum()), "rho_abs": float(spearmanr(np.abs(eT), np.abs(eVb))[0]),
                 "rho_sign": float(spearmanr(eT, eVb)[0])}
    # A4: fit-failure contagion (inclusive only)
    if pop == "incl":
        hot = kv & (pr["T"] > 3000.0)
        cold = kv & (pr["T"] <= 3000.0)
        rec["A4"] = {"n_hot": int(hot.sum()),
                     "ratio": float(np.median(np.abs(pr["eV"][hot])) / np.median(np.abs(pr["eV"][cold])))
                     if hot.sum() >= 10 else None}
    # E2: increment rank correlation on the largest |dT| rows
    dT, dV = pr["dT"], pr["dV"]
    top = np.abs(dT) >= np.quantile(np.abs(dT), 0.9)
    rec["E2"] = {"n_rows": int(top.sum()), "rho_top10": float(spearmanr(dT[top], dV[top])[0]),
                 "rho_all": float(spearmanr(dT, dV)[0])}
    # E1 / E1b
    fit23 = json.loads((DATA / ".b23_vrot_jump_fit.json").read_text(encoding="utf-8"))["fits"][f"{pop}_s{s}"]
    cases = {"CES_TI": (0, BULK_SIGMA_N["CES_TI_cut" if pop == "cut" else "CES_TI_incl"], REG["lam"], REG["sigma_j"]),
             "CES_VT": (1, BULK_SIGMA_N["CES_VT"], REG["lam"], REG["sigma_j"]),
             "CES_VT_b23fit": (1, BULK_SIGMA_N["CES_VT"], fit23["lam_per_step"], fit23["sigma_j"])}
    rec["E1"] = {}
    for key, (t, sn, lam, sj) in cases.items():
        carried, obs_prev = series(blocks, t)
        w, em, ed = filter_e1(carried, obs_prev, lam, sj ** 2, sn ** 2)
        rec["E1"][key] = {"n_updates": int(len(w)), "ambiguous_frac": float(np.mean((w > 0.2) & (w < 0.8))),
                          "jump_frac": float(np.mean(w >= 0.8)),
                          "medae_ratio_median_over_mean": float(np.median(ed) / np.median(em))}
    return rec


def main():
    out = {}
    for pop in MANIFEST:
        grid, dims = load_grid_files(DATA, True, CUT_EV[pop])
        for s in SEEDS:
            r = one_split(pop, s, grid, dims)
            out[f"{pop}_s{s}"] = r
            e1 = r["E1"]
            print(f"[b26] {pop} s{s}: A1 R {r['A1']['R_median']:.3f} ({r['A1']['n_shots']} shots) | A2 slope {r['A2']['slope']:+.3f}"
                  f" | A3 abs {r['A3']['rho_abs']:+.3f} sign {r['A3']['rho_sign']:+.3f}"
                  + (f" | A4 {r['A4']['ratio']:.2f} (n {r['A4']['n_hot']})" if pop == 'incl' and r['A4']['ratio'] else "")
                  + f" | E2 top10 {r['E2']['rho_top10']:+.3f} all {r['E2']['rho_all']:+.3f} | E1 amb "
                  + " ".join(f"{k} {v['ambiguous_frac']:.3f}/{v['medae_ratio_median_over_mean']:.4f}" for k, v in e1.items()),
                  flush=True)
    verdict = {}
    for pop in MANIFEST:
        rs = [out[f"{pop}_s{s}"] for s in SEEDS]
        v = {"A1": sum(abs(r["A1"]["R_median"] - 1.41) <= 0.30 for r in rs),
             "A1_below_1.11": sum(r["A1"]["R_median"] < 1.11 for r in rs),
             "A2": sum(abs(r["A2"]["slope"] - 0.5) <= 0.15 for r in rs),
             "A3": sum(r["A3"]["rho_abs"] >= 0.15 and abs(r["A3"]["rho_sign"]) < 0.1 for r in rs),
             "A3_both_below_0.1": sum(abs(r["A3"]["rho_abs"]) < 0.1 and abs(r["A3"]["rho_sign"]) < 0.1 for r in rs),
             "A3_sign_ge_0.2": sum(abs(r["A3"]["rho_sign"]) >= 0.2 for r in rs),
             "E2": sum(abs(r["E2"]["rho_top10"]) >= 0.15 for r in rs)}
        if pop == "incl":
            v["A4"] = sum((r["A4"]["ratio"] or 0) >= 2 for r in rs)
        for key in ("CES_TI", "CES_VT", "CES_VT_b23fit"):
            v[f"E1_{key}"] = sum(r["E1"][key]["ambiguous_frac"] >= 0.05 for r in rs)
            v[f"E1b_{key}"] = sum(r["E1"][key]["medae_ratio_median_over_mean"] <= 0.99 for r in rs)
        verdict[pop] = {k: {"splits": n, "pass": n >= 3} for k, n in v.items()}
    out["verdict"] = verdict
    (DATA / ".b26_precheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({p: {k: f"{v['splits']}/4" for k, v in d.items()} for p, d in verdict.items()}, indent=1))
    print(f"wrote {DATA / '.b26_precheck.json'}")


if __name__ == "__main__":
    main()
