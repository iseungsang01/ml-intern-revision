"""Regime transfer (PREREGISTRATION_REGIME_TRANSFER.md, 2026-11-15): per-discharge regime number
on KSTAR against which CES-only estimator is the right bar (T1) and against the margin of the
backbone over the causal GP (T2). CPU, frozen TEST rows of the eight canonical cells.

Run from the repo root:  py docs/paper/review_loop/analyses/R3/regime_transfer.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "R2"))
sys.path.insert(0, str(HERE.parents[2] / "ppcf"))
import r2_common as C  # noqa: E402
import r2ac_vrot_running_mean as RA  # noqa: E402  (file_tables, attach, val_alpha, T_RULE, ALPHAS)
import kstar_regime as KR  # noqa: E402  (regime, held_free; definition unchanged)

SEED, NBOOT, MIN_ROWS = 20261116, 2000, 30
TARGETS = {"T_i": "CES_TI", "V_rot": "CES_VT"}
R2A = json.loads((HERE.parent / "R2" / "r2ac_vrot_running_mean.json").read_text(encoding="utf-8"))
A2 = {(c["pop"], c["seed"]): c for c in json.loads(
    (HERE.parent / "A2" / "a2_decomposition.json").read_text(encoding="utf-8"))["a_decomposition"]}


def regime_numbers(files):
    out = {}
    for f in files:
        df = pd.read_csv(f, usecols=["time", "CES_TI", "CES_VT"])
        t = df["time"].to_numpy(float)
        y = df["CES_TI"].to_numpy(float).copy()
        y[y > 3000.0] = np.nan
        v = KR.held_free(df["CES_VT"].to_numpy(float))
        v[np.abs(v) > 1000.0] = np.nan
        out[Path(f).name] = {"T_i": KR.regime(t, y)[0], "V_rot": KR.regime(t, v)[0]}
    return out


def val_alpha_ti(tabs, names, name_to_idx):
    s = np.zeros(len(RA.ALPHAS))
    n = 0
    for nm in names:
        i = name_to_idx.get(nm)
        if i is None:
            continue
        tb = tabs[i]
        k = ~np.isnan(tb["pers"])
        if k.any():
            s += ((tb["eu"][k] - tb["v"][k][:, None]) ** 2).sum(0)
            n += int(k.sum())
    return int(np.argmin(s)), n


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def boot_spearman(x, y, rng):
    n = len(x)
    idx = rng.integers(0, n, (NBOOT, n))
    vals = np.array([spearman(x[i], y[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"rho": spearman(x, y), "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0), "n_discharges": int(n)}


def boot_ratio(a, r, rng):
    n = len(a)
    idx = rng.integers(0, n, (NBOOT, n))
    sk = 1.0 - a[idx].sum(1) / r[idx].sum(1)
    lo, hi = np.percentile(sk, [2.5, 97.5])
    return {"point": float(1.0 - a.sum() / r.sum()), "ci95": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0 or hi < 0), "n_discharges": int(n)}


def sfmt(s, key="rho"):
    star = "*" if s["excludes_0"] else " "
    return f"{s[key]:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]{star} n={s['n_discharges']}"


def main():
    t0 = time.time()
    out = {"registration": "ces_prediction/experiments/PREREGISTRATION_REGIME_TRANSFER.md",
           "seed": SEED, "n_boot": NBOOT, "min_rows_per_appearance": MIN_ROWS, "alignment": {}, "results": {}}
    reg = None
    for pop in C.POPS:
        ds = C.build_dataset(pop)
        names = [Path(p).name for p in ds.valid_files]
        name_to_idx = {n: i for i, n in enumerate(names)}
        if reg is None:
            reg = regime_numbers(ds.valid_files)
            print(f"regime numbers for {len(reg)} files ({time.time() - t0:.0f} s)", flush=True)
        tc = int(ds._column_slices["time"])
        for tname, tkey in TARGETS.items():
            j_t = 0 if tname == "T_i" else 1
            col = int(ds._column_slices["target"][j_t])
            tabs = [RA.file_tables(fa, tc, col) for fa in ds.file_arrays]
            acc = {}
            for s in C.SEEDS:
                man = C.manifest(pop, s)
                d = C.canonical(pop, s, tkey)
                check = {"rows": 0, "persistence_mismatch": 0}
                tag = f"{pop} s{s} {tname}"
                if tname == "T_i":
                    ju, nv = val_alpha_ti(tabs, man["val_files"], name_to_idx)
                    d = RA.attach(d, tabs, ju, ju, tag, check)
                    keep = np.ones(len(d["shot"]), bool)
                    rm = d["ewma_u"]
                    alpha = float(RA.ALPHAS[ju])
                    ref = A2[(pop, s)]
                    al = {"alpha": alpha, "n_val_rows": nv,
                          "model_vs_gp_causal": C.skill(d["shot"], d["model"], d["gp_causal"])["point"],
                          "stored_model_vs_gp_causal": ref["backbone_vs_gp_causal"]["skill_point"],
                          "model_vs_running_mean": C.skill(d["shot"], d["model"], rm)["point"],
                          "stored_model_vs_ewma_a2_coarse_grid": ref["backbone_vs_ewma"]["skill_point"],
                          "stored_alpha_a2": ref["alpha_star"]}
                    al["ok"] = bool(abs(al["model_vs_gp_causal"] - al["stored_model_vs_gp_causal"]) < 1e-9)
                else:
                    ju, jg, nv, _, _ = RA.val_alpha(tabs, man["val_files"], name_to_idx)
                    d = RA.attach(d, tabs, ju, jg, tag, check)
                    keep = d["amax"] <= RA.T_RULE
                    rm = d["ewma_g"]
                    alpha = float(RA.ALPHAS[jg])
                    ref = R2A["pops"][pop]["canonical"][f"s{s}"]
                    al = {"alpha": alpha, "n_val_rows": nv, "n_dropped": int((~keep).sum()),
                          "model_vs_gp_causal": C.skill(d["shot"][keep], d["model"][keep], d["gp_causal"][keep])["point"],
                          "stored_model_vs_gp_causal": ref["model_vs_gp_causal"]["point"],
                          "running_mean_vs_persistence": C.skill(d["shot"][keep], rm[keep], d["persistence"][keep])["point"],
                          "stored_running_mean_vs_persistence": ref["ewma_g_vs_persistence"]["point"]}
                    al["ok"] = bool(abs(al["model_vs_gp_causal"] - al["stored_model_vs_gp_causal"]) < 1e-9
                                    and abs(al["running_mean_vs_persistence"] - al["stored_running_mean_vs_persistence"]) < 1e-9)
                al["persistence_mismatch_rows"] = check["persistence_mismatch"]
                al["rows"] = check["rows"]
                out["alignment"][f"{tname}_{pop}_s{s}"] = al
                print(f"[align] {tag}: ok={al['ok']} pers_mismatch={check['persistence_mismatch']}/{check['rows']} "
                      f"alpha={alpha} model/gpc {al['model_vs_gp_causal']:+.4f} (stored {al['stored_model_vs_gp_causal']:+.4f})",
                      flush=True)
                if not al["ok"] or check["persistence_mismatch"]:
                    raise SystemExit("FATAL: row alignment check failed; no statistic computed")
                sh = d["shot"][keep]
                u, inv = np.unique(sh, return_inverse=True)
                cols = {"model": d["model"][keep], "gpc": d["gp_causal"][keep],
                        "pers": d["persistence"][keep], "rm": rm[keep]}
                sums = {k: np.bincount(inv, weights=v, minlength=len(u)) for k, v in cols.items()}
                cnt = np.bincount(inv, minlength=len(u))
                for i, fi in enumerate(u):
                    a = acc.setdefault(int(fi), {"model": 0.0, "gpc": 0.0, "pers": 0.0, "rm": 0.0, "rows": 0, "app": 0})
                    for k in sums:
                        a[k] += float(sums[k][i])
                    a["rows"] += int(cnt[i])
                    a["app"] += 1
            rows = []
            for fi, a in acc.items():
                rn = reg[names[fi]][tname]
                if not np.isfinite(rn) or a["rows"] / a["app"] < MIN_ROWS or a["pers"] <= 0 or a["gpc"] <= 0:
                    continue
                rows.append((rn, a["model"], a["gpc"], a["pers"], a["rm"]))
            T = np.array(rows)
            rn, m, g, p, r = T.T
            D1 = (r - g) / p
            D1b = 1.0 - r / p
            D2 = 1.0 - m / g
            rng = np.random.default_rng(SEED)
            res = {"n_discharges_scored": len(acc), "n_included": int(len(rn)),
                   "regime_quartiles_included": [float(q) for q in np.percentile(rn, [25, 50, 75])],
                   "share_regime_above_0": float((rn > 0).mean()),
                   "T1_spearman_regime_vs_gp_minus_running_mean": boot_spearman(rn, D1, rng),
                   "T1b_spearman_regime_vs_running_mean_skill": boot_spearman(rn, D1b, rng),
                   "T2_spearman_regime_vs_backbone_over_gp": boot_spearman(rn, D2, rng),
                   "tertiles": []}
            edges = np.percentile(rn, [100 / 3, 200 / 3])
            lab = np.digitize(rn, edges)
            for q in range(3):
                k = lab == q
                res["tertiles"].append({
                    "regime_range": [float(rn[k].min()), float(rn[k].max())], "n_discharges": int(k.sum()),
                    "gp_vs_running_mean": boot_ratio(g[k], r[k], rng),
                    "running_mean_vs_persistence": boot_ratio(r[k], p[k], rng),
                    "gp_vs_persistence": boot_ratio(g[k], p[k], rng),
                    "backbone_vs_gp": boot_ratio(m[k], g[k], rng)})
            out["results"][f"{tname}_{pop}"] = res
            print(f"== {tname} {pop}: included {len(rn)} of {len(acc)} scored discharges")
            print("   T1  rho(regime, GP - running mean)  ", sfmt(res["T1_spearman_regime_vs_gp_minus_running_mean"]))
            print("   T1b rho(regime, running mean skill) ", sfmt(res["T1b_spearman_regime_vs_running_mean_skill"]))
            print("   T2  rho(regime, backbone over GP)   ", sfmt(res["T2_spearman_regime_vs_backbone_over_gp"]))
            for q, tt in enumerate(res["tertiles"]):
                print(f"   tertile {q} regime [{tt['regime_range'][0]:+.2f}, {tt['regime_range'][1]:+.2f}] n={tt['n_discharges']}: "
                      f"GP/RM {sfmt(tt['gp_vs_running_mean'], 'point')} | RM/pers {sfmt(tt['running_mean_vs_persistence'], 'point')} | "
                      f"GP/pers {sfmt(tt['gp_vs_persistence'], 'point')} | bb/GP {sfmt(tt['backbone_vs_gp'], 'point')}", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (HERE / "regime_transfer.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote regime_transfer.json ({out['minutes']} min)")


if __name__ == "__main__":
    main()
