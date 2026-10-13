"""P04 extras: (1) exact 8n reproduction (inclusive population, all input-complete rows,
held rule only -- the diagnose_shift.py definition); (2) R^2 of per-file level on bins of
consecutive shots (run-day proxy) at several bin sizes; (3) the 'free loss' test: does the
absolute ECEI/BES level carry information about the within-shot T_i mapping beyond what the
CES level (which the history anchor carries) already gives?  ASCII only."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p04_campaign_drift import DATA, OUT, STUCK_GAP, TRAIN_RANGE, TEST_RANGE, stuck_mask, partial_corr, r2_linear


def load_inclusive(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    groups = {"BES": [c for c in cols if c.startswith("BES_")], "ECEI": [c for c in cols if c.startswith("ECEI_")],
              "MC": [c for c in cols if c.startswith("MC")]}
    t = df["time"].to_numpy(float)
    block = np.concatenate(([0], np.cumsum(np.diff(t) >= STUCK_GAP))) if len(t) > 1 else np.zeros(len(t), int)
    out = {}
    for g, cs in groups.items():
        out[g] = df[cs].to_numpy(float)
    for c in ("CES_TI", "CES_VT"):
        v = df[c].to_numpy(float, copy=True)
        v[stuck_mask(v, block)] = np.nan
        out[c] = v[:, None]
    return out


def main():
    files = sorted(DATA.glob("s*.csv"))
    pooled = {"train": {}, "test": {}}
    for f in files:
        shot = int(re.search(r"s(\d+)", f.name).group(1))
        side = "train" if TRAIN_RANGE[0] <= shot <= TRAIN_RANGE[1] else ("test" if TEST_RANGE[0] <= shot <= TEST_RANGE[1] else None)
        if not side:
            continue
        d = load_inclusive(f)
        for name, arr in d.items():
            for k in range(arr.shape[1]):
                v = arr[:, k]; v = v[np.isfinite(v)]
                s = pooled[side].setdefault((name, k), [0, 0.0, 0.0])
                s[0] += v.size; s[1] += v.sum(); s[2] += (v * v).sum()
    print("=== (1) exact 8n reproduction (inclusive, all rows, held rule only) ===")
    rep = {}
    for name in ("BES", "ECEI", "MC", "CES_TI", "CES_VT"):
        zs, ratios = [], []
        for key in sorted(k for k in pooled["train"] if k[0] == name):
            a, b = pooled["train"][key], pooled["test"][key]
            ma, mb = a[1] / a[0], b[1] / b[0]
            sa = np.sqrt(max(a[2] / a[0] - ma ** 2, 0)); sb = np.sqrt(max(b[2] / b[0] - mb ** 2, 0))
            zs.append(abs(mb - ma) / sa); ratios.append(sb / sa)
        rep[name] = dict(z_median=float(np.median(zs)), z_max=float(np.max(zs)), std_ratio_median=float(np.median(ratios)))
        print("%-7s z median %.3f  z max %.3f  std ratio %.3f" % (name, *rep[name].values()))
    z_ces = np.median([rep["CES_TI"]["z_median"], rep["CES_VT"]["z_median"]])
    print("CES(target) group z median (median of TI, VT) = %.3f" % z_ces)

    df = pd.read_csv(OUT / "p04_per_file_stats.csv").sort_values("shot").reset_index(drop=True)
    print("\n=== (2) R^2 of per-file mean on bins of K consecutive shots (run-day proxy) ===")
    binr = {}
    print("%-6s " % "K" + " ".join("%7s" % q for q in ("bes", "ecei", "mc", "ti", "vt")))
    for K in (5, 10, 20, 40, 80, 214):
        b = np.arange(len(df)) // K
        row = []
        for q in ("bes", "ecei", "mc", "ti", "vt"):
            y = df[q + "_mean"].to_numpy(float); m = np.isfinite(y); yy, bb = y[m], b[m]
            ss_tot = ((yy - yy.mean()) ** 2).sum()
            ss_b = sum((yy[bb == k]).size * (yy[bb == k].mean() - yy.mean()) ** 2 for k in np.unique(bb))
            # adjusted R^2 (bins eat degrees of freedom)
            n, p = yy.size, np.unique(bb).size
            r2 = ss_b / ss_tot; adj = 1 - (1 - r2) * (n - 1) / (n - p)
            row.append(adj)
        binr[K] = dict(zip(("bes", "ecei", "mc", "ti", "vt"), map(float, row)))
        print("%-6d " % K + " ".join("%7.3f" % v for v in row))

    print("\n=== (3) free-loss test: per-shot gain beta_f = dT_i/dz(ECEI or BES) [eV per within-shot sigma] ===")
    # recompute per-shot gains from the raw files (cut population, pipeline rows)
    from p04_campaign_drift import load
    recs = []
    for f in files:
        shot = int(re.search(r"s(\d+)", f.name).group(1))
        d = load(f)
        m = np.isfinite(d["ti"])
        if m.sum() < 8:
            continue
        rec = dict(shot=shot, ti_mean=np.nanmean(d["ti"]), ti_std=np.nanstd(d["ti"]))
        for q in ("bes", "ecei"):
            x = d[q]; lvl = x.mean(); sd = x.std()
            if sd <= 0:
                continue
            z = (x[m] - lvl) / sd
            if z.std() <= 0:
                continue
            rec[q + "_level"] = lvl; rec[q + "_sd"] = sd
            rec[q + "_beta"] = np.cov(d["ti"][m], z)[0, 1] / z.var()
            # the raw-unit slope: dT_i / dx (eV per raw volt) -- what a global-norm model must learn
            rec[q + "_beta_raw"] = rec[q + "_beta"] / sd
        recs.append(rec)
    g = pd.DataFrame(recs)
    fl = {}
    for q in ("bes", "ecei"):
        beta = g[q + "_beta"].to_numpy(float); lvl = g[q + "_level"].to_numpy(float)
        ti = g[q.replace(q, "ti_mean")].to_numpy(float); tistd = g["ti_std"].to_numpy(float)
        pr_lvl, r_lvl, n = partial_corr(beta, lvl, ti)           # level info about the gain, beyond T_i level
        pr_ti, r_ti, _ = partial_corr(beta, ti, lvl)             # T_i level info about the gain, beyond ECEI level
        pr_std, r_std, _ = partial_corr(tistd, lvl, ti)          # level info about T_i dynamics amplitude
        r2_lvl_only, _ = r2_linear(lvl, beta); r2_ti_only, _ = r2_linear(ti, beta)
        ok = np.isfinite(beta) & np.isfinite(lvl) & np.isfinite(ti)
        bb = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), lvl[ok], ti[ok]]), beta[ok], rcond=None)[0]; X = np.column_stack([np.ones(len(beta)), lvl, ti]); r2_both = 1 - (beta[ok] - X[ok] @ bb).var() / beta[ok].var()
        # raw-unit slope spread: how much the global-norm mapping must vary across shots
        braw = g[q + "_beta_raw"].to_numpy(float)
        fl[q] = dict(n=int(n), r_beta_level=float(r_lvl), partial_r_beta_level_given_ti=float(pr_lvl),
                     r_beta_ti=float(r_ti), partial_r_beta_ti_given_level=float(pr_ti),
                     r2_beta_on_level=float(r2_lvl_only), r2_beta_on_ti=float(r2_ti_only), r2_beta_on_both=float(r2_both),
                     partial_r_tistd_level_given_ti=float(pr_std),
                     beta_median_eV_per_sigma=float(np.nanmedian(beta)), beta_iqr=float(np.nanpercentile(beta, 75) - np.nanpercentile(beta, 25)),
                     beta_raw_cv=float(np.nanstd(braw) / abs(np.nanmean(braw))))
        print("%-5s n=%d  beta median %.1f eV/sigma (IQR %.1f)" % (q, n, fl[q]["beta_median_eV_per_sigma"], fl[q]["beta_iqr"]))
        print("      corr(beta, level) = %+.3f ; partial | T_i level = %+.3f" % (r_lvl, pr_lvl))
        print("      corr(beta, T_i level) = %+.3f ; partial | level = %+.3f" % (r_ti, pr_ti))
        print("      R2 beta ~ level %.3f | ~ T_i level %.3f | ~ both %.3f  (increment from level = %.3f)" % (
            r2_lvl_only, r2_ti_only, r2_both, r2_both - r2_ti_only))
        print("      partial corr(T_i within-shot std, level | T_i level) = %+.3f" % pr_std)
        print("      raw-unit slope dT_i/dx across shots: CV = %.2f (global-norm model must absorb this)" % fl[q]["beta_raw_cv"])
    (OUT / "p04_extra_results.json").write_text(json.dumps(dict(repro_8n_exact=rep, r2_by_bin=binr, free_loss=fl), indent=1), encoding="ascii")
    print("\nwrote", OUT / "p04_extra_results.json")


if __name__ == "__main__":
    main()
