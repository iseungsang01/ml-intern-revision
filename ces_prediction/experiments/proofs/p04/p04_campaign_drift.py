"""P04: campaign drift of the fast diagnostics is instrument/geometry physics.

Per-file statistics over the 641 shot CSVs (held-free, 3 keV CES_TI cut), then:
  (a) R^2 of each per-file mean (and std) on shot number / campaign third
  (b) between-file / total variance share (eta^2) per quantity
  (c) between-shot corr(ECEI or BES level, CES_TI level), partial on shot number,
      versus the within-shot correlation (Fisher-averaged)
  (d) reproduction of the 8n train->test z drift from the same loader
  (e) EWMA / expanding-window online standardization bias vs the offline
      per-shot z-score, binned by time since block start
  (f) per-shot linear trend slopes (in per-shot sigma per second) feeding the
      EWMA bias formula  bias = -b * tau_w
All output ASCII only. No model is trained or scored. No tracked file is touched.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent
STUCK_GAP = 0.5
CUT_EV = 3000.0
TRAIN_RANGE = (30801, 31991)   # 8n manifest
TEST_RANGE = (32312, 32751)


def stuck_mask(values, block):
    mask = np.zeros(values.shape[0], dtype=bool)
    obs = np.flatnonzero(~np.isnan(values))
    if obs.size <= 1:
        return mask
    ov, ob = values[obs], block[obs]
    dup = (ov[1:] == ov[:-1]) & (ob[1:] == ob[:-1])
    mask[obs[1:][dup]] = True
    return mask


def load(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    bes = [c for c in cols if c.startswith("BES_")]
    ecei = [c for c in cols if c.startswith("ECEI_")]
    mc = [c for c in cols if c.startswith("MC")]
    df = df.dropna(subset=["time"] + bes + ecei + mc).reset_index(drop=True)
    t = df["time"].to_numpy(float)
    block = np.concatenate(([0], np.cumsum(np.diff(t) >= STUCK_GAP))) if len(t) > 1 else np.zeros(len(t), int)
    ti = df["CES_TI"].to_numpy(float, copy=True)
    ti[ti > CUT_EV] = np.nan                      # spike cut first (dataset.py order)
    ti[stuck_mask(ti, block)] = np.nan
    vt = df["CES_VT"].to_numpy(float, copy=True)
    vt[stuck_mask(vt, block)] = np.nan
    keep = ~(np.isnan(ti) & np.isnan(vt))         # pipeline keeps rows with >=1 target
    return dict(
        t=t[keep], block=block[keep], ti=ti[keep], vt=vt[keep],
        bes=df[bes].to_numpy(float)[keep].mean(axis=1),
        ecei=df[ecei].to_numpy(float)[keep].mean(axis=1),
        mc=df[mc].to_numpy(float)[keep].mean(axis=1),
        bes_ch=df[bes].to_numpy(float)[keep], ecei_ch=df[ecei].to_numpy(float)[keep],
        mc_ch=df[mc].to_numpy(float)[keep],
    )


def r2_linear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1.0 - res.var() / y.var(), beta[1]


def r2_third(third, y):
    m = np.isfinite(y)
    third, y = third[m], y[m]
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_b = sum(((y[third == k]).size) * (y[third == k].mean() - y.mean()) ** 2 for k in np.unique(third))
    return ss_b / ss_tot


def partial_corr(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], np.corrcoef(x, y)[0, 1], m.sum()


def ewma_z(x, t, tau):
    """dt-aware causal EWMA mean/var, state starts at the first sample."""
    z = np.empty_like(x)
    m, v = x[0], 0.0
    z[0] = 0.0
    for i in range(1, len(x)):
        a = 1.0 - np.exp(-(t[i] - t[i - 1]) / tau)
        d = x[i] - m
        m = m + a * d
        v = (1 - a) * (v + a * d * d)
        z[i] = (x[i] - m) / np.sqrt(v) if v > 1e-12 else 0.0
    return z


def expanding_z(x):
    n = np.arange(1, len(x) + 1)
    cm = np.cumsum(x) / n
    cv = np.cumsum(x * x) / n - cm ** 2
    cv = np.maximum(cv, 0)
    z = np.where(cv > 1e-12, (x - cm) / np.sqrt(cv), 0.0)
    z[0] = 0.0
    return z


def main():
    files = sorted(DATA.glob("s*.csv"))
    assert len(files) == 641, len(files)
    rows = []
    pooled = {"train": {}, "test": {}}
    ewma_err = {}       # (quantity, tau) -> list of (elapsed, |z_on - z_off|)
    slopes = {"bes": [], "ecei": []}
    within_r = {"bes": [], "ecei": []}
    taus = (0.5, 1.0, 2.0, "expanding")
    for f in files:
        shot = int(re.search(r"s(\d+)", f.name).group(1))
        d = load(f)
        n = len(d["t"])
        if n < 3:
            continue
        rec = dict(shot=shot, n=n, n_ti=int(np.isfinite(d["ti"]).sum()), n_vt=int(np.isfinite(d["vt"]).sum()))
        for q in ("bes", "ecei", "mc"):
            rec[q + "_mean"] = d[q].mean(); rec[q + "_std"] = d[q].std()
        for q in ("ti", "vt"):
            rec[q + "_mean"] = np.nanmean(d[q]) if rec["n_" + q] else np.nan
            rec[q + "_std"] = np.nanstd(d[q]) if rec["n_" + q] > 1 else np.nan
        rows.append(rec)
        # pooled per-channel stats for the 8n reproduction
        side = "train" if TRAIN_RANGE[0] <= shot <= TRAIN_RANGE[1] else ("test" if TEST_RANGE[0] <= shot <= TEST_RANGE[1] else None)
        if side:
            for name, arr in (("BES", d["bes_ch"]), ("ECEI", d["ecei_ch"]), ("MC", d["mc_ch"]),
                              ("CES_TI", d["ti"][:, None]), ("CES_VT", d["vt"][:, None])):
                for k in range(arr.shape[1]):
                    v = arr[:, k]; v = v[np.isfinite(v)]
                    s = pooled[side].setdefault((name, k), [0, 0.0, 0.0])
                    s[0] += v.size; s[1] += v.sum(); s[2] += (v * v).sum()
        # within-shot correlation (needs >= 8 TI points) and slopes, per block
        for q in ("bes", "ecei"):
            m = np.isfinite(d["ti"])
            if m.sum() >= 8 and d[q][m].std() > 0 and d["ti"][m].std() > 0:
                within_r[q].append(np.corrcoef(d[q][m], d["ti"][m])[0, 1])
            for b in np.unique(d["block"]):
                sel = d["block"] == b
                if sel.sum() >= 10 and d[q].std() > 0:
                    _, slope = r2_linear(d["t"][sel], d[q][sel] / d[q].std())  # per-shot-sigma per second
                    slopes[q].append(slope)
            # online vs offline z-score, per contiguous block
            z_off = (d[q] - d[q].mean()) / (d[q].std() if d[q].std() > 0 else 1.0)
            for b in np.unique(d["block"]):
                sel = np.flatnonzero(d["block"] == b)
                if sel.size < 10:
                    continue
                tt, xx = d["t"][sel], d[q][sel]
                for tau in taus:
                    z_on = expanding_z(xx) if tau == "expanding" else ewma_z(xx, tt, tau)
                    ewma_err.setdefault((q, tau), []).append(np.column_stack([tt - tt[0], np.abs(z_on - z_off[sel])]))

    df = pd.DataFrame(rows).sort_values("shot").reset_index(drop=True)
    df.to_csv(OUT / "p04_per_file_stats.csv", index=False)
    third = np.digitize(np.arange(len(df)), [len(df) / 3, 2 * len(df) / 3])   # chronological thirds
    shot = df["shot"].to_numpy(float)
    res = {"n_files": int(len(df)), "n_rows": int(df["n"].sum())}

    print("=== (a) R^2 of per-file statistic on chronological position (641 files) ===")
    print("%-10s %-5s %8s %8s %12s" % ("quantity", "stat", "R2_shot", "R2_third", "slope/1000sh"))
    tab = {}
    for q in ("bes", "ecei", "mc", "ti", "vt"):
        for st in ("mean", "std"):
            y = df[q + "_" + st].to_numpy(float)
            r2s, slope = r2_linear(shot, y)
            r2t = r2_third(third, y)
            tab[(q, st)] = dict(R2_shot=float(r2s), R2_third=float(r2t), slope_per_1000_shots=float(slope * 1000))
            print("%-10s %-5s %8.3f %8.3f %12.4g" % (q, st, r2s, r2t, slope * 1000))
    res["r2"] = {f"{q}_{st}": v for (q, st), v in tab.items()}

    print("\n=== third means (per-file mean averaged within chronological third) ===")
    thirds = {}
    for q in ("bes", "ecei", "mc", "ti", "vt"):
        y = df[q + "_mean"].to_numpy(float)
        ms = [float(np.nanmean(y[third == k])) for k in range(3)]
        sds = [float(np.nanmean(df[q + "_std"].to_numpy(float)[third == k])) for k in range(3)]
        thirds[q] = dict(mean_by_third=ms, std_by_third=sds)
        print("%-6s mean: %10.4g %10.4g %10.4g | within-shot std: %10.4g %10.4g %10.4g" % (q, *ms, *sds))
    res["thirds"] = thirds

    print("\n=== (b) between-file variance share eta^2 = SS_between / SS_total (row-weighted) ===")
    eta = {}
    for q in ("bes", "ecei", "mc", "ti", "vt"):
        m = df[q + "_mean"].to_numpy(float); s = df[q + "_std"].to_numpy(float)
        n = (df["n_" + q] if q in ("ti", "vt") else df["n"]).to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(s) & (n > 1)
        m, s, n = m[ok], s[ok], n[ok]
        gm = (n * m).sum() / n.sum()
        ss_b = (n * (m - gm) ** 2).sum(); ss_w = (n * s ** 2).sum()
        eta[q] = dict(eta2=float(ss_b / (ss_b + ss_w)), between_share_unweighted=float(m.var() / (m.var() + np.mean(s ** 2))))
        print("%-6s eta2 = %.3f   (unweighted %.3f)" % (q, eta[q]["eta2"], eta[q]["between_share_unweighted"]))
    res["eta2"] = eta

    print("\n=== (c) between-shot level correlation with CES_TI level, raw and partial on shot number ===")
    cc = {}
    for q in ("bes", "ecei", "mc"):
        for tgt in ("ti", "vt"):
            pr, r, nn = partial_corr(df[q + "_mean"].to_numpy(float), df[tgt + "_mean"].to_numpy(float), shot)
            cc[f"{q}_vs_{tgt}"] = dict(r=float(r), partial_r_given_shot=float(pr), n=int(nn))
            print("%-5s vs %-3s  r = %+.3f   r|shot = %+.3f   (n = %d)" % (q, tgt, r, pr, nn))
    r_ti_shot = np.corrcoef(shot[np.isfinite(df["ti_mean"])], df["ti_mean"].dropna())[0, 1]
    print("corr(shot, ti_mean) = %+.3f ; corr(shot, ecei_mean) = %+.3f ; corr(shot, bes_mean) = %+.3f" % (
        r_ti_shot, np.corrcoef(shot, df["ecei_mean"])[0, 1], np.corrcoef(shot, df["bes_mean"])[0, 1]))
    res["between_corr"] = cc
    wr = {}
    for q in ("bes", "ecei"):
        z = np.arctanh(np.clip(np.array(within_r[q]), -0.999, 0.999))
        wr[q] = dict(fisher_mean_r=float(np.tanh(z.mean())), median_r=float(np.median(within_r[q])), n_shots=int(len(within_r[q])),
                     frac_positive=float(np.mean(np.array(within_r[q]) > 0)))
        print("within-shot corr(%s, ti): Fisher mean %+.3f, median %+.3f, frac>0 %.2f (n=%d shots)" % (
            q, wr[q]["fisher_mean_r"], wr[q]["median_r"], wr[q]["frac_positive"], wr[q]["n_shots"]))
    res["within_corr"] = wr

    print("\n=== (d) 8n reproduction: train(30801-31991) -> test(32312-32751) z = |dmean|/sd_train, pooled rows ===")
    rep = {}
    for name in ("BES", "ECEI", "MC", "CES_TI", "CES_VT"):
        zs, ratios = [], []
        for key in sorted(k for k in pooled["train"] if k[0] == name):
            a, b = pooled["train"][key], pooled["test"][key]
            ma, mb = a[1] / a[0], b[1] / b[0]
            sa = np.sqrt(max(a[2] / a[0] - ma ** 2, 0)); sb = np.sqrt(max(b[2] / b[0] - mb ** 2, 0))
            zs.append(abs(mb - ma) / sa); ratios.append(sb / sa)
        rep[name] = dict(z_median=float(np.median(zs)), z_max=float(np.max(zs)), std_ratio_median=float(np.median(ratios)))
        print("%-7s z median %.3f  z max %.3f  std ratio %.3f" % (name, rep[name]["z_median"], rep[name]["z_max"], rep[name]["std_ratio_median"]))
    res["repro_8n"] = rep

    print("\n=== (f) per-shot linear trend slope b of the (per-shot-sigma-scaled) signal, per contiguous block ===")
    sl = {}
    for q in ("bes", "ecei"):
        b = np.abs(np.array(slopes[q]))
        sl[q] = dict(median_abs_slope_sigma_per_s=float(np.median(b)), p90=float(np.percentile(b, 90)), n_blocks=int(b.size))
        print("%-5s |b| median %.3f sigma/s, p90 %.3f sigma/s (n=%d blocks)" % (q, sl[q]["median_abs_slope_sigma_per_s"], sl[q]["p90"], b.size))
    res["slopes"] = sl

    print("\n=== (e) online z (EWMA tau_w / expanding) vs offline per-shot z: mean |dz| by elapsed time since block start ===")
    bins = [0, 0.5, 1, 2, 4, 8, 1e9]
    labels = ["0-0.5s", "0.5-1s", "1-2s", "2-4s", "4-8s", ">8s"]
    print("%-5s %-9s " % ("q", "tau") + " ".join("%8s" % l for l in labels))
    ew = {}
    for (q, tau), chunks in sorted(ewma_err.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        arr = np.concatenate(chunks)
        idx = np.digitize(arr[:, 0], bins[1:-1])
        vals = [float(arr[idx == k, 1].mean()) if (idx == k).any() else np.nan for k in range(len(labels))]
        ew[f"{q}_{tau}"] = dict(zip(labels, vals))
        print("%-5s %-9s " % (q, tau) + " ".join("%8.3f" % v for v in vals))
    res["online_vs_offline"] = ew

    (OUT / "p04_results.json").write_text(json.dumps(res, indent=1), encoding="ascii")
    print("\nwrote", OUT / "p04_results.json")


if __name__ == "__main__":
    main()
