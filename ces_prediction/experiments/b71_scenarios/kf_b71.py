"""B.71 filter rows (KF0, KFX under F): the B.56 filter with the B.70 phase-2 parameters (PREREGISTRATION_B70.md 2.9).

The recursion is b56_kfx/kfx_model.run (that module fits and scores at import, so it cannot be imported); it is
copied below, batched over the cuts of one block, with one change the registration names: a sample that is not
available makes no update (predict step only).  Filter rows: kfx_model's own loader (the CSV without its first and
last row, float64), filter row j = grid row j + 1 (checked on the time column, as B.70 phase 2 does); T_i = the D1
grid's (above 3 keV and held repeats missing, B.70 phase 2's filter grid); BES drive c = mean log of the BES channels,
g = mean log(col 07) - mean log(col 05); the state restarts at every grid block start.  Output at row t = exp(x-),
the estimate from CES samples before t and BES up to t.

Parameter files: B.70 phase 2 (b70_d1/p2_b56_fit.py) writes data/.b70/kf_params_r{r}_f{k}.json with
    {"KF0": [log q, log r, log k], "KFX": [log q, log r, log k, b_c, b_g], "use_x": {"KF0": false, "KFX": true},
     "clip": 0.2, "fold": {"repeat": r, "fold": k}, "held_repeats_removed": true, ...}
A missing fold -> nothing is computed and the readout states that the filter rows are left out.
Output: data/.b71/kf_r{r}_f{k}.npz with kf0_F, kfx_F in the order of the F units of rows_r{r}_f{k}.npz.
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np

import run_b71 as M


def kf_replay(theta, use_x, clip, ly, dc, dg, steps, avail):
    """kfx_model.run for one block, batched over cuts: ly, dc, dg, steps (L,); avail (B, L) -> out (B, L)."""
    lq, lr, lk = theta[:3]
    bc, bg = (theta[3], theta[4]) if use_x else (0.0, 0.0)
    q, r, k = np.exp(lq), np.exp(lr), np.exp(lk)
    nb, L = avail.shape
    x = np.full(nb, np.nan)
    P = np.full(nb, np.nan)
    out = np.full((nb, L), np.nan)
    for j in range(L):
        xm = x + bc * np.clip(dc[j], -clip, clip) + bg * np.clip(dg[j], -clip, clip)
        Pm = P + q * steps[j]
        out[:, j] = np.exp(xm)
        obs = avail[:, j] & np.isfinite(ly[j])
        start = obs & ~np.isfinite(xm)
        v = ly[j] - xm
        S = Pm + r
        w = np.minimum(1.0, k * np.sqrt(S) / np.maximum(np.abs(v), 1e-12))
        Kg = Pm / S
        upd = obs & np.isfinite(xm)
        x = np.where(upd, xm + Kg * w * v, xm)
        P = np.where(upd, (1 - w * Kg) * Pm, Pm)
        x = np.where(start, ly[j], x)
        P = np.where(start, r, P)
    return out


def drives(X, nm):
    """Per grid row, from kfx_model's loader (the CSV without its first and last row, float64; NaN on those two rows):
    time, log T_i (missing where the D1 grid has none: above 3 keV and held repeats, as B.70 phase 2's filter grid),
    BES drives c and g; plus the grid time (float32 -> float64) for the cut comparisons."""
    import pandas as pd
    arr = X.C.grid[nm]
    n = len(arr)
    df = pd.read_csv(M.DATA / nm).iloc[1:-1].reset_index(drop=True)              # kfx_model's loader
    bes = [c for c in df.columns if c.startswith("BES_")]
    c05 = [c for c in bes if c[6:8] == "05"]
    c07 = [c for c in bes if c[6:8] == "07"]
    if not c05 or not c07:
        raise SystemExit("FATAL: BES columns 05 / 07 not found")
    t_csv = df["time"].to_numpy(float)
    if len(t_csv) != n - 2 or np.max(np.abs(t_csv - arr[1:-1, 0].astype(np.float64))) > 1e-5:
        raise SystemExit(f"FATAL: {nm}: filter row j is not grid row j + 1")
    Bv = df[bes].to_numpy(float)
    c, g = np.full(n, np.nan), np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        c[1:-1] = np.nanmean(np.log(np.where(Bv > 0, Bv, np.nan)), 1)
    g[1:-1] = np.log(np.clip(df[c07].to_numpy(float), 1e-9, None)).mean(1) - \
        np.log(np.clip(df[c05].to_numpy(float), 1e-9, None)).mean(1)
    tm = np.full(n, np.nan)
    tm[1:-1] = t_csv
    ti = np.full(n, np.nan)
    ti[1:-1] = np.where(np.isfinite(arr[1:-1, 1]), df["CES_TI"].to_numpy(float), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ly = np.log(np.where(ti > 0, ti, np.nan))
    return arr[:, 0].astype(np.float64), tm, ly, c, g


def fold(X, r, k, prm):
    z = np.load(M.CACHE / f"rows_r{r}_f{k}.npz")
    cuts = z["cuts"]
    col = {c: i for i, c in enumerate(z["cut_cols"].tolist())}
    kept = cuts[cuts[:, col["kept"]] == 1]
    fid, frow, fcut = (z[c].astype(np.int64) for c in ("F__file", "F__row", "F__cut"))
    out0, outx = np.full(len(frow), np.nan), np.full(len(frow), np.nan)
    if prm.get("use_x") != {"KF0": False, "KFX": True} or not prm.get("held_repeats_removed") \
            or prm.get("fold") != {"repeat": r, "fold": k} or len(prm["KF0"]) != 3 or len(prm["KFX"]) != 5:
        raise SystemExit(f"FATAL: the filter parameters of r{r} f{k} are not B.70 phase 2's D1 filters")
    th0, thx = np.asarray(prm["KF0"], float), np.asarray(prm["KFX"], float)
    clip = np.inf if prm.get("clip") is None else float(prm["clip"])
    for f in np.unique(kept[:, col["file"]]):
        nm = X.C.names[int(f)]
        tm_grid, tm, ly, c, g = drives(X, nm)
        n = len(tm)
        tq = M.tq_of(tm_grid)
        kc = np.flatnonzero(kept[:, col["file"]] == f)
        for (a, b) in {(int(kept[i, col["a"]]), int(kept[i, col["b"]])) for i in kc}:
            lo, hi = max(a, 1), min(b, n - 1)                       # kfx_model drops the first and last row
            ids = [i for i in kc if int(kept[i, col["a"]]) == a]
            dt = np.diff(tm[lo:hi], prepend=np.nan)
            reset = ~(dt < 0.5)
            steps = np.where(reset | ~np.isfinite(dt), 1.0, np.clip(dt / 0.01, 0.5, 50))
            dc = np.diff(c[lo:hi], prepend=np.nan)
            dg = np.diff(g[lo:hi], prepend=np.nan)
            dc[reset | ~np.isfinite(dc)] = 0.0
            dg[reset | ~np.isfinite(dg)] = 0.0
            if reset[1:].any():
                raise SystemExit(f"FATAL: {nm}: a filter reset inside a grid block")
            av = np.stack([np.isfinite(ly[lo:hi]) & (tq[lo:hi] < kept[i, col["cq"]]) for i in ids])
            o0 = kf_replay(th0, False, clip, ly[lo:hi], dc, dg, steps, av)
            ox = kf_replay(thx, True, clip, ly[lo:hi], dc, dg, steps, av)
            for j, i in enumerate(ids):
                cut_id = int(i)                                      # F__cut = index among the fold's kept cuts
                m = (fid == f) & (fcut == cut_id)
                rr = frow[m] - lo
                out0[m] = o0[j, rr]
                outx[m] = ox[j, rr]
    return out0, outx


def check(folds):
    """Filter checks (no error against a measured T_i): (a) at the first replay row t0 of every cut, F has the samples
    of U, so KF0 F and KFX F must equal B.70 phase 2's stored U-scenario filter estimates on that row
    (data/.b70p2/b56/r{r}_f{k}.npz, grid row = gi + 1, discharge = shot); (b) KF0 F is constant after every cut
    (predict step only, no drive); (c) every F unit has a finite filter output."""
    res = {}
    for r, k in folds:
        z = np.load(M.CACHE / f"rows_r{r}_f{k}.npz")
        kz = np.load(M.CACHE / f"kf_r{r}_f{k}.npz")
        col = {c: i for i, c in enumerate(z["cut_cols"].tolist())}
        kept = z["cuts"][z["cuts"][:, col["kept"]] == 1]
        f, row, cut = (z[c].astype(np.int64) for c in ("F__file", "F__row", "F__cut"))
        first = row == kept[cut, col["a"]] + kept[cut, col["t0"]]
        p2 = np.load(M.DATA / ".b70p2" / "b56" / f"r{r}_f{k}.npz")
        ok = p2["gi"] >= 0
        lut = {(int(s), int(g) + 1): j for j, (s, g) in enumerate(zip(p2["shot"][ok], p2["gi"][ok]))}
        idx = np.flatnonzero(first)
        pairs = [(i, lut[(int(f[i]), int(row[i]))]) for i in idx if (int(f[i]), int(row[i])) in lut]
        a = np.array([p for p, _ in pairs], dtype=np.int64)
        b = np.flatnonzero(ok)[np.array([q for _, q in pairs], dtype=np.int64)] if pairs else np.zeros(0, np.int64)
        d0 = np.abs(kz["kf0_F"][a] - p2["est_kf0"][b]) if len(a) else np.zeros(0)
        dx = np.abs(kz["kfx_F"][a] - p2["est_kfx"][b]) if len(a) else np.zeros(0)
        const = max((float(np.ptp(kz["kf0_F"][cut == c])) for c in np.unique(cut)), default=0.0)
        res[f"r{r}f{k}"] = {"first_rows_compared": int(len(a)), "max_dev_kf0_eV": float(d0.max()) if len(a) else None,
                            "max_dev_kfx_eV": float(dx.max()) if len(a) else None,
                            "kf0_max_range_within_a_cut_eV": const,
                            "units_without_filter_output": int((~np.isfinite(kz["kf0_F"])).sum()
                                                               + (~np.isfinite(kz["kfx_F"])).sum())}
    worst = lambda key: max(v[key] for v in res.values() if v[key] is not None)       # noqa: E731
    out = {"per_fold": res, "max_dev_kf0_eV": worst("max_dev_kf0_eV"), "max_dev_kfx_eV": worst("max_dev_kfx_eV"),
           "kf0_max_range_within_a_cut_eV": worst("kf0_max_range_within_a_cut_eV"),
           "units_without_filter_output": sum(v["units_without_filter_output"] for v in res.values()),
           "first_rows_compared": sum(v["first_rows_compared"] for v in res.values())}
    out["passes"] = bool(out["first_rows_compared"] > 0 and out["max_dev_kf0_eV"] <= 1e-6 and out["max_dev_kfx_eV"] <= 1e-6
                         and out["kf0_max_range_within_a_cut_eV"] <= 1e-9 and out["units_without_filter_output"] == 0)
    (M.CACHE / "kf_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({kk: v for kk, v in out.items() if kk != "per_fold"}), flush=True)
    return out


def main(params_fmt, folds):
    paths = {(r, k): Path(str(params_fmt).format(r=r, k=k)) for r, k in folds}
    missing = [f"r{r}_f{k}" for (r, k), p in paths.items() if not p.exists()]
    if missing:
        print(f"filter rows left out: B.70 phase-2 parameter files missing for {missing} ({params_fmt})", flush=True)
        return
    t0 = _time.time()
    X = M.B71()
    for (r, k), p in paths.items():
        key = f"r{r}_f{k}"
        prm = json.loads(p.read_text(encoding="utf-8"))
        o0, ox = fold(X, r, k, prm)
        np.savez_compressed(M.CACHE / f"kf_r{r}_f{k}.npz", kf0_F=o0, kfx_F=ox,
                            params=np.array(json.dumps({"file": str(p), key: prm})))
        M.log(t0, f"{key}: filter rows written ({np.isfinite(o0).sum()} of {len(o0)} F units with a filter output)")
