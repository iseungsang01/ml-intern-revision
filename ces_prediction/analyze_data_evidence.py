# -*- coding: utf-8 -*-
"""Regenerate the three data-level evidence claims used in the thesis/talk.

Run from the repo root (set ``CES_DATA_DIR`` if the shot CSVs live elsewhere)::

    python ces_prediction/analyze_data_evidence.py

Everything here reads the real shot CSVs only — no synthetic data, no sampling
shortcuts for the headline missingness ledger (that one is a full 641-file scan).

A. Missingness ledger — NaN missing vs. held ("same value padded") rows.
   Counting NaN alone understates CES_VT badly: NaN is 23.9%, but 41.1% more of
   the grid repeats the previous observation bit-for-bit, so 65.0% of rows carry
   no independent CES_VT information. CES_TI is unaffected (1 held row in 641
   files), which is why the audit is a CES_VT-specific instrument characteristic.

B. Mirnov (MC) lag-1 autocorrelation vs. BES/ECEI on the same 100 Hz grid.
   BES/ECEI are temporally continuous (r ~ +0.57); MC is exactly uncorrelated
   (r ~ 0.00). dB/dt oscillates at the kHz mode frequency, so sampling it at
   100 Hz without an anti-aliasing filter folds those components in at random
   phase — the amplitude/mode information is gone before the model sees it.

C. Does Te proxy the NBI torque? ECEI (Te surrogate) correlated against both
   targets, between shots and within shots. The first half of the causal chain
   holds (Te ~ CES_TI is strong), the second half does not (Te ~ CES_VT is null)
   — power is not torque.

Results are printed and written to ``<out>/data_evidence.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CONTIGUOUS_GAP_S = 0.5  # same block iff time delta < 0.5 s (matches dataset.py)


def shot_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "s*.csv")))
    if not files:
        raise SystemExit(
            f"no shot CSVs under {data_dir!r}. Point CES_DATA_DIR at the real data folder."
        )
    return files


def contiguous_blocks(t: np.ndarray, min_len: int = 4) -> list[tuple[int, int]]:
    out, start = [], 0
    for i in range(1, len(t)):
        d = t[i] - t[i - 1]
        if d >= CONTIGUOUS_GAP_S or d <= 0:
            if i - start >= min_len:
                out.append((start, i))
            start = i
    if len(t) - start >= min_len:
        out.append((start, len(t)))
    return out


# --------------------------------------------------------------------------
# A. missingness ledger
# --------------------------------------------------------------------------
def missingness_ledger(files: list[str]) -> dict:
    acc = {c: dict(n_rows=0, nan=0, obs=0, held=0, files_held=0, files_allnan=0, runs=[])
           for c in ("CES_TI", "CES_VT")}
    for f in files:
        df = pd.read_csv(f, usecols=["time", "CES_TI", "CES_VT"])
        n = len(df)
        for col, a in acc.items():
            a["n_rows"] += n
            s = df[col]
            nan = int(s.isna().sum())
            a["nan"] += nan
            if nan == n:
                a["files_allnan"] += 1
            v = s.dropna().to_numpy()
            a["obs"] += len(v)
            if len(v) < 2:
                continue
            same = v[1:] == v[:-1]
            held = int(same.sum())
            a["held"] += held
            if held:
                a["files_held"] += 1
                run = 1
                for flag in same:
                    if flag:
                        run += 1
                    else:
                        if run > 1:
                            a["runs"].append(run)
                        run = 1
                if run > 1:
                    a["runs"].append(run)

    out = {}
    for col, a in acc.items():
        runs = np.array(a["runs"]) if a["runs"] else np.array([0])
        N, obs, held, nan = a["n_rows"], a["obs"], a["held"], a["nan"]
        out[col] = {
            "n_rows": N,
            "nan": nan, "nan_pct": 100 * nan / N,
            "held": held, "held_pct_of_rows": 100 * held / N,
            "held_pct_of_observed": 100 * held / obs if obs else 0.0,
            "effective_missing": nan + held,
            "effective_missing_pct": 100 * (nan + held) / N,
            "independent_obs": obs - held,
            "independent_obs_pct": 100 * (obs - held) / N,
            "files_with_held": a["files_held"], "files_all_nan": a["files_allnan"],
            "n_files": len(files),
            "held_run_median": float(np.median(runs)),
            "held_run_max": int(runs.max()),
        }
    return out


# --------------------------------------------------------------------------
# B. lag-1 autocorrelation by diagnostic family
# --------------------------------------------------------------------------
def lag1_autocorrelation(files: list[str]) -> dict:
    acc: dict[str, list[float]] = {}
    for f in files:
        df = pd.read_csv(f)
        t = df["time"].to_numpy()
        blocks = contiguous_blocks(t, min_len=8)
        if not blocks:
            continue
        for c in df.columns:
            if c.startswith("BES_"):
                fam = "BES"
            elif c.startswith("ECEI_"):
                fam = "ECEI"
            elif c.startswith("MC"):
                fam = "MC"
            else:
                continue
            v = df[c].to_numpy(dtype=float)
            for a, b in blocks:
                seg = v[a:b]
                if len(seg) < 8 or not np.all(np.isfinite(seg)) or seg.std() == 0:
                    continue
                r = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
                if np.isfinite(r):
                    acc.setdefault(fam, []).append(r)

    out = {}
    for fam, vals in acc.items():
        a = np.array(vals)
        out[fam] = {"n_blocks": int(a.size), "mean_r": float(a.mean()),
                    "median_r": float(np.median(a)),
                    "frac_abs_r_below_0.1": float((np.abs(a) < 0.1).mean())}
    return out


# --------------------------------------------------------------------------
# C. does Te proxy the NBI torque?
# --------------------------------------------------------------------------
def te_torque_probe(files: list[str], min_obs: int = 20) -> dict:
    from scipy import stats

    rows, within = [], {k: [] for k in ("Te~CES_TI", "Te~CES_VT")}
    for f in files:
        df = pd.read_csv(f)
        ec = [c for c in df.columns if c.startswith("ECEI_")]
        be = [c for c in df.columns if c.startswith("BES_")]
        if not ec:
            continue
        te = df[ec].mean(axis=1)
        rows.append({
            "te_mean": te.mean(), "te_std": te.std(),
            "bes_std": df[be].std().mean() if be else np.nan,
            "ti_mean": df["CES_TI"].mean(),
            "vt_mean": df["CES_VT"].mean(),
            "vt_absmean": df["CES_VT"].abs().mean(),
            "n_ti": int(df["CES_TI"].notna().sum()),
            "n_vt": int(df["CES_VT"].notna().sum()),
        })
        tev = te.to_numpy(dtype=float)
        for a, b in contiguous_blocks(df["time"].to_numpy(), min_len=10):
            for tgt, key in (("CES_TI", "Te~CES_TI"), ("CES_VT", "Te~CES_VT")):
                y = df[tgt].to_numpy(dtype=float)[a:b]
                x = tev[a:b]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 10 or x[m].std() == 0 or y[m].std() == 0:
                    continue
                within[key].append(float(np.corrcoef(x[m], y[m])[0, 1]))

    t = pd.DataFrame(rows)
    t = t[(t["n_ti"] >= min_obs) & (t["n_vt"] >= min_obs)]

    def pair(x, y):
        m = np.isfinite(t[x]) & np.isfinite(t[y])
        r, p = stats.pearsonr(t[x][m], t[y][m])
        rs, ps = stats.spearmanr(t[x][m], t[y][m])
        return {"n_shots": int(m.sum()), "pearson_r": float(r), "pearson_p": float(p),
                "spearman_rho": float(rs), "spearman_p": float(ps)}

    between = {
        "Te~CES_TI": pair("te_mean", "ti_mean"),
        "Te~CES_VT": pair("te_mean", "vt_mean"),
        "Te~|CES_VT|": pair("te_mean", "vt_absmean"),
        "Te_std~|CES_VT|": pair("te_std", "vt_absmean"),
        "BES_std~|CES_VT|": pair("bes_std", "vt_absmean"),
    }
    within_out = {}
    for k, vals in within.items():
        a = np.array([v for v in vals if np.isfinite(v)])
        within_out[k] = {"n_blocks": int(a.size), "mean_r": float(a.mean()),
                         "mean_abs_r": float(np.abs(a).mean()),
                         "frac_abs_r_above_0.3": float((np.abs(a) > 0.3).mean())}
    return {"between_shot": between, "within_shot": within_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("CES_DATA_DIR", "data"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--corr-sample", type=int, default=0,
                    help="limit shots for parts B/C (0 = use all; A always uses all)")
    args = ap.parse_args()

    files = shot_files(args.data_dir)
    print(f"shot files: {len(files)}  (data-dir: {args.data_dir})")

    print("\n=== A. missingness ledger (all files) ===")
    ledger = missingness_ledger(files)
    for col, v in ledger.items():
        print(f"{col}: rows={v['n_rows']:,}  NaN={v['nan']:,} ({v['nan_pct']:.1f}%)  "
              f"held={v['held']:,} ({v['held_pct_of_rows']:.1f}% of rows, "
              f"{v['held_pct_of_observed']:.1f}% of observed)  "
              f"effective-missing={v['effective_missing']:,} ({v['effective_missing_pct']:.1f}%)  "
              f"files_with_held={v['files_with_held']}/{v['n_files']}  "
              f"run median/max={v['held_run_median']:.0f}/{v['held_run_max']}")

    sub = files
    if args.corr_sample and args.corr_sample < len(files):
        sub = list(np.random.default_rng(0).choice(files, size=args.corr_sample, replace=False))
        print(f"\n(parts B/C on a {len(sub)}-shot sample, seed 0)")

    print("\n=== B. lag-1 autocorrelation within contiguous blocks ===")
    lag1 = lag1_autocorrelation(sub)
    for fam in ("BES", "ECEI", "MC"):
        if fam in lag1:
            v = lag1[fam]
            print(f"{fam:5s} n_blocks={v['n_blocks']:6d}  mean r={v['mean_r']:+.3f}  "
                  f"median r={v['median_r']:+.3f}  |r|<0.1: {v['frac_abs_r_below_0.1']:.1%}")

    print("\n=== C. does Te proxy the NBI torque? ===")
    te = te_torque_probe(sub)
    for k, v in te["between_shot"].items():
        print(f"between-shot {k:18s} n={v['n_shots']:4d}  r={v['pearson_r']:+.3f} "
              f"(p={v['pearson_p']:.2g})  rho={v['spearman_rho']:+.3f}")
    for k, v in te["within_shot"].items():
        print(f"within-shot  {k:18s} n={v['n_blocks']:5d}  mean r={v['mean_r']:+.3f}  "
              f"mean |r|={v['mean_abs_r']:.3f}  |r|>0.3: {v['frac_abs_r_above_0.3']:.1%}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "data_evidence.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n_files": len(files), "missingness": ledger,
                   "lag1_autocorr": lag1, "te_torque": te}, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
