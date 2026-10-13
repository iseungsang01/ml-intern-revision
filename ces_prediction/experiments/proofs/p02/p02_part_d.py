"""P02 part D -- (i) multi-scale event-timing test on raw CSVs: k-step V_rot change
vs fast-channel activity inside the window; (ii) PCHIP fallback rows in the npz."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\lss\Documents\GitHub\ml-intern-revision"
NPZ = os.path.join(ROOT, "data", ".b1_seqv2_s42_i42", "comparison_errors_test.npz")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p02_part_d.json")
GAP_S = 0.5
MIN_PAIRS = 40
KS = (1, 2, 5, 10, 20)


def stuck_mask(values, block):
    stuck = np.zeros(values.shape, bool)
    last_val, last_block = np.nan, -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if block[i] == last_block and not np.isnan(last_val) and v == last_val:
            stuck[i] = True
        else:
            last_val, last_block = v, block[i]
    return stuck


def modality_mean(df, cols):
    a = df[cols].to_numpy(float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


class Pool:
    def __init__(self):
        self.sxy = self.sxx = self.syy = 0.0
        self.n = 0
        self.block_r = []

    def add(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        self.sxy += (x * y).sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.n += len(x)
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            self.block_r.append(float((x * y).sum() / den))

    def summary(self):
        br = np.array(self.block_r)
        den = np.sqrt(self.sxx * self.syy)
        return dict(r=float(self.sxy / den) if den > 0 else float("nan"), n_pairs=int(self.n),
                    n_blocks=len(br), se_block=float(br.std(ddof=1) / np.sqrt(len(br))) if len(br) > 1 else float("nan"),
                    frac_blocks_positive=float((br > 0).mean()) if len(br) else float("nan"))


def part_d_raw():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")))
    pools = {}
    for p in files:
        df = pd.read_csv(p)
        t = df["time"].to_numpy(float)
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP_S)))
        groups = {m: [c for c in df.columns if c.startswith(m)] for m in ("BES", "ECEI", "MC")}
        drv = {m: modality_mean(df, cols) for m, cols in groups.items() if cols}
        ddr = {m: np.concatenate(([np.nan], np.diff(v))) for m, v in drv.items()}
        tg = {}
        for c in ("CES_TI", "CES_VT"):
            v = df[c].to_numpy(float).copy()
            if c == "CES_TI":
                v[v > 3000] = np.nan
            v[stuck_mask(v, block)] = np.nan
            tg[c] = v
        for b in np.unique(block):
            sel = np.where(block == b)[0]
            if len(sel) < MIN_PAIRS:
                continue
            for c in ("CES_TI", "CES_VT"):
                y = tg[c][sel]
                for k in KS:
                    if len(y) <= k:
                        continue
                    dy = y[k:] - y[:-k]                      # k-step change, both ends observed
                    ok = np.isfinite(dy)
                    if ok.sum() < MIN_PAIRS:
                        continue
                    for m in drv:
                        x = drv[m][sel]
                        dx = ddr[m][sel]
                        dxk = x[k:] - x[:-k]                  # net change of the driver over the window
                        # activity inside the window: sum of |one-step driver increments| over (t, t+k]
                        adx = np.abs(dx)
                        adx[~np.isfinite(adx)] = 0.0
                        cs = np.concatenate(([0.0], np.cumsum(adx)))
                        act = cs[k + 1:len(cs)] - cs[1:len(cs) - k]   # sum over indices t+1..t+k
                        act = act[:len(dy)]
                        okk = ok & np.isfinite(dxk)
                        if okk.sum() < MIN_PAIRS:
                            continue
                        pools.setdefault((c, m, k, "signed_net"), Pool()).add(dxk[okk], dy[okk])
                        pools.setdefault((c, m, k, "abs_net"), Pool()).add(np.abs(dxk[okk]), np.abs(dy[okk]))
                        pools.setdefault((c, m, k, "abs_vs_activity"), Pool()).add(act[okk], np.abs(dy[okk]))
    return {f"{c}|{m}|k={k}|{kind}": pl.summary() for (c, m, k, kind), pl in sorted(pools.items())}


def part_d_npz():
    d = np.load(NPZ)
    out = {}
    for tgt in ("CES_VT", "CES_TI"):
        se_m = d[f"{tgt}_se_model"].astype(float)
        se_p = d[f"{tgt}_se_pchip"]
        se_s = d[f"{tgt}_se_persistence"]
        fb = np.isclose(se_p, se_s, rtol=0, atol=1e-9)
        o = dict(n=len(se_m), fallback_frac=float(fb.mean()))
        for name, m in (("fallback", fb), ("bridged", ~fb)):
            o[name] = dict(n=int(m.sum()),
                           skill_model_vs_pchip=float(1 - se_m[m].mean() / se_p[m].mean()),
                           median_ratio_model_over_pchip=float(np.median(np.sqrt(se_m[m])) / np.median(np.sqrt(se_p[m]))),
                           median_ratio_pchip_over_pers=float(np.median(np.sqrt(se_p[m])) / np.median(np.sqrt(se_s[m]))),
                           share_of_pchip_sse=float(se_p[m].sum() / se_p.sum()),
                           share_of_model_sse=float(se_m[m].sum() / se_m.sum()))
        out[tgt] = o
    return out


def main():
    res = dict(raw=part_d_raw(), npz=part_d_npz())
    json.dump(res, open(OUT, "w"), indent=1)
    print("=== D.i multi-scale: k-step target change vs driver (held-free, blocks>=40) ===")
    for key, v in res["raw"].items():
        print(f"  {key:40s} r={v['r']:+.4f} se_block={v['se_block']:.4f} n={v['n_pairs']:7d}"
              f" blocks={v['n_blocks']:4d} fracpos={v['frac_blocks_positive']:.3f}")
    print("=== D.ii PCHIP fallback rows in the s42 cut npz ===")
    for tgt, o in res["npz"].items():
        print(f"[{tgt}] n={o['n']} fallback_frac={o['fallback_frac']:.3f}")
        for name in ("fallback", "bridged"):
            v = o[name]
            print(f"   {name:9s} n={v['n']:6d} skill m/p={v['skill_model_vs_pchip']:+.3f}"
                  f" med m/p={v['median_ratio_model_over_pchip']:.3f} med p/pers={v['median_ratio_pchip_over_pers']:.3f}"
                  f" share_pchip_sse={v['share_of_pchip_sse']:.3f} share_model_sse={v['share_of_model_sse']:.3f}")
    print("json ->", OUT)


if __name__ == "__main__":
    main()
