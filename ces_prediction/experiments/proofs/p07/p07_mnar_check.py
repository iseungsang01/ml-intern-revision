"""P07 data check: is CES missingness concentrated where the plasma state moves?

Descriptive only (no model, no split, no TEST). Reads the 641 raw shot CSVs.
Blocks: a time gap >= 0.5 s starts a new block (dataset.py rule).
Two treatments of the masks:
  raw      : missing = NaN in the CSV
  genuine  : missing = NaN or a held repeat (value bit-identical to the previous
             observed value of the same target in the same block; the loader's
             _stuck_repeat_mask rule, applied to both targets as dataset.py does)

Outputs (ASCII only, console is cp949):
  (a)/(b) missing rate per activity quintile, for T_i and V_rot, with four
          activity proxies: neighbour jump J (CES neighbours within 5 steps,
          excluding the row itself), |dECEI| and |dBES| (centered increments of
          the per-file z-scored channel mean, defined at every row), and the
          BES level (density proxy, per-file z-score).
          Q5-Q1 difference with a shot-clustered bootstrap CI (B=1000).
  (c)     mask agreement, P(both), P(T)P(V), phi, conditional rates, and phi
          within |dECEI| quintiles (does a shared driver explain the coupling?)
  (d)     missing-run length distribution (median, mean, p90) and the fraction
          of missing rows with a past genuine observation within 2 / 4 rows.
Writes p07_results.json next to this script.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("C:/Users/lss/Documents/GitHub/ml-intern-revision", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p07_results.json")
GAP_S = 0.5
NEIGH = 5
RNG = np.random.default_rng(12345)
B_BOOT = 1000


def stuck_mask(values, block):
    mask = np.zeros(values.shape[0], dtype=bool)
    obs = np.flatnonzero(~np.isnan(values))
    if obs.size <= 1:
        return mask
    ov, ob = values[obs], block[obs]
    dup = (ov[1:] == ov[:-1]) & (ob[1:] == ob[:-1])
    mask[obs[1:][dup]] = True
    return mask


def neighbour_jump(values, block, k=NEIGH):
    """|y(t+k2) - y(t-k1)| with the nearest observed neighbours (1..k steps)
    on each side, same block, excluding the row itself. NaN when either side
    has no observed neighbour within k steps."""
    n = values.shape[0]
    out = np.full(n, np.nan)
    obs = ~np.isnan(values)
    idx = np.arange(n)
    # previous observed index (strictly before)
    prev = np.where(obs, idx, -1)
    prev = np.maximum.accumulate(np.concatenate(([-1], prev[:-1])))
    # next observed index (strictly after)
    nxt = np.where(obs, idx, n)
    nxt = np.minimum.accumulate(np.concatenate((nxt[1:], [n]))[::-1])[::-1]
    ok = (prev >= 0) & (nxt < n)
    ok &= (idx - prev <= k) & (nxt - idx <= k)
    ok[ok] &= (block[prev[ok]] == block[idx[ok]]) & (block[nxt[ok]] == block[idx[ok]])
    out[ok] = np.abs(values[nxt[ok]] - values[prev[ok]])
    return out


def centered_increment(x, block):
    """|x(t+1) - x(t-1)| within a block; NaN at block edges."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < 3:
        return out
    inner = np.arange(1, n - 1)
    same = (block[inner - 1] == block[inner]) & (block[inner + 1] == block[inner])
    out[inner[same]] = np.abs(x[inner + 1][same] - x[inner - 1][same])
    return out


def zscore(x):
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / s


def run_lengths(mask):
    """Lengths of True runs."""
    if mask.size == 0:
        return np.array([], dtype=int)
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts


def past_obs_within(miss, block, k):
    """For missing rows: is there an observed row within k rows in the past,
    same block?"""
    n = miss.shape[0]
    obs = ~miss
    idx = np.arange(n)
    prev = np.where(obs, idx, -1)
    prev = np.maximum.accumulate(np.concatenate(([-1], prev[:-1])))
    ok = (prev >= 0) & (idx - prev <= k)
    ok[ok] &= block[prev[ok]] == block[idx[ok]]
    return ok[miss]


def quintile_table(proxy, miss, shot, label):
    """Missing rate per proxy quintile (over rows where proxy is defined), the
    Q5-Q1 difference with a shot-clustered bootstrap CI, and Spearman-like
    monotonicity (rank corr between quintile index and rate)."""
    ok = ~np.isnan(proxy)
    p, m, s = proxy[ok], miss[ok], shot[ok]
    edges = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, p, side="right")  # 0..4
    rates, counts = [], []
    for i in range(5):
        sel = q == i
        counts.append(int(sel.sum()))
        rates.append(float(m[sel].mean()) if sel.any() else float("nan"))
    # shot-clustered bootstrap of Q5 - Q1
    shots = np.unique(s)
    # per-shot sums for fast resampling
    df = pd.DataFrame({"s": s, "q": q, "m": m.astype(float)})
    g = df.groupby(["s", "q"]).agg(n=("m", "size"), k=("m", "sum")).unstack(fill_value=0)
    n_tab = g["n"].reindex(index=shots, columns=range(5)).fillna(0).to_numpy()
    k_tab = g["k"].reindex(index=shots, columns=range(5)).fillna(0).to_numpy()
    diffs = np.empty(B_BOOT)
    for b in range(B_BOOT):
        pick = RNG.integers(0, len(shots), len(shots))
        n_b = n_tab[pick].sum(0)
        k_b = k_tab[pick].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = k_b / n_b
        diffs[b] = r[4] - r[0]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    mono = all(rates[i] <= rates[i + 1] for i in range(4))
    res = {
        "label": label,
        "rows": int(ok.sum()),
        "coverage": float(ok.mean()),
        "edges": [float(e) for e in edges],
        "counts": counts,
        "rates": rates,
        "q5_minus_q1": float(rates[4] - rates[0]),
        "ci95": [float(lo), float(hi)],
        "ratio_q5_q1": float(rates[4] / rates[0]) if rates[0] > 0 else float("nan"),
        "monotone": bool(mono),
    }
    print("  %-28s n=%7d cov=%.3f  rates=%s  Q5-Q1=%+.4f [%+.4f,%+.4f] ratio=%.2f mono=%s" % (
        label, res["rows"], res["coverage"], " ".join("%.4f" % r for r in rates),
        res["q5_minus_q1"], lo, hi, res["ratio_q5_q1"], mono))
    return res


def phi_coef(a, b):
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def agreement_block(mT, mV, label):
    pT, pV = float(mT.mean()), float(mV.mean())
    both = float((mT & mV).mean())
    agree = float((mT == mV).mean())
    res = {
        "label": label,
        "p_T": pT, "p_V": pV,
        "p_both": both, "p_T_times_p_V": pT * pV,
        "p_V_given_T_miss": float(mV[mT].mean()) if mT.any() else float("nan"),
        "p_V_given_T_obs": float(mV[~mT].mean()),
        "p_T_given_V_miss": float(mT[mV].mean()) if mV.any() else float("nan"),
        "p_T_given_V_obs": float(mT[~mV].mean()),
        "agreement": agree,
        "agreement_if_independent": pT * pV + (1 - pT) * (1 - pV),
        "phi": phi_coef(mT, mV),
    }
    print("  %-8s P(T)=%.4f P(V)=%.4f P(both)=%.4f P(T)P(V)=%.4f agree=%.4f indep_agree=%.4f phi=%.4f  P(V|Tmiss)=%.4f P(V|Tobs)=%.4f" % (
        label, pT, pV, both, pT * pV, agree, res["agreement_if_independent"], res["phi"],
        res["p_V_given_T_miss"], res["p_V_given_T_obs"]))
    return res


def run_stats(miss, block, label):
    # runs must not cross blocks: split per block
    lens = []
    for bval in np.unique(block):
        sel = block == bval
        lens.append(run_lengths(miss[sel]))
    lens = np.concatenate(lens) if lens else np.array([], dtype=int)
    res = {
        "label": label,
        "n_runs": int(lens.size),
        "median": float(np.median(lens)) if lens.size else float("nan"),
        "mean": float(lens.mean()) if lens.size else float("nan"),
        "p90": float(np.quantile(lens, 0.9)) if lens.size else float("nan"),
        "frac_runs_len1": float((lens == 1).mean()) if lens.size else float("nan"),
        "frac_missing_rows_in_runs_le2": float(lens[lens <= 2].sum() / lens.sum()) if lens.size else float("nan"),
        "in_domain_W2": float(past_obs_within(miss, block, 2).mean()) if miss.any() else float("nan"),
        "in_domain_W4": float(past_obs_within(miss, block, 4).mean()) if miss.any() else float("nan"),
    }
    print("  %-10s runs=%6d median=%.0f mean=%.2f p90=%.0f len1=%.3f rows_in_runs<=2=%.3f inW2=%.3f inW4=%.3f" % (
        label, res["n_runs"], res["median"], res["mean"], res["p90"], res["frac_runs_len1"],
        res["frac_missing_rows_in_runs_le2"], res["in_domain_W2"], res["in_domain_W4"]))
    return res


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print("no CSVs found in %s" % DATA_DIR)
        sys.exit(1)
    cols = {"time": [], "ti": [], "vt": [], "ecei": [], "bes": [], "block": [], "shot": []}
    nblocks = 0
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        t = df["time"].to_numpy(float)
        blk = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP_S))) + nblocks
        nblocks = int(blk.max()) + 1
        bes_cols = [c for c in df.columns if c.startswith("BES_")]
        ecei_cols = [c for c in df.columns if c.startswith("ECEI_")]
        cols["time"].append(t)
        cols["ti"].append(df["CES_TI"].to_numpy(float))
        cols["vt"].append(df["CES_VT"].to_numpy(float))
        cols["ecei"].append(zscore(df[ecei_cols].to_numpy(float).mean(1)))
        cols["bes"].append(zscore(df[bes_cols].to_numpy(float).mean(1)))
        cols["block"].append(blk)
        cols["shot"].append(np.full(len(t), i))
    T = np.concatenate(cols["ti"]); V = np.concatenate(cols["vt"])
    E = np.concatenate(cols["ecei"]); Bm = np.concatenate(cols["bes"])
    block = np.concatenate(cols["block"]); shot = np.concatenate(cols["shot"])
    n = T.shape[0]
    print("files %d rows %d blocks %d" % (len(files), n, nblocks))

    # masks
    missT_raw, missV_raw = np.isnan(T), np.isnan(V)
    heldT, heldV = stuck_mask(T, block), stuck_mask(V, block)
    missT_gen, missV_gen = missT_raw | heldT, missV_raw | heldV
    T_gen = T.copy(); T_gen[heldT] = np.nan
    V_gen = V.copy(); V_gen[heldV] = np.nan
    summary = {
        "files": len(files), "rows": n, "blocks": nblocks,
        "missT_raw": float(missT_raw.mean()), "missV_raw": float(missV_raw.mean()),
        "heldT_frac_rows": float(heldT.mean()), "heldV_frac_rows": float(heldV.mean()),
        "heldT_frac_obs": float(heldT.sum() / (~missT_raw).sum()),
        "heldV_frac_obs": float(heldV.sum() / (~missV_raw).sum()),
        "missT_gen": float(missT_gen.mean()), "missV_gen": float(missV_gen.mean()),
    }
    print("missing raw  T=%.4f V=%.4f | held rows T=%.4f V=%.4f (of observed T=%.4f V=%.4f) | genuine-missing T=%.4f V=%.4f" % (
        summary["missT_raw"], summary["missV_raw"], summary["heldT_frac_rows"], summary["heldV_frac_rows"],
        summary["heldT_frac_obs"], summary["heldV_frac_obs"], summary["missT_gen"], summary["missV_gen"]))

    # activity proxies
    dE = centered_increment(E, block)
    dB = centered_increment(Bm, block)
    JT_raw = neighbour_jump(T, block)
    JV_raw = neighbour_jump(V, block)
    JT_gen = neighbour_jump(T_gen, block)
    JV_gen = neighbour_jump(V_gen, block)
    # a V_rot proxy independent of V_rot's own held pattern: the T_i jump
    out = {"summary": summary, "quintiles": {}, "agreement": {}, "phi_within_dE_quintile": {}, "runs": {}}

    print("\n(a) T_i missing rate per activity quintile")
    out["quintiles"]["T_raw"] = [
        quintile_table(JT_raw, missT_raw, shot, "J_Ti (raw mask)"),
        quintile_table(dE, missT_raw, shot, "|dECEI| (raw mask)"),
        quintile_table(dB, missT_raw, shot, "|dBES| (raw mask)"),
        quintile_table(Bm, missT_raw, shot, "BES level (raw mask)"),
        quintile_table(E, missT_raw, shot, "ECEI level (raw mask)"),
    ]
    out["quintiles"]["T_gen"] = [
        quintile_table(JT_gen, missT_gen, shot, "J_Ti (genuine mask)"),
        quintile_table(dE, missT_gen, shot, "|dECEI| (genuine mask)"),
        quintile_table(dB, missT_gen, shot, "|dBES| (genuine mask)"),
        quintile_table(Bm, missT_gen, shot, "BES level (genuine mask)"),
    ]
    print("\n(b) V_rot missing rate per activity quintile")
    out["quintiles"]["V_raw"] = [
        quintile_table(JV_raw, missV_raw, shot, "J_Vrot (raw mask)"),
        quintile_table(JT_raw, missV_raw, shot, "J_Ti (raw V mask)"),
        quintile_table(dE, missV_raw, shot, "|dECEI| (raw mask)"),
        quintile_table(dB, missV_raw, shot, "|dBES| (raw mask)"),
        quintile_table(Bm, missV_raw, shot, "BES level (raw mask)"),
    ]
    out["quintiles"]["V_gen"] = [
        quintile_table(JV_gen, missV_gen, shot, "J_Vrot (genuine mask)"),
        quintile_table(JT_gen, missV_gen, shot, "J_Ti (genuine V mask)"),
        quintile_table(dE, missV_gen, shot, "|dECEI| (genuine mask)"),
        quintile_table(dB, missV_gen, shot, "|dBES| (genuine mask)"),
        quintile_table(Bm, missV_gen, shot, "BES level (genuine mask)"),
    ]

    print("\n(b2) stratified: missing rate by BES-level quintile (rows) x activity quintile (cols), genuine masks")
    out["stratified"] = {}
    for name, act in (("dECEI", dE), ("J_Ti", JT_gen), ("dBES", dB)):
        out["stratified"][name] = cross_table(Bm, act, missT_gen, missV_gen, name)

    print("\n(c) mask agreement")
    out["agreement"]["raw"] = agreement_block(missT_raw, missV_raw, "raw")
    out["agreement"]["genuine"] = agreement_block(missT_gen, missV_gen, "genuine")
    # phi within |dECEI| quintiles (shared driver -> coupling should weaken within strata)
    ok = ~np.isnan(dE)
    edges = np.quantile(dE[ok], [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, dE[ok], side="right")
    for name, mT, mV in (("raw", missT_raw, missV_raw), ("genuine", missT_gen, missV_gen)):
        rows = []
        for i in range(5):
            sel = q == i
            a, b = mT[ok][sel], mV[ok][sel]
            rows.append({"q": i + 1, "n": int(sel.sum()), "p_T": float(a.mean()), "p_V": float(b.mean()),
                         "p_both": float((a & b).mean()), "phi": phi_coef(a, b)})
        out["phi_within_dE_quintile"][name] = rows
        print("  %-8s phi within |dECEI| quintiles: %s" % (name, " ".join("Q%d=%.3f" % (r["q"], r["phi"]) for r in rows)))
        print("  %-8s P(both) within |dECEI| quintiles: %s" % (name, " ".join("Q%d=%.4f" % (r["q"], r["p_both"]) for r in rows)))

    print("\n(d) missing-run lengths")
    out["runs"]["T_raw"] = run_stats(missT_raw, block, "T raw")
    out["runs"]["T_gen"] = run_stats(missT_gen, block, "T genuine")
    out["runs"]["V_raw"] = run_stats(missV_raw, block, "V raw")
    out["runs"]["V_gen"] = run_stats(missV_gen, block, "V genuine")

    # extra: T_i neighbour level (low-N proxy via relative error) and spike proximity
    print("\n(e) extra: missing rate vs T_i neighbour mean level (genuine mask)")
    # mean of the two nearest observed neighbours (same construction as J)
    out["extra"] = {}
    out["extra"]["T_level"] = quintile_table(_neighbour_mean(T_gen, block), missT_gen, shot, "Ti neighbour level (gen)")
    out["extra"]["V_vs_Tlevel"] = quintile_table(_neighbour_mean(T_gen, block), missV_gen, shot, "Vrot miss vs Ti level (gen)")

    # (f) span-matched jump: J2 = |y(t+1) - y(t-1)| only where BOTH immediate
    # neighbours are observed, so an isolated missing frame and an observed
    # frame are compared over the same 20 ms span (removes the mechanical
    # "longer span -> larger jump" bias of J).  Also the 3 keV cut population
    # (T_i > 3000 eV treated as missing, the protocol's fit-failure rule).
    print("\n(f) span-matched J2 (both immediate neighbours observed) and 3 keV cut variant")
    T_cut = T_gen.copy(); T_cut[T_cut > 3000.0] = np.nan
    missT_cut = np.isnan(T_cut)
    out["span_matched"] = {
        "T_gen": quintile_table(neighbour_jump(T_gen, block, k=1), missT_gen, shot, "J2_Ti (genuine)"),
        "T_cut": quintile_table(neighbour_jump(T_cut, block, k=1), missT_cut, shot, "J2_Ti (3keV cut)"),
        "V_gen": quintile_table(neighbour_jump(V_gen, block, k=1), missV_gen, shot, "J2_Vrot (genuine)"),
        "V_raw": quintile_table(neighbour_jump(V, block, k=1), missV_raw, shot, "J2_Vrot (raw)"),
    }
    out["cut_variant"] = {
        "J_Ti_cut": quintile_table(neighbour_jump(T_cut, block), missT_cut, shot, "J_Ti (3keV cut, k=5)"),
        "BES_level_cut": quintile_table(Bm, missT_cut, shot, "BES level (3keV cut)"),
        "dBES_cut": quintile_table(dB, missT_cut, shot, "|dBES| (3keV cut)"),
        "missT_cut": float(missT_cut.mean()),
        "spike_rows": int((T_gen > 3000.0).sum()),
    }
    print("  T_i missing after 3 keV cut = %.4f  (spike rows %d)" % (out["cut_variant"]["missT_cut"], out["cut_variant"]["spike_rows"]))
    # J2 within BES level strata, T genuine
    print("  J2_Ti x BES level (genuine):")
    out["span_matched"]["J2_x_BES"] = cross_table(Bm, neighbour_jump(T_gen, block, k=1), missT_gen, missV_gen, "J2_Ti")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %s" % OUT)


def cross_table(level, act, mT, mV, name):
    ok = ~np.isnan(level) & ~np.isnan(act)
    L, A = level[ok], act[ok]
    le = np.quantile(L, [0.2, 0.4, 0.6, 0.8]); ae = np.quantile(A, [0.2, 0.4, 0.6, 0.8])
    ql = np.searchsorted(le, L, side="right"); qa = np.searchsorted(ae, A, side="right")
    res = {"T": [], "V": [], "n": []}
    for i in range(5):
        rowT, rowV, rown = [], [], []
        for j in range(5):
            sel = (ql == i) & (qa == j)
            rown.append(int(sel.sum()))
            rowT.append(float(mT[ok][sel].mean()) if sel.any() else float("nan"))
            rowV.append(float(mV[ok][sel].mean()) if sel.any() else float("nan"))
        res["T"].append(rowT); res["V"].append(rowV); res["n"].append(rown)
        print("  %-6s BESq%d  T: %s   V: %s   n: %s" % (name, i + 1,
              " ".join("%.4f" % r for r in rowT), " ".join("%.4f" % r for r in rowV), " ".join("%6d" % c for c in rown)))
    return res


def _neighbour_mean(values, block, k=NEIGH):
    n = values.shape[0]
    out = np.full(n, np.nan)
    obs = ~np.isnan(values)
    idx = np.arange(n)
    prev = np.where(obs, idx, -1)
    prev = np.maximum.accumulate(np.concatenate(([-1], prev[:-1])))
    nxt = np.where(obs, idx, n)
    nxt = np.minimum.accumulate(np.concatenate((nxt[1:], [n]))[::-1])[::-1]
    ok = (prev >= 0) & (nxt < n) & (idx - prev <= k) & (nxt - idx <= k)
    ok[ok] &= (block[prev[ok]] == block[idx[ok]]) & (block[nxt[ok]] == block[idx[ok]])
    out[ok] = 0.5 * (values[nxt[ok]] + values[prev[ok]])
    return out


if __name__ == "__main__":
    main()
