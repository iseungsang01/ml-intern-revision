"""P01 part 2.
(A2) npz: verify sequential row order, build the two-sided transient measure
     J = max(|y_t - y_prev|, |y_next - y_t|) on 10 ms neighbours, stratify, and run the
     OU-bridge check against the acausal GP (the near-optimal two-sided arm).
(B2) raw CSVs: lag -1/0/+1 increment correlations in the tail (simultaneity test).
ASCII prints only.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = r"C:\Users\lss\Documents\GitHub\ml-intern-revision"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 1, 7, 123]
RES = {}


def sk(a, b):
    return 1.0 - float(a.sum()) / float(b.sum())


def fmt(x):
    return f"{x:+.3f}"


def a2(tag, path, tgt="CES_TI"):
    d = np.load(path)
    shot = d[f"{tgt}_shot"]; dt = d[f"{tgt}_dt_ms"]; y = d[f"{tgt}_y_true"].astype(np.float64)
    se_m = d[f"{tgt}_se_model"].astype(np.float64); se_p = d[f"{tgt}_se_pchip"].astype(np.float64)
    se_g = d[f"{tgt}_se_gp"].astype(np.float64); se_gc = d[f"{tgt}_se_gp_causal"].astype(np.float64)
    se_pe = d[f"{tgt}_se_persistence"].astype(np.float64); se_l = d[f"{tgt}_se_linear"].astype(np.float64)
    n = len(y)
    # --- sequential-order check: for row i+1 with same shot and dt==10, |y[i+1]-y[i]| must equal sqrt(se_pers[i+1])
    same = (shot[1:] == shot[:-1]) & (np.abs(dt[1:] - 10.0) < 0.6)
    back = np.sqrt(se_pe[1:])
    diff = np.abs(y[1:] - y[:-1])
    agree = np.abs(back - diff) < 1e-3 * np.maximum(1.0, back)
    frac_seq = float(agree[same].mean())
    # two-sided jump
    jb = np.sqrt(se_pe)                                   # backward jump (gap = dt_ms)
    jf = np.full(n, np.nan)
    idx = np.flatnonzero(same & agree)                    # row i has a verified 10 ms successor i+1
    jf[idx] = diff[idx]
    J = np.where(np.isfinite(jf), np.maximum(jb, jf), jb)
    has_f = np.isfinite(jf)
    qs = np.quantile(J, [0.5, 0.9, 0.95, 0.99])
    lab = np.digitize(J, qs)
    names = ["<q50", "q50-90", "q90-95", "q95-99", ">=q99"]
    r = {"n": n, "frac_sequential": frac_seq, "frac_has_forward": float(has_f.mean()),
         "quantiles_J": [float(q) for q in qs],
         "skill_vs_pchip": sk(se_m, se_p), "skill_vs_gp": sk(se_m, se_g),
         "gpc_vs_gp": sk(se_gc, se_g), "gpc_vs_pchip": sk(se_gc, se_p), "gp_vs_pchip": sk(se_g, se_p)}
    strata = {}
    acc_p = acc_g = 0.0
    for k, nm in enumerate(names):
        s = lab == k
        f_p = float(se_p[s].sum() / se_p.sum()); f_g = float(se_g[s].sum() / se_g.sum())
        st = {"n": int(s.sum()), "f_pchip": f_p, "f_gp": f_g, "f_model": float(se_m[s].sum() / se_m.sum()),
              "m_vs_pchip": sk(se_m[s], se_p[s]), "m_vs_gp": sk(se_m[s], se_g[s]),
              "gpc_vs_gp": sk(se_gc[s], se_g[s]), "gpc_vs_pchip": sk(se_gc[s], se_p[s]),
              "pers_vs_gp": sk(se_pe[s], se_g[s]), "m_vs_gpc": sk(se_m[s], se_gc[s]),
              "rmse_pchip": float(np.sqrt(se_p[s].mean())), "rmse_model": float(np.sqrt(se_m[s].mean())),
              "rmse_gp": float(np.sqrt(se_g[s].mean())), "medae_model": float(np.sqrt(np.median(se_m[s]))),
              "medae_pchip": float(np.sqrt(np.median(se_p[s]))), "medae_gp": float(np.sqrt(np.median(se_g[s])))}
        acc_p += f_p * st["m_vs_pchip"]; acc_g += f_g * st["m_vs_gp"]
        strata[nm] = st
    r["strata"] = strata
    r["identity_err_pchip"] = acc_p - r["skill_vs_pchip"]; r["identity_err_gp"] = acc_g - r["skill_vs_gp"]
    # top-5 formula: skill = f_I r_I - (1-f_I) l_B
    I = J >= qs[2]
    f_I = float(se_p[I].sum() / se_p.sum()); r_I = sk(se_m[I], se_p[I]); r_B = sk(se_m[~I], se_p[~I])
    r["top5"] = {"f_I_pchip": f_I, "r_I": r_I, "r_B": r_B, "l_B": -r_B, "formula": f_I * r_I + (1 - f_I) * r_B,
                 "sqvar_share_J_top5": float((J[I] ** 2).sum() / (J ** 2).sum()),
                 "f_I_gp": float(se_g[I].sum() / se_g.sum()), "r_I_gp": sk(se_m[I], se_g[I]), "r_B_gp": sk(se_m[~I], se_g[~I]),
                 "gpc_vs_gp_B": sk(se_gc[~I], se_g[~I]), "gpc_vs_gp_I": sk(se_gc[I], se_g[I])}
    # per-shot: skill vs transient share (J-based)
    ps = []
    for s_ in np.unique(shot):
        s = shot == s_
        if s.sum() < 30:
            continue
        ps.append((sk(se_m[s], se_p[s]), float(se_p[s & I].sum() / se_p[s].sum()),
                   float((J[s & I] ** 2).sum() / (J[s] ** 2).sum())))
    ps = np.array(ps)
    r["per_shot"] = {"n": int(len(ps)),
                     "rho_skill_fI": float(spearmanr(ps[:, 0], ps[:, 1])[0]), "p_fI": float(spearmanr(ps[:, 0], ps[:, 1])[1]),
                     "rho_skill_varshare": float(spearmanr(ps[:, 0], ps[:, 2])[0]), "p_var": float(spearmanr(ps[:, 0], ps[:, 2])[1])}
    terc = np.quantile(ps[:, 2], [1 / 3, 2 / 3]); tl = np.digitize(ps[:, 2], terc)
    r["per_shot"]["tercile_winrate"] = [float((ps[tl == k, 0] > 0).mean()) for k in range(3)]
    r["per_shot"]["tercile_median_skill"] = [float(np.median(ps[tl == k, 0])) for k in range(3)]
    RES[tag] = r
    print(f"\n[{tag} {tgt}] n={n} sequential={frac_seq:.4f} has_forward={r['frac_has_forward']:.3f} "
          f"J q50/q90/q95/q99={'/'.join(f'{q:.0f}' for q in qs)} | m vs PCHIP {fmt(r['skill_vs_pchip'])} m vs GP {fmt(r['skill_vs_gp'])} "
          f"GPc vs GP {fmt(r['gpc_vs_gp'])} GPc vs PCHIP {fmt(r['gpc_vs_pchip'])} GP vs PCHIP {fmt(r['gp_vs_pchip'])}")
    print("   stratum      n   f_pchip f_gp  | m/PCHIP  m/GP   GPc/GP  GPc/PCHIP pers/GP m/GPc | rmse P/M/G      medAE M/P/G")
    for nm, st in strata.items():
        print(f"   {nm:>7} {st['n']:6d}  {st['f_pchip']:.3f} {st['f_gp']:.3f} | {fmt(st['m_vs_pchip'])} {fmt(st['m_vs_gp'])} {fmt(st['gpc_vs_gp'])} "
              f"{fmt(st['gpc_vs_pchip'])}  {fmt(st['pers_vs_gp'])} {fmt(st['m_vs_gpc'])} | {st['rmse_pchip']:6.1f}/{st['rmse_model']:6.1f}/{st['rmse_gp']:6.1f} "
              f"{st['medae_model']:5.1f}/{st['medae_pchip']:5.1f}/{st['medae_gp']:5.1f}")
    t = r["top5"]
    print(f"   identity err (pchip/gp) = {r['identity_err_pchip']:.1e}/{r['identity_err_gp']:.1e}; top5 J: f_I={t['f_I_pchip']:.3f} r_I={fmt(t['r_I'])} "
          f"r_B={fmt(t['r_B'])} formula={fmt(t['formula'])} sqvar share={t['sqvar_share_J_top5']:.3f} | vs GP: f_I={t['f_I_gp']:.3f} r_I={fmt(t['r_I_gp'])} r_B={fmt(t['r_B_gp'])} "
          f"| GPc vs GP bulk {fmt(t['gpc_vs_gp_B'])} tail {fmt(t['gpc_vs_gp_I'])}")
    p = r["per_shot"]
    print(f"   per-shot n={p['n']} rho(skill,f_I)={p['rho_skill_fI']:+.3f} (p={p['p_fI']:.1e}) rho(skill,varshare)={p['rho_skill_varshare']:+.3f} (p={p['p_var']:.1e}) "
          f"terciles win={['%.2f' % w for w in p['tercile_winrate']]} med skill={['%+.3f' % w for w in p['tercile_median_skill']]}")


# ------------------------------------------------------------- B2
GAP = 0.5; CUT = 3000.0


def stuck_mask(values, block):
    stuck = np.zeros(values.shape, dtype=bool); last_val, last_block = np.nan, -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if block[i] == last_block and not np.isnan(last_val) and v == last_val:
            stuck[i] = True
        else:
            last_val, last_block = v, block[i]
    return stuck


def modality_mean(df, cols):
    a = df[cols].to_numpy(dtype=float); mu = np.nanmean(a, axis=0, keepdims=True); sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


def b2():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "s*.csv")))
    cols = {"dT": [], "dE_m1": [], "dE_0": [], "dE_p1": [], "dB_m1": [], "dB_0": [], "dB_p1": [], "lvlE": [], "lvlB": []}
    for f in files:
        df = pd.read_csv(f); t = df["time"].to_numpy(dtype=float)
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP)))
        ti = df["CES_TI"].to_numpy(dtype=float).copy(); ti[ti > CUT] = np.nan; ti[stuck_mask(ti, block)] = np.nan
        E = modality_mean(df, [c for c in df.columns if c.startswith("ECEI_")])
        Bm = modality_mean(df, [c for c in df.columns if c.startswith("BES_")])
        dE = np.full(len(t), np.nan); dB = np.full(len(t), np.nan)
        okstep = (block[1:] == block[:-1]) & (np.diff(t) < 0.015)
        dE[1:][okstep] = (E[1:] - E[:-1])[okstep]; dB[1:][okstep] = (Bm[1:] - Bm[:-1])[okstep]
        ok = np.isfinite(ti)
        pair = np.zeros(len(t), dtype=bool); pair[1:] = ok[1:] & ok[:-1] & okstep
        idx = np.flatnonzero(pair)
        idx = idx[(idx >= 2) & (idx <= len(t) - 2)]
        m = np.isfinite(dE[idx - 1]) & np.isfinite(dE[idx]) & np.isfinite(dE[idx + 1]) & np.isfinite(dB[idx - 1]) & np.isfinite(dB[idx]) & np.isfinite(dB[idx + 1])
        idx = idx[m]
        if idx.size == 0:
            continue
        cols["dT"].append(ti[idx] - ti[idx - 1])
        cols["dE_m1"].append(dE[idx - 1]); cols["dE_0"].append(dE[idx]); cols["dE_p1"].append(dE[idx + 1])
        cols["dB_m1"].append(dB[idx - 1]); cols["dB_0"].append(dB[idx]); cols["dB_p1"].append(dB[idx + 1])
        cols["lvlE"].append(E[idx]); cols["lvlB"].append(Bm[idx])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    a = np.abs(c["dT"]); q95 = np.quantile(a, 0.95)
    res = {"n_pairs": int(len(a)), "q95": float(q95)}
    for name, sel in (("all", np.ones_like(a, dtype=bool)), ("above_q95", a >= q95), ("below_q50", a < np.quantile(a, 0.5))):
        row = {"n": int(sel.sum())}
        for lag in ("m1", "0", "p1"):
            row[f"r_E_{lag}"] = float(pearsonr(c["dT"][sel], c[f"dE_{lag}"][sel])[0])
            row[f"r_B_{lag}"] = float(pearsonr(c["dT"][sel], c[f"dB_{lag}"][sel])[0])
        # Spearman at lag 0 (robust to outliers)
        row["rho_E_0"] = float(spearmanr(c["dT"][sel], c["dE_0"][sel])[0]); row["rho_B_0"] = float(spearmanr(c["dT"][sel], c["dB_0"][sel])[0])
        # level-rate (S8ar style) for reference
        row["r_lvlE_dT"] = float(pearsonr(c["lvlE"][sel], c["dT"][sel])[0]); row["r_lvlB_dT"] = float(pearsonr(c["lvlB"][sel], c["dT"][sel])[0])
        res[name] = row
        print(f"  [B2 {name:>9}] n={row['n']:7d} r(dT,dE) lag-1/0/+1 = {row['r_E_m1']:+.3f}/{row['r_E_0']:+.3f}/{row['r_E_p1']:+.3f} | "
              f"r(dT,dB) = {row['r_B_m1']:+.3f}/{row['r_B_0']:+.3f}/{row['r_B_p1']:+.3f} | spearman lag0 E {row['rho_E_0']:+.3f} B {row['rho_B_0']:+.3f} | "
              f"level-rate E {row['r_lvlE_dT']:+.3f} B {row['r_lvlB_dT']:+.3f}")
    RES["B2"] = res


if __name__ == "__main__":
    for s in SEEDS:
        a2(f"cut_s{s}", os.path.join(ROOT, "data", f".b1_seqv2_s{s}_i{s}", "comparison_errors_test.npz"))
    a2("cut_s42_VT", os.path.join(ROOT, "data", ".b1_seqv2_s42_i42", "comparison_errors_test.npz"), tgt="CES_VT")
    print("\n[cut T_i 4-split, J = max(back, fwd) top-5%] seed | skill | f_I | r_I | r_B | f_I*r_I | share of skill | GPc/GP bulk | m/GP bulk | m/GP tail")
    for s in SEEDS:
        r = RES[f"cut_s{s}"]; t = r["top5"]
        print(f"  s{s:<4} {fmt(r['skill_vs_pchip'])} {t['f_I_pchip']:.3f} {fmt(t['r_I'])} {fmt(t['r_B'])} {fmt(t['f_I_pchip'] * t['r_I'])} "
              f"{t['f_I_pchip'] * t['r_I'] / r['skill_vs_pchip']:.2f} {fmt(t['gpc_vs_gp_B'])} {fmt(t['r_B_gp'])} {fmt(t['r_I_gp'])}")
    print("\n[B2 lag structure, raw CSVs]")
    b2()
    with open(os.path.join(OUT, "p01_part2_results.json"), "w") as fh:
        json.dump(RES, fh, indent=1)
