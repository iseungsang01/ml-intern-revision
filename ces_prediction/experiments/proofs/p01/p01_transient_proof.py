"""P01 -- the T_i headline skill is transient physics.

Part A (frozen TEST npz, no training / scoring):
  skill decomposition by stratum (is_peak, and backward-jump quantile),
  exact identity  skill = sum_k f_k * s_k  (f_k = PCHIP SSE share, s_k = stratum skill),
  per-shot margin vs per-shot transient share.
Part B (raw CSVs, 641 files, held-free, 3 keV cut, blocks at gap >= 0.5 s):
  Pearson r between dT_i and d(ECEI mean), d(BES mean) -- all rows vs |dT_i| >= q95.
All prints are ASCII.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = r"C:\Users\lss\Documents\GitHub\ml-intern-revision"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 1, 7, 123]
RES = {}


def skill(se_m, se_b):
    return 1.0 - float(se_m.sum()) / float(se_b.sum())


def strata_decomp(se_m, se_b, labels, names):
    """Exact identity: skill = sum_k f_k s_k with f_k = SSE_b share, s_k = stratum skill."""
    tot = skill(se_m, se_b)
    rows = []
    acc = 0.0
    for k, nm in enumerate(names):
        sel = labels == k
        if sel.sum() == 0:
            rows.append((nm, 0, 0.0, 0.0, float("nan"), 0.0))
            continue
        f_b = float(se_b[sel].sum() / se_b.sum())
        f_m = float(se_m[sel].sum() / se_m.sum())
        s_k = skill(se_m[sel], se_b[sel])
        acc += f_b * s_k
        rows.append((nm, int(sel.sum()), f_b, f_m, s_k, f_b * s_k))
    return tot, acc, rows


def part_a(tag, path):
    d = np.load(path)
    out = {}
    for tgt in ("CES_TI", "CES_VT"):
        se_m = d[f"{tgt}_se_model"].astype(np.float64)
        se_p = d[f"{tgt}_se_pchip"].astype(np.float64)
        se_pers = d[f"{tgt}_se_persistence"].astype(np.float64)
        se_gpc = d[f"{tgt}_se_gp_causal"].astype(np.float64)
        peak = d[f"{tgt}_is_peak"].astype(bool)
        shot = d[f"{tgt}_shot"]
        n = len(se_m)
        r = {"n": n, "skill_vs_pchip": skill(se_m, se_p),
             "skill_vs_pers": skill(se_m, se_pers),
             "skill_gpcausal_vs_pchip": skill(se_gpc, se_p)}

        # --- (1) is_peak decomposition
        tot, acc, rows = strata_decomp(se_m, se_p, peak.astype(int), ["nonpeak", "peak"])
        r["peak"] = {nm: {"n": nn, "f_pchip": fb, "f_model": fm, "skill": sk, "contrib": c}
                     for nm, nn, fb, fm, sk, c in rows}
        r["peak"]["identity_sum"] = acc
        r["peak"]["identity_err"] = acc - tot
        r["peak"]["peak_frac_rows"] = float(peak.mean())
        # causal GP (history-only) in the same strata -> OU-bridge check
        r["peak"]["gpcausal_skill_nonpeak"] = skill(se_gpc[~peak], se_p[~peak])
        r["peak"]["gpcausal_skill_peak"] = skill(se_gpc[peak], se_p[peak])

        # --- (2) backward-jump |y_t - y_prev| = sqrt(se_persistence) quantile strata
        jump = np.sqrt(se_pers)
        qs = np.quantile(jump, [0.5, 0.9, 0.95, 0.99])
        lab = np.digitize(jump, qs)  # 0:<q50 1:q50-90 2:q90-95 3:q95-99 4:>=q99
        names = ["<q50", "q50-90", "q90-95", "q95-99", ">=q99"]
        tot2, acc2, rows2 = strata_decomp(se_m, se_p, lab, names)
        r["jump"] = {"quantiles": [float(x) for x in qs],
                     "strata": {nm: {"n": nn, "f_pchip": fb, "f_model": fm, "skill": sk, "contrib": c}
                                for nm, nn, fb, fm, sk, c in rows2},
                     "identity_err": acc2 - tot2}
        # causal GP in jump strata
        r["jump"]["gpcausal_skill"] = {nm: skill(se_gpc[lab == k], se_p[lab == k]) for k, nm in enumerate(names)}
        # share of squared backward variation in the top 5 %
        top5 = jump >= qs[2]
        r["jump"]["sq_var_share_top5"] = float(se_pers[top5].sum() / se_pers.sum())
        r["jump"]["f_pchip_top5"] = float(se_p[top5].sum() / se_p.sum())
        r["jump"]["skill_top5"] = skill(se_m[top5], se_p[top5])
        r["jump"]["skill_bottom95"] = skill(se_m[~top5], se_p[~top5])
        r["jump"]["skill_pers_vs_pchip_bottom95"] = skill(se_pers[~top5], se_p[~top5])
        # formula with I = top5 : skill = f_I r_I - (1-f_I) l_B
        f_I = r["jump"]["f_pchip_top5"]; r_I = r["jump"]["skill_top5"]; l_B = -r["jump"]["skill_bottom95"]
        r["jump"]["formula_top5"] = f_I * r_I - (1 - f_I) * l_B

        # --- (3) per-shot: margin vs transient share
        shots = np.unique(shot)
        ps = []
        for s in shots:
            sel = shot == s
            if sel.sum() < 30:
                continue
            sk = skill(se_m[sel], se_p[sel])
            fpk = float(se_p[sel & peak].sum() / se_p[sel].sum()) if se_p[sel].sum() > 0 else np.nan
            ft5 = float(se_p[sel & top5].sum() / se_p[sel].sum()) if se_p[sel].sum() > 0 else np.nan
            # share of the shot's own squared backward variation in its top-5% (global threshold) rows
            vshare = float(se_pers[sel & top5].sum() / se_pers[sel].sum()) if se_pers[sel].sum() > 0 else np.nan
            ps.append((sk, fpk, ft5, vshare, int(sel.sum())))
        ps = np.array(ps)
        ok = np.isfinite(ps).all(axis=1)
        ps = ps[ok]
        r["per_shot"] = {"n_shots": int(len(ps)),
                         "spearman_skill_vs_fpeak": float(spearmanr(ps[:, 0], ps[:, 1])[0]),
                         "p_fpeak": float(spearmanr(ps[:, 0], ps[:, 1])[1]),
                         "spearman_skill_vs_ftop5": float(spearmanr(ps[:, 0], ps[:, 2])[0]),
                         "p_ftop5": float(spearmanr(ps[:, 0], ps[:, 2])[1]),
                         "spearman_skill_vs_varshare": float(spearmanr(ps[:, 0], ps[:, 3])[0]),
                         "p_varshare": float(spearmanr(ps[:, 0], ps[:, 3])[1]),
                         "win_rate": float((ps[:, 0] > 0).mean())}
        # terciles of transient share
        terc = np.quantile(ps[:, 3], [1 / 3, 2 / 3])
        tl = np.digitize(ps[:, 3], terc)
        r["per_shot"]["tercile_winrate"] = [float((ps[tl == k, 0] > 0).mean()) for k in range(3)]
        r["per_shot"]["tercile_median_skill"] = [float(np.median(ps[tl == k, 0])) for k in range(3)]
        out[tgt] = r
    RES[tag] = out
    return out


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:+.3f}"


def print_a():
    for tag, out in RES.items():
        if not tag.startswith(("cut", "incl")):
            continue
        for tgt in ("CES_TI", "CES_VT"):
            r = out[tgt]
            print(f"\n[{tag} {tgt}] n={r['n']} skill_vs_pchip={fmt(r['skill_vs_pchip'])} "
                  f"vs_pers={fmt(r['skill_vs_pers'])} gpcausal_vs_pchip={fmt(r['skill_gpcausal_vs_pchip'])}")
            p = r["peak"]
            print(f"  peak rows frac={p['peak_frac_rows']:.3f}  "
                  f"nonpeak: n={p['nonpeak']['n']} f_pchip={p['nonpeak']['f_pchip']:.3f} skill={fmt(p['nonpeak']['skill'])} contrib={fmt(p['nonpeak']['contrib'])} | "
                  f"peak: n={p['peak']['n']} f_pchip={p['peak']['f_pchip']:.3f} f_model={p['peak']['f_model']:.3f} skill={fmt(p['peak']['skill'])} contrib={fmt(p['peak']['contrib'])} | "
                  f"identity err={p['identity_err']:.2e}")
            print(f"  causal GP vs PCHIP: nonpeak {fmt(p['gpcausal_skill_nonpeak'])}  peak {fmt(p['gpcausal_skill_peak'])}")
            j = r["jump"]
            print(f"  backward-jump quantiles q50/q90/q95/q99 = " + "/".join(f"{q:.1f}" for q in j["quantiles"]))
            for nm, st in j["strata"].items():
                print(f"    {nm:>7}: n={st['n']:6d} f_pchip={st['f_pchip']:.3f} f_model={st['f_model']:.3f} "
                      f"skill={fmt(st['skill'])} contrib={fmt(st['contrib'])}  gpcausal={fmt(j['gpcausal_skill'][nm])}")
            print(f"    identity err={j['identity_err']:.2e}; top5: sqvar share={j['sq_var_share_top5']:.3f} "
                  f"f_I={j['f_pchip_top5']:.3f} r_I={fmt(j['skill_top5'])} bottom95 skill={fmt(j['skill_bottom95'])} "
                  f"(pers vs pchip bottom95 {fmt(j['skill_pers_vs_pchip_bottom95'])}) formula={fmt(j['formula_top5'])}")
            s = r["per_shot"]
            print(f"  per-shot (n={s['n_shots']}): rho(skill,f_peak)={s['spearman_skill_vs_fpeak']:+.3f} (p={s['p_fpeak']:.1e}) "
                  f"rho(skill,f_top5)={s['spearman_skill_vs_ftop5']:+.3f} (p={s['p_ftop5']:.1e}) "
                  f"rho(skill,varshare)={s['spearman_skill_vs_varshare']:+.3f} (p={s['p_varshare']:.1e}) win={s['win_rate']:.2f}")
            print(f"    terciles of transient share: winrate={['%.2f' % w for w in s['tercile_winrate']]} "
                  f"median skill={['%+.3f' % w for w in s['tercile_median_skill']]}")


# ---------------------------------------------------------------- Part B
GAP = 0.5
CUT = 3000.0


def stuck_mask(values, block):
    stuck = np.zeros(values.shape, dtype=bool)
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
    a = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


def part_b():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "s*.csv")))
    dT, dE, dB, dTe_lvl = [], [], [], []
    n_files = 0
    n_rows_total = 0
    n_held = 0
    n_cut = 0
    for f in files:
        df = pd.read_csv(f)
        t = df["time"].to_numpy(dtype=float)
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP)))
        ti = df["CES_TI"].to_numpy(dtype=float).copy()
        n_cut += int(np.sum(ti > CUT))
        ti[ti > CUT] = np.nan
        sm = stuck_mask(ti, block)
        n_held += int(sm.sum())
        ti[sm] = np.nan
        ecei_cols = [c for c in df.columns if c.startswith("ECEI_")]
        bes_cols = [c for c in df.columns if c.startswith("BES_")]
        E = modality_mean(df, ecei_cols)
        Bm = modality_mean(df, bes_cols)
        ok = np.isfinite(ti)
        # consecutive rows (i-1, i), both observed, same block, grid step (dt < 0.015 s)
        pair = ok[1:] & ok[:-1] & (block[1:] == block[:-1]) & (np.diff(t) < 0.015)
        pair &= np.isfinite(E[1:]) & np.isfinite(E[:-1]) & np.isfinite(Bm[1:]) & np.isfinite(Bm[:-1])
        idx = np.flatnonzero(pair) + 1
        if idx.size == 0:
            continue
        n_files += 1
        n_rows_total += len(df)
        dT.append(ti[idx] - ti[idx - 1])
        dE.append(E[idx] - E[idx - 1])
        dB.append(Bm[idx] - Bm[idx - 1])
    dT = np.concatenate(dT); dE = np.concatenate(dE); dB = np.concatenate(dB)
    a = np.abs(dT)
    q95 = float(np.quantile(a, 0.95)); q99 = float(np.quantile(a, 0.99)); q50 = float(np.quantile(a, 0.5))
    res = {"n_files": n_files, "n_pairs": int(len(dT)), "n_cut_rows": n_cut, "n_held_rows": n_held,
           "abs_dT_q50": q50, "abs_dT_q95": q95, "abs_dT_q99": q99,
           "sqvar_share_top5": float((dT[a >= q95] ** 2).sum() / (dT ** 2).sum()),
           "sqvar_share_top1": float((dT[a >= q99] ** 2).sum() / (dT ** 2).sum())}

    def corr_block(sel, name):
        rE, pE = pearsonr(dT[sel], dE[sel]); rB, pB = pearsonr(dT[sel], dB[sel])
        # magnitude coupling
        rEa = pearsonr(np.abs(dT[sel]), np.abs(dE[sel]))[0]; rBa = pearsonr(np.abs(dT[sel]), np.abs(dB[sel]))[0]
        res[name] = {"n": int(sel.sum()), "r_dT_dECEI": float(rE), "p_E": float(pE),
                     "r_dT_dBES": float(rB), "p_B": float(pB),
                     "r_abs_dT_abs_dECEI": float(rEa), "r_abs_dT_abs_dBES": float(rBa),
                     "sign_agree_E": float((np.sign(dT[sel]) == np.sign(dE[sel])).mean()),
                     "sign_agree_B": float((np.sign(dT[sel]) == np.sign(dB[sel])).mean())}

    corr_block(np.ones_like(a, dtype=bool), "all")
    corr_block(a < q50, "below_q50")
    corr_block(a >= q95, "above_q95")
    corr_block(a >= q99, "above_q99")
    # negative-only tail (crash) and positive-only (recovery)
    corr_block((a >= q95) & (dT < 0), "above_q95_neg")
    corr_block((a >= q95) & (dT > 0), "above_q95_pos")
    # bootstrap CI for the two headline numbers (pair-level, 2000 resamples)
    rng = np.random.default_rng(0)
    for name, sel in (("all", np.ones_like(a, dtype=bool)), ("above_q95", a >= q95)):
        ii = np.flatnonzero(sel)
        bs = []
        for _ in range(2000):
            jj = rng.choice(ii, size=len(ii), replace=True)
            bs.append((np.corrcoef(dT[jj], dE[jj])[0, 1], np.corrcoef(dT[jj], dB[jj])[0, 1]))
        bs = np.array(bs)
        res[name]["ci95_E"] = [float(x) for x in np.quantile(bs[:, 0], [0.025, 0.975])]
        res[name]["ci95_B"] = [float(x) for x in np.quantile(bs[:, 1], [0.025, 0.975])]
    RES["raw_increments"] = res
    print(f"\n[Part B] files={n_files} pairs={len(dT)} cut_rows={n_cut} held_rows={n_held} "
          f"|dT| q50/q95/q99 = {q50:.1f}/{q95:.1f}/{q99:.1f} eV; sq-var share top5%={res['sqvar_share_top5']:.3f} top1%={res['sqvar_share_top1']:.3f}")
    for k in ("all", "below_q50", "above_q95", "above_q99", "above_q95_neg", "above_q95_pos"):
        r = res[k]
        ci = ""
        if "ci95_E" in r:
            ci = f" CI_E=[{r['ci95_E'][0]:+.3f},{r['ci95_E'][1]:+.3f}] CI_B=[{r['ci95_B'][0]:+.3f},{r['ci95_B'][1]:+.3f}]"
        print(f"  {k:>14}: n={r['n']:7d} r(dT,dECEI)={r['r_dT_dECEI']:+.3f} r(dT,dBES)={r['r_dT_dBES']:+.3f} "
              f"|.| r_E={r['r_abs_dT_abs_dECEI']:+.3f} r_B={r['r_abs_dT_abs_dBES']:+.3f} "
              f"sign-agree E={r['sign_agree_E']:.2f} B={r['sign_agree_B']:.2f}{ci}")


if __name__ == "__main__":
    for s in SEEDS:
        part_a(f"cut_s{s}", os.path.join(ROOT, "data", f".b1_seqv2_s{s}_i{s}", "comparison_errors_test.npz"))
    for s in SEEDS:
        p = os.path.join(ROOT, "data", f".b5i_seqv2_s{s}", "comparison_errors_test.npz")
        if os.path.exists(p):
            part_a(f"incl_s{s}", p)
    print_a()
    # 4-split summary for T_i (cut)
    print("\n[cut T_i 4-split summary] seed | skill | f_peak | s_peak | s_nonpeak | f_I(top5 jump) | r_I | s_bottom95 | identity")
    for s in SEEDS:
        r = RES[f"cut_s{s}"]["CES_TI"]
        p = r["peak"]; j = r["jump"]
        print(f"  s{s:<4} {fmt(r['skill_vs_pchip'])} {p['peak']['f_pchip']:.3f} {fmt(p['peak']['skill'])} {fmt(p['nonpeak']['skill'])} "
              f"{j['f_pchip_top5']:.3f} {fmt(j['skill_top5'])} {fmt(j['skill_bottom95'])} {fmt(j['formula_top5'])}")
    part_b()
    with open(os.path.join(OUT, "p01_results.json"), "w") as fh:
        json.dump(RES, fh, indent=1)
    print("\nwrote", os.path.join(OUT, "p01_results.json"))
