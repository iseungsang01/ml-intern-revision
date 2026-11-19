"""B.73 checks (registration section 10, Addendum 1 section 10).

    py ces_prediction/experiments/b73_targets/stage_checks.py repro     # (a) (b) (d) (e) (f)
    py ces_prediction/experiments/b73_targets/stage_checks.py leakage   # (c)
    py ces_prediction/experiments/b73_targets/stage_checks.py write     # docs/paper/b73_checks.md
Output: data/.b73/checks_repro.json, data/.b73/checks_leakage.json, docs/paper/b73_checks.md
"""
from __future__ import annotations

import json
import sys
import warnings

import common as C
import designs as DS

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
TOL_B70, TOL_REV = 0.001, 0.005
# the reviewer's values (data/.critique_20261120_stats/logs: gbt_lean_r2.log, gbt_variants2_r3.log, gbt_lean_r4.log)
REVIEWER = {2: {"TX/T0": 0.0877, "T0/G(stored)": 0.0971, "TX/c2": 0.0002, "TX/c": 0.0921, "T0/t": 0.0509, "T0/t2": 0.0087},
            3: {"TX/T0": 0.0852, "T0/G(stored)": 0.0992, "TX/c2": -0.0044, "TX/c": 0.0908, "T0/t": 0.0558, "T0/t2": 0.0087},
            4: {"TX/T0": 0.0859, "T0/G(stored)": 0.0999, "TX/c2": 0.0030, "TX/c": 0.0909, "T0/t": 0.0659, "T0/t2": 0.0163}}
TREES = C.CACHE / "trees"


def assemble(R, unit_prefix, learner, target):
    """Cross-fitted estimates over all D1 rows from the per-unit files of one re-partition / assignment."""
    est = np.full(len(R["y"]), np.nan)
    for k in range(C.K):
        z = C.load_npz(TREES / f"{unit_prefix}_k{k}__{learner}__{target}.npz")
        est[z["idx"]] = z["est"]
    if not np.all(np.isfinite(est)):
        raise SystemExit(f"FATAL: {unit_prefix} {learner} {target}: rows without an estimate")
    return est


def repro():
    R = C.rows()
    y = R["y"]
    out = {"a": {}, "b": {}, "d": {}, "e": {}, "f": {}}
    # (a) B.70 D1-1 .. D1-4
    b70 = json.loads((C.PAPER / "b70_d1.json").read_text(encoding="utf-8"))["statements"]
    vals = {}
    for r in C.REPEATS:
        se = {e: (R[f"r{r}_{e}"] - y) ** 2 for e in ("c", "t", "c2", "t2")}
        se["c"], se["t"] = R[f"r{r}_se_c"], R[f"r{r}_se_t"]
        se["gp"] = R[f"r{r}_se_gp"]
        vals[r] = {"D1-1": C.skill(se["c2"], se["gp"]), "D1-2": C.skill(se["c2"], se["t2"]),
                   "D1-3": C.skill(se["c"], se["t"]), "D1-4": C.skill(se["c"], se["gp"])}
    ok_a = True
    for s in ("D1-1", "D1-2", "D1-3", "D1-4"):
        ours = {str(r): vals[r][s] for r in C.REPEATS}
        ref = b70[s]["per_repeat"]
        dev = max(abs(ours[k] - ref[k]) for k in ref)
        devm = abs(np.mean(list(ours.values())) - b70[s]["value"][0])
        ok = dev <= TOL_B70 and devm <= TOL_B70
        ok_a &= ok
        out["a"][s] = {"ours": ours, "mean_ours": float(np.mean(list(ours.values()))), "b70": ref, "b70_mean": b70[s]["value"][0],
                       "max_dev": dev, "mean_dev": devm, "passes": ok}
    out["a"]["passes"] = bool(ok_a)
    # (b) the reviewer's network-free values
    ok_b = True
    for r in C.REPEATS:
        t0 = assemble(R, f"V1_r{r}", "T0", "F")
        tx = assemble(R, f"V1_r{r}", "TX", "F")
        se = {"T0": (t0 - y) ** 2, "TX": (tx - y) ** 2, "G": R[f"r{r}_se_gp"],
              **{e: (R[f"r{r}_{e}"] - y) ** 2 for e in ("c", "t", "c2", "t2")}}
        ours = {"TX/T0": C.skill(se["TX"], se["T0"]), "T0/G(stored)": C.skill(se["T0"], se["G"]),
                "TX/c2": C.skill(se["TX"], se["c2"]), "TX/c": C.skill(se["TX"], se["c"]),
                "T0/t": C.skill(se["T0"], se["t"]), "T0/t2": C.skill(se["T0"], se["t2"])}
        dev = {k: ours[k] - REVIEWER[r][k] for k in ours}
        ok = all(abs(v) <= TOL_REV for v in dev.values())
        ok_b &= ok
        out["b"][f"r{r}"] = {"ours": ours, "reviewer": REVIEWER[r], "dev": dev, "passes": ok,
                            "median_abs_eV": {"TX": float(np.median(np.sqrt(se["TX"]))), "T0": float(np.median(np.sqrt(se["T0"])))}}
    out["b"]["passes"] = bool(ok_b)
    # (d) folds
    us, tab, fifth = DS.units(R)
    shot_of = dict(zip(tab["fidx"].tolist(), tab["shot"].tolist()))
    clus_of = dict(zip(tab["fidx"].tolist(), tab["clus"].tolist()))
    v2_ok = True
    for u in us:
        if u["design"] == "V2":
            ct = {clus_of[int(i)] for i in u["test"]}
            cf = {clus_of[int(i)] for i in u["fit"]}
            v2_ok &= not (ct & cf)
    time_ok, detail = True, {}
    for u in us:
        if u["design"] in ("V3", "V4"):
            mf = max(shot_of[int(i)] for i in u["fit"])
            mt = min(shot_of[int(i)] for i in u["test"])
            detail[u["name"]] = {"last_fit_shot": int(mf), "first_test_shot": int(mt)}
            time_ok &= mf < mt
    disjoint = all(not (set(u["fit"].tolist()) & set(u["test"].tolist())) for u in us)
    cover = {}
    for d, subs in (("V1", C.REPEATS), ("V2", C.V2_SEEDS)):
        for s in subs:
            tests = np.concatenate([u["test"] for u in us if u["design"] == d and u["sub"] == s])
            cover[f"{d}_{s}"] = bool(len(tests) == len(tab["fidx"]) and len(np.unique(tests)) == len(tests))
    out["d"] = {"v2_clusters_whole": bool(v2_ok), "v3_v4_fit_before_test": bool(time_ok), "detail": detail,
                "fit_test_disjoint": bool(disjoint), "every_discharge_tested_once": cover,
                "passes": bool(v2_ok and time_ok and disjoint and all(cover.values()))}
    # (e) the filter code against B.70
    p = C.CACHE / "filters" / "CHECK_r2_k0.json"
    if p.exists():
        e = json.loads(p.read_text(encoding="utf-8"))
        out["e"] = {"KF0_theta_max_abs_dev": e["KF0_theta_max_abs_dev"], "KFX_b_max_abs_dev": e["KFX_b_max_abs_dev"],
                    "ours": e["ours"], "b70": e["b70"],
                    "passes": bool(e["KF0_theta_max_abs_dev"] < 1e-6 and e["KFX_b_max_abs_dev"] < 1e-6)}
    else:
        out["e"] = {"passes": None, "note": "not run yet"}
    # (f) base GP against the comparison files' GP
    dg = np.abs(np.abs(R["gp"] - y) - np.sqrt(R["r2_se_gp"]))
    dh = np.abs(np.abs(R["hold"] - y) - np.sqrt(R["r2_se_hold"]))
    out["f"] = {"gp_share_dev_gt_0.5eV": float(np.mean(dg > 0.5)), "gp_max_dev_eV": float(dg.max()),
                "hold_share_dev_gt_0.5eV": float(np.mean(dh > 0.5)), "hold_max_dev_eV": float(dh.max())}
    C.save_json(C.CACHE / "checks_repro.json", out)
    print(json.dumps({k: (v.get("passes") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
    print(json.dumps(out["b"], indent=1, default=float))
    print(json.dumps({s: out["a"][s]["max_dev"] for s in ("D1-1", "D1-2", "D1-3", "D1-4")}))


# ------------------------------------------------------------------------------------------ (c) leakage
def leakage():
    import stage_trees as ST
    G = C.grid()
    R = C.rows()
    us, _, _ = DS.units(R)
    u = next(x for x in us if x["name"] == "V1_r2_k0")
    rng = C.rng_for("leakage")
    off, tm, seg, fid = G["off"], G["time"], G["seg"], G["fidx"]
    ti0, fast0 = G["ti"], G["fast"]
    ti1, fast1 = ti0.copy(), fast0.copy()
    cut = {}
    for i in u["test"]:
        q = np.flatnonzero(R["fidx"] == i)
        gstar = int(R["gi"][rng.choice(q)])
        cut[int(i)] = gstar
        b = off[i + 1]
        seg_rows = np.arange(gstar, b)
        m = np.isfinite(ti1[seg_rows])
        ti1[seg_rows[m]] *= rng.uniform(0.5, 1.5, m.sum())
        miss = seg_rows[~m]
        give = miss[rng.random(len(miss)) < 0.5]
        ti1[give] = rng.uniform(100.0, 2000.0, len(give))
        fr = np.arange(gstar + 1, b)
        fast1[fr] = (fast1[fr].astype(np.float64) * rng.uniform(0.5, 1.5, (len(fr), fast1.shape[1]))).astype(np.float32)
    test_rows = np.flatnonzero(np.isin(R["fidx"], u["test"]))
    upto = np.array([R["gi"][j] <= cut[int(R["fidx"][j])] for j in test_rows])
    rows_chk = test_rows[upto]
    res = {"unit": u["name"], "test_discharges": int(len(u["test"])), "rows_checked": int(len(rows_chk)),
           "rows_at_cut": int(len(cut))}
    # history inputs (own recomputation from the grid) and targets change only after the cut
    def hist(ti):
        H = np.full((len(test_rows), 8), np.nan)
        for i in u["test"]:
            a, b = off[i], off[i + 1]
            sel = np.flatnonzero(R["fidx"][test_rows] == i)
            h_, a_, th_, m_ = C.file_history(tm[a:b], ti[a:b], seg[a:b], R["grow"][test_rows[sel]])
            H[sel] = np.column_stack([h_, a_, th_, m_])
        return H
    H0, H1 = hist(ti0), hist(ti1)
    same_h = np.array_equal(np.nan_to_num(H0[upto], nan=-9e9), np.nan_to_num(H1[upto], nan=-9e9))
    changed_after = int(np.sum(np.any(np.nan_to_num(H0[~upto], nan=-9e9) != np.nan_to_num(H1[~upto], nan=-9e9), 1)))
    res["history_inputs_unchanged_up_to_cut"] = bool(same_h)
    res["rows_after_cut_whose_history_inputs_changed"] = changed_after
    res["history_recomputed_equals_table"] = bool(np.allclose(np.nan_to_num(H0[:, 0]), np.nan_to_num(R["hold"][test_rows])) and
                                                  np.allclose(np.nan_to_num(H0[:, 1]), np.nan_to_num(R["age"][test_rows])))
    # causal GP at the cut rows
    sys.path.insert(0, str(C.REPO / "ces_prediction"))
    import baselines_interpolation as BI
    dev_gp, dev_base = 0.0, 0.0
    for i, gstar in cut.items():
        a = off[i]
        blk = np.flatnonzero((seg[a:off[i + 1]] == seg[gstar])) + a
        v = []
        for ti in (ti0, ti1):
            obs = blk[np.isfinite(ti[blk])]
            obs = obs[obs < gstar]
            obs = obs[-16:]
            v.append(BI.predict_gp_causal(tm[obs], ti[obs], tm[gstar]))
        dev_gp = max(dev_gp, abs(v[0] - v[1]))
        j = np.flatnonzero(R["gi"] == gstar)[0]
        dev_base = max(dev_base, abs(v[0] - R["gp"][j]))
    res["gp_at_cut_max_change_eV"] = float(dev_gp)
    res["gp_at_cut_recomputed_vs_table_max_dev_eV"] = float(dev_base)
    # trees: the unit's fits redone on the perturbed grid, estimates up to the cut
    H10_0 = ST.history_matrix(R)
    R1 = dict(R)
    th1 = R["thist"].copy()
    hold1 = R["hold"].copy()
    age1 = R["age"].copy()
    th1[test_rows] = H1[:, 2:7]
    hold1[test_rows], age1[test_rows] = H1[:, 0], H1[:, 1]
    R1["thist"], R1["hold"], R1["age"] = th1, hold1, age1
    # G of rows after the cut would change too; only rows up to the cut are compared, where it is unchanged (above)
    H10_1 = ST.history_matrix(R1)
    est = {}
    fit_d = np.isin(R["fidx"], u["fit"])
    for tag, fast, H10 in (("orig", fast0, H10_0), ("pert", fast1, H10_1)):
        FE = ST.Feats(fast)
        wf = FE.window(u["fit"])
        fa, fl = FE.at(wf, R["gi"])
        ok = ST.complete_mask(H10, fa, fl, R["gp"])
        XS = ST.learner_inputs(H10, fa, fl)
        est[tag] = {"XS": XS, "ok": ok, "fa": fa, "fl": fl}
    same_fit_inputs = all(np.array_equal(np.nan_to_num(est["orig"]["XS"][L][fit_d], nan=-9e9),
                                         np.nan_to_num(est["pert"]["XS"][L][fit_d], nan=-9e9)) for L in est["orig"]["XS"])
    res["fit_rows_inputs_identical"] = bool(same_fit_inputs)
    res["fast_scalars_unchanged_up_to_cut"] = bool(
        np.array_equal(np.nan_to_num(est["orig"]["fa"][rows_chk], nan=-9e9), np.nan_to_num(est["pert"]["fa"][rows_chk], nan=-9e9)) and
        np.array_equal(np.nan_to_num(est["orig"]["fl"][rows_chk], nan=-9e9), np.nan_to_num(est["pert"]["fl"][rows_chk], nan=-9e9)))
    tree_dev = {}
    cache_dev = {}
    for L in ("T0", "TX", "TB", "TE", "TXlag"):
        for X in ("F", "L5"):
            tgt = R[f"tg_{X}"]
            ok = est["orig"]["ok"]
            fr = fit_d & ok & np.isfinite(tgt)
            m = ST.gbr().fit(est["orig"]["XS"][L][fr], (tgt - R["gp"])[fr])
            e = {}
            for tag in ("orig", "pert"):
                okt = est[tag]["ok"][rows_chk]
                v = R["gp"][rows_chk].copy()
                v[okt] += m.predict(est[tag]["XS"][L][rows_chk[okt]])
                e[tag] = v
            tree_dev[f"{L}[{X}]"] = float(np.max(np.abs(e["orig"] - e["pert"])))
            pth = TREES / f"{u['name']}__{L}__{X}.npz"
            if pth.exists():
                z = C.load_npz(pth)
                pos = np.searchsorted(z["idx"], rows_chk)
                cache_dev[f"{L}[{X}]"] = float(np.max(np.abs(z["est"][pos] - e["orig"])))
    res["tree_estimates_max_change_up_to_cut_eV"] = tree_dev
    res["tree_estimates_vs_stage_cache_max_dev_eV"] = cache_dev
    # filters: the unit's F parameters, perturbed filter arrays
    import stage_filters as SF
    g = SF.ns_held()
    pj = C.CACHE / "filters" / f"{u['name']}__F.json"
    par = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else \
        json.loads((C.CACHE / "filters" / "CHECK_r2_k0.json").read_text(encoding="utf-8"))["ours"]
    res["filter_parameters_from"] = pj.name if pj.exists() else "CHECK_r2_k0.json"
    run, CLIP = g["run"], g["CLIP"]
    CLIP[0] = par["clip"]
    keep = {k: g[k].copy() for k in ("Y", "LY", "dC", "dG")}
    outs = {}
    for tag in ("orig", "pert"):
        if tag == "pert":
            for i, gstar in cut.items():
                j0 = gstar - off[i] - 1                          # filter row of the cut row
                nk = len(g["grid"][i]["t"])
                jj = np.arange(j0, nk)
                yv = g["Y"][i, jj]
                m = np.isfinite(yv)
                yv[m] *= rng.uniform(0.5, 1.5, m.sum())
                give = (~m) & (rng.random(len(jj)) < 0.5)
                yv[give] = rng.uniform(100.0, 2000.0, give.sum())
                g["Y"][i, jj] = yv
                g["LY"][i, jj] = np.log(np.where(yv > 0, yv, np.nan))
                j1 = np.arange(j0 + 1, nk)
                g["dC"][i, j1] += rng.normal(0, 0.05, len(j1))
                g["dG"][i, j1] += rng.normal(0, 0.05, len(j1))
        o0 = run(np.r_[par["theta_k0"], 0, 0], None, False)
        ox = run(np.r_[par["theta_k0"], par["b_kx"]], None, True)
        f, j = R["fidx"][rows_chk], R["grow"][rows_chk] - 1
        outs[tag] = (o0[f, j], ox[f, j])
    for k_, v in keep.items():
        g[k_][...] = v
    res["filter_K0_max_change_up_to_cut_eV"] = float(np.nanmax(np.abs(outs["orig"][0] - outs["pert"][0])))
    res["filter_KX_max_change_up_to_cut_eV"] = float(np.nanmax(np.abs(outs["orig"][1] - outs["pert"][1])))
    res["passes"] = bool(same_h and res["fast_scalars_unchanged_up_to_cut"] and same_fit_inputs and dev_gp == 0.0
                         and max(tree_dev.values()) == 0.0 and res["filter_K0_max_change_up_to_cut_eV"] == 0.0
                         and res["filter_KX_max_change_up_to_cut_eV"] == 0.0)
    C.save_json(C.CACHE / "checks_leakage.json", res)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------ md
def fmt(x, d=4):
    return f"{x:+.{d}f}"


def write():
    a = json.loads((C.CACHE / "checks_repro.json").read_text(encoding="utf-8"))
    pl = C.CACHE / "checks_leakage.json"
    lk = json.loads(pl.read_text(encoding="utf-8")) if pl.exists() else None
    info = json.loads((C.CACHE / "rows_info.json").read_text(encoding="utf-8"))
    L = ["# B.73 checks", "",
         f"Registration: `{C.REG}` section 10, with Addendum 2 section 10 (implementation choices; Addendum 1 is the session lead's). Run before any readout; the batch continues only if (a) and (b) pass.", ""]
    L += ["## (a) Stored estimates against B.70 (V1, target F; tolerance 0.001)", "",
          "| statement | r2 | r3 | r4 | mean | B.70 mean | largest deviation | passes |", "|---|---|---|---|---|---|---|---|"]
    for s in ("D1-1", "D1-2", "D1-3", "D1-4"):
        v = a["a"][s]
        L.append(f"| {s} | {fmt(v['ours']['2'])} | {fmt(v['ours']['3'])} | {fmt(v['ours']['4'])} | {fmt(v['mean_ours'])} | "
                 f"{fmt(v['b70_mean'])} | {max(v['max_dev'], v['mean_dev']):.1e} | {bool(v['passes'])} |")
    L += ["", f"(a) passes: **{a['a']['passes']}**", ""]
    L += ["## (b) T0 and TX against the reviewer's values (V1, target F; tolerance 0.005)", "",
          "Reviewer's values: `data/.critique_20261120_stats/logs/` (gbt_lean r2, gbt_variants2 r3, gbt_lean r4), the C7 rows of "
          "`docs/paper/review_loop/critique_20261120_stats.md`. G(stored) = the causal GP of the comparison files, as the reviewer used.", "",
          "| comparison | r2 ours / reviewer | r3 ours / reviewer | r4 ours / reviewer | largest deviation |", "|---|---|---|---|---|"]
    for c in REVIEWER[2]:
        cells = [f"{fmt(a['b'][f'r{r}']['ours'][c])} / {fmt(REVIEWER[r][c])}" for r in C.REPEATS]
        dev = max(abs(a["b"][f"r{r}"]["dev"][c]) for r in C.REPEATS)
        L.append(f"| {c} | " + " | ".join(cells) + f" | {dev:.4f} |")
    L += ["", "Median absolute error (eV), TX / T0: " + "; ".join(
        f"r{r} {a['b'][f'r{r}']['median_abs_eV']['TX']:.1f} / {a['b'][f'r{r}']['median_abs_eV']['T0']:.1f}" for r in C.REPEATS)
        + " (reviewer: 39.8 / 41.8, 39.8 / 41.8, 39.7 / 41.7).", "", f"(b) passes: **{a['b']['passes']}**", ""]
    L += ["## (c) No input of row t contains a value of row t or later", ""]
    if lk:
        L += [f"Unit {lk['unit']}: {lk['test_discharges']} test discharges, one cut row each, {lk['rows_checked']:,} D1 rows up to the cuts compared. "
              "T_i values from the cut on multiplied by factors in [0.5, 1.5], half of the missing ones from the cut on given a value; fast channels after the cut multiplied by factors in [0.5, 1.5] (filter: its BES drive perturbed after the cut).", "",
              "| quantity | result |", "|---|---|",
              f"| history inputs (hold, age, five scalars, M5) at rows up to the cut | unchanged: {lk['history_inputs_unchanged_up_to_cut']} (rows after the cut changed: {lk['rows_after_cut_whose_history_inputs_changed']:,}) |",
              f"| recomputed hold and age equal the row table | {lk['history_recomputed_equals_table']} |",
              f"| causal GP at the cut rows | largest change {lk['gp_at_cut_max_change_eV']:.2e} eV; recomputed vs table {lk['gp_at_cut_recomputed_vs_table_max_dev_eV']:.2e} eV |",
              f"| eight fast scalars of row t and of row t-1 | unchanged: {lk['fast_scalars_unchanged_up_to_cut']} |",
              f"| inputs of the fit rows | identical: {lk['fit_rows_inputs_identical']} |",
              "| tree estimates up to the cut (fits redone), largest change (eV) | " + ", ".join(f"{k} {v:.1e}" for k, v in lk["tree_estimates_max_change_up_to_cut_eV"].items()) + " |",
              "| the same against the stage's cached estimates (eV) | " + (", ".join(f"{k} {v:.1e}" for k, v in lk["tree_estimates_vs_stage_cache_max_dev_eV"].items()) or "-") + " |",
              f"| filters K0, KX up to the cut ({lk['filter_parameters_from']}) | largest change {lk['filter_K0_max_change_up_to_cut_eV']:.1e}, {lk['filter_KX_max_change_up_to_cut_eV']:.1e} eV |",
              "", f"(c) passes: **{lk['passes']}**", ""]
    else:
        L += ["not run", ""]
    o1p, o2p = (C.CACHE / s / "checks_leakage.json" for s in ("superseded_global_cumsum", "superseded_perfile_sumsq"))
    if o1p.exists() and o2p.exists():
        o1, o2 = (json.loads(p.read_text(encoding="utf-8")) for p in (o1p, o2p))
        r1, r2 = (json.loads((p.parent / "checks_repro.json").read_text(encoding="utf-8")) for p in (o1p, o2p))
        tx = lambda r: ", ".join(f"{r['b'][f'r{k}']['ours']['TX/T0']:+.4f}" for k in C.REPEATS)   # noqa: E731
        L += ["**Two earlier runs of (c), superseded before any readout.** Both failed on bit-identity of the fast scalars, "
              "not on causality. In both, the history inputs, the causal GP and the filters were unchanged up to the cut.", "",
              "| run | implementation of the fast scalars | what (c) found | (b) TX over T0, r2 / r3 / r4 |", "|---|---|---|---|",
              "| 1 | `window_feats` once over all files laid end to end (running sums crossing file boundaries); channel "
              "statistics from per-file sums and sums of squares | fast scalars of rows up to the cut and of the fit rows not "
              "bit-identical after perturbing later rows of other files (rounding, at most about 1e-8); tree estimates up to the cut "
              f"unchanged (largest change {max(o1['tree_estimates_max_change_up_to_cut_eV'].values()):.1e} eV) | {tx(r1)} |",
              "| 2 | `window_feats` per file; statistics as in run 1 | the standard deviations differed in the last digits "
              "between the two processes (sum of squares minus squared mean, numpy reductions whose rounding depends on memory "
              "alignment), so the fast scalars of fit rows differed by up to 1.7e-7; one tree estimate up to the cut moved by "
              f"{max(o2['tree_estimates_max_change_up_to_cut_eV'].values()):.1f} eV when a value crossed a bin edge | {tx(r2)} |",
              "| 3 (above) | `window_feats` per file; mean and sd by two passes of exactly rounded sums (`math.fsum`); array "
              "means by explicit sequential sums | see the table above | see (b) |", "",
              "Every tree estimator and classifier was refitted after each change. The readout uses run 3 only. "
              "Run 1 to run 2 also shows how sensitive the boosted trees are to rounding-level changes of their inputs: "
              "TX over T0 moved by up to 0.002 on a re-partition.", ""]
    d = a["d"]
    L += ["## (d) Folds", "", f"- V2: every cluster whole in each fold: {d['v2_clusters_whole']}",
          f"- V3, V4: every fit discharge earlier than the first test discharge: {d['v3_v4_fit_before_test']} (" +
          "; ".join(f"{k}: last fit shot {v['last_fit_shot']}, first test shot {v['first_test_shot']}" for k, v in d["detail"].items()) + ")",
          f"- fit and test discharges disjoint in every unit: {d['fit_test_disjoint']}; V1 and V2: every D1 discharge tested exactly once per re-partition / assignment: {d['every_discharge_tested_once']}",
          "", f"(d) passes: **{d['passes']}**", ""]
    e = a["e"]
    L += ["## (e) The filter code against B.70", ""]
    if e.get("passes") is not None:
        L += [f"F fit on the manifest training files of r2 f0 against `data/.b70/kf_params_r2_f0.json`: largest deviation of the three K0 parameters {e['KF0_theta_max_abs_dev']:.1e}, of the two gains {e['KFX_b_max_abs_dev']:.1e}.",
              "", f"(e) passes: **{e['passes']}**", ""]
    else:
        L += ["not run yet", ""]
    f = a["f"]
    L += ["## (f) Per-row references", "",
          f"Causal GP of `data/.b64/base.npz` against the comparison files: share of D1 rows apart by more than 0.5 eV {f['gp_share_dev_gt_0.5eV']:.5f} (largest {f['gp_max_dev_eV']:.2f} eV); hold {f['hold_share_dev_gt_0.5eV']:.5f} (largest {f['hold_max_dev_eV']:.2f} eV).", ""]
    L += ["## Counts", "", f"D1 rows {info['d1_rows']:,} of {info['d1_discharges']} discharges; rows with the target defined: " +
          ", ".join(f"{k} {v:,}" for k, v in info["target_rows"].items()) +
          f"; classifier rows {info['clf_rows']:,} ({info['clf_positive_share']:.4f} positive).", ""]
    C.save_text(C.CHECKS_MD, "\n".join(L))
    print("written", C.CHECKS_MD)


if __name__ == "__main__":
    {"repro": repro, "leakage": leakage, "write": write}[sys.argv[1]]()
