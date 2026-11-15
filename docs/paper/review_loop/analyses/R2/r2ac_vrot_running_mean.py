"""R2-a and R2-c (REGISTRATION.md): V_rot running-mean arm under the A1 rule F(1000), paired
against the backbone; Table 1's V_rot RMSE column (seed 42) under the rule.

Run from the repo root:  py docs/paper/review_loop/analyses/R2/r2ac_vrot_running_mean.py
"""
import json
import time
from pathlib import Path

import numpy as np

import r2_common as C
import analyze_b27c as A  # registered two-level reader (B = 20,000, seed 2728)
import baselines_interpolation as BI

T_RULE = 1000.0
ALPHAS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
PRIMARY = tuple(range(2, 10))
ARMS_NPZ = ("model", "persistence", "pchip", "gp_causal", "gp", "linear")


def file_tables(fa, tc, vc):
    """Per observed V_rot row of one file: amax (A1 anchors), persistence anchor, has_past,
    EWMA state before the row for every alpha, ungated and gated (|v| > T_RULE never enters)."""
    t = fa[:, tc].astype(np.float64)
    v = fa[:, vc].astype(np.float64)
    n = len(t)
    seg = np.concatenate(([0], np.cumsum(~(np.diff(t) < C.GAP)))) if n > 1 else np.zeros(n, int)
    amax = np.full(n, np.nan)
    pers = np.full(n, np.nan)
    eu = np.full((n, len(ALPHAS)), np.nan)
    eg = np.full((n, len(ALPHAS)), np.nan)
    obs_all = np.flatnonzero(~np.isnan(v))
    for s in np.unique(seg[obs_all]) if len(obs_all) else []:
        o = obs_all[seg[obs_all] == s]
        vv = v[o]
        mu = None
        mg = None
        for k, r in enumerate(o):
            lo, hi = max(0, k - 2), min(len(o), k + 3)
            amax[r] = np.max(np.abs(vv[lo:hi]))
            if k >= 1:
                pers[r] = vv[k - 1]
                eu[r] = mu
                if mg is not None:
                    eg[r] = mg
            x = vv[k]
            mu = np.full(len(ALPHAS), x) if mu is None else mu + ALPHAS * (x - mu)
            if abs(x) <= T_RULE:
                mg = np.full(len(ALPHAS), x) if mg is None else mg + ALPHAS * (x - mg)
    return {"amax": amax, "pers": pers, "eu": eu, "eg": eg, "v": v}


def val_alpha(tabs, names, name_to_idx):
    """alpha* (ungated, gated) on VAL files: observed rows with a past observation, F(1000)."""
    su = np.zeros(len(ALPHAS))
    sg = np.zeros(len(ALPHAS))
    n = 0
    for nm in names:
        i = name_to_idx.get(nm)
        if i is None:
            continue
        tb = tabs[i]
        k = (~np.isnan(tb["pers"])) & (tb["amax"] <= T_RULE)
        if not k.any():
            continue
        y = tb["v"][k]
        su += ((tb["eu"][k] - y[:, None]) ** 2).sum(0)
        sg += ((tb["eg"][k] - y[:, None]) ** 2).sum(0)
        n += int(k.sum())
    return int(np.argmin(su)), int(np.argmin(sg)), n, (su / n).tolist(), (sg / n).tolist()


def attach(d, tabs, ju, jg, tag, check):
    n = len(d["shot"])
    amax = np.empty(n)
    eu = np.empty(n)
    eg = np.empty(n)
    pers = np.empty(n)
    for i in range(n):
        tb = tabs[int(d["shot"][i])]
        r = int(d["row"][i])
        amax[i] = tb["amax"][r]
        eu[i] = tb["eu"][r, ju]
        eg[i] = tb["eg"][r, jg]
        pers[i] = tb["pers"][r]
        if abs(tb["v"][r] - d["y"][i]) > 1e-3 * max(1.0, abs(d["y"][i])):
            raise SystemExit(f"FATAL {tag}: target mismatch shot {d['shot'][i]} row {r}")
    bad = ~np.isclose((pers - d["y"]) ** 2, d["persistence"], rtol=1e-3, atol=1e-2)
    check["rows"] += n
    check["persistence_mismatch"] += int(bad.sum())
    d["amax"] = amax
    d["ewma_u"] = (eu - d["y"]) ** 2
    d["ewma_g"] = (eg - d["y"]) ** 2
    return d


def load_fold(pop, r, k):
    z = np.load(C.DATA / f".b27_seqv2_{pop}_r{r}_f{k}" / "comparison_errors_test.npz")
    d = {"shot": z["CES_VT_shot"], "row": z["CES_VT_row"], "y": z["CES_VT_y_true"].astype(np.float64)}
    for a in ARMS_NPZ:
        d[a] = z[f"CES_VT_se_{a}"].astype(np.float64)
    return d


def hier(P, keeps, a, b):
    sums = {}
    for r in PRIMARY:
        k = keeps[r]
        sums[r] = tuple(C.shot_sums(P[r]["shot"][k], P[r][a][k], P[r][b][k]))
    h = A.hier(sums, PRIMARY, np.random.default_rng(A.SEED))
    return {"median": h["median"], "ci95": h["ci95"],
            "excludes_0": bool(h["ci95"][0] > 0 or h["ci95"][1] < 0),
            "per_repeat_pooled": {f"r{r}": float(1 - sums[r][0].sum() / sums[r][1].sum()) for r in PRIMARY}}


def hfmt(h):
    return f"{h['median']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]{'*' if h['excludes_0'] else ' '}"


def main():
    t0 = time.time()
    out = {"rule": "A1 F(1000): drop row in every arm if |V_rot| > 1000 km/s at the target, the persistence "
                   "anchor or up to 2 observed neighbours on each side (held-free series, 0.5 s segments)",
           "alphas": ALPHAS.tolist(),
           "two_level": "analyze_b27c.hier, repeats r2-r9, B=20000, seed 2728",
           "per_split": "shot-cluster bootstrap B=10000 seed 12345", "pops": {}, "table1_vrot_seed42": {}}
    for pop in C.POPS:
        ds = C.build_dataset(pop)
        tc = int(ds._column_slices["time"])
        vc = int(ds._column_slices["target"][1])
        tabs = [file_tables(fa, tc, vc) for fa in ds.file_arrays]
        name_to_idx = {Path(p).name: i for i, p in enumerate(ds.valid_files)}
        check = {"rows": 0, "persistence_mismatch": 0}
        res = {"folds_alpha": {}, "canonical": {}}
        print(f"[{pop}] tables built ({time.time() - t0:.0f} s)", flush=True)

        # ---- (1) two-level over the B.27c repeats
        P = {}
        for r in PRIMARY:
            parts = []
            for k in range(5):
                man = json.loads((C.DATA / f".b27_split_r{r}_f{k}" / "split_manifest.json").read_text(encoding="utf-8"))
                ju, jg, nv, _, _ = val_alpha(tabs, man["val_files"], name_to_idx)
                res["folds_alpha"][f"r{r}_f{k}"] = {"alpha_ungated": float(ALPHAS[ju]), "alpha_gated": float(ALPHAS[jg]),
                                                    "n_val_rows": nv}
                parts.append(attach(load_fold(pop, r, k), tabs, ju, jg, f"{pop} r{r} f{k}", check))
            P[r] = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
        keeps = {r: P[r]["amax"] <= T_RULE for r in PRIMARY}
        nd = [int((~keeps[r]).sum()) for r in PRIMARY]
        tl = {"rows_dropped_per_repeat": nd}
        for a, b in (("model", "ewma_g"), ("model", "ewma_u"),
                     ("ewma_g", "persistence"), ("ewma_g", "pchip"), ("ewma_g", "gp_causal"), ("ewma_g", "gp"),
                     ("ewma_u", "persistence"), ("ewma_g", "ewma_u"),
                     ("model", "persistence"), ("model", "pchip"), ("model", "gp_causal")):
            tl[f"{a}_vs_{b}"] = hier(P, keeps, a, b)
        allrows = {r: np.ones(len(P[r]["shot"]), bool) for r in PRIMARY}
        tl["norule_model_vs_ewma_u"] = hier(P, allrows, "model", "ewma_u")
        tl["norule_model_vs_pchip_check"] = hier(P, allrows, "model", "pchip")
        res["two_level"] = tl
        ag = [v["alpha_gated"] for v in res["folds_alpha"].values()]
        au = [v["alpha_ungated"] for v in res["folds_alpha"].values()]
        print(f"[{pop}] two-level F(1000), dropped/repeat {min(nd)}-{max(nd)}; alpha* gated {min(ag)}-{max(ag)}, "
              f"ungated {min(au)}-{max(au)}")
        for kname in ("model_vs_ewma_g", "model_vs_ewma_u", "ewma_g_vs_persistence", "ewma_g_vs_pchip",
                      "ewma_g_vs_gp_causal", "ewma_g_vs_gp", "model_vs_persistence", "model_vs_pchip",
                      "model_vs_gp_causal", "norule_model_vs_ewma_u", "norule_model_vs_pchip_check"):
            print(f"    {kname:30s} {hfmt(tl[kname])}")

        # ---- (2) canonical splits
        for s in C.SEEDS:
            man = C.manifest(pop, s)
            ju, jg, nv, _, _ = val_alpha(tabs, man["val_files"], name_to_idx)
            d = attach(C.canonical(pop, s, "CES_VT"), tabs, ju, jg, f"{pop} s{s}", check)
            k = d["amax"] <= T_RULE
            cell = {"alpha_ungated": float(ALPHAS[ju]), "alpha_gated": float(ALPHAS[jg]), "n_val_rows": nv,
                    "n_rows": int(len(k)), "n_dropped": int((~k).sum())}
            for a, b in (("model", "ewma_g"), ("model", "ewma_u"), ("ewma_g", "persistence"),
                         ("ewma_g", "pchip"), ("ewma_g", "gp_causal"), ("model", "gp_causal"),
                         ("model", "persistence"), ("model", "pchip")):
                cell[f"{a}_vs_{b}"] = C.skill(d["shot"][k], d[a][k], d[b][k])
            res["canonical"][f"s{s}"] = cell
            print(f"[{pop}] s{s}: alpha* g {cell['alpha_gated']} u {cell['alpha_ungated']} dropped {cell['n_dropped']} | "
                  f"bb/ewma_g {C.fmt(cell['model_vs_ewma_g'])} bb/ewma_u {C.fmt(cell['model_vs_ewma_u'])} "
                  f"ewma_g/pers {C.fmt(cell['ewma_g_vs_persistence'])} ewma_g/gpc {C.fmt(cell['ewma_g_vs_gp_causal'])}")

            # ---- R2-c: Table 1 V_rot column, seed 42
            if s == 42:
                arms = {a: d[a] for a in ARMS_NPZ}
                arms["surrogate_file_vt_branch"] = d["surrogate"]
                arms["running_mean_gated"] = d["ewma_g"]
                arms["running_mean_ungated"] = d["ewma_u"]
                ar = np.empty(len(k))
                for i in range(len(k)):
                    fa = ds.file_arrays[int(d["shot"][i])]
                    tt, vv, t_tgt = BI.build_neighbor_set(fa, tc, vc, int(d["row"][i]))
                    ar[i] = BI.predict_ar_local(tt, vv, t_tgt)
                arms["two_point_extrap"] = (ar - d["y"]) ** 2
                ctl = np.load(C.DATA / C.POPS[pop]["control"].format(s=42) / "comparison_errors_test.npz")
                if len(ctl["CES_VT_shot"]) == len(k) and np.array_equal(ctl["CES_VT_se_pchip"],
                                                                         np.load(C.DATA / C.POPS[pop]["run"].format(s=42) / "comparison_errors_test.npz")["CES_VT_se_pchip"]):
                    arms["windowed_control"] = ctl["CES_VT_se_model"].astype(np.float64)
                tab = {a: {"rmse_no_rule": float(np.sqrt(v.mean())), "rmse_F1000": float(np.sqrt(v[k].mean())),
                           "sse_share_dropped": float(v[~k].sum() / v.sum())} for a, v in arms.items()}
                out["table1_vrot_seed42"][pop] = {"n_rows": int(len(k)), "n_dropped": int((~k).sum()),
                                                  "alpha_gated": cell["alpha_gated"],
                                                  "alpha_ungated": cell["alpha_ungated"], "arms": tab}
                print(f"[{pop}] Table 1 V_rot RMSE km/s (seed 42): no rule -> F(1000), dropped {int((~k).sum())}")
                for a, v in sorted(tab.items(), key=lambda kv: kv[1]["rmse_F1000"]):
                    print(f"    {a:26s} {v['rmse_no_rule']:7.2f} -> {v['rmse_F1000']:7.2f}  (dropped rows' SSE share {v['sse_share_dropped']:.3f})")
        res["anchor_check"] = check
        print(f"[{pop}] anchor check: {check}")
        out["pops"][pop] = res
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (C.HERE / "r2ac_vrot_running_mean.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote r2ac_vrot_running_mean.json ({out['minutes']} min)")


if __name__ == "__main__":
    main()
