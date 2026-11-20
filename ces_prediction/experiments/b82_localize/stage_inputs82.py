"""B.82 stage 1: inputs that are not estimates (Addendum 1 sections 1-3, 6).

    py ces_prediction/experiments/b82_localize/stage_inputs82.py mc       # Mirnov columns joined to the grid -> mc.npz
    py ces_prediction/experiments/b82_localize/stage_inputs82.py strata   # tau of every D1 row; c and its class per unit, rule
    py ces_prediction/experiments/b82_localize/stage_inputs82.py facts    # part E: file facts of the data set
Output: data/.b82/mc.npz, data/.b82/tau.npz, data/.b82/cclass.npz, data/.b82/facts.json, data/.b82/checks/mc_join.json
No target is compared with an estimate here; part E reads the reported values and L5 as data.
"""
from __future__ import annotations

import sys
import time

import common82 as M

import numpy as np  # noqa: E402

C, K75 = M.C, M.K75


def run_mc():
    import pandas as pd
    G = K75.grid_lean(("names", "off", "time"))
    names, off, tm = G["names"], G["off"], G["time"]
    hdr = pd.read_csv(C.DATA / str(names[0]), nrows=0).columns.tolist()
    bes = [c for c in hdr if c.startswith("BES_")]
    mcs = [c for c in hdr if c.startswith("MC")]
    if tuple(bes) != M.BES_NAMES or tuple(mcs) != M.MC_COLS:
        raise SystemExit(f"FATAL: columns {bes} {mcs}")
    mc = np.full((len(tm), 2), np.nan, np.float32)
    bad_len, bad_time, bad_hdr = [], [], []
    for i, nm in enumerate(names):
        h = pd.read_csv(C.DATA / str(nm), nrows=0).columns.tolist()
        if [c for c in h if c.startswith("BES_")] != list(M.BES_NAMES) or [c for c in h if c.startswith("MC")] != list(M.MC_COLS):
            bad_hdr.append(str(nm))
        df = pd.read_csv(C.DATA / str(nm), usecols=["time", *M.MC_COLS])[["time", *M.MC_COLS]]
        a, b = off[i], off[i + 1]
        if len(df) != b - a:
            bad_len.append(str(nm))
            continue
        t32 = df["time"].to_numpy(np.float64).astype(np.float32).astype(np.float64)
        if not np.array_equal(t32, tm[a:b]):
            bad_time.append(str(nm))
        mc[a:b] = df[list(M.MC_COLS)].to_numpy(np.float64).astype(np.float32)
    out = {"files": int(len(names)), "rows": int(len(tm)), "files_other_row_count": bad_len, "files_other_time": bad_time,
           "files_other_bes_or_mc_header": bad_hdr, "nonfinite_mc_values": int((~np.isfinite(mc)).sum()),
           "bes_columns": list(M.BES_NAMES), "mc_columns": list(M.MC_COLS)}
    out["passes"] = bool(not bad_len and not bad_time and not bad_hdr and out["nonfinite_mc_values"] == 0)
    M.save_json(M.CHK / "mc_join.json", out)
    M.save_npz(M.CACHE / "mc.npz", mc=mc)
    M.log(f"mc: {out}")


def run_strata():
    R = K75.rows_lean()
    G = K75.grid_lean(("time",))
    jS = K75.load_inputs("S")["jS_g"]
    tau = M.tau_ms(G["time"], R["gi"], jS)
    M.save_npz(M.CACHE / "tau.npz", tau=tau, cls=M.tau_class(tau))
    ctx = M.B81.Ctx()
    us, _, _ = M.units()
    S = ctx.S
    arr = {}
    for u in us:
        _, ms_b, ms_e = ctx.unit_arrays(u)
        fit_d = np.isin(R["fidx"], u["fit"])
        test_d = np.isin(R["fidx"], u["test"])
        for r in M.RULES:
            I = ctx.inputs(r)
            A = K75.anchor_scalars(R["gi"], I["anc_g"], ms_b, ms_e, ctx.mr_b, ctx.mr_e)
            c = M.c_of(A)
            fr = fit_d & S[M.SETOF[r]]
            cf = c[fr]
            cuts = np.quantile(cf[np.isfinite(cf)], [1.0 / 3.0, 2.0 / 3.0])
            cls = np.full(len(c), -1, np.int8)
            cls[test_d] = M.c_class(c[test_d], cuts).astype(np.int8)
            arr[f"{u['name']}|{r}|cls"] = cls
            arr[f"{u['name']}|{r}|cuts"] = cuts
            arr[f"{u['name']}|{r}|c_test"] = np.where(test_d, c, np.nan)
        M.log(f"strata {u['name']}: done")
    M.save_npz(M.CACHE / "cclass.npz", **arr)


def pct(x, ps=(10, 50, 90)):
    return {str(p): float(np.percentile(x, p)) for p in ps}


def run_facts():
    """Part E (registration section 7)."""
    R = K75.rows_lean()
    G = K75.grid_lean(("off", "time", "ti", "seg", "fidx", "grow", "shot"))
    off, tm, ti, seg, fid, grow = G["off"], G["time"], G["ti"], G["seg"], G["fidx"], G["grow"]
    last = np.concatenate((fid[1:] != fid[:-1], [True]))
    gridrow = (grow > 0) & ~last
    d1 = np.unique(R["fidx"])
    rows_per, absent_per = [], []
    for i in d1:
        a, b = off[i], off[i + 1]
        m = gridrow[a:b]
        rows_per.append(int(m.sum()))
        t_, s_ = tm[a:b][m], seg[a:b][m]
        dt = np.diff(t_)
        absent_per.append(int(((s_[1:] == s_[:-1]) & (dt > K75.AB_LO) & (dt < K75.AB_HI)).sum()))
    rows_per, absent_per = np.array(rows_per), np.array(absent_per)
    ind1 = np.isin(fid, d1) & gridrow
    l5 = R["tg_L5"][np.isfinite(R["tg_L5"])]
    out = {"d1_discharges": int(len(d1)), "grid_rows_of_d1_discharges": int(ind1.sum()),
           "d1_rows": int(len(R["y"])), "d1_rows_with_L5": int(len(l5)),
           "grid_rows_per_discharge": pct(rows_per), "absent_steps_per_discharge": pct(absent_per),
           "absent_steps_total": int(absent_per.sum()), "discharges_without_absent_step": int((absent_per == 0).sum()),
           "reported_Ti_eV_over_d1_rows": pct(R["y"]), "L5_eV_over_d1_rows": pct(l5),
           "share_of_grid_rows_without_usable_Ti": float(np.mean(~np.isfinite(ti[ind1]))),
           "grid_rows_without_usable_Ti": int((~np.isfinite(ti[ind1])).sum())}
    # beam (B.79's joined columns; sources NB1B, NB1C, NB2A, NB2B, NB2C; on above 0.1 MW; empty power = 0)
    sys.path.insert(0, str(M.HERE.parent / "b79_beam_inputs"))
    import common79 as B79
    P = np.load(B79.CACHE / "beam_cols.npz")["P"]
    Pz = np.where(np.isfinite(P), P, 0.0)
    Pt = Pz.sum(1)
    non = (Pz > B79.ON_MW).sum(1)
    inval = np.isin(G["shot"][fid], B79.invalid_files())
    sel = ind1 & (non > 0) & ~inval
    out["beam"] = {"total_power_MW_over_grid_rows_with_a_source_on_outside_13_files": pct(Pt[sel]),
                   "rows_of_that_line": int(sel.sum()),
                   "files_left_out_of_that_line": [int(s) for s in B79.invalid_files()],
                   "d1_discharges_among_them": int(len(np.unique(fid[ind1 & inval]))),
                   "share_of_grid_rows_by_sources_on_all_d1_discharges": {str(k): float(np.mean(non[ind1] == k)) for k in range(6)},
                   "share_of_grid_rows_by_sources_on_outside_13_files": {str(k): float(np.mean(non[ind1 & ~inval] == k))
                                                                         for k in range(6)},
                   "grid_rows_all_d1_discharges": int(ind1.sum()), "grid_rows_outside_13_files": int((ind1 & ~inval).sum())}
    M.save_json(M.CACHE / "facts.json", out)
    M.log("facts written")


if __name__ == "__main__":
    t0 = time.time()
    {"mc": run_mc, "strata": run_strata, "facts": run_facts}[sys.argv[1]]()
    M.log(f"done ({time.time() - t0:.0f}s)")
