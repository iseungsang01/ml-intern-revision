"""MAST gate, addendum 2: post hoc breakdown by campaign and machine configuration.

Registration: ces_prediction/experiments/PREREGISTRATION_MAST_GATE_ADDENDUM2.md.
Nothing of the gate is changed and nothing is retuned: the hyperparameters are read from
docs/paper/mast_gate_addendum1.json, the TEST row tables are rebuilt with run_gate functions,
and the pooled addendum-1 numbers must be reproduced before any stratum is read.
Only gate-list shots are opened.  CPU, single process.

Run from the repo root:  py ces_prediction/experiments/mast/run_gate_addendum2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as rg  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_MAST_GATE_ADDENDUM2.md"
FIRST_JSON = rg.ROOT / "docs" / "paper" / "mast_gate.json"
ADD1_JSON = rg.ROOT / "docs" / "paper" / "mast_gate_addendum1.json"
OUT_JSON = rg.ROOT / "docs" / "paper" / "mast_gate_addendum2.json"
OUT_MD = rg.ROOT / "docs" / "paper" / "mast_gate_addendum2.md"
ARMS3 = ("causal_gp", "slope_filter", "running_mean")
MIN_SHOTS = 8
CAMPS = ("M6", "M7", "M8", "M9")
META_COLS = ["shot_id", "campaign", "heating", "rmp_coil", "plasma_shape", "current_range"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "no record"
    s = str(v)
    return "no record" if s in ("None", "nan", "") else s


def heating_class(h):
    if h == "no record":
        return "no record"
    if h == "2 Beams,SS Beam,SW Beam":
        return "2 beams"
    return h


def has_ss(h):
    return h in ("2 beams", "SS Beam")


def params_from(add1, tg):
    t = add1["retuned"]["tuning"][tg]
    return {"tau_ms": t["running_mean"]["selected_tau_ms"],
            "q_level": t["slope_filter"]["selected_q_level"],
            "q_slope": t["slope_filter"]["selected_q_slope"],
            "gp_causal": (t["gp_causal"]["selected_length_ms"], t["gp_causal"]["selected_amp"]),
            "gp_offline": (t["gp_offline"]["selected_length_ms"], t["gp_offline"]["selected_amp"])}


def same(a, b, tol=1e-9):
    return all(abs(a[k] - b[k]) <= tol * max(1.0, abs(b[k])) for k in ("point", "lo", "hi"))


def strip(e, few):
    """Point only when the stratum is too small for an interval."""
    if e is None or e.get("point") is None:
        return {"point": None, "lo": None, "hi": None, "side": None, "ref": None}
    if few:
        return {"point": e["point"], "lo": None, "hi": None, "side": "too few", "ref": e["ref"]}
    return {k: e[k] for k in ("point", "lo", "hi", "side", "ref")}


def fmt(e, nd=3, signed=True):
    if e is None or e.get("point") is None:
        return "n/a"
    f = "{:" + ("+" if signed else "") + "." + str(nd) + "f}"
    if e["lo"] is None:
        return f"{f.format(e['point'])} (too few)"
    return f"{f.format(e['point'])} [{f.format(e['lo'])}, {f.format(e['hi'])}]"


def side(e):
    if e is None or e.get("point") is None:
        return "n/a"
    if e["lo"] is None:
        return "too few"
    return {"above": f"above {e['ref']:g}", "below": f"below {e['ref']:g}",
            "includes": f"includes {e['ref']:g}"}[e["side"]]


def main():
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    sha_first, sha_add1 = sha256(FIRST_JSON), sha256(ADD1_JSON)
    add1 = json.loads(ADD1_JSON.read_text(encoding="utf-8"))

    shots = rg.read_gate_shots()
    meta = pd.read_parquet(rg.META, columns=META_COLS)
    meta = meta[meta.shot_id.isin(shots)].set_index("shot_id")
    data = {s: d for s in shots if (d := rg.load_shot(s)) is not None}
    have = sorted(data)
    camp = {s: str(meta.loc[s, "campaign"]) for s in have}
    split = {}
    for c in sorted(set(camp.values())):
        for i, s in enumerate(sorted(s for s in have if camp[s] == c)):
            split[s] = "TUNE" if i % 2 == 0 else "TEST"
    usable, med_dt = {}, {}
    for s in have:
        tt = data[s]["time_ms"]
        dd = np.diff(tt[np.isfinite(tt)])
        m = float(np.median(dd)) if len(dd) else float("nan")
        med_dt[s] = m
        usable[s] = bool(np.isfinite(m) and abs(m - rg.STEP_MS) <= 0.1)
    test_shots = [s for s in have if split[s] == "TEST"]
    tpos = {s: i for i, s in enumerate(test_shots)}
    nT = len(test_shots)
    rng = np.random.default_rng(rg.SEED)
    bidx_all = rng.integers(0, nT, size=(rg.N_BOOT, nT))
    counts_all = np.zeros((rg.N_BOOT, nT))
    for b in range(rg.N_BOOT):
        counts_all[b] = np.bincount(bidx_all[b], minlength=nT)
    log(f"gate shots {len(shots)}, with data {len(have)}, TEST {nT}")

    # ---- strata labels per shot
    lab = {}
    for s in have:
        h = heating_class(tag(meta.loc[s, "heating"]))
        ncol = int(data[s]["T"].shape[1])
        sp = "5 ms" if usable[s] else (f"{med_dt[s]:.1f} ms" if np.isfinite(med_dt[s]) else "n/a")
        lab[s] = {"campaign": camp[s], "heating": h, "rmp_coil": tag(meta.loc[s, "rmp_coil"]),
                  "chord_columns": str(ncol), "frame_spacing": sp,
                  "plasma_shape": tag(meta.loc[s, "plasma_shape"]),
                  "current_range": tag(meta.loc[s, "current_range"])}
    dims = ("campaign", "heating", "rmp_coil", "chord_columns", "frame_spacing", "plasma_shape",
            "current_range")

    prep = {(tg, pop): {s: (rg.prepare(data[s], tg, rel) if usable[s] else {"status": "unusable"})
                        for s in have}
            for tg in rg.TARGETS for pop, rel in rg.POPS.items()}

    # ---- counts over all shots with data
    counts_tab = {}
    for dim in dims:
        counts_tab[dim] = {}
        for v in sorted(set(lab[s][dim] for s in have)):
            ss = [s for s in have if lab[s][dim] == v]
            row = {"with_data": len(ss), "usable": int(sum(usable[s] for s in ss)),
                   "TEST": int(sum(split[s] == "TEST" for s in ss))}
            for pop in rg.POPS:
                st = [prep["T_i", pop][s]["status"] for s in ss]
                row[pop] = {k: int(sum(x == k for x in st))
                            for k in ("unusable", "no_block", "no_scored_chord", "ok")}
            counts_tab[dim][v] = row

    res = {"registration": REG, "post_hoc": True, "seed": rg.SEED, "n_boot": rg.N_BOOT,
           "hyperparameters_from": "docs/paper/mast_gate_addendum1.json (no retuning)",
           "sha256": {"mast_gate.json_before": sha_first, "mast_gate_addendum1.json_before": sha_add1},
           "min_shots_for_interval": MIN_SHOTS, "counts": counts_tab, "integrity": {},
           "strata": {}, "statements": {}}

    per_target = {}
    for tg in rg.TARGETS:
        params = params_from(add1, tg)
        acc = {"shot": [], "R": [], "sig2": [], "r2": {a: [] for a in rg.ARMS},
               "den": {a: [] for a in rg.ARMS}}
        reg = np.full(nT, np.nan)
        q_rows, q_shot = [], []
        for s in test_shots:
            p = prep[tg, "G"][s]
            if p["status"] != "ok":
                continue
            G, S2 = p["G"], p["S2"]
            reg[tpos[s]] = rg.regime_number(np.where(G, p["Y0"], np.nan), G)
            d1 = p["Y0"][2:] - 2.0 * p["Y0"][1:-1] + p["Y0"][:-2]
            ok = G[2:] & G[1:-1] & G[:-2]
            var = 4.0 * S2[1:-1] + S2[2:] + S2[:-2]
            q_rows.append((d1[ok] ** 2) / var[ok])
            q_shot.append(np.full(int(ok.sum()), tpos[s]))
            if p["n_rows"] == 0:
                continue
            rows = p["rows"]
            yt = p["Y0"][p["hid"]][rows]
            s2t = p["S2"][p["hid"]][rows]
            acc["shot"].append(np.full(len(yt), tpos[s]))
            acc["R"].append(np.broadcast_to(p["R"][None, :], rows.shape)[rows])
            acc["sig2"].append(s2t)
            for arm in rg.ARMS:
                pred, vw = rg.arm_outputs(p, arm, params, mask=rows)
                acc["r2"][arm].append((pred[rows] - yt) ** 2)
                acc["den"][arm].append(s2t + vw[rows])
        tab = {"shot": np.concatenate(acc["shot"]).astype(np.int64),
               "R": np.concatenate(acc["R"]), "sig2": np.concatenate(acc["sig2"]),
               "r2": {a: np.concatenate(v) for a, v in acc["r2"].items()},
               "den": {a: np.concatenate(v) for a, v in acc["den"].items()}}
        q_rows = np.concatenate(q_rows)
        q_shot = np.concatenate(q_shot).astype(np.int64)
        per_target[tg] = (tab, reg, q_rows, q_shot)
        log(f"{tg}: TEST rows {len(tab['shot'])}")

        # ---- integrity: pooled numbers of addendum 1
        pooled = rg.readouts(tab, counts_all, nT, with_median=False)
        ref = add1["retuned"]["readouts"][f"{tg}|G"]["arms"]
        ok_int = {"n_rows": pooled["n_rows"] == add1["retuned"]["readouts"][f"{tg}|G"]["n_rows"]}
        for arm in rg.ARMS:
            ok_int[f"{arm} skill"] = same(pooled["arms"][arm]["skill_vs_persistence"],
                                          ref[arm]["skill_vs_persistence"])
            ok_int[f"{arm} chi2 mean"] = same(pooled["arms"][arm]["chi2_mean_vs_1p5"],
                                              ref[arm]["chi2_mean_vs_1p5"])
        breg = np.full(rg.N_BOOT, np.nan)
        for b in range(rg.N_BOOT):
            x = reg[bidx_all[b]]
            if np.isfinite(x).any():
                breg[b] = np.nanmedian(x)
        e_reg = rg.interval(float(np.nanmedian(reg)), breg, -0.5)
        ok_int["regime number"] = same(
            e_reg, add1["retuned"]["regime"][tg]["regime_number_median_over_TEST_shots"])
        e_q = rg.interval(float(np.median(q_rows)) / rg.GAUSS_MED_CHI2_1,
                          rg.boot_weighted_median(q_rows, q_shot, counts_all) / rg.GAUSS_MED_CHI2_1, 1.0)
        ok_int["error ratio"] = same(e_q, add1["retuned"]["gate_a"][tg]["coreported_ratio"])
        res["integrity"][tg] = ok_int
        if not all(ok_int.values()):
            raise SystemExit(f"integrity check failed for {tg}: {ok_int}")
        log(f"{tg}: pooled addendum-1 numbers reproduced")

    # ---- strata
    for di, dim in enumerate(dims):
        res["strata"][dim] = {}
        for vi, v in enumerate(sorted(set(lab[s][dim] for s in test_shots))):
            members = np.array([tpos[s] for s in test_shots if lab[s][dim] == v])
            srng = np.random.default_rng([rg.SEED, di, vi])
            bi = members[srng.integers(0, len(members), size=(rg.N_BOOT, len(members)))]
            cts = np.zeros((rg.N_BOOT, nT))
            for b in range(rg.N_BOOT):
                cts[b] = np.bincount(bi[b], minlength=nT)
            inm = np.zeros(nT, bool)
            inm[members] = True
            out = {"test_shots": int(len(members))}
            for tg in rg.TARGETS:
                tab, reg, q_rows, q_shot = per_target[tg]
                sel = inm[tab["shot"]]
                n_with_rows = int(len(np.unique(tab["shot"][sel])))
                few = n_with_rows < MIN_SHOTS
                o = {"shots_with_rows": n_with_rows, "n_rows": int(sel.sum()), "too_few": few}
                rs = reg[members]
                if np.isfinite(rs).any():
                    breg = np.full(rg.N_BOOT, np.nan)
                    for b in range(rg.N_BOOT):
                        x = reg[bi[b]]
                        if np.isfinite(x).any():
                            breg[b] = np.nanmedian(x)
                    o["regime_number"] = strip(rg.interval(float(np.nanmedian(rs)), breg, 0.0), few)
                    o["regime_shots"] = int(np.isfinite(rs).sum())
                    o["regime_share_of_shots_positive"] = float(np.mean(rs[np.isfinite(rs)] > 0))
                else:
                    o["regime_number"] = strip(None, few)
                if sel.sum() > 0:
                    r = rg.readouts(rg.subset(tab, sel), cts, nT, with_median=False)
                    for arm in ARMS3:
                        o[f"skill_{arm}"] = strip(r["arms"][arm]["skill_vs_persistence"], few)
                        o[f"chi2_mean_{arm}"] = strip(r["arms"][arm]["chi2_mean_vs_1p5"], few)
                    o["rmse_persistence"] = r["arms"]["persistence"]["rmse"]
                qs = inm[q_shot]
                if qs.sum() > 0:
                    with np.errstate(all="ignore"):
                        e = rg.interval(
                            float(np.median(q_rows[qs])) / rg.GAUSS_MED_CHI2_1,
                            rg.boot_weighted_median(q_rows[qs], q_shot[qs], cts) / rg.GAUSS_MED_CHI2_1,
                            1.0)
                    o["error_ratio"] = strip(e, few)
                out[tg] = o
            res["strata"][dim][v] = out
        log(f"strata of {dim} done")

    # ---- statements
    st = res["statements"]
    camp_s = res["strata"]["campaign"]
    for name, key, cond in (
            ("S1", "regime_number", lambda e: e["side"] == "above"),
            ("S2", "chi2_mean_causal_gp", lambda e: e["side"] == "above"),
            ("S3", "skill_running_mean", lambda e: e["side"] in ("below", "includes"))):
        st[name] = {}
        for tg in rg.TARGETS:
            per = {c: camp_s[c][tg].get(key) for c in CAMPS if c in camp_s}
            hold = {c: bool(e is not None and e.get("side") not in (None, "too few") and cond(e))
                    for c, e in per.items()}
            st[name][tg] = {"per_campaign": per, "holds_in": hold,
                            "n_campaigns_holding": int(sum(hold.values())),
                            "holds": bool(len(hold) == 4 and all(hold.values()))}
            if name in ("S1", "S2"):
                thr = 0.0 if name == "S1" else 1.5
                st[name][tg]["n_campaigns_point_above"] = int(sum(
                    e is not None and e.get("point") is not None and e["point"] > thr
                    for e in per.values()))
    s4 = {}
    for pop in rg.POPS:
        rows = {}
        for h, r in counts_tab["heating"].items():
            n = r["with_data"]
            nb = r[pop]["no_block"]
            rows[h] = {"with_data": n, "unusable": r[pop]["unusable"], "no_block": nb,
                       "no_scored_chord": r[pop]["no_scored_chord"], "ok": r[pop]["ok"],
                       "share_no_block": nb / n if n else None}
        ss = [v for h, v in rows.items() if has_ss(h)]
        no = [v for h, v in rows.items() if h in ("SW Beam", "Ohmic")]
        a = sum(v["no_block"] for v in ss) / max(1, sum(v["with_data"] for v in ss))
        b = sum(v["no_block"] for v in no) / max(1, sum(v["with_data"] for v in no))
        s4[pop] = {"by_heating": rows,
                   "share_no_block_with_SS_beam": a, "n_with_SS_beam": sum(v["with_data"] for v in ss),
                   "share_no_block_without_SS_beam": b,
                   "n_without_SS_beam": sum(v["with_data"] for v in no),
                   "holds": bool(b >= 0.8 and a <= 0.2)}
    st["S4"] = s4
    # gate shots that carry no act arrays at all (store present, arrays absent), by heating tag
    meta_all = pd.read_parquet(rg.META, columns=["shot_id", "heating"])
    meta_all = meta_all[meta_all.shot_id.isin(shots)].set_index("shot_id")
    nd = {}
    for s in shots:
        h = heating_class(tag(meta_all.loc[s, "heating"]))
        r = nd.setdefault(h, {"gate_shots": 0, "without_act_arrays": 0})
        r["gate_shots"] += 1
        r["without_act_arrays"] += int(s not in data)
    st["S4"]["gate_shots_without_act_arrays_by_heating"] = nd

    res["stratum_f"] = stratum_f({"data": data, "prep": prep, "test_shots": test_shots, "tpos": tpos,
                                  "nT": nT, "counts_all": counts_all, "bidx_all": bidx_all,
                                  "add1": add1, "usable": usable, "log": log})
    res["sha256"]["mast_gate.json_after"] = sha256(FIRST_JSON)
    res["sha256"]["mast_gate_addendum1.json_after"] = sha256(ADD1_JSON)
    res["earlier_outputs_unchanged"] = bool(
        res["sha256"]["mast_gate.json_after"] == sha_first
        and res["sha256"]["mast_gate_addendum1.json_after"] == sha_add1)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    log(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    for k in ("S1", "S2", "S3"):
        print(k, {tg: (st[k][tg]["holds"], st[k][tg]["n_campaigns_holding"]) for tg in rg.TARGETS},
              flush=True)
    print("S4", {p: (round(s4[p]["share_no_block_without_SS_beam"], 3),
                     round(s4[p]["share_no_block_with_SS_beam"], 3), s4[p]["holds"]) for p in rg.POPS},
          flush=True)
    return res


RHO_EDGES = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
RHO_BINS = tuple(f"{a:.1f}-{b:.1f}" for a, b in zip(RHO_EDGES[:-1], RHO_EDGES[1:]))
KSTAR_BIN = "0.7-0.9"


def rho_bin(rho):
    """Bin index of each rho (left-closed); -1 outside (0, 1) or not finite; rho <= 0 is not low-field."""
    r = np.asarray(rho, float)
    out = np.full(r.shape, -1, int)
    ok = np.isfinite(r) & (r > 0.0) & (r < 1.0)
    out[ok] = np.searchsorted(np.asarray(RHO_EDGES), r[ok], side="right") - 1
    return out


def w_block(d):
    """Beam block on population W (T_i masks), as rg.prepare defines it."""
    good = rg.good_masks(d, rg.POPS["W"])["T_i"]
    frames = np.nonzero(good.sum(1) >= 10)[0]
    if len(frames) == 0:
        return None
    a, b = int(frames[0]), int(frames[-1])
    return (a, b) if b - a + 1 >= 20 else None


def stratum_f(ctx):
    data, prep, test_shots, tpos, nT = ctx["data"], ctx["prep"], ctx["test_shots"], ctx["tpos"], ctx["nT"]
    counts_all, bidx_all, add1, usable, log = (ctx["counts_all"], ctx["bidx_all"], ctx["add1"],
                                               ctx["usable"], ctx["log"])
    cat = pd.read_parquet(rg.META, columns=["shot_id", "radii_magnetic__radius_efit",
                                            "radii_router_radius_efit", "radii_router_radius_d_alpha"])
    cat = cat[cat.shot_id.isin(test_shots)].set_index("shot_id")
    geo, src = {}, {"efit": 0, "d_alpha_outer": 0, "missing": 0}
    for s in test_shots:
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        ro = float(cat.loc[s, "radii_router_radius_efit"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            geo[s] = (ax, ro)
            src["efit"] += 1
            continue
        ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            geo[s] = (ax, ro)
            src["d_alpha_outer"] += 1
        else:
            src["missing"] += 1
    ax_all = np.array([geo[s][0] for s in geo])
    ro_all = np.array([geo[s][1] for s in geo])
    out = {"bins": list(RHO_BINS), "kstar_matched_bin": KSTAR_BIN, "geometry_source_TEST_shots": src,
           "R_axis_median_m": float(np.median(ax_all)), "R_out_median_m": float(np.median(ro_all)),
           "readouts": {}, "pre_rule": {}, "statements": {}}

    def rho_of(s, R):
        if s not in geo:
            return np.full(np.shape(R), np.nan)
        ax, ro = geo[s]
        return (np.asarray(R, float) - ax) / (ro - ax)

    boot_reg = {}
    for tg in rg.TARGETS:
        params = params_from(add1, tg)
        for pop in rg.POPS:
            acc = {"shot": [], "R": [], "sig2": [], "bin": [], "r2": {a: [] for a in rg.ARMS},
                   "den": {a: [] for a in rg.ARMS}}
            reg = np.full((len(RHO_BINS), nT), np.nan)
            q_rows, q_shot, q_bin = [], [], []
            dropped = {"rho_ge_1": 0, "high_field_side": 0, "no_geometry_or_radius": 0}
            for s in test_shots:
                p = prep[tg, pop][s]
                if p["status"] != "ok":
                    continue
                rho = rho_of(s, p["R"])
                cb = rho_bin(rho)
                dropped["no_geometry_or_radius"] += int((~np.isfinite(rho)).sum())
                dropped["rho_ge_1"] += int((np.isfinite(rho) & (rho >= 1.0)).sum())
                dropped["high_field_side"] += int((np.isfinite(rho) & (rho <= 0.0)).sum())
                G, S2 = p["G"], p["S2"]
                Yn = np.where(G, p["Y0"], np.nan)
                for k in range(len(RHO_BINS)):
                    c = cb == k
                    if c.any():
                        reg[k, tpos[s]] = rg.regime_number(Yn[:, c], G[:, c])
                d1 = p["Y0"][2:] - 2.0 * p["Y0"][1:-1] + p["Y0"][:-2]
                ok = G[2:] & G[1:-1] & G[:-2]
                var = 4.0 * S2[1:-1] + S2[2:] + S2[:-2]
                q_rows.append((d1[ok] ** 2) / var[ok])
                q_shot.append(np.full(int(ok.sum()), tpos[s]))
                q_bin.append(np.broadcast_to(cb[None, :], ok.shape)[ok])
                if p["n_rows"] == 0:
                    continue
                rows = p["rows"]
                yt = p["Y0"][p["hid"]][rows]
                s2t = p["S2"][p["hid"]][rows]
                acc["shot"].append(np.full(len(yt), tpos[s]))
                acc["R"].append(np.broadcast_to(p["R"][None, :], rows.shape)[rows])
                acc["bin"].append(np.broadcast_to(cb[None, :], rows.shape)[rows])
                acc["sig2"].append(s2t)
                for arm in rg.ARMS:
                    pred, vw = rg.arm_outputs(p, arm, params, mask=rows)
                    acc["r2"][arm].append((pred[rows] - yt) ** 2)
                    acc["den"][arm].append(s2t + vw[rows])
            tab = {"shot": np.concatenate(acc["shot"]).astype(np.int64),
                   "R": np.concatenate(acc["R"]), "sig2": np.concatenate(acc["sig2"]),
                   "r2": {a: np.concatenate(v) for a, v in acc["r2"].items()},
                   "den": {a: np.concatenate(v) for a, v in acc["den"].items()}}
            rbin = np.concatenate(acc["bin"])
            q_rows = np.concatenate(q_rows)
            q_shot = np.concatenate(q_shot).astype(np.int64)
            q_bin = np.concatenate(q_bin)
            cell = {"scored_chords_dropped": dropped, "rows_total": int(len(rbin)),
                    "rows_in_a_bin": int((rbin >= 0).sum()), "bins": {}}
            for k, name in enumerate(RHO_BINS):
                sel = rbin == k
                n_sh = int(len(np.unique(tab["shot"][sel])))
                few = n_sh < MIN_SHOTS
                o = {"shots_with_rows": n_sh, "n_rows": int(sel.sum()), "too_few": few}
                rs = reg[k]
                n_reg = int(np.isfinite(rs).sum())
                o["regime_shots"] = n_reg
                if n_reg > 0:
                    br = np.full(rg.N_BOOT, np.nan)
                    for b in range(rg.N_BOOT):
                        x = rs[bidx_all[b]]
                        if np.isfinite(x).any():
                            br[b] = np.nanmedian(x)
                    boot_reg[tg, pop, name] = br
                    o["regime_number"] = strip(rg.interval(float(np.nanmedian(rs)), br, 0.0),
                                               n_reg < MIN_SHOTS)
                if sel.sum() > 0:
                    r = rg.readouts(rg.subset(tab, sel), counts_all, nT, with_median=True)
                    for arm in ARMS3:
                        o[f"skill_{arm}"] = strip(r["arms"][arm]["skill_vs_persistence"], few)
                        o[f"chi2_mean_{arm}"] = strip(r["arms"][arm]["chi2_mean_vs_1p5"], few)
                        o[f"chi2_median_{arm}"] = strip(r["arms"][arm]["chi2_median"], few)
                    o["rmse_persistence"] = r["arms"]["persistence"]["rmse"]
                qs = q_bin == k
                if qs.sum() > 0:
                    with np.errstate(all="ignore"):
                        e = rg.interval(
                            float(np.median(q_rows[qs])) / rg.GAUSS_MED_CHI2_1,
                            rg.boot_weighted_median(q_rows[qs], q_shot[qs], counts_all)
                            / rg.GAUSS_MED_CHI2_1, 1.0)
                    o["error_ratio"] = strip(e, int(len(np.unique(q_shot[qs]))) < MIN_SHOTS)
                cell["bins"][name] = o
            out["readouts"][f"{tg}|{pop}"] = cell
            log(f"stratum f {tg} {pop}: rows in a bin {cell['rows_in_a_bin']} of {cell['rows_total']}")

    # ---- pre-rule samples on the population-W block
    nB = len(RHO_BINS)
    S = {k: np.zeros((nB, nT)) for k in ("n", "finite", "fail01", "fail05")}
    vals = {k: [[] for _ in range(nB)] for k in ("rel", "verr", "emis")}
    vsh = {k: [[] for _ in range(nB)] for k in ("rel", "verr", "emis")}
    skipped = {"unusable": 0, "no_W_block": 0, "no_geometry": 0, "radius_columns_mismatch": 0}
    used, emis_shots = 0, 0
    for s in test_shots:
        d = data[s]
        if not usable[s]:
            skipped["unusable"] += 1
            continue
        blk = w_block(d)
        if blk is None:
            skipped["no_W_block"] += 1
            continue
        if s not in geo:
            skipped["no_geometry"] += 1
            continue
        if d["R"].shape[0] != d["T"].shape[1]:
            skipped["radius_columns_mismatch"] += 1
            continue
        a, b = blk
        T, Te, Ve = d["T"][a:b + 1], d["Te"][a:b + 1], d["Ve"][a:b + 1]
        fin = np.isfinite(T)
        cols = fin.any(0)
        cb = rho_bin(rho_of(s, d["R"]))
        g01 = rg.good_masks(d, 0.1)["T_i"][a:b + 1]
        g05 = rg.good_masks(d, 0.5)["T_i"][a:b + 1]
        with np.errstate(all="ignore"):
            okrel = fin & np.isfinite(Te) & (T > 0) & (Te > 0)
            rel = Te / T
        em = None
        with np.load(rg.ACT_DIR / f"{s}.npz") as z:
            if "act_ss_cx_emissivity" in z.files and z["act_ss_cx_emissivity"].shape[-1] == T.shape[1]:
                em = z["act_ss_cx_emissivity"].astype(float)[a:b + 1]
        used += 1
        emis_shots += int(em is not None)
        for k in range(nB):
            c = cols & (cb == k)
            if not c.any():
                continue
            S["n"][k, tpos[s]] = c.sum() * T.shape[0]
            S["finite"][k, tpos[s]] = fin[:, c].sum()
            S["fail01"][k, tpos[s]] = (fin[:, c] & ~g01[:, c]).sum()
            S["fail05"][k, tpos[s]] = (fin[:, c] & ~g05[:, c]).sum()
            m = okrel[:, c]
            vals["rel"][k].append(rel[:, c][m])
            vsh["rel"][k].append(np.full(int(m.sum()), tpos[s]))
            mv = m & np.isfinite(Ve[:, c]) & (Ve[:, c] > 0)
            vals["verr"][k].append(Ve[:, c][mv])
            vsh["verr"][k].append(np.full(int(mv.sum()), tpos[s]))
            if em is not None:
                me = np.isfinite(em[:, c]) & fin[:, c]
                vals["emis"][k].append(em[:, c][me])
                vsh["emis"][k].append(np.full(int(me.sum()), tpos[s]))
    pre = {"shots_used": used, "shots_with_emissivity_columns_matching": emis_shots,
           "shots_skipped": skipped, "bins": {}}
    for k, name in enumerate(RHO_BINS):
        n_sh = int((S["n"][k] > 0).sum())
        few = n_sh < MIN_SHOTS
        o = {"shots": n_sh, "samples": int(S["n"][k].sum()), "finite_samples": int(S["finite"][k].sum())}
        if S["finite"][k].sum() > 0:
            bf = counts_all @ S["finite"][k]
            bn = counts_all @ S["n"][k]
            with np.errstate(all="ignore"):
                for key in ("fail01", "fail05"):
                    o[f"share_{key}"] = strip(rg.interval(S[key][k].sum() / S["finite"][k].sum(),
                                                          (counts_all @ S[key][k]) / bf, 0.0), few)
                o["share_non_finite"] = strip(rg.interval(1.0 - S["finite"][k].sum() / S["n"][k].sum(),
                                                          1.0 - bf / bn, 0.0), few)
        for key, label in (("rel", "median_rel_Ti_error"), ("verr", "median_Vi_error_kms"),
                           ("emis", "median_emissivity")):
            if vals[key][k]:
                v = np.concatenate(vals[key][k])
                sh = np.concatenate(vsh[key][k]).astype(np.int64)
                if len(v):
                    e = rg.interval(float(np.median(v)), rg.boot_weighted_median(v, sh, counts_all), 0.0)
                    o[label] = strip(e, int(len(np.unique(sh))) < MIN_SHOTS)
                    o[label + "_shots"] = int(len(np.unique(sh)))
        pre["bins"][name] = o
    out["pre_rule"] = pre
    log("stratum f pre-rule readout done")

    # ---- statements S5-S8
    def has_ci(e):
        return e is not None and e.get("point") is not None and e.get("lo") is not None

    def monotone(points, increasing=True):
        d = np.diff(points)
        return bool(len(points) >= 2 and (np.all(d >= 0) if increasing else np.all(d <= 0)))

    s5 = {}
    for tg in rg.TARGETS:
        s5[tg] = {}
        for pop in rg.POPS:
            e = out["readouts"][f"{tg}|{pop}"]["bins"][KSTAR_BIN].get("regime_number")
            s5[tg][pop] = {"regime_number": e, "holds": bool(has_ci(e) and e["side"] == "above"),
                           "readable": bool(has_ci(e))}
        s5[tg]["read_on"] = "G" if s5[tg]["G"]["readable"] else "W"
        s5[tg]["holds"] = s5[tg][s5[tg]["read_on"]]["holds"]
    s6 = {}
    for tg in rg.TARGETS:
        s6[tg] = {}
        for pop in rg.POPS:
            pts = [(n, out["readouts"][f"{tg}|{pop}"]["bins"][n].get("chi2_mean_causal_gp")) for n in RHO_BINS]
            pts = [(n, e["point"]) for n, e in pts if has_ci(e)]
            s6[tg][pop] = {"bins_with_interval": [n for n, _ in pts], "points": [p for _, p in pts],
                           "holds": monotone([p for _, p in pts], True)}
    s7 = {}
    for key in ("share_fail01", "median_rel_Ti_error"):
        pts = [(n, pre["bins"][n].get(key)) for n in RHO_BINS]
        pts = [(n, e["point"]) for n, e in pts if has_ci(e)]
        vals_ = [p for _, p in pts]
        s7[key] = {"bins_with_interval": [n for n, _ in pts], "points": vals_,
                   "non_decreasing": monotone(vals_, True),
                   "largest_in_0.9-1.0": bool(pts and pts[-1][0] == "0.9-1.0" and vals_[-1] == max(vals_)),
                   "outermost_bin_with_interval": pts[-1][0] if pts else None,
                   "largest_in_outermost_bin_with_interval": bool(pts and vals_[-1] == max(vals_))}
        s7[key]["holds"] = bool(s7[key]["non_decreasing"] and s7[key]["largest_in_0.9-1.0"])
    s7["holds"] = bool(s7["share_fail01"]["holds"] and s7["median_rel_Ti_error"]["holds"])
    s8 = {}
    for pop in ("W", "G"):
        pts = [(n, out["readouts"][f"T_i|{pop}"]["bins"][n].get("regime_number")) for n in RHO_BINS]
        pts = [(n, e["point"]) for n, e in pts if has_ci(e)]
        o = {"bins_with_interval": [n for n, _ in pts], "points": [p for _, p in pts],
             "non_increasing": monotone([p for _, p in pts], False)}
        ka, kb = ("T_i", pop, KSTAR_BIN), ("T_i", pop, "0.0-0.3")
        ea = out["readouts"][f"T_i|{pop}"]["bins"][KSTAR_BIN].get("regime_number")
        eb = out["readouts"][f"T_i|{pop}"]["bins"]["0.0-0.3"].get("regime_number")
        if has_ci(ea) and has_ci(eb):
            o["difference_0.7-0.9_minus_0.0-0.3"] = rg.interval(ea["point"] - eb["point"],
                                                                boot_reg[ka] - boot_reg[kb], 0.0)
            o["holds"] = bool(o["non_increasing"] and o["difference_0.7-0.9_minus_0.0-0.3"]["side"] == "below")
        else:
            o["difference_0.7-0.9_minus_0.0-0.3"] = None
            o["holds"] = False
        s8[pop] = o
    s8["holds"] = s8["W"]["holds"]
    out["statements"] = {"S5": s5, "S6": s6, "S7": s7, "S8": s8}
    return out


def render_f(f):
    L = ["## 5. Stratum (f): normalized radius", "",
         "rho = (R_chord - R_axis) / (R_out - R_axis) from the catalogue EFIT scalars, one value per "
         "discharge (time variation inside the beam block ignored), low-field side only. Geometry source "
         f"over the TEST shots: {f['geometry_source_TEST_shots']}; median R_axis {f['R_axis_median_m']:.3f} m, "
         f"median R_out {f['R_out_median_m']:.3f} m. The KSTAR-matched bin is {f['kstar_matched_bin']} "
         "(nominal KSTAR geometry, (2.198 - 1.8) / 0.5 = 0.80; not an EFIT mapping of the KSTAR discharges). "
         "Intervals: 2,000 bootstrap resamples of TEST shots.", ""]
    for key, cell in f["readouts"].items():
        tg, pop = key.split("|")
        L += [f"**{tg}, population {pop}** - rows in a bin {cell['rows_in_a_bin']} of {cell['rows_total']}; "
              f"scored chords outside the bins: {cell['scored_chords_dropped']}", "",
              "| rho | shots with rows | rows | regime number (ref 0) | shots with a regime number | causal GP mean chi-square (ref 1.5) | causal GP median chi-square (ref 1) | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, t in cell["bins"].items():
            L.append(f"| {n} | {t['shots_with_rows']} | {t['n_rows']} | {fmt(t.get('regime_number'))} | "
                     f"{t['regime_shots']} | {fmt(t.get('chi2_mean_causal_gp'), 2, False)} | "
                     f"{fmt(t.get('chi2_median_causal_gp'), 3, False)} | {fmt(t.get('skill_causal_gp'))} | "
                     f"{fmt(t.get('skill_slope_filter'))} | {fmt(t.get('skill_running_mean'))} | "
                     f"{fmt(t.get('error_ratio'), 3, False)} |")
        L.append("")
    p = f["pre_rule"]
    L += ["### Does the measurement weaken outward (samples before the fit rule)", "",
          f"Usable TEST shots with a population-W beam block, geometry and a radius per chord column: "
          f"{p['shots_used']} (skipped: {p['shots_skipped']}); emissivity columns match the chords in "
          f"{p['shots_with_emissivity_columns_matching']} of them. Every low-field chord column with at "
          "least one finite T_i value in the block, every block frame.", "",
          "| rho | shots | samples | fail share, cut 0.1 | fail share, cut 0.5 | non-finite share | median relative T_i error | median V_i error (km/s) | median emissivity (shots) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, o in p["bins"].items():
        L.append(f"| {n} | {o['shots']} | {o['samples']} | {fmt(o.get('share_fail01'), 3, False)} | "
                 f"{fmt(o.get('share_fail05'), 3, False)} | {fmt(o.get('share_non_finite'), 3, False)} | "
                 f"{fmt(o.get('median_rel_Ti_error'), 3, False)} | {fmt(o.get('median_Vi_error_kms'), 2, False)} | "
                 f"{fmt(o.get('median_emissivity'), 3, False)} ({o.get('median_emissivity_shots', 0)}) |")
    L += ["", "The emissivity intervals are not usable: a few shots store emissivity on a scale about "
          "1e14 times larger than the rest, so the upper bootstrap bounds jump to that scale. The "
          "medians are reported as stored; no statement rests on them."]
    st = f["statements"]
    L += ["", "### Statements S5-S8", ""]
    for tg in ("T_i", "V_i"):
        s = st["S5"][tg]
        L.append(f"- S5, {tg}: regime number in rho {f['kstar_matched_bin']} - population G "
                 f"{fmt(s['G']['regime_number'])}, {side(s['G']['regime_number'])}; population W "
                 f"{fmt(s['W']['regime_number'])}, {side(s['W']['regime_number'])}. Read on population "
                 f"{s['read_on']}: {'holds' if s['holds'] else 'does not hold'}.")
    for tg in ("T_i", "V_i"):
        for pop in ("G", "W"):
            s = st["S6"][tg][pop]
            L.append(f"- S6, {tg}, population {pop}: causal GP mean reduced chi-square over bins "
                     f"{s['bins_with_interval']} = {[round(x, 2) for x in s['points']]}; non-decreasing: "
                     f"{'yes' if s['holds'] else 'no'}" + (" (registered population)." if pop == "G" else " (co-reported)."))
    for key, label in (("share_fail01", "fail share at the cut 0.1"),
                       ("median_rel_Ti_error", "median relative T_i error")):
        s = st["S7"][key]
        L.append(f"- S7, {label}: bins {s['bins_with_interval']} = {[round(x, 3) for x in s['points']]}; "
                 f"non-decreasing: {'yes' if s['non_decreasing'] else 'no'}; largest in 0.9-1.0: "
                 f"{'yes' if s['largest_in_0.9-1.0'] else 'no'}; {'holds' if s['holds'] else 'does not hold'}.")
    L.append(f"- S7 overall: {'holds' if st['S7']['holds'] else 'does not hold'}.")
    for pop in ("W", "G"):
        s = st["S8"][pop]
        L.append(f"- S8, T_i regime number, population {pop}: bins {s['bins_with_interval']} = "
                 f"{[round(x, 3) for x in s['points']]}; non-increasing: {'yes' if s['non_increasing'] else 'no'}; "
                 f"difference (0.7-0.9 minus 0.0-0.3) {fmt(s['difference_0.7-0.9_minus_0.0-0.3'])}, "
                 f"{side(s['difference_0.7-0.9_minus_0.0-0.3'])}; {'holds' if s['holds'] else 'does not hold'}"
                 + (" (registered population)." if pop == "W" else " (co-reported)."))
    L.append("")
    return L


TITLES = {"campaign": "Campaign (primary)", "heating": "Heating (catalogue)",
          "rmp_coil": "RMP coil (catalogue)", "chord_columns": "CXRS chord columns (stored arrays)",
          "frame_spacing": "Frame spacing (stored times)", "plasma_shape": "Plasma shape (auxiliary)",
          "current_range": "Current range (auxiliary)"}


def render_md(res):
    L = ["# MAST gate, addendum 2 - breakdown by campaign and machine configuration", "",
         f"Registration: `{res['registration']}`. Post hoc: the pooled TEST results were seen before "
         "this breakdown was registered. Hyperparameters are those of addendum 1; nothing is retuned. "
         "TEST shots, population G (relative T_i error < 0.1). Intervals: 2,000 bootstrap resamples "
         f"of the shots inside the stratum; a stratum with fewer than {res['min_shots_for_interval']} "
         "TEST shots carrying scored rows has point values only (\"too few\"). No correction for "
         "multiplicity; only S1-S4 are statements.", "",
         "Integrity: the pooled addendum-1 numbers (rows, skills, mean reduced chi-square of every "
         "arm, regime number, error ratio) are reproduced exactly for both targets; "
         f"`mast_gate.json` and `mast_gate_addendum1.json` unchanged: {res['earlier_outputs_unchanged']}.",
         "", "## 1. Statements", ""]
    st = res["statements"]
    desc = {"S1": "regime number above 0 in every campaign",
            "S2": "mean reduced chi-square of the causal GP above 1.5 in every campaign",
            "S3": "running-mean skill vs persistence not above 0 in every campaign"}
    L += ["| statement | target | M6 | M7 | M8 | M9 | campaigns holding | verdict |", "|---|---|---|---|---|---|---|---|"]
    for k in ("S1", "S2", "S3"):
        for tg in ("T_i", "V_i"):
            s = st[k][tg]
            cells = " | ".join(f"{fmt(s['per_campaign'].get(c), 3, k != 'S2')}, {side(s['per_campaign'].get(c))}"
                               for c in CAMPS)
            L.append(f"| {k}: {desc[k]} | {tg} | {cells} | {s['n_campaigns_holding']} / 4 | "
                     f"{'holds' if s['holds'] else 'does not hold'} |")
    L += ["", "S4: gate shots without a beam block, by catalogue heating (all 264 shots with data; "
          "\"no block\" = usable frame spacing but no block of at least 20 frames with 10 good chords).", ""]
    for pop in ("G", "W"):
        s = st["S4"][pop]
        L += [f"Population {pop}:", "", "| heating | with data | unusable spacing | no block | block, no scored chord | block and chords | share no block |",
              "|---|---|---|---|---|---|---|"]
        for h, r in s["by_heating"].items():
            L.append(f"| {h} | {r['with_data']} | {r['unusable']} | {r['no_block']} | {r['no_scored_chord']} | "
                     f"{r['ok']} | {r['share_no_block']:.3f} |")
        L += ["", f"Without the SS beam (SW Beam, Ohmic; {s['n_without_SS_beam']} shots): share without a block "
              f"{s['share_no_block_without_SS_beam']:.3f} (registered: at least 0.8). With the SS beam "
              f"({s['n_with_SS_beam']} shots): {s['share_no_block_with_SS_beam']:.3f} (registered: at most 0.2). "
              f"S4 {'holds' if s['holds'] else 'does not hold'} on population {pop}.", ""]
    nd = st["S4"]["gate_shots_without_act_arrays_by_heating"]
    L += ["Not part of the registered S4 reading: of the 300 gate shots, those that carry no `act` "
          "arrays at all, by heating tag - "
          + "; ".join(f"{h}: {r['without_act_arrays']} of {r['gate_shots']}" for h, r in sorted(nd.items()))
          + ". Every SW-beam-only gate shot is among the shots without arrays, so the registered "
          "comparison has only the Ohmic shots on its no-SS side.", "",
          "## 2. Readouts per stratum", ""]
    for dim, title in TITLES.items():
        L += [f"### {title}", ""]
        for tg in ("T_i", "V_i"):
            L += [f"**{tg}**", "",
                  "| stratum | TEST shots | with rows | rows | regime number (ref 0) | causal GP mean chi-square (ref 1.5) | slope filter mean chi-square | causal GP skill | slope filter skill | running mean skill | error ratio (ref 1) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for v, o in res["strata"][dim].items():
                t = o[tg]
                L.append(
                    f"| {v} | {o['test_shots']} | {t['shots_with_rows']} | {t['n_rows']} | "
                    f"{fmt(t.get('regime_number'))} | {fmt(t.get('chi2_mean_causal_gp'), 2, False)} | "
                    f"{fmt(t.get('chi2_mean_slope_filter'), 2, False)} | {fmt(t.get('skill_causal_gp'))} | "
                    f"{fmt(t.get('skill_slope_filter'))} | {fmt(t.get('skill_running_mean'))} | "
                    f"{fmt(t.get('error_ratio'), 3, False)} |")
            L.append("")
    L += ["## 3. Counts over the 264 gate shots with data (TUNE and TEST)", ""]
    for dim, title in TITLES.items():
        L += [f"### {title}", "",
              "| stratum | with data | usable spacing | TEST | G: no block | G: block, no scored chord | G: block and chords | W: no block | W: block, no scored chord | W: block and chords |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for v, r in res["counts"][dim].items():
            L.append(f"| {v} | {r['with_data']} | {r['usable']} | {r['TEST']} | {r['G']['no_block']} | "
                     f"{r['G']['no_scored_chord']} | {r['G']['ok']} | {r['W']['no_block']} | "
                     f"{r['W']['no_scored_chord']} | {r['W']['ok']} |")
        L.append("")
    L += ["## 4. Notes", "",
          "- Strata are confounded with each other and with campaign; half of the heating, RMP and shape tags are missing. No causal reading is attached to a configuration stratum.",
          "- The stratum bootstrap resamples the TEST shots of that stratum (shots without scored rows included, as in the gate); the seed sequence is (20261116, dimension index, stratum index).",
          "- Counts use the T_i masks; the V_i counts equal them in both populations (gate report).",
          "- `chord_columns` is the column count of `act_ss_temperature`; `frame_spacing` is the stored median frame interval, \"5 ms\" meaning inside 5.0 +- 0.1 ms (the usable rule).", ""]
    L += render_f(res["stratum_f"])
    return "\n".join(L)


if __name__ == "__main__":
    main()
    sys.exit(0)
