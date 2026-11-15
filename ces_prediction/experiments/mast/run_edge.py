"""MAST edge confirmation (PREREGISTRATION_MAST_EDGE.md): fit-failure share, regime number and
CES-only skill by normalized radius on the MAST discharges outside the gate and pilot lists.

Reuses run_gate.py (masks, block, arms, regime number) and the hyperparameters selected on the
gate TUNE shots in addendum 1; nothing is retuned.  Shots are read one at a time and only per-shot,
per-bin sums are kept.  Run from the repo root:  py ces_prediction/experiments/mast/run_edge.py
Writes docs/paper/mast_edge.json and docs/paper/mast_edge.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate as rg  # noqa: E402

REG = "ces_prediction/experiments/PREREGISTRATION_MAST_EDGE.md"
HERE = Path(__file__).resolve().parent
ALL_TXT = HERE / "all_shots.txt"
GATE_TXT = HERE / "gate_shots.txt"
PILOT = {17703, 18330, 18064, 21770, 24411, 22622, 26581, 27244, 27288, 30448, 30013, 28991, 30147}
ADD1_JSON = rg.ROOT / "docs" / "paper" / "mast_gate_addendum1.json"
FETCH_LOG = rg.ROOT / "data" / "mast" / "fetch.log"
SOURCES = rg.ROOT / "data" / "mast" / "meta" / "l1_sources.parquet"
OUT_JSON = rg.ROOT / "docs" / "paper" / "mast_edge.json"
OUT_MD = rg.ROOT / "docs" / "paper" / "mast_edge.md"

EDGES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
BINS = tuple(f"{a:g}-{b:g}" for a, b in zip(EDGES[:-1], EDGES[1:]))
NB = len(BINS)                       # 7 primary bins
MERGED = {"0.9-1": (5, 6), "0-0.7": (0, 1, 2)}
REF, EDGE_A, EDGE_B, CORE = "0.5-0.7", "0.8-0.9", "0.9-1", "0-0.7"
E1_BINS = ("0.8-0.9", "0.9-0.95", "0.95-1")
ARMS = ("persistence", "running_mean", "rm15", "rm40", "slope_filter", "causal_gp")
MIN_SHOTS = 8
CAMPS = ("M6", "M7", "M8", "M9")
COMBOS = [(tg, pop) for tg in rg.TARGETS for pop in rg.POPS]


def rho_bin(rho):
    r = np.asarray(rho, float)
    out = np.full(r.shape, -1, int)
    ok = np.isfinite(r) & (r > 0.0) & (r < 1.0)
    out[ok] = np.searchsorted(np.asarray(EDGES), r[ok], side="right") - 1
    return out


def params_from(add1, tg):
    t = add1["retuned"]["tuning"][tg]
    return {"tau_ms": t["running_mean"]["selected_tau_ms"],
            "q_level": t["slope_filter"]["selected_q_level"],
            "q_slope": t["slope_filter"]["selected_q_slope"],
            "gp_causal": (t["gp_causal"]["selected_length_ms"], t["gp_causal"]["selected_amp"]),
            "gp_offline": (t["gp_offline"]["selected_length_ms"], t["gp_offline"]["selected_amp"])}


def arm_out(p, arm, params, rows):
    if arm == "rm15":
        return rg.arm_outputs(p, "running_mean", dict(params, tau_ms=15.0))
    if arm == "rm40":
        return rg.arm_outputs(p, "running_mean", dict(params, tau_ms=40.0))
    return rg.arm_outputs(p, arm, params, mask=rows)


def w_block(d):
    good = rg.good_masks(d, rg.POPS["W"])["T_i"]
    frames = np.nonzero(good.sum(1) >= 10)[0]
    if len(frames) == 0:
        return None
    a, b = int(frames[0]), int(frames[-1])
    return (a, b) if b - a + 1 >= 20 else None


def process(shot, geo, params):
    """Per-shot record: status, pre-rule sums per bin, and per (target, population) regime
    numbers and squared-error sums per bin."""
    d = rg.load_shot(shot)
    if d is None:
        return {"status": "no_file"}
    tt = d["time_ms"]
    dd = np.diff(tt[np.isfinite(tt)])
    m = float(np.median(dd)) if len(dd) else float("nan")
    if not (np.isfinite(m) and abs(m - rg.STEP_MS) <= 0.1):
        return {"status": "unusable"}
    if geo is None:
        return {"status": "no_geometry"}
    if d["R"].shape[0] != d["T"].shape[1]:
        return {"status": "radius_columns_mismatch"}
    ax, ro = geo
    rho_all = (d["R"] - ax) / (ro - ax)
    cb_all = rho_bin(rho_all)
    rec = {"status": "ok", "pre": None, "cells": {}}
    blk = w_block(d)
    if blk is not None:
        a, b = blk
        T, Te, Ve = d["T"][a:b + 1], d["Te"][a:b + 1], d["Ve"][a:b + 1]
        fin = np.isfinite(T)
        g01 = rg.good_masks(d, 0.1)["T_i"][a:b + 1]
        g05 = rg.good_masks(d, 0.5)["T_i"][a:b + 1]
        with np.errstate(all="ignore"):
            okrel = fin & np.isfinite(Te) & (T > 0) & (Te > 0)
            rel = Te / T
        pre = np.full((NB, 5), np.nan)       # finite, fail01, fail05, med rel, med verr
        cols = fin.any(0)
        for k in range(NB):
            c = cols & (cb_all == k)
            if not c.any():
                continue
            pre[k, 0] = fin[:, c].sum()
            pre[k, 1] = (fin[:, c] & ~g01[:, c]).sum()
            pre[k, 2] = (fin[:, c] & ~g05[:, c]).sum()
            mm = okrel[:, c]
            if mm.any():
                pre[k, 3] = np.median(rel[:, c][mm])
            mv = mm & np.isfinite(Ve[:, c]) & (Ve[:, c] > 0)
            if mv.any():
                pre[k, 4] = np.median(Ve[:, c][mv])
        rec["pre"] = pre
    for tg, pop in COMBOS:
        p = rg.prepare(d, tg, rg.POPS[pop])
        if p["status"] != "ok":
            rec["cells"][tg, pop] = None
            continue
        rho = (p["R"] - ax) / (ro - ax)
        cb = rho_bin(rho)
        G = p["G"]
        Yn = np.where(G, p["Y0"], np.nan)
        reg = np.full(NB + len(MERGED), np.nan)
        for k in range(NB):
            c = cb == k
            if c.any():
                reg[k] = rg.regime_number(Yn[:, c], G[:, c])
        for j, (name, ks) in enumerate(MERGED.items()):
            c = np.isin(cb, ks)
            if c.any():
                reg[NB + j] = rg.regime_number(Yn[:, c], G[:, c])
        sse = np.zeros((NB, len(ARMS)))
        chi = np.zeros(NB)
        nrow = np.zeros(NB)
        if p["n_rows"] > 0:
            rows = p["rows"]
            yt = p["Y0"][p["hid"]]
            s2t = p["S2"][p["hid"]]
            cbr = np.broadcast_to(cb[None, :], rows.shape)
            for i, arm in enumerate(ARMS):
                pred, vw = arm_out(p, arm, params[tg], rows)
                r2 = np.where(rows, (pred - yt) ** 2, 0.0)
                for k in range(NB):
                    sel = rows & (cbr == k)
                    if sel.any():
                        sse[k, i] = r2[sel].sum()
                        if arm == "causal_gp":
                            chi[k] = (r2[sel] / (s2t[sel] + vw[sel])).sum()
                            nrow[k] = sel.sum()
        rec["cells"][tg, pop] = {"reg": reg, "sse": sse, "chi": chi, "nrow": nrow}
    return rec


# --------------------------------------------------------------------------- statistics

def ci(point, boots, ref=0.0, n=None):
    if n is not None and n < MIN_SHOTS:
        return {"point": None if point is None or not np.isfinite(point) else float(point),
                "lo": None, "hi": None, "ref": ref, "side": "too few", "n_shots": int(n)}
    e = rg.interval(point, boots, ref)
    out = {k: e[k] for k in ("point", "lo", "hi", "ref", "side")}
    if n is not None:
        out["n_shots"] = int(n)
    return out


def boot_median(x, bidx):
    """Median over shots (NaN = shot does not contribute) for each resample."""
    out = np.full(bidx.shape[0], np.nan)
    if not np.isfinite(x).any():
        return out
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(0, bidx.shape[0], 250):
                out[s:s + 250] = np.nanmedian(x[bidx[s:s + 250]], axis=1)
    return out


def merged_cols(arr, ks):
    return arr[:, list(ks)].sum(1)


def analyse(recs, shots, label, log):
    """All read-outs for one set of shots (records with status ok)."""
    n = len(shots)
    rng = np.random.default_rng(rg.SEED)
    bidx = rng.integers(0, n, size=(rg.N_BOOT, n))
    counts = np.zeros((rg.N_BOOT, n))
    for b in range(rg.N_BOOT):
        counts[b] = np.bincount(bidx[b], minlength=n)
    names = list(BINS) + list(MERGED)
    out = {"label": label, "n_shots": n, "bins": names, "pre_rule": {}, "cells": {}, "statements": {}}

    # ---- pre-rule read-outs
    pre = np.full((n, NB, 5), np.nan)
    for i, s in enumerate(shots):
        if recs[s]["pre"] is not None:
            pre[i] = recs[s]["pre"]
    out["pre_rule"]["shots_with_W_block"] = int(np.isfinite(pre[:, :, 0]).any(1).sum())
    fin = np.nan_to_num(pre[:, :, 0])
    f01 = np.nan_to_num(pre[:, :, 1])
    f05 = np.nan_to_num(pre[:, :, 2])

    def colsum(a, name):
        if name in MERGED:
            return merged_cols(a, MERGED[name])
        return a[:, BINS.index(name)]

    share_b = {}
    tab = {}
    for name in names:
        a, b1, b5 = colsum(fin, name), colsum(f01, name), colsum(f05, name)
        nsh = int((a > 0).sum())
        o = {"shots": nsh, "finite_samples": int(a.sum())}
        with np.errstate(all="ignore"):
            for key, num in (("fail_share_0p1", b1), ("fail_share_0p5", b5)):
                bb = (counts @ num) / (counts @ a)
                o[key] = ci(num.sum() / a.sum() if a.sum() > 0 else np.nan, bb, 0.0, nsh)
                share_b[key, name] = bb
        if name not in MERGED:
            for key, q in (("median_rel_Ti_error", 3), ("median_Vi_error_kms", 4)):
                x = pre[:, BINS.index(name), q]
                nn = int(np.isfinite(x).sum())
                o[key] = ci(float(np.nanmedian(x)) if nn else np.nan, boot_median(x, bidx), 0.0, nn)
        tab[name] = o
    out["pre_rule"]["bins"] = tab
    e1 = {}
    for name in E1_BINS:
        pt = tab[name]["fail_share_0p1"]["point"]
        rf = tab[REF]["fail_share_0p1"]["point"]
        nsh = min(tab[name]["shots"], tab[REF]["shots"])
        e1[name] = ci(None if pt is None or rf is None else pt - rf,
                      share_b["fail_share_0p1", name] - share_b["fail_share_0p1", REF], 0.0, nsh)
    out["statements"]["E1"] = {"differences_vs_0.5-0.7": e1,
                               "holds": all(v["side"] == "above" for v in e1.values())}

    # ---- per (target, population)
    regb = {}
    for tg, pop in COMBOS:
        reg = np.full((n, len(names)), np.nan)
        sse = np.zeros((n, NB, len(ARMS)))
        chi = np.zeros((n, NB))
        nrow = np.zeros((n, NB))
        ok_cell = 0
        for i, s in enumerate(shots):
            c = recs[s]["cells"].get((tg, pop))
            if c is None:
                continue
            ok_cell += 1
            reg[i], sse[i], chi[i], nrow[i] = c["reg"], c["sse"], c["chi"], c["nrow"]
        cell = {"shots_with_block_and_chords": ok_cell, "bins": {}}
        skb = {}
        for j, name in enumerate(names):
            x = reg[:, j]
            nreg = int(np.isfinite(x).sum())
            bm = boot_median(x, bidx)
            regb[tg, pop, name] = (bm, nreg, float(np.nanmedian(x)) if nreg else np.nan)
            ks = MERGED[name] if name in MERGED else (BINS.index(name),)
            nr = nrow[:, list(ks)].sum(1)
            nsh = int((nr > 0).sum())
            o = {"regime_number": ci(regb[tg, pop, name][2], bm, 0.0, nreg),
                 "shots_with_rows": nsh, "n_rows": int(nr.sum())}
            sp = sse[:, list(ks), 0].sum(1)
            with np.errstate(all="ignore"):
                bp = counts @ sp
                for a_i, arm in enumerate(ARMS[1:], start=1):
                    sa = sse[:, list(ks), a_i].sum(1)
                    bs = 1.0 - (counts @ sa) / bp
                    skb[arm, name] = bs
                    o[f"skill_{arm}"] = ci(1.0 - sa.sum() / sp.sum() if sp.sum() > 0 else np.nan,
                                           bs, 0.0, nsh)
                cs = chi[:, list(ks)].sum(1)
                o["chi2_mean_causal_gp"] = ci(cs.sum() / nr.sum() if nr.sum() > 0 else np.nan,
                                              (counts @ cs) / (counts @ nr), 1.5, nsh)
                o["rmse_persistence"] = float(np.sqrt(sp.sum() / nr.sum())) if nr.sum() > 0 else None
            cell["bins"][name] = o
        # differences
        diffs = {}
        for name in (EDGE_A, EDGE_B):
            bm, nreg, pt = regb[tg, pop, name]
            bmr, nref, ptr = regb[tg, pop, REF]
            diffs[f"regime_{name}_minus_{REF}"] = ci(pt - ptr, bm - bmr, 0.0, min(nreg, nref))
            for arm in ("causal_gp", "rm15", "rm40", "running_mean", "slope_filter"):
                a_pt = cell["bins"][name][f"skill_{arm}"]["point"]
                c_pt = cell["bins"][CORE][f"skill_{arm}"]["point"]
                nsh = min(cell["bins"][name]["shots_with_rows"], cell["bins"][CORE]["shots_with_rows"])
                diffs[f"skill_{arm}_{name}_minus_core"] = ci(
                    None if a_pt is None or c_pt is None else a_pt - c_pt,
                    skb[arm, name] - skb[arm, CORE], 0.0, nsh)
        cell["differences"] = diffs
        out["cells"][f"{tg}|{pop}"] = cell
        log(f"{label}: {tg} {pop} cells {ok_cell}")

    def cellget(tg, pop):
        return out["cells"][f"{tg}|{pop}"]

    def below(e):
        return e["side"] == "below"

    def above(e):
        return e["side"] == "above"

    cw, cg = cellget("T_i", "W"), cellget("T_i", "G")
    dA = cw["differences"][f"regime_{EDGE_A}_minus_{REF}"]
    dB = cw["differences"][f"regime_{EDGE_B}_minus_{REF}"]
    out["statements"]["E2"] = {"T_i_W": {EDGE_A: dA, EDGE_B: dB}, "holds": below(dA) and below(dB),
                               "T_i_G_co_reported": {
                                   EDGE_A: cg["differences"][f"regime_{EDGE_A}_minus_{REF}"],
                                   EDGE_B: cg["differences"][f"regime_{EDGE_B}_minus_{REF}"]}}
    rb = cw["bins"][EDGE_B]["regime_number"]
    out["statements"]["E2b"] = {"T_i_W_regime_0.9-1": rb,
                                "holds": rb["side"] in ("includes", "below")}
    g3 = {nm: cg["differences"][f"skill_causal_gp_{nm}_minus_core"] for nm in (EDGE_A, EDGE_B)}
    w3 = {nm: cw["differences"][f"skill_causal_gp_{nm}_minus_core"] for nm in (EDGE_A, EDGE_B)}
    out["statements"]["E3"] = {"T_i_G": g3, "holds": all(below(v) for v in g3.values()),
                               "T_i_W_co_reported": w3}
    d4 = cw["differences"][f"skill_rm15_{EDGE_B}_minus_core"]
    s4 = cw["bins"][EDGE_B]["skill_rm15"]
    out["statements"]["E4"] = {
        "T_i_W_rm15_0.9-1_minus_core": d4, "T_i_W_rm15_0.9-1": s4,
        "holds": above(d4) and s4["side"] in ("includes", "above"),
        "co_reported": {"T_i_G_rm15_0.9-1_minus_core": cg["differences"][f"skill_rm15_{EDGE_B}_minus_core"],
                        "T_i_G_rm15_0.9-1": cg["bins"][EDGE_B]["skill_rm15"],
                        "T_i_W_rm40_0.9-1_minus_core": cw["differences"][f"skill_rm40_{EDGE_B}_minus_core"],
                        "T_i_W_rm40_0.9-1": cw["bins"][EDGE_B]["skill_rm40"]}}
    vw = cellget("V_i", "W")
    vA = vw["differences"][f"regime_{EDGE_A}_minus_{REF}"]
    vB = vw["differences"][f"regime_{EDGE_B}_minus_{REF}"]
    out["statements"]["E5"] = {"V_i_W": {EDGE_A: vA, EDGE_B: vB}, "holds": below(vA) and below(vB)}
    return out


# --------------------------------------------------------------------------- output

def f(e, nd=3, signed=True):
    if e is None or e.get("point") is None:
        return "n/a"
    fm = "{:" + ("+" if signed else "") + "." + str(nd) + "f}"
    if e.get("lo") is None:
        return f"{fm.format(e['point'])} (too few)"
    return f"{fm.format(e['point'])} [{fm.format(e['lo'])}, {fm.format(e['hi'])}]"


def sd(e):
    if e is None or e.get("point") is None:
        return "n/a"
    if e["side"] == "too few":
        return "too few shots"
    ref = f"{e['ref']:g}"
    return {"above": f"excludes {ref} (above)", "below": f"excludes {ref} (below)",
            "includes": f"includes {ref}"}[e["side"]]


def render_set(a, L, brief=False):
    L.append(f"Shots analysed: {a['n_shots']}; with a population-W beam block: "
             f"{a['pre_rule']['shots_with_W_block']}.")
    L.append("")
    L.append("| rho | shots | finite samples | fail share, cut 0.1 | fail share, cut 0.5 | median relative T_i error | median V_i error (km/s) |")
    L.append("|---|---|---|---|---|---|---|")
    for nm in a["bins"]:
        o = a["pre_rule"]["bins"][nm]
        L.append(f"| {nm} | {o['shots']} | {o['finite_samples']} | {f(o['fail_share_0p1'], 3, False)} | "
                 f"{f(o['fail_share_0p5'], 3, False)} | {f(o.get('median_rel_Ti_error'), 3, False)} | "
                 f"{f(o.get('median_Vi_error_kms'), 2, False)} |")
    L.append("")
    for key in ([f"{tg}|{pop}" for tg, pop in COMBOS] if not brief else ["T_i|W", "V_i|W"]):
        c = a["cells"][key]
        L.append(f"**{key.replace('|', ', population ')}** - shots with a block and scored chords: "
                 f"{c['shots_with_block_and_chords']}")
        L.append("")
        L.append("| rho | regime number (ref 0) | shots | rows | causal GP skill | slope filter skill | running mean tau 2.5 ms | running mean tau 15 ms | running mean tau 40 ms | causal GP mean chi-square (ref 1.5) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for nm in a["bins"]:
            o = c["bins"][nm]
            L.append(f"| {nm} | {f(o['regime_number'])} | {o['regime_number'].get('n_shots', '')} | "
                     f"{o['n_rows']} | {f(o['skill_causal_gp'])} | {f(o['skill_slope_filter'])} | "
                     f"{f(o['skill_running_mean'])} | {f(o['skill_rm15'])} | {f(o['skill_rm40'])} | "
                     f"{f(o['chi2_mean_causal_gp'], 2, False)} |")
        L.append("")


def render_statements(a, L):
    s = a["statements"]
    L.append("| statement | read-out | value | side | verdict |")
    L.append("|---|---|---|---|---|")
    for nm, e in s["E1"]["differences_vs_0.5-0.7"].items():
        L.append(f"| E1 (primary) | fail share {nm} minus 0.5-0.7 | {f(e)} | {sd(e)} | "
                 f"{'holds' if s['E1']['holds'] else 'does not hold'} |")
    for nm, e in s["E2"]["T_i_W"].items():
        L.append(f"| E2 (primary) | T_i regime number (W) {nm} minus 0.5-0.7 | {f(e)} | {sd(e)} | "
                 f"{'holds' if s['E2']['holds'] else 'does not hold'} |")
    for nm, e in s["E2"]["T_i_G_co_reported"].items():
        L.append(f"| E2, co-reported | T_i regime number (G) {nm} minus 0.5-0.7 | {f(e)} | {sd(e)} | - |")
    e = s["E2b"]["T_i_W_regime_0.9-1"]
    L.append(f"| E2b | T_i regime number (W) in 0.9-1 | {f(e)} | {sd(e)} | "
             f"{'holds' if s['E2b']['holds'] else 'does not hold'} |")
    for nm, e in s["E3"]["T_i_G"].items():
        L.append(f"| E3 | causal GP skill (T_i, G) {nm} minus core | {f(e)} | {sd(e)} | "
                 f"{'holds' if s['E3']['holds'] else 'does not hold'} |")
    for nm, e in s["E3"]["T_i_W_co_reported"].items():
        L.append(f"| E3, co-reported | causal GP skill (T_i, W) {nm} minus core | {f(e)} | {sd(e)} | - |")
    e = s["E4"]["T_i_W_rm15_0.9-1_minus_core"]
    L.append(f"| E4 | running mean 15 ms skill (T_i, W) 0.9-1 minus core | {f(e)} | {sd(e)} | "
             f"{'holds' if s['E4']['holds'] else 'does not hold'} |")
    e = s["E4"]["T_i_W_rm15_0.9-1"]
    L.append(f"| E4 | running mean 15 ms skill (T_i, W) in 0.9-1 | {f(e)} | {sd(e)} | same |")
    for k, e in s["E4"]["co_reported"].items():
        L.append(f"| E4, co-reported | {k} | {f(e)} | {sd(e)} | - |")
    for nm, e in s["E5"]["V_i_W"].items():
        L.append(f"| E5 | V_i regime number (W) {nm} minus 0.5-0.7 | {f(e)} | {sd(e)} | "
                 f"{'holds' if s['E5']['holds'] else 'does not hold'} |")
    L.append("")


def render_md(res):
    L = ["# MAST edge confirmation - fit-failure share, regime number and CES-only skill by normalized radius",
         "",
         f"Generated by `ces_prediction/experiments/mast/run_edge.py`. Registration: `{REG}` "
         "(committed before any value of these shots was read). The hypothesis was read post hoc in "
         "the 300-shot gate (`mast_gate_addendum2.md`); the shots here took no part in it. "
         "Hyperparameters are those selected on the gate TUNE shots (addendum 1); nothing is retuned. "
         "Every interval is a 95 % interval from 2,000 bootstrap resamples of discharges.",
         "",
         "## 1. Counts", ""]
    c = res["counts"]
    L.append("| set | listed | with file | unusable (frame spacing) | no geometry | radius columns mismatch | analysed |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("Not Checked", "Bad"):
        o = c[k]
        L.append(f"| {k} | {o['listed']} | {o['with_file']} | {o['unusable']} | {o['no_geometry']} | "
                 f"{o['radius_columns_mismatch']} | {o['ok']} |")
    L.append("")
    L.append(f"Geometry source among analysed primary shots: {json.dumps(c['geometry_source'])}. "
             f"Campaigns (primary, analysed): {json.dumps(c['campaigns'])}.")
    L.append("")
    L.append("## 2. Statements (primary set: catalogue quality Not Checked)")
    L.append("")
    render_statements(res["primary"], L)
    L.append("## 3. Read-outs by normalized radius (primary set)")
    L.append("")
    render_set(res["primary"], L)
    L.append("## 4. Expectation check")
    L.append("")
    L.append("| expectation | found | met |")
    L.append("|---|---|---|")
    for e in res["expectations"]:
        L.append(f"| {e['text']} | {e['found']} | {'yes' if e['met'] else 'no'} |")
    L.append("")
    L.append("## 5. Per campaign (primary set; statements re-read, not separately registered)")
    L.append("")
    for cp in CAMPS:
        if cp not in res["campaigns"]:
            continue
        L.append(f"### {cp}")
        L.append("")
        render_statements(res["campaigns"][cp], L)
        render_set(res["campaigns"][cp], L, brief=True)
    L.append("## 6. Catalogue quality Bad (co-reported)")
    L.append("")
    if res.get("bad"):
        render_statements(res["bad"], L)
        render_set(res["bad"], L, brief=True)
    else:
        L.append("No analysable shot.")
        L.append("")
    L.append("## 7. Notes")
    L.append("")
    for n in res["notes"]:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


def expectations(res):
    p = res["primary"]
    pre = p["pre_rule"]["bins"]
    tw, tg_, vw = p["cells"]["T_i|W"]["bins"], p["cells"]["T_i|G"]["bins"], p["cells"]["V_i|W"]["bins"]
    out = []

    def add(text, val, lo, hi, nd=3):
        ok = val is not None and lo <= val <= hi
        out.append({"text": text, "found": "n/a" if val is None else f"{val:+.{nd}f}", "met": bool(ok)})

    out.append({"text": "analysed primary shots with a W block: 2,000 to 3,000",
                "found": str(p["pre_rule"]["shots_with_W_block"]),
                "met": 2000 <= p["pre_rule"]["shots_with_W_block"] <= 3000})
    for nm, lo, hi in (("0.5-0.7", 0.22, 0.30), ("0.8-0.9", 0.38, 0.52), ("0.9-0.95", 0.50, 0.68),
                       ("0.95-1", 0.62, 0.85)):
        add(f"fail share (0.1) in {nm}: {lo} to {hi}", pre[nm]["fail_share_0p1"]["point"], lo, hi)
    for nm, lo, hi in (("0.5-0.7", 0.25, 0.40), ("0.8-0.9", 0.02, 0.22), ("0.9-1", -0.25, 0.05)):
        add(f"T_i regime number W in {nm}: {lo:+} to {hi:+}", tw[nm]["regime_number"]["point"], lo, hi)
    add("T_i regime number G in 0.8-0.9: +0.0 to +0.25", tg_["0.8-0.9"]["regime_number"]["point"], 0.0, 0.25)
    for key in ("E1", "E2", "E2b", "E3", "E5"):
        out.append({"text": f"{key} holds", "found": "holds" if p["statements"][key]["holds"] else "does not hold",
                    "met": bool(p["statements"][key]["holds"])})
    add("causal GP skill (T_i, G) in 0.8-0.9: -0.15 to +0.08", tg_["0.8-0.9"]["skill_causal_gp"]["point"], -0.15, 0.08)
    add("causal GP skill (T_i, G) in the core: +0.08 to +0.16", tg_["0-0.7"]["skill_causal_gp"]["point"], 0.08, 0.16)
    add("running mean 15 ms (T_i, W) in the core: -0.45 to -0.20", tw["0-0.7"]["skill_rm15"]["point"], -0.45, -0.20)
    add("running mean 15 ms (T_i, W) in 0.9-1: -0.15 to +0.10", tw["0.9-1"]["skill_rm15"]["point"], -0.15, 0.10)
    e4 = p["statements"]["E4"]
    out.append({"text": "E4 difference above 0 and the 0.9-1 interval includes 0",
                "found": f"difference {sd(e4['T_i_W_rm15_0.9-1_minus_core'])}; edge {sd(e4['T_i_W_rm15_0.9-1'])}",
                "met": e4["T_i_W_rm15_0.9-1_minus_core"]["side"] == "above"
                and e4["T_i_W_rm15_0.9-1"]["side"] == "includes"})
    for nm, lo, hi in (("0.5-0.7", 0.35, 0.50), ("0.8-0.9", 0.05, 0.30), ("0.9-1", -0.05, 0.25)):
        add(f"V_i regime number W in {nm}: {lo:+} to {hi:+}", vw[nm]["regime_number"]["point"], lo, hi)
    return out


def main():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    last = FETCH_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
    if not last.startswith("done"):
        raise SystemExit("download not finished: " + last[:80])
    gate = {int(x) for x in GATE_TXT.read_text().split()}
    listed = [int(x) for x in ALL_TXT.read_text().split()]
    assert not (set(listed) & gate) and not (set(listed) & PILOT)
    src = pd.read_parquet(SOURCES, columns=["shot_id", "name", "quality"])
    qual = dict(src[src.name == "act"][["shot_id", "quality"]].itertuples(index=False, name=None))
    cat = pd.read_parquet(rg.META, columns=["shot_id", "campaign", "radii_magnetic__radius_efit",
                                            "radii_router_radius_efit", "radii_router_radius_d_alpha"])
    cat = cat.set_index("shot_id")
    add1 = json.loads(ADD1_JSON.read_text(encoding="utf-8"))
    params = {tg: params_from(add1, tg) for tg in rg.TARGETS}

    def geo_of(s):
        if s not in cat.index:
            return None, "missing"
        ax = float(cat.loc[s, "radii_magnetic__radius_efit"])
        ro = float(cat.loc[s, "radii_router_radius_efit"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            return (ax, ro), "efit"
        ro = float(cat.loc[s, "radii_router_radius_d_alpha"])
        if np.isfinite(ax) and np.isfinite(ro) and ro > ax:
            return (ax, ro), "d_alpha_outer"
        return None, "missing"

    recs, counts = {}, {}
    gsrc = {"efit": 0, "d_alpha_outer": 0}
    for q in ("Not Checked", "Bad"):
        counts[q] = {"listed": 0, "with_file": 0, "unusable": 0, "no_geometry": 0,
                     "radius_columns_mismatch": 0, "ok": 0}
    for i, s in enumerate(listed):
        q = qual.get(s, "Not Checked")
        if q not in counts:
            continue
        counts[q]["listed"] += 1
        if not (rg.ACT_DIR / f"{s}.npz").exists():
            continue
        counts[q]["with_file"] += 1
        geo, gs = geo_of(s)
        r = process(s, geo, params)
        if r["status"] == "ok":
            counts[q]["ok"] += 1
            recs[s] = r
            if q == "Not Checked":
                gsrc[gs] += 1
        elif r["status"] in counts[q]:
            counts[q][r["status"]] += 1
        if (i + 1) % 500 == 0:
            log(f"{i + 1}/{len(listed)} shots scanned, analysed {len(recs)}")
    prim = sorted(s for s in recs if qual.get(s, "Not Checked") == "Not Checked")
    bad = sorted(s for s in recs if qual.get(s) == "Bad")
    camp = {s: str(cat.loc[s, "campaign"]) for s in recs}
    counts["geometry_source"] = gsrc
    counts["campaigns"] = {c: int(sum(camp[s] == c for s in prim)) for c in CAMPS}
    log(f"analysed primary {len(prim)}, bad {len(bad)}")

    res = {"registration": REG, "seed": rg.SEED, "n_boot": rg.N_BOOT, "bin_edges": list(EDGES),
           "hyperparameters": {tg: {k: (list(v) if isinstance(v, tuple) else v) for k, v in params[tg].items()}
                               for tg in rg.TARGETS},
           "counts": counts}
    res["primary"] = analyse(recs, prim, "primary", log)
    res["campaigns"] = {}
    for c in CAMPS:
        ss = [s for s in prim if camp[s] == c]
        if len(ss) >= MIN_SHOTS:
            res["campaigns"][c] = analyse(recs, ss, c, log)
    res["bad"] = analyse(recs, bad, "bad", log) if len(bad) >= MIN_SHOTS else None
    res["expectations"] = expectations(res)
    res["notes"] = [
        "Skill, regime number and chi-square use the scored chords and scored rows of each target and "
        "population (rg.prepare); the pre-rule read-outs use every low-field chord with a finite T_i "
        "sample inside the population-W beam block.",
        "The fixed running means (tau 15 ms, 40 ms) and the tuned one are exponentially weighted means "
        "of the visible samples of the chord; tau = 2.5 ms is the persistence limit.",
        "Merged bins 0.9-1 and 0-0.7: sums of the bin sums for skill and fail share; the regime number "
        "is recomputed over the chords of the merged bin.",
        "rho is a geometric midplane ratio from one EFIT scalar pair per discharge; the catalogue has no "
        "confinement-mode label, so the statements are about normalized radius, not about a pedestal.",
        "Per-campaign and Bad-quality tables re-read the statements on subsets; only the primary set "
        "is the registered test.",
    ]
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    log(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    p = res["primary"]["statements"]
    print("E1", p["E1"]["holds"], "E2", p["E2"]["holds"], "E2b", p["E2b"]["holds"],
          "E3", p["E3"]["holds"], "E4", p["E4"]["holds"], "E5", p["E5"]["holds"])


if __name__ == "__main__":
    main()
