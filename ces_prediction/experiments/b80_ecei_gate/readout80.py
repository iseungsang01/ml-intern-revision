"""B.80 readout (registration sections 4-5, Addendum 1 sections 3-6): every readout and statement in one pass into the
JSON, then the markdown.  No model is fitted: stored estimates of B.75 and B.79 and the input-only measures of
measures80.py.  The progress log prints row counts only.

    py ces_prediction/experiments/b80_ecei_gate/readout80.py
Output: docs/paper/b80_ecei_gate.json / .md
"""
from __future__ import annotations

import json
import time
import warnings

import check80 as CK
import common80 as M
import measures80 as ME

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

warnings.filterwarnings("ignore")
C = M.C
N = M.N_BOOT
FORMS = M.FORMS
BLOCK_NAMES = [f"block {b[0]}" for b in C.BLOCKS]
EARLY = "blocks up to 32310"
TAGS = {0: "low", 2: "high"}


def learners(d):
    return ("T0", "TX", "TN", "TXN") + (("TB", "TE") if d in M.HALF_DESIGNS else ())


def skill_pairs(d):
    return [("TX", "T0"), ("TXN", "TN")] + ([("TB", "T0"), ("TE", "T0")] if d in M.HALF_DESIGNS else [])


# ------------------------------------------------------------------------------------------ strata
def strata_of(d, cls_r, blk_r):
    one = np.ones(len(cls_r), bool)
    S = {"all": one}

    def add_classes(prefix, base):
        for k, nm in enumerate(M.CLASS_NAMES):
            m = base & (cls_r == k)
            if m.any():
                S[f"{prefix}class {nm}"] = m
        m = base & (cls_r == M.NO_CLASS)
        if m.any():
            S[f"{prefix}no class"] = m
        lo, hi = base & (cls_r == 0), base & (cls_r == 2)
        if lo.any() and hi.any():
            S[f"{prefix}low+high"] = lo | hi

    add_classes("", one)
    if d == "V1":
        for b, nm in enumerate(BLOCK_NAMES):
            base = blk_r == b
            if base.any():
                S[nm] = base
                add_classes(nm + "|", base)
        base = blk_r <= 1
        S[EARLY] = base
        add_classes(EARLY + "|", base)
    return S


# ------------------------------------------------------------------------------------------ quantities
def diff_pack(g, s, a, b):
    """(1 - SSE_high(a)/SSE_high(b)) - (1 - SSE_low(a)/SSE_low(b)) draw by draw; quantiles over finite draws."""
    fn = lambda A1, B1, A2, B2: (1 - A1 / B1) - (1 - A2 / B2)   # noqa: E731
    es = (f"{a}@high", f"{b}@high", f"{a}@low", f"{b}@low")
    with np.errstate(invalid="ignore", divide="ignore"):
        v = g._f(s, fn, *es)
        v["draws_not_finite"] = {}
        for form in FORMS:
            t = g.tot[(s, form)]
            x = fn(*[t[e][1] for e in es])
            v["draws_not_finite"][form] = int((~np.isfinite(x)).sum())
    return v


def stratum_block(g, s, d):
    o = {"counts": g.counts(s), "rmse_eV": {L: g.rmse(s, L) for L in learners(d) + ("TG",)}}
    for a, b in skill_pairs(d):
        o[f"{a}/{b}"] = g.skill(s, a, b)
    o["TG/T0"] = g.skill(s, "TG", "T0")
    o["TG/TX"] = g.skill(s, "TG", "TX")
    if s.endswith("low+high"):
        for a, b in skill_pairs(d):
            o[f"high minus low: {a}/{b}"] = diff_pack(g, s, a, b)
    return o


def cell(d, kind, R, sets, us, cls_row, unit_d, unit_c):
    r = M.RULE_OF[kind]
    subs = M.subs_of(us, d)
    n = len(R["y"])
    scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
    rows = np.flatnonzero(scored & sets[f"{kind}_L5"])
    cr, br = cls_row[rows], R["blk"][rows]
    strata = strata_of(d, cr, br)
    g = M.Group(f"{d}|L5|{kind}", rows, strata, unit_d, unit_c, len(subs))
    y = R["tg_L5"][rows]
    gate = cr >= 1                               # C_E of 0.7 or above: TX; otherwise (low, no C_E): T0
    for si, units in enumerate(subs):
        se = {}
        for L in learners(d):
            se[L] = (M.load_est(units, L, r, n)[rows] - y) ** 2
        se["TG"] = np.where(gate, se["TX"], se["T0"])
        for L, v in se.items():
            g.add(L, si, v)
        for L in learners(d):
            for k, tag in TAGS.items():
                g.add(f"{L}@{tag}", si, se[L] * (cr == k))
    out = {}
    for s in strata:
        g.finish(s)
        out[s] = stratum_block(g, s, d)
        g.tot.clear()
    M.log(f"{d}|{r}: rows {len(rows):,}, strata {len(strata)} done")
    return out


# ------------------------------------------------------------------------------------------ Spearman
def spearman_w(x, z, c):
    """Spearman correlation of (x, z) with each pair entering c times (average ranks for ties)."""
    xs, zs = np.repeat(x, c), np.repeat(z, c)
    rx, rz = rankdata(xs), rankdata(zs)
    rx -= rx.mean()
    rz -= rz.mean()
    den = np.sqrt((rx * rx).sum() * (rz * rz).sum())
    return float((rx * rz).sum() / den) if den > 0 else np.nan


def spearman_section(R, sets, us, ce_file):
    n = len(R["y"])
    y = R["tg_L5"]
    rows = np.flatnonzero(sets["S_L5"])
    subs = M.subs_of(us, "V1")
    scored = np.isin(R["fidx"], np.concatenate([u["test"] for u in subs[0]]))
    if not np.all(scored[rows]):
        raise SystemExit("FATAL: an S-set row not scored under V1")
    ud, first, inv = np.unique(R["fidx"][rows], return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    ce_d = ce_file[ud]
    blk_d, clus_d = R["blk"][rows][first], R["clus"][rows][first]
    sk = np.zeros((len(subs), len(ud)))
    for si, units in enumerate(subs):
        t0 = M.load_est(units, "T0", "S", n)[rows]
        tx = M.load_est(units, "TX", "S", n)[rows]
        s0 = np.bincount(inv, (t0 - y[rows]) ** 2, len(ud))
        sx = np.bincount(inv, (tx - y[rows]) ** 2, len(ud))
        sk[si] = 1.0 - sx / s0
    elig = (cnt >= M.MIN_SCORED) & np.isfinite(ce_d)
    out = {"population": "D1 discharges with C_E and at least 50 S-set rows (L5); per-discharge skill of TX over T0 "
                          "under S per re-partition; point = mean over the three re-partitions",
           "discharges with C_E but fewer than 50 S-set rows": int(((cnt < M.MIN_SCORED) & np.isfinite(ce_d)).sum()),
           "discharges with at least 50 S-set rows but no C_E": int(((cnt >= M.MIN_SCORED) & ~np.isfinite(ce_d)).sum())}
    for st, dm in [("all blocks", np.ones(len(ud), bool))] + [(BLOCK_NAMES[b], blk_d == b) for b in range(3)]:
        D = np.flatnonzero(elig & dm)
        x = ce_d[D]
        pts = [spearman_w(x, sk[si, D], np.ones(len(D), np.int64)) for si in range(len(subs))]
        o = {"discharges": int(len(D)), "clusters": int(len(np.unique(clus_d[D]))), "point": float(np.mean(pts)),
             "per re-partition": [float(p) for p in pts],
             "median per-discharge skill (mean over re-partitions)": float(np.median(sk[:, D].mean(0)))}
        for form in FORMS:
            if form == "discharge":
                uof = np.arange(len(D))
            else:
                _, uof = np.unique(clus_d[D], return_inverse=True)
            nu = int(uof.max()) + 1
            W = M.R75.draws(f"V1|L5|S|spearman|{st}|{form}", len(subs), nu)
            vals = np.empty(N)
            for si, (sel, Wm) in enumerate(W):
                z = sk[si, D]
                for k, dr in enumerate(sel):
                    vals[dr] = spearman_w(x, z, Wm[k][uof].astype(np.int64))
            fin = vals[np.isfinite(vals)]
            o[form] = [float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))]
            o[f"draws_not_finite_{form}"] = int((~np.isfinite(vals)).sum())
        out[st] = o
        M.log(f"Spearman {st}: discharges {len(D)} done")
    return out


# ------------------------------------------------------------------------------------------ statements
def statements(res):
    S = {}

    def get(d, r, s, key):
        c = res["cells"][f"{d}|{r}"]
        return c[s][key] if s in c and key in c[s] else None

    def put(k, text, v, rule, exp, note=""):
        if v is None:
            S[k] = {"text": text, "value": None, "holds": None, "formed": False, "expected": exp,
                    "note": "not formed: the stratum has no rows" + ("; " + note if note else "")}
            return
        val = {kk: v[kk] for kk in ("point", "discharge", "cluster")}
        if "draws_not_finite" in v:
            val["draws_not_finite"] = v["draws_not_finite"]
        S[k] = {"text": text, "value": val, "holds": bool(rule(v)), "formed": True, "expected": exp, "note": note}

    def ivs(v):
        return "not formed" if v is None else f"{v['point']:+.3f}; c [{v['cluster'][0]:+.3f}, {v['cluster'][1]:+.3f}]"

    above = lambda v: v["cluster"][0] > 0          # noqa: E731
    not_above = lambda v: v["cluster"][0] <= 0     # noqa: E731
    not_below = lambda v: v["cluster"][1] >= 0     # noqa: E731
    dk = "high minus low: TX/T0"

    def cls_note(d, r, prefix=""):
        return (f"high {ivs(get(d, r, prefix + 'class high', 'TX/T0'))}; low {ivs(get(d, r, prefix + 'class low', 'TX/T0'))}; "
                f"middle {ivs(get(d, r, prefix + 'class middle', 'TX/T0'))}")

    put("G1", "V1, S: TX over T0 in the high class minus that in the low class lies above zero",
        get("V1", "S", "low+high", dk), above, "holds", cls_note("V1", "S"))
    put("G2", "V1, S: TE over T0 in the low class has an interval that contains zero or lies below it (the ECEI half adds "
              "nothing where its channels are not coherent)", get("V1", "S", "class low", "TE/T0"), not_above, "holds",
        f"TB over T0 in the low class: {ivs(get('V1', 'S', 'class low', 'TB/T0'))}")
    put("G3", "V1, S: TE over T0 in the high class lies above zero", get("V1", "S", "class high", "TE/T0"), above, "holds",
        f"TB over T0 in the high class: {ivs(get('V1', 'S', 'class high', 'TB/T0'))}")
    put("G4", "V1, D650: TX over T0 in the high class minus that in the low class lies above zero",
        get("V1", "D650", "low+high", dk), above, "holds", cls_note("V1", "D650"))
    put("G5", "V1, S, blocks up to 32310 only: TX over T0 in the high class minus that in the low class lies above zero "
              "(the ordering holds inside the earlier blocks, not only between blocks)",
        get("V1", "S", f"{EARLY}|low+high", dk), above, "uncertain, leaning to holds", cls_note("V1", "S", f"{EARLY}|"))
    put("G6", "V3, S: TX over T0 in the high class lies above zero", get("V3", "S", "class high", "TX/T0"), above, "holds",
        f"low class: {ivs(get('V3', 'S', 'class low', 'TX/T0'))}")
    put("G7", "V1, S, block from 32312: TG over T0 lies above zero", get("V1", "S", BLOCK_NAMES[2], "TG/T0"), above,
        "uncertain", f"TX over T0 in the block: {ivs(get('V1', 'S', BLOCK_NAMES[2], 'TX/T0'))}; "
                     f"TG over TX: {ivs(get('V1', 'S', BLOCK_NAMES[2], 'TG/TX'))}")
    put("G8", "V4 (fit to 31991), S: TG over T0 lies above zero", get("V4a", "S", "all", "TG/T0"), above, "uncertain",
        f"V4b (fit to 32310): {ivs(get('V4b', 'S', 'all', 'TG/T0'))}; V4a TX over T0: {ivs(get('V4a', 'S', 'all', 'TX/T0'))}")
    put("G9", "V1, S, all blocks: TG is not worse than TX (the interval of TG over TX is not below zero)",
        get("V1", "S", "all", "TG/TX"), not_below, "holds", f"TG over T0: {ivs(get('V1', 'S', 'all', 'TG/T0'))}")
    return S


# ------------------------------------------------------------------------------------------ main
def main():
    t_start = time.time()
    R = M.rows()
    sets = M.sets()
    us, _, _ = M.DS.units(R)
    meas = M.load_npz(M.CACHE / "measures.npz")
    cls_row = meas["cls"][R["fidx"]]
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED,
           "estimates": {"B.75": "data/.b75/fits/", "B.79": "data/.b79/fits/ (main run, beam columns as delivered)"},
           "classes": {"low": "C_E < 0.7", "middle": "0.7 <= C_E < 0.9", "high": "C_E >= 0.9",
                       "gate": "TG = TX where C_E >= 0.7, T0 otherwise (also where C_E is undefined)"},
           "distribution": ME.distribution(meas, np.unique(R["fidx"])),
           "check_stage": json.loads((M.CACHE / "check.json").read_text(encoding="utf-8")),
           "cells": {}}
    for d in M.DESIGNS:
        for kind in ("S", "common"):
            res["cells"][f"{d}|{M.RULE_OF[kind]}"] = cell(d, kind, R, sets, us, cls_row, unit_d, unit_c)
    # the check of section 8 recomputed in this pass
    md = CK.published_md()
    chk = []
    for d in M.DESIGNS:
        for r in ("D650", "S"):
            v = res["cells"][f"{d}|{r}"]["all"]["TX/T0"]
            chk.append({"design": d, "rule": r, "point": v["point"], "published_md": md[(d, r)],
                        "pass": f"{v['point']:+.3f}" == f"{md[(d, r)]:+.3f}"})
    res["check_in_readout"] = {"cells": chk, "all_pass": all(c["pass"] for c in chk)}
    if not res["check_in_readout"]["all_pass"]:
        raise SystemExit("FATAL: the check of section 8 fails in the readout pass")
    res["spearman"] = spearman_section(R, sets, us, meas["ce"])
    res["statements"] = statements(res)
    res["expectations"] = M.expectation_met(res["statements"])
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    RR = M.load_by_path("render80", M.HERE / "render80.py")
    M.save_text(M.OUT_MD, RR.render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


if __name__ == "__main__":
    main()
