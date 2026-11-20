"""B.84 readout (registration sections 4-5, Addendum 1 section 5): every table and statement in one pass, then the
markdown.  Refuses to run unless the four checks have passed.

    py ces_prediction/experiments/b84_onefit/readout84.py
Output: docs/paper/b84_onefit.json / .md
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings

import common84 as M

import numpy as np  # noqa: E402

sys.path.insert(0, str(M.HERE.parent / "b81_controls"))
sys.path.insert(0, str(M.HERE.parent / "b82_localize"))
import boot82 as BT  # noqa: E402  (draws, q, Units; cluster form drawn first)

warnings.filterwarnings("ignore")
C = M.C
K75 = M.K75
N = C.N_BOOT
DESIGNS = {"V1": tuple(f"V1_r2_k{k}" for k in range(5)), "V3": tuple(f"V3_q{q}" for q in range(1, 5)), "V4a": ("V4a",)}
DLAB = {"V1": "V1 random, re-partition r2 (five folds pooled)", "V3": "V3 rolling origin (fifths 2-5 pooled)",
        "V4a": "V4a chronological: fit to 31991, score from 32312"}


def est_list(r):
    out = ["T0", "TX"]
    if r not in ("D10", "K10"):
        out += ["T0a", "TXa"]
    out += list(M.MIX_LEARNERS)
    if r in M.REL_RULES:
        out.append("TXrel")
    return out


def est_path(un, L, r):
    if L in ("T0", "TX", "T0a", "TXa"):
        return (M.M78.FITS / f"{un}__{L}__K10__L5.npz") if r == "K10" else (K75.FITS / f"{un}__{L}__{r}__L5.npz")
    return M.FITS / f"{un}__{L}__{r}.npz"


def skill_pairs(r, have):
    pairs = [("T0all", "T0"), ("TXall", "TX"), ("T0raw", "T0"), ("TXraw", "TX"), ("TXall", "T0all"), ("T0raw", "T0all"),
             ("TXraw", "T0all"), ("TXraw", "TXall"), ("TX", "T0")]
    if "TXa" in have:
        pairs += [("TXall", "TXa"), ("T0all", "T0a"), ("TXraw", "TXa")]
    if "TXrel" in have:
        pairs += [("TXrel", "TX"), ("TXrel", "T0"), ("TXrel", "TXall")]
    return pairs


def boot(stream, U, vecs, stat):
    """Interval of stat(per-unit sums) by the cluster and the discharge draws of one stream (cluster first)."""
    rng = M.rng_for(stream)
    res = {}
    for form in BT.FORMS:
        inv, nu = U.idx[form]
        (sel, Wm), = BT.draws(rng, 1, nu)
        sums = {k: np.bincount(inv, v, nu) for k, v in vecs.items()}
        vals = stat({k: Wm @ s for k, s in sums.items()})
        res[form] = vals
        del Wm
    pt = stat({k: np.array([v.sum()]) for k, v in vecs.items()})
    return {"point": float(pt[0]), "discharge": BT.q(res["discharge"]), "cluster": BT.q(res["cluster"]), "stream": stream}


def boot_multi(stream, U, vecs, stats):
    """Several statistics on the same draws of one stream: stats = {name: fn}."""
    rng = M.rng_for(stream)
    res = {k: {} for k in stats}
    for form in BT.FORMS:
        inv, nu = U.idx[form]
        (sel, Wm), = BT.draws(rng, 1, nu)
        S = {k: Wm @ np.bincount(inv, v, nu) for k, v in vecs.items()}
        for nm, fn in stats.items():
            res[nm][form] = BT.q(fn(S))
        del Wm
    P = {k: np.array([v.sum()]) for k, v in vecs.items()}
    return {nm: {"point": float(fn(P)[0]), "discharge": res[nm]["discharge"], "cluster": res[nm]["cluster"], "stream": stream}
            for nm, fn in stats.items()}


class Cell:
    def __init__(self, d, r, R, unit_d, unit_c, ctx, by):
        self.d, self.r = d, r
        idx, est = [], {}
        self.have = est_list(r)
        for un in DESIGNS[d]:
            te = ctx.rows_of(by[un], r, "test")
            idx.append(te)
            for L in self.have:
                z = np.load(est_path(un, L, r))
                if not np.array_equal(z["idx"], te):
                    raise SystemExit(f"FATAL: {d} {r} {L} {un}: rows differ")
                est.setdefault(L, []).append(z["est"])
        self.rows = np.concatenate(idx)
        self.y = R["tg_L5"][self.rows]
        self.est = {L: np.concatenate(v) for L, v in est.items()}
        for L, v in self.est.items():
            if not np.all(np.isfinite(v)):
                raise SystemExit(f"FATAL: {d} {r} {L}: rows without an estimate")
        self.se = {L: (v - self.y) ** 2 for L, v in self.est.items()}
        self.ae = {L: np.abs(v - self.y) for L, v in self.est.items()}
        self.w10 = {L: (self.ae[L] / self.y <= 0.10).astype(np.float64) for L in self.est}
        self.U = BT.Units(self.rows, unit_d, unit_c)
        self.counts = {"rows": int(len(self.rows)), "discharges": self.U.count("discharge"), "clusters": self.U.count("cluster")}

    def rmse(self):
        return {L: float(np.sqrt(v.mean())) for L, v in self.se.items()}

    def skill(self, a, b, stream):
        return boot(stream, self.U, {"A": self.se[a], "B": self.se[b]}, lambda S: 1 - S["A"] / S["B"])

    def robust(self):
        vecs = {"n": np.ones(len(self.rows))}
        stats = {}
        for L in self.est:
            vecs[f"ae|{L}"] = self.ae[L]
            vecs[f"w|{L}"] = self.w10[L]
            stats[f"{L}|MAE"] = (lambda S, L=L: S[f"ae|{L}"] / S["n"])
            stats[f"{L}|share within 10 %"] = (lambda S, L=L: S[f"w|{L}"] / S["n"])
        res = boot_multi(f"{self.d}|{self.r}|robust", self.U, vecs, stats)
        out = {}
        for k, v in res.items():
            L, m = k.split("|")
            out.setdefault(L, {})[m] = v
        return out


STATEMENTS = (
    # id, design, rule, A, B, judged, text, expectation
    ("U1", "V1", "D650", "TXall", "TX", "gt-0.05", "V1, D650: TXall is not worse than the per-rule TX by more than 5 %",
     "uncertain, leaning to holds"),
    ("U2", "V1", "D10", "TXall", "TX", "gt-0.05", "V1, D10: TXall is not worse than the per-rule TX by more than 5 %",
     "uncertain, leaning to holds"),
    ("U3", "V1", "S", "TXall", "TX", "gt-0.05", "V1, S: TXall is not worse than the per-rule TX by more than 5 %",
     "uncertain"),
    ("U4", "V1", "D650", "TXall", "TXa", "gt0",
     "V1, D650: TXall over TXa (fitted with the previous frame visible, applied) lies above zero", "holds"),
    ("U5", "V1", "D650", "TXall", "T0all", "gt0",
     "V1, D650: TXall over T0all lies above zero (the fast-channel increment inside the one fit)", "holds"),
    ("U6", "V1", "K10", "TXall", "T0", "gt0",
     "V1, K10 (not in the mixture): TXall over the per-rule T0 of K10 lies above zero", "uncertain, leaning to holds"),
    ("U7", "V3", "D650", "TXall", "TX", "gt-0.05", "V3, D650: TXall is not worse than the per-rule TX by more than 5 %",
     "uncertain"),
    ("U8", "V1", "D650", "TXraw", "TXall", "gt-0.05",
     "V1, D650: TXraw is not worse than TXall by more than 5 % (the hand-made scalars are not needed)", "uncertain"),
    ("U9", "V3", "D650", "TXrel", "TX", "gt0",
     "V3, D650: TXrel over the per-rule TX lies above zero (without absolute levels the estimate is better in time order)",
     "uncertain"),
    ("U10", "V4a", "S", "TXrel", "T0", "gt0",
     "V4a, S: TXrel over the per-rule T0 lies above zero (the last block under the schedule)",
     "uncertain, leaning to does not hold"),
    ("U11", "V1", "D650", "TXrel", "TX", "gt-0.05", "V1, D650: TXrel is not worse than the per-rule TX by more than 5 %",
     "uncertain"),
)
JUDGED = {"gt0": "lower end of the cluster interval above zero", "gt-0.05": "lower end of the cluster interval above -0.05"}


def main():
    t_start = time.time()
    chk = {}
    for n in ("reuse", "mixture", "raw", "rows"):
        p = M.CHK / f"{n}.json"
        if not p.exists():
            raise SystemExit(f"check {n} has not been run")
        chk[n] = json.loads(p.read_text(encoding="utf-8"))
        if not chk[n]["pass"]:
            raise SystemExit(f"check {n} failed; no readout")
    ctx = M.Ctx()
    R = M.load_keys(C.CACHE / "rows.npz", ("fidx", "clus", "tg_L5"))
    _, unit_d = np.unique(R["fidx"], return_inverse=True)
    _, unit_c = np.unique(R["clus"], return_inverse=True)
    us, tab, fifth = M.units()
    by = {u["name"]: u for u in us}
    res = {"registration": M.REG, "addendum": M.ADD, "n_boot": N, "seed": M.SEED, "target": M.X,
           "mixture_rules": list(M.MIX), "rules_reported": list(M.ALL_RULES),
           "rule_sets": {"D10..D1000": "common set (common_L5 of data/.b75/sets.npz)", "S": "S set (S_L5 of data/.b75/sets.npz)",
                         "K10": "B.78 common set (common_L5 of data/.b78/sets.npz)"},
           "sources": {"T0, TX, T0a, TXa (D10..S)": "data/.b75/fits (B.75)", "T0, TX (K10)": "data/.b78/fits (B.78)",
                       "T0all, TXall, T0raw, TXraw, TXrel": "data/.b84/fits (B.84)"},
           "designs": {d: {"label": DLAB[d], "units": [{"name": n, "fit": int(len(by[n]["fit"])), "test": int(len(by[n]["test"]))}
                                                         for n in DESIGNS[d]]} for d in DESIGNS},
           "cells": {}}
    cells = {}
    for d in DESIGNS:
        res["cells"][d] = {}
        for r in M.ALL_RULES:
            c = Cell(d, r, R, unit_d, unit_c, ctx, by)
            cells[(d, r)] = c
            o = {"counts": c.counts, "estimators_scored": c.have, "rmse_eV": c.rmse(), "estimators": c.robust(), "skills": {}}
            for a, b in skill_pairs(r, c.have):
                o["skills"][f"{a} over {b}"] = c.skill(a, b, f"{d}|{r}|{a} over {b}")
            res["cells"][d][r] = o
            M.log(f"{d} {r}: rows {c.counts['rows']:,}")
    # ---- statements
    st = {}
    for sid, d, r, a, b, how, text, exp in STATEMENTS:
        v = cells[(d, r)].skill(a, b, sid)
        lo = v["cluster"][0]
        holds = lo > 0 if how == "gt0" else lo > -0.05
        st[sid] = {"text": text, "location": [d, r], "comparison": f"{a} over {b}", "judged": JUDGED[how], "value": v,
                   "holds": bool(holds), "expected": exp}
    res["statements"] = st
    ex = {}
    for k, s in st.items():
        e = s["expected"]
        met = s["holds"] if e.startswith("holds") else (not s["holds"]) if e.startswith("does not hold") else None
        ex[k] = {"verdict_as_expected": met}
    res["expectations"] = ex
    # ---- fit rows
    fr = {}
    for un in M.UNIT_NAMES:
        fr[un] = {}
        for L in M.MIX_LEARNERS:
            fr[un][L] = json.loads((M.FITS / f"{un}__{L}__fit.json").read_text(encoding="utf-8"))
        for r in M.REL_RULES:
            z = np.load(M.FITS / f"{un}__TXrel__{r}.npz")
            fr[un][f"TXrel|{r}"] = {"fit_rows": int(z["fit_rows"]), "n_iter": int(z["n_iter"])}
    res["fit_rows"] = fr
    res["checks"] = chk
    res["runtime_readout_s"] = round(time.time() - t_start, 1)
    M.save_json(M.OUT_JSON, res)
    spec = importlib.util.spec_from_file_location("render84", M.HERE / "render84.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    M.save_text(M.OUT_MD, mod.render(res))
    M.log(f"written {M.OUT_JSON.name}, {M.OUT_MD.name}")


if __name__ == "__main__":
    main()
